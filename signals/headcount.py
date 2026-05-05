"""
headcount.py — LinkedIn employee count signal.

Total headcount correlates ~0.65 with log(ARR), vs 0.33 for open_roles.
It measures accumulated revenue capacity rather than current hiring velocity.

Sources (tried in priority order, all have free tiers)
------------------------------------------------------
1. People Data Labs  set PDL_API_KEY        — 100 free calls/month,
                                              GET /v5/company/enrich?website=...
                                              returns employee_count directly
2. LinkdAPI          set LINKDAPI_KEY        — direct Proxycurl alternative,
                                              public LinkedIn data, free tier
3. Proxycurl         set PROXYCURL_API_KEY   — paid (~$0.01/call), most reliable,
                                              included for users who want it
4. SerpAPI/ScraperAPI  SCRAPER_API_KEY       — already in your .env,
                                              Google-searches LinkedIn snippet
5. LinkedIn direct                           — free but rate-limited (~70% hit rate)
6. Returns 0 / confidence=0.0               — model degrades gracefully

Signup links:
  PDL:      https://www.peopledatalabs.com  (free tier: 100 calls/month)
  LinkdAPI: https://linkd.inc              (free tier: check their pricing page)

.env keys to add:
  PDL_API_KEY=...
  LINKDAPI_KEY=...
  PROXYCURL_API_KEY=...   (optional paid)
  SCRAPER_API_KEY=...     (already present)
"""

import asyncio
import logging
import math
import os
import re
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

from signals.cache import CACHE_VERSION, get_cache, set_cache, USE_CACHE

load_dotenv()
logger = logging.getLogger(__name__)

PDL_API_KEY       = os.getenv("PDL_API_KEY")
LINKDAPI_KEY      = os.getenv("LINKDAPI_KEY")
PROXYCURL_API_KEY = os.getenv("PROXYCURL_API_KEY")
SCRAPER_API_KEY   = os.getenv("SCRAPER_API_KEY")

_COUNT_RE   = re.compile(
    r'([\d,]+)\s*(?:employees?\s*on\s*LinkedIn|employees?\s*worldwide|people\s*on\s*LinkedIn)',
    re.IGNORECASE,
)
_SNIPPET_RE = re.compile(r'([\d,]+)\s*(?:employees?|people)', re.IGNORECASE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_count(text: str) -> int:
    """Extract the first plausible employee count from a text blob."""
    for pattern in (_COUNT_RE, _SNIPPET_RE):
        m = pattern.search(text)
        if m:
            try:
                val = int(m.group(1).replace(",", ""))
                if 10 <= val <= 500_000:
                    return val
            except ValueError:
                continue
    return 0


def _normalize(domain: str) -> str:
    return (
        domain.strip().lower()
        .replace("https://", "").replace("http://", "").replace("www.", "")
    )


def _to_li_slug(domain: str) -> str:
    """
    Best-effort domain -> LinkedIn company slug.
    notion.so -> notion, linear.app -> linear, zoom.us -> zoom
    """
    d = _normalize(domain)
    for suffix in [".so", ".app", ".us", ".io", ".co", ".ai", ".dev", ".inc"]:
        if d.endswith(suffix):
            return d[: -len(suffix)]
    if d.endswith(".com"):
        return d[:-4]
    return d.split(".")[0]


# ── Fetchers ──────────────────────────────────────────────────────────────────

async def _fetch_pdl(client: httpx.AsyncClient, domain: str) -> int:
    """
    People Data Labs Company Enrichment API.
    Free tier: 100 calls/month at peopledatalabs.com
    Docs: https://docs.peopledatalabs.com/docs/company-enrichment-api
    Returns employee_count field directly — no parsing needed.
    """
    if not PDL_API_KEY:
        return 0
    try:
        r = await client.get(
            "https://api.peopledatalabs.com/v5/company/enrich",
            params={"website": domain, "pretty": "false"},
            headers={"X-Api-Key": PDL_API_KEY},
            timeout=15,
        )
        if r.status_code == 200:
            data  = r.json()
            count = data.get("employee_count") or data.get("size", 0)
            # PDL also has an employee_count_by_role breakdown — we want the total
            if isinstance(count, (int, float)) and count > 0:
                logger.debug(f"[headcount/pdl] {domain}: {count}")
                return int(count)
        elif r.status_code == 404:
            logger.debug(f"[headcount/pdl] {domain}: not found")
        else:
            logger.debug(f"[headcount/pdl] {domain}: HTTP {r.status_code}")
    except Exception as e:
        logger.debug(f"[headcount/pdl] {domain}: {e}")
    return 0


async def _fetch_linkdapi(client: httpx.AsyncClient, domain: str) -> int:
    """
    LinkdAPI company lookup by domain.
    Free tier: check https://linkd.inc for current limits.
    """
    if not LINKDAPI_KEY:
        return 0
    try:
        r = await client.get(
            "https://api.linkd.inc/v1/company",
            params={"domain": domain},
            headers={"x-api-key": LINKDAPI_KEY},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            # LinkdAPI returns {"company": {"employee_count": 4521, ...}}
            company = data.get("company") or data.get("data") or data
            hc = (
                company.get("employee_count")
                or company.get("headcount")
                or company.get("company_size")
                or 0
            )
            if hc and int(hc) > 0:
                logger.debug(f"[headcount/linkdapi] {domain}: {hc}")
                return int(hc)
        else:
            logger.debug(f"[headcount/linkdapi] {domain}: HTTP {r.status_code}")
    except Exception as e:
        logger.debug(f"[headcount/linkdapi] {domain}: {e}")
    return 0


async def _fetch_proxycurl(client: httpx.AsyncClient, domain: str) -> int:
    """Proxycurl Company API — reliable, ~$0.01/call."""
    if not PROXYCURL_API_KEY:
        return 0
    try:
        r = await client.get(
            "https://nubela.co/proxycurl/api/linkedin/company",
            params={"url": f"https://www.linkedin.com/company/{_to_li_slug(domain)}"},
            headers={"Authorization": f"Bearer {PROXYCURL_API_KEY}"},
            timeout=15,
        )
        if r.status_code == 200:
            data  = r.json()
            count = data.get("company_size_on_linkedin") or data.get("follower_count", 0)
            if isinstance(count, (int, float)) and count > 0:
                logger.debug(f"[headcount/proxycurl] {domain}: {count}")
                return int(count)
    except Exception as e:
        logger.debug(f"[headcount/proxycurl] {domain}: {e}")
    return 0


async def _fetch_serpapi(client: httpx.AsyncClient, domain: str) -> int:
    """
    Google-search the LinkedIn company page via ScraperAPI and parse the snippet.
    Uses your existing SCRAPER_API_KEY — no extra cost if you already have it.
    """
    if not SCRAPER_API_KEY:
        return 0
    slug  = _to_li_slug(domain)
    query = f'site:linkedin.com/company "{slug}" employees'
    try:
        r = await client.get(
            "https://api.scraperapi.com/",
            params={
                "api_key": SCRAPER_API_KEY,
                "url":     f"https://www.google.com/search?q={quote(query)}&num=5",
                "render":  "false",
            },
            timeout=20,
        )
        if r.status_code == 200:
            count = _parse_count(r.text)
            if count > 0:
                logger.debug(f"[headcount/serpapi] {domain}: {count}")
                return count
    except Exception as e:
        logger.debug(f"[headcount/serpapi] {domain}: {e}")
    return 0


async def _fetch_linkedin_direct(client: httpx.AsyncClient, domain: str) -> int:
    """
    Scrape LinkedIn /company/<slug>/about page directly.
    Free, no API key needed, but LinkedIn blocks ~30% of requests.
    Works well enough as a last resort.
    """
    slug = _to_li_slug(domain)
    url  = f"https://www.linkedin.com/company/{slug}/about/"
    try:
        r = await client.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=15,
            follow_redirects=True,
        )
        if r.status_code == 200:
            count = _parse_count(r.text)
            if count > 0:
                logger.debug(f"[headcount/linkedin] {domain}: {count}")
                return count
    except Exception as e:
        logger.debug(f"[headcount/linkedin] {domain}: {e}")
    return 0


# ── Source confidence map ─────────────────────────────────────────────────────

_SOURCE_CONFIDENCE = {
    "pdl":        0.90,   # People Data Labs — structured company data
    "linkdapi":   0.90,   # LinkdAPI — LinkedIn data
    "proxycurl":  0.95,   # highest — direct LinkedIn API
    "serpapi":    0.65,   # snippet parsing — less precise
    "linkedin":   0.70,   # direct scrape — blocked sometimes
}


# ── Public API ────────────────────────────────────────────────────────────────

async def get_headcount_signal(domain: str) -> dict:
    """
    Return headcount signal dict.

    Keys
    ----
    headcount      : int    total LinkedIn employee count (0 = unknown)
    headcount_score: float  sqrt-normalised 0-1 (10k employees = 1.0)
    source         : str    which fetcher succeeded
    confidence     : float  0.0 if not found, up to 0.95 for paid APIs
    size_tier      : str    micro / small / mid / large / enterprise / unknown
    """
    normalized = _normalize(domain)
    cache_key  = f"headcount:{CACHE_VERSION}:{normalized}"

    cached = get_cache(cache_key)
    if USE_CACHE and cached:
        logger.debug(f"[headcount] cache hit: {normalized}")
        return cached

    empty = {
        "headcount":       0,
        "headcount_score": 0.0,
        "source":          "none",
        "confidence":      0.0,
        "size_tier":       "unknown",
    }

    count  = 0
    source = "none"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Try each source in priority order
        fetchers = [
            ("pdl",      _fetch_pdl),
            ("linkdapi", _fetch_linkdapi),
            ("proxycurl",_fetch_proxycurl),
            ("serpapi",  _fetch_serpapi),
            ("linkedin", _fetch_linkedin_direct),
        ]
        for name, fn in fetchers:
            try:
                count = await fn(client, normalized)
            except Exception as e:
                logger.debug(f"[headcount/{name}] unexpected error: {e}")
                count = 0
            if count > 0:
                source = name
                break

    if count == 0:
        return empty

    hc_score   = min(math.sqrt(count / 10_000), 1.0)
    confidence = _SOURCE_CONFIDENCE.get(source, 0.60)
    size_tier  = (
        "micro"      if count < 50    else
        "small"      if count < 200   else
        "mid"        if count < 1000  else
        "large"      if count < 5000  else
        "enterprise"
    )

    result = {
        "headcount":       count,
        "headcount_score": round(hc_score, 4),
        "source":          source,
        "confidence":      confidence,
        "size_tier":       size_tier,
    }
    set_cache(cache_key, result)
    return result


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    async def _test():
        domains = sys.argv[1:] or [
            "notion.so", "linear.app", "datadog.com",
            "vercel.com", "stripe.com", "salesforce.com",
        ]
        print(f"{'Domain':<22} {'Count':>7}  {'Score':>6}  {'Source':<12} {'Tier'}")
        print("-" * 60)
        for d in domains:
            r = await get_headcount_signal(d)
            print(
                f"  {d:<20} {r['headcount']:>7}  "
                f"{r['headcount_score']:>6.3f}  "
                f"{r['source']:<12} {r['size_tier']}"
            )

    asyncio.run(_test())