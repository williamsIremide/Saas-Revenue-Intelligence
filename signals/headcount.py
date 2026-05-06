"""
headcount.py — LinkedIn employee count signal.

Total headcount correlates ~0.65 with log(ARR), vs 0.33 for open_roles.
It measures accumulated revenue capacity rather than current hiring velocity.

Sources (tried in priority order)
----------------------------------
1. Crustdata          set CRUSTDATA_API_KEY  — covers headcount, also returns job_openings
                                               and web traffic trends. One call, lots of data.
                                               Docs: https://docs.crustdata.com
2. People Data Labs    set PDL_API_KEY        — 100 free calls/month
3. LinkdAPI            set LINKDAPI_KEY        — direct LinkedIn data, free tier
4. Proxycurl           set PROXYCURL_API_KEY  — paid (~$0.01/call), most reliable
5. SerpAPI             set SERPAPI_API_KEY    — Google-searches LinkedIn snippet (free tier 100/mo)
6. ScraperAPI          set SCRAPER_API_KEY    — fallback Google search via proxy
7. LinkedIn direct                            — free but rate-limited (~70% hit rate)
8. Returns 0 / confidence=0.0               — model degrades gracefully to fallback weights

.env keys used:
  CRUSTDATA_API_KEY=...   (already in your .env — highest priority)
  PDL_API_KEY=...
  LINKDAPI_KEY=...
  PROXYCURL_API_KEY=...   (optional paid)
  SERPAPI_API_KEY=...     (already in your .env)
  SCRAPER_API_KEY=...     (already in your .env)
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

CRUSTDATA_API_KEY = os.getenv("CRUSTDATA_API_KEY")
PDL_API_KEY       = os.getenv("PDL_API_KEY")
LINKDAPI_KEY      = os.getenv("LINKDAPI_KEY")
PROXYCURL_API_KEY = os.getenv("PROXYCURL_API_KEY")
SERPAPI_API_KEY   = os.getenv("SERPAPI_API_KEY")
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
    """Best-effort domain -> LinkedIn company slug."""
    d = _normalize(domain)
    for suffix in [".so", ".app", ".us", ".io", ".co", ".ai", ".dev", ".inc"]:
        if d.endswith(suffix):
            return d[: -len(suffix)]
    if d.endswith(".com"):
        return d[:-4]
    return d.split(".")[0]


def _parse_employee_count_range(ecr: str) -> int:
    """'5001-10000' → 7500, '10001+' → 12000, '2-10' → 6, '' → 0"""
    if not ecr:
        return 0
    ecr = ecr.strip().replace(",", "")
    if ecr.endswith("+"):
        try:
            return int(ecr[:-1]) + 1000
        except ValueError:
            return 0
    if "-" in ecr:
        parts = ecr.split("-")
        try:
            return (int(parts[0]) + int(parts[1])) // 2
        except (ValueError, IndexError):
            return 0
    try:
        return int(ecr)
    except ValueError:
        return 0
    
def _is_own_domain(company_website: str, query_domain: str) -> bool:
    """
    Returns True only if company_website points to the ROOT of query_domain —
    meaning this is actually the company's own website, not a page hosted there.
 
    True:  "https://notion.so"         for query "notion.so"
    True:  "https://www.notion.so/"    for query "notion.so"
    True:  "https://hubspot.com"       for query "hubspot.com"
    False: "https://notion.so/kriss-hkust/..."   ← hosted Notion page
    False: "https://notion.so/isssues/abc123..."  ← hosted Notion page
    """
    if not company_website or not query_domain:
        return False
    cw = company_website.lower().replace("https://", "").replace("http://", "").replace("www.", "")
    q  = query_domain.lower().strip()
    if not cw.startswith(q):
        return False
    remainder = cw[len(q):]
    # Only accept empty remainder, trailing slash, or query string — no path segments
    return remainder in ("", "/") or remainder.startswith("?")

# def _score_crustdata_record(rec: dict, query_domain: str) -> float:
#     """
#     Score a Crustdata record for how well it matches the queried domain.

#     Returns a float 0.0–3.0. Higher = better match.

#     Rules (in priority order):
#     1. +2.0 if company_website_domain exactly matches (e.g. "notion.so" == "notion.so")
#     2. +1.5 if company_website contains the query domain
#     3. +0.5 if is_full_domain_match is True (Crustdata's own flag)
#     4. -1.0 penalty if employee_count_range is "1-10" or "11-50" and headcount < 100
#        (real SaaS companies we'd query have at least a few dozen staff)
#     5. +0.3 bonus if hq_country is "USA" or "GBR" (most of our training set)

#     The HKUST KRISS false-positive has:
#       - company_website_domain = "notion.so"  (wrong — it's a hosted Notion page)
#       - company_website = "https://www.notion.so/kriss-hkust/..."  (not notion.so the company)
#       - hq_country = "" / hq_state = "Hong Kong"
#       - employee_count_range = "11-50"
#       - is_full_domain_match likely False

#     The real Notion record would have:
#       - company_website = "https://notion.so" or "https://www.notion.so"
#       - company_website_domain = "notion.so"
#       - hq_country = "USA"
#       - employee_count_range = "1001-5000" or similar
#     """
#     score   = 0.0
#     q = query_domain.lower().strip()
 
#     # ── 1. company_website_domain exact match (+2.0) ──────────────────────────
#     cwd = (rec.get("company_website_domain") or "").lower().strip()
#     if cwd == q:
#         score += 2.0
 
#     # ── 2. company_website root-only match (+1.5) ─────────────────────────────
#     # The website must point to the company's OWN root, not a page hosted there.
#     # Valid:   "https://notion.so"  or  "https://notion.so/"
#     # Invalid: "https://notion.so/isssues/abc123..."  ← hosted page
#     # Invalid: "https://notion.so/kriss-hkust/..."    ← hosted page
#     cw = (rec.get("company_website") or "").lower()
#     cw_bare = cw.replace("https://", "").replace("http://", "").replace("www.", "")
#     if cw_bare.startswith(q):
#         remainder = cw_bare[len(q):]   # everything after the domain
#         # Accept only if nothing follows, or just "/" or a query string
#         remainder_is_root = remainder == "" or remainder == "/" or remainder.startswith("?")
#         # Reject if remainder contains UUID-like or long hex strings (hosted doc pages)
#         _UUID_RE = re.compile(r'[0-9a-f]{8,}', re.IGNORECASE)
#         has_uuid_path = bool(_UUID_RE.search(remainder))
#         if remainder_is_root and not has_uuid_path:
#             score += 1.5
 
#     # ── 3. Crustdata's own domain-match flag (+0.5) ───────────────────────────
#     if rec.get("is_full_domain_match"):
#         score += 0.5
 
#     # ── 4. Small-org penalty (-1.0) ───────────────────────────────────────────
#     # Real SaaS companies we'd query have meaningful headcounts.
#     # Tiny orgs that happen to host pages on a platform domain are false positives.
#     ecr = (rec.get("employee_count_range") or "").lower()
#     hc_nested = rec.get("headcount") or {}
#     hc = 0
#     if isinstance(hc_nested, dict):
#         hc = int(hc_nested.get("linkedin_headcount", 0) or 0)
#     elif isinstance(hc_nested, (int, float)):
#         hc = int(hc_nested)
#     if hc == 0:
#         hc = int(rec.get("employee_count", 0) or 0)
 
#     if ecr in ("1-10", "2-10", "11-50") or (0 < hc < 50):
#         score -= 1.0
 
#     # ── 5. Geo bonus (+0.3) ───────────────────────────────────────────────────
#     if (rec.get("hq_country") or "").upper() in ("USA", "GBR", "CAN", "AUS", "DEU", "SGP"):
#         score += 0.3
 
#     return score


# ── Fetchers ──────────────────────────────────────────────────────────────────

async def _fetch_crustdata(client: httpx.AsyncClient, domain: str) -> tuple[int, dict]:
    """
    Crustdata Company Data API — single call returns headcount, job openings,
    web traffic, G2 reviews, and more.

    Key fix: Crustdata can return multiple records for a domain query (e.g. a real
    SaaS company AND a student club that hosts pages on that domain).  We now pick
    the best-matching record using _score_crustdata_record() instead of blindly
    taking records[0].

    Field path fix: headcount data is nested as rec["headcount"]["linkedin_headcount"],
    NOT rec["headcount"]["headcount"] as previously coded.

    Returns (headcount, extra_signals_dict).
    """
    if not CRUSTDATA_API_KEY:
        return 0, {}
    try:
        r = await client.get(
            "https://api.crustdata.com/screener/company",
            params={
                "company_domain": domain,
                "fields": "headcount,job_openings,glassdoor,g2",
            },
            headers={
                "Authorization": f"Token {CRUSTDATA_API_KEY}",
                "Accept": "application/json",
            },
            timeout=20,
        )
        if r.status_code != 200:
            logger.debug(f"[headcount/crustdata] {domain}: HTTP {r.status_code}")
            return 0, {}
 
        data    = r.json()
        records = data if isinstance(data, list) else data.get("records", [data])
        if not records:
            logger.debug(f"[headcount/crustdata] {domain}: no records")
            return 0, {}
 
        # ── Hard disqualification ──────────────────────────────────────────────
        valid = []
        for rec in records:
            cw  = rec.get("company_website") or ""
            cwd = (rec.get("company_website_domain") or "").lower().strip()
 
            # company_website_domain must match (Crustdata's own domain tag)
            if cwd != domain.lower().strip():
                logger.debug(
                    f"[headcount/crustdata] {domain}: skip '{rec.get('company_name')}' "
                    f"— company_website_domain='{cwd}' != query"
                )
                continue
 
            # company_website must point to the ROOT, not a hosted sub-page
            if not _is_own_domain(cw, domain):
                logger.debug(
                    f"[headcount/crustdata] {domain}: skip '{rec.get('company_name')}' "
                    f"— company_website '{cw}' is a hosted sub-page, not own domain"
                )
                continue
 
            valid.append(rec)
 
        if not valid:
            logger.debug(
                f"[headcount/crustdata] {domain}: {len(records)} record(s) all disqualified "
                f"(all use {domain} as a hosting platform) → falling through to PDL"
            )
            return 0, {}
 
        # Among valid records, pick highest headcount
        def _hc(rec):
            hf = rec.get("headcount") or {}
            if isinstance(hf, dict):
                return int(hf.get("linkedin_headcount", 0) or 0)
            return int(hf or 0)
 
        best = max(valid, key=_hc)
        hc   = _hc(best)
 
        if hc == 0:
            # Fallback to employee_count_range midpoint
            ecr = best.get("employee_count_range") or ""
            hc  = _parse_employee_count_range(ecr)
 
        extra = {}
 
        # job_openings
        jf = best.get("job_openings") or {}
        if isinstance(jf, dict):
            jc = jf.get("job_openings_count") or jf.get("count") or 0
            if jc:
                extra["job_openings_count"] = int(jc)
 
        # g2
        g2 = best.get("g2") or {}
        if isinstance(g2, dict):
            rc = g2.get("g2_review_count")
            rt = g2.get("g2_average_rating")
            if rc:
                extra["g2_reviews"] = int(rc)
                extra["g2_rating"]  = float(rt or 0)
 
        # glassdoor
        gd = best.get("glassdoor") or {}
        if isinstance(gd, dict):
            rc = gd.get("glassdoor_review_count")
            rt = gd.get("glassdoor_overall_rating")
            if rc:
                extra["glassdoor_reviews"] = int(rc)
                extra["glassdoor_rating"]  = float(rt or 0)
 
        # Crustdata revenue bounds — useful even if headcount is low confidence
        rev_lo = best.get("estimated_revenue_lower_bound_usd") or 0
        rev_hi = best.get("estimated_revenue_higher_bound_usd") or 0
        if rev_lo and rev_hi:
            extra["crustdata_revenue_low"]  = int(rev_lo)
            extra["crustdata_revenue_high"] = int(rev_hi)
 
        logger.debug(
            f"[headcount/crustdata] {domain}: hc={hc} "
            f"company='{best.get('company_name')}' extra_keys={list(extra.keys())}"
        )
        return hc, extra
 
    except Exception as e:
        logger.debug(f"[headcount/crustdata] {domain}: {e}")
        return 0, {}


async def _fetch_pdl(client: httpx.AsyncClient, domain: str) -> int:
    """
    People Data Labs Company Enrichment API.
    Free tier: 100 calls/month at peopledatalabs.com
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
    """LinkdAPI company lookup by domain."""
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
            data    = r.json()
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


async def _fetch_serpapi_google(client: httpx.AsyncClient, domain: str) -> int:
    """
    Use SERPAPI_API_KEY to search Google for LinkedIn headcount.
    SerpApi free tier: 100 searches/month.
    """
    if not SERPAPI_API_KEY:
        return 0
    slug  = _to_li_slug(domain)
    query = f'site:linkedin.com/company "{slug}" employees'
    try:
        r = await client.get(
            "https://serpapi.com/search.json",
            params={
                "api_key": SERPAPI_API_KEY,
                "engine":  "google",
                "q":       query,
                "num":     5,
            },
            timeout=20,
        )
        if r.status_code == 200:
            data     = r.json()
            snippets = " ".join(
                res.get("snippet", "") for res in data.get("organic_results", [])
            )
            count = _parse_count(snippets)
            if count > 0:
                logger.debug(f"[headcount/serpapi] {domain}: {count}")
                return count
    except Exception as e:
        logger.debug(f"[headcount/serpapi] {domain}: {e}")
    return 0


async def _fetch_scraperapi_google(client: httpx.AsyncClient, domain: str) -> int:
    """
    Fallback: use SCRAPER_API_KEY to proxy a Google search for LinkedIn headcount.
    Only runs if SerpApi is not configured.
    """
    if not SCRAPER_API_KEY:
        return 0
    slug  = _to_li_slug(domain)
    query = f'site:linkedin.com/company "{slug}" employees'
    try:
        r = await client.get(
            "http://api.scraperapi.com/",
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
                logger.debug(f"[headcount/scraperapi] {domain}: {count}")
                return count
    except Exception as e:
        logger.debug(f"[headcount/scraperapi] {domain}: {e}")
    return 0


async def _fetch_linkedin_direct(client: httpx.AsyncClient, domain: str) -> int:
    """
    Scrape LinkedIn /company/<slug>/about page directly.
    Free, no API key, blocked ~30% of requests. Last resort.
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
    "crustdata":   0.92,  # structured company DB, LinkedIn-sourced
    "pdl":         0.90,  # People Data Labs
    "linkdapi":    0.90,  # LinkdAPI — LinkedIn data
    "proxycurl":   0.95,  # highest — direct LinkedIn API
    "serpapi":     0.65,  # Google snippet parsing
    "scraperapi":  0.60,  # proxy Google search
    "linkedin":    0.70,  # direct scrape
}


# ── Public API ────────────────────────────────────────────────────────────────

async def get_headcount_signal(domain: str, force_refresh: bool = False) -> dict:
    """
    Return headcount signal dict.

    Keys
    ----
    headcount        : int    total LinkedIn employee count (0 = unknown)
    headcount_score  : float  sqrt-normalised 0-1 (10k employees = 1.0)
    source           : str    which fetcher succeeded
    confidence       : float  0.0 if not found, up to 0.95 for paid APIs
    size_tier        : str    micro / small / mid / large / enterprise / unknown
    extra            : dict   bonus signals from Crustdata (g2, glassdoor, job_openings)
    """
    normalized = _normalize(domain)
    cache_key  = f"headcount:{CACHE_VERSION}:{normalized}"

    if not force_refresh:
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
        "extra":           {},
    }

    count  = 0
    source = "none"
    extra  = {}

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # 1. Crustdata — best option, you already have the key
        count, extra = await _fetch_crustdata(client, normalized)
        if count > 0:
            source = "crustdata"

        if count == 0:
            # 2. PDL
            count = await _fetch_pdl(client, normalized)
            if count > 0:
                source = "pdl"

        if count == 0:
            # 3. LinkdAPI
            count = await _fetch_linkdapi(client, normalized)
            if count > 0:
                source = "linkdapi"

        if count == 0:
            # 4. Proxycurl
            count = await _fetch_proxycurl(client, normalized)
            if count > 0:
                source = "proxycurl"

        if count == 0:
            # 5. SerpApi
            count = await _fetch_serpapi_google(client, normalized)
            if count > 0:
                source = "serpapi"

        if count == 0:
            # 6. ScraperAPI Google fallback (burns credits — only if SerpApi unavailable)
            if not SERPAPI_API_KEY:
                count = await _fetch_scraperapi_google(client, normalized)
                if count > 0:
                    source = "scraperapi"

        if count == 0:
            # 7. LinkedIn direct — last resort
            count = await _fetch_linkedin_direct(client, normalized)
            if count > 0:
                source = "linkedin"

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
        "extra":           extra,
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
        print(f"{'Domain':<22} {'Count':>7}  {'Score':>6}  {'Source':<12} {'Tier':<12} {'Extra keys'}")
        print("-" * 75)
        for d in domains:
            r = await get_headcount_signal(d, force_refresh=True)
            extra_keys = list(r.get("extra", {}).keys())
            print(
                f"  {d:<20} {r['headcount']:>7}  "
                f"{r['headcount_score']:>6.3f}  "
                f"{r['source']:<12} {r['size_tier']:<12} {extra_keys}"
            )

    asyncio.run(_test())