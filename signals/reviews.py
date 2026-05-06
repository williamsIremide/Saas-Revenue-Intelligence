import asyncio, os, re, math, json
import httpx
from bs4 import BeautifulSoup
from datetime import UTC, datetime, timedelta
from dotenv import load_dotenv
from signals.cache import get_cache_if_fresh, set_cache, USE_CACHE, CACHE_VERSION

load_dotenv()

_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if _anthropic_api_key:
    try:
        from anthropic import AsyncAnthropic
        from anthropic.types import TextBlock
        anthropic_client = AsyncAnthropic(api_key=_anthropic_api_key)
    except Exception:
        anthropic_client = None
else:
    anthropic_client = None


SLUG_OVERRIDES = {
    "hubspot":    "hubspot-crm",
    "salesforce": "salesforce-crm",
    "microsoft":  "microsoft-teams",
}


def normalize_domain(domain: str) -> str:
    return (
        domain.strip()
        .replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .lower()
    )


def extract_slug(domain: str) -> str:
    return normalize_domain(domain).split(".")[0]


REVIEW_REGEX = [
    r'see all\s+([\d,]+)\s+\w',
    r'([\d,]+)\s+reviews?',
    r'([\d,]+)\s+\w+\s+reviews?',
]
RATING_REGEX = re.compile(r'(\d\.\d)\s*out\s*of\s*5', re.IGNORECASE)


def extract_review_count(text: str) -> int:
    for pattern in REVIEW_REGEX:
        match = re.search(pattern, text.lower())
        if match:
            raw = match.group(1).replace(",", "").strip()
            if raw and raw.isdigit():
                val = int(raw)
                if val > 0:
                    return val
    return 0


def extract_rating_from_text(text: str) -> float:
    match = RATING_REGEX.search(text)
    if match:
        return float(match.group(1))
    return 0.0


def extract_trustpilot_rating(soup: BeautifulSoup) -> float:
    el = soup.find(attrs={"itemprop": "ratingValue"})
    if el:
        try:
            return float(el.get("content") or el.get_text(strip=True))
        except (ValueError, TypeError):
            pass

    meta = soup.find("meta", attrs={"name": "trustpilot-score"})
    if meta:
        try:
            return float(meta["content"])
        except (KeyError, ValueError):
            pass

    el = soup.find(attrs={"data-rating": True})
    if el:
        try:
            return float(el["data-rating"])
        except (KeyError, ValueError):
            pass

    for el in soup.find_all(attrs={"aria-label": True}):
        text = el.get("aria-label", "")
        match = re.search(r'(\d\.\d)', text)
        if match:
            val = float(match.group(1))
            if 1.0 <= val <= 5.0:
                return val

    return 0.0


def extract_dates(soup: BeautifulSoup) -> list[datetime]:
    dates = []
    for el in soup.find_all("time"):
        dt_attr = el.get("datetime", "")
        if dt_attr:
            try:
                dt = datetime.fromisoformat(dt_attr.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                dates.append(dt)
                continue
            except ValueError:
                pass
        t = el.get_text(strip=True)
        for fmt in ["%B %d, %Y", "%b %d, %Y", "%b %Y"]:
            try:
                dt = datetime.strptime(t, fmt).replace(tzinfo=UTC)
                dates.append(dt)
                break
            except ValueError:
                continue
    return dates


def calc_velocity_90d(dates: list[datetime]) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=90)
    return sum(1 for d in dates if d >= cutoff)


def sentiment_from_rating(rating: float) -> float:
    if rating <= 0:
        return 0.0
    return round((rating - 1) / 4, 2)


def momentum_score(total: int, velocity: int, rating: float) -> float:
    volume = min(math.log10(max(total, 1)) / 3, 1.0)
    vel    = min(velocity / 50, 1.0)
    rating_score = max((rating - 1) / 4, 0)
    return round(volume * 0.3 + vel * 0.5 + rating_score * 0.2, 2)


def compute_confidence(total: int, rating: float, velocity: int) -> float:
    if total == 0:
        return 0.0
    score = 0.4
    if total > 50:
        score += 0.2
    elif total > 10:
        score += 0.1
    if rating > 0:
        score += 0.2
    if velocity > 0:
        score += 0.2
    return min(score, 1.0)


POSITIVE_WORDS = ["good", "great", "love", "excellent", "amazing"]
NEGATIVE_WORDS = ["bad", "poor", "slow", "bug", "issue"]


def extract_sentiment_keywords(text: str) -> float:
    t   = text.lower()
    pos = sum(t.count(w) for w in POSITIVE_WORDS)
    neg = sum(t.count(w) for w in NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.5
    return pos / (pos + neg)


async def extract_with_claude(text: str) -> tuple[float, int]:
    if anthropic_client is None:
        return 0.0, 0
    try:
        prompt = (
            "Extract review data from this page text.\n"
            "Return JSON only, no markdown, no explanation:\n"
            '{"rating": number, "total_reviews": number}\n\n'
            "Text:\n" + text[:3000]
        )
        res = await anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        block = res.content[0]
        if not isinstance(block, TextBlock):
            return 0.0, 0
        content = block.text.strip()
        if content.startswith("```"):
            content = content.split("```")[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()
        data = json.loads(content)
        return float(data.get("rating", 0)), int(data.get("total_reviews", 0))
    except Exception as e:
        print(f"[reviews] Claude fallback error: {e}")
        return 0.0, 0


async def fetch_page(url: str) -> str:
    """
    Race all available scraping providers IN PARALLEL.
    Hard timeout: 12 seconds per call — fast fail, never blocks the pipeline.
    """
    from urllib.parse import quote

    providers = []
    scrapfly_key   = os.getenv("SCRAPFLY_API_KEY")
    scrapedo_key   = os.getenv("SCRAPEDO_API_KEY")
    scraperapi_key = os.getenv("SCRAPER_API_KEY")

    if scrapfly_key:
        encoded = quote(url, safe="")
        providers.append(("scrapfly",   f"https://api.scrapfly.io/scrape?key={scrapfly_key}&url={encoded}&asp=true"))
    if scrapedo_key:
        encoded = quote(url, safe="")
        providers.append(("scrapedo",   f"https://api.scrape.do?token={scrapedo_key}&url={encoded}"))
    if scraperapi_key:
        providers.append(("scraperapi", f"http://api.scraperapi.com?api_key={scraperapi_key}&url={url}"))
    providers.append(("direct", url))

    async def _try_one(name: str, proxy_url: str) -> tuple[str, str]:
        timeout = 10 if name != "direct" else 6
        async with httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
            follow_redirects=True,
        ) as client:
            res = await client.get(proxy_url)
            if res.status_code != 200:
                raise ValueError(f"HTTP {res.status_code}")
            if name == "scrapfly":
                data = res.json()
                sc = data.get("result", {}).get("status_code", 200)
                if sc != 200:
                    raise ValueError(f"scrapfly target status {sc}")
                html = data.get("result", {}).get("content", "")
            else:
                html = res.text
            if not html or len(html) < 5000:
                raise ValueError(f"too little content ({len(html) if html else 0} bytes)")
            return name, html

    tasks = [asyncio.create_task(_try_one(name, proxy_url)) for name, proxy_url in providers]

    try:
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=12,
        )
        # Cancel and fully await losers to prevent "exception never retrieved" noise
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        for t in done:
            exc = t.exception()
            if exc is None:
                name, html = t.result()
                print(f"[reviews/{name}] success, {len(html)} bytes")
                return html
            # else: this provider failed, try next done task

    except Exception as e:
        print(f"[reviews] fetch_page error: {e}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    return ""


async def _fetch_urls_racing(urls: list[str], label: str) -> str:
    """
    Try multiple URLs (slug variants) all at once — return first that works.
    Replaces the sequential for-loop in fetch_g2 / fetch_trustpilot.
    """
    tasks = [asyncio.create_task(fetch_page(url)) for url in urls]
    try:
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=14,
        )
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        for t in done:
            if t.exception() is None:
                html = t.result()
                if html:
                    return html
    except Exception:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return ""


async def fetch_trustpilot(slug: str, domain: str) -> tuple[str, str]:
    urls = [
        f"https://www.trustpilot.com/review/{domain}",
        f"https://www.trustpilot.com/review/{slug}.com",
        f"https://www.trustpilot.com/review/{slug}.io",
    ]
    html = await _fetch_urls_racing(urls, "trustpilot")
    return html, "trustpilot"


async def fetch_g2(slug: str) -> tuple[str, str]:
    primary = SLUG_OVERRIDES.get(slug, slug)
    slugs_to_try = [primary] if primary != slug else [slug, f"{slug}-app", f"{slug}-software"]
    urls = [f"https://www.g2.com/products/{s}/reviews" for s in slugs_to_try]
    html = await _fetch_urls_racing(urls, "g2")
    if html and "reviews" in html.lower():
        return html, "g2"
    return "", "g2"


async def _get_reviews_signal_impl(domain: str, force_refresh: bool = False) -> dict:
    normalized = normalize_domain(domain)
    slug       = extract_slug(domain)

    cache_key = f"reviews:{CACHE_VERSION}:{normalized}"
    cached = get_cache_if_fresh(cache_key, force_refresh=force_refresh)
    if USE_CACHE and cached:
        return cached

    empty: dict = {
        "total_reviews":       0,
        "rating":              0.0,
        "review_velocity_90d": 0,
        "sentiment_score":     0.0,
        "momentum_score":      0.0,
        "source":              "none",
        "product_slug":        slug,
        "confidence":          0.0,
    }

    # Run G2 and Trustpilot fetches concurrently
    tp_task = asyncio.create_task(fetch_trustpilot(slug, normalized))
    g2_task = asyncio.create_task(fetch_g2(slug))

    tp_html, _ = await tp_task
    g2_html, _ = await g2_task

    if g2_html:
        html, source = g2_html, "g2"
    elif tp_html:
        html, source = tp_html, "trustpilot"
    else:
        return empty

    soup  = BeautifulSoup(html, "html.parser")
    text  = soup.get_text(" ", strip=True)

    total  = extract_review_count(text)
    rating = extract_rating_from_text(text)

    if rating == 0.0 and source == "trustpilot":
        rating = extract_trustpilot_rating(soup)

    dates    = extract_dates(soup)
    velocity = calc_velocity_90d(dates)

    if velocity == 0 and total > 0:
        velocity = int(total * 0.03)

    confidence = compute_confidence(total, rating, velocity)

    if confidence < 0.6:
        llm_rating, llm_total = await extract_with_claude(text)
        if llm_total > total:
            total = llm_total
        if llm_rating > 0:
            rating = llm_rating
        confidence = compute_confidence(total, rating, velocity)

    if source == "trustpilot":
        rating     = round(rating + (3.0 - rating) * 0.4, 2)
        confidence = round(min(confidence, 0.7), 2)

    if source == "g2" and total < 5:
        return empty

    sentiment_score   = sentiment_from_rating(rating)
    keyword_sentiment = extract_sentiment_keywords(text[:2000])
    sentiment_score   = round(sentiment_score * 0.7 + keyword_sentiment * 0.3, 2)

    momentum = momentum_score(total, velocity, rating)

    result = {
        "total_reviews":       total,
        "rating":              rating,
        "review_velocity_90d": velocity,
        "sentiment_score":     sentiment_score,
        "momentum_score":      momentum,
        "source":              source,
        "product_slug":        slug,
        "confidence":          round(confidence, 2),
    }

    set_cache(cache_key, result)
    return result


async def get_reviews_signal(domain: str, force_refresh: bool = False) -> dict:
    """
    Public entry point — wraps implementation with a hard 20s timeout.
    If reviews can't be fetched in time, returns empty gracefully so the
    other 4 signals still produce a result within the 60s Context Protocol limit.
    """
    empty = {
        "total_reviews":       0,
        "rating":              0.0,
        "review_velocity_90d": 0,
        "sentiment_score":     0.0,
        "momentum_score":      0.0,
        "source":              "timeout",
        "product_slug":        extract_slug(domain),
        "confidence":          0.0,
    }
    try:
        return await asyncio.wait_for(
            _get_reviews_signal_impl(domain, force_refresh),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        print(f"[reviews] 20s timeout for {domain} — returning empty")
        return empty
    except Exception as e:
        print(f"[reviews] error for {domain}: {e}")
        return empty