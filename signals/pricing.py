import asyncio, os, json, re
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from signals.cache import CACHE_VERSION, get_cache_if_fresh, set_cache, USE_CACHE

load_dotenv()

_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if _anthropic_api_key:
    from anthropic import AsyncAnthropic
    from anthropic.types import TextBlock
    anthropic_client = AsyncAnthropic(api_key=_anthropic_api_key)
else:
    anthropic_client = None
    print("[pricing] ANTHROPIC_API_KEY not set — Claude fallback disabled.")


PRICING_PATHS = ["/pricing", "/plans", "/pricing-plans"]
PRICE_REGEX   = re.compile(r'\$\s*(\d+(?:\.\d{1,2})?)')


async def fetch_pricing_page(client: httpx.AsyncClient, domain: str) -> tuple[str, str]:
    for path in PRICING_PATHS:
        url = f"https://{domain}{path}"
        try:
            res = await client.get(url)
            if res.status_code == 200:
                return url, res.text
        except Exception as e:
            print(f"[pricing] fetch {url}: {e}")
            continue
    return "", ""


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in ["script", "style", "nav", "footer", "header"]:
        for el in soup.find_all(tag):
            el.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text[:3000]


def extract_prices(text: str) -> list[float]:
    matches = PRICE_REGEX.findall(text)
    prices = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            if 5 <= val < 10000:
                prices.append(val)
        except Exception:
            continue
    return sorted(list(set(prices)))[:3]


def extract_prices_from_dom(html: str) -> list[float]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for el in soup.find_all():
        text = el.get_text(strip=True)
        if "$" in text:
            matches = re.findall(r'\$\s*(\d+(?:\.\d{1,2})?)', text)
            for m in matches:
                try:
                    val = float(m)
                    if 5 <= val < 10000:
                        candidates.append(val)
                except Exception:
                    continue
    return candidates


def detect_pricing_model(text: str, prices: list[float]) -> str:
    t = text.lower()
    if any(k in t for k in ["per user", "per seat"]):
        return "per_seat"
    if "usage" in t or "per api call" in t:
        return "usage"
    if "free" in t and prices:
        return "freemium"
    if prices:
        return "flat"
    if "contact sales" in t or "contact us" in t:
        return "enterprise"
    return "unknown"


def estimate_acv(model: str, prices: list[float]) -> float:
    try:
        if model == "per_seat":
            if not prices:
                return 0.0
            mid = prices[len(prices) // 2]
            return mid * 10 * 12
        elif model == "flat":
            if not prices:
                return 0.0
            return max(prices) * 12
        elif model == "freemium":
            paid = [p for p in prices if p > 0]
            if not paid:
                return 0.0
            return min(paid) * 5 * 12
        elif model == "enterprise":
            return 50000.0
        elif model == "usage":
            return 0.0
    except Exception:
        return 0.0
    return 0.0


def compute_confidence(has_page: bool, prices: list[float], model: str) -> float:
    if not has_page:
        return 0.0
    if model == "enterprise":
        return 0.3
    if not prices:
        return 0.2
    if len(prices) >= 2:
        return 0.9
    return 0.6


def detect_currency(text: str) -> str:
    if "€" in text:
        return "EUR"
    if "£" in text:
        return "GBP"
    return "USD"


async def extract_with_claude(text: str) -> tuple[str, list[float], str]:
    """Use Claude Haiku to extract pricing when heuristics have low confidence."""
    if anthropic_client is None:
        return "unknown", [], "USD"

    try:
        from anthropic.types import TextBlock
        prompt = (
            "Extract pricing information from this SaaS pricing page text.\n"
            "Return a JSON object only, no markdown, no explanation, no code fences.\n"
            "Fields:\n"
            '- pricing_model: one of "per_seat", "usage", "flat", "freemium", "enterprise", "unknown"\n'
            "- price_points: list of numbers (monthly USD prices only, empty list if none found)\n"
            '- currency: currency code string (default "USD")\n\n'
            "Text:\n" + text
        )

        res = await anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        block = res.content[0]
        if not isinstance(block, TextBlock):
            return "unknown", [], "USD"
        content = block.text.strip()

        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        return (
            data.get("pricing_model", "unknown"),
            [float(p) for p in data.get("price_points", [])],
            data.get("currency", "USD"),
        )
    except Exception as e:
        print(f"[pricing] Claude fallback error: {e}")
        return "unknown", [], "USD"


async def get_pricing_signal(domain: str, force_refresh: bool = False) -> dict:
    normalized_domain = (
        domain.strip()
        .replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .lower()
    )

    cache_key = f"pricing:{CACHE_VERSION}:{normalized_domain}"
    cached = get_cache_if_fresh(cache_key, force_refresh=force_refresh)
    if USE_CACHE and cached:
        print(f"[pricing] cache hit: {normalized_domain}")
        return cached

    empty_result = {
        "pricing_model":      "unknown",
        "has_public_pricing": False,
        "price_points":       [],
        "estimated_acv":      0.0,
        "currency":           "USD",
        "source_url":         "",
        "confidence":         0.0,
    }

    async with httpx.AsyncClient(
        timeout=6,
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
    ) as client:
        url, html = await fetch_pricing_page(client, normalized_domain)
        if not html:
            return empty_result

    text = clean_html(html)

    prices_dom  = extract_prices_from_dom(html)
    prices_text = extract_prices(text)
    prices      = sorted(list(set(prices_dom + prices_text)))[:5]

    model      = detect_pricing_model(text, prices)
    currency   = detect_currency(text)
    confidence = compute_confidence(True, prices, model)

    if prices_dom and prices_text:
        confidence = min(1.0, confidence + 0.1)

    if confidence < 0.6:
        claude_model, claude_prices, claude_currency = await extract_with_claude(text)
        if claude_prices:
            prices   = claude_prices
            currency = claude_currency
        if claude_model != "unknown":
            model = claude_model
        confidence = compute_confidence(True, prices, model)

    acv = estimate_acv(model, prices)

    result = {
        "pricing_model":      model,
        "has_public_pricing": True,
        "price_points":       prices,
        "estimated_acv":      acv,
        "currency":           currency,
        "source_url":         url,
        "confidence":         confidence,
    }

    set_cache(cache_key, result)
    return result