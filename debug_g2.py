"""
Run this to see exactly what ScraperAPI is returning for G2.
python debug_g2.py
"""
import asyncio, os, re
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")


def scraper_url(target: str, render: bool = False) -> str:
    base = f"http://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={target}"
    if render:
        base += "&render=true"
    return base


async def probe(label: str, url: str, render: bool = False):
    proxied = scraper_url(url, render=render)
    print(f"\n{'='*60}")
    print(f"PROBE: {label}")
    print(f"render={render}  →  {url}")
    print("="*60)

    async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
        try:
            res = await client.get(proxied)
            print(f"Status : {res.status_code}")
            print(f"Length : {len(res.text):,} chars")

            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text(" ", strip=True)

            # Key signals
            print(f"Has 'reviews' in text : {'reviews' in text.lower()}")
            print(f"Has 'datadome'        : {'datadome' in res.text.lower()}")
            print(f"Has 'captcha'         : {'captcha' in res.text.lower()}")
            print(f"Has 'blocked'         : {'blocked' in res.text.lower()}")

            # Rating / review count patterns
            rating_m = re.search(r'(\d\.\d)\s*out\s*of\s*5', text, re.IGNORECASE)
            count_m  = re.search(r'([\d,]+)\s+reviews?', text, re.IGNORECASE)
            print(f"Rating match : {rating_m.group(0) if rating_m else 'NONE'}")
            print(f"Review count : {count_m.group(0)  if count_m  else 'NONE'}")

            # itemprop schema.org fields
            el_rating = soup.find(attrs={"itemprop": "ratingValue"})
            el_count  = soup.find(attrs={"itemprop": "reviewCount"})
            print(f"itemprop ratingValue  : {el_rating}")
            print(f"itemprop reviewCount  : {el_count}")

            # First 600 chars of visible text
            print(f"\n--- First 600 chars of visible text ---")
            print(text[:600])

            # Links to /products/*/reviews (for search page probe)
            product_links = re.findall(r'/products/([^/"\s]+)/reviews', res.text)
            if product_links:
                print(f"\n--- Product slugs found in page ---")
                for s in product_links[:10]:
                    print(f"  {s}")

        except Exception as e:
            print(f"ERROR: {e}")


async def main():
    if not SCRAPER_API_KEY:
        print("ERROR: SCRAPER_API_KEY not set in .env")
        return

    # 1. G2 product page — no render
    await probe("G2 notion — no render", "https://www.g2.com/products/notion/reviews", render=False)

    # 2. G2 product page — with render
    await probe("G2 notion — render=true", "https://www.g2.com/products/notion/reviews", render=True)

    # 3. G2 search page — no render
    await probe("G2 search 'notion' — no render", "https://www.g2.com/search?query=notion", render=False)

    # 4. G2 search page — with render
    await probe("G2 search 'notion' — render=true", "https://www.g2.com/search?query=notion", render=True)

    # 5. G2 linear (trickier slug)
    await probe("G2 linear — render=true", "https://www.g2.com/products/linear/reviews", render=True)

    # 6. ScraperAPI account status
    print(f"\n{'='*60}")
    print("SCRAPER API — account info")
    print("="*60)
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            res = await client.get(
                f"http://api.scraperapi.com/account?api_key={SCRAPER_API_KEY}"
            )
            print(res.text)
        except Exception as e:
            print(f"ERROR: {e}")


asyncio.run(main())