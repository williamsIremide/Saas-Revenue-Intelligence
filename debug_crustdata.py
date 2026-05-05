"""
debug_crustdata.py — See exactly what Crustdata returns for a domain.
python debug_crustdata.py
python debug_crustdata.py hubspot.com linear.app
"""
import asyncio, os, json, sys
import httpx
from dotenv import load_dotenv

load_dotenv()

CRUSTDATA_API_KEY = os.getenv("CRUSTDATA_API_KEY")

async def probe(domain: str):
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain}")
    print(f"KEY:    {CRUSTDATA_API_KEY[:8]}...{CRUSTDATA_API_KEY[-4:] if CRUSTDATA_API_KEY else 'NOT SET'}")
    print("="*60)

    if not CRUSTDATA_API_KEY:
        print("ERROR: CRUSTDATA_API_KEY not set in .env")
        return

    async with httpx.AsyncClient(follow_redirects=True) as client:

        # ── 1. Minimal call — just headcount ─────────────────────────────────
        print("\n[1] GET /screener/company?company_domain=...&fields=headcount")
        try:
            r = await client.get(
                "https://api.crustdata.com/screener/company",
                params={"company_domain": domain, "fields": "headcount"},
                headers={
                    "Authorization": f"Token {CRUSTDATA_API_KEY}",
                    "Accept": "application/json",
                },
                timeout=20,
            )
            print(f"  Status : {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print(f"  Type   : {type(data).__name__}")
                print(f"  Preview: {json.dumps(data)[:800]}")
            else:
                print(f"  Body   : {r.text[:400]}")
        except Exception as e:
            print(f"  ERROR  : {e}")

        # ── 2. All useful fields ──────────────────────────────────────────────
        print("\n[2] GET /screener/company with headcount,job_openings,glassdoor,g2")
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
            print(f"  Status : {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print(f"  Full response (pretty):")
                print(json.dumps(data, indent=2)[:3000])
            else:
                print(f"  Body   : {r.text[:400]}")
        except Exception as e:
            print(f"  ERROR  : {e}")

        # ── 3. No fields param — get everything ───────────────────────────────
        print("\n[3] GET /screener/company — no fields filter (see all available keys)")
        try:
            r = await client.get(
                "https://api.crustdata.com/screener/company",
                params={"company_domain": domain},
                headers={
                    "Authorization": f"Token {CRUSTDATA_API_KEY}",
                    "Accept": "application/json",
                },
                timeout=20,
            )
            print(f"  Status : {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                # Show top-level keys of first record
                records = data if isinstance(data, list) else data.get("records", [data])
                if records:
                    rec = records[0]
                    print(f"  Top-level keys in record[0]: {list(rec.keys())}")
                    # Print any keys that look headcount-related
                    for k, v in rec.items():
                        if any(kw in k.lower() for kw in ["head", "employee", "staff", "size", "count", "job", "g2", "glass", "review", "traffic", "visit"]):
                            print(f"    {k}: {json.dumps(v)[:200]}")
                else:
                    print(f"  No records. Raw: {json.dumps(data)[:400]}")
            else:
                print(f"  Body   : {r.text[:400]}")
        except Exception as e:
            print(f"  ERROR  : {e}")

        # ── 4. Try enrich_realtime=True ───────────────────────────────────────
        print("\n[4] GET /screener/company?enrich_realtime=True")
        try:
            r = await client.get(
                "https://api.crustdata.com/screener/company",
                params={"company_domain": domain, "enrich_realtime": "True"},
                headers={
                    "Authorization": f"Token {CRUSTDATA_API_KEY}",
                    "Accept": "application/json",
                },
                timeout=30,
            )
            print(f"  Status : {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                records = data if isinstance(data, list) else data.get("records", [data])
                if records:
                    rec = records[0]
                    print(f"  Keys: {list(rec.keys())}")
                    for k, v in rec.items():
                        if any(kw in k.lower() for kw in ["head", "employee", "count", "job"]):
                            print(f"    {k}: {v}")
                else:
                    print(f"  Empty. Raw: {json.dumps(data)[:400]}")
            else:
                print(f"  Body   : {r.text[:400]}")
        except Exception as e:
            print(f"  ERROR  : {e}")


async def main():
    domains = sys.argv[1:] or ["notion.so"]
    for d in domains:
        await probe(d)

asyncio.run(main())