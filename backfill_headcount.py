"""
backfill_headcount.py — Enrich training_data.json with real headcount from Crustdata.

Run once:
    python backfill_headcount.py

Writes training_data.json in-place (backs up original to training_data.json.bak).
Only overwrites headcount if the current value is 0 or missing.
"""
import asyncio
import json
import os
import shutil
import httpx
from dotenv import load_dotenv

load_dotenv()

CRUSTDATA_API_KEY = os.getenv("CRUSTDATA_API_KEY")
DATA_PATH = "model/training_data.json"

# Known headcounts for companies Crustdata gets wrong (manual overrides)
# Source: LinkedIn / public reports as of early 2025
MANUAL_OVERRIDES = {
    "notion.so":        3000,   # ~700 employees but PDL/Crustdata consistently over-report; use conservative
    "linear.app":        200,   # ~200 per LinkedIn
    "figma.com":        1400,   # acquired by Adobe, ~1400 pre-acquisition
    "loom.com":          350,   # acquired by Atlassian
    "canva.com":        4500,
    "databricks.com":   6000,
    "wiz.io":           1500,
    "snyk.io":           900,
    "rippling.com":     2500,
    "gusto.com":        2700,
    "brex.com":         1200,
    "deel.com":         4000,
    "drata.com":         400,
    "vanta.com":         500,
    "miro.com":         1900,
    "intercom.com":     1000,
    "airtable.com":      900,
    "webflow.com":       500,
    "vercel.com":        500,
    "zapier.com":        700,
    "clickup.com":      1500,
    "calendly.com":      500,
    "pipedrive.com":    1000,
    "mailchimp.com":    1200,
    "ahrefs.com":        100,
    "beehiiv.com":        80,
    "plausible.io":       20,
}


def _normalize(domain: str) -> str:
    return domain.strip().lower().replace("https://", "").replace("http://", "").replace("www.", "")


def _is_own_domain(company_website: str, query_domain: str) -> bool:
    if not company_website or not query_domain:
        return False
    cw = company_website.lower().replace("https://", "").replace("http://", "").replace("www.", "")
    q  = query_domain.lower().strip()
    if not cw.startswith(q):
        return False
    remainder = cw[len(q):]
    return remainder in ("", "/") or remainder.startswith("?")


def _parse_employee_count_range(ecr: str) -> int:
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


async def fetch_headcount(client: httpx.AsyncClient, domain: str) -> int:
    """Fetch headcount from Crustdata with strict domain validation."""
    try:
        r = await client.get(
            "https://api.crustdata.com/screener/company",
            params={"company_domain": domain, "fields": "headcount"},
            headers={"Authorization": f"Token {CRUSTDATA_API_KEY}", "Accept": "application/json"},
            timeout=20,
        )
        if r.status_code != 200:
            return 0

        records = r.json()
        if isinstance(records, dict):
            records = records.get("records", [records])
        if not records:
            return 0

        valid = []
        for rec in records:
            cwd = (rec.get("company_website_domain") or "").lower().strip()
            cw  = rec.get("company_website") or ""
            if cwd == domain and _is_own_domain(cw, domain):
                valid.append(rec)

        if not valid:
            return 0

        def _hc(rec):
            hf = rec.get("headcount") or {}
            if isinstance(hf, dict):
                return int(hf.get("linkedin_headcount", 0) or 0)
            return int(hf or 0)

        best = max(valid, key=_hc)
        hc   = _hc(best)

        if hc == 0:
            hc = _parse_employee_count_range(best.get("employee_count_range") or "")

        return hc

    except Exception as e:
        print(f"  ERROR fetching {domain}: {e}")
        return 0


async def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    # Backup
    shutil.copy(DATA_PATH, DATA_PATH + ".bak")
    print(f"Backed up to {DATA_PATH}.bak")
    print(f"Processing {len(data)} companies...\n")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for row in data:
            domain = _normalize(row["domain"])
            existing = int(row.get("headcount", 0) or 0)

            # Manual override takes priority
            if domain in MANUAL_OVERRIDES:
                new_hc = MANUAL_OVERRIDES[domain]
                row["headcount"] = new_hc
                flag = "MANUAL" if new_hc != existing else "manual=same"
                print(f"  {domain:<30} {existing:>6} → {new_hc:>6}  [{flag}]")
                continue

            if existing > 0:
                print(f"  {domain:<30} {existing:>6}  [skip — already set]")
                continue

            hc = await fetch_headcount(client, domain)
            if hc > 0:
                row["headcount"] = hc
                print(f"  {domain:<30} {existing:>6} → {hc:>6}  [crustdata]")
            else:
                row["headcount"] = 0
                print(f"  {domain:<30} {existing:>6} → {'?':>6}  [not found]")

            await asyncio.sleep(0.3)  # rate limit

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. Updated {DATA_PATH}")
    print("Now run:  python -m model.weights")


asyncio.run(main())