"""
enrich_and_retrain.py — Fetch real headcount for every company in training_data.json,
write it back, then retrain the model.

Run once:
    python enrich_and_retrain.py

What it does
------------
1. Loads model/training_data.json
2. For each row missing headcount (or with headcount=0), calls get_headcount_signal()
3. Writes headcount, headcount_score, headcount_conf back into the row
4. Saves the enriched file
5. Retrains the model via model/weights.py train_model()

This is safe to re-run — rows that already have headcount > 0 are skipped.
Use --force to re-fetch everything.

Cost: one API call per company to Crustdata/PDL/SerpApi. ~100 companies in training set.
Time: ~3-5 minutes (parallel batches of 5).
"""

import asyncio
import json
import math
import sys
import argparse
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--force",  action="store_true", help="Re-fetch even if headcount already present")
parser.add_argument("--dry-run", action="store_true", help="Print what would be fetched, don't write")
parser.add_argument("--limit",  type=int, default=0,  help="Only process first N rows (for testing)")
parser.add_argument("--batch",  type=int, default=5,  help="Concurrent fetches per batch")
args = parser.parse_args()

TRAINING_DATA_PATH = Path("model/training_data.json")

# ── Load ──────────────────────────────────────────────────────────────────────

with open(TRAINING_DATA_PATH) as f:
    data = json.load(f)

print(f"Loaded {len(data)} training rows from {TRAINING_DATA_PATH}")

# ── Identify rows needing enrichment ──────────────────────────────────────────

to_enrich = []
already_have = []

for row in data:
    hc = row.get("headcount", 0) or 0
    if hc > 0 and not args.force:
        already_have.append(row["domain"])
    else:
        to_enrich.append(row)

print(f"  Already have headcount : {len(already_have)}")
print(f"  Need to fetch          : {len(to_enrich)}")

if args.limit:
    to_enrich = to_enrich[:args.limit]
    print(f"  (limited to first {args.limit} rows by --limit flag)")

if args.dry_run:
    print("\nDry run — would fetch headcount for:")
    for row in to_enrich:
        print(f"  {row['domain']:<30}  {row['company']}")
    sys.exit(0)

if not to_enrich:
    print("\nAll rows already have headcount. Use --force to re-fetch.")
    print("Running retrain anyway...")
else:
    # ── Fetch headcount in batches ─────────────────────────────────────────────
    from signals.headcount import get_headcount_signal

    async def fetch_batch(rows: list[dict]) -> list[dict]:
        results = await asyncio.gather(
            *[get_headcount_signal(row["domain"], force_refresh=True) for row in rows],
            return_exceptions=True,
        )
        enriched = []
        for row, result in zip(rows, results):
            if isinstance(result, Exception):
                print(f"  ⚠  {row['domain']}: {result}")
                enriched.append(row)
                continue

            assert isinstance(result, dict)
            hc       = result.get("headcount", 0)
            hc_score = result.get("headcount_score", 0.0)
            hc_conf  = result.get("confidence", 0.0)
            source   = result.get("source", "none")

            row["headcount"]       = hc
            row["headcount_score"] = hc_score
            row["headcount_conf"]  = hc_conf

            status = "✅" if hc > 0 else "❌"
            print(f"  {status} {row['domain']:<30} hc={hc:>7,}  score={hc_score:.3f}  source={source}")
            enriched.append(row)
        return enriched

    async def fetch_all(rows: list[dict], batch_size: int) -> list[dict]:
        all_enriched = []
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            print(f"\nBatch {i//batch_size + 1} / {math.ceil(len(rows)/batch_size)}")
            enriched = await fetch_batch(batch)
            all_enriched.extend(enriched)
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(rows):
                await asyncio.sleep(1.0)
        return all_enriched

    print(f"\nFetching headcount for {len(to_enrich)} companies (batch size={args.batch})...")
    enriched_rows = asyncio.run(fetch_all(to_enrich, args.batch))

    # Merge enriched rows back into data (match by domain)
    enriched_by_domain = {row["domain"]: row for row in enriched_rows}
    for i, row in enumerate(data):
        if row["domain"] in enriched_by_domain:
            data[i] = enriched_by_domain[row["domain"]]

    # ── Save enriched training data ────────────────────────────────────────────
    with open(TRAINING_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Saved enriched training data → {TRAINING_DATA_PATH}")

    # Summary
    have_hc  = sum(1 for row in data if (row.get("headcount") or 0) > 0)
    missing  = sum(1 for row in data if not (row.get("headcount") or 0) > 0)
    print(f"   Rows with headcount : {have_hc}/{len(data)}")
    print(f"   Still missing       : {missing}/{len(data)}")
    if missing > 0:
        print("   Missing domains:")
        for row in data:
            if not (row.get("headcount") or 0) > 0:
                print(f"     {row['domain']}")

# ── Retrain ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("Retraining model...")
print("="*60)

from model.weights import train_model
train_model()

print("\n✅ Done. Model retrained with headcount signal.")
print("   Test with:")
print("   python -c \"import asyncio,server; r=asyncio.run(server.get_revenue_estimate('notion.so')); print(r.arr_estimate/1e6)\"")