"""
patch_timeouts.py — Reduce httpx timeouts in hiring.py and pricing.py.

The hiring fetcher tries 6 job boards. At 10s each that's 60s before the
server-level 15s timeout kills it — meaning hiring always times out on
companies not using Greenhouse/Lever/etc.

Fix: reduce per-request timeout to 5s in hiring and pricing fetchers.
All 6 hiring boards then fail-fast in ~5s total (they run with as_completed).

Run: python patch_timeouts.py
"""
import re

PATCHES = [
    {
        "file": "signals/hiring.py",
        "old":  'timeout=10,\n        headers={"User-Agent": "Mozilla/5.0"},',
        "new":  'timeout=5,\n        headers={"User-Agent": "Mozilla/5.0"},',
        "desc": "hiring httpx timeout 10s → 5s",
    },
    {
        "file": "signals/pricing.py",
        "old":  'timeout=10,\n        headers={"User-Agent": "Mozilla/5.0"},\n        follow_redirects=True,',
        "new":  'timeout=6,\n        headers={"User-Agent": "Mozilla/5.0"},\n        follow_redirects=True,',
        "desc": "pricing httpx timeout 10s → 6s",
    },
]

for patch in PATCHES:
    with open(patch["file"], "r") as f:
        content = f.read()

    if patch["old"] not in content:
        print(f"  SKIP {patch['file']} — pattern not found (may already be patched)")
        continue

    content = content.replace(patch["old"], patch["new"], 1)
    with open(patch["file"], "w") as f:
        f.write(content)
    print(f"  OK   {patch['file']} — {patch['desc']}")

print("\nDone. Now test:")
print('  python -c "import asyncio,time,server; t=__import__(\'time\').time(); asyncio.run(server.get_revenue_estimate(\'pagerduty.com\')); print(time.time()-t)"')