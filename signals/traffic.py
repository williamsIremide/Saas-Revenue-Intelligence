"""
traffic.py — Real web-traffic signal using the Tranco top-1M domain ranking.

Tranco (https://tranco-list.eu) aggregates Alexa, Cisco Umbrella, Majestic,
and Farsight DNSDB into a single weekly-averaged rank list. It is:
  - Free, no API key required
  - Research-grade (cited in 800+ academic papers)
  - Updated daily, top-1M domains

Strategy
--------
1. On first call (or after TRANCO_CACHE_DAYS), download the latest Tranco CSV
   (~10 MB compressed) into a local file cache at TRANCO_CACHE_DIR.
2. Build an in-process dict { domain -> rank } from the full 1M rows.
3. Look up the requested domain (try bare domain, then www-stripped, then
   the apex from any subdomain).
4. Convert rank to a 0-1 score on a log scale and estimate monthly visits.
5. Cache the per-domain result in Redis (same TTL as other signals).

The list is loaded once into module-level memory and reused across requests,
so the 10 MB CSV only parses once per process lifetime.
"""

import asyncio
import csv
import io
import logging
import math
import os
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from signals.cache import CACHE_VERSION, get_cache, set_cache, USE_CACHE

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Where to store the downloaded Tranco CSV (survives process restarts)
TRANCO_CACHE_DIR  = Path(os.getenv("TRANCO_CACHE_DIR", ".tranco"))
# Re-download the list after this many days
TRANCO_CACHE_DAYS = int(os.getenv("TRANCO_CACHE_DAYS", "3"))
# Tranco API base
TRANCO_API        = "https://tranco-list.eu"

# ── Module-level list cache (loaded once per process) ────────────────────────
_rank_dict: dict[str, int] = {}       # domain -> rank (1-indexed, lower = better)
_list_loaded_at: float     = 0.0      # unix timestamp of last load


# ── Tranco download helpers ───────────────────────────────────────────────────

def _tranco_meta_path() -> Path:
    return TRANCO_CACHE_DIR / "meta.txt"


def _tranco_csv_path(list_id: str) -> Path:
    return TRANCO_CACHE_DIR / f"{list_id}.csv"


def _cached_list_id() -> str | None:
    """Return the list_id stored from the last successful download, or None."""
    meta = _tranco_meta_path()
    if not meta.exists():
        return None
    try:
        parts = meta.read_text().strip().split("|")
        list_id, ts = parts[0], float(parts[1])
        if time.time() - ts < TRANCO_CACHE_DAYS * 86400:
            csv_path = _tranco_csv_path(list_id)
            if csv_path.exists():
                return list_id
    except Exception:
        pass
    return None


def _save_meta(list_id: str) -> None:
    TRANCO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _tranco_meta_path().write_text(f"{list_id}|{time.time()}")


async def _fetch_list_id(client: httpx.AsyncClient) -> str:
    """Ask Tranco for the most recent available daily list ID."""
    # Try the last 7 days to find one that's published
    for days_back in range(1, 8):
        date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            r = await client.get(
                f"{TRANCO_API}/daily_list_id",
                params={"date": date, "subdomains": "false"},
                timeout=10,
            )
            if r.status_code == 200 and r.text.strip():
                return r.text.strip()
        except Exception as e:
            logger.debug(f"[traffic] list_id fetch {date}: {e}")
    raise RuntimeError("Could not retrieve a recent Tranco list ID after 7 attempts.")


async def _download_list(list_id: str, client: httpx.AsyncClient) -> None:
    """Download the Tranco ZIP for list_id and save as a plain CSV."""
    TRANCO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _tranco_csv_path(list_id)

    # Try ZIP endpoint first (faster)
    zip_url = f"{TRANCO_API}/download_daily/{list_id}"
    try:
        r = await client.get(zip_url, timeout=60)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open("top-1m.csv") as f:
                    csv_path.write_bytes(f.read())
            logger.info(f"[traffic] Downloaded Tranco list {list_id} via ZIP.")
            _save_meta(list_id)
            return
    except Exception as e:
        logger.debug(f"[traffic] ZIP download failed ({e}), trying plain CSV...")

    # Fallback: plain CSV endpoint
    plain_url = f"{TRANCO_API}/download/{list_id}/1000000"
    r = await client.get(plain_url, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"Tranco plain CSV download failed: HTTP {r.status_code}")
    csv_path.write_bytes(r.content)
    logger.info(f"[traffic] Downloaded Tranco list {list_id} via plain CSV.")
    _save_meta(list_id)


def _load_rank_dict(list_id: str) -> dict[str, int]:
    """Parse the CSV into { domain: rank }.  Runs once per process."""
    csv_path = _tranco_csv_path(list_id)
    ranks: dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    ranks[row[1].lower().strip()] = int(row[0])
                except ValueError:
                    continue
    logger.info(f"[traffic] Loaded {len(ranks):,} domains from Tranco list {list_id}.")
    return ranks


async def _ensure_list_loaded() -> None:
    """Download + parse the Tranco list if it isn't already in memory."""
    global _rank_dict, _list_loaded_at

    # Already loaded and fresh enough?
    if _rank_dict and (time.time() - _list_loaded_at) < TRANCO_CACHE_DAYS * 86400:
        return

    # Check local file cache first (avoids re-download after process restart)
    cached_id = _cached_list_id()
    if cached_id:
        if not _rank_dict:                         # parse into memory if needed
            _rank_dict      = _load_rank_dict(cached_id)
            _list_loaded_at = time.time()
        return

    # Need a fresh download
    async with httpx.AsyncClient(
        headers={"User-Agent": "saas-revenue-intelligence/1.0"},
        follow_redirects=True,
    ) as client:
        list_id = await _fetch_list_id(client)
        await _download_list(list_id, client)
        _rank_dict      = _load_rank_dict(list_id)
        _list_loaded_at = time.time()


# ── Rank lookup ───────────────────────────────────────────────────────────────

def _normalize(domain: str) -> str:
    return (
        domain.strip()
        .lower()
        .replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .split("/")[0]
    )


def _lookup_rank(domain: str) -> int:
    """
    Try the domain as-is, then the apex (e.g. app.notion.so -> notion.so).
    Returns -1 if not in the top-1M list.
    """
    d = _normalize(domain)
    if d in _rank_dict:
        return _rank_dict[d]

    # Try apex domain (last two labels)
    parts = d.split(".")
    if len(parts) > 2:
        apex = ".".join(parts[-2:])
        if apex in _rank_dict:
            return _rank_dict[apex]

    return -1


# ── Score conversion ──────────────────────────────────────────────────────────

def _rank_to_score(rank: int) -> float:
    """
    Convert Tranco rank to a 0-1 score on a log scale.
    rank=1 -> 1.0, rank=1,000,000 -> ~0.0, rank=-1 (unlisted) -> 0.0
    """
    if rank <= 0:
        return 0.0
    score = 1.0 - (math.log10(rank) / 6.0)
    return round(max(0.0, min(1.0, score)), 3)


def _rank_to_monthly_visits(rank: int) -> int:
    """
    Estimate monthly visits from rank using a log-linear interpolation
    anchored to known Similarweb reference points:
      rank ~1     → 100M visits/mo
      rank ~1,000 → ~1M visits/mo
      rank ~10k   → ~150k visits/mo
      rank ~100k  → ~20k visits/mo
      rank ~1M    → ~5k visits/mo
    """
    if rank <= 0:
        return 0
    log_rank   = math.log10(max(rank, 1))
    log_visits = 8.0 - (0.717 * log_rank)
    return int(10 ** log_visits)


def _rank_to_confidence(rank: int) -> float:
    """
    Confidence in the traffic signal.
    Listed domains get full confidence; unlisted get 0.
    Low-rank (high-traffic) domains get slightly higher confidence
    because the Tranco aggregation is more stable for popular sites.
    """
    if rank <= 0:
        return 0.0
    if rank <= 10_000:
        return 0.9
    if rank <= 100_000:
        return 0.75
    return 0.6


# ── Public API ────────────────────────────────────────────────────────────────

async def get_traffic_signal(domain: str) -> dict:
    """
    Return a traffic signal dict for the given domain.

    Keys
    ----
    monthly_visits_estimate : int    estimated monthly unique visits
    rank                    : int    Tranco rank (1=highest traffic, -1=unlisted)
    rank_score              : float  0-1 score (independent of hiring)
    source                  : str    always "tranco"
    confidence              : float  0-1 confidence in the estimate
    """
    normalized = _normalize(domain)
    cache_key  = f"traffic:{CACHE_VERSION}:{normalized}"

    cached = get_cache(cache_key)
    if USE_CACHE and cached:
        logger.debug(f"[traffic] cache hit: {normalized}")
        return cached

    empty = {
        "monthly_visits_estimate": 0,
        "rank":                    -1,
        "rank_score":              0.0,
        "source":                  "tranco",
        "confidence":              0.0,
    }

    try:
        await _ensure_list_loaded()
    except Exception as e:
        logger.error(f"[traffic] Could not load Tranco list: {e}")
        return empty

    rank      = _lookup_rank(domain)
    score     = _rank_to_score(rank)
    visits    = _rank_to_monthly_visits(rank)
    confidence = _rank_to_confidence(rank)

    result = {
        "monthly_visits_estimate": visits,
        "rank":                    rank,
        "rank_score":              score,
        "source":                  "tranco",
        "confidence":              confidence,
    }

    # Only cache positive results — unlisted might just be a new domain
    if rank > 0:
        set_cache(cache_key, result)

    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    test_domains = [
        "notion.so",
        "hubspot.com",
        "linear.app",
        "salesforce.com",
        "plausible.io",
        "vercel.com",
        "beehiiv.com",
        "carrd.co",
    ]

    async def _test():
        for d in test_domains:
            result = await get_traffic_signal(d)
            print(
                f"{d:30s}  rank={result['rank']:>9,}  "
                f"score={result['rank_score']:.4f}  source={result['source']}"
            )

    asyncio.run(_test())