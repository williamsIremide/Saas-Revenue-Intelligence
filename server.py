import asyncio
import os
import math
from datetime import datetime
from pydantic import BaseModel
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from fastmcp.exceptions import ToolError
from dotenv import load_dotenv

from ctxprotocol import verify_context_request, ContextError

from signals.hiring import get_hiring_signal
from signals.pricing import get_pricing_signal
from signals.reviews import get_reviews_signal
from signals.traffic import get_traffic_signal
from signals.headcount import get_headcount_signal
from model.weights import predict_arr

load_dotenv()

_MODEL_PATH = "model/model.pkl"
if not os.path.exists(_MODEL_PATH):
    print(f"[startup] {_MODEL_PATH} not found — training now...")
    from model.weights import train_model
    train_model()
    print("[startup] model trained.")

mcp = FastMCP("saas-revenue-intelligence")

# ── Per-signal timeouts (seconds) ─────────────────────────────────────────────
# Each signal gets its own budget. Reviews is the slowest scraper so gets most.
# Headcount/hiring/pricing/traffic are fast APIs or cached — 15s is generous.
_TIMEOUTS = {
    "hiring":    12,
    "pricing":   20,
    "reviews":   25,
    "traffic":   20,   # first run downloads Tranco list (~10s), then disk-cached
    "headcount": 20,
}

_EMPTY_HIRING    = {"open_roles": 0, "engineering_roles": 0, "sales_roles": 0, "source": "timeout", "velocity_score": 0.0, "raw_titles": []}
_EMPTY_PRICING   = {"pricing_model": "unknown", "has_public_pricing": False, "price_points": [], "estimated_acv": 0.0, "currency": "USD", "source_url": "", "confidence": 0.0}
_EMPTY_REVIEWS   = {"total_reviews": 0, "rating": 0.0, "review_velocity_90d": 0, "sentiment_score": 0.0, "momentum_score": 0.0, "source": "timeout", "product_slug": "", "confidence": 0.0}
_EMPTY_TRAFFIC   = {"monthly_visits_estimate": 0, "rank": -1, "rank_score": 0.0, "source": "tranco", "confidence": 0.0}
_EMPTY_HEADCOUNT = {"headcount": 0, "headcount_score": 0.0, "source": "none", "confidence": 0.0, "size_tier": "unknown", "extra": {}}


async def _with_timeout(coro, timeout: int, empty: dict, label: str) -> dict:
    """Run a signal coroutine with a hard timeout, returning empty dict on failure."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[server] {label} timed out after {timeout}s")
        return empty
    except Exception as e:
        print(f"[server] {label} error: {e}")
        return empty


class ContextProtocolAuth(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        if context.message.params.name == "health_check":
            return await call_next(context)
        headers = get_http_headers()
        auth_header = headers.get("authorization", "")
        try:
            await verify_context_request(authorization_header=auth_header)
        except ContextError as e:
            raise ToolError(f"Unauthorized: {e.message}")
        except Exception as e:
            raise ToolError(f"Unauthorized: {e}")
        return await call_next(context)


mcp.add_middleware(ContextProtocolAuth())


class RevenueEstimate(BaseModel):
    arr_estimate:     float
    range_low:        float
    range_high:       float
    confidence_score: float
    confidence_label: str
    signal_breakdown: dict
    fetched_at:       str


@mcp.tool(
    description=(
        "Estimate the Annual Recurring Revenue (ARR) of a SaaS company "
        "from its domain. Returns a calibrated estimate with confidence "
        "score and signal breakdown (headcount, hiring velocity, pricing "
        "model, review momentum, traffic rank). Calibrated against S-1 ARR data."
    ),
    meta={
        "surface": "both",
        "queryEligible": True,
        "latencyClass": "slow",
        "pricing": {
            "executeUsd": "0.001",
        },
        "rateLimit": {
            "maxRequestsPerMinute": 10,
            "cooldownMs": 2000,
            "maxConcurrency": 2,
            "supportsBulk": False,
            "notes": (
                "Results are cached 24 h. "
                "Fresh fetches (force_refresh=True) take 10–30 s — 5 signals run async."
            ),
        },
    },
)
async def get_revenue_estimate(
    domain: str,
    force_refresh: bool = False,
) -> RevenueEstimate:
    """
    Parameters
    ----------
    domain : str
        Company domain to analyse, e.g. 'notion.so' or 'hubspot.com'
    force_refresh : bool
        If True, bypass the 24-hour cache and fetch fresh signals for all sources.
    """
    # Run all 5 signals concurrently, each with its own hard timeout.
    # asyncio.gather runs them in parallel — total wall time = slowest signal, not sum.
    hiring, pricing, reviews, traffic, headcount = await asyncio.gather(
        _with_timeout(get_hiring_signal(domain,    force_refresh=force_refresh), _TIMEOUTS["hiring"],    _EMPTY_HIRING,    "hiring"),
        _with_timeout(get_pricing_signal(domain,   force_refresh=force_refresh), _TIMEOUTS["pricing"],   _EMPTY_PRICING,   "pricing"),
        _with_timeout(get_reviews_signal(domain,   force_refresh=force_refresh), _TIMEOUTS["reviews"],   _EMPTY_REVIEWS,   "reviews"),
        _with_timeout(get_traffic_signal(domain,   force_refresh=force_refresh), _TIMEOUTS["traffic"],   _EMPTY_TRAFFIC,   "traffic"),
        _with_timeout(get_headcount_signal(domain, force_refresh=force_refresh), _TIMEOUTS["headcount"], _EMPTY_HEADCOUNT, "headcount"),
    )

    crustdata_extra = headcount.get("extra", {})

    hc_raw   = headcount.get("headcount", 0)
    hc_score = min(math.sqrt(hc_raw / 10_000), 1.0) if hc_raw > 0 else 0.0

    open_roles = hiring["open_roles"]
    if open_roles == 0 and crustdata_extra.get("job_openings_count", 0) > 0:
        open_roles = crustdata_extra["job_openings_count"]

    reviews_total  = reviews["total_reviews"]
    reviews_rating = reviews["rating"]
    if reviews_total == 0 and crustdata_extra.get("g2_reviews", 0) > 0:
        reviews_total  = crustdata_extra["g2_reviews"]
        reviews_rating = crustdata_extra.get("g2_rating", 0.0)

    signals = {
        "headcount":       hc_raw,
        "headcount_score": hc_score,
        "headcount_conf":  headcount.get("confidence", 0.0),
        "open_roles":      open_roles,
        "velocity_score":  hiring["velocity_score"],
        "acv":             pricing["estimated_acv"],
        "pricing_conf":    pricing["confidence"],
        "momentum":        reviews["momentum_score"],
        "review_conf":     reviews["confidence"],
        "rank_score":      traffic["rank_score"],
        "traffic_conf":    traffic["confidence"],
    }

    estimate = predict_arr(signals)

    return RevenueEstimate(
        arr_estimate=estimate["arr_estimate"],
        range_low=estimate["range_low"],
        range_high=estimate["range_high"],
        confidence_score=estimate["confidence_score"],
        confidence_label=estimate["confidence_label"],
        signal_breakdown={
            "headcount": {
                "total":      hc_raw,
                "size_tier":  headcount.get("size_tier", "unknown"),
                "source":     headcount.get("source", "none"),
                "confidence": headcount.get("confidence", 0.0),
            },
            "hiring": {
                "open_roles":        open_roles,
                "open_roles_source": "crustdata" if hiring["open_roles"] == 0 and open_roles > 0 else hiring["source"],
                "velocity_score":    hiring["velocity_score"],
                "scraper_source":    hiring["source"],
            },
            "pricing": {
                "model":         pricing["pricing_model"],
                "estimated_acv": pricing["estimated_acv"],
                "confidence":    pricing["confidence"],
            },
            "reviews": {
                "total_reviews":       reviews_total,
                "rating":              reviews_rating,
                "momentum_score":      reviews["momentum_score"],
                "source":              reviews["source"],
                "g2_from_crustdata":   crustdata_extra.get("g2_reviews", 0) > 0 and reviews["total_reviews"] == 0,
            },
            "traffic": {
                "monthly_visits_estimate": traffic["monthly_visits_estimate"],
                "rank":                    traffic["rank"],
                "rank_score":              traffic["rank_score"],
                "source":                  traffic["source"],
                "confidence":              traffic["confidence"],
            },
        },
        fetched_at=datetime.utcnow().isoformat(),
    )


@mcp.tool(description="Returns server health status. Free diagnostic endpoint.")
def health_check() -> dict:
    return {"status": "ok", "server": "saas-revenue-intelligence"}


if __name__ == "__main__":
    mcp.run(
        transport="http",
        port=int(os.getenv("PORT", 3000)),
    )