import asyncio
import os
from datetime import datetime
from pydantic import BaseModel
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from fastmcp.exceptions import ToolError
from dotenv import load_dotenv

# ── Eager imports (fail at startup, not at first request) ─────────────────────
from ctxprotocol import verify_context_request, ContextError

from signals.hiring import get_hiring_signal
from signals.pricing import get_pricing_signal
from signals.reviews import get_reviews_signal
from signals.traffic import get_traffic_signal
from signals.headcount import get_headcount_signal
from model.weights import predict_arr

load_dotenv()

# ── Startup model check ───────────────────────────────────────────────────────
_MODEL_PATH = "model/model.pkl"
if not os.path.exists(_MODEL_PATH):
    print(f"[startup] {_MODEL_PATH} not found — training now...")
    from model.weights import train_model
    train_model()
    print("[startup] model trained.")

# ── App ───────────────────────────────────────────────────────────────────────
mcp = FastMCP("saas-revenue-intelligence")


# ── Auth middleware ───────────────────────────────────────────────────────────
class ContextProtocolAuth(Middleware):
    """
    Verify Context Protocol JWT on every tool/call.
    health_check is a free ($0.00) tool — skip auth and let FastMCP handle it.
    All other tools are paid, so auth is mandatory per Context Protocol rules.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        # health_check is free — pass straight through
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


# ── Response models ───────────────────────────────────────────────────────────
class RevenueEstimate(BaseModel):
    arr_estimate:     float
    range_low:        float
    range_high:       float
    confidence_score: float   # 0.0 – 1.0
    confidence_label: str     # "Low" | "Medium" | "High"
    signal_breakdown: dict
    fetched_at:       str


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Estimate the Annual Recurring Revenue (ARR) of a SaaS company "
        "from its domain. Returns a calibrated estimate with confidence "
        "score and signal breakdown (hiring velocity, pricing model, "
        "review momentum, traffic rank). Calibrated against S-1 ARR data."
    ),
    meta={
        "surface": "both",
        "queryEligible": True,
        "latencyClass": "slow",   # live scraping; use "instant" if you always serve cache first
        "pricing": {
            "executeUsd": "0.001",   # required — without this the tool is invisible in Execute mode
        },
        "rateLimit": {
            "maxRequestsPerMinute": 10,
            "cooldownMs": 2000,
            "maxConcurrency": 2,
            "supportsBulk": False,
            "notes": (
                "Results are cached 24 h. "
                "Fresh fetches (force_refresh=True) take 10–30 s — 4 scrapers run async."
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
        If True, bypass the 24-hour cache and fetch fresh signals
    """
    hiring, pricing, reviews, traffic, headcount = await asyncio.gather(
        get_hiring_signal(domain),
        get_pricing_signal(domain),
        get_reviews_signal(domain),
        get_traffic_signal(domain),
        get_headcount_signal(domain),
    )

    # headcount_score: sqrt-normalised to 0-1 (10k employees = max)
    import math
    hc_raw   = headcount.get("headcount", 0)
    hc_score = min(math.sqrt(hc_raw / 10_000), 1.0) if hc_raw > 0 else 0.0

    signals = {
        "headcount":       hc_raw,
        "headcount_score": hc_score,
        "headcount_conf":  headcount.get("confidence", 0.0),
        "open_roles":      hiring["open_roles"],
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
                "open_roles":     hiring["open_roles"],
                "velocity_score": hiring["velocity_score"],
                "source":         hiring["source"],
            },
            "pricing": {
                "model":         pricing["pricing_model"],
                "estimated_acv": pricing["estimated_acv"],
                "confidence":    pricing["confidence"],
            },
            "reviews": {
                "total_reviews":  reviews["total_reviews"],
                "rating":         reviews["rating"],
                "momentum_score": reviews["momentum_score"],
                "source":         reviews["source"],
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


@mcp.tool(
    description="Returns server health status. Free diagnostic endpoint.",
)
def health_check() -> dict:
    """Check if the server is running."""
    return {"status": "ok", "server": "saas-revenue-intelligence"}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(
        transport="http",
        port=int(os.getenv("PORT", 3000)),
    )