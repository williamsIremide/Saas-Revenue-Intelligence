import asyncio
import json
import math
import os
import uvicorn
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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

app = FastAPI(title="saas-revenue-intelligence")

# ── Tightened timeouts — keeps total fan-out under 30s ───────────────────────
# Previous: hiring=12, pricing=20, reviews=25, traffic=20, headcount=20
# They ran concurrently but scrapers within each signal were sequential.
# Now each signal has a hard ceiling that still allows one good scraper attempt.
_TIMEOUTS = {
    "hiring":    8,   # ashby/greenhouse/lever — fast APIs, 8s is generous
    "pricing":   10,  # single HTTP fetch + Claude fallback
    "reviews":   12,  # G2/Trustpilot race, hard cap
    "traffic":   8,   # Tranco disk lookup — near-instant when cached
    "headcount": 12,  # Crustdata API call
}

_EMPTY_HIRING    = {"open_roles": 0, "engineering_roles": 0, "sales_roles": 0, "source": "timeout", "velocity_score": 0.0, "raw_titles": []}
_EMPTY_PRICING   = {"pricing_model": "unknown", "has_public_pricing": False, "price_points": [], "estimated_acv": 0.0, "currency": "USD", "source_url": "", "confidence": 0.0}
_EMPTY_REVIEWS   = {"total_reviews": 0, "rating": 0.0, "review_velocity_90d": 0, "sentiment_score": 0.0, "momentum_score": 0.0, "source": "timeout", "product_slug": "", "confidence": 0.0}
_EMPTY_TRAFFIC   = {"monthly_visits_estimate": 0, "rank": -1, "rank_score": 0.0, "source": "tranco", "confidence": 0.0}
_EMPTY_HEADCOUNT = {"headcount": 0, "headcount_score": 0.0, "source": "none", "confidence": 0.0, "size_tier": "unknown", "extra": {}}


async def _with_timeout(coro, timeout: int, empty: dict, label: str) -> dict:
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[server] {label} timed out after {timeout}s")
        return empty
    except Exception as e:
        print(f"[server] {label} error: {e}")
        return empty


TOOLS = [
    {
        "name": "get_revenue_estimate",
        "description": (
            "Estimate the Annual Recurring Revenue (ARR) of a SaaS company from its domain. "
            "Returns a calibrated estimate with confidence score and signal breakdown "
            "(headcount, hiring velocity, pricing model, review momentum, traffic rank). "
            "Calibrated against S-1 ARR data. Results cached 24h — fresh fetches take 10-30s."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Company domain to analyse",
                    "default": "notion.so",
                    "examples": ["notion.so", "hubspot.com", "stripe.com", "linear.app"],
                },
                "force_refresh": {
                    "type": "boolean",
                    "description": "Bypass 24-hour cache and fetch fresh signals",
                    "default": False,
                },
            },
            "required": ["domain"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "arr_estimate":     {"type": "number", "description": "Calibrated ARR point estimate in USD"},
                "range_low":        {"type": "number", "description": "Low end of confidence range in USD"},
                "range_high":       {"type": "number", "description": "High end of confidence range in USD"},
                "confidence_score": {"type": "number", "description": "Signal confidence 0.0-1.0"},
                "confidence_label": {"type": "string", "enum": ["Low", "Medium", "High"]},
                "signal_breakdown": {
                    "type": "object",
                    "description": "Per-signal data used to generate the estimate",
                    "properties": {
                        "headcount": {"type": "object"},
                        "hiring":    {"type": "object"},
                        "pricing":   {"type": "object"},
                        "reviews":   {"type": "object"},
                        "traffic":   {"type": "object"},
                    },
                },
                "fetched_at": {"type": "string", "description": "ISO 8601 timestamp"},
            },
            "required": [
                "arr_estimate", "range_low", "range_high",
                "confidence_score", "confidence_label",
                "signal_breakdown", "fetched_at",
            ],
        },
        "_meta": {
            "surface": "both",
            "queryEligible": True,
            "latencyClass": "slow",
            "pricing": {"executeUsd": "0.001"},
            "rateLimit": {
                "maxRequestsPerMinute": 10,
                "cooldownMs": 2000,
                "maxConcurrency": 2,
                "supportsBulk": False,
                "notes": "Results cached 24h. Cached responses return in <3s. Cold-cache fetches run 5 scrapers in parallel with 8-12s individual timeouts, typically completing in 15-25s.",
            },
        },
    }
]


async def run_estimate(domain: str, force_refresh: bool = False) -> dict:
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

    # Pass pricing_model through so predict_arr can apply freemium multiplier
    signals = {
        "headcount":       hc_raw,
        "headcount_score": hc_score,
        "headcount_conf":  headcount.get("confidence", 0.0),
        "open_roles":      open_roles,
        "velocity_score":  hiring["velocity_score"],
        "acv":             pricing["estimated_acv"],
        "pricing_conf":    pricing["confidence"],
        "pricing_model":   pricing["pricing_model"],   # NEW — for freemium multiplier
        "momentum":        reviews["momentum_score"],
        "review_conf":     reviews["confidence"],
        "rank_score":      traffic["rank_score"],
        "traffic_conf":    traffic["confidence"],
    }

    estimate = predict_arr(signals)

    return {
        "arr_estimate":     float(estimate["arr_estimate"]),
        "range_low":        float(estimate["range_low"]),
        "range_high":       float(estimate["range_high"]),
        "confidence_score": float(estimate["confidence_score"]),
        "confidence_label": str(estimate["confidence_label"]),
        "signal_breakdown": {
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
                "total_reviews":     reviews_total,
                "rating":            reviews_rating,
                "momentum_score":    reviews["momentum_score"],
                "source":            reviews["source"],
                "g2_from_crustdata": crustdata_extra.get("g2_reviews", 0) > 0 and reviews["total_reviews"] == 0,
            },
            "traffic": {
                "monthly_visits_estimate": traffic["monthly_visits_estimate"],
                "rank":                    traffic["rank"],
                "rank_score":              traffic["rank_score"],
                "source":                  traffic["source"],
                "confidence":              traffic["confidence"],
            },
        },
        "fetched_at": datetime.utcnow().isoformat(),
    }


async def handle_mcp(body: dict, authorized: bool) -> dict:
    method = body.get("method")
    id_    = body.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "saas-revenue-intelligence", "version": "1.1.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": id_, "result": {"tools": TOOLS}}

    if method == "tools/call":
        if not authorized:
            return {
                "jsonrpc": "2.0", "id": id_,
                "error": {"code": -32001, "message": "Unauthorized"},
            }

        name = body.get("params", {}).get("name")
        args = body.get("params", {}).get("arguments", {})

        if name == "get_revenue_estimate":
            domain = args.get("domain", "").strip()
            if not domain:
                return {
                    "jsonrpc": "2.0", "id": id_,
                    "result": {
                        "content": [{"type": "text", "text": "Error: domain is required"}],
                        "isError": True,
                    },
                }
            try:
                # Hard 50s ceiling — well within Context Protocol's 60s limit
                result = await asyncio.wait_for(
                    run_estimate(domain, force_refresh=args.get("force_refresh", False)),
                    timeout=50,
                )
                return {
                    "jsonrpc": "2.0", "id": id_,
                    "result": {
                        "structuredContent": result,
                        "content": [{"type": "text", "text": json.dumps(result)}],
                    },
                }
            except asyncio.TimeoutError:
                return {
                    "jsonrpc": "2.0", "id": id_,
                    "result": {
                        "content": [{"type": "text", "text": f"Timeout fetching signals for {domain}. Try again — results will be cached after first fetch."}],
                        "isError": True,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0", "id": id_,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0", "id": id_,
            "result": {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True},
        }

    return {"jsonrpc": "2.0", "id": id_, "error": {"code": -32601, "message": "Method not found"}}


@app.get("/health")
async def health():
    return {"status": "ok", "server": "saas-revenue-intelligence", "version": "1.1.0"}


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    body = await request.json()
    method = body.get("method", "")

    authorized = False
    if method == "tools/call":
        auth_header = request.headers.get("authorization", "")
        try:
            await verify_context_request(authorization_header=auth_header)
            authorized = True
        except Exception:
            authorized = False

    result = await handle_mcp(body, authorized)
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))