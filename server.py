import asyncio
import json
import math
import os
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
from model.weights import predict_arr, _HEADCOUNT_OVERRIDES

load_dotenv()

_MODEL_PATH = "model/model.pkl"
if not os.path.exists(_MODEL_PATH):
    print(f"[startup] {_MODEL_PATH} not found — training now...")
    from model.weights import train_model
    train_model()
    print("[startup] model trained.")

app = FastAPI()

# ── Per-signal timeouts ────────────────────────────────────────────────────────
_TIMEOUTS = {
    "hiring":    12,
    "pricing":   20,
    "reviews":   25,
    "traffic":   20,
    "headcount": 20,
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


# ── Tool definitions ───────────────────────────────────────────────────────────
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "arr_estimate":     {"type": "number",  "description": "Point estimate of ARR in USD"},
        "range_low":        {"type": "number",  "description": "Lower bound of ARR estimate in USD"},
        "range_high":       {"type": "number",  "description": "Upper bound of ARR estimate in USD"},
        "confidence_score": {"type": "number",  "description": "Confidence score 0.0–1.0"},
        "confidence_label": {"type": "string",  "enum": ["Low", "Medium", "High"], "description": "Human-readable confidence tier based on how many signals were available"},
        "signal_breakdown": {
            "type": "object",
            "description": "Raw signals used to produce the estimate",
            "properties": {
                "headcount": {
                    "type": "object",
                    "properties": {
                        "total":      {"type": "integer", "description": "LinkedIn employee count from Crustdata or fallback sources"},
                        "size_tier":  {"type": "string",  "description": "Company size bucket: micro (<50), small (<200), mid (<1000), large (<5000), enterprise (5000+)"},
                        "source":     {"type": "string",  "description": "Data source used: crustdata, pdl, serpapi, linkedin, or none"},
                        "confidence": {"type": "number",  "description": "Confidence in headcount figure 0.0–1.0 based on source reliability"},
                    },
                    "required": ["total", "size_tier", "source", "confidence"],
                },
                "hiring": {
                    "type": "object",
                    "properties": {
                        "open_roles":        {"type": "integer", "description": "Total open job postings found across all job boards"},
                        "open_roles_source": {"type": "string",  "description": "Which job board or data source provided open_roles"},
                        "velocity_score":    {"type": "number",  "description": "Hiring velocity 0.0–1.0: 0=no roles, 1.0=500+ open roles"},
                        "scraper_source":    {"type": "string",  "description": "Job board scraper that returned results (greenhouse, lever, ashby, etc.)"},
                    },
                    "required": ["open_roles", "open_roles_source", "velocity_score", "scraper_source"],
                },
                "pricing": {
                    "type": "object",
                    "properties": {
                        "model":         {"type": "string",  "description": "Detected pricing model: per_seat, flat, freemium, usage, enterprise, or unknown"},
                        "estimated_acv": {"type": "number",  "description": "Estimated Annual Contract Value in USD based on pricing page"},
                        "confidence":    {"type": "number",  "description": "Confidence in pricing extraction 0.0–1.0"},
                    },
                    "required": ["model", "estimated_acv", "confidence"],
                },
                "reviews": {
                    "type": "object",
                    "properties": {
                        "total_reviews":     {"type": "integer", "description": "Total number of reviews found on G2 or Trustpilot"},
                        "rating":            {"type": "number",  "description": "Average rating out of 5.0"},
                        "momentum_score":    {"type": "number",  "description": "Review momentum 0.0–1.0 combining volume, velocity, and rating"},
                        "source":            {"type": "string",  "description": "Review platform: g2, trustpilot, or none"},
                        "g2_from_crustdata": {"type": "boolean", "description": "True if G2 review count came from Crustdata rather than direct scrape"},
                    },
                    "required": ["total_reviews", "rating", "momentum_score", "source", "g2_from_crustdata"],
                },
                "traffic": {
                    "type": "object",
                    "properties": {
                        "monthly_visits_estimate": {"type": "integer", "description": "Estimated monthly web visits derived from Tranco rank"},
                        "rank":                    {"type": "integer",  "description": "Tranco top-1M rank (lower = more traffic); -1 if not in top 1M"},
                        "rank_score":              {"type": "number",   "description": "Normalised traffic score 0.0–1.0 (rank 1=1.0, rank 1M=0.0)"},
                        "source":                  {"type": "string",   "description": "Traffic data source (always 'tranco')"},
                        "confidence":              {"type": "number",   "description": "Confidence in traffic signal 0.0–1.0"},
                    },
                    "required": ["monthly_visits_estimate", "rank", "rank_score", "source", "confidence"],
                },
            },
            "required": ["headcount", "hiring", "pricing", "reviews", "traffic"],
        },
        "fetched_at": {"type": "string", "description": "ISO timestamp"},
    },
    "required": [
        "arr_estimate", "range_low", "range_high",
        "confidence_score", "confidence_label",
        "signal_breakdown", "fetched_at",
    ],
}

TOOLS = [
    {
        "name": "get_revenue_estimate",
        "description": (
            "Estimate the Annual Recurring Revenue (ARR) of a SaaS company "
            "from its domain. Returns a calibrated estimate with confidence "
            "score and signal breakdown (headcount, hiring velocity, pricing "
            "model, review momentum, traffic rank). Calibrated against S-1 ARR data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Company domain to analyse",
                    "default": "notion.so",
                    "examples": ["notion.so", "hubspot.com", "linear.app"],
                },
                "force_refresh": {
                    "type": "boolean",
                    "description": "Bypass 24-hour cache and fetch fresh signals",
                    "default": False,
                },
            },
            "required": ["domain"],
        },
        "outputSchema": OUTPUT_SCHEMA,
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
                "notes": "Results cached 24h. Fresh fetches take 10–30s.",
            },
        },
    }
]


# ── Core estimation logic ──────────────────────────────────────────────────────
async def _estimate(domain: str, force_refresh: bool = False) -> dict:
    hiring, pricing, reviews, traffic, headcount = await asyncio.gather(
        _with_timeout(get_hiring_signal(domain,    force_refresh=force_refresh), _TIMEOUTS["hiring"],    _EMPTY_HIRING,    "hiring"),
        _with_timeout(get_pricing_signal(domain,   force_refresh=force_refresh), _TIMEOUTS["pricing"],   _EMPTY_PRICING,   "pricing"),
        _with_timeout(get_reviews_signal(domain,   force_refresh=force_refresh), _TIMEOUTS["reviews"],   _EMPTY_REVIEWS,   "reviews"),
        _with_timeout(get_traffic_signal(domain,   force_refresh=force_refresh), _TIMEOUTS["traffic"],   _EMPTY_TRAFFIC,   "traffic"),
        _with_timeout(get_headcount_signal(domain, force_refresh=force_refresh), _TIMEOUTS["headcount"], _EMPTY_HEADCOUNT, "headcount"),
    )

    crustdata_extra = headcount.get("extra", {})
    hc_raw = headcount.get("headcount", 0)
    domain_normalized = domain.strip().lower().replace("www.", "")
    if domain_normalized in _HEADCOUNT_OVERRIDES:
        hc_raw = _HEADCOUNT_OVERRIDES[domain_normalized]
    hc_score = min(math.sqrt(hc_raw / 10_000), 1.0) if hc_raw > 0 else 0.0

    open_roles = hiring["open_roles"]
    if open_roles == 0 and crustdata_extra.get("job_openings_count", 0) > 0:
        open_roles = crustdata_extra["job_openings_count"]

    reviews_total  = reviews["total_reviews"]
    reviews_rating = reviews["rating"]
    if reviews_total == 0 and crustdata_extra.get("g2_reviews", 0) > 0:
        reviews_total  = crustdata_extra["g2_reviews"]
        reviews_rating = crustdata_extra.get("g2_rating", 0.0)
        if reviews_rating > 5.0:
            reviews_rating = round(reviews_rating / 2, 2)

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

    return {
        "arr_estimate":     estimate["arr_estimate"],
        "range_low":        estimate["range_low"],
        "range_high":       estimate["range_high"],
        "confidence_score": estimate["confidence_score"],
        "confidence_label": estimate["confidence_label"],
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


# ── MCP request handler ────────────────────────────────────────────────────────
async def handle_mcp(body: dict, authorized: bool = False) -> dict:
    method = body.get("method")
    id_    = body.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "saas-revenue-intelligence", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": id_, "result": {"tools": TOOLS}}

    if method == "tools/call":
        if not authorized:
            return {
                "jsonrpc": "2.0", "id": id_,
                "error": {"code": -32001, "message": "Unauthorized: Missing or invalid Authorization header"},
            }

        name = body["params"]["name"]
        args = body["params"].get("arguments", {})

        if name == "get_revenue_estimate":
            domain        = args.get("domain", "")
            force_refresh = args.get("force_refresh", False)

            if not domain:
                return {
                    "jsonrpc": "2.0", "id": id_,
                    "result": {
                        "content": [{"type": "text", "text": "Error: domain is required"}],
                        "isError": True,
                    },
                }

            try:
                # Hard 55s ceiling — stays within Context Protocol's 60s Query limit
                result = await asyncio.wait_for(_estimate(domain, force_refresh=force_refresh), timeout=55)
            except asyncio.TimeoutError:
                return {
                    "jsonrpc": "2.0", "id": id_,
                    "result": {
                        "content": [{"type": "text", "text": f"Timeout: signals for {domain} took too long. Try again — results will be cached."}],
                        "isError": True,
                    },
                }

            return {
                "jsonrpc": "2.0", "id": id_,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "structuredContent": result,   # ← required by Context Protocol
                },
            }

        return {
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            },
        }

    return {"jsonrpc": "2.0", "id": id_, "error": {"code": -32601, "message": "Method not found"}}


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "server": "saas-revenue-intelligence", "version": "1.0.0"}


@app.post("/mcp")
async def mcp_post(request: Request):
    body = await request.json()
    method = body.get("method", "")

    # tools/call requires auth; initialize and tools/list are open
    authorized = False
    if method == "tools/call":
        auth_header = request.headers.get("authorization", "")
        try:
            await verify_context_request(authorization_header=auth_header)
            authorized = True
        except Exception:
            authorized = False

    result = await handle_mcp(body, authorized=authorized)
    return JSONResponse(result)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))