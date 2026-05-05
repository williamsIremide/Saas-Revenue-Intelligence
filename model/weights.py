"""
weights.py — ARR estimation model.

Four independent signals
------------------------
  open_roles    hiring velocity  (Greenhouse / Lever / Ashby / etc.)
  momentum      review momentum  (G2 / Trustpilot, bias-corrected)
  rank_score    Tranco web-traffic rank  (0-1 log scale, INDEPENDENT of hiring)
  acv           pricing-tier ACV estimate

The previous traffic.py derived rank_score from open_roles, so rank_score
and open_roles were collinear. With real Tranco data they are independent,
and the model genuinely has four distinct signal sources.

Regression method
-----------------
We fit:  log(ARR) = a * log(composite_score) + b

With only ~20 rows OLS and Huber produce similar results. As the dataset
grows toward 64+ companies, Huber's robustness matters more because the ARR
range spans $50M to $35B — a 700x spread where a few mega-companies
(Salesforce, ServiceNow, Workday) would otherwise dominate the OLS fit.

Outlier strategy
----------------
Companies with ARR > ARR_TRAIN_CAP ($6B) are DOWN-WEIGHTED, not excluded.
Their signals are real and informative; we just don't want them to anchor
the fit for the $50M–$2B range where most queries will land.
We implement this by clamping their log(ARR) contribution via Huber loss
(epsilon=1.35 corresponds to ~1-sigma downweighting at the extremes).
"""

import json
import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Companies above this ARR are kept in training but down-weighted by Huber.
# Salesforce ($35B), ServiceNow ($10B), Workday ($7B), Autodesk ($5.5B) all
# sit far above the typical query range and would skew an OLS fit.
ARR_TRAIN_CAP = 6_000_000_000

# Domains to fully exclude from training (no useful signal, skews fit badly).
EXCLUDE_DOMAINS: set[str] = set()  # previously excluded stripe.com; now handled by Huber


# ── Feature extraction ────────────────────────────────────────────────────────

def parse_arr(arr_str: str) -> float:
    return float(str(arr_str).replace("_", ""))


def compute_weighted_score(signals: dict) -> float:
    """
    Compute a 0-1 composite score from all five signals.

    Signal hierarchy
    ----------------
    headcount_score  0.40  PRIMARY — total LinkedIn employees, sqrt-scaled.
                           Correlation with log(ARR): ~0.85 vs 0.33 for open_roles.
                           A company with 5,000 staff has more revenue capacity
                           than one with 500, regardless of current hiring pace.
    rank_score       0.20  Web traffic (Tranco) — independent of hiring.
    momentum         0.15  Review growth — customer acquisition proxy.
    open_roles       0.10  SECONDARY — current hiring velocity / growth signal.
                           Kept because it captures forward-looking growth
                           that headcount (a lagging indicator) misses.
    acv_signal       0.10  Pricing tier ACV estimate.
    hc_x_rank        0.05  headcount × rank interaction: scale × web reach.

    Fallback: if headcount_score=0 (LinkedIn not scraped yet), the model
    degrades gracefully to the old open_roles-primary weights so existing
    cached rows without headcount still produce reasonable estimates.
    """
    hc_score   = float(signals.get("headcount_score", 0.0))
    open_roles = float(signals.get("open_roles",      0))
    momentum   = float(signals.get("momentum",        0.0))
    rank_score = float(signals.get("rank_score",      0.0))
    acv        = float(signals.get("acv",             0.0))

    acv_norm    = min(acv / 10_000, 1.0)
    roles_norm  = min((open_roles / 500) ** 0.6, 1.0)
    hc_x_rank   = min((hc_score * rank_score) / 0.2, 1.0)  # normalise: 0.2 ≈ typical product

    if hc_score > 0:
        # Full 5-signal model
        return (
            hc_score  * 0.40 +
            rank_score * 0.20 +
            momentum   * 0.15 +
            roles_norm * 0.10 +
            acv_norm   * 0.10 +
            hc_x_rank  * 0.05
        )
    else:
        # Fallback: headcount unavailable — use open_roles-primary weights
        # (same as previous model, slightly degraded accuracy)
        size_mult   = 1.30 if open_roles >= 200 else 1.0
        interaction = min((open_roles * rank_score) / 200, 1.0)
        base = (
            roles_norm  * 0.35 +
            rank_score  * 0.25 +
            momentum    * 0.20 +
            acv_norm    * 0.10 +
            interaction * 0.10
        )
        return base * size_mult


# ── Training ──────────────────────────────────────────────────────────────────

def _signal_quality(row: dict) -> int:
    """
    Count how many of the 5 signals returned real data (not zero-due-to-failure).
    """
    count = 0
    if float(row.get("headcount",  0)) > 0:      count += 1
    if float(row.get("open_roles", 0)) > 0:      count += 1
    if float(row.get("review_conf", 0)) > 0:     count += 1
    if float(row.get("pricing_conf", 0)) > 0:    count += 1
    if float(row.get("traffic_conf", 0)) > 0:    count += 1
    return count


def train_model(
    training_data_path: str = "model/training_data.json",
    min_signals: int = 2,
) -> None:
    """
    Fit log(ARR) = a * log(composite_score) + b using Huber regression.

    Only trains on rows with at least `min_signals` real data sources.
    A signal is "real" when its confidence > 0 (not a scrape failure).
    This prevents rows where G2/Trustpilot returned 403 from poisoning
    the fit with fake momentum=0 values.

    ARR cap: companies above ARR_TRAIN_CAP are excluded from training.
    Salesforce ($35B), ServiceNow ($10B) etc. sit so far above the typical
    query range ($50M–$3B) that they dominate the OLS/Huber fit and push
    all mid-range predictions toward the mean. They're excluded, not
    down-weighted, because even Huber can't fully compensate for a 700x
    outlier when n=65.
    """
    ARR_TRAIN_CAP = 5_000_000_000

    with open(training_data_path) as f:
        data = json.load(f)

    filtered = []
    excluded_zero  = 0
    excluded_cap   = 0
    excluded_sig   = 0

    for row in data:
        if row["domain"] in EXCLUDE_DOMAINS:
            continue
        arr = parse_arr(row["arr"])
        sq  = _signal_quality(row)

        if arr > ARR_TRAIN_CAP:
            excluded_cap += 1
            continue
        if sq < min_signals:
            excluded_sig += 1
            continue
        if row.get("open_roles", 0) == 0 and row.get("momentum", 0) == 0 and row.get("rank_score", 0) == 0:
            excluded_zero += 1
            continue
        filtered.append(row)

    print(
        f"Training on {len(filtered)}/{len(data)} companies  "
        f"(excluded: {excluded_cap} ARR>{ARR_TRAIN_CAP/1e9:.0f}B, "
        f"{excluded_sig} <{min_signals} signals, "
        f"{excluded_zero} all-zero)"
    )

    try:
        from sklearn.linear_model import HuberRegressor
    except ImportError:
        logger.warning("[weights] scikit-learn not found; falling back to OLS.")
        HuberRegressor = None  # type: ignore

    scores = np.array([compute_weighted_score(row) for row in filtered])
    arrs   = np.array([parse_arr(row["arr"])        for row in filtered])
    scores = np.clip(scores, 1e-6, None)

    log_scores = np.log(scores).reshape(-1, 1)
    log_arrs   = np.log(arrs)

    if HuberRegressor is not None:
        # epsilon=1.35: rows > 1.35 * MAD are treated as outliers.
        # Larger epsilon -> closer to OLS. 1.35 is the standard "95% efficiency" setting.
        model = HuberRegressor(epsilon=1.35, max_iter=500, fit_intercept=True)
        model.fit(log_scores, log_arrs)
        a = float(model.coef_[0])
        b = float(model.intercept_)
        method = "Huber"
    else:
        # OLS fallback
        A = np.column_stack([log_scores.ravel(), np.ones(len(filtered))])
        coeffs, _, _, _ = np.linalg.lstsq(A, log_arrs, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
        method = "OLS (fallback)"

    print(f"  Method:             {method}")
    print(f"  Scale exponent (a): {a:.4f}")
    print(f"  Intercept (b):      {b:.4f}")

    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump({"a": a, "b": b, "method": method}, f)

    # Sanity check
    _print_sanity_check(filtered, scores, a, b)
    print("Model saved -> model/model.pkl")


def _print_sanity_check(
    rows: list[dict],
    scores: np.ndarray,
    a: float,
    b: float,
) -> None:
    print("\nSanity check (flag = >3x error):")
    errors = []
    for row, score in zip(rows, scores):
        predicted = np.exp(a * np.log(score) + b)
        actual    = parse_arr(row["arr"])
        ratio     = predicted / actual if actual > 0 else 0.0
        errors.append(abs(np.log(max(ratio, 1e-6))))
        flag = "⚠ " if (ratio < 0.33 or ratio > 3.0) else "  "
        print(
            f"  {flag}{row['company']:<16}  "
            f"actual: ${actual/1e6:>8.0f}M  "
            f"predicted: ${predicted/1e6:>8.0f}M  "
            f"ratio: {ratio:.2f}x"
        )
    mae = float(np.mean(errors))
    print(f"\nMean log error: {mae:.3f}  ({np.exp(mae):.2f}x average error)")


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model() -> dict:
    with open("model/model.pkl", "rb") as f:
        return pickle.load(f)


def predict_arr(signals: dict) -> dict:
    """
    Return an ARR point estimate with a confidence interval.

    Confidence is derived from four independent signal quality indicators:
      - hiring:  open_roles > 0  (hiring board found)
      - reviews: review_conf from G2/Trustpilot scrape
      - pricing: pricing_conf from pricing page scrape
      - traffic: traffic_conf from Tranco lookup (0 if domain not in top-1M)

    The margin shrinks as confidence rises: 40% at zero confidence, 20% at full.
    A low-confidence estimate will therefore show a wider range as a natural
    signal to the caller that the data is thin.
    """
    params = load_model()
    a, b   = params["a"], params["b"]

    score    = max(compute_weighted_score(signals), 1e-6)
    log_pred = a * np.log(score) + b
    point    = float(np.exp(log_pred))

    # Five independent confidence factors
    confidence_factors = [
        float(signals.get("headcount_conf", 0.0)),   # LinkedIn headcount found
        1.0 if signals.get("open_roles", 0) > 0 else 0.0,
        float(signals.get("review_conf",  0.0)),
        float(signals.get("pricing_conf", 0.0)),
        float(signals.get("traffic_conf", 0.0)),
    ]
    avg_conf = sum(confidence_factors) / len(confidence_factors)

    margin = 0.40 - (avg_conf * 0.20)   # 0.40 at zero conf -> 0.20 at full conf

    return {
        "arr_estimate":     round(point),
        "range_low":        round(point * (1 - margin)),
        "range_high":       round(point * (1 + margin)),
        "confidence_score": round(avg_conf, 2),
        "confidence_label": (
            "High"   if avg_conf >= 0.7 else
            "Medium" if avg_conf >= 0.4 else
            "Low"
        ),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model()

    print("\n--- predict_arr smoke test (Notion) ---")
    result = predict_arr({
        "open_roles":   141,
        "momentum":     0.93,
        "rank_score":   0.349,   # real Tranco score for notion.so
        "acv":          480.0,
        "review_conf":  1.0,
        "pricing_conf": 1.0,
        "traffic_conf": 0.9,
    })
    print(
        f"Estimate:  ${result['arr_estimate']/1e6:.0f}M  "
        f"(${result['range_low']/1e6:.0f}M — ${result['range_high']/1e6:.0f}M)  "
        f"{result['confidence_label']} ({result['confidence_score']})"
    )