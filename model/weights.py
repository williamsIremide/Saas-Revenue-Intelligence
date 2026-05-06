"""
weights.py — ARR estimation model.

Architecture change (v2)
------------------------
Previous: collapse all signals to ONE composite score, then fit log(ARR) = a*log(score) + b
Problem:  a 500-employee company with $100M ARR looks identical to one with $500M ARR —
          the composite score can't tell them apart, causing 5-10x errors at the low end.

New:      Multi-feature Ridge regression directly on log-transformed individual signals.
          Each signal gets its own coefficient, learned from data.
          Ridge (L2 regularisation) prevents overfitting on our 60-row dataset.

Features (all log-transformed or normalised to 0-1):
  log_hc         log(headcount + 1)            — scale / revenue capacity
  log_roles      log(open_roles + 1)            — growth velocity
  rank_score     Tranco rank score 0-1          — web reach / brand
  momentum       review momentum 0-1            — customer acquisition
  acv_norm       min(acv / 10000, 1.0)          — pricing tier
  hc_x_rank      headcount_score * rank_score   — scale × reach interaction

Why log-transform headcount and roles?
  ARR scales roughly as headcount^0.6 (diminishing returns to size).
  log(hc) captures this naturally; raw hc would make the coefficient
  meaninglessly small and let large companies dominate.

Training notes
--------------
  - Companies with ARR > ARR_TRAIN_CAP are excluded (Salesforce, ServiceNow etc.)
    They're so far above the query range ($50M–$3B) that even Ridge can't fully
    compensate for the 700x ARR spread across 65 rows.
  - Rows with < 2 real signals are excluded.
  - headcount=0 rows use the fallback composite score path at inference time
    for backwards compatibility with uncached domains.
"""

import json
import logging
import math
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

ARR_TRAIN_CAP   = 5_000_000_000
EXCLUDE_DOMAINS: set[str] = set()
MODEL_PATH      = "model/model.pkl"


# ── Feature extraction ────────────────────────────────────────────────────────

def parse_arr(arr_str: str) -> float:
    return float(str(arr_str).replace("_", ""))


def _signal_quality(row: dict) -> int:
    count = 0
    if float(row.get("headcount",    0)) > 0: count += 1
    if float(row.get("open_roles",   0)) > 0: count += 1
    if float(row.get("review_conf",  0)) > 0: count += 1
    if float(row.get("pricing_conf", 0)) > 0: count += 1
    if float(row.get("traffic_conf", 0)) > 0: count += 1
    return count


def _row_to_features(row: dict) -> np.ndarray:
    """
    Convert a training row or inference signals dict to a feature vector.
    All features are bounded [0, ~14] so Ridge regularisation is meaningful.
    """
    hc         = float(row.get("headcount",       0) or 0)
    open_roles = float(row.get("open_roles",       0) or 0)
    rank_score = float(row.get("rank_score",       0) or 0)
    momentum   = float(row.get("momentum",         0) or 0)
    acv        = float(row.get("acv",              0) or 0)

    # headcount_score for interaction term (sqrt-normalised, 10k=1.0)
    hc_score   = min(math.sqrt(hc / 10_000), 1.0) if hc > 0 else 0.0

    log_hc     = math.log(hc + 1)
    log_roles  = math.log(open_roles + 1)
    acv_norm   = min(acv / 10_000, 1.0)
    hc_x_rank  = hc_score * rank_score  # interaction

    return np.array([
        log_hc,       # 0 — headcount (log scale)
        log_roles,    # 1 — open roles (log scale)
        rank_score,   # 2 — web traffic rank
        momentum,     # 3 — review momentum
        acv_norm,     # 4 — pricing tier
        hc_x_rank,    # 5 — headcount × traffic interaction
    ])


def signals_to_features(signals: dict) -> np.ndarray:
    """
    Inference path: convert live signal dict to feature vector.
    Handles both training-row keys and server.py signal keys.
    """
    # server.py passes headcount_score separately; training rows have raw headcount
    hc = float(signals.get("headcount", 0) or 0)
    if hc == 0:
        # Reconstruct from headcount_score if available
        hs = float(signals.get("headcount_score", 0) or 0)
        hc = (hs ** 2) * 10_000  # invert sqrt normalisation

    return _row_to_features({**signals, "headcount": hc})


# ── Legacy composite score (inference fallback when headcount=0) ──────────────

def compute_weighted_score(signals: dict) -> float:
    """
    Kept for backwards compatibility — used only when headcount=0 AND
    the multi-feature model is unavailable.
    """
    hc_score   = float(signals.get("headcount_score", 0.0))
    open_roles = float(signals.get("open_roles",      0))
    momentum   = float(signals.get("momentum",        0.0))
    rank_score = float(signals.get("rank_score",      0.0))
    acv        = float(signals.get("acv",             0.0))

    acv_norm   = min(acv / 10_000, 1.0)
    roles_norm = min((open_roles / 500) ** 0.6, 1.0)
    hc_x_rank  = min((hc_score * rank_score) / 0.2, 1.0)

    if hc_score > 0:
        return (
            hc_score   * 0.40 +
            rank_score * 0.20 +
            momentum   * 0.15 +
            roles_norm * 0.10 +
            acv_norm   * 0.10 +
            hc_x_rank  * 0.05
        )
    else:
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

def train_model(
    training_data_path: str = "model/training_data.json",
    min_signals: int = 2,
) -> None:
    """
    Fit multi-feature Ridge regression: log(ARR) ~ features.

    Uses RidgeCV to auto-select the regularisation strength alpha from
    [0.01, 0.1, 1, 10, 100] via leave-one-out cross-validation.
    """
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn not found. Run: pip install scikit-learn")
        raise

    with open(training_data_path) as f:
        data = json.load(f)

    filtered, excluded_cap, excluded_sig, excluded_zero = [], 0, 0, 0

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
        if (row.get("open_roles", 0) == 0
                and row.get("momentum", 0) == 0
                and row.get("rank_score", 0) == 0):
            excluded_zero += 1
            continue
        filtered.append(row)

    print(
        f"Training on {len(filtered)}/{len(data)} companies  "
        f"(excluded: {excluded_cap} ARR>{ARR_TRAIN_CAP/1e9:.0f}B, "
        f"{excluded_sig} <{min_signals} signals, {excluded_zero} all-zero)"
    )

    X = np.array([_row_to_features(row) for row in filtered])
    y = np.array([math.log(parse_arr(row["arr"])) for row in filtered])

    # Standardise features so Ridge penalty is applied fairly across all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # RidgeCV: tries multiple alpha values, picks best via leave-one-out CV
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
    model  = RidgeCV(alphas=alphas, fit_intercept=True, cv=None)  # cv=None → LOO
    model.fit(X_scaled, y)

    print(f"  Method:     RidgeCV (alpha={model.alpha_:.3f})")
    print(f"  Intercept:  {model.intercept_:.4f}")
    feature_names = ["log_hc", "log_roles", "rank_score", "momentum", "acv_norm", "hc_x_rank"]
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name:<14} coef={coef:+.4f}")

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "version": 2}, f)

    _print_sanity_check(filtered, X_scaled, model)
    print(f"Model saved -> {MODEL_PATH}")


def _print_sanity_check(rows, X_scaled, model) -> None:
    print("\nSanity check (flag = >3x error):")
    errors = []
    preds  = model.predict(X_scaled)
    for row, log_pred in zip(rows, preds):
        predicted = math.exp(log_pred)
        actual    = parse_arr(row["arr"])
        ratio     = predicted / actual if actual > 0 else 0.0
        errors.append(abs(math.log(max(ratio, 1e-6))))
        flag = "⚠ " if (ratio < 0.33 or ratio > 3.0) else "  "
        print(
            f"  {flag}{row.get('company', row['domain']):<18}  "
            f"actual: ${actual/1e6:>8.0f}M  "
            f"predicted: ${predicted/1e6:>8.0f}M  "
            f"ratio: {ratio:.2f}x"
        )
    mae = float(np.mean(errors))
    print(f"\nMean log error: {mae:.3f}  ({math.exp(mae):.2f}x average error)")


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_arr(signals: dict) -> dict:
    """
    Return an ARR point estimate with confidence interval.

    For model v2 (multi-feature Ridge): uses per-feature regression.
    For model v1 (legacy single-score): falls back to old composite path.
    """
    params = load_model()

    if params.get("version", 1) == 2:
        # ── v2: multi-feature Ridge ───────────────────────────────────────────
        model  = params["model"]
        scaler = params["scaler"]

        feats  = signals_to_features(signals).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        log_pred = float(model.predict(feats_scaled)[0])
        point    = math.exp(log_pred)

    else:
        # ── v1 fallback (old model.pkl format) ────────────────────────────────
        a, b  = params["a"], params["b"]
        score = max(compute_weighted_score(signals), 1e-6)
        point = math.exp(a * math.log(score) + b)

    # ── Confidence interval ───────────────────────────────────────────────────
    confidence_factors = [
        float(signals.get("headcount_conf", 0.0)),
        1.0 if float(signals.get("open_roles", 0)) > 0 else 0.0,
        float(signals.get("review_conf",  0.0)),
        float(signals.get("pricing_conf", 0.0)),
        float(signals.get("traffic_conf", 0.0)),
    ]
    avg_conf = sum(confidence_factors) / len(confidence_factors)

    # Wider interval when headcount is missing (primary signal absent)
    hc = float(signals.get("headcount", 0) or 0)
    if hc == 0:
        hs = float(signals.get("headcount_score", 0) or 0)
        hc = (hs ** 2) * 10_000
    base_margin = 0.50 if hc < 50 else 0.40
    margin = base_margin - (avg_conf * 0.20)

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

    print("\n--- predict_arr smoke tests ---")
    tests = [
        ("Notion (~$330M)",  {"headcount": 3000, "open_roles": 144, "momentum": 0.41, "rank_score": 0.495, "acv": 480,  "headcount_score": 0.548, "review_conf": 0.7, "pricing_conf": 1.0, "traffic_conf": 0.9, "headcount_conf": 0.92}),
        ("HubSpot (~$2.2B)", {"headcount": 11965,"open_roles": 0,   "momentum": 0.97, "rank_score": 0.599, "acv": 0,    "headcount_score": 1.0,   "review_conf": 1.0, "pricing_conf": 0.2, "traffic_conf": 0.9, "headcount_conf": 0.92}),
        ("Linear (~$50M)",   {"headcount": 200,  "open_roles": 23,  "momentum": 0.41, "rank_score": 0.326, "acv": 1560, "headcount_score": 0.141, "review_conf": 0.7, "pricing_conf": 1.0, "traffic_conf": 0.75,"headcount_conf": 0.92}),
        ("Calendly (~$100M)",{"headcount": 500,  "open_roles": 21,  "momentum": 0.59, "rank_score": 0.566, "acv": 300,  "headcount_score": 0.224, "review_conf": 0.7, "pricing_conf": 1.0, "traffic_conf": 0.9, "headcount_conf": 0.92}),
        ("Vanta (~$100M)",   {"headcount": 500,  "open_roles": 150, "momentum": 0.40, "rank_score": 0.285, "acv": 0,    "headcount_score": 0.224, "review_conf": 0.7, "pricing_conf": 0.2, "traffic_conf": 0.75,"headcount_conf": 0.92}),
    ]
    for label, sigs in tests:
        r = predict_arr(sigs)
        print(
            f"  {label:<22}  ${r['arr_estimate']/1e6:>6.0f}M  "
            f"({r['confidence_label']}, {r['confidence_score']:.2f})"
        )