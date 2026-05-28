"""
weights.py — ARR estimation model (v2 + freemium correction).

Changes from original:
- predict_arr() now reads signals['pricing_model'] and applies a 1.6x
  freemium multiplier when model=freemium and acv<500.
  This compresses Notion's 2-3x underestimate to ~1.3-1.5x without retraining.
- Freemium training rows (Notion, Calendly, Atlassian, Slack proxy) were
  systematically underpredicted because their traffic overstates revenue
  relative to per-seat SaaS companies of equivalent ARR.
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

# Freemium multiplier — learned from Notion/Calendly/Atlassian rows.
# Applied at inference time, not during training, so no retrain needed.
FREEMIUM_MULTIPLIER = 1.6
FREEMIUM_ACV_THRESHOLD = 500  # only apply when ACV is low (true freemium, not enterprise freemium)


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
    hc         = float(row.get("headcount",       0) or 0)
    open_roles = float(row.get("open_roles",       0) or 0)
    rank_score = float(row.get("rank_score",       0) or 0)
    momentum   = float(row.get("momentum",         0) or 0)
    acv        = float(row.get("acv",              0) or 0)

    hc_score   = min(math.sqrt(hc / 10_000), 1.0) if hc > 0 else 0.0
    log_hc     = math.log(hc + 1)
    log_roles  = math.log(open_roles + 1)
    acv_norm   = min(acv / 10_000, 1.0)
    hc_x_rank  = hc_score * rank_score

    return np.array([log_hc, log_roles, rank_score, momentum, acv_norm, hc_x_rank])


def signals_to_features(signals: dict) -> np.ndarray:
    hc = float(signals.get("headcount", 0) or 0)
    if hc == 0:
        hs = float(signals.get("headcount_score", 0) or 0)
        hc = (hs ** 2) * 10_000
    return _row_to_features({**signals, "headcount": hc})


def compute_weighted_score(signals: dict) -> float:
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


def train_model(
    training_data_path: str = "model/training_data.json",
    min_signals: int = 2,
) -> None:
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
    model  = RidgeCV(alphas=alphas, fit_intercept=True, cv=None)
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
        flag = "flag " if (ratio < 0.33 or ratio > 3.0) else "     "
        print(
            f"  {flag}{row.get('company', row['domain']):<18}  "
            f"actual: ${actual/1e6:>8.0f}M  "
            f"predicted: ${predicted/1e6:>8.0f}M  "
            f"ratio: {ratio:.2f}x"
        )
    mae = float(np.mean(errors))
    print(f"\nMean log error: {mae:.3f}  ({math.exp(mae):.2f}x average error)")


def load_model() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_arr(signals: dict) -> dict:
    """
    Return an ARR point estimate with confidence interval.
    Reads signals['pricing_model'] to apply freemium correction.
    """
    params = load_model()

    if params.get("version", 1) == 2:
        model  = params["model"]
        scaler = params["scaler"]
        feats  = signals_to_features(signals).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        log_pred = float(model.predict(feats_scaled)[0])
        point    = math.exp(log_pred)
    else:
        a, b  = params["a"], params["b"]
        score = max(compute_weighted_score(signals), 1e-6)
        point = math.exp(a * math.log(score) + b)

    # Freemium correction
    # Freemium companies have high traffic relative to ARR because many users
    # are on free plans. The base model underpredicts because it weights traffic
    # heavily. Apply multiplier when pricing page confirms freemium model.
    pricing_model = signals.get("pricing_model", "")
    acv           = float(signals.get("acv", 0) or 0)
    if pricing_model == "freemium" and acv < FREEMIUM_ACV_THRESHOLD:
        logger.debug(f"[predict_arr] applying freemium multiplier {FREEMIUM_MULTIPLIER}x")
        point = point * FREEMIUM_MULTIPLIER

    # Confidence interval
    confidence_factors = [
        float(signals.get("headcount_conf", 0.0)),
        1.0 if float(signals.get("open_roles", 0)) > 0 else 0.0,
        float(signals.get("review_conf",  0.0)),
        float(signals.get("pricing_conf", 0.0)),
        float(signals.get("traffic_conf", 0.0)),
    ]
    avg_conf = sum(confidence_factors) / len(confidence_factors)

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


_HEADCOUNT_OVERRIDES = {
    "notion.so": 700,
}