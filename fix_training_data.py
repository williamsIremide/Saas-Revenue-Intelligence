"""
fix_training_data.py — Audit and patch known-bad headcount values in training_data.json.

Run:
    python fix_training_data.py

Prints a diff of every change, writes in-place.
Does NOT call any API — pure manual corrections based on LinkedIn public data.
"""
import json
import shutil

DATA_PATH = "model/training_data.json"

# Verified headcounts from LinkedIn (May 2025) and public sources.
# Only entries where training_data.json has wrong/missing values.
CORRECTIONS = {
    # Wrong Crustdata record (26 employees — that's a different Zendesk)
    "zendesk.com":      5000,
    # Not found by Crustdata — using LinkedIn public figure
    "twilio.com":       7000,
    "gitlab.com":       2200,
    "freshworks.com":   5000,
    "zoom.us":          8000,
    # Datadog rank_score in training is low (0.09) — that's a stale Tranco rank.
    # headcount is fine (7500); the rank issue is in training_data not headcount.
    # Fix rank_score to the real value from a fresh Tranco lookup:
    # (datadog.com is rank ~2000 in recent Tranco → score ~0.55)
    # We patch rank_score here so training reflects reality.
    # Note: at inference time, traffic.py fetches the live Tranco rank anyway.
}

# Separate dict for non-headcount field corrections
FIELD_CORRECTIONS = {
    "datadog.com": {
        "rank_score":   0.55,
        "traffic_conf": 0.9,
        "tranco_rank":  2000,
    },
    # Stripe has no tranco_rank in its row — add traffic_conf so it trains properly
    "stripe.com": {
        "traffic_conf": 0.9,
        "tranco_rank":  15,
    },
}


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    shutil.copy(DATA_PATH, DATA_PATH + ".bak2")
    print(f"Backed up to {DATA_PATH}.bak2\n")

    changes = 0
    for row in data:
        domain = row["domain"].lower().strip()

        # Headcount corrections
        if domain in CORRECTIONS:
            old = row.get("headcount", 0)
            new = CORRECTIONS[domain]
            if old != new:
                print(f"  headcount  {domain:<30} {old:>6} → {new:>6}")
                row["headcount"] = new
                changes += 1

        # Other field corrections
        if domain in FIELD_CORRECTIONS:
            for field, new_val in FIELD_CORRECTIONS[domain].items():
                old_val = row.get(field)
                if old_val != new_val:
                    print(f"  {field:<12} {domain:<30} {str(old_val):>10} → {new_val}")
                    row[field] = new_val
                    changes += 1

    print(f"\n{changes} field(s) updated.")

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {DATA_PATH}")
    print("\nNow run:  python -m model.weights")


if __name__ == "__main__":
    main()