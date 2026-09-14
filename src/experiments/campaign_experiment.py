from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT /
    "artifacts/models/final_xgboost_model.joblib"
)

TEST_PATH = (
    ROOT /
    "data/processed/test.csv"
)

OUTPUT_PATH = (
    ROOT /
    "data/experiments/campaign_strategy_results.csv"
)


# --------------------------------------------------
# Campaign assumptions
# --------------------------------------------------

BUDGET = 5000
CONTACT_COST = 20
CONVERSION_VALUE = 1000

RANDOM_STATE = 42


# --------------------------------------------------
# Load model and test data
# --------------------------------------------------

def load_data():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(TEST_PATH)

    X = df.drop(
        columns=["y"],
        errors="ignore"
    )

    y = df["y"].astype(int)

    probabilities = model.predict_proba(X)[:, 1]

    df = df.copy()

    df["conversion_probability"] = probabilities

    return df


# --------------------------------------------------
# Number of customers we can contact
# --------------------------------------------------

def calculate_capacity():

    return int(
        BUDGET // CONTACT_COST
    )


# --------------------------------------------------
# Strategy 1: Random
# --------------------------------------------------

def random_strategy(df, capacity):

    return df.sample(
        n=min(capacity, len(df)),
        random_state=RANDOM_STATE
    )


# --------------------------------------------------
# Strategy 2: Highest probability
# --------------------------------------------------

def probability_strategy(df, capacity):

    return (
        df.sort_values(
            "conversion_probability",
            ascending=False
        )
        .head(capacity)
    )


# --------------------------------------------------
# Strategy 3: Expected profit
# --------------------------------------------------

def profit_strategy(df, capacity):

    result = df.copy()

    result["expected_revenue"] = (
        result["conversion_probability"]
        * CONVERSION_VALUE
    )

    result["expected_profit"] = (
        result["expected_revenue"]
        - CONTACT_COST
    )

    return (
        result.sort_values(
            "expected_profit",
            ascending=False
        )
        .head(capacity)
    )


# --------------------------------------------------
# Strategy 4: Fatigue-aware targeting
# --------------------------------------------------

def fatigue_strategy(df, capacity):

    result = df.copy()

    # Campaign-contact history proxy.
    #
    # In the Bank Marketing dataset,
    # previous campaign contacts are represented
    # by campaign.
    #
    # This is a business policy assumption,
    # NOT a causal effect.

    if "campaign" in result.columns:

        result["fatigue_penalty"] = (
            np.log1p(
                result["campaign"]
            )
        )

    else:

        result["fatigue_penalty"] = 0

    result["fatigue_adjusted_score"] = (
        result["conversion_probability"]
        / (
            1 +
            0.10 *
            result["fatigue_penalty"]
        )
    )

    return (
        result.sort_values(
            "fatigue_adjusted_score",
            ascending=False
        )
        .head(capacity)
    )


# --------------------------------------------------
# Evaluate campaign
# --------------------------------------------------

def evaluate_strategy(
    selected,
    strategy_name
):

    contacts = len(selected)

    conversions = int(
        selected["y"].sum()
    )

    conversion_rate = (
        conversions / contacts
        if contacts
        else 0
    )

    revenue = (
        conversions *
        CONVERSION_VALUE
    )

    cost = (
        contacts *
        CONTACT_COST
    )

    profit = revenue - cost

    roi = (
        profit / cost
        if cost
        else 0
    )

    expected_conversions = (
        selected[
            "conversion_probability"
        ].sum()
    )

    return {
        "strategy": strategy_name,
        "contacts": contacts,
        "actual_conversions": conversions,
        "actual_conversion_rate":
            conversion_rate,
        "expected_conversions":
            expected_conversions,
        "revenue": revenue,
        "campaign_cost": cost,
        "profit": profit,
        "roi": roi
    }


# --------------------------------------------------
# Run experiment
# --------------------------------------------------

def main():

    print("=" * 60)
    print("CAMPAIGN STRATEGY EXPERIMENT")
    print("=" * 60)

    df = load_data()

    capacity = calculate_capacity()

    print()
    print(f"Budget       : {BUDGET}")
    print(f"Contact cost : {CONTACT_COST}")
    print(f"Capacity     : {capacity}")
    print(f"Test records : {len(df)}")

    strategies = {

        "Random":
            random_strategy(
                df,
                capacity
            ),

        "Highest Probability":
            probability_strategy(
                df,
                capacity
            ),

        "Expected Profit":
            profit_strategy(
                df,
                capacity
            ),

        "Fatigue-Aware":
            fatigue_strategy(
                df,
                capacity
            )
    }

    results = []

    for name, selected in strategies.items():

        result = evaluate_strategy(
            selected,
            name
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "profit",
        ascending=False
    )

    print()
    print(
        results_df.to_string(
            index=False
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(
        f"Saved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
