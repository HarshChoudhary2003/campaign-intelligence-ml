from pathlib import Path
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "final_xgboost_model.joblib"
)

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "model_data.csv"
)


def load_model():

    return joblib.load(MODEL_PATH)


def load_customers():

    return pd.read_csv(DATA_PATH)


def score_customers(
    model,
    customers
):

    X = customers.drop(
        columns=["y"],
        errors="ignore"
    )

    probabilities = (
        model
        .predict_proba(X)[:, 1]
    )

    result = customers.copy()

    result["conversion_probability"] = (
        probabilities
    )

    return result


def apply_campaign_policy(
    scored_customers
):

    result = scored_customers.copy()

    if "campaign" in result.columns:

        result["fatigue_penalty"] = (
            result["campaign"]
            .clip(
                lower=1,
                upper=10
            )
            / 10
        )

    else:

        result["fatigue_penalty"] = 0

    result["adjusted_score"] = (
        result["conversion_probability"]
        * (
            1
            - 0.20
            * result["fatigue_penalty"]
        )
    )

    return result


def optimize_campaign(
    scored_customers,
    budget,
    contact_cost,
    conversion_value,
    strategy,
    max_contacts
):

    result = apply_campaign_policy(scored_customers)

    result["expected_revenue"] = (
        result["conversion_probability"]
        * conversion_value
    )

    result["expected_profit"] = (
        result["expected_revenue"]
        - contact_cost
    )

    if strategy == (
        "Highest Conversion Probability"
    ):

        result = result.sort_values(
            "conversion_probability",
            ascending=False
        )

    elif strategy == (
        "Highest Expected Profit"
    ):

        result = result.sort_values(
            "expected_profit",
            ascending=False
        )

    else:

        result = result.sort_values(
            "adjusted_score",
            ascending=False
        )

    result = result.head(max_contacts)

    result["cumulative_cost"] = (
        pd.Series(
            range(1, len(result) + 1),
            index=result.index
        )
        * contact_cost
    )

    return result[
        result["cumulative_cost"] <= budget
    ].copy()
