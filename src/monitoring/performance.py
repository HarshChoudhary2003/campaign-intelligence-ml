from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


PREDICTION_PATH = Path(
    "data/monitoring/predictions.csv"
)

OUTCOME_PATH = Path(
    "data/monitoring/outcomes.csv"
)


MIN_RECORDS = 10


def calculate_performance():

    if not (
        PREDICTION_PATH.exists()
        and OUTCOME_PATH.exists()
    ):

        return {
            "status": "insufficient_data",
            "message":
                "Prediction or outcome data unavailable."
        }

    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    outcomes = pd.read_csv(
        OUTCOME_PATH
    )

    merged = predictions.merge(
        outcomes,
        on=[
            "campaign_id",
            "customer_id"
        ],
        how="inner"
    )

    if len(merged) < MIN_RECORDS:

        return {
            "status": "insufficient_data",
            "matched_records":
                len(merged),
            "required_records":
                MIN_RECORDS
        }

    y_true = merged[
        "actual_outcome"
    ]

    y_prob = merged[
        "conversion_probability"
    ]

    result = {
        "status": "available",
        "matched_records":
            len(merged),
        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob
            ),
        "pr_auc":
            average_precision_score(
                y_true,
                y_prob
            ),
        "brier_score":
            brier_score_loss(
                y_true,
                y_prob
            ),
        "actual_conversion_rate":
            y_true.mean()
    }

    return result
