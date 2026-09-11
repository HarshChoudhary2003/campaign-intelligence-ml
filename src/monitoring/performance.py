from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "predictions.csv"
)

OUTCOME_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "outcomes.csv"
)


def evaluate_production_model():

    if not PREDICTION_PATH.exists():
        return {
            "status": "no_predictions"
        }

    if not OUTCOME_PATH.exists():
        return {
            "status": "no_outcomes"
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
            "customer_id",
            "model_version"
        ],
        how="inner"
    )

    if len(merged) < 10:

        return {
            "status": "insufficient_data",
            "matched_records":
                int(len(merged))
        }

    y_true = merged[
        "actual_outcome"
    ]

    y_prob = merged[
        "conversion_probability"
    ]

    return {
        "status": "evaluated",

        "matched_records":
            int(len(merged)),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    y_prob
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    y_prob
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    y_prob
                )
            )
    }
