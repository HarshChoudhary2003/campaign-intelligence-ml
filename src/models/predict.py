from pathlib import Path
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "xgboost_campaign_model.joblib"
)


def load_model():

    return joblib.load(
        MODEL_PATH
    )


def predict_probability(
    model,
    customer_data
):

    if isinstance(
        customer_data,
        dict
    ):
        customer_data = pd.DataFrame(
            [customer_data]
        )

    probability = (
        model
        .predict_proba(customer_data)[:, 1]
    )

    return float(
        probability[0]
    )
