from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LOG_PATH = Path(
    "data/monitoring/predictions.csv"
)


def log_prediction(
    prediction_id: str,
    campaign_id: str,
    customer_id: str,
    model_version: str,
    conversion_probability: float,
    expected_value: float,
):

    record = pd.DataFrame([
        {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "campaign_id": campaign_id,
            "customer_id": customer_id,
            "model_version": model_version,
            "conversion_probability":
                conversion_probability,
            "expected_value":
                expected_value,
        }
    ])

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if LOG_PATH.exists():

        record.to_csv(
            LOG_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        record.to_csv(
            LOG_PATH,
            index=False
        )
