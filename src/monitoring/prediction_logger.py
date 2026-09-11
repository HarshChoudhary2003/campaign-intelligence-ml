from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "predictions.csv"
)


def log_prediction(
    customer_id,
    probability,
    expected_value,
    model_version
):

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    record = pd.DataFrame([
        {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "customer_id": str(
                customer_id
            ),

            "model_version":
                model_version,

            "conversion_probability":
                float(probability),

            "expected_value":
                float(expected_value)
        }
    ])

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
