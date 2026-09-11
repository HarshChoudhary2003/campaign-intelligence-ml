from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTCOME_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "outcomes.csv"
)


def log_outcome(
    customer_id,
    actual_outcome,
    model_version
):

    OUTCOME_PATH.parent.mkdir(
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

            "actual_outcome": int(
                actual_outcome
            ),

            "model_version":
                model_version
        }
    ])

    if OUTCOME_PATH.exists():

        record.to_csv(
            OUTCOME_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        record.to_csv(
            OUTCOME_PATH,
            index=False
        )
