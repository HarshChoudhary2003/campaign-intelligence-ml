from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LOG_PATH = Path(
    "data/monitoring/outcomes.csv"
)


def log_outcome(
    outcome_id: str,
    campaign_id: str,
    customer_id: str,
    actual_outcome: int,
):

    record = pd.DataFrame([
        {
            "outcome_id": outcome_id,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "campaign_id": campaign_id,
            "customer_id": customer_id,
            "actual_outcome":
                actual_outcome,
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
