import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)

def test_prediction():
    df = pd.read_csv(
        "data/processed/model_data.csv"
    )

    import numpy as np
    customer = (
        df
        .drop(
            columns=["y"],
            errors="ignore"
        )
        .iloc[0]
        .replace({np.nan: None})
        .to_dict()
    )

    response = client.post(
        "/predict",
        json={
            "customer": customer
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert (
        0
        <= result["conversion_probability"]
        <= 1
    )

    assert (
        "model_version"
        in result
    )
