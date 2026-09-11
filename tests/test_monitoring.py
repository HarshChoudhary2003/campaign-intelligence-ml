import pandas as pd

from src.monitoring.drift import (
    numeric_drift
)


def test_no_drift():

    reference = pd.DataFrame({
        "age": [20, 25, 30, 35, 40]
    })

    current = reference.copy()

    result = numeric_drift(
        reference,
        current
    )

    assert not result[
        "drift_detected"
    ].any()


def test_drift_detection():

    reference = pd.DataFrame({
        "age": [
            20, 21, 22, 23, 24,
            25, 26, 27, 28, 29
        ]
    })

    current = pd.DataFrame({
        "age": [
            50, 51, 52, 53, 54,
            55, 56, 57, 58, 59
        ]
    })

    result = numeric_drift(
        reference,
        current
    )

    assert result[
        "drift_detected"
    ].any()
