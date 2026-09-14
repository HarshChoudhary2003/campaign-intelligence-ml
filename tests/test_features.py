from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    ROOT /
    "data/processed/test.csv"
)


def test_duration_not_used():

    df = pd.read_csv(TEST_PATH)

    assert "duration" not in df.columns, (
        "duration must not be used for "
        "pre-contact targeting"
    )
