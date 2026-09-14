from pathlib import Path
import joblib


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT /
    "artifacts/models/"
    "final_xgboost_model.joblib"
)


def test_model_exists():

    assert MODEL_PATH.exists()


def test_model_loads():

    model = joblib.load(
        MODEL_PATH
    )

    assert model is not None
