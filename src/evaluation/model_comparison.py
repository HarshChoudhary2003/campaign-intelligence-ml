from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score
)

from src.models.train_model import build_pipeline

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT /
    "artifacts/models/"
    "final_xgboost_model.joblib"
)

TRAIN_PATH = (
    ROOT /
    "data/processed/train.csv"
)

TEST_PATH = (
    ROOT /
    "data/processed/test.csv"
)

OUTPUT_PATH = (
    ROOT /
    "data/experiments/"
    "model_comparison.csv"
)


def evaluate(name, model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "model": name,
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "brier_score": brier_score_loss(y, probabilities),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0)
    }


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["y"], errors="ignore")
    y_train = train_df["y"]

    X_test = test_df.drop(columns=["y"], errors="ignore")
    y_test = test_df["y"]

    xgb_model = joblib.load(MODEL_PATH)
    results = []

    results.append(evaluate("XGBoost", xgb_model, X_test, y_test))

    # Baseline with proper preprocessing
    pipeline = build_pipeline(X_train)
    pipeline.steps.pop() # Remove XGBoost
    pipeline.steps.append(("model", LogisticRegression(max_iter=1000)))

    try:
        pipeline.fit(X_train, y_train)
        results.append(evaluate("Logistic Regression", pipeline, X_test, y_test))
    except Exception as error:
        print("Baseline could not be trained.")
        print(f"Reason: {error}")

    result_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print(result_df.to_string(index=False))
    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
