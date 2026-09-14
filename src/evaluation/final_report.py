from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "artifacts/models/final_xgboost_model.joblib"
TEST_PATH = ROOT / "data/processed/test.csv"
OUTPUT_DIR = ROOT / "data/experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_PATH)

    X_test = test.drop(columns=["y"], errors="ignore")
    y_test = test["y"]

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    results = {
        "model": "XGBoost",
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities
        ),
        "brier_score": brier_score_loss(
            y_test,
            probabilities
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
    }

    return results


def save_results(results):

    output_file = (
        OUTPUT_DIR /
        "final_model_evaluation.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    return output_file


def main():

    print("=" * 50)
    print("CAMPAIGN INTELLIGENCE")
    print("FINAL MODEL EVALUATION")
    print("=" * 50)

    model, X_test, y_test = load_data()

    results = evaluate_model(
        model,
        X_test,
        y_test
    )

    for metric, value in results.items():

        if isinstance(value, float):

            print(
                f"{metric:15}: {value:.4f}"
            )

        else:

            print(
                f"{metric:15}: {value}"
            )

    output_file = save_results(results)

    print()
    print(
        f"Results saved to: {output_file}"
    )


if __name__ == "__main__":
    main()
