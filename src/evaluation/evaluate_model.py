import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    brier_score_loss
)

MODEL_PATH = "artifacts/models/final_xgboost_model.joblib"
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"

def compute_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0)
    }

def evaluate():
    xgb_pipeline = joblib.load(MODEL_PATH)
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    X_train = train_df.drop(columns=["y"], errors="ignore")
    y_train = train_df["y"]
    
    X_test = test_df.drop(columns=["y"], errors="ignore")
    y_test = test_df["y"]
    
    # Train Logistic Regression Baseline
    # Extract preprocessor from the loaded XGBoost pipeline
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]
    
    lr_metrics = compute_metrics(y_test, lr_probs)
    xgb_metrics = compute_metrics(y_test, xgb_probs)
    
    # Save to CSV
    results_df = pd.DataFrame([
        {"model": "Logistic Regression", **lr_metrics},
        {"model": "XGBoost", **xgb_metrics}
    ])
    
    os.makedirs("data/experiments", exist_ok=True)
    results_df.to_csv("data/experiments/model_evaluation.csv", index=False)
    
    print("\n--- Logistic Regression Baseline ---")
    for metric, value in lr_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\n--- XGBoost Final Model ---")
    for metric, value in xgb_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Calibration Curve
    fraction_positive, mean_predicted = calibration_curve(y_test, xgb_probs, n_bins=10)
    plt.figure(figsize=(7, 5))
    plt.plot(mean_predicted, fraction_positive, marker="o", label="XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Model Calibration")
    plt.legend()
    # Save instead of show to avoid blocking
    plt.savefig("data/experiments/calibration_curve.png")

if __name__ == "__main__":
    evaluate()
