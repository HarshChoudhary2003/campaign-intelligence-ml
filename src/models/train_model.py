from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "model_data.csv"
)

MODEL_DIR = (
    ROOT
    / "artifacts"
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "final_xgboost_model.joblib"
)


def load_data():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["y"])
    y = df["y"]

    return X, y


def build_pipeline(X):

    categorical_features = (
        X.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    numerical_features = (
        X.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    return pipeline


def train():

    X, y = load_data()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    train_data = X_train.copy()
    train_data["y"] = y_train
    test_data = X_test.copy()
    test_data["y"] = y_test

    train_data.to_csv("data/processed/train.csv", index=False)
    test_data.to_csv("data/processed/test.csv", index=False)

    pipeline = build_pipeline(X_train)

    pipeline.fit(
        X_train,
        y_train
    )

    probabilities = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print(
        f"Brier Score: {brier:.4f}"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    train()
