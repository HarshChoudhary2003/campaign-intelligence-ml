from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "bank_marketing_features.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "model_data.csv"
)


def prepare_data():

    df = pd.read_csv(INPUT_PATH)

    # Information available before making the call
    features = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "age_group",
        "balance_group",
        "has_previous_contact",
        "high_campaign_contact"
    ]

    target = "y"

    model_df = df[features + [target]].copy()

    model_df = model_df.dropna(
        subset=[target]
    )

    return model_df


if __name__ == "__main__":

    df = prepare_data()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Model dataset created")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")
