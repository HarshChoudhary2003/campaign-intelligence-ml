from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT /
    "data" /
    "processed" /
    "bank_marketing_clean.csv"
)

OUTPUT_PATH = (
    ROOT /
    "data" /
    "processed" /
    "bank_marketing_features.csv"
)


def build_features(df):

    df = df.copy()

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 65, 100],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "66+"
        ]
    )

    # Balance groups
    df["balance_group"] = pd.qcut(
        df["balance"],
        q=5,
        labels=False,
        duplicates="drop"
    )

    # Previous campaign contact indicator
    df["has_previous_contact"] = (
        df["previous"] > 0
    ).astype(int)

    # Campaign intensity
    df["high_campaign_contact"] = (
        df["campaign"] >= 3
    ).astype(int)

    return df


def main():

    df = pd.read_csv(INPUT_PATH)

    df = build_features(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
