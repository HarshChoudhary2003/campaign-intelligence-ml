from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "bank-full.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "bank_marketing_clean.csv"


def load_raw_data():
    return pd.read_csv(RAW_PATH, sep=";")


def clean_data(df):

    df = df.copy()

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Convert target to binary
    df["y"] = df["y"].map({
        "yes": 1,
        "no": 0
    })

    # Treat "unknown" as missing for categorical fields
    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    for column in categorical_columns:
        df[column] = df[column].replace(
            "unknown",
            pd.NA
        )

    return df


def save_data(df):
    PROCESSED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_PATH,
        index=False
    )


if __name__ == "__main__":

    df = load_raw_data()

    print(f"Raw rows: {len(df):,}")

    df = clean_data(df)

    print(f"Clean rows: {len(df):,}")

    print("\nMissing values:")
    print(df.isna().sum())

    save_data(df)

    print(
        f"\nClean dataset saved to:\n{PROCESSED_PATH}"
    )
