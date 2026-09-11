from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH, sep=";")


def audit_data(df):
    print("\n" + "=" * 60)
    print("DATASET AUDIT")
    print("=" * 60)

    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing)

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Target Distribution ---")
    print(df["y"].value_counts())
    print(df["y"].value_counts(normalize=True).round(4))

    print("\n--- Numerical Summary ---")
    print(df.describe().T)


if __name__ == "__main__":
    df = load_data()
    audit_data(df)
