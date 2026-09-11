import pandas as pd
from scipy.stats import ks_2samp

def numeric_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.05
):
    results = []
    numeric_columns = reference.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        if column not in current.columns:
            continue

        ref = reference[column].dropna()
        cur = current[column].dropna()

        if len(ref) == 0 or len(cur) == 0:
            continue

        statistic, p_value = ks_2samp(
            ref,
            cur
        )

        results.append({
            "feature": column,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": (
                p_value < threshold
            )
        })

    return pd.DataFrame(results)

def categorical_drift(
    reference,
    current,
    threshold=0.05
):
    results = []

    categorical_columns = (
        reference
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
    )

    for column in categorical_columns:
        if column not in current.columns:
            continue

        ref_distribution = (
            reference[column]
            .value_counts(
                normalize=True
            )
        )

        cur_distribution = (
            current[column]
            .value_counts(
                normalize=True
            )
        )

        categories = (
            set(ref_distribution.index)
            | set(cur_distribution.index)
        )

        difference = sum(
            abs(
                ref_distribution.get(
                    category,
                    0
                )
                -
                cur_distribution.get(
                    category,
                    0
                )
            )
            for category in categories
        )

        results.append({
            "feature": column,
            "distribution_difference": difference,
            "drift_detected": difference > threshold
        })

    return pd.DataFrame(results)

def run_drift_monitoring(
    reference,
    current
):
    numerical = numeric_drift(
        reference,
        current
    )

    categorical = categorical_drift(
        reference,
        current
    )

    return {
        "numerical": numerical,
        "categorical": categorical
    }
