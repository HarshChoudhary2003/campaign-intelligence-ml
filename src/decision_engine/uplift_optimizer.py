import pandas as pd


def calculate_incremental_value(
    df,
    uplift_column,
    conversion_value,
    contact_cost
):

    result = df.copy()

    result["incremental_value"] = (
        result[uplift_column]
        * conversion_value
        - contact_cost
    )

    return result


def rank_by_incremental_value(
    df,
    uplift_column,
    conversion_value,
    contact_cost
):

    result = calculate_incremental_value(
        df,
        uplift_column,
        conversion_value,
        contact_cost
    )

    result = result[
        result["incremental_value"] > 0
    ]

    return result.sort_values(
        "incremental_value",
        ascending=False
    )
