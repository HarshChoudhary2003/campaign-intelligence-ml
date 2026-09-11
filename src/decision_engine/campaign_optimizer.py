from dataclasses import dataclass
import pandas as pd


@dataclass
class CampaignConfig:
    conversion_value: float = 1000.0
    contact_cost: float = 20.0
    budget: float = 100000.0


def calculate_expected_value(
    probability: pd.Series,
    config: CampaignConfig
) -> pd.Series:

    return (
        probability * config.conversion_value
        - config.contact_cost
    )


def rank_customers(
    df: pd.DataFrame,
    probability_column: str,
    config: CampaignConfig
) -> pd.DataFrame:

    result = df.copy()

    result["expected_value"] = (
        calculate_expected_value(
            result[probability_column],
            config
        )
    )

    result = result.sort_values(
        "expected_value",
        ascending=False
    )

    result["priority_rank"] = range(
        1,
        len(result) + 1
    )

    return result


def select_customers(
    ranked_df: pd.DataFrame,
    config: CampaignConfig
) -> pd.DataFrame:

    result = ranked_df.copy()

    result["contact_cost"] = (
        config.contact_cost
    )

    result["cumulative_cost"] = (
        result["contact_cost"].cumsum()
    )

    result = result[
        (result["cumulative_cost"] <= config.budget)
        &
        (result["expected_value"] > 0)
    ]

    return result
