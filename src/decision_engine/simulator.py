from dataclasses import dataclass
import pandas as pd


@dataclass
class CampaignScenario:
    budget: float
    contact_cost: float
    conversion_value: float


def simulate_campaign(
    ranked_customers: pd.DataFrame,
    scenario: CampaignScenario
) -> dict:

    df = ranked_customers.copy()

    # Expected value per customer
    df["expected_revenue"] = (
        df["xgb_probability"]
        * scenario.conversion_value
    )

    df["contact_cost"] = scenario.contact_cost

    df["expected_profit"] = (
        df["expected_revenue"]
        - df["contact_cost"]
    )

    # Only economically worthwhile customers
    df = df[
        df["expected_profit"] > 0
    ].copy()

    # Highest expected profit first
    df = df.sort_values(
        "expected_profit",
        ascending=False
    )

    # Respect campaign budget
    df["cumulative_cost"] = (
        df["contact_cost"].cumsum()
    )

    selected = df[
        df["cumulative_cost"] <= scenario.budget
    ].copy()

    if len(selected) == 0:
        return {
            "customers_targeted": 0,
            "expected_conversions": 0,
            "campaign_cost": 0,
            "expected_revenue": 0,
            "expected_profit": 0,
            "expected_roi": 0,
            "target_list": selected
        }

    expected_conversions = (
        selected["xgb_probability"].sum()
    )

    campaign_cost = (
        selected["contact_cost"].sum()
    )

    expected_revenue = (
        selected["expected_revenue"].sum()
    )

    expected_profit = (
        selected["expected_profit"].sum()
    )

    expected_roi = (
        expected_profit / campaign_cost
    )

    return {
        "customers_targeted": len(selected),
        "expected_conversions": expected_conversions,
        "campaign_cost": campaign_cost,
        "expected_revenue": expected_revenue,
        "expected_profit": expected_profit,
        "expected_roi": expected_roi,
        "target_list": selected
    }
