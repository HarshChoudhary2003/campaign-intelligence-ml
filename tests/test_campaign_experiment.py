from src.experiments.campaign_experiment import (
    calculate_capacity,
)


def test_campaign_capacity():

    capacity = calculate_capacity()

    assert capacity == 250
