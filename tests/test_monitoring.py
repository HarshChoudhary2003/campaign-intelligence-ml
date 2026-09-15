from src.monitoring.retraining import (
    should_retrain
)


def test_retraining_not_needed():

    performance = {
        "status": "available",
        "matched_records": 150,
        "pr_auc": 0.31
    }

    assert should_retrain(
        performance,
        baseline_pr_auc=0.30
    ) is False


def test_retraining_needed():

    performance = {
        "status": "available",
        "matched_records": 150,
        "pr_auc": 0.20
    }

    assert should_retrain(
        performance,
        baseline_pr_auc=0.30
    ) is True
