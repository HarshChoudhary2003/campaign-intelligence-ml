def should_retrain(
    performance,
    baseline_pr_auc,
    minimum_records=100,
):

    if performance.get(
        "status"
    ) != "available":

        return False

    if performance.get(
        "matched_records",
        0
    ) < minimum_records:

        return False

    current_pr_auc = performance[
        "pr_auc"
    ]

    degradation = (
        baseline_pr_auc -
        current_pr_auc
    )

    return degradation >= 0.05
