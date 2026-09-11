def retraining_required(
    performance,
    baseline_pr_auc,
    minimum_records=100
):

    if performance.get(
        "status"
    ) != "evaluated":

        return False

    if performance[
        "matched_records"
    ] < minimum_records:

        return False

    current_pr_auc = performance[
        "pr_auc"
    ]

    degradation = (
        baseline_pr_auc
        - current_pr_auc
    )

    return degradation >= 0.05
