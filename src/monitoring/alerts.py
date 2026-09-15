def generate_alerts(
    performance,
    baseline_pr_auc=None
):

    alerts = []

    if performance.get(
        "status"
    ) != "available":

        return alerts

    current_pr_auc = performance.get(
        "pr_auc"
    )

    if (
        baseline_pr_auc is not None
        and
        current_pr_auc <
        baseline_pr_auc - 0.05
    ):

        alerts.append({
            "severity": "HIGH",
            "type": "MODEL_DEGRADATION",
            "message":
                "Production PR-AUC has degraded "
                "by at least 0.05 versus baseline."
        })

    return alerts
