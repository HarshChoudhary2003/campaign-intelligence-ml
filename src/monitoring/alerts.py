def check_prediction_health(
    summary
):

    alerts = []

    mean_probability = (
        summary["mean_probability"]
    )

    high_probability_rate = (
        summary["high_probability_rate"]
    )

    if mean_probability is not None:

        if mean_probability < 0.05:

            alerts.append(
                "Average prediction probability "
                "is unusually low."
            )

        elif mean_probability > 0.80:

            alerts.append(
                "Average prediction probability "
                "is unusually high."
            )

    if high_probability_rate is not None:

        if high_probability_rate > 0.90:

            alerts.append(
                "Unusually large percentage of "
                "customers have high predicted probability."
            )

    return alerts
