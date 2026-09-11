import pandas as pd


def prediction_summary(
    predictions
):

    if predictions.empty:

        return {
            "count": 0,
            "mean_probability": None,
            "median_probability": None,
            "high_probability_rate": None
        }

    probabilities = (
        predictions[
            "conversion_probability"
        ]
    )

    return {

        "count":
            int(len(probabilities)),

        "mean_probability":
            float(probabilities.mean()),

        "median_probability":
            float(probabilities.median()),

        "high_probability_rate":
            float(
                (
                    probabilities >= 0.70
                ).mean()
            )
    }
