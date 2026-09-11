import pandas as pd
import numpy as np


def explain_customer(
    shap_values,
    feature_names,
    top_n=5
):

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values
    })

    explanation["impact"] = np.where(
        explanation["shap_value"] > 0,
        "positive",
        "negative"
    )

    explanation["absolute_impact"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        "absolute_impact",
        ascending=False
    )

    return explanation.head(top_n)
