import shap
import pandas as pd


def create_explainer(model):
    """
    Create a SHAP TreeExplainer for the trained
    tree-based model.
    """

    return shap.TreeExplainer(model)


def explain_prediction(
    explainer,
    features: pd.DataFrame
):

    shap_values = explainer.shap_values(
        features
    )

    return shap_values

def top_prediction_reasons(
    shap_values,
    feature_names,
    top_n=5
):

    values = pd.Series(
        shap_values,
        index=feature_names
    )

    positive = (
        values[values > 0]
        .sort_values(
            ascending=False
        )
        .head(top_n)
    )

    negative = (
        values[values < 0]
        .sort_values()
        .head(top_n)
    )

    return positive, negative
