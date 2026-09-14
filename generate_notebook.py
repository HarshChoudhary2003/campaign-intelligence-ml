import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

code1 = """import joblib
import pandas as pd
import shap

model = joblib.load(
    "../artifacts/models/final_xgboost_model.joblib"
)"""

code2 = """df = pd.read_csv(
    "../data/processed/model_data.csv"
)

X = df.drop(
    columns=["y", "duration"],
    errors="ignore"
)"""

code3 = """# Extract model and transform data because it's a Pipeline
xgb_model = model.named_steps['model']
preprocessor = model.named_steps['preprocessor']
X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_transformed)"""

code4 = """shap.summary_plot(
    shap_values,
    features=X_transformed,
    feature_names=feature_names
)"""

code5 = """importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
})

importance = (
    importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)

importance.head(15)"""

code6 = """import os
os.makedirs("../data/experiments", exist_ok=True)
importance.to_csv(
    "../data/experiments/shap_feature_importance.csv",
    index=False
)"""

code7 = """customer = X.iloc[[0]]
customer_transformed = preprocessor.transform(customer)
customer_shap = explainer.shap_values(customer_transformed)

shap.force_plot(
    explainer.expected_value,
    customer_shap[0],
    customer_transformed[0],
    feature_names=feature_names,
    matplotlib=True
)"""

nb['cells'] = [
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_code_cell(code6),
    nbf.v4.new_code_cell(code7),
]

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/12_shap_explainability.ipynb", "w") as f:
    nbf.write(nb, f)
