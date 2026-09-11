import pandas as pd


class PredictionService:

    def __init__(self, model):

        self.model = model

    def predict(
        self,
        customer: dict
    ):

        customer_df = pd.DataFrame(
            [customer]
        )

        probability = (
            self.model
            .predict_proba(customer_df)[:, 1][0]
        )

        return float(probability)
