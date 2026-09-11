from src.models.predict import (
    load_model,
    predict_probability
)

from src.config import settings


def score_customer(customer):

    model = load_model()

    probability = predict_probability(
        model,
        customer
    )

    expected_value = (
        probability
        * settings.default_conversion_value
        - settings.default_contact_cost
    )

    return {
        "conversion_probability":
            probability,

        "expected_value":
            expected_value
    }
