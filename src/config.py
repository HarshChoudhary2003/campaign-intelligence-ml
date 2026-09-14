import os


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "1.0.0"
)

CONVERSION_VALUE = float(
    os.getenv(
        "CONVERSION_VALUE",
        "1000"
    )
)

CONTACT_COST = float(
    os.getenv(
        "CONTACT_COST",
        "20"
    )
)
