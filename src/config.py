import os

from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    api_url: str = os.getenv(
        "API_URL",
        "http://127.0.0.1:8000"
    )

    model_version: str = os.getenv(
        "MODEL_VERSION",
        "1.0.0"
    )

    conversion_value: float = float(
        os.getenv(
            "CONVERSION_VALUE",
            "1000"
        )
    )

    contact_cost: float = float(
        os.getenv(
            "CONTACT_COST",
            "20"
        )
    )

settings = Settings()
