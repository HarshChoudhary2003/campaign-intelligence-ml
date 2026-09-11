from pathlib import Path
import sys
import json
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from app.services.campaign_service import (
    load_model,
    load_customers,
    score_customers,
    optimize_campaign
)
from src.models.prediction_service import PredictionService
from src.utils.logger import get_logger

from src.monitoring.drift import run_drift_monitoring
from src.monitoring.prediction_logger import log_prediction
from src.monitoring.prediction_monitor import prediction_summary
from src.monitoring.alerts import check_prediction_health
from src.monitoring.outcome_logger import log_outcome
from src.monitoring.performance import evaluate_production_model

logger = get_logger(__name__)

app = FastAPI(
    title="Campaign Intelligence API",
    description=(
        "ML API for customer conversion "
        "prediction and campaign targeting."
    ),
    version="1.0.0"
)


model = load_model()
prediction_service = PredictionService(model)
customers = load_customers()


MODEL_METADATA_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "model_metadata.json"
)

def load_metadata():
    with open(MODEL_METADATA_PATH, "r") as file:
        return json.load(file)

metadata = load_metadata()


class PredictionRequest(BaseModel):
    customer: dict

    @field_validator("customer")
    @classmethod
    def validate_customer(cls, value):
        if not value:
            raise ValueError("Customer data cannot be empty.")
        return value


class PredictionResponse(BaseModel):
    conversion_probability: float
    expected_value: float
    model_version: str


class CampaignRequest(BaseModel):
    budget: float = Field(gt=0)
    contact_cost: float = Field(gt=0)
    conversion_value: float = Field(gt=0)
    max_contacts: int = Field(gt=0)
    strategy: str = "Highest Expected Profit"


class OutcomeRequest(BaseModel):
    customer_id: str
    actual_outcome: int = Field(ge=0, le=1)


@app.get("/")
def root():
    return {
        "service": "Campaign Intelligence API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": metadata.get("model_version", "1.0.0")
    }


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )
    return {
        "status": "ready"
    }


@app.get("/model/info")
def model_info():
    return metadata


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    logger.info("Prediction request received")
    start_time = time.perf_counter()

    try:
        probability = prediction_service.predict(request.customer)

        conversion_value = 1000.0
        contact_cost = 20.0

        expected_value = (
            probability
            * conversion_value
            - contact_cost
        )

        customer_id = request.customer.get(
            "customer_id",
            "unknown"
        )
        
        log_prediction(
            customer_id=customer_id,
            probability=probability,
            expected_value=expected_value,
            model_version=metadata[
                "model_version"
            ]
        )

        latency = time.perf_counter() - start_time
        logger.info(f"Prediction latency: {latency:.4f}s")
        logger.info("Prediction generated successfully")

        return {
            "conversion_probability": float(probability),
            "expected_value": float(expected_value),
            "model_version": metadata["model_version"]
        }

    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@app.post("/campaign/optimize")
def campaign_optimize(
    request: CampaignRequest
):
    scored = score_customers(
        model,
        customers
    )

    selected = optimize_campaign(
        scored,
        request.budget,
        request.contact_cost,
        request.conversion_value,
        request.strategy,
        request.max_contacts
    )

    expected_conversions = (
        selected[
            "conversion_probability"
        ].sum()
    )

    expected_revenue = (
        selected[
            "expected_revenue"
        ].sum()
    )

    campaign_cost = (
        len(selected)
        * request.contact_cost
    )

    expected_profit = (
        expected_revenue
        - campaign_cost
    )

    return {
        "customers_targeted":
            len(selected),

        "expected_conversions":
            float(expected_conversions),

        "expected_revenue":
            float(expected_revenue),

        "campaign_cost":
            float(campaign_cost),

        "expected_profit":
            float(expected_profit)
    }

@app.get("/monitoring/drift")
def monitoring_drift():
    try:
        reference = pd.read_csv(
            ROOT
            / "data"
            / "monitoring"
            / "reference.csv"
        )

        current = pd.read_csv(
            ROOT
            / "data"
            / "processed"
            / "model_data.csv"
        )

        results = run_drift_monitoring(
            reference,
            current
        )

        numerical = results[
            "numerical"
        ]

        categorical = results[
            "categorical"
        ]

        numerical_drift = int(
            numerical[
                "drift_detected"
            ].sum()
        )

        categorical_drift = int(
            categorical[
                "drift_detected"
            ].sum()
        )

        return {
            "status": "completed",
            "numerical_features_with_drift":
                numerical_drift,
            "categorical_features_with_drift":
                categorical_drift
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.get(
    "/monitoring/predictions"
)
def prediction_monitoring():
    try:
        log_path = (
            ROOT
            / "data"
            / "monitoring"
            / "predictions.csv"
        )

        if not log_path.exists():
            return {
                "count": 0,
                "message":
                    "No predictions logged yet."
            }

        predictions = pd.read_csv(
            log_path
        )

        return prediction_summary(
            predictions
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.get(
    "/monitoring/alerts"
)
def monitoring_alerts():
    log_path = (
        ROOT
        / "data"
        / "monitoring"
        / "predictions.csv"
    )

    if not log_path.exists():
        return {
            "status": "no_data",
            "alerts": []
        }

    predictions = pd.read_csv(
        log_path
    )

    summary = prediction_summary(
        predictions
    )

    alerts = check_prediction_health(
        summary
    )

    return {
        "status": (
            "warning"
            if alerts
            else "healthy"
        ),
        "alerts": alerts
    }

@app.post("/outcomes")
def record_outcome(
    request: OutcomeRequest
):
    try:
        log_outcome(
            customer_id=request.customer_id,
            actual_outcome=request.actual_outcome,
            model_version=metadata[
                "model_version"
            ]
        )

        return {
            "status": "recorded",
            "customer_id":
                request.customer_id,
            "actual_outcome":
                request.actual_outcome
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.get(
    "/monitoring/performance"
)
def production_performance():
    try:
        return evaluate_production_model()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
