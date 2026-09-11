import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

def check_health():
    response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )
    response.raise_for_status()
    return response.json()

def predict_customer(customer):
    response = requests.post(
        f"{API_URL}/predict",
        json={
            "customer": customer
        },
        timeout=10
    )
    response.raise_for_status()
    return response.json()

def get_model_info():
    response = requests.get(
        f"{API_URL}/model/info",
        timeout=5
    )
    response.raise_for_status()
    return response.json()

def optimize_campaign_api(budget, contact_cost, conversion_value, strategy, max_contacts):
    response = requests.post(
        f"{API_URL}/campaign/optimize",
        json={
            "budget": budget,
            "contact_cost": contact_cost,
            "conversion_value": conversion_value,
            "strategy": strategy,
            "max_contacts": max_contacts
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def get_drift_status():
    response = requests.get(
        f"{API_URL}/monitoring/drift",
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def get_prediction_monitoring():
    response = requests.get(
        f"{API_URL}/monitoring/predictions",
        timeout=10
    )
    response.raise_for_status()
    return response.json()

def get_production_performance():
    response = requests.get(
        f"{API_URL}/monitoring/performance",
        timeout=10
    )
    response.raise_for_status()
    return response.json()
