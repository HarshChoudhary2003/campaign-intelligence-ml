from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health():
    response = client.get(
        "/health"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_ready():
    response = client.get(
        "/ready"
    )
    assert response.status_code == 200

def test_model_info():
    response = client.get(
        "/model/info"
    )
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
