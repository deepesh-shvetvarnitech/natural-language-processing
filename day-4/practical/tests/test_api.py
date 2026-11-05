from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_search_request_validation():
    response = client.post("/similar-reviews", json={"query": "battery", "top_k": 0})
    assert response.status_code == 422
