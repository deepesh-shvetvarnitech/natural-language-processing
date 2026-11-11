from fastapi.testclient import TestClient

from src.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_sentiment():
    with TestClient(app) as client:
        response = client.post(
            "/sentiment",
            json={"review": "The battery drains quickly."}
        )

    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "confidence" in response.json()


def test_keywords():
    with TestClient(app) as client:
        response = client.post(
            "/keywords",
            json={"review": "The camera quality is poor."}
        )

    assert response.status_code == 200
    assert "keywords" in response.json()


def test_entities():
    with TestClient(app) as client:
        response = client.post(
            "/entities",
            json={"review": "Samsung Galaxy S24 battery drains quickly."}
        )

    assert response.status_code == 200
    assert "entities" in response.json()
    assert "issues" in response.json()


def test_similar():
    with TestClient(app) as client:
        response = client.post(
            "/similar",
            json={
                "review": "My phone battery drains very fast.",
                "top_k": 2
            }
        )

    assert response.status_code == 200
    assert "similar_reviews" in response.json()


def test_analyze_review():
    with TestClient(app) as client:
        response = client.post(
            "/analyze-review",
            json={
                "review": "The Samsung Galaxy S24 battery drains quickly.",
                "top_keywords": 5,
                "similar_reviews": 3
            }
        )

    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "keywords" in response.json()
    assert "entities" in response.json()
    assert "issues" in response.json()
    assert "similar_reviews" in response.json()


def test_empty_review():
    with TestClient(app) as client:
        response = client.post(
            "/sentiment",
            json={"review": " "}
        )

    assert response.status_code == 400


def test_too_long_review():
    with TestClient(app) as client:
        response = client.post(
            "/sentiment",
            json={"review": "a" * 5001}
        )

    assert response.status_code == 400