from fastapi.testclient import TestClient

from src.main import app
import src.main as main_module

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_experiments_endpoint_has_required_values():
    response = client.get("/experiments")
    assert response.status_code == 200
    body = response.json()["experiments"]
    assert [x["temperature"] for x in body["temperature"]] == [0.0, 0.3, 0.7, 1.2]
    assert [x["top_p"] for x in body["top_p"]] == [0.1, 0.5, 0.9, 1.0]


def test_chat_endpoint_returns_assignment_response(monkeypatch):
    async def fake_complete_chat(**kwargs):
        assert kwargs["model"] is None
        return {
            "model": "test-model", "reply": "RAG retrieves context before generation.",
            "prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22,
        }

    monkeypatch.setattr(main_module, "complete_chat", fake_complete_chat)
    response = client.post("/chat", json={"message": "Explain RAG"})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"].startswith("RAG retrieves")
    assert body["model"] == "test-model"
    assert body["prompt_tokens"] == 10
    assert body["completion_tokens"] == 12
    assert body["total_tokens"] == 22
    assert body["parameters"] == {"temperature": 0.7, "top_p": 1.0, "max_tokens": 512}
    assert "latency_ms" in body


def test_chat_rejects_invalid_temperature_with_400():
    response = client.post("/chat", json={"message": "Hello", "temperature": 3.0})
    assert response.status_code == 400


def test_chat_rejects_message_over_4000_characters():
    response = client.post("/chat", json={"message": "x" * 4001})
    assert response.status_code == 400


def test_chat_maps_provider_failure_to_502(monkeypatch):
    from src.llm_client import LLMClientError

    async def fail_chat(**kwargs):
        raise LLMClientError("Provider unavailable")

    monkeypatch.setattr(main_module, "complete_chat", fail_chat)
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Provider unavailable"
