import pytest
from pydantic import ValidationError

from src.schemas import ChatRequest


def test_chat_request_defaults():
    request = ChatRequest(message="Explain RAG")
    assert request.temperature == 0.7
    assert request.top_p == 1.0
    assert request.max_tokens == 512
    assert request.model is None


def test_chat_request_accepts_custom_parameters_and_model():
    request = ChatRequest(
        message="Explain transformers", temperature=0.2,
        top_p=0.8, max_tokens=256, model="gemini-test"
    )
    assert request.temperature == 0.2
    assert request.top_p == 0.8
    assert request.max_tokens == 256
    assert request.model == "gemini-test"


@pytest.mark.parametrize("payload", [
    {"message": ""},
    {"message": "x" * 4001},
    {"message": "Hello", "temperature": 2.1},
    {"message": "Hello", "temperature": -0.1},
    {"message": "Hello", "top_p": -0.1},
    {"message": "Hello", "top_p": 1.1},
    {"message": "Hello", "max_tokens": 0},
    {"message": "Hello", "max_tokens": 2001},
])
def test_invalid_chat_requests_are_rejected(payload):
    with pytest.raises(ValidationError):
        ChatRequest(**payload)
