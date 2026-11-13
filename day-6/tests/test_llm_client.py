from src.prompts import build_messages, MENTOR_SYSTEM_PROMPT
from src.experiments import get_experiment_cases


def test_build_messages_includes_system_and_user():
    messages = build_messages("Explain RAG", MENTOR_SYSTEM_PROMPT)
    assert messages[0] == {"role": "system", "content": MENTOR_SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "Explain RAG"}


def test_build_messages_without_system_prompt():
    assert build_messages("Hello") == [{"role": "user", "content": "Hello"}]


def test_parameter_experiment_plan_contains_exact_values():
    cases = get_experiment_cases()
    assert [x["temperature"] for x in cases["temperature"]] == [0.0, 0.3, 0.7, 1.2]
    assert [x["top_p"] for x in cases["top_p"]] == [0.1, 0.5, 0.9, 1.0]
