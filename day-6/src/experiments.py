

TEMPERATURE_VALUES = [0.0, 0.3, 0.7, 1.2]
TOP_P_VALUES = [0.1, 0.5, 0.9, 1.0]
EXPERIMENT_PROMPT = (
    "Explain vector databases to a backend developer in exactly 3 bullet points."
)


def get_experiment_cases() -> dict:
    """Return independent temperature and top-p experiment plans."""
    return {
        "prompt": EXPERIMENT_PROMPT,
        "temperature": [
            {"temperature": value, "top_p": 1.0} for value in TEMPERATURE_VALUES
        ],
        "top_p": [
            {"temperature": 0.7, "top_p": value} for value in TOP_P_VALUES
        ],
        "max_tokens": 300,
        "note": "Change one sampling parameter at a time; keep the other fixed.",
    }
