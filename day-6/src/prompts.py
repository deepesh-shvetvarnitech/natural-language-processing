MENTOR_SYSTEM_PROMPT = (
    "You are a concise Python, FastAPI, SQL, and AI mentor. "
    "Explain concepts with practical backend examples."
)


def build_messages(
    message: str,
    system_prompt: str | None = None
) -> list[dict]:
   

    messages: list[dict] = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": message
    })

    return messages
