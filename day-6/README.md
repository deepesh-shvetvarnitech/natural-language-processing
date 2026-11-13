# Day-6: LLM Basics and Chat Completion

A FastAPI chat-completion service using an OpenAI-compatible provider endpoint. The project demonstrates message roles, system prompts, request validation, configurable sampling parameters, token usage, retries, timeouts, and experiment planning.

## 1. Setup (Windows PowerShell)
Run from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set your provider's API key and model. The example is configured for a Gemini-compatible endpoint; confirm the model name available to your Google AI account. Do not share or commit `.env`.

```env
LLM_API_KEY=your_real_key_here
BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
MODEL=your-supported-model-name
TIMEOUT_SECONDS=30
MAX_RETRIES=3
MAX_TOKENS_FIELD=max_tokens
```

If you already have a working `.env`, do not overwrite it with `.env.example`.

## 2. Start the API

```powershell
uvicorn src.main:app --reload
```

Open Swagger: `http://127.0.0.1:8000/docs`.

## 3. Endpoints

- `GET /health` — API status, whether an API key is configured, and configured model.
- `POST /chat` — sends a chat-completion request.
- `GET /experiments` — returns the required temperature/top-p experiment values.

Example request:

```json
{
  "message": "Explain RAG with a practical backend example.",
  "system_prompt": "You are a concise technical mentor.",
  "temperature": 0.2,
  "top_p": 1.0,
  "max_tokens": 300
}
```

The `model` field is optional. If omitted, the model from `.env` is used. You can supply `model` to override it when the provider supports that model.

The response includes `reply`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `parameters`, and `latency_ms`. Some providers may not return token usage; in that case token counts are reported as zero.

## 4. Run tests

```powershell
pytest -v
```

Tests mock the external provider call, so they do not require a live API key or incur API charges.

## 5. Run parameter experiments

1. Keep the API running.
2. Open `notebooks/day06_llm_parameters.ipynb` in VS Code/Jupyter.
3. Run all cells. The notebook executes four temperature runs (`0.0`, `0.3`, `0.7`, `1.2`) with `top_p=1.0`, then four top-p runs (`0.1`, `0.5`, `0.9`, `1.0`) with `temperature=0.7`.
4. Review the actual outputs and complete `reports/parameter_experiment.md` with the measured observations. Do not fill the report with invented results.

## 6. Project structure

```text
day06_llm_chat_service/
├── requirements.txt
├── .env.example
├── notebooks/
│   └── day06_llm_parameters.ipynb
├── src/
│   ├── config.py
│   ├── experiments.py
│   ├── llm_client.py
│   ├── prompts.py
│   ├── schemas.py
│   └── main.py
├── reports/
│   ├── parameter_experiment.md
│   └── parameter_policy.md
├── tests/
│   ├── test_schemas.py
│   ├── test_llm_client.py
│   └── test_api.py
├── vector_db_design.md
└── README.md
```

## 7. Security and operational notes

- Never print, paste, or commit API keys. `.gitignore` excludes `.env`.
- `/health` reveals only whether a key is configured, never the key itself.
- The API maps invalid request data to HTTP 400 and upstream LLM failures to HTTP 502 as required by the assignment.
- Logs include token counts and latency but do not log prompts, API keys, or provider response bodies.
- Retry behavior uses exponential backoff for HTTP 429/5xx and transient network/timeouts.
- Provider model names and parameter support vary. Verify them against your provider's current documentation.

## 8. Vector database design

See [`vector_db_design.md`](vector_db_design.md) for the conceptual design of a vector search collection for 10,000 product FAQs, including embeddings, dimensions, similarity, metadata filters, top-k, and an example query.
