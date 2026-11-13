# LLM Parameter Selection Policy

## Defaults
- `temperature=0.7`: general-purpose starting point.
- `top_p=1.0`: leave nucleus sampling unrestricted unless an experiment justifies changing it.
- `max_tokens=512`: moderate response budget; increase only when the task needs a longer answer.

## Recommended starting points

| Task | Temperature | Top-p | Max tokens | Reason |
|---|---:|---:|---:|---|
| Code explanation / technical documentation | 0.0–0.3 | 1.0 | 500–1000 | Favor focused, repeatable wording |
| Ecommerce support FAQ | 0.1–0.3 | 1.0 | 250–500 | Keep answers consistent and concise |
| General learning chat | 0.5–0.8 | 1.0 | 512 | Balance clarity and natural wording |
| Brainstorming / creative writing | 0.9–1.2 | 1.0 | 700–1200 | Allow more varied ideas |

These are starting values only. Confirm that the selected provider/model supports the parameters and evaluate outputs for correctness.

## Operating rules
1. During experiments, change temperature OR top-p, not both at once.
2. Keep prompt, model, system prompt, and token limit fixed during comparisons.
3. Use retrieval/approved knowledge for factual support answers; sampling settings do not make unsupported facts reliable.
4. Never expose system prompts, API keys, or internal instructions in responses.
5. Set `max_tokens` high enough for the requested answer but keep a sensible limit to control latency and cost.
6. Log model, parameter values, token usage, latency, and sanitized errors. Never log credentials.

## Ecommerce support policy
Use a low-variance setting such as temperature 0.2 and top-p 1.0 as an initial policy. Retrieve the relevant product FAQ first, answer only from retrieved/approved information, and ask a clarifying question when retrieval confidence is low. Escalate policy, payment, safety, or account-specific issues to the appropriate support workflow.
