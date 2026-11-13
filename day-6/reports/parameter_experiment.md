# Day-6 LLM Parameter Experiment Report

## Objective
Observe how `temperature` and `top_p` change a chat model's response while changing only one sampling parameter per experiment.

## Fixed experiment setup
- Prompt for every run: `Explain vector databases to a backend developer in exactly 3 bullet points.`
- Model: record the actual model returned by `/chat` for each run.
- `max_tokens`: 300 for every run.
- Keep system prompt and all other request fields identical.
- Run each condition at least once; repeat runs if you want to assess variability.

## Experiment A — Temperature
Keep `top_p=1.0` fixed. Change only temperature.

| Run | Temperature | Top-p | Actual model | Output (paste exact reply) | Notes: changed? / quality |
|---|---:|---:|---|---|---|
| A1 | 0.0 | 1.0 | Pending live run | Pending live run | Pending |
| A2 | 0.3 | 1.0 | Pending live run | Pending live run | Pending |
| A3 | 0.7 | 1.0 | Pending live run | Pending live run | Pending |
| A4 | 1.2 | 1.0 | Pending live run | Pending live run | Pending |

**Questions to answer after running:** Did the wording or examples change? Which setting was most consistent? Which was best for technical documentation? Which was best for creative writing?

## Experiment B — Top-p
Keep `temperature=0.7` fixed. Change only top-p.

| Run | Temperature | Top-p | Actual model | Output (paste exact reply) | Notes: style / consistency / usefulness |
|---|---:|---:|---|---|---|
| B1 | 0.7 | 0.1 | Pending live run | Pending live run | Pending |
| B2 | 0.7 | 0.5 | Pending live run | Pending live run | Pending |
| B3 | 0.7 | 0.9 | Pending live run | Pending live run | Pending |
| B4 | 0.7 | 1.0 | Pending live run | Pending live run | Pending |

## Why change either temperature or top-p, not both?
Both parameters influence token sampling. Changing both at the same time makes it difficult to attribute an output difference to one setting. A controlled experiment changes one variable and keeps the other fixed. Provider/model behavior can differ, so record actual outputs instead of assuming a universal effect.

## Interpretation guide (not measured results)
- Lower temperature usually makes sampling more focused and repeatable.
- Higher temperature usually allows more variation.
- Lower top-p restricts sampling to a smaller probability-mass set.
- These are tendencies, not guarantees; verify behavior with the configured provider/model.

## Final conclusion
Complete this section only after the eight live requests have been executed. Do not report example outputs or predicted behavior as measured results. Include any provider errors or unsupported-parameter responses as part of the experiment record.
