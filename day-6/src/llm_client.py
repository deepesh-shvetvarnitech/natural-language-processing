
import asyncio
import httpx

from src.config import settings
from src.prompts import build_messages, MENTOR_SYSTEM_PROMPT


class LLMClientError(RuntimeError):
    
    pass


async def complete_chat(
    message: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
    model: str | None = None,
) -> dict:
    

   
    if not settings.llm_configured:
        raise LLMClientError(
            "LLM API key is not configured. Add LLM_API_KEY or API_KEY to .env."
        )

   
    selected_model = model or settings.model

    
    selected_system_prompt = (
        system_prompt
        if system_prompt is not None
        else MENTOR_SYSTEM_PROMPT
    )

    messages = build_messages(
        message,
        selected_system_prompt,
    )

  
    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        settings.max_tokens_field: max_tokens,
    }

    # Step 5: Authentication headers
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }

    
    url = (
        f"{settings.base_url.rstrip('/')}"
        "/chat/completions"
    )

    
    for attempt in range(settings.max_retries + 1):

        try:
            async with httpx.AsyncClient(
                timeout=settings.timeout_seconds
            ) as client:

                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )

           
            if (
                response.status_code == 429
                or response.status_code >= 500
            ):
                if attempt < settings.max_retries:
                    wait_time = min(2 ** attempt, 8)
                    await asyncio.sleep(wait_time)
                    continue

           
            response.raise_for_status()

           
            data = response.json()

            
            answer = data["choices"][0]["message"]["content"]

            
            if isinstance(answer, list):
                answer = "".join(
                    part.get("text", "")
                    for part in answer
                    if isinstance(part, dict)
                )

            if not isinstance(answer, str):
                raise LLMClientError(
                    "The LLM returned an unexpected response."
                )

            
            usage = data.get("usage") or {}

            prompt_tokens = int(
                usage.get("prompt_tokens", 0) or 0
            )

            completion_tokens = int(
                usage.get("completion_tokens", 0) or 0
            )

            total_tokens = int(
                usage.get(
                    "total_tokens",
                    prompt_tokens + completion_tokens,
                ) or 0
            )

            
            return {
                "reply": answer,
                "model": data.get("model", selected_model),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

       
        except (httpx.TimeoutException, httpx.NetworkError):

            if attempt < settings.max_retries:
                wait_time = min(2 ** attempt, 8)
                await asyncio.sleep(wait_time)
                continue

            raise LLMClientError(
                "Could not reach the LLM provider."
            ) from None

      
        except httpx.HTTPStatusError as exc:

            status_code = exc.response.status_code

            raise LLMClientError(
                f"LLM provider returned HTTP {status_code}."
            ) from None

        
        except (KeyError, IndexError, ValueError, TypeError):

            raise LLMClientError(
                "The LLM provider returned an invalid response format."
            ) from None

   
    raise LLMClientError(
        "The LLM request failed after retries."
    )
