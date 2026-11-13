import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config import settings
from src.experiments import get_experiment_cases
from src.llm_client import complete_chat, LLMClientError
from src.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("day06_chat")

app = FastAPI(
    title="Day-6 LLM Chat Service",
    description="A configurable chat-completions API for the Day-6 assignment.",
    version="1.1.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        api_configured=settings.llm_configured,
        model=settings.model,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.perf_counter()
    try:
        result = await complete_chat(
            message=request.message,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            model=request.model,
        )
    except LLMClientError as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error("LLM request failed after %.2f ms: %s", latency_ms, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from None

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    parameters = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
    }
    logger.info(
        "LLM request succeeded model=%s prompt_tokens=%s completion_tokens=%s "
        "total_tokens=%s latency_ms=%.2f",
        result["model"], result["prompt_tokens"],
        result["completion_tokens"], result["total_tokens"], latency_ms,
    )
    return ChatResponse(
        reply=result["reply"],
        model=result["model"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
        parameters=parameters,
        latency_ms=latency_ms,
    )


@app.get("/experiments")
async def experiments():
    """Show example profiles; formal experiments vary one parameter at a time."""
    return {"experiments": get_experiment_cases()}
