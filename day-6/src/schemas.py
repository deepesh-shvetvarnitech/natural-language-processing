from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Validated input for one chat-completion request."""

    message: str = Field(..., min_length=1, max_length=4000)
    system_prompt: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=2000)
    
    model: str | None = None


class ChatResponse(BaseModel):
  

    reply: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    parameters: dict
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    api_configured: bool
    model: str
