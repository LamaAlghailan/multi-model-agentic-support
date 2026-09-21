from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

Intent = Literal['authentication', 'network', 'deployment', 'database', 'gpu', 'api', 'package', 'general']
Route = Literal['qa', 'tools', 'support', 'escalate']

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class RouteDecision(Contract):
    route: Route
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)

    @field_validator('intent', mode='before')
    @classmethod
    def canonical_intent(cls, value):
        from src.routing.normalization import normalize_intent
        return normalize_intent(value)

class ToolResult(Contract):
    tool: str
    ok: bool
    backend: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

class ChatMessage(Contract):
    role: Literal['system', 'user', 'assistant']
    content: str = Field(min_length=1, max_length=16000)

class ChatRequest(Contract):
    model: str = 'tuwaiq-tech-support-agent'
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    stream: bool = False
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=180, ge=1, le=512)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=512)
    # Common optional client fields. Sampling is deliberately greedy.
    top_p: float | None = Field(default=None, ge=0, le=1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    seed: int | None = None
    user: str | None = Field(default=None, max_length=200)
    n: int = Field(default=1, ge=1, le=1)
    stop: str | list[str] | None = None
