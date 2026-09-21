from typing import Literal
from pydantic import BaseModel, Field


RouteName = Literal[
    "qa",
    "tools",
    "support",
    "escalate",
]


IntentName = Literal[
    "authentication",
    "network",
    "deployment",
    "database",
    "gpu",
    "api",
    "package",
    "general",
]


class RoutingDecision(BaseModel):
    route: RouteName
    intent: IntentName
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    reason: str