from src.schemas import RoutingDecision


HIGH_RISK_PATTERNS = [
    "security breach",
    "data breach",
    "database corruption",
    "data corruption",
    "data loss",
    "production data loss",
    "compromised",
    "ransomware",
    "unauthorized access",
    "delete production",
    "drop table",
    "truncate table",
]


TOOL_PATTERNS = [
    "current status",
    "right now",
    "health check",
    "check logs",
    "logs",
    "ticket",
    "package version",
    "system status",
]


def apply_hard_rules(
    user_message: str,
) -> RoutingDecision | None:

    text = user_message.lower().strip()

    for pattern in HIGH_RISK_PATTERNS:
        if pattern in text:
            return RoutingDecision(
                route="escalate",
                intent="general",
                confidence=1.0,
                reason=(
                    f"Hard safety rule matched: {pattern}"
                ),
            )

    for pattern in TOOL_PATTERNS:
        if pattern in text:
            return RoutingDecision(
                route="tools",
                intent="general",
                confidence=1.0,
                reason=(
                    f"Tool-required rule matched: {pattern}"
                ),
            )

    return None