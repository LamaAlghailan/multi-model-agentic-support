from pydantic import BaseModel


class SystemStatusResult(BaseModel):
    service: str
    status: str
    latency_ms: int
    message: str


def check_system_status(
    service: str,
) -> SystemStatusResult:
    """
    Deterministic mock tool for checking
    the current status of a service.
    """

    mock_statuses = {
        "database": {
            "status": "healthy",
            "latency_ms": 42,
        },
        "api": {
            "status": "degraded",
            "latency_ms": 180,
        },
        "network": {
            "status": "healthy",
            "latency_ms": 25,
        },
    }

    result = mock_statuses.get(
        service.lower(),
        {
            "status": "unknown",
            "latency_ms": 0,
        },
    )

    return SystemStatusResult(
        service=service,
        status=result["status"],
        latency_ms=result["latency_ms"],
        message=(
            f"Current status for {service}: "
            f"{result['status']}"
        ),
    )