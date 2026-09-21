from pydantic import BaseModel
from typing import List


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


class LogResult(BaseModel):
    service: str
    entries: List[LogEntry]
    count: int


def get_logs(
    service: str,
    limit: int = 5,
) -> LogResult:
    """
    Deterministic mock tool for retrieving
    recent service logs.
    """

    mock_logs = {
        "api": [
            LogEntry(
                timestamp="2026-09-21T08:10:00",
                level="ERROR",
                message="HTTP 500 returned from /users endpoint",
            ),
            LogEntry(
                timestamp="2026-09-21T08:09:32",
                level="WARNING",
                message="API response latency exceeded threshold",
            ),
            LogEntry(
                timestamp="2026-09-21T08:08:15",
                level="INFO",
                message="API service started successfully",
            ),
        ],

        "database": [
            LogEntry(
                timestamp="2026-09-21T08:12:00",
                level="WARNING",
                message="Database query latency increased",
            ),
            LogEntry(
                timestamp="2026-09-21T08:11:10",
                level="INFO",
                message="Database connection pool is healthy",
            ),
        ],

        "network": [
            LogEntry(
                timestamp="2026-09-21T08:15:00",
                level="ERROR",
                message="DNS resolution failed temporarily",
            ),
            LogEntry(
                timestamp="2026-09-21T08:14:20",
                level="INFO",
                message="Network connectivity restored",
            ),
        ],
    }

    entries = mock_logs.get(
        service.lower(),
        [],
    )

    entries = entries[:limit]

    return LogResult(
        service=service,
        entries=entries,
        count=len(entries),
    )