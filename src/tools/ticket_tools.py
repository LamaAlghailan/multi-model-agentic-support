from pydantic import BaseModel
from typing import Literal
from uuid import uuid4


TicketPriority = Literal[
    "low",
    "medium",
    "high",
    "critical",
]


class TicketResult(BaseModel):
    ticket_id: str
    title: str
    description: str
    priority: TicketPriority
    status: str
    message: str


def create_ticket(
    title: str,
    description: str,
    priority: TicketPriority = "medium",
) -> TicketResult:
    """
    Deterministic mock tool for creating
    a technical support ticket.
    """

    ticket_id = f"TKT-{uuid4().hex[:8].upper()}"

    return TicketResult(
        ticket_id=ticket_id,
        title=title,
        description=description,
        priority=priority,
        status="created",
        message=(
            f"Support ticket {ticket_id} "
            f"was created successfully."
        ),
    )