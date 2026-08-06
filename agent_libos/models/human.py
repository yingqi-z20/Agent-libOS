from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_libos.models.base import HumanRequestID, PID, StrEnum


class HumanRequestStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    CANCELLED = "cancelled"
    DELIVERED = "delivered"


@dataclass
class HumanRequest:
    request_id: HumanRequestID
    pid: PID
    human: str
    payload: dict[str, Any]
    status: HumanRequestStatus
    decision: dict[str, Any] | None
    blocking: bool
    created_at: str
    updated_at: str
    revision: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("human request revision must be a non-negative integer")
