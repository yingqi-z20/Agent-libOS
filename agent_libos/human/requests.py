from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from agent_libos.models import HumanRequest
from agent_libos.storage.repositories import ProcessRepository


class HumanRequestService:
    """Durable HumanRequest repository boundary."""

    def __init__(self, processes: ProcessRepository) -> None:
        self._processes = processes

    def insert(self, request: HumanRequest) -> None:
        self._processes.insert_human_request(request)

    def update(self, request: HumanRequest) -> None:
        self._processes.update_human_request(request)

    def get(self, request_id: str) -> HumanRequest | None:
        return self._processes.get_human_request(request_id)

    def list(self, **filters: object) -> list[HumanRequest]:
        return self._processes.list_human_requests(**filters)

    def transaction(self) -> AbstractContextManager[Any]:
        return self._processes.transaction()
