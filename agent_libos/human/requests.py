from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from typing import Any

from agent_libos.models import HumanRequest
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage.repositories import ProcessRepository


class HumanRequestService:
    """Durable HumanRequest repository boundary."""

    def __init__(self, processes: ProcessRepository) -> None:
        self._processes = processes

    def insert(self, request: HumanRequest) -> None:
        self._processes.insert_human_request(request)

    def update(self, request: HumanRequest) -> None:
        """Compatibility update that fails closed when its revision is stale.

        New Human state transitions should use :meth:`compare_and_set` or
        :meth:`replace_current`, which keep the expected snapshot explicit and
        never mutate it before the durable compare-and-set succeeds.
        """

        if not self._processes.update_human_request(request):
            raise ValidationError(
                "human request changed concurrently: "
                f"{request.request_id} revision={request.revision}"
            )

    def compare_and_set(
        self,
        expected: HumanRequest,
        target: HumanRequest,
    ) -> HumanRequest:
        """Commit one exact revision/status transition or fail closed."""

        if not self._processes.compare_and_set_human_request(expected, target):
            raise ValidationError(
                "human request changed concurrently: "
                f"{expected.request_id} status={expected.status.value} "
                f"revision={expected.revision}"
            )
        return target

    def replace_current(
        self,
        expected: HumanRequest,
        **changes: Any,
    ) -> HumanRequest:
        """Build and CAS a next-revision value without mutating ``expected``."""

        if "revision" in changes:
            raise ValidationError("human request revision is managed by CAS")
        target = replace(
            expected,
            **changes,
            revision=expected.revision + 1,
        )
        return self.compare_and_set(expected, target)

    def get(self, request_id: str) -> HumanRequest | None:
        return self._processes.get_human_request(request_id)

    def list(self, **filters: object) -> list[HumanRequest]:
        return self._processes.list_human_requests(**filters)

    def transaction(self) -> AbstractContextManager[Any]:
        return self._processes.transaction()
