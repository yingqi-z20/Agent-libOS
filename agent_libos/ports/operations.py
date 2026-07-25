from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Protocol

from agent_libos.models import (
    OperationKind,
    OperationOutcome,
    OperationRecord,
)


class OperationPort(Protocol):
    """Narrow evidence surface needed by capability services."""

    def current_id(self) -> str | None:
        ...

    def current(self) -> Any | None:
        ...

    def operation_for_evidence(
        self,
        evidence_types: tuple[str, ...],
        evidence_id: str,
    ) -> list[Any]:
        ...

    def attach(self, operation_id: str) -> AbstractContextManager[Any]:
        ...

    def expect(self, *roles: str, operation_id: str | None = None) -> Any | None:
        ...

    def link_evidence(
        self,
        evidence_type: str,
        evidence_id: str,
        role: str,
        *,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...

    def scope(
        self,
        *,
        kind: str,
        name: str,
        actor: str,
        pid: str | None,
        expected_roles: list[str] | tuple[str, ...] = (),
        operation_id: str | None = None,
        parent_operation_id: str | None = None,
        auto_finish: bool = True,
    ) -> AbstractContextManager[Any]:
        ...

    def finish(self, outcome: str, *, operation_id: str | None = None) -> Any:
        ...


class RuntimePublicationOperationPort(Protocol):
    """Exact operation boundary required by publication reconciliation."""

    def current_id(self) -> str | None: ...

    def finish(
        self,
        outcome: OperationOutcome | str,
        *,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationRecord | None: ...

    def attach(self, operation_id: str) -> AbstractContextManager[Any]: ...

    def get_operation(self, operation_id: str) -> OperationRecord | None: ...

    def bind_runtime_publication(
        self,
        operation_id: str,
        *,
        publication_id: str,
        publication_kind: str,
        expected_kind: OperationKind | str,
        expected_name: str,
        expected_actor: str,
        expected_pid: str | None,
    ) -> OperationRecord: ...

    def runtime_publication_binding_operation_ids(
        self,
        publication_id: str,
    ) -> list[str]: ...

    def reconcile_runtime_publication(
        self,
        operation_id: str,
        outcome: OperationOutcome | str,
        *,
        publication_id: str,
        publication_kind: str,
        publication_state: str,
        publication_phase: str,
        expected_kind: OperationKind | str,
        expected_name: str,
        expected_actor: str,
        expected_pid: str | None,
        _publication_reconciled_marker: Callable[..., bool] | None = None,
    ) -> OperationRecord: ...
