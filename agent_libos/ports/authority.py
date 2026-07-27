from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol

from agent_libos.models import (
    AgentProcess,
    Capability,
    CapabilityStatus,
    DataLabels,
    TaskAuthorityManifest,
)


class AuthorityManifestPort(Protocol):
    """Read/admission surface for task authority manifests."""

    def get_for_process(self, pid: str | None) -> TaskAuthorityManifest | None:
        ...

    def assert_data_flow_labels(self, pid: str, labels: DataLabels | Any) -> None:
        ...


class CapabilityStorePort(Protocol):
    """Persistence surface used by capability decision and mutation services."""

    def transaction(self, *, include_object_payloads: bool = False) -> AbstractContextManager[Any]:
        ...

    def insert_capability(self, cap: Capability) -> None:
        ...

    def update_capability(self, cap: Capability) -> None:
        ...

    def transition_capability_status(
        self,
        cap_id: str,
        *,
        expected_status: CapabilityStatus,
        status: CapabilityStatus,
        metadata: dict[str, Any],
    ) -> Capability | None:
        ...

    def get_capability(self, cap_id: str) -> Capability | None:
        ...

    def list_capabilities(
        self,
        subject: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[Capability]:
        ...

    def query_capabilities(
        self,
        subject: str | None = None,
        *,
        active_only: bool,
        after_cap_id: str | None,
        limit: int,
    ) -> list[Capability]:
        ...

    def consume_capability_uses(self, cap_id: str, count: int = 1) -> Capability | None:
        ...

    def reserve_capability_uses(
        self,
        cap_id: str,
        reservation_id: str,
        *,
        count: int = 1,
        reserved_by: str,
        reason: str,
        created_at: str,
    ) -> Capability | None:
        ...

    def commit_capability_use_reservation(self, reservation_id: str, *, updated_at: str) -> bool:
        ...

    def restore_capability_use_reservation(self, reservation_id: str, *, updated_at: str) -> Capability | None:
        ...

    def get_process(self, pid: str) -> AgentProcess | None:
        ...

    def append_process_capability_ids(
        self,
        pid: str,
        capability_ids: list[str],
    ) -> AgentProcess:
        ...
