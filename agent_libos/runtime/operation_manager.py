from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any, Callable, Iterable, Iterator, TypeVar, cast

from agent_libos.models import (
    OperationEvidenceLink,
    OperationEvidenceRole,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    OperationState,
    StaleOperationRecoverySummary,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    PolicyDenied,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    ResourceLimitExceeded,
    RuntimePublicationPending,
    RuntimeRecoveryRequired,
    ValidationError,
)
from agent_libos.storage import (
    OperationRepositoryProtocol,
    RuntimePublicationRepositoryProtocol,
)
from agent_libos.utils.ids import new_id, utc_now


@dataclass(frozen=True)
class _CurrentOperation:
    manager_identity: int
    operation_id: str


_CURRENT_OPERATION: ContextVar[_CurrentOperation | None] = ContextVar(
    "agent_libos_current_operation",
    default=None,
)

F = TypeVar("F", bound=Callable[..., Any])
_RUNTIME_PUBLICATION_BINDING_VERSION = 1
_RUNTIME_PUBLICATION_METADATA_PREFIX = "runtime_publication_"
_CHECKPOINT_FORK_OPERATION_NAME = "checkpoint.fork"
_CHECKPOINT_FORK_RECEIPT_ATTRIBUTE = "checkpoint_fork_receipt"
_CHECKPOINT_FORK_RECEIPT_KEYS = frozenset(
    {
        "checkpoint_id", "source_pid", "fork_root_pid", "pid_map",
        "object_map", "tool_map", "status", "main_state_committed",
        "reconciliation_pending", "post_commit_failures", "outcome_diagnostic",
    }
)
_CHECKPOINT_FORK_FAILURE_KEYS = frozenset(
    {
        "phase", "error_type", "message", "audit_error_type", "audit_error",
        "failure_record_error_type", "failure_record_error",
    }
)
_CHECKPOINT_FORK_DIAGNOSTIC_KEYS = frozenset(
    {
        "phase", "interruption_error_type", "interruption",
        "diagnostic_error_type", "diagnostic_error",
        "prepared_runtime_assets_retained", "fork_subtree_quarantined",
        "recovery_signal_record_id", "recovery_signal_error_type",
        "recovery_signal_error", "lifecycle_fence_requested",
        "operation_recovery_signal_recorded",
        "operation_recovery_signal_error_type",
        "operation_recovery_signal_error", "lifecycle_fence_error_type",
        "lifecycle_fence_error", "lifecycle_fenced",
    }
)
_CHECKPOINT_FORK_MAX_MAP_ITEMS = 4096
_CHECKPOINT_FORK_MAX_FAILURES = 32
_CHECKPOINT_FORK_MAX_TEXT = 1024


def _bounded_fork_text(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _CHECKPOINT_FORK_MAX_TEXT
    ):
        return None
    return value


def _canonical_checkpoint_fork_identity(
    value: Mapping[Any, Any],
) -> dict[str, str] | None:
    selected: dict[str, str] = {}
    for key in ("checkpoint_id", "source_pid", "fork_root_pid", "status"):
        text = _bounded_fork_text(value.get(key))
        if text is None:
            return None
        selected[key] = text
    return selected


def _canonical_checkpoint_fork_outcome(
    value: Mapping[Any, Any],
    status: str,
) -> tuple[bool | None, bool] | None:
    committed = value.get("main_state_committed")
    pending = value.get("reconciliation_pending", False)
    if committed is not None and not isinstance(committed, bool):
        return None
    if not isinstance(pending, bool):
        return None
    valid_combinations = {
        ("forked", True, False),
        ("forked_with_warnings", True, False),
        ("fork_outcome_unknown", None, True),
        ("fork_recovery_required", True, True),
    }
    if (status, committed, pending) not in valid_combinations:
        return None
    return committed, pending


def _canonical_checkpoint_fork_map(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping) or len(value) > _CHECKPOINT_FORK_MAX_MAP_ITEMS:
        return None
    normalized: dict[str, str] = {}
    for raw_source, raw_target in value.items():
        source = _bounded_fork_text(raw_source)
        target = _bounded_fork_text(raw_target)
        if source is None or target is None:
            return None
        normalized[source] = target
    return normalized if len(normalized) == len(value) else None


def _canonical_checkpoint_fork_maps(
    value: Mapping[Any, Any],
) -> dict[str, dict[str, str]] | None:
    selected: dict[str, dict[str, str]] = {}
    for key in ("pid_map", "object_map", "tool_map"):
        normalized = _canonical_checkpoint_fork_map(value.get(key))
        if normalized is None:
            return None
        selected[key] = normalized
    return selected


def _canonical_checkpoint_fork_failure(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping) or set(value) - _CHECKPOINT_FORK_FAILURE_KEYS:
        return None
    selected: dict[str, str] = {}
    for key, raw_text in value.items():
        if not isinstance(raw_text, str) or not raw_text:
            return None
        selected[str(key)] = raw_text[:_CHECKPOINT_FORK_MAX_TEXT]
    required = {"phase", "error_type", "message"}
    return selected if required <= set(selected) else None


def _checkpoint_fork_failure_count_matches_status(
    status: str,
    failures: list[dict[str, str]],
) -> bool:
    if status == "forked":
        return not failures
    if status in {"forked_with_warnings", "fork_recovery_required"}:
        return bool(failures)
    return True


def _canonical_checkpoint_fork_failures(
    value: Mapping[Any, Any],
    status: str,
) -> list[dict[str, str]] | None:
    raw_failures = value.get("post_commit_failures")
    if (
        not isinstance(raw_failures, list)
        or len(raw_failures) > _CHECKPOINT_FORK_MAX_FAILURES
    ):
        return None
    failures: list[dict[str, str]] = []
    for raw_failure in raw_failures:
        failure = _canonical_checkpoint_fork_failure(raw_failure)
        if failure is None:
            return None
        failures.append(failure)
    if not _checkpoint_fork_failure_count_matches_status(status, failures):
        return None
    return failures


def _canonical_checkpoint_fork_diagnostic(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) - _CHECKPOINT_FORK_DIAGNOSTIC_KEYS:
        return None
    diagnostic: dict[str, Any] = {}
    for raw_key, raw_item in value.items():
        key = _bounded_fork_text(raw_key)
        if key is None or not isinstance(raw_item, (str, bool, int, type(None))):
            return None
        diagnostic[key] = (
            raw_item[:_CHECKPOINT_FORK_MAX_TEXT]
            if isinstance(raw_item, str)
            else raw_item
        )
    return diagnostic


def _canonical_checkpoint_fork_receipt(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) - _CHECKPOINT_FORK_RECEIPT_KEYS:
        return None
    identity = _canonical_checkpoint_fork_identity(value)
    if identity is None:
        return None
    selected: dict[str, Any] = dict(identity)
    status = identity["status"]
    outcome = _canonical_checkpoint_fork_outcome(value, status)
    if outcome is None:
        return None
    maps = _canonical_checkpoint_fork_maps(value)
    if maps is None:
        return None
    failures = _canonical_checkpoint_fork_failures(value, status)
    if failures is None:
        return None
    committed, pending = outcome
    selected["main_state_committed"] = committed
    selected["reconciliation_pending"] = pending
    selected.update(maps)
    selected["post_commit_failures"] = failures
    raw_diagnostic = value.get("outcome_diagnostic")
    if raw_diagnostic is not None:
        diagnostic = _canonical_checkpoint_fork_diagnostic(raw_diagnostic)
        if diagnostic is None:
            return None
        selected["outcome_diagnostic"] = diagnostic
    elif pending:
        return None
    return selected


def _validated_public_operation_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = dict(metadata or {})
    if any(
        str(key).startswith(_RUNTIME_PUBLICATION_METADATA_PREFIX)
        for key in selected
    ):
        raise ValidationError(
            "runtime publication metadata is reserved for durable binding"
        )
    return selected


class OperationManager:
    """Durable causal scopes for protected Agent libOS operations."""

    def __init__(
        self,
        store: OperationRepositoryProtocol,
        publications: RuntimePublicationRepositoryProtocol | None = None,
        *,
        recovery_page_size: int = 500,
        require_recovery_lease: Callable[[], None] | None = None,
        recovery_terminalization_scope: (
            Callable[[str], AbstractContextManager[Any]] | None
        ) = None,
        current_mutation_admission_is_stale: Callable[[], bool] | None = None,
    ):
        if (
            isinstance(recovery_page_size, bool)
            or not isinstance(recovery_page_size, int)
            or recovery_page_size <= 0
        ):
            raise ValueError("operation recovery page size must be positive")
        self.store = store
        self.publications: RuntimePublicationRepositoryProtocol = (
            publications
            if publications is not None
            else cast(RuntimePublicationRepositoryProtocol, store)
        )
        self._identity = id(self)
        self._recovery_page_size = recovery_page_size
        self._require_recovery_lease = (
            require_recovery_lease
            if require_recovery_lease is not None
            else self._recovery_lease_not_configured
        )
        self._recovery_terminalization_scope = recovery_terminalization_scope
        self._current_mutation_admission_is_stale = (
            current_mutation_admission_is_stale
        )

    def current_id(self) -> str | None:
        current = _CURRENT_OPERATION.get()
        if current is None or current.manager_identity != self._identity:
            return None
        return current.operation_id

    def current(self) -> OperationRecord | None:
        operation_id = self.current_id()
        return self.store.get_operation(operation_id) if operation_id is not None else None

    def start(
        self,
        *,
        kind: OperationKind | str,
        name: str,
        actor: str,
        pid: str | None,
        parent_operation_id: str | None = None,
        expected_roles: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> OperationRecord:
        parent_id = parent_operation_id if parent_operation_id is not None else self.current_id()
        parent = self.store.get_operation(parent_id) if parent_id is not None else None
        if parent_id is not None and parent is None:
            raise ValueError(f"parent operation not found: {parent_id}")
        operation_id = new_id("op")
        now = utc_now()
        selected_metadata = _validated_public_operation_metadata(metadata)
        record = OperationRecord(
            operation_id=operation_id,
            root_operation_id=parent.root_operation_id if parent is not None else operation_id,
            parent_operation_id=parent_id,
            kind=OperationKind(kind),
            name=str(name),
            actor=str(actor),
            pid=str(pid) if pid is not None else None,
            state=OperationState.RUNNING,
            outcome=OperationOutcome.PENDING,
            expected_roles=sorted({str(value) for value in expected_roles}),
            metadata=selected_metadata,
            started_at=now,
            updated_at=now,
        )
        self.store.insert_operation(record)
        return record

    def resume(self, operation_id: str) -> OperationRecord:
        with self.store.locked():
            record = self._require(operation_id)
            if record.state == OperationState.TERMINAL:
                return record
            if record.state == OperationState.RUNNING:
                return record
            updated = replace(
                record,
                state=OperationState.RUNNING,
                outcome=OperationOutcome.PENDING,
                updated_at=utc_now(),
                completed_at=None,
            )
            if not self.store.update_operation(updated, expected_states=[OperationState.WAITING.value]):
                return self._require(operation_id)
            return updated

    def expect(self, *roles: OperationEvidenceRole | str, operation_id: str | None = None) -> OperationRecord | None:
        selected_id = operation_id or self.current_id()
        if selected_id is None:
            return None
        with self.store.locked():
            record = self._require(selected_id)
            expected = sorted({*record.expected_roles, *(str(role) for role in roles)})
            if expected == record.expected_roles:
                return record
            updated = replace(record, expected_roles=expected, updated_at=utc_now())
            self.store.update_operation(updated)
            return updated

    def merge_metadata(self, metadata: dict[str, Any], *, operation_id: str | None = None) -> OperationRecord | None:
        selected_metadata = _validated_public_operation_metadata(metadata)
        selected_id = operation_id or self.current_id()
        if selected_id is None:
            return None
        with self.store.locked():
            record = self._require(selected_id)
            updated = replace(
                record,
                metadata={**record.metadata, **selected_metadata},
                updated_at=utc_now(),
            )
            self.store.update_operation(updated)
            return updated

    def set_pid(self, pid: str, *, operation_id: str | None = None) -> OperationRecord | None:
        selected_id = operation_id or self.current_id()
        if selected_id is None:
            return None
        with self.store.locked():
            record = self._require(selected_id)
            if record.pid == str(pid):
                return record
            updated = replace(record, pid=str(pid), updated_at=utc_now())
            self.store.update_operation(updated)
            return updated

    def finish(
        self,
        outcome: OperationOutcome | str,
        *,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationRecord | None:
        return self._finish(
            outcome,
            operation_id=operation_id,
            metadata=_validated_public_operation_metadata(metadata),
        )

    def _finish(
        self,
        outcome: OperationOutcome | str,
        *,
        operation_id: str | None,
        metadata: dict[str, Any],
    ) -> OperationRecord | None:
        selected_id = operation_id or self.current_id()
        if selected_id is None:
            return None
        with self.store.locked():
            record = self._require(selected_id)
            if record.state == OperationState.TERMINAL:
                return record
            selected_outcome = OperationOutcome(outcome)
            selected_metadata = dict(metadata)
            if (
                selected_outcome == OperationOutcome.SUCCEEDED
                and self._has_unknown_external_effect(selected_id)
            ):
                selected_outcome = OperationOutcome.UNKNOWN
                selected_metadata.setdefault(
                    "outcome_adjustment",
                    "succeeded_with_unknown_external_effect",
                )
            now = utc_now()
            updated = replace(
                record,
                state=OperationState.TERMINAL,
                outcome=selected_outcome,
                metadata={**record.metadata, **selected_metadata},
                updated_at=now,
                completed_at=now,
            )
            if not self.store.update_operation(
                updated,
                expected_states=[OperationState.RUNNING.value, OperationState.WAITING.value],
            ):
                return self._require(selected_id)
            return updated

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
    ) -> OperationRecord:
        """Persist the publication-to-operation association during planning."""

        with self.store.locked():
            self._require_runtime_publication_plan(
                operation_id,
                publication_id=publication_id,
                publication_kind=publication_kind,
                required_state="planning",
            )

            record = self._require_runtime_publication_operation_identity(
                operation_id,
                publication_id=publication_id,
                expected_kind=expected_kind,
                expected_name=expected_name,
                expected_actor=expected_actor,
                expected_pid=expected_pid,
            )
            if record.state != OperationState.RUNNING:
                raise ValidationError(
                    "runtime publication can only bind its active operation: "
                    f"{publication_id} -> {operation_id}"
                )
            binding_operation_ids = self.runtime_publication_binding_operation_ids(
                publication_id
            )
            if binding_operation_ids not in ([], [str(operation_id)]):
                raise ValidationError(
                    "runtime publication is already bound to another operation: "
                    f"{publication_id} -> {binding_operation_ids}"
                )
            existing_publication_id = record.metadata.get("runtime_publication_id")
            if (
                existing_publication_id is not None
                and str(existing_publication_id) != str(publication_id)
            ):
                raise ValidationError(
                    "operation is already bound to another runtime publication: "
                    f"{operation_id} -> {existing_publication_id}"
                )
            metadata = {
                **record.metadata,
                "runtime_publication_id": str(publication_id),
                "runtime_publication_kind": str(publication_kind),
                "runtime_publication_bound": True,
                "runtime_publication_binding_version": (
                    _RUNTIME_PUBLICATION_BINDING_VERSION
                ),
            }
            if record.metadata == metadata:
                return record
            updated = replace(record, metadata=metadata, updated_at=utc_now())
            if not self.store.update_operation(
                updated,
                expected_states=[OperationState.RUNNING.value],
            ):
                latest = self._require(operation_id)
                if (
                    latest.state == OperationState.RUNNING
                    and latest.metadata.get("runtime_publication_id")
                    == str(publication_id)
                    and latest.metadata.get("runtime_publication_kind")
                    == str(publication_kind)
                    and latest.metadata.get("runtime_publication_bound") is True
                    and latest.metadata.get("runtime_publication_binding_version")
                    == _RUNTIME_PUBLICATION_BINDING_VERSION
                ):
                    return latest
                raise RuntimeError(
                    "operation changed during runtime publication binding: "
                    f"{operation_id}"
                )
            return updated

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
    ) -> OperationRecord:
        """Authoritatively converge an operation from its durable publication."""

        selected_outcome = OperationOutcome(outcome)
        if selected_outcome == OperationOutcome.PENDING:
            raise ValueError("runtime publication cannot reconcile to pending")
        with self.store.locked():
            self._require_runtime_publication_plan(
                operation_id,
                publication_id=publication_id,
                publication_kind=publication_kind,
                required_state=publication_state,
                required_phase=publication_phase,
            )
            record = self._require_runtime_publication_operation_identity(
                operation_id,
                publication_id=publication_id,
                expected_kind=expected_kind,
                expected_name=expected_name,
                expected_actor=expected_actor,
                expected_pid=expected_pid,
            )
            self._require_exact_runtime_publication_binding(
                record,
                publication_id=publication_id,
                publication_kind=publication_kind,
            )
            now = utc_now()
            metadata = {
                **record.metadata,
                "runtime_publication_id": str(publication_id),
                "runtime_publication_state": str(publication_state),
                "runtime_publication_phase": str(publication_phase),
                "runtime_publication_reconciled": True,
            }
            metadata.setdefault(
                "runtime_publication_original_operation_state",
                record.state.value,
            )
            metadata.setdefault(
                "runtime_publication_original_operation_outcome",
                record.outcome.value,
            )
            if (
                record.state == OperationState.TERMINAL
                and record.outcome == selected_outcome
                and record.metadata == metadata
            ):
                self._mark_runtime_publication_operation_reconciled(
                    operation_id,
                    publication_id=publication_id,
                    publication_kind=publication_kind,
                    publication_state=publication_state,
                    publication_phase=publication_phase,
                    reconciled_marker=_publication_reconciled_marker,
                )
                return record
            updated = replace(
                record,
                state=OperationState.TERMINAL,
                outcome=selected_outcome,
                metadata=metadata,
                updated_at=now,
                completed_at=record.completed_at or now,
            )
            if not self.store.update_operation(
                updated,
                expected_states=[record.state.value],
            ):
                latest = self._require(operation_id)
                if (
                    latest.state == OperationState.TERMINAL
                    and latest.outcome == selected_outcome
                    and latest.metadata.get("runtime_publication_id")
                    == str(publication_id)
                ):
                    self._require_exact_runtime_publication_binding(
                        latest,
                        publication_id=publication_id,
                        publication_kind=publication_kind,
                    )
                    self._mark_runtime_publication_operation_reconciled(
                        operation_id,
                        publication_id=publication_id,
                        publication_kind=publication_kind,
                        publication_state=publication_state,
                        publication_phase=publication_phase,
                        reconciled_marker=_publication_reconciled_marker,
                    )
                    return latest
                raise RuntimeError(
                    "operation changed during runtime publication reconciliation: "
                    f"{operation_id}"
                )
            self._mark_runtime_publication_operation_reconciled(
                operation_id,
                publication_id=publication_id,
                publication_kind=publication_kind,
                publication_state=publication_state,
                publication_phase=publication_phase,
                reconciled_marker=_publication_reconciled_marker,
            )
            return updated

    def _mark_runtime_publication_operation_reconciled(
        self,
        operation_id: str,
        *,
        publication_id: str,
        publication_kind: str,
        publication_state: str,
        publication_phase: str,
        reconciled_marker: Callable[..., bool] | None,
    ) -> None:
        marker = (
            reconciled_marker
            or self.publications.mark_runtime_publication_operation_reconciled
        )
        if not marker(
            publication_id,
            expected_kind=publication_kind,
            expected_state=publication_state,
            expected_phase=publication_phase,
            expected_operation_id=operation_id,
        ):
            raise RuntimeError(
                "runtime publication changed while marking operation reconciliation: "
                f"{publication_id} -> {operation_id}"
            )

    def runtime_publication_binding_operation_ids(
        self,
        publication_id: str,
    ) -> list[str]:
        """Return every operation carrying a reverse link to a publication."""

        return self.store.list_operation_ids_by_runtime_publication_id(
            str(publication_id)
        )

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        """Read one operation through the manager's typed repository boundary."""

        return self.store.get_operation(str(operation_id))

    def _require_runtime_publication_plan(
        self,
        operation_id: str,
        *,
        publication_id: str,
        publication_kind: str,
        required_state: str,
        required_phase: str | None = None,
    ) -> dict[str, Any]:
        publication = self.publications.get_runtime_publication(publication_id)
        if publication is None:
            raise ValidationError(
                f"runtime publication is missing: {publication_id}"
            )
        plan = publication["plan"]
        matches = (
            publication["kind"] == str(publication_kind)
            and str(plan.get("operation_id") or "") == str(operation_id)
            and plan.get("operation_binding_version")
            == _RUNTIME_PUBLICATION_BINDING_VERSION
            and publication["state"] == str(required_state)
            and (
                required_phase is None
                or publication["phase"] == str(required_phase)
            )
        )
        if not matches:
            raise ValidationError(
                "runtime publication binding changed: "
                f"{publication_id} -> {operation_id}"
            )
        return publication

    def _require_exact_runtime_publication_binding(
        self,
        record: OperationRecord,
        *,
        publication_id: str,
        publication_kind: str,
    ) -> None:
        binding_operation_ids = self.runtime_publication_binding_operation_ids(
            publication_id
        )
        metadata = record.metadata
        if (
            binding_operation_ids != [record.operation_id]
            or metadata.get("runtime_publication_bound") is not True
            or metadata.get("runtime_publication_kind") != str(publication_kind)
            or metadata.get("runtime_publication_binding_version")
            != _RUNTIME_PUBLICATION_BINDING_VERSION
        ):
            raise ValidationError(
                "operation is not the exact durable runtime publication binding: "
                f"{record.operation_id} -> {publication_id}"
            )

    def _require_runtime_publication_operation_identity(
        self,
        operation_id: str,
        *,
        publication_id: str,
        expected_kind: OperationKind | str,
        expected_name: str,
        expected_actor: str,
        expected_pid: str | None,
    ) -> OperationRecord:
        record = self.store.get_operation(operation_id)
        if record is None:
            raise ValidationError(
                "runtime publication references a missing operation: "
                f"{publication_id} -> {operation_id}"
            )

        selected_kind = OperationKind(expected_kind)
        selected_pid = str(expected_pid) if expected_pid is not None else None
        identity_mismatches: list[str] = []
        if record.kind != selected_kind:
            identity_mismatches.append(
                f"kind={record.kind.value!r} (expected {selected_kind.value!r})"
            )
        if record.name != str(expected_name):
            identity_mismatches.append(
                f"name={record.name!r} (expected {str(expected_name)!r})"
            )
        if record.actor != str(expected_actor):
            identity_mismatches.append(
                f"actor={record.actor!r} (expected {str(expected_actor)!r})"
            )
        if record.pid != selected_pid:
            identity_mismatches.append(
                f"pid={record.pid!r} (expected {selected_pid!r})"
            )
        if identity_mismatches:
            raise ValidationError(
                "runtime publication operation identity mismatch: "
                f"{publication_id} -> {operation_id} "
                f"({'; '.join(identity_mismatches)})"
            )
        return record

    def wait(
        self,
        *,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationRecord | None:
        selected_metadata = _validated_public_operation_metadata(metadata)
        selected_id = operation_id or self.current_id()
        if selected_id is None:
            return None
        with self.store.locked():
            record = self._require(selected_id)
            if record.state == OperationState.TERMINAL:
                return record
            updated = replace(
                record,
                state=OperationState.WAITING,
                outcome=OperationOutcome.PENDING,
                metadata={**record.metadata, **selected_metadata},
                updated_at=utc_now(),
                completed_at=None,
            )
            if not self.store.update_operation(
                updated,
                expected_states=[OperationState.RUNNING.value, OperationState.WAITING.value],
            ):
                return self._require(selected_id)
            return updated

    def link_evidence(
        self,
        evidence_type: str,
        evidence_id: str,
        role: OperationEvidenceRole | str,
        *,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationEvidenceLink | None:
        selected_id = operation_id or self.current_id()
        if selected_id is None:
            return None
        if self.store.get_operation(selected_id) is None:
            return None
        link = OperationEvidenceLink(
            link_id=new_id("oplink"),
            operation_id=selected_id,
            evidence_type=str(evidence_type),
            evidence_id=str(evidence_id),
            role=str(role),
            created_at=utc_now(),
            metadata=dict(metadata or {}),
        )
        return link if self.store.insert_operation_evidence(link) else None

    def operation_for_evidence(self, evidence_types: Iterable[str], evidence_id: str) -> list[OperationRecord]:
        links = self.store.list_operation_evidence(
            evidence_types=list(evidence_types),
            evidence_id=str(evidence_id),
        )
        ids = sorted({link.operation_id for link in links})
        return [record for operation_id in ids if (record := self.store.get_operation(operation_id)) is not None]

    def interrupt_stale_running(self) -> StaleOperationRecoverySummary:
        self._require_recovery_lease()
        interrupted_sample: list[str] = []
        interrupted_total = 0
        cursor = None
        with self.store.stale_operation_recovery_index():
            while True:
                page = self.store.scan_stale_running_operations(
                    after=cursor,
                    limit=self._recovery_page_size,
                )
                unknown_ids = self.store.operation_ids_with_unknown_external_effects(
                    record.operation_id for record in page.records
                )
                for record in page.records:
                    pending_effect = record.operation_id in unknown_ids
                    updated = self.finish(
                        (
                            OperationOutcome.UNKNOWN
                            if pending_effect
                            else OperationOutcome.INTERRUPTED
                        ),
                        operation_id=record.operation_id,
                        metadata={
                            "recovery": (
                                "stale_running_with_pending_external_effect"
                                if pending_effect
                                else "stale_running_operation"
                            )
                        },
                    )
                    if updated is not None and updated.outcome in {
                        OperationOutcome.INTERRUPTED,
                        OperationOutcome.UNKNOWN,
                    }:
                        interrupted_total += 1
                        if len(interrupted_sample) < self._recovery_page_size:
                            interrupted_sample.append(updated.operation_id)
                cursor = page.next_cursor
                if cursor is None:
                    break
        return StaleOperationRecoverySummary(
            total_count=interrupted_total,
            sample_operation_ids=tuple(interrupted_sample),
        )

    @staticmethod
    def _recovery_lease_not_configured() -> None:
        raise RuntimeError("operation recovery requires a configured recovery lease")

    @contextmanager
    def activate(self, operation_id: str) -> Iterator[OperationRecord]:
        record = self.resume(operation_id)
        token = self._set_current(record.operation_id)
        try:
            yield record
        finally:
            _CURRENT_OPERATION.reset(token)

    @contextmanager
    def attach(self, operation_id: str) -> Iterator[OperationRecord]:
        """Attach evidence to an operation without changing its lifecycle state."""
        record = self._require(operation_id)
        token = self._set_current(record.operation_id)
        try:
            yield record
        finally:
            _CURRENT_OPERATION.reset(token)

    def _owns_pending_runtime_publication(
        self,
        operation_id: str,
        pending: RuntimePublicationPending | RuntimeRecoveryRequired,
    ) -> bool:
        publication = self.publications.get_runtime_publication(
            pending.publication_id
        )
        operation = self.store.get_operation(operation_id)
        return bool(
            publication is not None
            and operation is not None
            and operation.state != OperationState.TERMINAL
            and publication["state"]
            in {"planning", "applying", "reconciliation_pending", "rollback_pending"}
            and str(publication["plan"].get("operation_id") or "") == operation_id
            and publication["plan"].get("operation_binding_version")
            == _RUNTIME_PUBLICATION_BINDING_VERSION
            and self.runtime_publication_binding_operation_ids(
                pending.publication_id
            )
            == [operation_id]
            and operation.metadata.get("runtime_publication_id")
            == pending.publication_id
            and operation.metadata.get("runtime_publication_kind")
            == publication["kind"]
            and operation.metadata.get("runtime_publication_bound") is True
            and operation.metadata.get("runtime_publication_binding_version")
            == _RUNTIME_PUBLICATION_BINDING_VERSION
            and self._runtime_publication_operation_contract_matches(
                operation,
                publication,
            )
            and pending.operation_id == operation_id
            and (
                not isinstance(pending, RuntimeRecoveryRequired)
                or pending.pid == publication["pid"]
            )
            and pending.state == publication["state"]
            and pending.phase == publication["phase"]
        )

    def _owns_grouped_pending_runtime_publication(
        self,
        operation_id: str,
        error: BaseExceptionGroup,
    ) -> bool:
        """Recognize exact pending signals carried by a control-flow group."""

        pending: list[RuntimePublicationPending | RuntimeRecoveryRequired] = []
        stack: list[BaseException] = [error]
        while stack:
            current = stack.pop()
            if isinstance(current, BaseExceptionGroup):
                stack.extend(current.exceptions)
            elif isinstance(
                current,
                (RuntimePublicationPending, RuntimeRecoveryRequired),
            ):
                pending.append(current)
        return bool(pending) and all(
            self._owns_pending_runtime_publication(operation_id, item)
            for item in pending
        )

    def _finish_runtime_publication_mismatch(
        self,
        operation: OperationRecord,
        error: BaseException,
    ) -> None:
        scope = self._recovery_terminalization_scope
        publication_id = self._validated_terminalization_publication_id(
            operation,
            error,
        )
        if scope is None or publication_id is None:
            self._finish_unless_admission_stale(
                (
                    OperationOutcome.UNKNOWN
                    if self._has_unknown_external_effect(operation.operation_id)
                    else OperationOutcome.FAILED
                ),
                operation_id=operation.operation_id,
                metadata={
                    "error_type": type(error).__name__,
                    "runtime_publication_mismatch": True,
                },
            )
            return
        with scope(publication_id):
            self._finish(
                (
                    OperationOutcome.UNKNOWN
                    if self._has_unknown_external_effect(operation.operation_id)
                    else OperationOutcome.FAILED
                ),
                operation_id=operation.operation_id,
                metadata={
                    "error_type": type(error).__name__,
                    "runtime_publication_mismatch": True,
                },
            )

    def _finish_unless_admission_stale(
        self,
        outcome: OperationOutcome | str,
        *,
        operation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> OperationRecord | None:
        stale = self._current_mutation_admission_is_stale
        if stale is not None and stale():
            return None
        try:
            return self._finish(
                outcome,
                operation_id=operation_id,
                metadata=dict(metadata or {}),
            )
        except BaseException:
            if stale is not None and stale():
                return None
            raise

    @staticmethod
    def _checkpoint_fork_exception_completion(
        record: OperationRecord,
        exc: BaseException,
    ) -> tuple[OperationOutcome, dict[str, Any]] | None:
        """Classify only a strictly formed receipt from the fork boundary."""

        if (
            record.kind != OperationKind.RUNTIME
            or record.name != _CHECKPOINT_FORK_OPERATION_NAME
        ):
            return None
        stable_receipt = _canonical_checkpoint_fork_receipt(
            getattr(exc, _CHECKPOINT_FORK_RECEIPT_ATTRIBUTE, None)
        )
        if stable_receipt is None:
            return None
        committed = stable_receipt["main_state_committed"]
        pending = stable_receipt["reconciliation_pending"]
        if pending or committed is None:
            outcome = OperationOutcome.UNKNOWN
        elif committed:
            outcome = OperationOutcome.SUCCEEDED
        else:
            outcome = OperationOutcome.FAILED
        return outcome, {
            "error_type": type(exc).__name__,
            "checkpoint_fork_receipt": stable_receipt,
        }

    def _validated_terminalization_publication_id(
        self,
        operation: OperationRecord,
        error: BaseException,
    ) -> str | None:
        """Resolve one exact durable publication without trusting the signal.

        A control-flow signal is only a lookup hint.  Its complete envelope,
        the publication plan, the unique reverse operation binding, and the
        durable operation ancestry must all agree before the builder-issued
        terminalization scope can be selected.
        """

        signals = self._runtime_publication_signals(error)
        if not signals:
            return None
        selected_publication_id: str | None = None
        for signal in signals:
            binding = self._validated_signal_binding(signal)
            if binding is None:
                return None
            publication_id, bound_operation = binding
            if not self._operations_are_ancestrally_related(
                operation,
                bound_operation,
            ):
                return None
            if (
                selected_publication_id is not None
                and selected_publication_id != publication_id
            ):
                return None
            selected_publication_id = publication_id
        return selected_publication_id

    def _validated_signal_binding(
        self,
        signal: RuntimePublicationPending | RuntimeRecoveryRequired,
    ) -> tuple[str, OperationRecord] | None:
        publication = self.publications.get_runtime_publication(
            signal.publication_id
        )
        if publication is None:
            return None
        publication_id = str(publication.get("publication_id") or "")
        plan = publication.get("plan")
        if not publication_id or not isinstance(plan, dict):
            return None
        bound_operation_id = str(plan.get("operation_id") or "")
        if not bound_operation_id:
            return None
        if self.runtime_publication_binding_operation_ids(publication_id) != [
            bound_operation_id
        ]:
            return None
        bound_operation = self.store.get_operation(bound_operation_id)
        if bound_operation is None or not self._bound_operation_matches_publication(
            bound_operation,
            publication,
        ):
            return None
        if not self._signal_matches_publication(
            signal,
            publication,
            bound_operation_id=bound_operation_id,
        ):
            return None
        return publication_id, bound_operation

    def _bound_operation_matches_publication(
        self,
        operation: OperationRecord,
        publication: dict[str, Any],
    ) -> bool:
        metadata = operation.metadata
        return bool(
            metadata.get("runtime_publication_id")
            == publication.get("publication_id")
            and metadata.get("runtime_publication_kind")
            == publication.get("kind")
            and metadata.get("runtime_publication_bound") is True
            and metadata.get("runtime_publication_binding_version")
            == _RUNTIME_PUBLICATION_BINDING_VERSION
            and self._runtime_publication_operation_contract_matches(
                operation,
                publication,
            )
        )

    @staticmethod
    def _signal_matches_publication(
        signal: RuntimePublicationPending | RuntimeRecoveryRequired,
        publication: dict[str, Any],
        *,
        bound_operation_id: str,
    ) -> bool:
        return bool(
            str(signal.publication_id)
            == str(publication.get("publication_id") or "")
            and str(signal.operation_id) == bound_operation_id
            and str(signal.state) == str(publication.get("state") or "")
            and str(signal.phase) == str(publication.get("phase") or "")
            and (
                not isinstance(signal, RuntimeRecoveryRequired)
                or str(signal.pid) == str(publication.get("pid") or "")
            )
        )

    @staticmethod
    def _runtime_publication_signals(
        error: BaseException,
    ) -> list[RuntimePublicationPending | RuntimeRecoveryRequired]:
        signals: list[RuntimePublicationPending | RuntimeRecoveryRequired] = []
        stack = [error]
        while stack:
            current = stack.pop()
            if isinstance(current, BaseExceptionGroup):
                stack.extend(current.exceptions)
            elif isinstance(
                current,
                (RuntimePublicationPending, RuntimeRecoveryRequired),
            ):
                signals.append(current)
        return signals

    def _operations_are_ancestrally_related(
        self,
        left: OperationRecord,
        right: OperationRecord,
    ) -> bool:
        if left.root_operation_id != right.root_operation_id:
            return False
        return self._operation_descends_from(left, right.operation_id) or (
            self._operation_descends_from(right, left.operation_id)
        )

    def _operation_descends_from(
        self,
        operation: OperationRecord,
        ancestor_operation_id: str,
    ) -> bool:
        current: OperationRecord | None = operation
        seen: set[str] = set()
        while current is not None and current.operation_id not in seen:
            if current.operation_id == ancestor_operation_id:
                return True
            seen.add(current.operation_id)
            current = (
                self.store.get_operation(current.parent_operation_id)
                if current.parent_operation_id is not None
                else None
            )
        return False

    @staticmethod
    def _runtime_publication_operation_contract_matches(
        operation: OperationRecord,
        publication: dict[str, Any],
    ) -> bool:
        plan = publication["plan"]
        publication_pid = str(publication["pid"])
        if (
            operation.kind != OperationKind.RUNTIME
            or str(plan.get("pid") or "") != publication_pid
        ):
            return False
        if publication["kind"] == "process_exec":
            return OperationManager._process_exec_publication_contract_matches(
                operation,
                publication_pid,
            )
        if publication["kind"] == "checkpoint_restore":
            return OperationManager._checkpoint_restore_publication_contract_matches(
                operation,
                plan,
            )
        if publication["kind"] != "process_launch":
            return False
        return OperationManager._process_launch_publication_contract_matches(
            operation,
            plan,
            publication_pid,
        )

    @staticmethod
    def _process_exec_publication_contract_matches(
        operation: OperationRecord,
        publication_pid: str,
    ) -> bool:
        return (
            operation.name == "process.exec"
            and operation.actor == publication_pid
            and operation.pid == publication_pid
        )

    @staticmethod
    def _checkpoint_restore_publication_contract_matches(
        operation: OperationRecord,
        plan: dict[str, Any],
    ) -> bool:
        actor = str(plan.get("actor") or "")
        return bool(
            actor
            and str(plan.get("checkpoint_id") or "")
            and operation.name == "checkpoint.restore"
            and operation.actor == actor
            and operation.pid == actor
        )

    @staticmethod
    def _process_launch_publication_contract_matches(
        operation: OperationRecord,
        plan: dict[str, Any],
        publication_pid: str,
    ) -> bool:
        launch_kind = str(plan.get("launch_kind") or "")
        if launch_kind == "spawn":
            return (
                plan.get("parent_pid") is None
                and operation.name == "process.spawn"
                and operation.actor == "runtime"
                and operation.pid in {None, publication_pid}
            )
        parent_pid = str(plan.get("parent_pid") or "")
        return bool(
            parent_pid
            and launch_kind in {"fork", "spawn_child"}
            and operation.name == f"process.{launch_kind}"
            and operation.actor == parent_pid
            and operation.pid == parent_pid
        )

    @contextmanager
    def scope(
        self,
        *,
        kind: OperationKind | str,
        name: str,
        actor: str,
        pid: str | None,
        expected_roles: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        operation_id: str | None = None,
        parent_operation_id: str | None = None,
        auto_finish: bool = True,
    ) -> Iterator[OperationRecord]:
        record = (
            self.resume(operation_id)
            if operation_id is not None
            else self.start(
                kind=kind,
                name=name,
                actor=actor,
                pid=pid,
                parent_operation_id=parent_operation_id,
                expected_roles=expected_roles,
                metadata=metadata,
            )
        )
        token = self._set_current(record.operation_id)
        try:
            yield record
        except (HumanApprovalRequired, ProcessWaitRequired, ProcessMessageWaitRequired) as exc:
            self._record_wait(record.operation_id, exc)
            raise
        except (RuntimePublicationPending, RuntimeRecoveryRequired) as exc:
            if not self._owns_pending_runtime_publication(record.operation_id, exc):
                self._finish_runtime_publication_mismatch(record, exc)
            raise
        except BaseExceptionGroup as exc:
            if not self._owns_grouped_pending_runtime_publication(
                record.operation_id,
                exc,
            ):
                self._finish_runtime_publication_mismatch(record, exc)
            raise
        except (CapabilityDenied, PolicyDenied, ResourceLimitExceeded) as exc:
            self._finish_unless_admission_stale(
                OperationOutcome.DENIED,
                operation_id=record.operation_id,
                metadata={"error_type": type(exc).__name__},
            )
            raise
        except asyncio.CancelledError as exc:
            completion = self._checkpoint_fork_exception_completion(record, exc)
            if completion is None:
                outcome = OperationOutcome.INTERRUPTED
                completion_metadata = None
            else:
                outcome, completion_metadata = completion
            self._finish_unless_admission_stale(
                outcome,
                operation_id=record.operation_id,
                metadata=completion_metadata,
            )
            raise
        except BaseException as exc:
            completion = self._checkpoint_fork_exception_completion(record, exc)
            if completion is None:
                outcome = (
                    OperationOutcome.UNKNOWN
                    if self._has_unknown_external_effect(record.operation_id)
                    else OperationOutcome.FAILED
                )
                completion_metadata = {"error_type": type(exc).__name__}
            else:
                outcome, completion_metadata = completion
            self._finish_unless_admission_stale(
                outcome,
                operation_id=record.operation_id,
                metadata=completion_metadata,
            )
            raise
        else:
            if auto_finish:
                self._finish_unless_admission_stale(
                    OperationOutcome.SUCCEEDED,
                    operation_id=record.operation_id,
                )
        finally:
            _CURRENT_OPERATION.reset(token)

    def protected(
        self,
        *,
        kind: OperationKind | str,
        name: str,
        actor_arg: str = "pid",
        pid_arg: str = "pid",
        expected_roles: Iterable[str] = (),
        result_pid: bool = False,
    ) -> Callable[[F], F]:
        """Decorator for public boundaries whose exceptions determine outcome."""

        def decorate(function: F) -> F:
            signature = inspect.signature(function)

            def selected(bound: inspect.BoundArguments, key: str) -> str | None:
                value = bound.arguments.get(key)
                return str(value) if value is not None else None

            if inspect.iscoroutinefunction(function):
                @wraps(function)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    bound = signature.bind_partial(*args, **kwargs)
                    actor = selected(bound, actor_arg) or "runtime"
                    pid = selected(bound, pid_arg)
                    with self.scope(
                        kind=kind,
                        name=name,
                        actor=actor,
                        pid=pid,
                        expected_roles=expected_roles,
                    ) as operation:
                        result = await function(*args, **kwargs)
                        if result_pid and isinstance(result, str):
                            self.set_pid(result, operation_id=operation.operation_id)
                        return result

                return async_wrapper  # type: ignore[return-value]

            @wraps(function)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                bound = signature.bind_partial(*args, **kwargs)
                actor = selected(bound, actor_arg) or "runtime"
                pid = selected(bound, pid_arg)
                with self.scope(
                    kind=kind,
                    name=name,
                    actor=actor,
                    pid=pid,
                    expected_roles=expected_roles,
                ) as operation:
                    result = function(*args, **kwargs)
                    if result_pid and isinstance(result, str):
                        self.set_pid(result, operation_id=operation.operation_id)
                    return result

            return sync_wrapper  # type: ignore[return-value]

        return decorate

    def _record_wait(self, operation_id: str, exc: BaseException) -> None:
        metadata: dict[str, Any] = {"wait_type": type(exc).__name__}
        if isinstance(exc, HumanApprovalRequired):
            metadata["request_id"] = exc.request_id
            self.link_evidence(
                "human_request",
                exc.request_id,
                OperationEvidenceRole.WAIT,
                operation_id=operation_id,
            )
        elif isinstance(exc, ProcessWaitRequired):
            metadata["child_pid"] = exc.child_pid
            self.link_evidence(
                "process",
                exc.child_pid,
                OperationEvidenceRole.WAIT,
                operation_id=operation_id,
            )
        elif isinstance(exc, ProcessMessageWaitRequired):
            metadata["recipient_pid"] = exc.recipient_pid
        self.expect(OperationEvidenceRole.WAIT, operation_id=operation_id)
        self.wait(operation_id=operation_id, metadata=metadata)

    def _set_current(self, operation_id: str) -> Token[_CurrentOperation | None]:
        return _CURRENT_OPERATION.set(
            _CurrentOperation(manager_identity=self._identity, operation_id=operation_id)
        )

    def _require(self, operation_id: str) -> OperationRecord:
        record = self.store.get_operation(operation_id)
        if record is None:
            raise ValueError(f"operation not found: {operation_id}")
        return record

    def _has_unknown_external_effect(self, operation_id: str) -> bool:
        self._require(operation_id)
        return self.store.operation_has_unknown_external_effect(operation_id)
