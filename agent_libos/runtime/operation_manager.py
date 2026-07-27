from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Mapping
from copy import deepcopy
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
from agent_libos.utils.public_errors import PublicErrorEnvelope
from agent_libos.utils.serde import dumps


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
        "phase", "code", "error_type", "message", "correlation_id",
        "internal_error", "audit_error_type", "audit_error",
        "audit_error_code", "audit_error_correlation_id",
        "audit_internal_error", "failure_record_error_type",
        "failure_record_error", "failure_record_error_code",
        "failure_record_error_correlation_id",
        "failure_record_internal_error",
    }
)
_CHECKPOINT_FORK_DIAGNOSTIC_KEYS = frozenset(
    {
        "phase", "interruption_error_type", "interruption",
        "interruption_code", "interruption_correlation_id",
        "interruption_internal_error",
        "diagnostic_error_type", "diagnostic_error",
        "diagnostic_code", "diagnostic_correlation_id",
        "diagnostic_internal_error",
        "prepared_runtime_assets_retained", "fork_subtree_quarantined",
        "recovery_signal_record_id", "recovery_signal_error_type",
        "recovery_signal_error", "recovery_signal_error_code",
        "recovery_signal_error_correlation_id",
        "recovery_signal_internal_error", "lifecycle_fence_requested",
        "operation_recovery_signal_recorded",
        "operation_recovery_signal_error_type",
        "operation_recovery_signal_error",
        "operation_recovery_signal_error_code",
        "operation_recovery_signal_error_correlation_id",
        "operation_recovery_signal_internal_error",
        "lifecycle_fence_error_type", "lifecycle_fence_error",
        "lifecycle_fence_error_code",
        "lifecycle_fence_error_correlation_id",
        "lifecycle_fence_internal_error", "lifecycle_fenced",
    }
)
_CHECKPOINT_FORK_MAX_MAP_ITEMS = 4096
_CHECKPOINT_FORK_MAX_FAILURES = 32
_CHECKPOINT_FORK_MAX_TEXT = 1024
_OPERATION_METADATA_MAX_BYTES = 131_072
_OPERATION_METADATA_MAX_DEPTH = 32
_OPERATION_METADATA_MAX_NODES = 4_096


def _validated_operation_id(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
    ):
        raise ValidationError(
            "operation id must be an exact non-empty string"
        )
    return value


def _selected_operation_id(
    operation_id: Any | None,
    current_operation_id: str | None,
) -> str | None:
    if operation_id is None:
        return current_operation_id
    return _validated_operation_id(operation_id)


def _bounded_fork_text(value: Any) -> str | None:
    if (
        type(value) is not str
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


def _canonical_internal_error_observation(
    value: Any,
    *,
    correlation_id: str,
) -> dict[str, Any] | None:
    if type(value) is not dict or set(value) != {
        "error_type",
        "correlation_id",
        "exception_text",
    }:
        return None
    error_type = _bounded_fork_text(value.get("error_type"))
    observed_correlation = _bounded_fork_text(value.get("correlation_id"))
    text = value.get("exception_text")
    if (
        error_type is None
        or observed_correlation != correlation_id
        or type(text) is not dict
        or set(text) != {"bytes", "sha256"}
    ):
        return None
    byte_count = text.get("bytes")
    sha256 = text.get("sha256")
    if (
        type(byte_count) is not int
        or byte_count < 0
        or byte_count > 2**63 - 1
        or type(sha256) is not str
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        return None
    return {
        "error_type": error_type,
        "correlation_id": observed_correlation,
        "exception_text": {"bytes": byte_count, "sha256": sha256},
    }


def _canonical_error_fields(
    value: Mapping[Any, Any],
    *,
    error_type_key: str,
    message_key: str,
    code_key: str,
    correlation_id_key: str,
    internal_error_key: str,
) -> dict[str, Any] | None:
    error_type = _bounded_fork_text(value.get(error_type_key))
    message = _bounded_fork_text(value.get(message_key))
    if error_type is None or message is None:
        return None
    modern_keys = {code_key, correlation_id_key, internal_error_key}
    if not any(key in value for key in modern_keys):
        return {error_type_key: error_type, message_key: message}
    code = _bounded_fork_text(value.get(code_key))
    correlation_id = _bounded_fork_text(value.get(correlation_id_key))
    if code is None or correlation_id is None:
        return None
    envelope = PublicErrorEnvelope.from_mapping(
        {
            "code": code,
            "error_type": error_type,
            "correlation_id": correlation_id,
        }
    )
    if envelope is None or message != envelope.message:
        return None
    selected: dict[str, Any] = {
        error_type_key: error_type,
        message_key: message,
        code_key: code,
        correlation_id_key: correlation_id,
    }
    if internal_error_key in value:
        internal = _canonical_internal_error_observation(
            value.get(internal_error_key),
            correlation_id=correlation_id,
        )
        if internal is None:
            return None
        selected[internal_error_key] = internal
    return selected


def _canonical_prefixed_checkpoint_error(
    value: Mapping[Any, Any],
    *,
    error_type_key: str,
    message_key: str,
    code_key: str,
    correlation_id_key: str,
    internal_error_key: str,
) -> dict[str, Any] | None:
    keys = {
        error_type_key,
        message_key,
        code_key,
        correlation_id_key,
        internal_error_key,
    }
    if not any(key in value for key in keys):
        return {}
    return _canonical_error_fields(
        value,
        error_type_key=error_type_key,
        message_key=message_key,
        code_key=code_key,
        correlation_id_key=correlation_id_key,
        internal_error_key=internal_error_key,
    )


def _canonical_checkpoint_fork_failure(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) - _CHECKPOINT_FORK_FAILURE_KEYS:
        return None
    phase = _bounded_fork_text(value.get("phase"))
    primary = _canonical_error_fields(
        value,
        error_type_key="error_type",
        message_key="message",
        code_key="code",
        correlation_id_key="correlation_id",
        internal_error_key="internal_error",
    )
    if phase is None or primary is None:
        return None
    selected: dict[str, Any] = {"phase": phase, **primary}
    for keys in (
        (
            "audit_error_type",
            "audit_error",
            "audit_error_code",
            "audit_error_correlation_id",
            "audit_internal_error",
        ),
        (
            "failure_record_error_type",
            "failure_record_error",
            "failure_record_error_code",
            "failure_record_error_correlation_id",
            "failure_record_internal_error",
        ),
    ):
        extra = _canonical_prefixed_checkpoint_error(
            value,
            error_type_key=keys[0],
            message_key=keys[1],
            code_key=keys[2],
            correlation_id_key=keys[3],
            internal_error_key=keys[4],
        )
        if extra is None:
            return None
        selected.update(extra)
    return selected


def _checkpoint_fork_failure_count_matches_status(
    status: str,
    failures: list[dict[str, Any]],
) -> bool:
    if status == "forked":
        return not failures
    if status in {"forked_with_warnings", "fork_recovery_required"}:
        return bool(failures)
    return True


def _canonical_checkpoint_fork_failures(
    value: Mapping[Any, Any],
    status: str,
) -> list[dict[str, Any]] | None:
    raw_failures = value.get("post_commit_failures")
    if (
        not isinstance(raw_failures, list)
        or len(raw_failures) > _CHECKPOINT_FORK_MAX_FAILURES
    ):
        return None
    failures: list[dict[str, Any]] = []
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
    phase = _bounded_fork_text(value.get("phase"))
    if phase is None:
        return None
    diagnostic: dict[str, Any] = {"phase": phase}
    for key in (
        "prepared_runtime_assets_retained",
        "fork_subtree_quarantined",
        "lifecycle_fence_requested",
        "operation_recovery_signal_recorded",
        "lifecycle_fenced",
    ):
        if key not in value:
            continue
        if type(value[key]) is not bool:
            return None
        diagnostic[key] = value[key]
    if "recovery_signal_record_id" in value:
        record_id = _bounded_fork_text(value["recovery_signal_record_id"])
        if record_id is None:
            return None
        diagnostic["recovery_signal_record_id"] = record_id
    error_bundles = (
        (
            "interruption_error_type", "interruption", "interruption_code",
            "interruption_correlation_id", "interruption_internal_error",
        ),
        (
            "diagnostic_error_type", "diagnostic_error", "diagnostic_code",
            "diagnostic_correlation_id", "diagnostic_internal_error",
        ),
        (
            "recovery_signal_error_type", "recovery_signal_error",
            "recovery_signal_error_code", "recovery_signal_error_correlation_id",
            "recovery_signal_internal_error",
        ),
        (
            "operation_recovery_signal_error_type",
            "operation_recovery_signal_error",
            "operation_recovery_signal_error_code",
            "operation_recovery_signal_error_correlation_id",
            "operation_recovery_signal_internal_error",
        ),
        (
            "lifecycle_fence_error_type", "lifecycle_fence_error",
            "lifecycle_fence_error_code", "lifecycle_fence_error_correlation_id",
            "lifecycle_fence_internal_error",
        ),
    )
    for keys in error_bundles:
        error_fields = _canonical_prefixed_checkpoint_error(
            value,
            error_type_key=keys[0],
            message_key=keys[1],
            code_key=keys[2],
            correlation_id_key=keys[3],
            internal_error_key=keys[4],
        )
        if error_fields is None:
            return None
        diagnostic.update(error_fields)
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


def _validate_metadata_string(value: str) -> None:
    if len(value) > _OPERATION_METADATA_MAX_BYTES:
        raise ValidationError(
            f"operation metadata exceeds {_OPERATION_METADATA_MAX_BYTES} bytes"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValidationError(
            "operation metadata strings must be valid UTF-8"
        ) from exc


def _claim_metadata_container(
    value: dict[str, Any] | list[Any],
    depth: int,
    container_ids: set[int],
) -> None:
    if depth > _OPERATION_METADATA_MAX_DEPTH:
        raise ValidationError(
            "operation metadata exceeds maximum JSON depth="
            f"{_OPERATION_METADATA_MAX_DEPTH}"
        )
    identity = id(value)
    if identity in container_ids:
        raise ValidationError(
            "operation metadata must not contain container cycles or aliases"
        )
    container_ids.add(identity)


def _metadata_container_children(
    value: Any,
    depth: int,
    container_ids: set[int],
) -> list[tuple[Any, int]] | None:
    value_type = type(value)
    if value_type is dict:
        _claim_metadata_container(value, depth, container_ids)
        if len(value) > _OPERATION_METADATA_MAX_NODES:
            raise ValidationError(
                "operation metadata exceeds maximum JSON nodes="
                f"{_OPERATION_METADATA_MAX_NODES}"
            )
        children: list[tuple[Any, int]] = []
        for key, child in value.items():
            if type(key) is not str:
                raise ValidationError(
                    "operation metadata keys must be exact strings"
                )
            _validate_metadata_string(key)
            children.append((child, depth + 1))
        return children
    if value_type is list:
        _claim_metadata_container(value, depth, container_ids)
        if len(value) > _OPERATION_METADATA_MAX_NODES:
            raise ValidationError(
                "operation metadata exceeds maximum JSON nodes="
                f"{_OPERATION_METADATA_MAX_NODES}"
            )
        return [(child, depth + 1) for child in value]
    return None


def _validate_metadata_leaf(value: Any) -> None:
    value_type = type(value)
    if value_type is str:
        _validate_metadata_string(value)
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValidationError("operation metadata numbers must be finite")
        return
    if value_type in {int, bool, type(None)}:
        return
    raise ValidationError("operation metadata values must use exact JSON types")


def _validate_metadata_graph(metadata: dict[str, Any]) -> None:
    pending: list[tuple[Any, int]] = [(metadata, 1)]
    container_ids: set[int] = set()
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > _OPERATION_METADATA_MAX_NODES:
            raise ValidationError(
                "operation metadata exceeds maximum JSON nodes="
                f"{_OPERATION_METADATA_MAX_NODES}"
            )
        children = _metadata_container_children(current, depth, container_ids)
        if children is None:
            _validate_metadata_leaf(current)
        else:
            pending.extend(children)


def _bounded_metadata_copy(metadata: dict[str, Any]) -> dict[str, Any]:
    selected = deepcopy(metadata)
    try:
        encoded_size = len(dumps(selected).encode("utf-8"))
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError("operation metadata is not bounded JSON") from exc
    if encoded_size > _OPERATION_METADATA_MAX_BYTES:
        raise ValidationError(
            f"operation metadata exceeds {_OPERATION_METADATA_MAX_BYTES} bytes"
        )
    return selected


def _validated_metadata_mapping(
    metadata: Any | None,
    *,
    reserve_runtime_publication_keys: bool,
) -> dict[str, Any]:
    if metadata is None:
        return {}
    if type(metadata) is not dict:
        raise ValidationError("operation metadata must be an exact JSON object")
    if len(metadata) > _OPERATION_METADATA_MAX_NODES:
        raise ValidationError(
            "operation metadata exceeds maximum JSON nodes="
            f"{_OPERATION_METADATA_MAX_NODES}"
        )
    if reserve_runtime_publication_keys and any(
        type(key) is str
        and key.startswith(_RUNTIME_PUBLICATION_METADATA_PREFIX)
        for key in metadata
    ):
        raise ValidationError(
            "runtime publication metadata is reserved for durable binding"
        )
    _validate_metadata_graph(metadata)
    return _bounded_metadata_copy(metadata)


def _validated_public_operation_metadata(
    metadata: Any | None,
) -> dict[str, Any]:
    return _validated_metadata_mapping(
        metadata,
        reserve_runtime_publication_keys=True,
    )


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
        selected_metadata = _validated_public_operation_metadata(metadata)
        parent_id = _selected_operation_id(
            parent_operation_id,
            self.current_id(),
        )
        parent = self.store.get_operation(parent_id) if parent_id is not None else None
        if parent_id is not None and parent is None:
            raise ValueError(f"parent operation not found: {parent_id}")
        operation_id = new_id("op")
        now = utc_now()
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
        selected_id = _validated_operation_id(operation_id)
        with self.store.locked():
            record = self._require(selected_id)
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
                return self._require(selected_id)
            return updated

    def expect(self, *roles: OperationEvidenceRole | str, operation_id: str | None = None) -> OperationRecord | None:
        selected_id = _selected_operation_id(operation_id, self.current_id())
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
        selected_id = _selected_operation_id(operation_id, self.current_id())
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
        selected_id = _selected_operation_id(operation_id, self.current_id())
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
        selected_id = _selected_operation_id(operation_id, self.current_id())
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

        selected_operation_id = _validated_operation_id(operation_id)
        with self.store.locked():
            self._require_runtime_publication_plan(
                selected_operation_id,
                publication_id=publication_id,
                publication_kind=publication_kind,
                required_state="planning",
            )

            record = self._require_runtime_publication_operation_identity(
                selected_operation_id,
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
            if binding_operation_ids not in ([], [selected_operation_id]):
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
                    f"{selected_operation_id} -> {existing_publication_id}"
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
                latest = self._require(selected_operation_id)
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
                    f"{selected_operation_id}"
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

        selected_operation_id = _validated_operation_id(operation_id)
        selected_outcome = OperationOutcome(outcome)
        if selected_outcome == OperationOutcome.PENDING:
            raise ValueError("runtime publication cannot reconcile to pending")
        with self.store.locked():
            self._require_runtime_publication_plan(
                selected_operation_id,
                publication_id=publication_id,
                publication_kind=publication_kind,
                required_state=publication_state,
                required_phase=publication_phase,
            )
            record = self._require_runtime_publication_operation_identity(
                selected_operation_id,
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
                    selected_operation_id,
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
                latest = self._require(selected_operation_id)
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
                        selected_operation_id,
                        publication_id=publication_id,
                        publication_kind=publication_kind,
                        publication_state=publication_state,
                        publication_phase=publication_phase,
                        reconciled_marker=_publication_reconciled_marker,
                    )
                    return latest
                raise RuntimeError(
                    "operation changed during runtime publication reconciliation: "
                    f"{selected_operation_id}"
                )
            self._mark_runtime_publication_operation_reconciled(
                selected_operation_id,
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

        return self.store.get_operation(_validated_operation_id(operation_id))

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
        selected_operation_id = _validated_operation_id(operation_id)
        record = self.store.get_operation(selected_operation_id)
        if record is None:
            raise ValidationError(
                "runtime publication references a missing operation: "
                f"{publication_id} -> {selected_operation_id}"
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
                f"{publication_id} -> {selected_operation_id} "
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
        selected_id = _selected_operation_id(operation_id, self.current_id())
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
        selected_metadata = _validated_metadata_mapping(
            metadata,
            reserve_runtime_publication_keys=False,
        )
        selected_id = _selected_operation_id(operation_id, self.current_id())
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
            metadata=selected_metadata,
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
        """Recognize a group made exclusively from one exact pending signal."""

        pending: list[RuntimePublicationPending | RuntimeRecoveryRequired] = []
        signal_identity: tuple[Any, ...] | None = None
        has_unrelated_leaf = False
        stack: list[BaseException] = [error]
        while stack:
            current = stack.pop()
            if isinstance(current, BaseExceptionGroup):
                stack.extend(current.exceptions)
            elif isinstance(
                current,
                (RuntimePublicationPending, RuntimeRecoveryRequired),
            ):
                current_identity = (
                    type(current),
                    current.publication_id,
                    current.operation_id,
                    current.state,
                    current.phase,
                    (
                        current.pid
                        if isinstance(current, RuntimeRecoveryRequired)
                        else None
                    ),
                )
                if signal_identity is None:
                    signal_identity = current_identity
                elif current_identity != signal_identity:
                    return False
                pending.append(current)
            else:
                has_unrelated_leaf = True
        owned = bool(pending) and all(
            self._owns_pending_runtime_publication(operation_id, item)
            for item in pending
        )
        if not owned or not has_unrelated_leaf:
            return owned
        # Merely claiming ownership (including through a faulty adapter) must
        # not let a pending-looking control signal mask another group leaf.
        # An exact durable publication is different: it authoritatively owns
        # the operation outcome, and a failed terminal transaction must leave
        # that operation nonterminal for recovery even while every original
        # exception is re-raised to the caller.
        for item in pending:
            binding = self._validated_signal_binding(item)
            if binding is None or binding[1].operation_id != operation_id:
                return False
        return True

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
        selected_operation_id = _validated_operation_id(operation_id)
        return _CURRENT_OPERATION.set(
            _CurrentOperation(
                manager_identity=self._identity,
                operation_id=selected_operation_id,
            )
        )

    def _require(self, operation_id: str) -> OperationRecord:
        selected_operation_id = _validated_operation_id(operation_id)
        record = self.store.get_operation(selected_operation_id)
        if record is None:
            raise ValueError(f"operation not found: {selected_operation_id}")
        return record

    def _has_unknown_external_effect(self, operation_id: str) -> bool:
        self._require(operation_id)
        return self.store.operation_has_unknown_external_effect(operation_id)
