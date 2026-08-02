from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
import inspect
from typing import TYPE_CHECKING, Any, ClassVar, cast

from agent_libos.capability.mutation import EXEC_ROLLBACK_TOKEN_KEY
from agent_libos.evidence.payload_retention import (
    PayloadRetentionCursor,
    PayloadRetentionPage,
    PayloadRetentionStore,
    PayloadRetentionTier,
    TaskRunExternalEffectRedactionStore,
)
from agent_libos.evidence.initial_goal_recovery import (
    redact_initial_goal_recovery_receipt_projection,
)
from agent_libos.models import (
    AgentObject,
    AgentImage,
    AgentProcess,
    Capability,
    CapabilityStatus,
    CapabilityUseReservationRecoverySummary,
    Checkpoint,
    CheckpointPayloadDeliveryAttempt,
    CheckpointPayloadDeliveryAttemptPage,
    CheckpointPayloadDeliveryAttemptState,
    Event,
    ExternalEffectRecord,
    HumanRequest,
    HumanRequestStatus,
    JITRehydrationArtifact,
    LLMCallRecord,
    ObjectHandle,
    ObjectMetadata,
    ObjectNamespace,
    ObjectNamespaceCursor,
    ObjectNamespacePage,
    ObjectLifecycleState,
    ObjectOwnerKind,
    ObjectPayloadRecoverySummary,
    ObjectRefCursor,
    ObjectRefPage,
    PersistedObjectState,
    ObjectTask,
    ObjectTaskRecoveryCursor,
    ObjectTaskRecoveryKind,
    ObjectTaskRecoveryPage,
    ObjectTaskStatus,
    OperationCursor,
    OperationEvidenceLink,
    OperationPage,
    OperationRecord,
    ProcessExecutionToken,
    ProcessCursor,
    ProcessPage,
    ProcessRestoreEpoch,
    ProcessToolBindingCursor,
    ProcessToolBindingPage,
    ProcessOutcome,
    ProcessMessageStatus,
    ProcessStatus,
    ProcessWaitState,
    StaleExecutionRecoverySummary,
    ResourceReservation,
    ResourceUsage,
    ResourceUsageReservation,
    ResourceUsageReservationCursor,
    ResourceUsageReservationPage,
    ResourceUsageReservationStatus,
    RuntimeModule,
    RuntimePublicationCursor,
    RuntimePublicationKind,
    RuntimePublicationPage,
    PayloadDeliveryState,
    RuntimePublicationRecord,
    RuntimePublicationState,
    ToolCandidate,
    ToolSpec,
    parse_runtime_publication_kind,
    validate_runtime_publication_record,
)
from agent_libos.models.exceptions import NotFound, ProcessRevisionConflict, ValidationError
from agent_libos.storage.contracts import (
    AuthorityRecoveryBackendProtocol,
    CheckpointPublicationWriterBackendProtocol,
    OperationEvidenceBackendProtocol,
    ObjectQueryBackendProtocol,
    ObjectRecoveryBackendProtocol,
    ProcessBackendProtocol,
    ProcessScaffoldCleanup,
    ResourceBackendProtocol,
    RuntimeModuleBackendProtocol,
    RuntimePublicationBackendProtocol,
    SnapshotCheckpointBackendProtocol,
    ToolArtifactRepositoryProtocol,
    TaskRunBackendProtocol,
    TransactionBackendProtocol,
    UnitOfWorkBackendProtocol,
)
from agent_libos.storage.base import StoreAssemblyReadiness
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, loads

if TYPE_CHECKING:
    from agent_libos.models.snapshot import ProcessSnapshot, SnapshotRows


# Keep doubled OR predicates plus fixed parameters below SQLite's historical
# 999-variable default while remaining comfortably inside PostgreSQL limits.
_SNAPSHOT_SQL_VALUE_BATCH_SIZE = 400


def _exact_backend_cas_result(value: Any, *, operation: str) -> bool:
    """Reject truthy non-booleans at typed backend mutation boundaries."""

    if type(value) is not bool:
        raise ValidationError(
            f"{operation} backend returned a non-boolean CAS result"
        )
    return value


def _exact_positive_capability_use_count(value: Any, *, operation: str) -> int:
    """Keep booleans and coercible scalar values out of authority mutations."""

    if type(value) is not int or value < 1:
        raise ValidationError(
            f"capability {operation} count must be a positive integer"
        )
    return value


def _exact_backend_capability_result(
    value: Any,
    *,
    operation: str,
) -> Capability | None:
    """Reject malformed custom-backend capability mutation projections."""

    if value is not None and not isinstance(value, Capability):
        raise ValidationError(
            f"{operation} backend returned a non-capability result"
        )
    return value


def _backend_mutation_scope(
    backend: object,
    transaction: Callable[[], AbstractContextManager[Any]],
) -> AbstractContextManager[Any]:
    """Open a transaction unless the caller already owns the backend frame."""

    probe = getattr(backend, "probe_runtime_assembly_readiness", None)
    readiness = probe() if callable(probe) else None
    if readiness is StoreAssemblyReadiness.ACTIVE_TRANSACTION:
        return nullcontext()
    return transaction()


def _transactional_backend_cas_result(
    backend: object,
    transaction: Callable[[], AbstractContextManager[Any]],
    mutation: Callable[[], Any],
    *,
    operation: str,
) -> bool:
    """Validate CAS results transactionally without hiding lost-ACK writes.

    A non-boolean return is a backend contract violation and rolls the outer
    transaction back. Backend exceptions retain their existing ambiguity
    semantics: the backend owns rollback versus commit-before-error, so the
    facade lets the outer frame close normally before re-raising the original
    failure for the runtime's reconciliation path.
    """

    failure: BaseException | None = None
    with _backend_mutation_scope(backend, transaction):
        try:
            result = mutation()
        except BaseException as exc:
            failure = exc
        else:
            return _exact_backend_cas_result(result, operation=operation)
    if failure is None:  # pragma: no cover - context-manager contract defense
        raise RuntimeError(f"{operation} backend mutation produced no result")
    raise failure.with_traceback(failure.__traceback__)


def _transactional_backend_cas_result_rollback_on_error(
    backend: object,
    transaction: Callable[[], AbstractContextManager[Any]],
    mutation: Callable[[], Any],
    *,
    operation: str,
) -> bool:
    """Rollback backend body failures while retaining commit-exit ambiguity.

    Use this for mutations whose backend performs its own in-transaction
    readback: an exception from the body proves that the candidate write is
    invalid and must not be committed. A failure raised by the transaction
    context while committing still propagates with its original ambiguity.
    """

    with _backend_mutation_scope(backend, transaction):
        return _exact_backend_cas_result(
            mutation(),
            operation=operation,
        )


def _snapshot_value_batches(values: Iterable[str]) -> Iterator[tuple[str, ...]]:
    selected = tuple(dict.fromkeys(str(value) for value in values))
    for offset in range(0, len(selected), _SNAPSHOT_SQL_VALUE_BATCH_SIZE):
        yield selected[offset : offset + _SNAPSHOT_SQL_VALUE_BATCH_SIZE]


_CHECKPOINT_OBJECT_IDENTITY_FIELDS = (
    "oid",
    "type",
    "schema_version",
    "immutable",
    "created_by",
    "created_at",
)
_CHECKPOINT_FORK_PENDING_STATUS_MESSAGE = "checkpoint_fork_pending_payload"
_TERMINAL_PROCESS_STATUSES = frozenset(
    {
        ProcessStatus.EXITED.value,
        ProcessStatus.FAILED.value,
        ProcessStatus.KILLED.value,
    }
)


def _checkpoint_fork_quarantine_process_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    item = deepcopy(dict(row))
    if str(item.get("status")) in _TERMINAL_PROCESS_STATUSES:
        return item
    item.update(
        {
            "status": ProcessStatus.CREATED.value,
            "status_message": _CHECKPOINT_FORK_PENDING_STATUS_MESSAGE,
            "wait_state_json": dumps(None),
            "outcome_json": dumps(None),
        }
    )
    return item


def _checkpoint_object_payload_was_superseded(
    current: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    current_marker: Any,
    present_marker: Mapping[str, Any],
) -> bool:
    """Recognize a later Object lifecycle without reviving its old payload."""

    if any(
        current.get(field) != expected.get(field)
        for field in _CHECKPOINT_OBJECT_IDENTITY_FIELDS
    ):
        return False
    current_version = current.get("version")
    expected_version = expected.get("version")
    if (
        isinstance(current_version, bool)
        or not isinstance(current_version, int)
        or isinstance(expected_version, bool)
        or not isinstance(expected_version, int)
        or current_version < expected_version
    ):
        return False
    lifecycle_state = current.get("lifecycle_state")
    if lifecycle_state == ObjectLifecycleState.LIVE.value:
        return current_version > expected_version and current_marker == present_marker
    if lifecycle_state != ObjectLifecycleState.RELEASED.value:
        return False
    if current_version == expected_version:
        release_fields = {"lifecycle_state", "deleted_at", "updated_at"}
        if any(
            current.get(field) != expected.get(field)
            for field in set(current) | set(expected)
            if field not in release_fields
        ):
            return False
    return current_marker in (
        {"storage": "runtime_memory", "present": False},
        {
            "storage": "runtime_memory",
            "present": False,
            "recovered_after_reopen": True,
        },
    )


class _RepositoryFacade:
    """Narrow domain adapter over a shared transactional SQL engine."""

    _METHODS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, store: TransactionBackendProtocol) -> None:
        self.__store = store

    def __getattr__(self, name: str) -> Any:
        if name not in self._METHODS:
            raise AttributeError(f"{type(self).__name__!s} has no repository operation {name!r}")
        return getattr(self.__store, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | self._METHODS)

    def _delegate(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        return getattr(self.__store, operation)(*args, **kwargs)

    def locked(self) -> AbstractContextManager[None]:
        return self.__store.locked()

    def transaction(self, *, include_object_payloads: bool = False) -> AbstractContextManager[Any]:
        return self.__store.transaction(include_object_payloads=include_object_payloads)


class TaskRunRepository(_RepositoryFacade):
    """Narrow facade for schema-v4 durable TaskRun persistence."""

    _METHODS = frozenset(
        {
            "runtime_epoch",
            "claim_runtime_epoch",
            "insert_task_run",
            "get_task_run",
            "list_task_runs",
            "list_recoverable_task_runs",
            "update_task_run_cas",
            "claim_task_run_epoch",
            "claim_terminal_task_run_epoch",
            "insert_task_run_requirement",
            "list_task_run_requirements",
            "update_task_run_requirement_cas",
            "insert_task_run_payload",
            "get_task_run_payload",
            "list_task_run_payloads",
            "purge_task_run_payloads",
            "upsert_task_run_resume_point",
            "get_task_run_resume_point",
            "list_task_run_resume_points",
            "delete_task_run_resume_point",
            "insert_task_run_command",
            "get_task_run_command",
            "get_task_run_command_by_client_request_id",
            "update_task_run_command_result",
            "append_task_run_ledger_item",
            "get_task_run_ledger_item",
            "list_task_run_ledger",
            "insert_task_run_link",
            "list_task_run_links",
            "list_processes_for_task_run",
            "list_task_run_process_ids",
            "list_active_capability_use_reservations_for_pids",
            "list_human_requests_for_pids",
            "list_task_run_human_requests",
            "redact_task_run_human_request_payload",
            "purge_task_run_messages",
            "purge_task_run_llm_pending_actions",
            "list_task_run_external_effects",
            "redact_task_run_external_effect_payload",
            "list_active_task_run_ids_for_pids",
            "list_task_run_llm_calls",
            "redact_task_run_llm_call_payload",
            "list_task_run_llm_tool_outputs",
            "redact_task_run_llm_tool_output",
        }
    )

    def __init__(self, backend: TaskRunBackendProtocol) -> None:
        super().__init__(backend)


class ProcessRepository(_RepositoryFacade):
    """Process lifecycle, messaging, human, and LLM persistence."""

    _METHODS = frozenset(
        {
            "insert_human_request",
            "update_human_request",
            "list_human_requests",
            "list_human_requests_for_pids",
            "insert_llm_call",
            "list_llm_calls",
            "get_llm_call",
            "get_latest_llm_call",
            "get_latest_successful_llm_call",
            "upsert_llm_tool_output",
            "list_llm_tool_outputs",
            "get_llm_context_generation",
            "set_llm_context_generation",
            "get_llm_context_label_history",
            "merge_llm_context_label_history",
            "upsert_llm_pending_action",
            "get_llm_pending_action",
            "list_llm_pending_actions",
            "list_llm_pending_action_validation_rows",
            "claim_llm_pending_action",
            "complete_llm_pending_action",
            "insert_process_message",
            "update_process_message",
            "update_process_message_metadata",
            "get_process_message",
            "list_process_messages",
            "count_process_messages",
            "get_process_activity_summaries",
            "insert_object_task",
            "update_object_task",
            "mark_object_tasks_abandoned",
            "upsert_agent_rating",
            "get_agent_rating",
            "get_agent_ratings_for_processes",
            "list_agent_ratings",
        }
    )

    def __init__(self, backend: ProcessBackendProtocol) -> None:
        super().__init__(backend)
        self._process_backend = backend

    def insert_process(self, process: AgentProcess) -> None:
        self._process_backend.insert_process(process)

    def get_process(self, pid: str) -> AgentProcess | None:
        return self._process_backend.get_process(pid)

    def list_processes(
        self,
        limit: int | None = None,
        *,
        active_first: bool = False,
    ) -> list[AgentProcess]:
        return cast(
            list[AgentProcess],
            self._process_backend.list_processes(limit, active_first=active_first),
        )

    def query_processes(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
    ) -> ProcessPage:
        return self._process_backend.query_processes(after=after, limit=limit)

    def query_process_tool_bindings(
        self,
        *,
        after: ProcessToolBindingCursor | None,
        limit: int,
    ) -> ProcessToolBindingPage:
        return self._process_backend.query_process_tool_bindings(
            after=after,
            limit=limit,
        )

    def get_processes_with_ancestors(self, pids: Iterable[str]) -> list[AgentProcess]:
        return cast(
            list[AgentProcess],
            self._process_backend.get_processes_with_ancestors(pids),
        )

    def list_processes_by_status(
        self,
        status: ProcessStatus | str,
    ) -> list[AgentProcess]:
        return cast(
            list[AgentProcess],
            self._process_backend.list_processes_by_status(status),
        )

    def query_orphaned_created_processes(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
    ) -> ProcessPage:
        return cast(
            ProcessPage,
            self._process_backend.query_orphaned_created_processes(
                after=after,
                limit=limit,
            ),
        )

    def list_child_processes(self, parent_pid: str) -> list[AgentProcess]:
        return cast(
            list[AgentProcess],
            self._process_backend.list_child_processes(parent_pid),
        )

    def get_human_request(self, request_id: str) -> HumanRequest | None:
        return self._process_backend.get_human_request(request_id)

    def ack_process_message(
        self,
        message_id: str,
        *,
        recipient_pid: str,
        acked_at: str,
        updated_at: str,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._process_backend,
            self.transaction,
            lambda: self._process_backend.ack_process_message(
                message_id,
                recipient_pid=recipient_pid,
                acked_at=acked_at,
                updated_at=updated_at,
            ),
            operation="ack process message",
        )

    def get_object_task(self, task_id: str) -> ObjectTask | None:
        return self._process_backend.get_object_task(task_id)

    def list_object_tasks(
        self,
        *,
        owner_oid: str | None = None,
        creator_pid: str | None = None,
        statuses: Iterable[str | ObjectTaskStatus] | None = None,
        include_terminal: bool = True,
        limit: int | None = None,
    ) -> list[ObjectTask]:
        return self._process_backend.list_object_tasks(
            owner_oid=owner_oid,
            creator_pid=creator_pid,
            statuses=statuses,
            include_terminal=include_terminal,
            limit=limit,
        )

    def query_object_task_recovery(
        self,
        *,
        kind: ObjectTaskRecoveryKind,
        after: ObjectTaskRecoveryCursor | None,
        limit: int,
    ) -> ObjectTaskRecoveryPage:
        return self._process_backend.query_object_task_recovery(
            kind=kind,
            after=after,
            limit=limit,
        )

    def abandon_object_task_after_reopen(
        self,
        task_id: str,
        *,
        expected_status: ObjectTaskStatus,
        reason: str,
        updated_at: str,
    ) -> ObjectTask | None:
        return self._process_backend.abandon_object_task_after_reopen(
            task_id,
            expected_status=expected_status,
            reason=reason,
            updated_at=updated_at,
        )

    def mark_object_task_result_unavailable_after_reopen(
        self,
        task_id: str,
        *,
        expected_result_oid: str,
        wait: Mapping[str, Any],
        error: str,
        updated_at: str,
    ) -> ObjectTask | None:
        return self._process_backend.mark_object_task_result_unavailable_after_reopen(
            task_id,
            expected_result_oid=expected_result_oid,
            wait=wait,
            error=error,
            updated_at=updated_at,
        )

    def patch_process(
        self,
        pid: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
        expected_status: ProcessStatus | str | None = None,
    ) -> AgentProcess:
        return cast(
            AgentProcess,
            self._process_backend.patch_process(
                pid,
                patch,
                expected_revision=expected_revision,
                expected_status=expected_status,
            ),
        )

    def patch_process_control(
        self,
        pid: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
        allowed_statuses: Iterable[ProcessStatus | str],
        reason: str,
    ) -> AgentProcess:
        return cast(
            AgentProcess,
            self._process_backend.patch_process_control(
                pid,
                patch,
                expected_revision=expected_revision,
                allowed_statuses=allowed_statuses,
                reason=reason,
            ),
        )

    def apply_process_state_transition(
        self,
        pid: str,
        status: ProcessStatus | str,
        *,
        expected_revision: int,
        expected_status: ProcessStatus | str | None = None,
        expected_state_generation: int | None = None,
        wait_state: ProcessWaitState | None = None,
        outcome: ProcessOutcome | None = None,
        status_message: str | None = None,
        control: bool = False,
        allowed_statuses: Iterable[ProcessStatus | str] | None = None,
        reason: str | None = None,
    ) -> AgentProcess:
        return cast(
            AgentProcess,
            self._process_backend.apply_process_state_transition(
                pid,
                status,
                expected_revision=expected_revision,
                expected_status=expected_status,
                expected_state_generation=expected_state_generation,
                wait_state=wait_state,
                outcome=outcome,
                status_message=status_message,
                control=control,
                allowed_statuses=allowed_statuses,
                reason=reason,
            ),
        )

    def append_process_memory_roots(
        self,
        pid: str,
        roots: Iterable[ObjectHandle],
    ) -> AgentProcess:
        return cast(
            AgentProcess,
            self._process_backend.append_process_memory_roots(pid, roots),
        )

    def remove_process_memory_roots(
        self,
        pid: str,
        oids: Iterable[str],
    ) -> AgentProcess:
        return cast(
            AgentProcess,
            self._process_backend.remove_process_memory_roots(pid, oids),
        )

    def append_process_capability_ids(
        self,
        pid: str,
        capability_ids: Iterable[str],
    ) -> AgentProcess:
        return cast(
            AgentProcess,
            self._process_backend.append_process_capability_ids(pid, capability_ids),
        )

    def patch_process_tool_tables(
        self,
        pid: str,
        *,
        tool_table: Mapping[str, str] | None = None,
        model_tool_table: Mapping[str, str] | None = None,
    ) -> AgentProcess:
        return cast(
            AgentProcess,
            self._process_backend.patch_process_tool_tables(
                pid,
                tool_table=tool_table,
                model_tool_table=model_tool_table,
            ),
        )

    def remove_process_tool_bindings(
        self,
        pid: str,
        bindings: Mapping[str, str],
    ) -> AgentProcess:
        return cast(
            AgentProcess,
            self._process_backend.remove_process_tool_bindings(pid, bindings),
        )

    def replace_process_for_restore(self, process: AgentProcess) -> None:
        self._process_backend.replace_process_for_restore(process)

    def reserve_process_restore_epochs(
        self,
        floors: Iterable[ProcessRestoreEpoch],
    ) -> tuple[ProcessRestoreEpoch, ...]:
        return self._process_backend.reserve_process_restore_epochs(floors)

    def commit_process_exec_epoch(
        self,
        pid: str,
        *,
        publication_id: str,
        expected_revision: int,
    ) -> AgentProcess:
        return cast(
            AgentProcess,
            self._process_backend.commit_process_exec_epoch(
                pid,
                publication_id=publication_id,
                expected_revision=expected_revision,
            ),
        )

    def claim_execution(
        self,
        pid: str,
        *,
        owner_id: str,
    ) -> ProcessExecutionToken | None:
        return cast(
            ProcessExecutionToken | None,
            self._process_backend.claim_execution(pid, owner_id=owner_id),
        )

    def claim_host_process_exec(
        self,
        pid: str,
        *,
        owner_id: str,
        expected_revision: int,
        expected_state_generation: int,
        expected_execution_generation: int,
    ) -> ProcessExecutionToken | None:
        return cast(
            ProcessExecutionToken | None,
            self._process_backend.claim_host_process_exec(
                pid,
                owner_id=owner_id,
                expected_revision=expected_revision,
                expected_state_generation=expected_state_generation,
                expected_execution_generation=expected_execution_generation,
            ),
        )

    def claim_worker_process_exec(
        self,
        pid: str,
        *,
        execution_token: ProcessExecutionToken,
        owner_id: str,
        expected_revision: int,
        expected_state_generation: int,
    ) -> ProcessExecutionToken | None:
        return cast(
            ProcessExecutionToken | None,
            self._process_backend.claim_worker_process_exec(
                pid,
                execution_token=execution_token,
                owner_id=owner_id,
                expected_revision=expected_revision,
                expected_state_generation=expected_state_generation,
            ),
        )

    def complete_execution(
        self,
        token: ProcessExecutionToken,
        *,
        status: ProcessStatus | str = ProcessStatus.RUNNABLE,
        status_message: str | None = None,
        wait_state: ProcessWaitState | None = None,
        outcome: ProcessOutcome | None = None,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._process_backend,
            self.transaction,
            lambda: self._process_backend.complete_execution(
                token,
                status=status,
                status_message=status_message,
                wait_state=wait_state,
                outcome=outcome,
            ),
            operation="complete execution",
        )

    def release_execution(self, token: ProcessExecutionToken) -> bool:
        return _transactional_backend_cas_result(
            self._process_backend,
            self.transaction,
            lambda: self._process_backend.release_execution(token),
            operation="release execution",
        )

    def get_process_terminal_cleanup(self, pid: str) -> dict[str, Any] | None:
        cleanup = self._process_backend.get_process_terminal_cleanup(pid)
        return dict(cleanup) if cleanup is not None else None

    def list_process_terminal_cleanups(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
        include_completed: bool = False,
    ) -> list[dict[str, Any]]:
        return [
            dict(cleanup)
            for cleanup in self._process_backend.list_process_terminal_cleanups(
                after=after,
                limit=limit,
                include_completed=include_completed,
            )
        ]

    def claim_process_terminal_cleanup(
        self,
        pid: str,
        *,
        owner_id: str,
        lease_id: str,
        expected_lease_id: str | None,
    ) -> dict[str, Any] | None:
        cleanup = self._process_backend.claim_process_terminal_cleanup(
            pid,
            owner_id=owner_id,
            lease_id=lease_id,
            expected_lease_id=expected_lease_id,
        )
        return dict(cleanup) if cleanup is not None else None

    def complete_process_terminal_cleanup_phase(
        self,
        pid: str,
        *,
        lease_id: str,
        phase: str,
    ) -> dict[str, Any] | None:
        cleanup = self._process_backend.complete_process_terminal_cleanup_phase(
            pid,
            lease_id=lease_id,
            phase=phase,
        )
        return dict(cleanup) if cleanup is not None else None

    def fail_process_terminal_cleanup(
        self,
        pid: str,
        *,
        lease_id: str,
        phase: str,
        error: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        cleanup = self._process_backend.fail_process_terminal_cleanup(
            pid,
            lease_id=lease_id,
            phase=phase,
            error=error,
        )
        return dict(cleanup) if cleanup is not None else None

    def finish_process_terminal_cleanup(
        self,
        pid: str,
        *,
        lease_id: str,
    ) -> dict[str, Any] | None:
        cleanup = self._process_backend.finish_process_terminal_cleanup(
            pid,
            lease_id=lease_id,
        )
        return dict(cleanup) if cleanup is not None else None

    def recover_stale_executions(
        self,
        *,
        owner_id: str,
        require_recovery_lease: Callable[[], None],
        on_recovered: Callable[[str], None],
    ) -> StaleExecutionRecoverySummary:
        require_recovery_lease()
        return cast(
            StaleExecutionRecoverySummary,
            self._process_backend.recover_stale_executions(
                owner_id=owner_id,
                require_recovery_lease=require_recovery_lease,
                on_recovered=on_recovered,
            ),
        )

    def tool_id_referenced_outside_process(
        self,
        tool_id: str,
        *,
        excluding_pid: str,
    ) -> bool:
        return self._process_backend.tool_id_referenced_outside_process(
            tool_id,
            excluding_pid=excluding_pid,
        )

    def delete_process_scaffold(
        self,
        pid: str,
        *,
        namespace: str,
        namespace_resource: str,
    ) -> ProcessScaffoldCleanup:
        statements = (
            (
                "capabilities",
                "DELETE FROM capabilities WHERE subject = ? OR resource = ?",
                (pid, namespace_resource),
            ),
            (
                "process_resource_reservations",
                "DELETE FROM process_resource_reservations "
                "WHERE parent_pid = ? OR child_pid = ?",
                (pid, pid),
            ),
            ("llm_pending_actions", "DELETE FROM llm_pending_actions WHERE pid = ?", (pid,)),
            ("authority_manifests", "DELETE FROM authority_manifests WHERE pid = ?", (pid,)),
            ("tool_candidates", "DELETE FROM tool_candidates WHERE pid = ?", (pid,)),
            (
                "process_messages",
                "DELETE FROM process_messages WHERE sender = ? OR recipient_pid = ?",
                (pid, pid),
            ),
            (
                "object_namespaces",
                "DELETE FROM object_namespaces WHERE namespace = ? AND created_by = ?",
                (namespace, pid),
            ),
            ("processes", "DELETE FROM processes WHERE pid = ?", (pid,)),
        )
        deleted: dict[str, int] = {}
        with self.transaction(include_object_payloads=True) as cursor:
            for table, sql, params in statements:
                deleted[table] = max(0, int(cursor.execute(sql, params).rowcount))
        return ProcessScaffoldCleanup(deleted_by_table=deleted)

class ObjectRepository(_RepositoryFacade):
    """Object metadata, volatile payload, namespace, and link persistence."""

    _METHODS = frozenset(
        {
            "set_object_payload",
            "forget_object_payload",
            "is_recovered_object_payload",
            "snapshot_object_payloads",
            "insert_object",
            "update_object",
            "get_object_ref_by_name",
            "object_name_exists",
            "list_objects",
            "list_object_oids_created_by",
            "list_objects_created_by",
            "list_object_oids_owned_by",
            "delete_object",
            "insert_namespace",
            "list_namespaces",
            "insert_link",
            "list_links",
            "select_table_rows",
        }
    )

    def __init__(self, backend: TransactionBackendProtocol) -> None:
        super().__init__(backend)
        self._object_backend = cast(SnapshotCheckpointBackendProtocol, backend)
        self._object_query_backend = cast(ObjectQueryBackendProtocol, backend)
        self._object_recovery_backend = cast(ObjectRecoveryBackendProtocol, backend)

    def get_object(self, oid: str) -> AgentObject | None:
        return self._object_backend.get_object(oid)

    def get_persisted_object_state(
        self,
        oid: str,
    ) -> PersistedObjectState | None:
        return self._object_recovery_backend.get_persisted_object_state(oid)

    def query_object_refs_page(
        self,
        namespace: str,
        *,
        after: ObjectRefCursor | None,
        limit: int,
    ) -> ObjectRefPage:
        return self._object_query_backend.query_object_refs_page(
            namespace,
            after=after,
            limit=limit,
        )

    def query_child_namespaces_page(
        self,
        parent_namespace: str,
        *,
        after: ObjectNamespaceCursor | None,
        limit: int,
    ) -> ObjectNamespacePage:
        return self._object_query_backend.query_child_namespaces_page(
            parent_namespace,
            after=after,
            limit=limit,
        )

    def recover_missing_runtime_object_payloads(
        self,
        *,
        require_recovery_lease: Callable[[], None],
    ) -> ObjectPayloadRecoverySummary:
        require_recovery_lease()
        return self._object_recovery_backend.recover_missing_runtime_object_payloads(
            require_recovery_lease=require_recovery_lease,
        )

    def rehydrate_root_spawn_goal_payload(
        self,
        oid: str,
        payload: Any,
        *,
        expected_version: int,
        expected_identity_sha256: str,
        require_recovery_lease: Callable[[], None],
    ) -> bool:
        require_recovery_lease()
        return self._object_recovery_backend.rehydrate_root_spawn_goal_payload(
            oid,
            payload,
            expected_version=expected_version,
            expected_identity_sha256=expected_identity_sha256,
            require_recovery_lease=require_recovery_lease,
        )

    def payload_marker(
        self,
        *,
        present: bool,
        recovered_after_reopen: bool = False,
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._delegate(
                "payload_marker",
                present=present,
                recovered_after_reopen=recovered_after_reopen,
            ),
        )

    def get_object_by_name(
        self,
        name: str,
        namespace: str,
    ) -> AgentObject | None:
        return cast(
            AgentObject | None,
            self._delegate("get_object_by_name", name, namespace),
        )

    def list_objects_owned_by(
        self,
        owner_kind: str | ObjectOwnerKind,
        owner_id: str,
    ) -> list[AgentObject]:
        return self._object_backend.list_objects_owned_by(owner_kind, owner_id)

    def object_payload(self, oid: str) -> Any:
        return self._object_backend.object_payload(oid)

    def has_object_payload(self, oid: str) -> bool:
        return self._object_backend.has_object_payload(oid)

    def list_namespaces_created_by(self, created_by: str) -> list[ObjectNamespace]:
        return self._object_backend.list_namespaces_created_by(created_by)

    def get_namespace(self, namespace: str) -> ObjectNamespace | None:
        return self._object_backend.get_namespace(namespace)

    def namespace_exists(self, namespace: str) -> bool:
        return self._object_backend.namespace_exists(namespace)

    def get_persisted_object_metadata(self, oid: str) -> ObjectMetadata | None:
        """Read durable labels even after the volatile object payload is released."""

        rows = self._delegate(
            "select_table_rows",
            "objects",
            "oid = ? AND lifecycle_state IN (?, ?)",
            (oid, "live", "released"),
        )
        if not rows:
            return None
        return ObjectMetadata.from_persisted(loads(rows[0].get("metadata_json"), {}))


class AuthorityRepository(_RepositoryFacade):
    """Authority, reservation, capability, and data-flow persistence."""

    _METHODS = frozenset(
        {
            "insert_authority_manifest",
            "get_authority_manifest",
            "get_authority_manifest_for_process",
            "list_authority_manifests",
            "upsert_resource_reservation",
            "get_resource_reservation",
            "list_resource_reservations",
            "delete_resource_reservation",
            "delete_resource_reservations_for_process",
            "insert_capability",
            "restore_capability_use_reservation",
            "get_capability_use_reservation",
            "update_capability",
            "transition_capability_status",
            "get_process",
            "append_process_capability_ids",
            "register_sink_trust",
            "unregister_sink_trust",
            "get_sink_trust",
            "inspect_sink_trust",
            "list_sink_trust",
            "get_sink_trust_generation",
            "insert_data_flow_decision",
            "get_data_flow_decision",
            "list_data_flow_decisions",
            "upsert_file_label_binding",
            "get_file_label_binding",
            "get_file_label_binding_by_id",
            "get_file_label_binding_generation",
            "list_file_label_bindings",
            "list_file_label_bindings_for_tree",
            "tombstone_file_label_binding",
        }
    )

    def __init__(self, backend: AuthorityRecoveryBackendProtocol) -> None:
        super().__init__(backend)
        self._authority_recovery_backend = backend

    def abandon_stale_capability_use_reservations(
        self,
        *,
        require_recovery_lease: Callable[[], None],
    ) -> CapabilityUseReservationRecoverySummary:
        require_recovery_lease()
        return self._authority_recovery_backend.abandon_stale_capability_use_reservations(
            require_recovery_lease=require_recovery_lease,
        )

    def get_capability(self, cap_id: str) -> Capability | None:
        return self._authority_recovery_backend.get_capability(cap_id)

    def list_capabilities(
        self,
        subject: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[Capability]:
        return self._authority_recovery_backend.list_capabilities(
            subject,
            limit=limit,
        )

    def query_capabilities(
        self,
        subject: str | None = None,
        *,
        active_only: bool,
        after_cap_id: str | None,
        limit: int,
    ) -> list[Capability]:
        return self._authority_recovery_backend.query_capabilities(
            subject,
            active_only=active_only,
            after_cap_id=after_cap_id,
            limit=limit,
        )

    def consume_capability_uses(
        self,
        cap_id: str,
        count: int = 1,
    ) -> Capability | None:
        selected_count = _exact_positive_capability_use_count(
            count,
            operation="consume",
        )
        with _backend_mutation_scope(
            self._authority_recovery_backend,
            self.transaction,
        ):
            result = self._delegate(
                "consume_capability_uses",
                cap_id,
                selected_count,
            )
            return _exact_backend_capability_result(
                result,
                operation="consume capability uses",
            )

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
        selected_count = _exact_positive_capability_use_count(
            count,
            operation="reservation",
        )
        with _backend_mutation_scope(
            self._authority_recovery_backend,
            self.transaction,
        ):
            result = self._delegate(
                "reserve_capability_uses",
                cap_id,
                reservation_id,
                count=selected_count,
                reserved_by=reserved_by,
                reason=reason,
                created_at=created_at,
            )
            return _exact_backend_capability_result(
                result,
                operation="reserve capability uses",
            )

    def commit_capability_use_reservation(
        self,
        reservation_id: str,
        *,
        updated_at: str,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._authority_recovery_backend,
            self.transaction,
            lambda: self._delegate(
                "commit_capability_use_reservation",
                reservation_id,
                updated_at=updated_at,
            ),
            operation="commit capability use reservation",
        )

    def delete_publication_capability(self, cap_id: str) -> None:
        """Delete one publication-owned capability and its use reservations."""

        with self.transaction() as cursor:
            cursor.execute(
                "DELETE FROM capability_use_reservations WHERE cap_id = ?",
                (cap_id,),
            )
            cursor.execute("DELETE FROM capabilities WHERE cap_id = ?", (cap_id,))


class ResourceRepository(_RepositoryFacade):
    """Typed hierarchical and provider-usage reservation persistence."""

    def __init__(self, backend: ResourceBackendProtocol) -> None:
        super().__init__(backend)
        self._resource_backend = backend

    def upsert_resource_reservation(self, reservation: ResourceReservation) -> None:
        self._resource_backend.upsert_resource_reservation(reservation)

    def get_resource_reservation(
        self,
        parent_pid: str,
        child_pid: str,
    ) -> ResourceReservation | None:
        return cast(
            ResourceReservation | None,
            self._resource_backend.get_resource_reservation(parent_pid, child_pid),
        )

    def list_resource_reservations(
        self,
        *,
        parent_pid: str | None = None,
        parent_pids: Iterable[str] | None = None,
        child_pid: str | None = None,
    ) -> list[ResourceReservation]:
        return cast(
            list[ResourceReservation],
            self._resource_backend.list_resource_reservations(
                parent_pid=parent_pid,
                parent_pids=parent_pids,
                child_pid=child_pid,
            ),
        )

    def delete_resource_reservation(self, parent_pid: str, child_pid: str) -> None:
        self._resource_backend.delete_resource_reservation(parent_pid, child_pid)

    def delete_resource_reservations_for_process(self, pid: str) -> None:
        self._resource_backend.delete_resource_reservations_for_process(pid)

    def insert_resource_usage_reservation(
        self,
        *,
        reservation_id: str,
        pid: str,
        usage: ResourceUsage,
        reserved_by: str,
        reason: str,
        created_at: str,
    ) -> None:
        self._resource_backend.insert_resource_usage_reservation(
            reservation_id=reservation_id,
            pid=pid,
            usage=usage,
            reserved_by=reserved_by,
            reason=reason,
            created_at=created_at,
        )

    def get_resource_usage_reservation(
        self,
        reservation_id: str,
    ) -> ResourceUsageReservation | None:
        record = self._resource_backend.get_resource_usage_reservation(reservation_id)
        return None if record is None else self._usage_reservation(record)

    def list_resource_usage_reservations(
        self,
        *,
        pid: str | None = None,
        status: ResourceUsageReservationStatus | str | None = None,
    ) -> list[ResourceUsageReservation]:
        records = self._resource_backend.list_resource_usage_reservations(
            pid=pid,
            status=status.value if isinstance(status, ResourceUsageReservationStatus) else status,
        )
        return [self._usage_reservation(record) for record in records]

    def query_resource_usage_reservation_recovery(
        self,
        *,
        after: ResourceUsageReservationCursor | None,
        limit: int,
    ) -> ResourceUsageReservationPage:
        page = self._resource_backend.query_resource_usage_reservation_recovery(
            after=after,
            limit=limit,
        )
        return ResourceUsageReservationPage(
            records=tuple(self._usage_reservation(record) for record in page.records),
            next_cursor=page.next_cursor,
        )

    def settle_resource_usage_reservation(
        self,
        reservation_id: str,
        *,
        status: ResourceUsageReservationStatus | str,
        settled_usage: ResourceUsage,
        updated_at: str,
    ) -> bool:
        selected_status = (
            status.value if isinstance(status, ResourceUsageReservationStatus) else status
        )
        return _transactional_backend_cas_result(
            self._resource_backend,
            self.transaction,
            lambda: self._resource_backend.settle_resource_usage_reservation(
                reservation_id,
                status=selected_status,
                settled_usage=settled_usage,
                updated_at=updated_at,
            ),
            operation="settle resource usage reservation",
        )

    @staticmethod
    def _usage_reservation(record: Any) -> ResourceUsageReservation:
        if isinstance(record, ResourceUsageReservation):
            return record
        if not isinstance(record, Mapping):
            raise TypeError("resource usage reservation backend returned an invalid record")
        usage = record.get("usage")
        settled_usage = record.get("settled_usage")
        if not isinstance(usage, ResourceUsage):
            raise TypeError("resource usage reservation has invalid usage")
        if settled_usage is not None and not isinstance(settled_usage, ResourceUsage):
            raise TypeError("resource usage reservation has invalid settled usage")
        return ResourceUsageReservation(
            reservation_id=str(record["reservation_id"]),
            pid=str(record["pid"]),
            usage=usage,
            status=ResourceUsageReservationStatus(str(record["status"])),
            reserved_by=str(record["reserved_by"]),
            reason=str(record["reason"]),
            settled_usage=settled_usage,
            created_at=str(record["created_at"]),
            updated_at=str(record["updated_at"]),
        )


class RuntimePublicationRepository(_RepositoryFacade):
    """Typed publication state-machine persistence."""

    def __init__(self, backend: RuntimePublicationBackendProtocol) -> None:
        super().__init__(backend)
        self._publication_backend = backend

    def insert_runtime_publication(
        self,
        *,
        publication_id: str,
        kind: RuntimePublicationKind | str,
        pid: str,
        owner_instance_id: str,
        plan: Mapping[str, Any],
        phase: str = "planned",
    ) -> RuntimePublicationRecord:
        try:
            selected_kind = parse_runtime_publication_kind(kind)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        selected_plan = deepcopy(dict(plan))
        with self.transaction():
            inserted = self._validated_record(
                self._publication_backend.insert_runtime_publication(
                    publication_id=publication_id,
                    kind=selected_kind,
                    pid=pid,
                    owner_instance_id=owner_instance_id,
                    plan=selected_plan,
                    phase=phase,
                ),
                expected_publication_id=publication_id,
            )
            self._require_insert_binding(
                inserted,
                publication_id=publication_id,
                kind=selected_kind,
                pid=pid,
                owner_instance_id=owner_instance_id,
                plan=selected_plan,
                phase=phase,
            )
            readback = self._publication_backend.get_runtime_publication(
                publication_id
            )
            if readback is None:
                raise ValidationError(
                    "runtime publication backend did not persist the inserted record"
                )
            persisted = self._validated_record(
                readback,
                expected_publication_id=publication_id,
            )
            if persisted != inserted:
                raise ValidationError(
                    "runtime publication insert result and durable readback differ"
                )
            return inserted

    def get_runtime_publication(
        self,
        publication_id: str,
    ) -> RuntimePublicationRecord | None:
        record = self._publication_backend.get_runtime_publication(publication_id)
        return (
            self._validated_record(
                record,
                expected_publication_id=publication_id,
            )
            if record is not None
            else None
        )

    def list_runtime_publications(
        self,
        *,
        states: Iterable[RuntimePublicationState | str] | None = None,
        pid: str | None = None,
    ) -> list[RuntimePublicationRecord]:
        return [
            self._validated_record(record)
            for record in self._publication_backend.list_runtime_publications(
                states=states,
                pid=pid,
            )
        ]

    def get_committed_root_spawn_publication(
        self,
        pid: str,
    ) -> RuntimePublicationRecord | None:
        record = self._publication_backend.get_committed_root_spawn_publication(pid)
        return self._validated_recovery_record(record) if record is not None else None

    def redact_root_spawn_initial_goal(
        self,
        publication_id: str,
        *,
        pid: str,
        goal_oid: str,
        expected_payload_sha256: str,
        expected_states: Iterable[RuntimePublicationState | str] = ("committed",),
    ) -> bool:
        return _transactional_backend_cas_result(
            self._publication_backend,
            self.transaction,
            lambda: self._publication_backend.redact_root_spawn_initial_goal(
                publication_id,
                pid=pid,
                goal_oid=goal_oid,
                expected_payload_sha256=expected_payload_sha256,
                expected_states=expected_states,
            ),
            operation="redact root spawn initial goal",
        )

    @staticmethod
    def _validated_record(
        record: Mapping[str, Any],
        *,
        expected_publication_id: str | None = None,
    ) -> RuntimePublicationRecord:
        try:
            validated = validate_runtime_publication_record(record)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid runtime publication repository record: {exc}"
            ) from exc
        if (
            expected_publication_id is not None
            and validated["publication_id"] != expected_publication_id
        ):
            raise ValidationError(
                "runtime publication repository returned "
                f"publication {validated['publication_id']!r} for "
                f"request {expected_publication_id!r}"
            )
        validated["receipt"] = redact_initial_goal_recovery_receipt_projection(
            validated["receipt"]
        )
        return validated

    @staticmethod
    def _validated_recovery_record(
        record: Mapping[str, Any],
    ) -> RuntimePublicationRecord:
        """Validate the Host-only recovery projection without exposing it generally."""

        try:
            return validate_runtime_publication_record(record)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid runtime publication recovery record: {exc}"
            ) from exc

    @staticmethod
    def _require_insert_binding(
        record: RuntimePublicationRecord,
        *,
        publication_id: str,
        kind: RuntimePublicationKind,
        pid: str,
        owner_instance_id: str,
        plan: Mapping[str, Any],
        phase: str,
    ) -> None:
        expected = {
            "publication_id": publication_id,
            "kind": kind,
            "pid": pid,
            "owner_instance_id": owner_instance_id,
            "state": "planning",
            "phase": phase,
            "plan": dict(plan),
            "receipt": {"phases": [], "artifacts": []},
            "error": None,
            "operation_reconciled": False,
            "payload_delivery_state": None,
            "payload_delivery_attempt_id": None,
            "payload_delivery_started_at": None,
        }
        drifted = [
            field_name
            for field_name, expected_value in expected.items()
            if record[field_name] != expected_value
        ]
        if drifted:
            raise ValidationError(
                "runtime publication insert binding drifted: "
                + ", ".join(drifted)
            )

    def query_runtime_publication_operation_reconciliation(
        self,
        *,
        kind: RuntimePublicationKind,
        state: RuntimePublicationState | str,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage:
        return self._publication_backend.query_runtime_publication_operation_reconciliation(
            kind=kind,
            state=state,
            after=after,
            limit=limit,
        )

    def query_runtime_publication_recovery(
        self,
        *,
        kind: RuntimePublicationKind,
        state: RuntimePublicationState | str,
        operation_reconciled: bool,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage:
        return self._publication_backend.query_runtime_publication_recovery(
            kind=kind,
            state=state,
            operation_reconciled=operation_reconciled,
            after=after,
            limit=limit,
        )

    def query_checkpoint_payload_delivery_attempts(
        self,
        *,
        after: CheckpointPayloadDeliveryAttempt | None,
        limit: int,
    ) -> CheckpointPayloadDeliveryAttemptPage:
        return self._publication_backend.query_checkpoint_payload_delivery_attempts(
            after=after,
            limit=limit,
        )

    def get_checkpoint_payload_delivery_attempt_state(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> CheckpointPayloadDeliveryAttemptState | None:
        return self._publication_backend.get_checkpoint_payload_delivery_attempt_state(
            attempt
        )

    def query_checkpoint_restore_payload_deliveries(
        self,
        *,
        delivery_state: PayloadDeliveryState | str,
        attempt_id: str | None,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage:
        return self._publication_backend.query_checkpoint_restore_payload_deliveries(
            delivery_state=delivery_state,
            attempt_id=attempt_id,
            after=after,
            limit=limit,
        )

    def mark_runtime_publication_operation_reconciled(
        self,
        publication_id: str,
        *,
        expected_kind: RuntimePublicationKind,
        expected_state: RuntimePublicationState | str,
        expected_phase: str,
        expected_operation_id: str | None,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._publication_backend,
            self.transaction,
            lambda: self._publication_backend.mark_runtime_publication_operation_reconciled(
                    publication_id,
                    expected_kind=expected_kind,
                    expected_state=expected_state,
                    expected_phase=expected_phase,
                    expected_operation_id=expected_operation_id,
            ),
            operation="reconcile runtime publication operation",
        )

    def runtime_publication_exists_for_pid(
        self,
        pid: str,
        *,
        kind: RuntimePublicationKind,
    ) -> bool:
        return self._publication_backend.runtime_publication_exists_for_pid(
            pid,
            kind=kind,
        )

    def claim_runtime_publication_recovery(
        self,
        publication_id: str,
        *,
        claimant_instance_id: str,
        expected_owner_instance_id: str,
        expected_state: RuntimePublicationState | str,
        classification: str,
        max_attempts: int | None = None,
        allow_orphaned_claim_takeover: bool = False,
        claimed_state: RuntimePublicationState | str = "rollback_pending",
    ) -> RuntimePublicationRecord | None:
        with self.transaction():
            before_raw = self._publication_backend.get_runtime_publication(
                publication_id
            )
            if before_raw is None:
                return None
            before = self._validated_record(
                before_raw,
                expected_publication_id=publication_id,
            )
            claimed_raw = self._publication_backend.claim_runtime_publication_recovery(
                publication_id,
                claimant_instance_id=claimant_instance_id,
                expected_owner_instance_id=expected_owner_instance_id,
                expected_state=expected_state,
                classification=classification,
                max_attempts=max_attempts,
                allow_orphaned_claim_takeover=allow_orphaned_claim_takeover,
                claimed_state=claimed_state,
            )
            if claimed_raw is None:
                after_raw = self._publication_backend.get_runtime_publication(
                    publication_id
                )
                after = (
                    None
                    if after_raw is None
                    else self._validated_record(
                        after_raw,
                        expected_publication_id=publication_id,
                    )
                )
                if after != before:
                    raise ValidationError(
                        "runtime publication recovery claim returned no record "
                        "after mutating durable state"
                    )
                return None
            claimed = self._validated_record(
                claimed_raw,
                expected_publication_id=publication_id,
            )
            readback_raw = self._publication_backend.get_runtime_publication(
                publication_id
            )
            if readback_raw is None:
                raise ValidationError(
                    "runtime publication recovery claim lost its durable record"
                )
            readback = self._validated_record(
                readback_raw,
                expected_publication_id=publication_id,
            )
            if readback != claimed:
                raise ValidationError(
                    "runtime publication recovery claim and durable readback differ"
                )
            self._require_claim_binding(
                before,
                claimed,
                claimant_instance_id=claimant_instance_id,
                expected_owner_instance_id=expected_owner_instance_id,
                expected_state=expected_state,
                classification=classification,
                max_attempts=max_attempts,
                claimed_state=claimed_state,
            )
            return claimed

    @classmethod
    def _require_claim_binding(
        cls,
        before: RuntimePublicationRecord,
        claimed: RuntimePublicationRecord,
        *,
        claimant_instance_id: str,
        expected_owner_instance_id: str,
        expected_state: RuntimePublicationState | str,
        classification: str,
        max_attempts: int | None,
        claimed_state: RuntimePublicationState | str,
    ) -> None:
        cls._require_claim_immutable_binding(
            before,
            claimed,
            expected_owner_instance_id=expected_owner_instance_id,
            expected_state=expected_state,
        )
        recovery, attempt = cls._validated_claim_lease(
            claimed,
            claimant_instance_id=claimant_instance_id,
            classification=classification,
        )
        cls._require_claim_state_binding(
            claimed,
            recovery,
            attempt=attempt,
            max_attempts=max_attempts,
            claimed_state=claimed_state,
        )

    @staticmethod
    def _require_claim_immutable_binding(
        before: RuntimePublicationRecord,
        claimed: RuntimePublicationRecord,
        *,
        expected_owner_instance_id: str,
        expected_state: RuntimePublicationState | str,
    ) -> None:
        immutable_fields = (
            "publication_id",
            "kind",
            "pid",
            "plan",
            "created_at",
            "payload_delivery_state",
            "payload_delivery_attempt_id",
            "payload_delivery_started_at",
        )
        drifted = [
            field_name
            for field_name in immutable_fields
            if before[field_name] != claimed[field_name]
        ]
        if drifted:
            raise ValidationError(
                "runtime publication recovery claim changed immutable binding: "
                + ", ".join(drifted)
            )
        if (
            before["owner_instance_id"] != expected_owner_instance_id
            or before["state"] != expected_state
        ):
            raise ValidationError(
                "runtime publication recovery claim ignored its owner/state fence"
            )

    @staticmethod
    def _validated_claim_lease(
        claimed: RuntimePublicationRecord,
        *,
        claimant_instance_id: str,
        classification: str,
    ) -> tuple[Mapping[str, Any], int]:
        recovery = claimed["receipt"].get("recovery")
        if not isinstance(recovery, Mapping):
            raise ValidationError(
                "runtime publication recovery claim is missing its durable lease"
            )
        attempt = recovery.get("attempt")
        lease_id = recovery.get("lease_id")
        if (
            claimed["owner_instance_id"] != claimant_instance_id
            or recovery.get("claimant_instance_id") != claimant_instance_id
            or recovery.get("classification") != classification
            or type(attempt) is not int
            or attempt < 1
            or type(lease_id) is not str
            or not lease_id
        ):
            raise ValidationError(
                "runtime publication recovery claim returned an invalid lease binding"
            )
        return recovery, attempt

    @staticmethod
    def _require_claim_state_binding(
        claimed: RuntimePublicationRecord,
        recovery: Mapping[str, Any],
        *,
        attempt: int,
        max_attempts: int | None,
        claimed_state: RuntimePublicationState | str,
    ) -> None:
        exhausted = (
            max_attempts is not None
            and type(max_attempts) is int
            and attempt > max_attempts
        )
        expected_claimed_state = "manual" if exhausted else claimed_state
        expected_phase = (
            "recovery_attempts_exhausted" if exhausted else "recovery_claimed"
        )
        expected_disposition = "manual" if exhausted else "retryable"
        if (
            claimed["state"] != expected_claimed_state
            or claimed["phase"] != expected_phase
            or recovery.get("disposition") != expected_disposition
            or claimed["error"] is not None
            or claimed["operation_reconciled"] is not False
        ):
            raise ValidationError(
                "runtime publication recovery claim returned an invalid state binding"
            )
        phases = claimed["receipt"].get("phases")
        if not isinstance(phases, list) or not any(
            isinstance(phase, Mapping)
            and phase.get("phase") == "recovery_claimed"
            and all(phase.get(key) == value for key, value in recovery.items())
            for phase in phases
        ):
            raise ValidationError(
                "runtime publication recovery claim lease is absent from its transcript"
            )

    def advance_runtime_publication(
        self,
        publication_id: str,
        *,
        state: RuntimePublicationState | str,
        phase: str,
        receipt: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        expected_states: Iterable[RuntimePublicationState | str] | None = None,
        expected_phase: str | None = None,
        recovery_lease_id: str | None = None,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._publication_backend,
            self.transaction,
            lambda: self._publication_backend.advance_runtime_publication(
                publication_id,
                state=state,
                phase=phase,
                receipt=receipt,
                error=error,
                expected_states=expected_states,
                expected_phase=expected_phase,
                recovery_lease_id=recovery_lease_id,
            ),
            operation="advance runtime publication",
        )

    def update_runtime_publication_plan(
        self,
        publication_id: str,
        update: Mapping[str, Any],
        *,
        expected_states: Iterable[RuntimePublicationState | str] | None = None,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._publication_backend,
            self.transaction,
            lambda: self._publication_backend.update_runtime_publication_plan(
                publication_id,
                update,
                expected_states=expected_states,
            ),
            operation="update runtime publication plan",
        )

    def record_runtime_publication_artifact(
        self,
        publication_id: str,
        artifact: Mapping[str, Any],
        *,
        expected_states: Iterable[RuntimePublicationState | str] | None = None,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._publication_backend,
            self.transaction,
            lambda: self._publication_backend.record_runtime_publication_artifact(
                publication_id,
                artifact,
                expected_states=expected_states,
            ),
            operation="record runtime publication artifact",
        )


class CheckpointRestorePublicationWriter:
    """Opaque mutation port for the checkpoint-restore publication machine.

    The token never crosses this storage-owned adapter.  Runtime orchestration
    receives only the typed port, while the generic Host-visible publication
    repository remains unable to mutate checkpoint restore rows.
    """

    def __init__(self, backend: CheckpointPublicationWriterBackendProtocol) -> None:
        self.__backend = backend
        self.__token = backend._issue_checkpoint_restore_writer_token()

    @staticmethod
    def _validated_record(
        record: Mapping[str, Any],
        *,
        expected_publication_id: str,
    ) -> RuntimePublicationRecord:
        try:
            validated = validate_runtime_publication_record(record)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid checkpoint restore publication writer record: {exc}"
            ) from exc
        if validated["publication_id"] != expected_publication_id:
            raise ValidationError(
                "checkpoint restore publication writer returned "
                f"publication {validated['publication_id']!r} for "
                f"request {expected_publication_id!r}"
            )
        return validated

    def insert_runtime_publication(
        self,
        *,
        publication_id: str,
        kind: RuntimePublicationKind | str,
        pid: str,
        owner_instance_id: str,
        plan: Mapping[str, Any],
        phase: str = "planned",
    ) -> RuntimePublicationRecord:
        return self._validated_record(
            self.__backend.insert_runtime_publication(
                publication_id=publication_id,
                kind=kind,
                pid=pid,
                owner_instance_id=owner_instance_id,
                plan=plan,
                phase=phase,
                _checkpoint_restore_writer_token=self.__token,
            ),
            expected_publication_id=publication_id,
        )

    def claim_runtime_publication_recovery(
        self,
        publication_id: str,
        *,
        claimant_instance_id: str,
        expected_owner_instance_id: str,
        expected_state: RuntimePublicationState | str,
        classification: str,
        max_attempts: int | None = None,
        allow_orphaned_claim_takeover: bool = False,
        claimed_state: RuntimePublicationState | str = "rollback_pending",
    ) -> RuntimePublicationRecord | None:
        record = self.__backend.claim_runtime_publication_recovery(
            publication_id,
            claimant_instance_id=claimant_instance_id,
            expected_owner_instance_id=expected_owner_instance_id,
            expected_state=expected_state,
            classification=classification,
            max_attempts=max_attempts,
            allow_orphaned_claim_takeover=allow_orphaned_claim_takeover,
            claimed_state=claimed_state,
            _checkpoint_restore_writer_token=self.__token,
        )

        if record is None:
            return None
        return self._validated_record(
            record,
            expected_publication_id=publication_id,
        )

    def advance_runtime_publication(
        self,
        publication_id: str,
        *,
        state: RuntimePublicationState | str,
        phase: str,
        receipt: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        expected_states: Iterable[RuntimePublicationState | str] | None = None,
        expected_phase: str | None = None,
        recovery_lease_id: str | None = None,
    ) -> bool:
        return _transactional_backend_cas_result(
            self.__backend,
            self.__backend.transaction,
            lambda: self.__backend.advance_runtime_publication(
                publication_id,
                state=state,
                phase=phase,
                receipt=receipt,
                error=error,
                expected_states=expected_states,
                expected_phase=expected_phase,
                recovery_lease_id=recovery_lease_id,
                _checkpoint_restore_writer_token=self.__token,
            ),
            operation="advance checkpoint restore publication",
        )

    def record_runtime_publication_artifact(
        self,
        publication_id: str,
        artifact: Mapping[str, Any],
        *,
        expected_states: Iterable[RuntimePublicationState | str] | None = None,
    ) -> bool:
        return _transactional_backend_cas_result(
            self.__backend,
            self.__backend.transaction,
            lambda: self.__backend.record_runtime_publication_artifact(
                publication_id,
                artifact,
                expected_states=expected_states,
                _checkpoint_restore_writer_token=self.__token,
            ),
            operation="record checkpoint restore publication artifact",
        )

    def mark_runtime_publication_operation_reconciled(
        self,
        publication_id: str,
        *,
        expected_kind: RuntimePublicationKind | str,
        expected_state: RuntimePublicationState | str,
        expected_phase: str,
        expected_operation_id: str | None,
    ) -> bool:
        return _transactional_backend_cas_result(
            self.__backend,
            self.__backend.transaction,
            lambda: self.__backend.mark_runtime_publication_operation_reconciled(
                    publication_id,
                    expected_kind=expected_kind,
                    expected_state=expected_state,
                    expected_phase=expected_phase,
                    expected_operation_id=expected_operation_id,
                    _checkpoint_restore_writer_token=self.__token,
            ),
            operation="reconcile checkpoint restore publication operation",
        )

    def transition_payload_delivery(
        self,
        publication_id: str,
        *,
        expected_delivery_state: str | None,
        delivery_state: str,
        expected_attempt: CheckpointPayloadDeliveryAttempt | None = None,
        delivery_attempt: CheckpointPayloadDeliveryAttempt | None = None,
        owner_instance_id: str | None = None,
        recovery_lease_id: str | None = None,
    ) -> bool:
        return _transactional_backend_cas_result_rollback_on_error(
            self.__backend,
            self.__backend.transaction,
            lambda: self.__backend.transition_checkpoint_restore_payload_delivery(
                    publication_id,
                    expected_delivery_state=expected_delivery_state,
                    delivery_state=delivery_state,
                    expected_attempt=expected_attempt,
                    delivery_attempt=delivery_attempt,
                    owner_instance_id=owner_instance_id,
                    recovery_lease_id=recovery_lease_id,
                    _checkpoint_restore_writer_token=self.__token,
            ),
            operation="transition checkpoint payload delivery",
        )

    def begin_checkpoint_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> bool:
        return _transactional_backend_cas_result(
            self.__backend,
            self.__backend.transaction,
            lambda: self.__backend.begin_checkpoint_payload_delivery_attempt(
                attempt,
                _checkpoint_restore_writer_token=self.__token,
            ),
            operation="begin checkpoint payload delivery attempt",
        )

    def ack_checkpoint_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> bool:
        return _transactional_backend_cas_result(
            self.__backend,
            self.__backend.transaction,
            lambda: self.__backend.ack_checkpoint_payload_delivery_attempt(
                attempt,
                _checkpoint_restore_writer_token=self.__token,
            ),
            operation="ack checkpoint payload delivery attempt",
        )

    def abort_checkpoint_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> bool:
        return _transactional_backend_cas_result(
            self.__backend,
            self.__backend.transaction,
            lambda: self.__backend.abort_checkpoint_payload_delivery_attempt(
                attempt,
                _checkpoint_restore_writer_token=self.__token,
            ),
            operation="abort checkpoint payload delivery attempt",
        )


class SnapshotCheckpointRepository(_RepositoryFacade):
    """Canonical snapshot/checkpoint persistence over one SQL backend.

    Runtime orchestration receives typed snapshot aggregates while generic
    table access, backend SQL, and Object-payload cache coordination remain
    confined to this repository boundary.
    """

    def __init__(self, backend: SnapshotCheckpointBackendProtocol) -> None:
        super().__init__(backend)
        self._snapshot_backend = backend

    def capture_process_exec_snapshot_rows(
        self,
        pid: str,
        *,
        process_namespace: str,
    ) -> tuple[SnapshotRows, dict[str, Any]]:
        with self.transaction():
            return self._capture_process_exec_snapshot_rows(
                pid,
                process_namespace=process_namespace,
            )

    def _capture_process_exec_snapshot_rows(
        self,
        pid: str,
        *,
        process_namespace: str,
    ) -> tuple[SnapshotRows, dict[str, Any]]:
        from agent_libos.models.snapshot import SnapshotRows

        process_rows = self._select_rows("processes", "pid = ?", (pid,))
        if not process_rows:
            raise NotFound(f"process not found: {pid}")
        object_rows = self._owned_object_rows(pid)
        object_oids = [str(row["oid"]) for row in object_rows]
        namespace_rows = self._owned_namespace_rows(pid, process_namespace)
        rows = SnapshotRows.from_trusted_durable_mapping(
            {
                "processes": process_rows,
                "object_namespaces": namespace_rows,
                "objects": object_rows,
                "object_links": self._object_link_rows(object_oids),
                "capabilities": self._select_rows(
                    "capabilities",
                    "subject = ?",
                    (pid,),
                    order_by="cap_id",
                ),
                "process_resource_reservations": self._select_rows(
                    "process_resource_reservations",
                    "parent_pid = ? OR child_pid = ?",
                    (pid, pid),
                    order_by="parent_pid, child_pid",
                ),
                "process_messages": [],
                "llm_pending_actions": self._select_rows(
                    "llm_pending_actions",
                    "pid = ?",
                    (pid,),
                ),
                "skills": [],
                "tools": [],
                "tool_candidates": self._select_rows(
                    "tool_candidates",
                    "pid = ?",
                    (pid,),
                    order_by="candidate_id",
                ),
            }
        )
        payloads = cast(
            dict[str, Any],
            self._snapshot_backend.snapshot_object_payloads(object_oids),
        )
        return rows, payloads

    def restore_process_exec_snapshot(
        self,
        snapshot: ProcessSnapshot,
        *,
        process_namespace: str,
        captured_tool_ids: Iterable[str],
        capability_rollback_token: str | None,
        fence_execution: bool = True,
    ) -> frozenset[str]:
        pid = snapshot.header.root_pid
        current_process = cast(
            AgentProcess | None,
            self._snapshot_backend.get_process(pid),
        )
        if current_process is None:
            raise NotFound(f"process not found during exec compensation: {pid}")
        before_process_rows = list(snapshot.rows.processes)
        if len(before_process_rows) != 1:
            raise NotFound(f"exec rollback snapshot has no unique process row: {pid}")
        before_process_row = dict(before_process_rows[0])
        current_object_oids = [
            str(row["oid"])
            for row in self._owned_object_rows(pid)
        ]
        before_object_oids = set(snapshot.owned_object_oids)
        current_object_oid_set = set(current_object_oids)
        borrowed_new_oids = self._externally_borrowed_oids(
            pid,
            current_object_oid_set - before_object_oids,
        )
        cleanup_object_oids = (
            current_object_oid_set - before_object_oids
        ) - borrowed_new_oids
        object_oids = sorted(before_object_oids | cleanup_object_oids)
        namespace_names = sorted(
            set(snapshot.owned_namespaces)
            | {
                str(row["namespace"])
                for row in self._owned_namespace_rows(pid, process_namespace)
            }
        )
        stale_tool_ids = set(current_process.tool_table.values()) - {
            str(tool_id) for tool_id in captured_tool_ids
        }
        with self.transaction(include_object_payloads=True) as cursor:
            self._delete_process_exec_scope(
                cursor,
                pid=pid,
                object_oids=object_oids,
                namespace_names=namespace_names,
            )
            self._insert_process_exec_snapshot_rows(snapshot)
            before_capability_rows = list(snapshot.rows.capabilities)
            self._restore_exec_revoked_capabilities(
                cursor,
                pid=pid,
                before_rows=before_capability_rows,
                rollback_token=capability_rollback_token,
            )
            self._remove_exec_created_capabilities(
                cursor,
                pid=pid,
                before_rows=before_capability_rows,
                cleanup_object_oids=cleanup_object_oids,
                rollback_token=capability_rollback_token,
            )
            capability_ids = [
                str(row["cap_id"])
                for row in cursor.execute(
                    "SELECT cap_id FROM capabilities WHERE subject = ? ORDER BY cap_id",
                    (pid,),
                )
            ]
            process_restored = _exact_backend_cas_result(
                self._snapshot_backend.restore_process_for_exec(
                    before_process_row,
                    expected_revision=current_process.revision,
                    publication_id=(
                        capability_rollback_token if fence_execution else None
                    ),
                    capability_ids=capability_ids,
                    fence_execution=fence_execution,
                ),
                operation="restore process for exec",
            )
            if not process_restored:
                raise ProcessRevisionConflict(
                    f"exec recovery conflict prevented process convergence: {pid}"
                )
        return frozenset(stale_tool_ids)

    def install_checkpoint_image_rows(
        self,
        rows: SnapshotRows,
        *,
        object_payloads: Mapping[str, Any],
    ) -> None:
        """Install remapped image memory/authority rows atomically."""

        with self.transaction(include_object_payloads=True) as cursor:
            for row in rows.object_namespaces:
                exists = cursor.execute(
                    "SELECT 1 FROM object_namespaces WHERE namespace = ?",
                    (row["namespace"],),
                ).fetchone()
                if exists is None:
                    self._snapshot_backend.insert_table_row(
                        "object_namespaces", dict(row)
                    )
            for row in rows.objects:
                item = dict(row)
                oid = str(item["oid"])
                if oid in object_payloads:
                    item["payload_json"] = dumps(object_payloads[oid])
                self._snapshot_backend.insert_table_row("objects", item)
                if oid in object_payloads:
                    self._snapshot_backend.set_object_payload(
                        oid,
                        deepcopy(object_payloads[oid]),
                    )
            for row in rows.object_links:
                self._snapshot_backend.insert_table_row("object_links", dict(row))
            for row in rows.capabilities:
                self._snapshot_backend.insert_table_row("capabilities", dict(row))

    def load_process_snapshot_rows(self, pids: Iterable[str]) -> SnapshotRows:
        from agent_libos.models.snapshot import SnapshotRows

        return SnapshotRows(
            processes=tuple(
                self._rows_by_values("processes", "pid", pids)
            )
        )

    def insert_checkpoint(
        self,
        checkpoint: Checkpoint,
        snapshot: ProcessSnapshot,
    ) -> None:
        selected_snapshot = self._validated_checkpoint_pair(
            checkpoint,
            snapshot,
            expected_checkpoint_id=checkpoint.checkpoint_id,
        )
        self._snapshot_backend.insert_checkpoint(
            checkpoint,
            selected_snapshot.to_mapping(),
        )

    def get_checkpoint_snapshot(
        self,
        checkpoint_id: str,
    ) -> tuple[Checkpoint, ProcessSnapshot] | None:
        from agent_libos.models.snapshot import ProcessSnapshot

        found = self._snapshot_backend.get_checkpoint_snapshot(checkpoint_id)
        if found is None:
            return None
        checkpoint, snapshot = found
        try:
            selected_snapshot = ProcessSnapshot.from_mapping(snapshot)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid checkpoint repository snapshot: {exc}"
            ) from exc
        selected_checkpoint = self._validated_checkpoint_pair(
            checkpoint,
            selected_snapshot,
            expected_checkpoint_id=checkpoint_id,
        )
        return cast(Checkpoint, checkpoint), selected_checkpoint

    @staticmethod
    def _validated_checkpoint_pair(
        checkpoint: Any,
        snapshot: ProcessSnapshot,
        *,
        expected_checkpoint_id: str,
    ) -> ProcessSnapshot:
        from agent_libos.models.snapshot import (
            ProcessSnapshot,
            SNAPSHOT_SCHEMA_VERSION,
        )

        if type(checkpoint) is not Checkpoint:
            raise ValidationError(
                "checkpoint repository backend returned an invalid checkpoint record"
            )
        if type(expected_checkpoint_id) is not str or not expected_checkpoint_id:
            raise ValidationError("checkpoint id must be exact non-empty text")
        if not isinstance(snapshot, ProcessSnapshot):
            raise ValidationError(
                "checkpoint repository backend returned an invalid snapshot record"
            )
        try:
            canonical_snapshot = ProcessSnapshot.from_mapping(snapshot.to_mapping())
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid checkpoint repository snapshot: {exc}"
            ) from exc
        header = canonical_snapshot.header
        if header.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported snapshot version: {header.schema_version}"
            )
        identity_fields = (
            (
                "checkpoint_id",
                checkpoint.checkpoint_id,
                header.checkpoint_id,
                str,
            ),
            ("pid", checkpoint.pid, header.root_pid, str),
            ("reason", checkpoint.reason, header.reason, str),
            ("created_at", checkpoint.created_at, header.created_at, str),
            ("created_by", checkpoint.created_by, header.created_by, str),
            (
                "snapshot_version",
                checkpoint.snapshot_version,
                header.schema_version,
                int,
            ),
        )
        for field_name, checkpoint_value, snapshot_value, expected_type in identity_fields:
            if (
                type(checkpoint_value) is not expected_type
                or type(snapshot_value) is not expected_type
                or checkpoint_value != snapshot_value
            ):
                raise ValidationError(
                    "checkpoint snapshot changed: row and snapshot header binding differ: "
                    f"{field_name}"
                )
        if checkpoint.checkpoint_id != expected_checkpoint_id:
            raise ValidationError(
                "checkpoint repository returned "
                f"checkpoint {checkpoint.checkpoint_id!r} for "
                f"request {expected_checkpoint_id!r}"
            )
        if (
            type(checkpoint.effect_ledger_seq) is not int
            or checkpoint.effect_ledger_seq < 0
            or (
                checkpoint.metadata is not None
                and type(checkpoint.metadata) is not dict
            )
        ):
            raise ValidationError(
                "checkpoint repository returned invalid checkpoint metadata"
            )
        return canonical_snapshot

    def list_checkpoints(
        self,
        *,
        pid: str | None = None,
        limit: int | None = None,
    ) -> list[Checkpoint]:
        return cast(
            list[Checkpoint],
            self._snapshot_backend.list_checkpoints(pid=pid, limit=limit),
        )

    def capture_checkpoint_rows(
        self,
        process_rows: Iterable[Mapping[str, Any]],
        *,
        object_oids: Iterable[str],
        namespace_names: Iterable[str],
    ) -> tuple[SnapshotRows, dict[str, Any]]:
        with self.transaction():
            return self._capture_checkpoint_rows(
                process_rows,
                object_oids=object_oids,
                namespace_names=namespace_names,
            )

    def _capture_checkpoint_rows(
        self,
        process_rows: Iterable[Mapping[str, Any]],
        *,
        object_oids: Iterable[str],
        namespace_names: Iterable[str],
    ) -> tuple[SnapshotRows, dict[str, Any]]:
        from agent_libos.models.snapshot import SnapshotRows

        selected_process_rows = [dict(row) for row in process_rows]
        pids = [str(row["pid"]) for row in selected_process_rows]
        selected_object_oids = list(dict.fromkeys(str(oid) for oid in object_oids))
        selected_namespaces = list(
            dict.fromkeys(str(namespace) for namespace in namespace_names)
        )
        capability_rows = [
            row
            for row in self._rows_by_values(
                "capabilities",
                "subject",
                pids,
            )
            if not str(row["resource"]).startswith("checkpoint:")
        ]
        reservation_rows: list[dict[str, Any]] = []
        if pids:
            scoped_pids = frozenset(pids)
            for batch in _snapshot_value_batches(pids):
                placeholders = ", ".join("?" for _ in batch)
                reservation_rows.extend(
                    row
                    for row in self._select_rows(
                        "process_resource_reservations",
                        f"parent_pid IN ({placeholders})",
                        batch,
                        order_by="parent_pid, child_pid",
                    )
                    if str(row["child_pid"]) in scoped_pids
                )
            reservation_rows.sort(
                key=lambda row: (str(row["parent_pid"]), str(row["child_pid"]))
            )
        tool_ids = {
            str(tool_id)
            for row in selected_process_rows
            for tool_id in loads(row.get("tool_table_json"), {}).values()
        }
        skill_ids = {
            str(skill_id)
            for row in selected_process_rows
            for skill_id in loads(row.get("loaded_skills_json"), {}).keys()
        }
        rows = SnapshotRows.from_trusted_durable_mapping(
            {
                "processes": selected_process_rows,
                "object_namespaces": self._rows_by_values(
                    "object_namespaces",
                    "namespace",
                    selected_namespaces,
                ),
                "objects": self._rows_by_values(
                    "objects",
                    "oid",
                    selected_object_oids,
                ),
                "object_links": self._object_link_rows(selected_object_oids),
                "capabilities": capability_rows,
                "process_resource_reservations": reservation_rows,
                "process_messages": self._rows_by_values(
                    "process_messages",
                    "recipient_pid",
                    pids,
                ),
                "llm_pending_actions": self._rows_by_values(
                    "llm_pending_actions",
                    "pid",
                    pids,
                ),
                "skills": self._rows_by_values(
                    "skills",
                    "skill_id",
                    sorted(skill_ids),
                ),
                "tools": self._rows_by_values(
                    "tools",
                    "tool_id",
                    sorted(tool_ids),
                ),
                "tool_candidates": self._rows_by_values(
                    "tool_candidates",
                    "pid",
                    pids,
                ),
            }
        ).without_task_run_private_replay_rows()
        return rows, cast(
            dict[str, Any],
            self._snapshot_backend.snapshot_object_payloads(selected_object_oids),
        )

    def prepare_checkpoint_restore_process_rows(
        self,
        rows: SnapshotRows,
        *,
        restored_capability_rows: Iterable[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        restored = [dict(row) for row in rows.processes]
        restored_pids = [str(row["pid"]) for row in restored]
        current_concurrency: dict[str, tuple[int, int, int]] = {}
        with self.transaction() as cursor:
            for batch in _snapshot_value_batches(restored_pids):
                placeholders = ", ".join("?" for _ in batch)
                for row in cursor.execute(
                    "SELECT pid, revision, execution_generation, state_generation "
                    f"FROM processes WHERE pid IN ({placeholders})",
                    batch,
                ):
                    current_concurrency[str(row["pid"])] = (
                        int(row["revision"]),
                        int(row["execution_generation"]),
                        int(row["state_generation"]),
                    )
            floors: list[ProcessRestoreEpoch] = []
            for process_row in restored:
                pid = str(process_row["pid"])
                current_revision, current_generation, current_state_generation = (
                    current_concurrency.get(pid, (0, 0, 0))
                )
                floors.append(
                    ProcessRestoreEpoch(
                        pid=pid,
                        revision=max(
                            current_revision,
                            int(process_row.get("revision") or 0),
                        ),
                        execution_generation=max(
                            current_generation,
                            int(process_row.get("execution_generation") or 0),
                        ),
                        state_generation=max(
                            current_state_generation,
                            int(process_row.get("state_generation") or 0),
                        ),
                    )
                )
            reserved = self._snapshot_backend.reserve_process_restore_epochs(
                floors,
                cursor=cursor,
            )
            reserved_by_pid = {str(item.pid): item for item in reserved}
            if set(reserved_by_pid) != set(restored_pids):
                raise ValidationError(
                    "process restore epoch reservation did not return the exact PID set"
                )
            for process_row in restored:
                epoch = reserved_by_pid[str(process_row["pid"])]
                process_row["revision"] = epoch.revision
                process_row["execution_generation"] = epoch.execution_generation
                process_row["state_generation"] = epoch.state_generation
                process_row["execution_owner_id"] = None
                process_row["execution_lease_id"] = None
        capability_ids: dict[str, list[str]] = {}
        for capability_row in restored_capability_rows:
            capability_ids.setdefault(str(capability_row["subject"]), []).append(
                str(capability_row["cap_id"])
            )
        for process_row in restored:
            process_row["capabilities_json"] = dumps(
                sorted(capability_ids.get(str(process_row["pid"]), []))
            )
        return tuple(restored)

    def cancel_pending_human_requests_after_checkpoint(
        self,
        pids: Iterable[str],
        checkpoint: Checkpoint,
    ) -> list[str]:
        cancelled: list[str] = []
        with self.transaction() as cursor:
            for pid in dict.fromkeys(str(pid) for pid in pids):
                requests = self._snapshot_backend.list_human_requests(
                    pid=pid,
                    status=HumanRequestStatus.PENDING,
                )
                for request in requests:
                    if request.created_at <= checkpoint.created_at:
                        continue
                    updated = cursor.execute(
                        "UPDATE human_requests "
                        "SET status = ?, decision_json = ?, updated_at = ? "
                        "WHERE request_id = ? AND status = ? AND created_at > ?",
                        (
                            HumanRequestStatus.CANCELLED.value,
                            dumps(
                                {
                                    "cancelled_by": (
                                        f"checkpoint:{checkpoint.checkpoint_id}"
                                    )
                                }
                            ),
                            utc_now(),
                            request.request_id,
                            HumanRequestStatus.PENDING.value,
                            checkpoint.created_at,
                        ),
                    )
                    if updated.rowcount == 1:
                        cancelled.append(str(request.request_id))
        return cancelled

    def supersede_messages_after_checkpoint(
        self,
        pids: Iterable[str],
        checkpoint: Checkpoint,
    ) -> list[str]:
        superseded: list[str] = []
        with self.transaction() as cursor:
            for pid in dict.fromkeys(str(pid) for pid in pids):
                messages = self._snapshot_backend.list_process_messages(
                    pid,
                    status=ProcessMessageStatus.UNREAD,
                )
                for message in messages:
                    if message.created_at <= checkpoint.created_at:
                        continue
                    payload = {
                        **message.payload,
                        "superseded_by_restore": checkpoint.checkpoint_id,
                        "superseded_at": utc_now(),
                    }
                    cursor.execute(
                        "UPDATE process_messages "
                        "SET payload_json = ?, status = ?, updated_at = ? "
                        "WHERE message_id = ?",
                        (
                            dumps(payload),
                            ProcessMessageStatus.SUPERSEDED_BY_RESTORE.value,
                            utc_now(),
                            message.message_id,
                        ),
                    )
                    superseded.append(str(message.message_id))
        return superseded

    def _iter_checkpoint_object_tasks(
        self,
        *,
        statuses: Iterable[ObjectTaskStatus],
        checkpoint_created_at: str,
        terminal_after: bool,
        pids: Iterable[str],
        object_oids: Iterable[str],
    ) -> Iterator[ObjectTask]:
        """Page only task rows intersecting one checkpoint's process/Object scope."""

        seen: set[str] = set()
        for scope_kind, values in (
            ("scope_pids", sorted({str(pid) for pid in pids})),
            ("owner_oids", sorted({str(oid) for oid in object_oids})),
        ):
            for batch in _snapshot_value_batches(values):
                after: ObjectTaskRecoveryCursor | None = None
                while True:
                    query_scope = {
                        "scope_pids": (),
                        "owner_oids": (),
                        scope_kind: batch,
                    }
                    page = self._snapshot_backend.query_checkpoint_object_tasks(
                        statuses=statuses,
                        checkpoint_created_at=checkpoint_created_at,
                        terminal_after=terminal_after,
                        after=after,
                        **query_scope,
                    )
                    if not isinstance(page, ObjectTaskRecoveryPage):
                        raise ValidationError(
                            "checkpoint task query returned an invalid page"
                        )
                    if after is not None and any(
                        (task.created_at, task.task_id)
                        <= (after.created_at, after.task_id)
                        for task in page.records
                    ):
                        raise ValidationError(
                            "checkpoint task query did not advance its cursor"
                        )
                    for task in page.records:
                        task_id = str(task.task_id)
                        if task_id in seen:
                            continue
                        seen.add(task_id)
                        yield task
                    if page.next_cursor is None:
                        break
                    after = page.next_cursor

    def supersede_object_tasks_after_checkpoint(
        self,
        pids: Iterable[str],
        object_oids: Iterable[str],
        checkpoint: Checkpoint,
    ) -> list[str]:
        scoped_pids = {str(pid) for pid in pids}
        scoped_oids = {str(oid) for oid in object_oids}
        terminal_statuses = (
            ObjectTaskStatus.SUCCEEDED,
            ObjectTaskStatus.FAILED,
            ObjectTaskStatus.CANCELLED,
            ObjectTaskStatus.ABANDONED,
            ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN,
        )
        superseded: list[str] = []
        now = utc_now()
        with self.transaction() as cursor:
            for task in self._iter_checkpoint_object_tasks(
                statuses=terminal_statuses,
                checkpoint_created_at=checkpoint.created_at,
                terminal_after=True,
                pids=scoped_pids,
                object_oids=scoped_oids,
            ):
                wait = {
                    **task.wait,
                    "superseded_by_restore": checkpoint.checkpoint_id,
                    "superseded_at": now,
                    "previous_status": task.status.value,
                    "previous_runner_pid": (
                        str(task.runner_pid)
                        if task.runner_pid is not None
                        else None
                    ),
                    "previous_result_oid": (
                        str(task.result_oid)
                        if task.result_oid is not None
                        else task.wait.get("previous_result_oid")
                    ),
                    "previous_error": task.error,
                }
                cursor.execute(
                    "UPDATE object_tasks SET runner_pid = NULL, status = ?, "
                    "result_oid = NULL, error = ?, wait_json = ?, updated_at = ? "
                    "WHERE task_id = ?",
                    (
                        ObjectTaskStatus.SUPERSEDED_BY_RESTORE.value,
                        "superseded by checkpoint restore "
                        f"{checkpoint.checkpoint_id}",
                        dumps(wait),
                        now,
                        task.task_id,
                    ),
                )
                superseded.append(str(task.task_id))
        return sorted(superseded)

    def replace_checkpoint_scope(
        self,
        snapshot: ProcessSnapshot,
        *,
        current_pids: Iterable[str],
        current_object_oids: Iterable[str],
        all_object_oids: Iterable[str],
        all_namespace_names: Iterable[str],
        snapshot_object_oids: Iterable[str],
        snapshot_namespaces: Iterable[str],
        restored_process_rows: Iterable[Mapping[str, Any]],
        restored_capability_rows: Iterable[Mapping[str, Any]],
        before_insert: Callable[[object, str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        selected_pids = list(dict.fromkeys(str(pid) for pid in current_pids))
        current_oids = {str(oid) for oid in current_object_oids}
        object_oids = {str(oid) for oid in all_object_oids}
        namespace_names = {str(item) for item in all_namespace_names}
        snapshot_oids = {str(oid) for oid in snapshot_object_oids}
        snapshot_namespace_names = {str(item) for item in snapshot_namespaces}
        restored_process_items = tuple(dict(row) for row in restored_process_rows)
        restored_capability_items = tuple(
            dict(row) for row in restored_capability_rows
        )
        with self.transaction(include_object_payloads=True) as cursor:
            self._delete_checkpoint_scope(
                cursor,
                pids=selected_pids,
                current_object_oids=current_oids,
                object_oids=object_oids,
                namespace_names=namespace_names,
                snapshot_object_oids=snapshot_oids,
            )
            self._insert_checkpoint_memory_rows(
                cursor,
                snapshot=snapshot,
                snapshot_object_oids=snapshot_oids,
                snapshot_namespaces=snapshot_namespace_names,
                before_insert=before_insert,
            )
            self._insert_checkpoint_process_rows(
                cursor,
                rows=snapshot.rows,
                restored_process_rows=restored_process_items,
                restored_capability_rows=restored_capability_items,
                before_insert=before_insert,
            )

    def reconcile_checkpoint_object_payloads(
        self,
        snapshot: ProcessSnapshot,
    ) -> tuple[str, ...]:
        """Rehydrate checkpoint-owned volatile payloads without reviving rows.

        The immutable checkpoint supplies the payload values, while exact row
        comparison proves the payload still belongs to the main restore that
        committed. A monotonically newer or explicitly released Object has
        already consumed that delivery and is skipped; missing, stale, or
        identity-drifted rows fail closed instead of reviving old payloads.
        """

        owned_oids = {str(oid) for oid in snapshot.owned_object_oids}
        payloads = {
            str(oid): deepcopy(payload)
            for oid, payload in snapshot.object_payloads.items()
            if str(oid) in owned_oids
        }
        if not payloads:
            return ()
        expected_rows = {
            str(row["oid"]): dict(row)
            for row in snapshot.rows.objects
            if str(row["oid"]) in payloads
        }
        if set(expected_rows) != set(payloads):
            raise ValidationError(
                "checkpoint Object payloads do not match the owned Object rows"
            )
        current_rows: dict[str, dict[str, Any]] = {}
        with self.transaction(include_object_payloads=True):
            for batch in _snapshot_value_batches(sorted(payloads)):
                placeholders = ", ".join("?" for _ in batch)
                for row in self._select_rows(
                    "objects",
                    f"oid IN ({placeholders})",
                    batch,
                    order_by="oid",
                ):
                    current_rows[str(row["oid"])] = row
            if set(current_rows) != set(payloads):
                raise ValidationError(
                    "checkpoint Object payload reconciliation found missing rows"
                )
            present_marker = self._snapshot_backend.payload_marker(present=True)
            hydrated: list[str] = []
            for oid in sorted(payloads):
                current = dict(current_rows[oid])
                expected = dict(expected_rows[oid])
                try:
                    current_marker = loads(current.pop("payload_json"), {})
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        f"checkpoint Object payload marker is invalid: {oid}"
                    ) from exc
                expected.pop("payload_json", None)
                exact_restore_row = (
                    current_marker == present_marker
                    and current.get("lifecycle_state")
                    == ObjectLifecycleState.LIVE.value
                    and current == expected
                )
                if exact_restore_row:
                    hydrated.append(oid)
                    continue
                if not _checkpoint_object_payload_was_superseded(
                    current,
                    expected,
                    current_marker=current_marker,
                    present_marker=present_marker,
                ):
                    raise ValidationError(
                        "checkpoint Object row changed before payload reconciliation: "
                        f"{oid}"
                    )
            for oid in hydrated:
                self._snapshot_backend.set_object_payload(oid, payloads[oid])
        return tuple(hydrated)

    def rehydrate_checkpoint_fork_object_payloads(
        self,
        rows: SnapshotRows,
        *,
        object_payloads: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Exactly restore volatile payloads after a lost fork commit ACK.

        The fork rows are immutable inputs allocated for one publication.  A
        durable row must still match that exact input and retain the present
        runtime-memory marker before its volatile payload is reconstructed.
        Validation and the cache replacement share the backend lock so a
        scheduler or Object mutation cannot observe an intermediate state.
        """

        if any(not isinstance(oid, str) or not oid for oid in object_payloads):
            raise ValidationError(
                "checkpoint fork Object payload ids must be non-empty strings"
            )
        payloads = {
            oid: deepcopy(payload)
            for oid, payload in object_payloads.items()
        }
        object_rows = tuple(dict(row) for row in rows.objects)
        if any(
            not isinstance(row.get("oid"), str) or not row.get("oid")
            for row in object_rows
        ):
            raise ValidationError(
                "checkpoint fork Object row ids must be non-empty strings"
            )
        expected_rows = {str(row["oid"]): row for row in object_rows}
        if len(expected_rows) != len(object_rows):
            raise ValidationError("checkpoint fork Object rows contain duplicate ids")
        if set(expected_rows) != set(payloads):
            raise ValidationError(
                "checkpoint fork Object payloads do not match Object rows"
            )
        if not payloads:
            return ()
        with self.locked():
            current_rows: dict[str, dict[str, Any]] = {}
            for batch in _snapshot_value_batches(sorted(payloads)):
                placeholders = ", ".join("?" for _ in batch)
                for row in self._select_rows(
                    "objects",
                    f"oid IN ({placeholders})",
                    batch,
                    order_by="oid",
                ):
                    current_rows[str(row["oid"])] = row
            if set(current_rows) != set(payloads):
                raise ValidationError(
                    "checkpoint fork payload rehydration found missing Object rows"
                )
            present_marker = self._snapshot_backend.payload_marker(present=True)
            for oid in sorted(payloads):
                current = dict(current_rows[oid])
                expected = dict(expected_rows[oid])
                try:
                    current_marker = loads(current.pop("payload_json"), {})
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        f"checkpoint fork Object payload marker is invalid: {oid}"
                    ) from exc
                expected.pop("payload_json", None)
                current_state = (
                    current_marker,
                    current.get("lifecycle_state"),
                    current,
                )
                expected_state = (
                    present_marker,
                    ObjectLifecycleState.LIVE.value,
                    expected,
                )
                if current_state != expected_state:
                    raise ValidationError(
                        "checkpoint fork Object row changed before payload "
                        f"rehydration: {oid}"
                    )
            return self._snapshot_backend.rehydrate_object_payload_cache(payloads)

    def publish_checkpoint_fork_process_rows(
        self,
        rows: SnapshotRows,
    ) -> tuple[str, ...]:
        """Atomically publish exact target states from durable quarantine."""

        target_rows = tuple(deepcopy(dict(row)) for row in rows.processes)
        target_pids = tuple(str(row["pid"]) for row in target_rows)
        if len(set(target_pids)) != len(target_pids):
            raise ValidationError("checkpoint fork process rows contain duplicate ids")
        with self.transaction() as cursor:
            for target in target_rows:
                quarantined = _checkpoint_fork_quarantine_process_row(target)
                if quarantined == target:
                    continue
                if (
                    target.get("revision") != 0
                    or target.get("state_generation") != 0
                    or target.get("execution_generation") != 0
                    or target.get("execution_owner_id") is not None
                    or target.get("execution_lease_id") is not None
                ):
                    raise ValidationError(
                        "checkpoint fork publication requires fresh process identity"
                    )
                updated = cursor.execute(
                    "UPDATE processes SET status = ?, status_message = ?, "
                    "wait_state_json = ?, outcome_json = ? "
                    "WHERE pid = ? AND status = ? AND status_message = ? "
                    "AND wait_state_json = ? AND outcome_json = ? "
                    "AND revision = 0 AND state_generation = 0 "
                    "AND execution_generation = 0 "
                    "AND execution_owner_id IS NULL "
                    "AND execution_lease_id IS NULL",
                    (
                        target["status"],
                        target.get("status_message"),
                        target["wait_state_json"],
                        target["outcome_json"],
                        target["pid"],
                        quarantined["status"],
                        quarantined["status_message"],
                        quarantined["wait_state_json"],
                        quarantined["outcome_json"],
                    ),
                )
                if updated.rowcount != 1:
                    raise ValidationError(
                        "checkpoint fork process left durable quarantine before "
                        f"publication: {target['pid']}"
                    )
                self._snapshot_backend.ensure_process_terminal_cleanup_intent(
                    str(target["pid"]),
                    str(target["status"]),
                )
        return tuple(sorted(target_pids))

    def checkpoint_fork_process_rows_match(
        self,
        rows: SnapshotRows,
        *,
        quarantined: bool = False,
    ) -> bool:
        expected_rows = {
            str(row["pid"]): (
                _checkpoint_fork_quarantine_process_row(row)
                if quarantined
                else dict(row)
            )
            for row in rows.processes
        }
        if len(expected_rows) != len(rows.processes):
            raise ValidationError("checkpoint fork process rows contain duplicate ids")
        current_rows: dict[str, dict[str, Any]] = {}
        with self.locked():
            for batch in _snapshot_value_batches(sorted(expected_rows)):
                placeholders = ", ".join("?" for _ in batch)
                for row in self._select_rows(
                    "processes",
                    f"pid IN ({placeholders})",
                    batch,
                    order_by="pid",
                ):
                    current_rows[str(row["pid"])] = row
        return current_rows == expected_rows

    def _delete_checkpoint_scope(
        self,
        cursor: Any,
        *,
        pids: list[str],
        current_object_oids: set[str],
        object_oids: set[str],
        namespace_names: set[str],
        snapshot_object_oids: set[str],
    ) -> None:
        self._invalidate_checkpoint_capability_reservations(
            cursor,
            pids=pids,
            object_oids=object_oids,
        )
        self._delete_checkpoint_object_links(cursor, object_oids)
        self._delete_rows_by_values(cursor, "objects", "oid", object_oids)
        self._delete_checkpoint_object_capabilities(
            cursor,
            current_object_oids - snapshot_object_oids,
        )
        for oid in object_oids:
            self._snapshot_backend.forget_object_payload(oid)
        self._delete_rows_by_values(
            cursor,
            "object_namespaces",
            "namespace",
            namespace_names,
        )
        self._delete_non_checkpoint_capabilities(cursor, pids)
        self._delete_checkpoint_resource_reservations(cursor, pids)
        for table in ("llm_pending_actions", "tool_candidates", "processes"):
            self._delete_rows_by_values(
                cursor,
                table,
                "pid",
                pids,
            )

    def _insert_checkpoint_memory_rows(
        self,
        cursor: Any,
        *,
        snapshot: ProcessSnapshot,
        snapshot_object_oids: set[str],
        snapshot_namespaces: set[str],
        before_insert: Callable[[object, str, Mapping[str, Any]], None] | None,
    ) -> None:
        for row in snapshot.rows.object_namespaces:
            if str(row.get("namespace")) in snapshot_namespaces:
                self._insert_backend_row(
                    cursor,
                    "object_namespaces",
                    row,
                    before_insert=before_insert,
                )
        for row in snapshot.rows.objects:
            item = dict(row)
            oid = str(item["oid"])
            if oid not in snapshot_object_oids:
                continue
            if oid in snapshot.object_payloads:
                item["payload_json"] = dumps(snapshot.object_payloads[oid])
            else:
                item["payload_json"] = dumps(
                    self._snapshot_backend.payload_marker(present=False)
                )
            self._insert_backend_row(
                cursor,
                "objects",
                item,
                before_insert=before_insert,
            )
            if oid in snapshot.object_payloads:
                self._snapshot_backend.set_object_payload(
                    oid,
                    deepcopy(snapshot.object_payloads[oid]),
                )
        for row in snapshot.rows.object_links:
            if (
                str(row.get("src_oid")) in snapshot_object_oids
                or str(row.get("dst_oid")) in snapshot_object_oids
            ):
                self._insert_backend_row(
                    cursor,
                    "object_links",
                    row,
                    before_insert=before_insert,
                )

    def _insert_checkpoint_process_rows(
        self,
        cursor: Any,
        *,
        rows: SnapshotRows,
        restored_process_rows: Iterable[Mapping[str, Any]],
        restored_capability_rows: Iterable[Mapping[str, Any]],
        before_insert: Callable[[object, str, Mapping[str, Any]], None] | None,
    ) -> None:
        for table, selected_rows in (
            ("capabilities", restored_capability_rows),
            (
                "process_resource_reservations",
                rows.process_resource_reservations,
            ),
            ("tool_candidates", rows.tool_candidates),
        ):
            for row in selected_rows:
                self._insert_backend_row(
                    cursor,
                    table,
                    row,
                    before_insert=before_insert,
                )
        for row in rows.tools:
            if cursor.execute(
                "SELECT 1 FROM tools WHERE tool_id = ?",
                (row["tool_id"],),
            ).fetchone() is None:
                self._insert_backend_row(
                    cursor,
                    "tools",
                    row,
                    before_insert=before_insert,
                )
        for table, selected_rows, key in (
            ("process_messages", rows.process_messages, "message_id"),
            ("llm_pending_actions", rows.llm_pending_actions, "pid"),
        ):
            for row in selected_rows:
                self._upsert_backend_row(
                    cursor,
                    table,
                    row,
                    key=key,
                    before_insert=before_insert,
                )
        for row in restored_process_rows:
            self._insert_backend_row(
                cursor,
                "processes",
                row,
                before_insert=before_insert,
            )
            self._snapshot_backend.ensure_process_terminal_cleanup_intent(
                str(row["pid"]),
                str(row["status"]),
            )
            self._snapshot_backend.set_llm_context_generation(
                str(row["pid"]),
                new_id("llmctx"),
            )

    def reconcile_restored_object_task_results(
        self,
        snapshot: ProcessSnapshot,
        checkpoint: Checkpoint,
    ) -> list[str]:
        snapshot_pids = set(snapshot.subtree_pids)
        snapshot_oids = set(snapshot.owned_object_oids)
        restored: list[str] = []
        now = utc_now()
        with self.transaction() as cursor:
            for task in self._iter_checkpoint_object_tasks(
                statuses=(ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN,),
                checkpoint_created_at=checkpoint.created_at,
                terminal_after=False,
                pids=snapshot_pids,
                object_oids=snapshot_oids,
            ):
                if task.wait.get("previous_status") != ObjectTaskStatus.SUCCEEDED.value:
                    continue
                previous_result_oid = task.wait.get("previous_result_oid")
                if (
                    not isinstance(previous_result_oid, str)
                    or previous_result_oid not in snapshot_oids
                ):
                    continue
                if self._snapshot_backend.get_object(previous_result_oid) is None:
                    continue
                wait = {
                    **task.wait,
                    "result_restored_by_checkpoint": checkpoint.checkpoint_id,
                    "result_restored_at": now,
                }
                cursor.execute(
                    "UPDATE object_tasks SET status = ?, result_oid = ?, "
                    "error = ?, wait_json = ?, updated_at = ? WHERE task_id = ?",
                    (
                        ObjectTaskStatus.SUCCEEDED.value,
                        previous_result_oid,
                        task.wait.get("previous_error"),
                        dumps(wait),
                        now,
                        task.task_id,
                    ),
                )
                restored.append(str(task.task_id))
        return sorted(restored)

    def insert_checkpoint_fork_rows(
        self,
        rows: SnapshotRows,
        *,
        object_payloads: Mapping[str, Any],
        before_insert: Callable[[object, str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        with self.transaction(include_object_payloads=True) as cursor:
            for row in rows.object_namespaces:
                if cursor.execute(
                    "SELECT 1 FROM object_namespaces WHERE namespace = ?",
                    (row["namespace"],),
                ).fetchone() is None:
                    self._insert_backend_row(
                        cursor,
                        "object_namespaces",
                        row,
                        before_insert=before_insert,
                    )
            for row in rows.objects:
                item = dict(row)
                oid = str(item["oid"])
                item["payload_json"] = dumps(object_payloads[oid])
                self._insert_backend_row(
                    cursor,
                    "objects",
                    item,
                    before_insert=before_insert,
                )
                self._snapshot_backend.set_object_payload(
                    oid,
                    deepcopy(object_payloads[oid]),
                )
            for table, selected_rows in (
                ("object_links", rows.object_links),
                ("capabilities", rows.capabilities),
                ("process_resource_reservations", rows.process_resource_reservations),
                ("process_messages", rows.process_messages),
                ("llm_pending_actions", rows.llm_pending_actions),
                ("tool_candidates", rows.tool_candidates),
            ):
                for row in selected_rows:
                    self._insert_backend_row(
                        cursor,
                        table,
                        row,
                        before_insert=before_insert,
                    )
            for row in rows.tools:
                if cursor.execute(
                    "SELECT 1 FROM tools WHERE tool_id = ?",
                    (row["tool_id"],),
                ).fetchone() is None:
                    self._insert_backend_row(
                        cursor,
                        "tools",
                        row,
                        before_insert=before_insert,
                    )
            for row in rows.processes:
                published = _checkpoint_fork_quarantine_process_row(row)
                self._insert_backend_row(
                    cursor,
                    "processes",
                    published,
                    before_insert=before_insert,
                )
                self._snapshot_backend.ensure_process_terminal_cleanup_intent(
                    str(published["pid"]),
                    str(published["status"]),
                )

    def tool_id_used_outside_scope(
        self,
        tool_id: str,
        *,
        scoped_pids: Iterable[str],
    ) -> bool:
        return self._snapshot_backend.tool_id_referenced_outside_scope(
            tool_id,
            scoped_pids=scoped_pids,
        )

    def get_jit_rehydration_artifacts(
        self,
        *,
        pid: str,
        tool_ids: Iterable[str],
    ) -> tuple[JITRehydrationArtifact, ...]:
        return self._snapshot_backend.get_jit_rehydration_artifacts(
            pid=pid,
            tool_ids=tool_ids,
        )

    def registered_jit_tool_ids_for_processes(
        self,
        pids: Iterable[str],
    ) -> frozenset[str]:
        selected: set[str] = set()
        for batch in _snapshot_value_batches(pids):
            placeholders = ", ".join("?" for _ in batch)
            rows = self._select_rows(
                "tool_candidates",
                f"pid IN ({placeholders}) "
                "AND registered_tool_id IS NOT NULL",
                batch,
                order_by="pid, registered_tool_id, candidate_id, status",
            )
            selected.update(
                str(row["registered_tool_id"])
                for row in rows
                if row.get("registered_tool_id")
            )
        return frozenset(selected)

    def delete_tool_if_unreferenced(
        self,
        tool_id: str,
        *,
        excluding_pid: str,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._snapshot_backend,
            self.transaction,
            lambda: self._snapshot_backend.delete_tool_if_unreferenced(
                tool_id,
                excluding_pid=excluding_pid,
            ),
            operation="delete unreferenced tool",
        )

    def _select_rows(
        self,
        table: str,
        where_sql: str = "",
        params: Iterable[Any] = (),
        *,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._snapshot_backend.select_table_rows(
                table,
                where_sql,
                params,
                order_by=order_by,
            ),
        )

    def _rows_by_values(
        self,
        table: str,
        column: str,
        values: Iterable[str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for batch in _snapshot_value_batches(values):
            placeholders = ", ".join("?" for _ in batch)
            rows.extend(
                self._select_rows(
                    table,
                    f"{column} IN ({placeholders})",
                    batch,
                    order_by=column,
                )
            )
        rows.sort(key=lambda row: str(row[column]))
        return rows

    def _insert_backend_row(
        self,
        cursor: Any,
        table: str,
        row: Mapping[str, Any],
        *,
        before_insert: Callable[[object, str, Mapping[str, Any]], None] | None,
    ) -> None:
        item = dict(row)
        if before_insert is not None:
            before_insert(cursor, table, item)
        # Snapshot values retain domain-native booleans so their JSON form is
        # unambiguous.  The shared SQL schema deliberately stores capability
        # flags as INTEGER for SQLite/PostgreSQL parity, however, and psycopg
        # correctly refuses to bind a Python bool to an INTEGER column.  Keep
        # the observer boundary semantic, then normalize only at persistence.
        if table == "capabilities":
            for column in ("delegable", "revocable"):
                if type(item.get(column)) is bool:
                    item[column] = int(item[column])
        self._snapshot_backend.insert_table_row(table, item)

    def _upsert_backend_row(
        self,
        cursor: Any,
        table: str,
        row: Mapping[str, Any],
        *,
        key: str,
        before_insert: Callable[[object, str, Mapping[str, Any]], None] | None,
    ) -> None:
        item = dict(row)
        if before_insert is not None:
            before_insert(cursor, table, item)
        selected_table = self._snapshot_backend.validate_table_identifier(table)
        selected_key = self._snapshot_backend.validate_column_identifier(
            selected_table,
            key,
        )
        columns = list(item)
        for column in columns:
            self._snapshot_backend.validate_column_identifier(
                selected_table,
                column,
            )
        assignments = ", ".join(
            f"{column} = excluded.{column}"
            for column in columns
            if column != selected_key
        )
        placeholders = ", ".join("?" for _ in columns)
        cursor.execute(
            f"INSERT INTO {selected_table} ({', '.join(columns)}) "
            f"VALUES ({placeholders}) ON CONFLICT({selected_key}) "
            f"DO UPDATE SET {assignments}",
            tuple(item[column] for column in columns),
        )

    def _delete_rows_by_values(
        self,
        cursor: Any,
        table: str,
        column: str,
        values: Iterable[str],
    ) -> None:
        selected_table = self._snapshot_backend.validate_table_identifier(table)
        selected_column = self._snapshot_backend.validate_column_identifier(
            selected_table,
            column,
        )
        for batch in _snapshot_value_batches(values):
            placeholders = ", ".join("?" for _ in batch)
            cursor.execute(
                f"DELETE FROM {selected_table} "
                f"WHERE {selected_column} IN ({placeholders})",
                batch,
            )

    @staticmethod
    def _delete_checkpoint_object_links(
        cursor: Any,
        object_oids: set[str],
    ) -> None:
        for batch in _snapshot_value_batches(sorted(object_oids)):
            placeholders = ", ".join("?" for _ in batch)
            cursor.execute(
                f"DELETE FROM object_links WHERE src_oid IN ({placeholders}) "
                f"OR dst_oid IN ({placeholders})",
                (*batch, *batch),
            )

    @staticmethod
    def _delete_checkpoint_object_capabilities(
        cursor: Any,
        object_oids: set[str],
    ) -> None:
        resources = [f"object:{oid}" for oid in sorted(object_oids)]
        for batch in _snapshot_value_batches(resources):
            placeholders = ", ".join("?" for _ in batch)
            cursor.execute(
                f"DELETE FROM capabilities WHERE resource IN ({placeholders})",
                batch,
            )

    @staticmethod
    def _invalidate_checkpoint_capability_reservations(
        cursor: Any,
        *,
        pids: list[str],
        object_oids: set[str],
    ) -> None:
        resources = [f"object:{oid}" for oid in sorted(object_oids)]
        updated_at = utc_now()
        for column, values in (("subject", pids), ("resource", resources)):
            for batch in _snapshot_value_batches(values):
                placeholders = ", ".join("?" for _ in batch)
                cursor.execute(
                    "UPDATE capability_use_reservations "
                    "SET status = ?, updated_at = ? "
                    "WHERE status = ? AND cap_id IN "
                    "(SELECT cap_id FROM capabilities "
                    f"WHERE {column} IN ({placeholders}))",
                    ("invalidated", updated_at, "reserved", *batch),
                )

    @staticmethod
    def _delete_non_checkpoint_capabilities(
        cursor: Any,
        pids: list[str],
    ) -> None:
        for batch in _snapshot_value_batches(pids):
            placeholders = ", ".join("?" for _ in batch)
            cursor.execute(
                f"DELETE FROM capabilities WHERE subject IN ({placeholders}) "
                "AND resource NOT LIKE 'checkpoint:%'",
                batch,
            )

    @staticmethod
    def _delete_checkpoint_resource_reservations(
        cursor: Any,
        pids: list[str],
    ) -> None:
        for batch in _snapshot_value_batches(pids):
            placeholders = ", ".join("?" for _ in batch)
            cursor.execute(
                "DELETE FROM process_resource_reservations "
                f"WHERE parent_pid IN ({placeholders}) "
                f"OR child_pid IN ({placeholders})",
                (*batch, *batch),
            )

    def _owned_object_rows(self, pid: str) -> list[dict[str, Any]]:
        return self._select_rows(
            "objects",
            "owner_kind = ? AND owner_id = ? AND lifecycle_state = ?",
            (ObjectOwnerKind.PROCESS.value, pid, "live"),
            order_by="oid",
        )

    def _owned_namespace_rows(
        self,
        pid: str,
        process_namespace: str,
    ) -> list[dict[str, Any]]:
        return self._select_rows(
            "object_namespaces",
            "created_by = ? OR namespace = ?",
            (pid, process_namespace),
            order_by="namespace",
        )

    def _object_link_rows(self, object_oids: Iterable[str]) -> list[dict[str, Any]]:
        rows_by_id: dict[str, dict[str, Any]] = {}
        for batch in _snapshot_value_batches(object_oids):
            placeholders = ", ".join("?" for _ in batch)
            for row in self._select_rows(
                "object_links",
                f"src_oid IN ({placeholders}) OR dst_oid IN ({placeholders})",
                (*batch, *batch),
                order_by="id",
            ):
                rows_by_id[str(row["id"])] = row
        return [rows_by_id[row_id] for row_id in sorted(rows_by_id)]

    def _externally_borrowed_oids(self, pid: str, oids: set[str]) -> set[str]:
        resources = [f"object:{oid}" for oid in sorted(oids)]
        rows: list[dict[str, Any]] = []
        for batch in _snapshot_value_batches(resources):
            placeholders = ", ".join("?" for _ in batch)
            rows.extend(
                self._select_rows(
                    "capabilities",
                    f"resource IN ({placeholders}) "
                    "AND subject <> ? AND status = ?",
                    (*batch, pid, CapabilityStatus.ACTIVE.value),
                )
            )
        return {
            str(row["resource"]).split(":", 1)[1]
            for row in rows
            if str(row.get("resource", "")).startswith("object:")
        }

    def _delete_process_exec_scope(
        self,
        cursor: Any,
        *,
        pid: str,
        object_oids: list[str],
        namespace_names: list[str],
    ) -> None:
        if object_oids:
            self._delete_checkpoint_object_links(cursor, set(object_oids))
            self._delete_rows_by_values(cursor, "objects", "oid", object_oids)
            for oid in object_oids:
                self._snapshot_backend.forget_object_payload(oid)
        if namespace_names:
            self._delete_rows_by_values(
                cursor,
                "object_namespaces",
                "namespace",
                namespace_names,
            )
        cursor.execute("DELETE FROM llm_pending_actions WHERE pid = ?", (pid,))
        cursor.execute("DELETE FROM tool_candidates WHERE pid = ?", (pid,))

    def _insert_process_exec_snapshot_rows(self, snapshot: ProcessSnapshot) -> None:
        for row in snapshot.rows.object_namespaces:
            self._snapshot_backend.insert_table_row("object_namespaces", dict(row))
        for row in snapshot.rows.objects:
            item = dict(row)
            oid = str(item["oid"])
            if oid in snapshot.object_payloads:
                item["payload_json"] = dumps(snapshot.object_payloads[oid])
            else:
                item["payload_json"] = dumps(
                    self._snapshot_backend.payload_marker(present=False)
                )
            self._snapshot_backend.insert_table_row("objects", item)
            if oid in snapshot.object_payloads:
                self._snapshot_backend.set_object_payload(
                    oid,
                    deepcopy(snapshot.object_payloads[oid]),
                )
        for row in snapshot.rows.object_links:
            self._snapshot_backend.insert_table_row("object_links", dict(row))
        for row in snapshot.rows.llm_pending_actions:
            self._snapshot_backend.insert_table_row("llm_pending_actions", dict(row))
        for row in snapshot.rows.tool_candidates:
            self._snapshot_backend.insert_table_row("tool_candidates", dict(row))

    def _remove_exec_created_capabilities(
        self,
        cursor: Any,
        *,
        pid: str,
        before_rows: list[dict[str, Any]],
        cleanup_object_oids: set[str],
        rollback_token: str | None,
    ) -> None:
        before_ids = {str(row["cap_id"]) for row in before_rows}
        current_rows = list(
            cursor.execute("SELECT * FROM capabilities WHERE subject = ?", (pid,))
        )
        cleanup_resources = {f"object:{oid}" for oid in cleanup_object_oids}
        for row in current_rows:
            cap_id = str(row["cap_id"])
            if (
                cap_id in before_ids
                or str(row["status"]) != CapabilityStatus.ACTIVE.value
            ):
                continue
            metadata = loads(row["metadata_json"], {})
            rollback_owned = (
                rollback_token is not None
                and metadata.get("runtime_publication_id") == rollback_token
            ) or str(row["resource"]) in cleanup_resources
            if not rollback_owned:
                continue
            cursor.execute(
                "DELETE FROM capabilities WHERE cap_id = ? AND subject = ? AND status = ?",
                (cap_id, pid, CapabilityStatus.ACTIVE.value),
            )

    @staticmethod
    def _restore_exec_revoked_capabilities(
        cursor: Any,
        *,
        pid: str,
        before_rows: list[dict[str, Any]],
        rollback_token: str | None,
    ) -> None:
        if not rollback_token:
            return
        before_by_id = {str(row["cap_id"]): row for row in before_rows}
        current_rows = list(
            cursor.execute(
                "SELECT cap_id, subject, status, metadata_json "
                "FROM capabilities WHERE subject = ?",
                (pid,),
            )
        )
        for current in current_rows:
            if str(current["status"]) != CapabilityStatus.EXEC_REVOKED.value:
                continue
            metadata = loads(current["metadata_json"], {})
            if metadata.get(EXEC_ROLLBACK_TOKEN_KEY) != rollback_token:
                continue
            before = before_by_id.get(str(current["cap_id"]))
            if before is None:
                continue
            cursor.execute(
                """
                UPDATE capabilities
                   SET status = ?, metadata_json = ?
                 WHERE cap_id = ? AND subject = ? AND status = ? AND metadata_json = ?
                """,
                (
                    str(before["status"]),
                    str(before["metadata_json"]),
                    str(current["cap_id"]),
                    pid,
                    CapabilityStatus.EXEC_REVOKED.value,
                    str(current["metadata_json"]),
                ),
            )


class EvidenceRepository(_RepositoryFacade):
    """Evidence, operations, and the publication reads that own their outcome."""

    _METHODS = frozenset(
        {
            "insert_event",
            "get_event",
            "insert_audit",
            "list_audit",
            "get_audit",
            "insert_context_materialization_manifest",
            "get_context_materialization_manifest",
            "list_context_materialization_manifests",
            "insert_external_effect",
            "finalize_external_effect",
            "transition_external_effect",
            "settle_external_effect_recovery",
            "list_task_run_external_effects",
            "redact_task_run_external_effect_payload",
            "abandon_external_effect_intent",
            "list_external_effects",
            "query_external_effect_recovery",
            "get_external_effect_by_idempotency",
        }
    )

    def __init__(self, store: OperationEvidenceBackendProtocol) -> None:
        super().__init__(store)
        self._operation_backend = store
        self._publication_backend = cast(RuntimePublicationBackendProtocol, store)
        self._task_run_external_effect_backend = cast(
            TaskRunExternalEffectRedactionStore,
            store,
        )

    def list_events(
        self,
        target: str | None = None,
        limit: int | None = None,
        before_event_id: str | None = None,
        after_event_id: str | None = None,
        *,
        include_gui_presentation: bool = True,
    ) -> list[Event]:
        filters: dict[str, Any] = {
            "target": target,
            "limit": limit,
            "before_event_id": before_event_id,
            "after_event_id": after_event_id,
        }
        # Preserve the pre-facade backend call shape for ordinary event reads.
        # The presentation filter is an opt-in narrowing flag; forwarding its
        # default explicitly would break compatible backend adapters that
        # implement the original list_events signature.
        if not include_gui_presentation:
            filters["include_gui_presentation"] = False
        return self._operation_backend.list_events(**filters)

    def stale_operation_recovery_index(self) -> AbstractContextManager[None]:
        return self._operation_backend.stale_operation_recovery_index()

    def operation_has_unknown_external_effect(self, operation_id: str) -> bool:
        return self._operation_backend.operation_has_unknown_external_effect(
            operation_id
        )

    def scan_stale_running_operations(
        self,
        *,
        after: OperationCursor | None,
        limit: int,
    ) -> OperationPage:
        return self._operation_backend.scan_stale_running_operations(
            after=after,
            limit=limit,
        )

    def operation_ids_with_unknown_external_effects(
        self,
        operation_ids: Iterable[str],
    ) -> set[str]:
        return self._operation_backend.operation_ids_with_unknown_external_effects(
            operation_ids
        )

    def insert_operation(self, record: OperationRecord) -> None:
        self._operation_backend.insert_operation(record)

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        return self._operation_backend.get_operation(operation_id)

    def list_operations(
        self,
        *,
        pid: str | None = None,
        root_operation_id: str | None = None,
        roots_only: bool = False,
        state: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[OperationRecord]:
        return self._operation_backend.list_operations(
            pid=pid,
            root_operation_id=root_operation_id,
            roots_only=roots_only,
            state=state,
            limit=limit,
            cursor=cursor,
        )

    def update_operation(
        self,
        record: OperationRecord,
        *,
        expected_states: Iterable[str] | None = None,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._operation_backend,
            self.transaction,
            lambda: self._operation_backend.update_operation(
                record,
                expected_states=expected_states,
            ),
            operation="update operation",
        )

    def insert_operation_evidence(self, link: OperationEvidenceLink) -> bool:
        return _transactional_backend_cas_result(
            self._operation_backend,
            self.transaction,
            lambda: self._operation_backend.insert_operation_evidence(link),
            operation="insert operation evidence",
        )

    def list_operation_evidence(
        self,
        *,
        operation_ids: Iterable[str] | None = None,
        evidence_types: Iterable[str] | None = None,
        evidence_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[OperationEvidenceLink]:
        return self._operation_backend.list_operation_evidence(
            operation_ids=operation_ids,
            evidence_types=evidence_types,
            evidence_id=evidence_id,
            limit=limit,
            cursor=cursor,
        )

    def get_external_effect(self, effect_id: str) -> ExternalEffectRecord | None:
        return self._operation_backend.get_external_effect(effect_id)

    def current_effect_ledger_seq(self) -> int:
        return self._operation_backend.current_effect_ledger_seq()

    def external_effect_transition_matches(
        self,
        transition_seq: int,
        effect_id: str,
        *,
        effect_state: str,
        transaction_state: str,
    ) -> bool:
        return self._operation_backend.external_effect_transition_matches(
            transition_seq,
            effect_id,
            effect_state=effect_state,
            transaction_state=transaction_state,
        )

    def list_external_effects_changed_after(
        self,
        effect_ledger_seq: int,
        *,
        pids: Iterable[str] | None = None,
    ) -> list[ExternalEffectRecord]:
        return self._operation_backend.list_external_effects_changed_after(
            effect_ledger_seq,
            pids=pids,
        )

    def list_task_run_external_effects(
        self,
        run_id: str,
        linked_pids: Iterable[str] = (),
    ) -> list[ExternalEffectRecord]:
        return self._task_run_external_effect_backend.list_task_run_external_effects(
            run_id,
            linked_pids,
        )

    def redact_task_run_external_effect_payload(
        self,
        record: ExternalEffectRecord,
        *,
        run_id: str,
        expected_payload_sha256: str,
        expected_tier: str,
        expected_effect_state: str,
        expected_transaction_state: str,
    ) -> bool:
        return (
            self._task_run_external_effect_backend.redact_task_run_external_effect_payload(
                record,
                run_id=run_id,
                expected_payload_sha256=expected_payload_sha256,
                expected_tier=expected_tier,
                expected_effect_state=expected_effect_state,
                expected_transaction_state=expected_transaction_state,
            )
        )

    def list_operation_ids_by_runtime_publication_id(
        self,
        publication_id: str,
    ) -> list[str]:
        return self._operation_backend.list_operation_ids_by_runtime_publication_id(
            publication_id
        )

    # Compatibility for standalone OperationManager construction. Runtime
    # composition injects UnitOfWork.publications directly.
    def get_runtime_publication(
        self,
        publication_id: str,
    ) -> RuntimePublicationRecord | None:
        return cast(
            RuntimePublicationRecord | None,
            self._publication_backend.get_runtime_publication(publication_id),
        )


class ExtensionRepository(_RepositoryFacade):
    """Tools, Skills, providers, images, modules, and checkpoints."""

    _METHODS = frozenset(
        {
            "insert_tool",
            "update_tool",
            "insert_tool_candidate",
            "update_tool_candidate",
            "upsert_skill",
            "get_skill",
            "list_skills",
            "insert_skill_trust",
            "delete_skill_trust",
            "is_skill_trusted",
            "list_skill_trust",
            "upsert_jsonrpc_endpoint",
            "get_jsonrpc_registry_binding",
            "get_jsonrpc_endpoint",
            "list_jsonrpc_endpoints",
            "delete_jsonrpc_endpoint",
            "upsert_mcp_server",
            "get_mcp_registry_binding",
            "get_mcp_server",
            "list_mcp_servers",
            "delete_mcp_server",
            "get_image",
            "list_images",
            "delete_image",
            "list_image_artifacts",
        }
    )

    def __init__(self, backend: TransactionBackendProtocol) -> None:
        super().__init__(backend)
        self._tool_artifact_backend = cast(ToolArtifactRepositoryProtocol, backend)
        self._checkpoint_backend = cast(SnapshotCheckpointBackendProtocol, backend)

    def delete_tool(
        self,
        tool_id: str,
        *,
        registered_by: str | None = None,
    ) -> None:
        self._checkpoint_backend.delete_tool(tool_id, registered_by=registered_by)

    def list_tools(self, limit: int | None = None) -> list[dict[str, Any]]:
        return self._checkpoint_backend.list_tools(limit=limit)

    def upsert_image(
        self,
        image: AgentImage,
        *,
        registered_by: str,
        source: str | None,
        created_at: str,
    ) -> None:
        self._checkpoint_backend.upsert_image(
            image,
            registered_by=registered_by,
            source=source,
            created_at=created_at,
        )

    def insert_image_artifact(
        self,
        *,
        artifact_id: str,
        kind: str,
        artifact: dict[str, Any],
        sha256: str,
        created_by: str,
        created_at: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._checkpoint_backend.insert_image_artifact(
            artifact_id=artifact_id,
            kind=kind,
            artifact=artifact,
            sha256=sha256,
            created_by=created_by,
            created_at=created_at,
            metadata=metadata,
        )

    def get_image_artifact(
        self,
        artifact_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        return self._checkpoint_backend.get_image_artifact(artifact_id)

    def get_existing_tool_ids(
        self,
        tool_ids: Iterable[str],
    ) -> frozenset[str]:
        return self._tool_artifact_backend.get_existing_tool_ids(tool_ids)

    def get_tool_spec(self, tool_id: str) -> ToolSpec | None:
        return self._tool_artifact_backend.get_tool_spec(tool_id)

    def get_jit_rehydration_artifacts(
        self,
        *,
        pid: str,
        tool_ids: Iterable[str],
    ) -> tuple[JITRehydrationArtifact, ...]:
        return self._tool_artifact_backend.get_jit_rehydration_artifacts(
            pid=pid,
            tool_ids=tool_ids,
        )

    def get_jit_rehydration_artifacts_for_tool_ids(
        self,
        tool_ids: Iterable[str],
    ) -> tuple[JITRehydrationArtifact, ...]:
        return self._tool_artifact_backend.get_jit_rehydration_artifacts_for_tool_ids(
            tool_ids
        )

    def get_tool_candidate(self, candidate_id: str) -> ToolCandidate | None:
        return self._tool_artifact_backend.get_tool_candidate(candidate_id)

    def delete_tool_candidate(self, candidate_id: str, pid: str) -> None:
        self._delegate(
            "delete_table_rows",
            "tool_candidates",
            "candidate_id = ? AND pid = ?",
            (candidate_id, pid),
        )

    def delete_jit_tool_rows(self, pid: str, tool_ids: Iterable[str]) -> None:
        self._tool_artifact_backend.delete_jit_tool_rows(pid, tool_ids)

    def list_registered_tool_candidate_rows(self) -> list[dict[str, Any]]:
        """Return durable sources that are eligible for runtime rehydration."""

        return self._delegate(
            "select_table_rows",
            "tool_candidates",
            "status = ?",
            ("registered",),
            order_by="updated_at, candidate_id",
        )

    def list_tool_candidate_rows_for_registration(
        self,
        pid: str,
        tool_id: str,
    ) -> list[dict[str, Any]]:
        return self._delegate(
            "select_table_rows",
            "tool_candidates",
            "pid = ? AND registered_tool_id = ?",
            (pid, tool_id),
            order_by="candidate_id",
        )


class RuntimeModuleRepository(_RepositoryFacade):
    """Explicit typed adapter for durable Runtime Module publications."""

    def __init__(self, backend: RuntimeModuleBackendProtocol) -> None:
        super().__init__(backend)
        self._module_backend = backend

    def upsert_runtime_module(self, module: RuntimeModule) -> RuntimeModule:
        self._module_backend.upsert_runtime_module(
            module_id=module.module_id,
            name=module.name,
            version=module.version,
            entrypoint=module.entrypoint,
            manifest_path=module.manifest_path,
            manifest_sha256=module.manifest_sha256,
            source_path=module.source_path,
            source_sha256=module.source_sha256,
            status=module.status.value,
            loaded_at=module.loaded_at,
            registered=module.registered.to_dict(),
            error=module.error,
            metadata=dict(module.metadata),
        )
        persisted = self.get_runtime_module(module.module_id)
        if persisted is None:
            raise RuntimeError(
                "runtime module publication disappeared after upsert: "
                f"{module.module_id}"
            )
        return persisted

    def get_runtime_module(self, module_id: str) -> RuntimeModule | None:
        value = self._module_backend.get_runtime_module(module_id)
        if value is None:
            return None
        return self._decode_runtime_module(cast(Mapping[str, Any], value))

    def list_runtime_modules(self, limit: int | None = None) -> list[RuntimeModule]:
        values = cast(
            list[Mapping[str, Any]],
            self._module_backend.list_runtime_modules(limit=limit),
        )
        return [self._decode_runtime_module(value) for value in values]

    @staticmethod
    def _decode_runtime_module(value: Mapping[str, Any]) -> RuntimeModule:
        try:
            return RuntimeModule.from_persisted(value)
        except (KeyError, TypeError, ValueError) as exc:
            module_id = value.get("module_id", "<unknown>")
            raise ValidationError(
                f"invalid persisted runtime module: {module_id}"
            ) from exc


class PayloadRetentionRepository(_RepositoryFacade):
    """Typed, bounded maintenance boundary for durable provider payloads."""

    def __init__(self, store: UnitOfWorkBackendProtocol) -> None:
        super().__init__(store)
        self._retention_backend: PayloadRetentionStore = store

    def scan_llm_call_payloads_for_retention(
        self,
        *,
        older_than: str,
        after: PayloadRetentionCursor | None,
        limit: int,
    ) -> PayloadRetentionPage[LLMCallRecord]:
        return self._retention_backend.scan_llm_call_payloads_for_retention(
            older_than=older_than,
            after=after,
            limit=limit,
        )

    def update_llm_call_payload_retention(
        self,
        record: LLMCallRecord,
        *,
        expected_payload_sha256: str,
        expected_tier: PayloadRetentionTier,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._retention_backend,
            self.transaction,
            lambda: self._retention_backend.update_llm_call_payload_retention(
                record,
                expected_payload_sha256=expected_payload_sha256,
                expected_tier=expected_tier,
            ),
            operation="update LLM call payload retention",
        )

    def scan_external_effect_payloads_for_retention(
        self,
        *,
        older_than: str,
        after: PayloadRetentionCursor | None,
        limit: int,
    ) -> PayloadRetentionPage[ExternalEffectRecord]:
        return self._retention_backend.scan_external_effect_payloads_for_retention(
            older_than=older_than,
            after=after,
            limit=limit,
        )

    def update_external_effect_payload_retention(
        self,
        record: ExternalEffectRecord,
        *,
        expected_payload_sha256: str,
        expected_tier: PayloadRetentionTier,
        expected_effect_state: str,
        expected_transaction_state: str,
    ) -> bool:
        return _transactional_backend_cas_result(
            self._retention_backend,
            self.transaction,
            lambda: self._retention_backend.update_external_effect_payload_retention(
                record,
                expected_payload_sha256=expected_payload_sha256,
                expected_tier=expected_tier,
                expected_effect_state=expected_effect_state,
                expected_transaction_state=expected_transaction_state,
            ),
            operation="update external effect payload retention",
        )


class ProtectedEffectRepository:
    """Cross-repository view used by the protected-operation SDK.

    Effect evidence and capability reservations remain in their owning
    repositories while sharing the UnitOfWork transaction boundary.
    """

    def __init__(self, unit_of_work: UnitOfWork) -> None:
        self.__unit_of_work = unit_of_work

    def transaction(self, *, include_object_payloads: bool = False) -> AbstractContextManager[Any]:
        return self.__unit_of_work.transaction(
            include_object_payloads=include_object_payloads
        )

    def insert_external_effect(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.insert_external_effect(*args, **kwargs)

    def get_external_effect(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.get_external_effect(*args, **kwargs)

    def list_external_effects(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.list_external_effects(*args, **kwargs)

    def query_external_effect_recovery(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.query_external_effect_recovery(
            *args,
            **kwargs,
        )

    def get_external_effect_by_idempotency(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.__unit_of_work.evidence.get_external_effect_by_idempotency(
            *args,
            **kwargs,
        )

    def finalize_external_effect(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.finalize_external_effect(*args, **kwargs)

    def transition_external_effect(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.transition_external_effect(*args, **kwargs)

    def settle_external_effect_recovery(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.settle_external_effect_recovery(
            *args,
            **kwargs,
        )

    def abandon_external_effect_intent(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.abandon_external_effect_intent(*args, **kwargs)

    def get_capability_use_reservation(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.authority.get_capability_use_reservation(
            *args,
            **kwargs,
        )

    def list_operation_evidence(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.list_operation_evidence(*args, **kwargs)

    def get_operation(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.evidence.get_operation(*args, **kwargs)

    def get_capability(self, *args: Any, **kwargs: Any) -> Any:
        return self.__unit_of_work.authority.get_capability(*args, **kwargs)


_MIGRATED_BACKEND_PROTOCOLS: tuple[tuple[str, type[Any]], ...] = (
    ("task-run", TaskRunBackendProtocol),
    ("authority-recovery", AuthorityRecoveryBackendProtocol),
    ("object-query", ObjectQueryBackendProtocol),
    ("object-recovery", ObjectRecoveryBackendProtocol),
    ("process", ProcessBackendProtocol),
    ("resource", ResourceBackendProtocol),
    ("runtime-publication", RuntimePublicationBackendProtocol),
    ("checkpoint-publication-writer", CheckpointPublicationWriterBackendProtocol),
    ("snapshot-checkpoint", SnapshotCheckpointBackendProtocol),
    ("runtime-module", RuntimeModuleBackendProtocol),
    ("operation-evidence", OperationEvidenceBackendProtocol),
    ("tool-artifact", ToolArtifactRepositoryProtocol),
    ("payload-retention", PayloadRetentionStore),
)
_MISSING_BACKEND_MEMBER = object()
_SIGNATURE_PROBE = object()


def _protocol_methods(protocol: type[Any]) -> dict[str, Callable[..., Any]]:
    methods: dict[str, Callable[..., Any]] = {}
    for base in reversed(protocol.__mro__):
        for name, value in base.__dict__.items():
            if name.startswith("__") or not inspect.isfunction(value):
                continue
            methods[name] = value
    return methods


def _bound_signature(value: Callable[..., Any], *, unbound: bool) -> inspect.Signature:
    signature = inspect.signature(value)
    parameters = list(signature.parameters.values())
    if unbound and parameters and parameters[0].name in {"self", "cls"}:
        signature = signature.replace(parameters=parameters[1:])
    return signature


def _protocol_call_shapes(
    signature: inspect.Signature,
) -> tuple[tuple[tuple[object, ...], dict[str, object]], ...]:
    shapes: list[tuple[tuple[object, ...], dict[str, object]]] = []
    for include_optional in (False, True):
        for keyword_style in (False, True):
            args: list[object] = []
            kwargs: dict[str, object] = {}
            for parameter in signature.parameters.values():
                if parameter.kind in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }:
                    continue
                if (
                    not include_optional
                    and parameter.default is not inspect.Parameter.empty
                ):
                    continue
                if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                    args.append(_SIGNATURE_PROBE)
                elif (
                    parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                    and not keyword_style
                ):
                    args.append(_SIGNATURE_PROBE)
                else:
                    kwargs[parameter.name] = _SIGNATURE_PROBE
            shape = (tuple(args), kwargs)
            if shape not in shapes:
                shapes.append(shape)
    return tuple(shapes)


def unit_of_work_backend_conformance_errors(
    backend: object | type[Any],
) -> tuple[str, ...]:
    """Return concrete missing/signature errors without trusting ``__getattr__``."""

    errors: list[str] = []
    backend_is_type = inspect.isclass(backend)
    for surface, protocol in _MIGRATED_BACKEND_PROTOCOLS:
        for name, protocol_method in _protocol_methods(protocol).items():
            static_member = inspect.getattr_static(
                backend,
                name,
                _MISSING_BACKEND_MEMBER,
            )
            if static_member is _MISSING_BACKEND_MEMBER:
                errors.append(f"{surface}.{name}: missing concrete method")
                continue
            backend_method = getattr(backend, name, None)
            if not callable(backend_method):
                errors.append(f"{surface}.{name}: concrete member is not callable")
                continue
            try:
                protocol_signature = _bound_signature(
                    protocol_method,
                    unbound=True,
                )
                backend_signature = _bound_signature(
                    backend_method,
                    unbound=backend_is_type,
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"{surface}.{name}: signature unavailable ({exc})")
                continue
            for args, kwargs in _protocol_call_shapes(protocol_signature):
                try:
                    backend_signature.bind(*args, **kwargs)
                except TypeError as exc:
                    errors.append(
                        f"{surface}.{name}: backend {backend_signature} rejects "
                        f"protocol {protocol_signature} ({exc})"
                    )
                    break
    return tuple(errors)


class UnitOfWork:
    """One transaction/lock boundary shared by all domain repositories.

    It does not own or initialize the store.  Existing helpers keep their
    nested-savepoint behavior, so calls through multiple repositories remain
    part of the outer transaction opened here.
    """

    def __init__(self, store: UnitOfWorkBackendProtocol) -> None:
        errors = unit_of_work_backend_conformance_errors(store)
        if errors:
            details = "; ".join(errors)
            raise TypeError(f"UnitOfWork backend contract violation: {details}")
        self.__store = store
        self.task_runs = TaskRunRepository(cast(TaskRunBackendProtocol, store))
        self.processes = ProcessRepository(store)
        self.objects = ObjectRepository(store)
        self.authority = AuthorityRepository(store)
        self.resources = ResourceRepository(store)
        self.publications = RuntimePublicationRepository(store)
        self.checkpoint_restore_publications = CheckpointRestorePublicationWriter(store)
        self.snapshots = SnapshotCheckpointRepository(store)
        self.evidence = EvidenceRepository(store)
        self.extensions = ExtensionRepository(store)
        self.module_publications = RuntimeModuleRepository(store)
        self.retention = PayloadRetentionRepository(store)
        self.protected_effects = ProtectedEffectRepository(self)

    def locked(self) -> AbstractContextManager[None]:
        return self.__store.locked()

    def probe_runtime_assembly_readiness(self) -> StoreAssemblyReadiness:
        return self.__store.probe_runtime_assembly_readiness()

    @contextmanager
    def transaction(self, *, include_object_payloads: bool = False) -> Iterator[UnitOfWork]:
        with self.__store.transaction(include_object_payloads=include_object_payloads):
            yield self
