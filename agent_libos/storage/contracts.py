from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from agent_libos.evidence.payload_retention import PayloadRetentionStore
from agent_libos.storage.base import StoreAssemblyReadiness
from agent_libos.models import (
    AgentObject,
    AgentImage,
    AgentProcess,
    AuditRecord,
    Capability,
    CapabilityUseReservationRecoverySummary,
    Checkpoint,
    CheckpointPayloadDeliveryAttempt,
    CheckpointPayloadDeliveryAttemptPage,
    CheckpointPayloadDeliveryAttemptState,
    Event,
    ExternalEffectRecord,
    ExternalEffectRecoverySettlement,
    HumanRequest,
    HumanRequestStatus,
    JITRehydrationArtifact,
    ObjectHandle,
    ObjectNamespace,
    ObjectNamespaceCursor,
    ObjectNamespacePage,
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
    ProcessMessage,
    ProcessMessageKind,
    ProcessMessageStatus,
    ProcessExecutionToken,
    ProcessOutcome,
    ProcessCursor,
    ProcessPage,
    ProcessRestoreEpoch,
    ProcessToolBindingCursor,
    ProcessToolBindingPage,
    ProcessStatus,
    ProcessWaitState,
    StaleExecutionRecoverySummary,
    ResourceReservation,
    ResourceUsage,
    ResourceUsageReservation,
    ResourceUsageReservationCursor,
    ResourceUsageReservationPage,
    RuntimeModule,
    RuntimePublicationCursor,
    RuntimePublicationKind,
    RuntimePublicationPage,
    PayloadDeliveryState,
    RuntimePublicationRecord,
    RuntimePublicationState,
    ToolCandidate,
    ToolSpec,
    TaskRunCommand,
    TaskRunCursor,
    TaskRunLedgerCursor,
    TaskRunLedgerItem,
    TaskRunLedgerPage,
    TaskRunLink,
    TaskRunPage,
    TaskRunPayload,
    TaskRunRecord,
    TaskRunRequirement,
    TaskRunRequirementStatus,
    TaskRunResumePoint,
    TaskRunStatus,
)
from agent_libos.ports.processes import ProcessRestoreEpochRepositoryPort

if TYPE_CHECKING:
    from agent_libos.models.snapshot import ProcessSnapshot, SnapshotRows


@dataclass(frozen=True, slots=True)
class ProcessScaffoldCleanup:
    """Rows deleted while compensating a partially published process."""

    deleted_by_table: Mapping[str, int]

    @property
    def deleted_rows(self) -> int:
        return sum(self.deleted_by_table.values())


class ProcessStateRepository(ProcessRestoreEpochRepositoryPort, Protocol):
    """Typed persistence boundary for process state and execution fencing."""

    def insert_process(self, process: AgentProcess) -> None: ...

    def get_process(self, pid: str) -> AgentProcess | None: ...

    def list_processes(
        self,
        limit: int | None = None,
        *,
        active_first: bool = False,
    ) -> list[AgentProcess]: ...

    def query_processes(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
    ) -> ProcessPage: ...

    def query_process_tool_bindings(
        self,
        *,
        after: ProcessToolBindingCursor | None,
        limit: int,
    ) -> ProcessToolBindingPage: ...

    def get_processes_with_ancestors(self, pids: Iterable[str]) -> list[AgentProcess]: ...

    def list_processes_by_status(self, status: ProcessStatus | str) -> list[AgentProcess]: ...

    def query_orphaned_created_processes(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
    ) -> ProcessPage: ...

    def list_child_processes(self, parent_pid: str) -> list[AgentProcess]: ...

    def get_human_request(self, request_id: str) -> HumanRequest | None: ...

    def list_human_requests_for_pids(
        self,
        pids: Iterable[str],
        *,
        statuses: Iterable[HumanRequestStatus | str] | None = None,
        limit: int,
        cursor: str | tuple[str, str] | None = None,
    ) -> Mapping[str, Any]: ...

    def get_object_task(self, task_id: str) -> ObjectTask | None: ...

    def list_object_tasks(
        self,
        *,
        owner_oid: str | None = None,
        creator_pid: str | None = None,
        statuses: Iterable[str | ObjectTaskStatus] | None = None,
        include_terminal: bool = True,
        limit: int | None = None,
    ) -> list[ObjectTask]: ...

    def query_object_task_recovery(
        self,
        *,
        kind: ObjectTaskRecoveryKind,
        after: ObjectTaskRecoveryCursor | None,
        limit: int,
    ) -> ObjectTaskRecoveryPage: ...

    def abandon_object_task_after_reopen(
        self,
        task_id: str,
        *,
        expected_status: ObjectTaskStatus,
        reason: str,
        updated_at: str,
    ) -> ObjectTask | None: ...

    def mark_object_task_result_unavailable_after_reopen(
        self,
        task_id: str,
        *,
        expected_result_oid: str,
        wait: Mapping[str, Any],
        error: str,
        updated_at: str,
    ) -> ObjectTask | None: ...

    def patch_process(
        self,
        pid: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
        expected_status: ProcessStatus | str | None = None,
    ) -> AgentProcess: ...

    def patch_process_control(
        self,
        pid: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
        allowed_statuses: Iterable[ProcessStatus | str],
        reason: str,
    ) -> AgentProcess: ...

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
    ) -> AgentProcess: ...

    def append_process_memory_roots(
        self,
        pid: str,
        roots: Iterable[ObjectHandle],
    ) -> AgentProcess: ...

    def remove_process_memory_roots(self, pid: str, oids: Iterable[str]) -> AgentProcess: ...

    def append_process_capability_ids(
        self,
        pid: str,
        capability_ids: Iterable[str],
    ) -> AgentProcess: ...

    def patch_process_tool_tables(
        self,
        pid: str,
        *,
        tool_table: Mapping[str, str] | None = None,
        model_tool_table: Mapping[str, str] | None = None,
    ) -> AgentProcess: ...

    def remove_process_tool_bindings(
        self,
        pid: str,
        bindings: Mapping[str, str],
    ) -> AgentProcess: ...

    def replace_process_for_restore(self, process: AgentProcess) -> None: ...

    def commit_process_exec_epoch(
        self,
        pid: str,
        *,
        publication_id: str,
        expected_revision: int,
    ) -> AgentProcess: ...

    def claim_runnable_process(self, pid: str) -> AgentProcess | None: ...

    def claim_execution(
        self,
        pid: str,
        *,
        owner_id: str,
    ) -> ProcessExecutionToken | None: ...

    def claim_host_process_exec(
        self,
        pid: str,
        *,
        owner_id: str,
        expected_revision: int,
        expected_state_generation: int,
        expected_execution_generation: int,
    ) -> ProcessExecutionToken | None: ...

    def claim_worker_process_exec(
        self,
        pid: str,
        *,
        execution_token: ProcessExecutionToken,
        owner_id: str,
        expected_revision: int,
        expected_state_generation: int,
    ) -> ProcessExecutionToken | None: ...

    def complete_execution(
        self,
        token: ProcessExecutionToken,
        *,
        status: ProcessStatus | str = ProcessStatus.RUNNABLE,
        status_message: str | None = None,
        wait_state: ProcessWaitState | None = None,
        outcome: ProcessOutcome | None = None,
    ) -> bool: ...

    def release_execution(self, token: ProcessExecutionToken) -> bool: ...

    def get_process_terminal_cleanup(self, pid: str) -> dict[str, Any] | None: ...

    def list_process_terminal_cleanups(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
        include_completed: bool = False,
    ) -> list[dict[str, Any]]: ...

    def claim_process_terminal_cleanup(
        self,
        pid: str,
        *,
        owner_id: str,
        lease_id: str,
        expected_lease_id: str | None,
    ) -> dict[str, Any] | None: ...

    def complete_process_terminal_cleanup_phase(
        self,
        pid: str,
        *,
        lease_id: str,
        phase: str,
    ) -> dict[str, Any] | None: ...

    def fail_process_terminal_cleanup(
        self,
        pid: str,
        *,
        lease_id: str,
        phase: str,
        error: Mapping[str, Any],
    ) -> dict[str, Any] | None: ...

    def finish_process_terminal_cleanup(
        self,
        pid: str,
        *,
        lease_id: str,
    ) -> dict[str, Any] | None: ...

    def recover_stale_executions(
        self,
        *,
        owner_id: str,
        require_recovery_lease: Callable[[], None],
        on_recovered: Callable[[str], None],
    ) -> StaleExecutionRecoverySummary: ...

    def tool_id_referenced_outside_process(
        self,
        tool_id: str,
        *,
        excluding_pid: str,
    ) -> bool: ...

    def delete_process_scaffold(
        self,
        pid: str,
        *,
        namespace: str,
        namespace_resource: str,
    ) -> ProcessScaffoldCleanup: ...


class TaskRunBackendProtocol(Protocol):
    """Durable TaskRun current rows, append-only evidence, and epoch fences."""

    @property
    def runtime_epoch(self) -> int | None: ...

    def claim_runtime_epoch(self, instance_id: str) -> int: ...

    def insert_task_run(self, run: TaskRunRecord, *, cursor: Any | None = None) -> None: ...

    def get_task_run(self, run_id: str) -> TaskRunRecord | None: ...

    def list_task_runs(
        self,
        *,
        statuses: Iterable[TaskRunStatus | str] | None = None,
        after: TaskRunCursor | None = None,
        limit: int | None = None,
    ) -> TaskRunPage: ...

    def list_recoverable_task_runs(
        self,
        *,
        after: TaskRunCursor | None,
        limit: int,
    ) -> TaskRunPage: ...

    def update_task_run_cas(
        self,
        run_id: str,
        expected_revision: int,
        *,
        updates: Mapping[str, Any],
        expected_runtime_epoch: int | None = None,
    ) -> TaskRunRecord: ...

    def claim_task_run_epoch(
        self,
        run_id: str,
        expected_revision: int,
        runtime_epoch: int | None = None,
    ) -> TaskRunRecord: ...

    def claim_terminal_task_run_epoch(
        self,
        run_id: str,
        expected_revision: int,
        runtime_epoch: int | None = None,
    ) -> TaskRunRecord: ...

    def insert_task_run_requirement(
        self,
        requirement: TaskRunRequirement,
        *,
        cursor: Any | None = None,
    ) -> None: ...

    def list_task_run_requirements(
        self,
        run_id: str,
        *,
        after: tuple[int, str] | None = None,
        limit: int | None = None,
    ) -> list[TaskRunRequirement]: ...

    def update_task_run_requirement_cas(
        self,
        requirement_id: str,
        *,
        expected_status: TaskRunRequirementStatus | str,
        status: TaskRunRequirementStatus | str,
        updated_at: str,
        started_at: str | None = None,
        completed_at: str | None = None,
        waived_by: str | None = None,
        waiver_reason: str | None = None,
    ) -> TaskRunRequirement | None: ...

    def insert_task_run_payload(
        self,
        payload: TaskRunPayload,
        *,
        cursor: Any | None = None,
    ) -> None: ...

    def get_task_run_payload(self, payload_id: str) -> TaskRunPayload | None: ...

    def list_task_run_payloads(self, run_id: str) -> list[TaskRunPayload]: ...

    def purge_task_run_payloads(self, run_id: str, *, purged_at: str) -> int: ...

    def upsert_task_run_resume_point(self, point: TaskRunResumePoint) -> None: ...

    def get_task_run_resume_point(
        self,
        pid: str,
        *,
        complete_only: bool = True,
    ) -> TaskRunResumePoint | None: ...

    def list_task_run_resume_points(
        self,
        run_id: str,
        *,
        complete_only: bool = True,
        limit: int | None = None,
    ) -> list[TaskRunResumePoint]: ...

    def delete_task_run_resume_point(self, pid: str) -> bool: ...

    def insert_task_run_command(
        self,
        command: TaskRunCommand,
        *,
        cursor: Any | None = None,
        expected_runtime_epoch: int | None = None,
    ) -> TaskRunCommand: ...

    def get_task_run_command(
        self,
        run_id: str,
        command_id: str,
    ) -> TaskRunCommand | None: ...

    def list_task_run_commands(
        self,
        run_id: str,
        *,
        limit: int,
    ) -> list[TaskRunCommand]: ...

    def get_task_run_command_by_client_request_id(
        self,
        client_request_id: str,
    ) -> TaskRunCommand | None: ...

    def update_task_run_command_result(
        self,
        run_id: str,
        command_id: str,
        *,
        expected_result_revision: int,
        result: Mapping[str, Any],
        result_revision: int,
        expected_runtime_epoch: int | None = None,
    ) -> TaskRunCommand: ...

    def append_task_run_ledger_item(
        self,
        item: Any,
        *,
        cursor: Any | None = None,
    ) -> Any: ...

    def get_task_run_ledger_item(
        self,
        run_id: str,
        item_id: str,
    ) -> TaskRunLedgerItem | None: ...

    def list_task_run_ledger(
        self,
        run_id: str,
        *,
        after: TaskRunLedgerCursor | None,
        limit: int,
    ) -> TaskRunLedgerPage: ...

    def insert_task_run_link(
        self,
        link: TaskRunLink,
        *,
        cursor: Any | None = None,
    ) -> None: ...

    def list_task_run_links(
        self,
        run_id: str,
        *,
        limit: int | None = None,
    ) -> list[TaskRunLink]: ...

    def list_processes_for_task_run(self, run_id: str) -> list[AgentProcess]: ...

    def list_task_run_process_ids(self, run_id: str) -> tuple[str, ...]: ...

    def list_active_capability_use_reservations_for_pids(
        self,
        pids: Iterable[str],
    ) -> list[dict[str, Any]]: ...

    def list_human_requests_for_pids(
        self,
        pids: Iterable[str],
        *,
        statuses: Iterable[HumanRequestStatus | str] | None = None,
        limit: int,
        cursor: str | tuple[str, str] | None = None,
    ) -> Mapping[str, Any]: ...

    def list_task_run_human_requests(
        self,
        run_id: str,
        linked_pids: Iterable[str] = (),
    ) -> list[HumanRequest]: ...

    def redact_task_run_human_request_payload(
        self,
        record: HumanRequest,
        *,
        run_id: str,
        expected_payload_sha256: str,
        expected_status: str,
    ) -> bool: ...

    def purge_task_run_messages(
        self,
        run_id: str,
        linked_pids: Iterable[str],
        *,
        purged_at: str,
    ) -> int: ...

    def purge_task_run_llm_pending_actions(
        self,
        run_id: str,
        linked_pids: Iterable[str],
        *,
        purged_at: str,
        cursor: Any | None = None,
    ) -> int: ...

    def list_task_run_external_effects(
        self,
        run_id: str,
        linked_pids: Iterable[str] = (),
    ) -> list[ExternalEffectRecord]: ...

    def redact_task_run_external_effect_payload(
        self,
        record: ExternalEffectRecord,
        *,
        run_id: str,
        expected_payload_sha256: str,
        expected_tier: str,
        expected_effect_state: str,
        expected_transaction_state: str,
    ) -> bool: ...

    def list_active_task_run_ids_for_pids(
        self,
        pids: Iterable[str],
    ) -> tuple[str, ...]: ...


class ResourceRepositoryProtocol(Protocol):
    """Typed persistence boundary for hierarchical resource accounting."""

    def upsert_resource_reservation(self, reservation: ResourceReservation) -> None: ...

    def get_resource_reservation(
        self,
        parent_pid: str,
        child_pid: str,
    ) -> ResourceReservation | None: ...

    def list_resource_reservations(
        self,
        *,
        parent_pid: str | None = None,
        parent_pids: Iterable[str] | None = None,
        child_pid: str | None = None,
    ) -> list[ResourceReservation]: ...

    def delete_resource_reservation(self, parent_pid: str, child_pid: str) -> None: ...

    def delete_resource_reservations_for_process(self, pid: str) -> None: ...

    def insert_resource_usage_reservation(
        self,
        *,
        reservation_id: str,
        pid: str,
        usage: ResourceUsage,
        reserved_by: str,
        reason: str,
        created_at: str,
    ) -> None: ...

    def get_resource_usage_reservation(
        self,
        reservation_id: str,
    ) -> ResourceUsageReservation | None: ...

    def list_resource_usage_reservations(
        self,
        *,
        pid: str | None = None,
        status: str | None = None,
    ) -> list[ResourceUsageReservation]: ...

    def query_resource_usage_reservation_recovery(
        self,
        *,
        after: ResourceUsageReservationCursor | None,
        limit: int,
    ) -> ResourceUsageReservationPage: ...

    def settle_resource_usage_reservation(
        self,
        reservation_id: str,
        *,
        status: str,
        settled_usage: ResourceUsage,
        updated_at: str,
    ) -> bool: ...


class RuntimePublicationRepositoryProtocol(Protocol):
    """Typed persistence boundary for publication state machines."""

    def insert_runtime_publication(
        self,
        *,
        publication_id: str,
        kind: RuntimePublicationKind,
        pid: str,
        owner_instance_id: str,
        plan: Mapping[str, Any],
        phase: str = "planned",
    ) -> RuntimePublicationRecord: ...

    def get_runtime_publication(
        self,
        publication_id: str,
    ) -> RuntimePublicationRecord | None: ...

    def list_runtime_publications(
        self,
        *,
        states: Iterable[RuntimePublicationState | str] | None = None,
        pid: str | None = None,
    ) -> list[RuntimePublicationRecord]: ...

    def get_committed_root_spawn_publication(
        self,
        pid: str,
    ) -> RuntimePublicationRecord | None: ...

    def redact_root_spawn_initial_goal(
        self,
        publication_id: str,
        *,
        pid: str,
        goal_oid: str,
        expected_payload_sha256: str,
        expected_states: Iterable[RuntimePublicationState | str] = ("committed",),
    ) -> bool: ...

    def query_runtime_publication_operation_reconciliation(
        self,
        *,
        kind: RuntimePublicationKind,
        state: RuntimePublicationState | str,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage: ...

    def query_runtime_publication_recovery(
        self,
        *,
        kind: RuntimePublicationKind,
        state: RuntimePublicationState | str,
        operation_reconciled: bool,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage: ...

    def get_checkpoint_payload_delivery_attempt_state(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> CheckpointPayloadDeliveryAttemptState | None: ...

    def mark_runtime_publication_operation_reconciled(
        self,
        publication_id: str,
        *,
        expected_kind: RuntimePublicationKind,
        expected_state: RuntimePublicationState | str,
        expected_phase: str,
        expected_operation_id: str | None,
    ) -> bool: ...

    def runtime_publication_exists_for_pid(
        self,
        pid: str,
        *,
        kind: RuntimePublicationKind,
    ) -> bool: ...

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
    ) -> RuntimePublicationRecord | None: ...

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
        """CAS a transition; ``None`` is unrestricted and empty matches none."""

        ...

    def update_runtime_publication_plan(
        self,
        publication_id: str,
        update: Mapping[str, Any],
        *,
        expected_states: Iterable[RuntimePublicationState | str] | None = None,
    ) -> bool:
        """CAS a plan update; ``None`` is unrestricted and empty matches none."""

        ...

    def record_runtime_publication_artifact(
        self,
        publication_id: str,
        artifact: Mapping[str, Any],
        *,
        expected_states: Iterable[RuntimePublicationState | str] | None = None,
    ) -> bool:
        """CAS a receipt append; ``None`` is unrestricted and empty matches none."""

        ...


class OperationRepositoryProtocol(Protocol):
    """Typed persistence boundary for causal operations and their evidence."""

    def locked(self) -> AbstractContextManager[None]: ...

    def list_events(
        self,
        target: str | None = None,
        limit: int | None = None,
        before_event_id: str | None = None,
        after_event_id: str | None = None,
        *,
        include_gui_presentation: bool = True,
    ) -> list[Event]: ...

    def insert_operation(self, record: OperationRecord) -> None: ...

    def get_operation(self, operation_id: str) -> OperationRecord | None: ...

    def list_operation_ids_by_runtime_publication_id(
        self,
        publication_id: str,
    ) -> list[str]: ...

    def list_operations(
        self,
        *,
        pid: str | None = None,
        root_operation_id: str | None = None,
        roots_only: bool = False,
        state: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[OperationRecord]: ...

    def scan_stale_running_operations(
        self,
        *,
        after: OperationCursor | None,
        limit: int,
    ) -> OperationPage: ...

    def operation_ids_with_unknown_external_effects(
        self,
        operation_ids: Iterable[str],
    ) -> set[str]: ...

    def stale_operation_recovery_index(self) -> AbstractContextManager[None]: ...

    def operation_has_unknown_external_effect(self, operation_id: str) -> bool: ...

    def update_operation(
        self,
        record: OperationRecord,
        *,
        expected_states: Iterable[str] | None = None,
    ) -> bool:
        """CAS an operation update; ``None`` is unrestricted and empty matches none."""

        ...

    def insert_operation_evidence(self, link: OperationEvidenceLink) -> bool: ...

    def list_operation_evidence(
        self,
        *,
        operation_ids: Iterable[str] | None = None,
        evidence_types: Iterable[str] | None = None,
        evidence_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[OperationEvidenceLink]: ...

    def get_external_effect(self, effect_id: str) -> ExternalEffectRecord | None: ...

    def settle_external_effect_recovery(
        self,
        effect_id: str,
        *,
        expected_transaction_state: str,
        provider_state: str,
        provider_metadata: dict[str, Any],
        provider_receipt: dict[str, Any],
        audit_record: AuditRecord,
        updated_at: str,
        run_id: str | None = None,
        runtime_epoch: int | None = None,
    ) -> ExternalEffectRecoverySettlement | None: ...

    def current_effect_ledger_seq(self) -> int: ...

    def external_effect_transition_matches(
        self,
        transition_seq: int,
        effect_id: str,
        *,
        effect_state: str,
        transaction_state: str,
    ) -> bool: ...

    def list_external_effects_changed_after(
        self,
        effect_ledger_seq: int,
        *,
        pids: Iterable[str] | None = None,
    ) -> list[ExternalEffectRecord]: ...


class ToolArtifactRepositoryProtocol(Protocol):
    """Typed exact-identity lookup used by publication compensation."""

    def get_existing_tool_ids(
        self,
        tool_ids: Iterable[str],
    ) -> frozenset[str]: ...

    def get_tool_spec(self, tool_id: str) -> ToolSpec | None: ...

    def get_tool_candidate(self, candidate_id: str) -> ToolCandidate | None: ...

    def get_jit_rehydration_artifacts(
        self,
        *,
        pid: str,
        tool_ids: Iterable[str],
    ) -> tuple[JITRehydrationArtifact, ...]: ...

    def get_jit_rehydration_artifacts_for_tool_ids(
        self,
        tool_ids: Iterable[str],
    ) -> tuple[JITRehydrationArtifact, ...]: ...

    def delete_jit_tool_rows(
        self,
        pid: str,
        tool_ids: Iterable[str],
    ) -> None: ...


class SnapshotCheckpointRepositoryProtocol(Protocol):
    """Typed persistence boundary for snapshot and checkpoint row sets.

    Runtime services exchange canonical ``SnapshotRows``/``ProcessSnapshot``
    values with this boundary.  Backend-specific SQL, generic table helpers,
    and Object-payload cache updates stay behind the repository.
    """

    def transaction(
        self,
        *,
        include_object_payloads: bool = False,
    ) -> AbstractContextManager[Any]: ...

    def capture_process_exec_snapshot_rows(
        self,
        pid: str,
        *,
        process_namespace: str,
    ) -> tuple[SnapshotRows, dict[str, Any]]: ...

    def restore_process_exec_snapshot(
        self,
        snapshot: ProcessSnapshot,
        *,
        process_namespace: str,
        captured_tool_ids: Iterable[str],
        capability_rollback_token: str | None,
        fence_execution: bool = True,
    ) -> frozenset[str]: ...

    def install_checkpoint_image_rows(
        self,
        rows: SnapshotRows,
        *,
        object_payloads: Mapping[str, Any],
    ) -> None: ...

    def load_process_snapshot_rows(self, pids: Iterable[str]) -> SnapshotRows: ...

    def insert_checkpoint(
        self,
        checkpoint: Checkpoint,
        snapshot: ProcessSnapshot,
    ) -> None: ...

    def get_checkpoint_snapshot(
        self,
        checkpoint_id: str,
    ) -> tuple[Checkpoint, ProcessSnapshot] | None: ...

    def list_checkpoints(
        self,
        *,
        pid: str | None = None,
        limit: int | None = None,
    ) -> list[Checkpoint]: ...

    def capture_checkpoint_rows(
        self,
        process_rows: Iterable[Mapping[str, Any]],
        *,
        object_oids: Iterable[str],
        namespace_names: Iterable[str],
    ) -> tuple[SnapshotRows, dict[str, Any]]: ...

    def prepare_checkpoint_restore_process_rows(
        self,
        rows: SnapshotRows,
        *,
        restored_capability_rows: Iterable[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]: ...

    def cancel_pending_human_requests_after_checkpoint(
        self,
        pids: Iterable[str],
        checkpoint: Checkpoint,
    ) -> list[str]: ...

    def supersede_messages_after_checkpoint(
        self,
        pids: Iterable[str],
        checkpoint: Checkpoint,
    ) -> list[str]: ...

    def supersede_object_tasks_after_checkpoint(
        self,
        pids: Iterable[str],
        object_oids: Iterable[str],
        checkpoint: Checkpoint,
    ) -> list[str]: ...

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
    ) -> None: ...

    def reconcile_checkpoint_object_payloads(
        self,
        snapshot: ProcessSnapshot,
    ) -> tuple[str, ...]: ...

    def rehydrate_checkpoint_fork_object_payloads(
        self,
        rows: SnapshotRows,
        *,
        object_payloads: Mapping[str, Any],
    ) -> tuple[str, ...]: ...

    def publish_checkpoint_fork_process_rows(
        self,
        rows: SnapshotRows,
    ) -> tuple[str, ...]: ...

    def checkpoint_fork_process_rows_match(
        self,
        rows: SnapshotRows,
        *,
        quarantined: bool = False,
    ) -> bool: ...

    def reconcile_restored_object_task_results(
        self,
        snapshot: ProcessSnapshot,
        checkpoint: Checkpoint,
    ) -> list[str]: ...

    def insert_checkpoint_fork_rows(
        self,
        rows: SnapshotRows,
        *,
        object_payloads: Mapping[str, Any],
        before_insert: Callable[[object, str, Mapping[str, Any]], None] | None = None,
    ) -> None: ...

    def tool_id_used_outside_scope(
        self,
        tool_id: str,
        *,
        scoped_pids: Iterable[str],
    ) -> bool: ...

    def get_jit_rehydration_artifacts(
        self,
        *,
        pid: str,
        tool_ids: Iterable[str],
    ) -> tuple[JITRehydrationArtifact, ...]: ...

    def registered_jit_tool_ids_for_processes(
        self,
        pids: Iterable[str],
    ) -> frozenset[str]: ...

    def delete_tool_if_unreferenced(
        self,
        tool_id: str,
        *,
        excluding_pid: str,
    ) -> bool: ...


class RuntimeModuleRepositoryProtocol(Protocol):
    """Typed persistence boundary for durable module publication metadata."""

    def transaction(
        self,
        *,
        include_object_payloads: bool = False,
    ) -> AbstractContextManager[Any]: ...

    def upsert_runtime_module(self, module: RuntimeModule) -> RuntimeModule: ...

    def get_runtime_module(self, module_id: str) -> RuntimeModule | None: ...

    def list_runtime_modules(self, limit: int | None = None) -> list[RuntimeModule]: ...


class TransactionBackendProtocol(Protocol):
    """Concrete transaction/lock surface required by typed repositories."""

    def locked(self) -> AbstractContextManager[None]: ...

    def transaction(
        self,
        *,
        include_object_payloads: bool = False,
    ) -> AbstractContextManager[Any]: ...

    def probe_runtime_assembly_readiness(self) -> StoreAssemblyReadiness: ...


class AuthorityRecoveryBackendProtocol(TransactionBackendProtocol, Protocol):
    """Startup-only authority reservation recovery surface."""

    def abandon_stale_capability_use_reservations(
        self,
        *,
        require_recovery_lease: Callable[[], None],
    ) -> CapabilityUseReservationRecoverySummary: ...

    def get_capability(self, cap_id: str) -> Capability | None: ...

    def list_capabilities(
        self,
        subject: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[Capability]: ...

    def query_capabilities(
        self,
        subject: str | None = None,
        *,
        active_only: bool,
        after_cap_id: str | None,
        limit: int,
    ) -> list[Capability]: ...


class ObjectRecoveryBackendProtocol(TransactionBackendProtocol, Protocol):
    """Lifecycle-gated cleanup of volatile Object payload rows."""

    def recover_missing_runtime_object_payloads(
        self,
        *,
        require_recovery_lease: Callable[[], None],
    ) -> ObjectPayloadRecoverySummary: ...

    def get_persisted_object_state(
        self,
        oid: str,
    ) -> PersistedObjectState | None: ...

    def rehydrate_root_spawn_goal_payload(
        self,
        oid: str,
        payload: Any,
        *,
        expected_version: int,
        expected_identity_sha256: str,
        require_recovery_lease: Callable[[], None],
    ) -> bool: ...


class ObjectQueryBackendProtocol(TransactionBackendProtocol, Protocol):
    """Payload-free, hard-bounded Object directory query surface."""

    def query_object_refs_page(
        self,
        namespace: str,
        *,
        after: ObjectRefCursor | None,
        limit: int,
    ) -> ObjectRefPage: ...

    def query_child_namespaces_page(
        self,
        parent_namespace: str,
        *,
        after: ObjectNamespaceCursor | None,
        limit: int,
    ) -> ObjectNamespacePage: ...


class ProcessBackendProtocol(
    TransactionBackendProtocol,
    ProcessRestoreEpochRepositoryPort,
    Protocol,
):
    """SQL backend operations consumed by ``ProcessRepository``."""

    def insert_process(self, process: AgentProcess) -> None: ...

    def get_process(self, pid: str) -> AgentProcess | None: ...

    def list_processes(
        self,
        limit: int | None = None,
        *,
        active_first: bool = False,
    ) -> list[AgentProcess]: ...

    def query_processes(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
    ) -> ProcessPage: ...

    def query_process_tool_bindings(
        self,
        *,
        after: ProcessToolBindingCursor | None,
        limit: int,
    ) -> ProcessToolBindingPage: ...

    def get_processes_with_ancestors(self, pids: Iterable[str]) -> list[AgentProcess]: ...

    def list_processes_by_status(self, status: ProcessStatus | str) -> list[AgentProcess]: ...

    def query_orphaned_created_processes(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
    ) -> ProcessPage: ...

    def list_child_processes(self, parent_pid: str) -> list[AgentProcess]: ...

    def get_human_request(self, request_id: str) -> HumanRequest | None: ...

    def list_human_requests_for_pids(
        self,
        pids: Iterable[str],
        *,
        statuses: Iterable[HumanRequestStatus | str] | None = None,
        limit: int,
        cursor: str | tuple[str, str] | None = None,
    ) -> Mapping[str, Any]: ...

    def list_llm_pending_action_validation_rows(
        self,
        *,
        after_pid: str | None = None,
        limit: int | None = None,
    ) -> Mapping[str, Any]: ...

    def count_process_messages(
        self,
        recipient_pid: str | None = None,
        *,
        status: ProcessMessageStatus | str | None = None,
        statuses: Iterable[ProcessMessageStatus | str] | None = None,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
    ) -> int: ...

    def ack_process_message(
        self,
        message_id: str,
        *,
        recipient_pid: str,
        acked_at: str,
        updated_at: str,
    ) -> bool:
        """CAS one process message from unread to acknowledged."""

        ...

    def get_object_task(self, task_id: str) -> ObjectTask | None: ...

    def list_object_tasks(
        self,
        *,
        owner_oid: str | None = None,
        creator_pid: str | None = None,
        statuses: Iterable[str | ObjectTaskStatus] | None = None,
        include_terminal: bool = True,
        limit: int | None = None,
    ) -> list[ObjectTask]: ...

    def query_object_task_recovery(
        self,
        *,
        kind: ObjectTaskRecoveryKind,
        after: ObjectTaskRecoveryCursor | None,
        limit: int,
    ) -> ObjectTaskRecoveryPage: ...

    def abandon_object_task_after_reopen(
        self,
        task_id: str,
        *,
        expected_status: ObjectTaskStatus,
        reason: str,
        updated_at: str,
    ) -> ObjectTask | None: ...

    def mark_object_task_result_unavailable_after_reopen(
        self,
        task_id: str,
        *,
        expected_result_oid: str,
        wait: Mapping[str, Any],
        error: str,
        updated_at: str,
    ) -> ObjectTask | None: ...

    def patch_process(
        self,
        pid: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
        expected_status: ProcessStatus | str | None = None,
    ) -> AgentProcess: ...

    def patch_process_control(
        self,
        pid: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
        allowed_statuses: Iterable[ProcessStatus | str],
        reason: str,
    ) -> AgentProcess: ...

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
    ) -> AgentProcess: ...

    def append_process_memory_roots(
        self,
        pid: str,
        roots: Iterable[ObjectHandle],
    ) -> AgentProcess: ...

    def remove_process_memory_roots(self, pid: str, oids: Iterable[str]) -> AgentProcess: ...

    def append_process_capability_ids(
        self,
        pid: str,
        capability_ids: Iterable[str],
    ) -> AgentProcess: ...

    def patch_process_tool_tables(
        self,
        pid: str,
        *,
        tool_table: Mapping[str, str] | None = None,
        model_tool_table: Mapping[str, str] | None = None,
    ) -> AgentProcess: ...

    def remove_process_tool_bindings(
        self,
        pid: str,
        bindings: Mapping[str, str],
    ) -> AgentProcess: ...

    def replace_process_for_restore(self, process: AgentProcess) -> None: ...

    def commit_process_exec_epoch(
        self,
        pid: str,
        *,
        publication_id: str,
        expected_revision: int,
    ) -> AgentProcess: ...

    def claim_runnable_process(self, pid: str) -> AgentProcess | None: ...

    def claim_execution(
        self,
        pid: str,
        *,
        owner_id: str,
    ) -> ProcessExecutionToken | None: ...

    def claim_host_process_exec(
        self,
        pid: str,
        *,
        owner_id: str,
        expected_revision: int,
        expected_state_generation: int,
        expected_execution_generation: int,
    ) -> ProcessExecutionToken | None: ...

    def claim_worker_process_exec(
        self,
        pid: str,
        *,
        execution_token: ProcessExecutionToken,
        owner_id: str,
        expected_revision: int,
        expected_state_generation: int,
    ) -> ProcessExecutionToken | None: ...

    def complete_execution(
        self,
        token: ProcessExecutionToken,
        *,
        status: ProcessStatus | str = ProcessStatus.RUNNABLE,
        status_message: str | None = None,
        wait_state: ProcessWaitState | None = None,
        outcome: ProcessOutcome | None = None,
    ) -> bool: ...

    def release_execution(self, token: ProcessExecutionToken) -> bool: ...

    def get_process_terminal_cleanup(self, pid: str) -> dict[str, Any] | None: ...

    def list_process_terminal_cleanups(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
        include_completed: bool = False,
    ) -> list[dict[str, Any]]: ...

    def claim_process_terminal_cleanup(
        self,
        pid: str,
        *,
        owner_id: str,
        lease_id: str,
        expected_lease_id: str | None,
    ) -> dict[str, Any] | None: ...

    def complete_process_terminal_cleanup_phase(
        self,
        pid: str,
        *,
        lease_id: str,
        phase: str,
    ) -> dict[str, Any] | None: ...

    def fail_process_terminal_cleanup(
        self,
        pid: str,
        *,
        lease_id: str,
        phase: str,
        error: Mapping[str, Any],
    ) -> dict[str, Any] | None: ...

    def finish_process_terminal_cleanup(
        self,
        pid: str,
        *,
        lease_id: str,
    ) -> dict[str, Any] | None: ...

    def recover_stale_executions(
        self,
        *,
        owner_id: str,
        require_recovery_lease: Callable[[], None],
        on_recovered: Callable[[str], None],
    ) -> StaleExecutionRecoverySummary: ...

    def tool_id_referenced_outside_process(
        self,
        tool_id: str,
        *,
        excluding_pid: str,
    ) -> bool: ...


class ResourceBackendProtocol(TransactionBackendProtocol, Protocol):
    """Raw SQL backend operations consumed by ``ResourceRepository``."""

    def upsert_resource_reservation(self, reservation: ResourceReservation) -> None: ...

    def get_resource_reservation(
        self,
        parent_pid: str,
        child_pid: str,
    ) -> ResourceReservation | None: ...

    def list_resource_reservations(
        self,
        *,
        parent_pid: str | None = None,
        parent_pids: Iterable[str] | None = None,
        child_pid: str | None = None,
    ) -> list[ResourceReservation]: ...

    def delete_resource_reservation(self, parent_pid: str, child_pid: str) -> None: ...

    def delete_resource_reservations_for_process(self, pid: str) -> None: ...

    def insert_resource_usage_reservation(
        self,
        *,
        reservation_id: str,
        pid: str,
        usage: ResourceUsage,
        reserved_by: str,
        reason: str,
        created_at: str,
    ) -> None: ...

    def get_resource_usage_reservation(
        self,
        reservation_id: str,
    ) -> Mapping[str, Any] | ResourceUsageReservation | None: ...

    def list_resource_usage_reservations(
        self,
        *,
        pid: str | None = None,
        status: str | None = None,
    ) -> list[Mapping[str, Any] | ResourceUsageReservation]: ...

    def query_resource_usage_reservation_recovery(
        self,
        *,
        after: ResourceUsageReservationCursor | None,
        limit: int,
    ) -> ResourceUsageReservationPage: ...

    def settle_resource_usage_reservation(
        self,
        reservation_id: str,
        *,
        status: str,
        settled_usage: ResourceUsage,
        updated_at: str,
    ) -> bool: ...


class RuntimePublicationBackendProtocol(TransactionBackendProtocol, Protocol):
    """Raw publication state-machine operations supplied by the SQL backend."""

    def _issue_checkpoint_restore_writer_token(self) -> object: ...

    def insert_runtime_publication(
        self,
        *,
        publication_id: str,
        kind: RuntimePublicationKind | str,
        pid: str,
        owner_instance_id: str,
        plan: Mapping[str, Any],
        phase: str = "planned",
        _checkpoint_restore_writer_token: object | None = None,
    ) -> Mapping[str, Any]: ...

    def get_runtime_publication(
        self,
        publication_id: str,
    ) -> Mapping[str, Any] | None: ...

    def list_runtime_publications(
        self,
        *,
        states: Iterable[RuntimePublicationState | str] | None = None,
        pid: str | None = None,
    ) -> list[Mapping[str, Any]]: ...

    def get_committed_root_spawn_publication(
        self,
        pid: str,
    ) -> Mapping[str, Any] | None: ...

    def redact_root_spawn_initial_goal(
        self,
        publication_id: str,
        *,
        pid: str,
        goal_oid: str,
        expected_payload_sha256: str,
        expected_states: Iterable[RuntimePublicationState | str] = ("committed",),
    ) -> bool: ...

    def query_runtime_publication_operation_reconciliation(
        self,
        *,
        kind: RuntimePublicationKind | str,
        state: RuntimePublicationState | str,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage: ...

    def query_runtime_publication_recovery(
        self,
        *,
        kind: RuntimePublicationKind | str,
        state: RuntimePublicationState | str,
        operation_reconciled: bool,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage: ...

    def query_checkpoint_payload_delivery_attempts(
        self,
        *,
        after: CheckpointPayloadDeliveryAttempt | None,
        limit: int,
    ) -> CheckpointPayloadDeliveryAttemptPage: ...

    def get_checkpoint_payload_delivery_attempt_state(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> CheckpointPayloadDeliveryAttemptState | None: ...

    def query_checkpoint_restore_payload_deliveries(
        self,
        *,
        delivery_state: PayloadDeliveryState | str,
        attempt_id: str | None,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage: ...

    def mark_runtime_publication_operation_reconciled(
        self,
        publication_id: str,
        *,
        expected_kind: RuntimePublicationKind | str,
        expected_state: RuntimePublicationState | str,
        expected_phase: str,
        expected_operation_id: str | None,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool: ...

    def runtime_publication_exists_for_pid(
        self,
        pid: str,
        *,
        kind: RuntimePublicationKind | str,
    ) -> bool: ...

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
        _checkpoint_restore_writer_token: object | None = None,
    ) -> Mapping[str, Any] | None: ...

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
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        """CAS a transition; ``None`` is unrestricted and empty matches none."""

        ...

    def update_runtime_publication_plan(
        self,
        publication_id: str,
        update: Mapping[str, Any],
        *,
        expected_states: Iterable[RuntimePublicationState | str] | None = None,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        """CAS a plan update; ``None`` is unrestricted and empty matches none."""

        ...

    def record_runtime_publication_artifact(
        self,
        publication_id: str,
        artifact: Mapping[str, Any],
        *,
        expected_states: Iterable[RuntimePublicationState | str] | None = None,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        """CAS a receipt append; ``None`` is unrestricted and empty matches none."""

        ...


class CheckpointPublicationWriterBackendProtocol(
    RuntimePublicationBackendProtocol,
    Protocol,
):
    """Storage-internal publication backend capable of issuing writer tokens."""

    def transition_checkpoint_restore_payload_delivery(
        self,
        publication_id: str,
        *,
        expected_delivery_state: str | None,
        delivery_state: str,
        expected_attempt: CheckpointPayloadDeliveryAttempt | None = None,
        delivery_attempt: CheckpointPayloadDeliveryAttempt | None = None,
        owner_instance_id: str | None = None,
        recovery_lease_id: str | None = None,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool: ...

    def begin_checkpoint_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
        *,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool: ...

    def ack_checkpoint_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
        *,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool: ...

    def abort_checkpoint_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
        *,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool: ...


class SnapshotCheckpointBackendProtocol(
    TransactionBackendProtocol,
    ProcessRestoreEpochRepositoryPort,
    Protocol,
):
    """Explicit raw SQL helpers owned by ``SnapshotCheckpointRepository``."""

    def snapshot_object_payloads(self, oids: Iterable[str]) -> dict[str, Any]: ...

    def get_process(self, pid: str) -> AgentProcess | None: ...

    def get_jit_rehydration_artifacts(
        self,
        *,
        pid: str,
        tool_ids: Iterable[str],
    ) -> tuple[JITRehydrationArtifact, ...]: ...

    def restore_process_for_exec(
        self,
        before_row: Mapping[str, Any],
        *,
        expected_revision: int,
        publication_id: str | None = None,
        capability_ids: Iterable[str] | None = None,
        fence_execution: bool = True,
    ) -> bool: ...

    def select_table_rows(
        self,
        table: str,
        where_sql: str = "",
        params: Iterable[Any] = (),
        *,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def insert_table_row(self, table: str, row: dict[str, Any]) -> None: ...

    def ensure_process_terminal_cleanup_intent(
        self,
        pid: str,
        status: ProcessStatus | str,
    ) -> None: ...

    def validate_table_identifier(self, table: str) -> str: ...

    def validate_column_identifier(self, table: str, column: str) -> str: ...

    def payload_marker(
        self,
        *,
        present: bool,
        recovered_after_reopen: bool = False,
    ) -> dict[str, Any]: ...

    def set_object_payload(self, oid: str, payload: Any) -> None: ...

    def rehydrate_object_payload_cache(
        self,
        object_payloads: Mapping[str, Any],
    ) -> tuple[str, ...]: ...

    def forget_object_payload(self, oid: str) -> None: ...

    def insert_checkpoint(self, checkpoint: Checkpoint, snapshot: dict[str, Any]) -> None: ...

    def get_checkpoint_snapshot(
        self,
        checkpoint_id: str,
    ) -> tuple[Checkpoint, dict[str, Any]] | None: ...

    def list_checkpoints(
        self,
        pid: str | None = None,
        limit: int | None = None,
    ) -> list[Checkpoint]: ...

    def reserve_process_restore_epoch(
        self,
        pid: str,
        *,
        revision_floor: int,
        execution_generation_floor: int,
        state_generation_floor: int,
        cursor: Any | None = None,
    ) -> tuple[int, int, int]: ...

    def reserve_process_restore_epochs(
        self,
        floors: Iterable[ProcessRestoreEpoch],
        *,
        cursor: Any | None = None,
    ) -> tuple[ProcessRestoreEpoch, ...]: ...

    def list_human_requests(
        self,
        pid: str | None = None,
        *,
        human: str | None = None,
        status: HumanRequestStatus | str | None = None,
        limit: int | None = None,
        newest: bool = False,
    ) -> list[HumanRequest]: ...

    def list_process_messages(
        self,
        recipient_pid: str | None = None,
        *,
        status: ProcessMessageStatus | str | None = None,
        statuses: Iterable[ProcessMessageStatus | str] | None = None,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        sender_prefix: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ProcessMessage]: ...

    def count_process_messages(
        self,
        recipient_pid: str | None = None,
        *,
        status: ProcessMessageStatus | str | None = None,
        statuses: Iterable[ProcessMessageStatus | str] | None = None,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
    ) -> int: ...

    def list_object_tasks(
        self,
        *,
        owner_oid: str | None = None,
        creator_pid: str | None = None,
        statuses: Iterable[str | ObjectTaskStatus] | None = None,
        include_terminal: bool = True,
        limit: int | None = None,
    ) -> list[ObjectTask]: ...

    def query_checkpoint_object_tasks(
        self,
        *,
        statuses: Iterable[str | ObjectTaskStatus],
        checkpoint_created_at: str,
        terminal_after: bool,
        scope_pids: Iterable[str] = (),
        owner_oids: Iterable[str] = (),
        after: ObjectTaskRecoveryCursor | None,
        limit: int | None = None,
    ) -> ObjectTaskRecoveryPage: ...

    def set_llm_context_generation(self, pid: str, generation: str) -> None: ...

    def get_object(self, oid: str) -> AgentObject | None: ...

    def list_objects_owned_by(
        self,
        owner_kind: str | ObjectOwnerKind,
        owner_id: str,
    ) -> list[AgentObject]: ...

    def object_payload(self, oid: str) -> Any: ...

    def has_object_payload(self, oid: str, *, row: Any | None = None) -> bool: ...

    def list_namespaces_created_by(self, created_by: str) -> list[ObjectNamespace]: ...

    def get_namespace(self, namespace: str) -> ObjectNamespace | None: ...

    def namespace_exists(self, namespace: str) -> bool: ...

    def list_tools(self, limit: int | None = None) -> list[dict[str, Any]]: ...

    def delete_tool(
        self,
        tool_id: str,
        *,
        registered_by: str | None = None,
    ) -> None: ...

    def upsert_image(
        self,
        image: AgentImage,
        *,
        registered_by: str,
        source: str | None,
        created_at: str,
    ) -> None: ...

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
    ) -> None: ...

    def get_image_artifact(
        self,
        artifact_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None: ...

    def tool_id_referenced_outside_scope(
        self,
        tool_id: str,
        *,
        scoped_pids: Iterable[str],
    ) -> bool: ...

    def delete_tool_if_unreferenced(
        self,
        tool_id: str,
        *,
        excluding_pid: str,
    ) -> bool: ...


class RuntimeModuleBackendProtocol(TransactionBackendProtocol, Protocol):
    """Raw durable module row operations supplied by the SQL backend."""

    def upsert_runtime_module(
        self,
        *,
        module_id: str,
        name: str,
        version: str,
        entrypoint: str,
        manifest_path: str,
        manifest_sha256: str,
        source_path: str,
        source_sha256: str,
        status: str,
        loaded_at: str | None,
        registered: dict[str, Any],
        error: str | None,
        metadata: dict[str, Any],
    ) -> None: ...

    def get_runtime_module(self, module_id: str) -> Mapping[str, Any] | None: ...

    def list_runtime_modules(
        self,
        limit: int | None = None,
    ) -> list[Mapping[str, Any]]: ...


class OperationEvidenceBackendProtocol(TransactionBackendProtocol, Protocol):
    """Causal-operation and linked-evidence SQL backend surface."""

    def list_events(
        self,
        target: str | None = None,
        limit: int | None = None,
        before_event_id: str | None = None,
        after_event_id: str | None = None,
        *,
        include_gui_presentation: bool = True,
    ) -> list[Event]: ...

    def insert_operation(self, record: OperationRecord) -> None: ...

    def get_operation(self, operation_id: str) -> OperationRecord | None: ...

    def list_operation_ids_by_runtime_publication_id(
        self,
        publication_id: str,
    ) -> list[str]: ...

    def list_operations(
        self,
        *,
        pid: str | None = None,
        root_operation_id: str | None = None,
        roots_only: bool = False,
        state: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[OperationRecord]: ...

    def scan_stale_running_operations(
        self,
        *,
        after: OperationCursor | None,
        limit: int,
    ) -> OperationPage: ...

    def operation_ids_with_unknown_external_effects(
        self,
        operation_ids: Iterable[str],
    ) -> set[str]: ...

    def stale_operation_recovery_index(self) -> AbstractContextManager[None]: ...

    def operation_has_unknown_external_effect(self, operation_id: str) -> bool: ...

    def update_operation(
        self,
        record: OperationRecord,
        *,
        expected_states: Iterable[str] | None = None,
    ) -> bool:
        """CAS an operation update; ``None`` is unrestricted and empty matches none."""

        ...

    def insert_operation_evidence(self, link: OperationEvidenceLink) -> bool: ...

    def list_operation_evidence(
        self,
        *,
        operation_ids: Iterable[str] | None = None,
        evidence_types: Iterable[str] | None = None,
        evidence_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[OperationEvidenceLink]: ...

    def get_external_effect(self, effect_id: str) -> ExternalEffectRecord | None: ...

    def current_effect_ledger_seq(self) -> int: ...

    def external_effect_transition_matches(
        self,
        transition_seq: int,
        effect_id: str,
        *,
        effect_state: str,
        transaction_state: str,
    ) -> bool: ...

    def list_external_effects_changed_after(
        self,
        effect_ledger_seq: int,
        *,
        pids: Iterable[str] | None = None,
    ) -> list[ExternalEffectRecord]: ...


class UnitOfWorkBackendProtocol(
    ProcessBackendProtocol,
    TaskRunBackendProtocol,
    ObjectQueryBackendProtocol,
    ObjectRecoveryBackendProtocol,
    AuthorityRecoveryBackendProtocol,
    ResourceBackendProtocol,
    CheckpointPublicationWriterBackendProtocol,
    SnapshotCheckpointBackendProtocol,
    RuntimeModuleBackendProtocol,
    OperationEvidenceBackendProtocol,
    ToolArtifactRepositoryProtocol,
    PayloadRetentionStore,
    Protocol,
):
    """Complete concrete backend contract required to assemble a UnitOfWork."""
