from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agent_libos.models import (
    TASK_RUN_DISPATCHABLE_STATUSES,
    TASK_RUN_TERMINAL_STATUSES,
    DataFlowContext,
    DataLabels,
    CapabilityEffect,
    EventType,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ObjectTaskStatus,
    OperationKind,
    OperationOutcome,
    ProcessMessageKind,
    ProcessStatus,
    StaleExecutionProcessWait,
    TaskRunCommand,
    TaskRunCursor,
    TaskRunLedgerCursor,
    TaskRunLedgerItem,
    TaskRunLedgerKind,
    TaskRunLink,
    TaskRunPayload,
    TaskRunPayloadRetention,
    TaskRunRecord,
    TaskRunRequirement,
    TaskRunRequirementKind,
    TaskRunRequirementStatus,
    TaskRunResumePoint,
    TaskRunRetention,
    TaskRunSpecV1,
    TaskRunStatus,
    TaskRunSummary,
    canonical_task_run_json,
    task_run_payload_sha256,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    NotFound,
    TaskRunCommandConflict,
    TaskRunCompletionContractError,
    TaskRunRevisionConflict,
    ValidationError,
)
from agent_libos.llm.pending import (
    pending_message_filters,
    pending_metadata,
    pending_resume_token,
    pending_task_run_transcript_call_id,
)
from agent_libos.llm.task_runs import (
    TaskRunDispatchDeferred,
    normalize_validated_action_manifest,
)
from agent_libos.runtime.task_run_reference import (
    TASK_RUN_REFERENCE_KEY,
    TASK_RUN_REFERENCE_SCHEMA_VERSION,
    is_task_run_reference_payload,
)
from agent_libos.runtime.task_run_bindings import (
    canonical_sha256 as task_run_binding_sha256,
    expected_activated_process_projection,
    loaded_skill_hashes,
    pre_action_binding,
    require_exact_activated_projection,
    tool_binding_hash,
    validate_pre_action_binding,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import to_jsonable

if TYPE_CHECKING:
    from agent_libos.config import AgentLibOSConfig
    from agent_libos.runtime.runtime import Runtime


_TERMINAL_PROCESS_STATUSES = frozenset(
    {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
)
_UNKNOWN_EFFECT_STATES = frozenset({"dispatched", "unknown"})
_SETTLED_EFFECT_STATES = frozenset({"committed", "failed", "compensated"})
_SIGNED_BIGINT_MAX = 2**63 - 1
_COMMAND_RESULT_BASE_KEYS = frozenset({"schema_version", "summary"})
_CONTROL_ADMISSION_FIELDS = frozenset(
    {
        "admission_ledger_seq",
        "admission_ledger_item_id",
        "admission_evidence_sha256",
    }
)
_CONTROL_COMPLETION_PRESERVED_FIELDS = _CONTROL_ADMISSION_FIELDS | frozenset(
    {"interrupt_provenance_sha256"}
)
_INTERRUPT_RECEIPT_FIELDS = frozenset(
    {
        "settlement_state",
        "settlement_kind",
        "pause_generation",
        "cancel_generation",
        "prior_status",
        "interrupt_provenance_sha256",
        "admission_runtime_epoch",
        "resume_fences",
        *_CONTROL_ADMISSION_FIELDS,
    }
)
_INTERRUPT_COMPLETE_RESULT_KEYS = (
    _COMMAND_RESULT_BASE_KEYS | _CONTROL_ADMISSION_FIELDS | frozenset(
    {
        "settlement_state",
        "settlement_kind",
        "pause_generation",
        "cancel_generation",
        "prior_status",
        "interrupt_provenance_sha256",
    }
    )
)
_INTERRUPT_PENDING_RESULT_KEYS = _INTERRUPT_COMPLETE_RESULT_KEYS | frozenset(
    {"admission_runtime_epoch", "resume_fences"}
)
_RUN_RESULT_KEYS = _COMMAND_RESULT_BASE_KEYS | _CONTROL_ADMISSION_FIELDS | frozenset(
    {"settlement_state", "admission_revision"}
)
_RESUME_RESULT_KEYS = _COMMAND_RESULT_BASE_KEYS | _CONTROL_ADMISSION_FIELDS | frozenset(
    {"settlement_state", "pause_generation"}
)
_CANCEL_RESULT_KEYS = _COMMAND_RESULT_BASE_KEYS | _CONTROL_ADMISSION_FIELDS | frozenset(
    {"settlement_state", "cancel_generation"}
)
_DEADLINE_RESULT_KEYS = _CANCEL_RESULT_KEYS | frozenset({"settlement_kind"})
_EFFECT_RECEIPT_RESULT_KEYS = _COMMAND_RESULT_BASE_KEYS | frozenset(
    {
        "settlement_state",
        "settlement_kind",
        "cancel_generation",
        "effect_id",
        "expected_transaction_state",
        "admission_runtime_epoch",
        "settlement_transition_seq",
        "settlement_audit_record_id",
    }
)
_TERMINALIZE_RESULT_KEYS = _DEADLINE_RESULT_KEYS
_LINKED_RESULT_KEYS = _COMMAND_RESULT_BASE_KEYS | frozenset(
    {"run_id", "revision", "new_run_id", "new_run_summary"}
)
_LINKED_RESULT_FIELDS = _LINKED_RESULT_KEYS - _COMMAND_RESULT_BASE_KEYS
_COMPLETED_OUTCOME_KEYS = frozenset(
    {
        "schema_version",
        "state",
        "paired_outputs_persisted",
        "data_labels",
        "result",
        "durable_wait",
        "previous_response_id_used",
    }
)
_TASK_RUN_REQUIREMENT_BINDING_KEY = "task_run_requirement_binding_v1"
_PROMPT_REQUIREMENT_BINDING_KEYS = frozenset(
    {"schema_version", "run_id", "pid", "context_generation", "requirements"}
)
_PROMPT_REQUIREMENT_ENTRY_KEYS = frozenset(
    {"requirement_id", "ordinal", "requirement_sha256"}
)
_REQUIREMENT_COMPLETION_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "run_id",
        "pid",
        "call_id",
        "context_generation",
        "action_manifest_sha256",
        "outcome_sha256",
        "requirement_binding",
        "requirement_binding_sha256",
        "requirement_outcomes",
    }
)
_REQUIREMENT_COMPLETION_OUTCOME_KEYS = frozenset(
    {"requirement_id", "status", "reported_status", "evidence_receipt_ids"}
)
_DURABLE_WAIT_TYPES = frozenset({"llm_release", "human", "child", "message"})
_BINDING_TRANSITION_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "source_payload_id",
        "source_payload_sha256",
        "pre_binding_projection_sha256",
        "pre_image_binding_hash",
        "pre_tool_binding_hash",
        "pre_provider_binding_hash",
        "post_image_binding_hash",
        "post_tool_binding_hash",
        "post_provider_binding_hash",
        "action_manifest_sha256",
        "action_result_sha256",
        "skill_id",
        "package_sha256",
        "loaded_skill_sha256",
        "settled_effect_seq",
    }
)
_PUBLIC_BLOCKER_KINDS = frozenset(
    {
        "unknown_effect",
        "payload_missing",
        "payload_corrupt",
        "binding_drift",
        "pending_action_unreplayable",
        "active_object_task",
        "requirements_unsatisfied",
        "cleanup_failed",
        "authority_revoked",
        "deadline_reached",
        "effect_unsettled",
        "reservation_unsettled",
        "publication_unsettled",
        "manual_recovery_required",
    }
)
_LAUNCH_OPTION_FIELDS = frozenset(
    {
        "capabilities",
        "resource_budget",
        "working_directory",
        "llm_profile_id",
    }
)

_WAIT_BOUNDARY_STATUSES = frozenset(
    {
        TaskRunStatus.WAITING_HUMAN,
        TaskRunStatus.WAITING_PROCESS,
        TaskRunStatus.WAITING_MESSAGE,
        TaskRunStatus.WAITING_TOOL,
        TaskRunStatus.PAUSED,
        TaskRunStatus.NEEDS_ATTENTION,
        *TASK_RUN_TERMINAL_STATUSES,
    }
)


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_settled_gui_presentation_effect(effect: Any) -> bool:
    """Validate the shared, non-mutating Human presentation envelope."""

    return all(
        (
            effect.provider == "human",
            effect.operation == "write",
            effect.effect_state == "finalized",
            effect.transaction_state == "committed",
            effect.rollback_class
            is ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            effect.rollback_status
            is ExternalEffectRollbackStatus.NOT_REQUIRED,
            effect.state_mutation is False,
            effect.information_flow is True,
        )
    )


def _gui_presentation_request_id(effect: Any) -> str | None:
    """Return the bound request only for the exact public GUI envelope."""

    metadata = effect.provider_metadata
    if not isinstance(metadata, Mapping):
        return None
    request_id = str(metadata.get("request_id") or "")
    if not all(
        (
            request_id,
            metadata.get("presented") is True,
            metadata.get("channel") == "gui",
        )
    ):
        return None
    return request_id


def _has_exact_gui_presentation_context(
    effect: Any,
    request_id: str,
) -> bool:
    """Fail closed unless the protected observation is metadata-only."""

    metadata = effect.provider_metadata
    if not isinstance(metadata, Mapping):
        return False
    context = metadata.get("context")
    if not isinstance(context, Mapping):
        return False
    observation = context.get("prompt_observation")
    if not isinstance(observation, Mapping):
        return False
    return all(
        (
            context.get("request_id") == request_id,
            context.get("purpose") == "gui_presentation",
            context.get("operation") == "write",
            context.get("channel") == "gui",
            observation.get("metadata_only") is True,
            observation.get("redacted") is True,
            observation.get("truncated") is True,
            observation.get("preview") == "<redacted protected payload>",
            _is_lower_sha256(observation.get("sha256")),
        )
    )


def _timestamp_is_strictly_after(value: str, boundary: str) -> bool:
    try:
        selected = datetime.fromisoformat(value.replace("Z", "+00:00"))
        lower_bound = datetime.fromisoformat(boundary.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return False
    if selected.tzinfo is None or lower_bound.tzinfo is None:
        return False
    return selected.astimezone(timezone.utc) > lower_bound.astimezone(
        timezone.utc
    )


def _is_post_purge_gui_presentation_candidate(
    record: TaskRunRecord,
    effect: Any,
    process: Any | None,
) -> bool:
    """Validate immutable Run/effect facts before consulting repositories."""

    purged_at = record.payloads_purged_at
    return all(
        (
            purged_at is not None,
            record.status in TASK_RUN_TERMINAL_STATUSES,
            process is not None,
            process is not None and process.task_run_id == record.run_id,
            process is not None and effect.pid == process.pid,
            _is_settled_gui_presentation_effect(effect),
            effect.record_id is not None,
            effect.event_id is not None,
            _is_lower_sha256(effect.canonical_args_hash),
            isinstance(effect.idempotency_key, str),
            bool(effect.idempotency_key),
            effect.provider_receipt == {},
            purged_at is not None
            and _timestamp_is_strictly_after(effect.created_at, purged_at),
        )
    )


def _matches_durable_human_wait(
    wrapper: Mapping[str, Any],
    *,
    request_id: str,
    pid: str,
) -> bool:
    snapshot = wrapper.get("wait_snapshot")
    return all(
        (
            wrapper.get("kind") == "durable_wait_action",
            wrapper.get("state") == "waiting",
            isinstance(snapshot, Mapping),
            isinstance(snapshot, Mapping) and snapshot.get("wait_type") == "human",
            isinstance(snapshot, Mapping)
            and snapshot.get("request_id") == request_id,
            isinstance(snapshot, Mapping) and snapshot.get("pid") == pid,
        )
    )


@dataclass(frozen=True, slots=True)
class TaskRunListPage:
    """Public summary page with an opaque, wire-safe keyset cursor."""

    records: tuple[TaskRunSummary, ...]
    next_cursor: str | None = None
    schema_version: int = 1

    @property
    def items(self) -> tuple[TaskRunSummary, ...]:
        return self.records


@dataclass(frozen=True, slots=True)
class TaskRunRequirementPage:
    records: tuple[Mapping[str, Any], ...]
    next_cursor: str | None = None
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class TaskRunRecoveryOption:
    option_id: str
    kind: str
    label: str
    requires_receipt: bool = False
    effect_id: str | None = None
    expected_transaction_state: str | None = None
    runtime_epoch: int | None = None
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class TaskRunLedgerListPage:
    records: tuple[TaskRunLedgerItem, ...]
    next_cursor: str | None = None
    schema_version: int = 1


class TaskRunManager:
    """Durable Host controller for one supervised ``AgentProcess`` tree.

    The SQL Store owns all durable CAS and epoch checks.  This service owns
    state projection and dispatch policy; in particular, it never turns an
    unknown/dispatched external effect into permission to retry.
    """

    def __init__(
        self,
        host: Runtime,
        *,
        config: AgentLibOSConfig,
    ) -> None:
        self._host = host
        self._store = host.store
        self._uow = host.uow
        self._process = host.process
        self._messages = host.messages
        self._audit = host.audit
        self._events = host.events
        self._object_tasks = host.object_tasks
        self.config = config
        self._condition = threading.Condition(threading.RLock())
        self._dispatch_scope = threading.local()
        self._active_run_dispatches: dict[str, int] = {}
        self._active_external_dispatches: dict[str, int] = {}
        self._control_mutations: set[tuple[str, str]] = set()
        self._external_dispatch_context: ContextVar[
            tuple[str, str, str] | None
        ] = ContextVar(
            f"task_run_external_dispatch_{id(self)}",
            default=None,
        )
        self._runtime_epoch = self._claim_runtime_epoch()
        self._recovered = False
        self._recovered_total_count = 0
        self._prevalidated_blockers: dict[str, dict[str, Any]] = {}
        self._prompt_requirement_bindings: dict[
            tuple[str, str], dict[str, Any]
        ] = {}

    @property
    def runtime_epoch(self) -> int:
        return self._runtime_epoch

    @property
    def recovered_total_count(self) -> int:
        return self._recovered_total_count

    # ------------------------------------------------------------------
    # Public reads

    def get(self, run_id: str) -> TaskRunSummary:
        record = self._require_run(run_id)
        if record.status not in TASK_RUN_TERMINAL_STATUSES:
            record = self._reconcile_deadline(record)
            record = self._project_paused_terminal_root(record)
        return self._summary(record)

    def list(
        self,
        *,
        statuses: Iterable[TaskRunStatus | str] | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> TaskRunListPage:
        selected_limit = (
            self.config.task_runs.list_page_size if limit is None else limit
        )
        if (
            type(selected_limit) is not int
            or selected_limit <= 0
            or selected_limit > self.config.task_runs.list_hard_limit
        ):
            raise ValidationError("TaskRun list limit is outside configured bounds")
        selected_statuses = (
            None
            if statuses is None
            else tuple(dict.fromkeys(TaskRunStatus(status) for status in statuses))
        )
        # Status filtering must be applied after every elapsed absolute
        # deadline has converged.  Otherwise a queued filter can return a Run
        # that this same call cancelled, while a cancelled filter can miss it.
        self._converge_expired_deadlines()
        page = self._store.list_task_runs(
            statuses=selected_statuses,
            after=self._decode_run_cursor(cursor),
            limit=selected_limit,
        )
        records: list[TaskRunSummary] = []
        for record in page.records:
            # ``get`` and ``list`` are both authoritative Host projections.
            # An elapsed absolute deadline must therefore persist its cancel
            # intent regardless of which bounded read first observes it.
            if record.status not in TASK_RUN_TERMINAL_STATUSES:
                record = self._reconcile_deadline(record)
                record = self._project_paused_terminal_root(record)
            if (
                selected_statuses is not None
                and record.status not in selected_statuses
            ):
                # Cover the narrow wall-clock race between the convergence
                # scan and the final page query without lying about filters.
                continue
            records.append(self._summary(record))
        return TaskRunListPage(
            records=tuple(records),
            next_cursor=self._encode_run_cursor(page.next_cursor),
        )

    def list_requirements(
        self,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> TaskRunRequirementPage:
        self._require_run(run_id)
        selected_limit = (
            self.config.task_runs.list_page_size if limit is None else limit
        )
        if (
            type(selected_limit) is not int
            or selected_limit <= 0
            or selected_limit > self.config.task_runs.list_hard_limit
        ):
            raise ValidationError(
                "TaskRun requirement list limit is outside configured bounds"
            )
        after = self._decode_requirement_cursor(cursor)
        records = tuple(
            self._uow.task_runs.list_task_run_requirements(
                run_id,
                after=after,
                limit=selected_limit + 1,
            )
        )
        selected = records[:selected_limit]
        next_cursor = None
        if len(records) > selected_limit and selected:
            last = selected[-1]
            next_cursor = self._encode_opaque_cursor(
                {"ordinal": last.ordinal, "requirement_id": last.requirement_id}
            )
        return TaskRunRequirementPage(
            tuple(self._requirement_view(item) for item in selected),
            next_cursor,
        )

    def list_ledger(
        self,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> TaskRunLedgerListPage:
        self._require_run(run_id)
        self._project_evidence(run_id)
        selected_limit = (
            self.config.task_runs.ledger_page_size if limit is None else limit
        )
        if (
            type(selected_limit) is not int
            or selected_limit <= 0
            or selected_limit > self.config.task_runs.ledger_page_hard_limit
        ):
            raise ValidationError("TaskRun ledger limit is outside configured bounds")
        page = self._store.list_task_run_ledger(
            run_id,
            after=self._decode_ledger_cursor(cursor),
            limit=selected_limit,
        )
        return TaskRunLedgerListPage(
            records=page.records,
            next_cursor=self._encode_ledger_cursor(page.next_cursor),
        )

    def list_human_requests(
        self,
        run_id: str,
        *,
        statuses: Iterable[str] | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> Any:
        """Return a bounded Human-request page without a global snapshot scan."""

        self._require_run(run_id)
        selected_limit = (
            self.config.task_runs.list_page_size if limit is None else limit
        )
        if (
            type(selected_limit) is not int
            or selected_limit <= 0
            or selected_limit > self.config.task_runs.list_hard_limit
        ):
            raise ValidationError(
                "TaskRun Human request limit is outside configured bounds"
            )
        pids = self._member_pids(run_id)
        if len(pids) > self.config.task_runs.recovery_page_hard_limit:
            raise ValidationError("TaskRun process tree exceeds bounded Human lookup")
        repository = getattr(self._host.uow, "task_runs", None)
        method = getattr(repository, "list_human_requests_for_pids", None)
        if not callable(method):
            method = getattr(self._store, "list_human_requests_for_pids", None)
        if not callable(method):
            # Fail closed instead of falling back to HumanManager.list(), whose
            # legacy snapshot can be globally truncated before PID filtering.
            raise ValidationError(
                "Store does not support bounded TaskRun Human request lookup"
            )
        return method(
            pids,
            statuses=statuses,
            limit=selected_limit,
            cursor=cursor,
        )

    # ------------------------------------------------------------------
    # Creation and control

    def create(
        self,
        spec: TaskRunSpecV1 | Mapping[str, Any],
        *,
        client_request_id: str,
        auto_run: bool = False,
    ) -> TaskRunSummary:
        if not self.config.task_runs.enabled:
            raise ValidationError("Durable TaskRuns are disabled")
        if not self.config.task_runs.plaintext_payloads_enabled:
            raise ValidationError(
                "durable TaskRun plaintext payloads are disabled; Host must "
                "explicitly enable task_runs.plaintext_payloads_enabled"
            )
        selected_spec = (
            spec if isinstance(spec, TaskRunSpecV1) else TaskRunSpecV1.from_mapping(spec)
        )
        if type(auto_run) is not bool:
            raise ValidationError("TaskRun auto_run must be boolean")
        if selected_spec.deadline_at is not None:
            deadline = datetime.fromisoformat(
                selected_spec.deadline_at.replace("Z", "+00:00")
            )
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc):
                raise ValidationError("TaskRun deadline must be in the future")
        request_id = self._identifier(client_request_id, "client_request_id")
        request_hash = self._request_hash(
            "create",
            {
                "client_request_id": request_id,
                "spec": selected_spec.to_mapping(),
            },
        )
        existing = self._store.get_task_run_command_by_client_request_id(request_id)
        if existing is not None:
            self._require_same_command(existing, "create", request_hash)
            replayed = self._summary_from_command(existing)
            if auto_run:
                run_command_id = f"{request_id}:run"
                run_command = self._store.get_task_run_command(
                    existing.run_id,
                    run_command_id,
                )
                if run_command is not None:
                    self._require_same_command(
                        run_command,
                        "run",
                        self._request_hash(
                            "run",
                            {
                                "expected_revision": replayed.revision,
                                "max_quanta": None,
                            },
                        ),
                    )
                    # Re-enter the exact replay path so a provisional receipt
                    # can finish local settlement without another dispatch.
                    return self.run_until_blocked(
                        existing.run_id,
                        expected_revision=replayed.revision,
                        command_id=run_command_id,
                    )
                current = self.get(existing.run_id)
                if (
                    current.status is not TaskRunStatus.QUEUED
                    or current.revision != replayed.revision
                ):
                    # Creation committed but its requested dispatch did not
                    # leave a command result.  The immutable create receipt,
                    # not the newest Run row, is the only authority for the
                    # deterministic auto-run command's revision identity.
                    raise TaskRunRevisionConflict(
                        "TaskRun auto-run result is not durably reconstructable"
                    )
                return self.run_until_blocked(
                    current.run_id,
                    expected_revision=replayed.revision,
                    command_id=run_command_id,
                )
            return replayed

        run_id = new_id("run")
        now = utc_now()
        image_id = selected_spec.image_id or self.config.runtime.default_image_id
        goal_payload = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=run_id,
            role="goal",
            label="TaskRun goal",
            value={
                "goal": selected_spec.goal,
                "data_labels": DataLabels(origin="host").to_dict(),
            },
            created_at=now,
        )
        self._require_payload_bound(goal_payload)
        marker = {
            TASK_RUN_REFERENCE_KEY: {
                "run_id": run_id,
                "payload_sha256": goal_payload.sha256,
                "schema_version": TASK_RUN_REFERENCE_SCHEMA_VERSION,
            }
        }
        assert is_task_run_reference_payload(marker)
        binding_hash = self._binding_hash(
            image_id=image_id,
            launch_options=selected_spec.launch_options,
            authority_manifest_id=selected_spec.authority_manifest_id,
        )
        record = TaskRunRecord.from_spec(
            run_id,
            selected_spec,
            image_id=image_id,
            runtime_epoch=self._runtime_epoch,
            binding_hash=binding_hash,
            requirement_count=1,
            created_at=now,
            updated_at=now,
        )
        requirement = TaskRunRequirement(
            requirement_id=new_id("trreq"),
            run_id=run_id,
            ordinal=0,
            kind=TaskRunRequirementKind.INITIAL,
            status=TaskRunRequirementStatus.PENDING,
            payload_id=goal_payload.payload_id,
            requirement_sha256=goal_payload.sha256,
            label="Initial goal",
            created_by="host",
            created_at=now,
            updated_at=now,
        )

        launch = self._launch_kwargs(selected_spec)
        creation_evidence: dict[str, Any] = {}

        def commit_task_run_create(
            pid: str,
            publication_id: str,
            process_event_id: str,
            process_audit_id: str,
        ) -> None:
            """Publish the complete TaskRun create evidence in spawn's commit."""

            nonlocal record
            record, evidence = self._commit_task_run_create_evidence(
                record=record,
                requirement=requirement,
                goal_payload=goal_payload,
                binding_hash=binding_hash,
                pid=pid,
                publication_id=publication_id,
                process_event_id=process_event_id,
                process_audit_id=process_audit_id,
                created_at=now,
            )
            creation_evidence.update(evidence)

        with self._uow.transaction(include_object_payloads=True):
            self._store.insert_task_run(record)
            self._store.insert_task_run_payload(goal_payload)
            self._store.insert_task_run_requirement(requirement)
            pid = self._spawn_root(
                run_id=run_id,
                epoch=self._runtime_epoch,
                image_id=image_id,
                marker=marker,
                launch=launch,
                authority_manifest_id=selected_spec.authority_manifest_id,
                task_run_commit=commit_task_run_create,
            )
            if not creation_evidence or record.root_pid != pid:
                raise RuntimeError("TaskRun create evidence callback did not commit")
            command = TaskRunCommand(
                command_id=f"create:{request_id}",
                client_request_id=request_id,
                run_id=run_id,
                command_kind="create",
                request_hash=request_hash,
                result=self._command_result(record),
                result_revision=record.revision,
                created_at=now,
            )
            self._store.insert_task_run_command(
                command,
                expected_runtime_epoch=self._runtime_epoch,
            )
        self._notify_updated()
        summary = self._summary(record)
        if auto_run:
            return self.run_until_blocked(
                run_id,
                expected_revision=summary.revision,
                command_id=f"{request_id}:run",
            )
        return summary

    def _commit_task_run_create_evidence(
        self,
        *,
        record: TaskRunRecord,
        requirement: TaskRunRequirement,
        goal_payload: TaskRunPayload,
        binding_hash: str,
        pid: str,
        publication_id: str,
        process_event_id: str,
        process_audit_id: str,
        created_at: str,
    ) -> tuple[TaskRunRecord, dict[str, str]]:
        record = self._store.update_task_run_cas(
            record.run_id,
            record.revision,
            updates={
                "root_pid": pid,
                "active_pid": pid,
                "updated_at": created_at,
            },
            expected_runtime_epoch=self._runtime_epoch,
        )
        requirement_ledger = self._append_ledger(
            record.run_id,
            kind=TaskRunLedgerKind.REQUIREMENT,
            status="pending",
            label="initial requirement",
            requirement_id=requirement.requirement_id,
            payload_id=goal_payload.payload_id,
            metadata={"requirement_sha256": goal_payload.sha256},
        )
        operation = self._host.operations.start(
            kind=OperationKind.RUNTIME,
            name="task_run.create",
            actor="host",
            pid=pid,
            expected_roles=("context", "event", "audit", "result"),
            metadata={
                "schema_version": 1,
                "run_id": record.run_id,
                "runtime_epoch": self._runtime_epoch,
                "retention": record.retention.value,
            },
        )
        task_event = self._events.emit(
            EventType.TASK_RUN_CREATED,
            source="runtime",
            target=record.run_id,
            payload={
                "schema_version": 1,
                "run_id": record.run_id,
                "revision": record.revision,
                "status": record.status.value,
                "root_pid": pid,
                "display_title": record.display_title,
                "retention": record.retention.value,
            },
            correlation_id=operation.operation_id,
        )
        task_audit = self._audit.record(
            actor="host",
            action="task_run.create",
            target=f"task_run:{record.run_id}",
            input_refs=[goal_payload.payload_id],
            output_refs=[pid],
            decision={
                "status": record.status.value,
                "root_pid": pid,
                "retention": record.retention.value,
                "binding_hash": binding_hash,
                "runtime_epoch": self._runtime_epoch,
            },
            correlation_id=operation.operation_id,
        )
        process_ledger = self._append_ledger(
            record.run_id,
            kind=TaskRunLedgerKind.PROCESS,
            status="created",
            label="root process and TaskRun create evidence committed",
            pid=pid,
            operation_id=operation.operation_id,
            metadata={
                "runtime_epoch": self._runtime_epoch,
                "publication_id": publication_id,
            },
        )
        evidence = (
            ("task_run", record.run_id, "context"),
            ("payload", goal_payload.payload_id, "context"),
            ("event", task_event.event_id, "event"),
            ("audit", task_audit.record_id, "audit"),
            ("process", pid, "result"),
            ("runtime_publication", publication_id, "result"),
            ("event", process_event_id, "event"),
            ("audit", process_audit_id, "audit"),
        )
        for evidence_type, evidence_id, role in evidence:
            self._host.operations.link_evidence(
                evidence_type,
                evidence_id,
                role,
                operation_id=operation.operation_id,
                metadata={"task_run_id": record.run_id},
            )
            self._store.insert_task_run_link(
                TaskRunLink(
                    link_id=new_id("trlink"),
                    run_id=record.run_id,
                    ledger_seq=process_ledger.seq,
                    evidence_type=evidence_type,
                    evidence_id=evidence_id,
                    role=role,
                    created_at=created_at,
                    metadata={"operation_id": operation.operation_id},
                )
            )
        self._store.insert_task_run_link(
            TaskRunLink(
                link_id=new_id("trlink"),
                run_id=record.run_id,
                ledger_seq=requirement_ledger.seq,
                evidence_type="requirement",
                evidence_id=requirement.requirement_id,
                role="initial",
                created_at=created_at,
            )
        )
        self._host.operations.finish(
            OperationOutcome.SUCCEEDED,
            operation_id=operation.operation_id,
            metadata={"task_run_revision": record.revision},
        )
        return record, {
            "operation_id": operation.operation_id,
            "event_id": task_event.event_id,
            "audit_id": task_audit.record_id,
        }

    def active_runs_for_pids(
        self,
        scoped_pids: Iterable[str],
    ) -> tuple[TaskRunSummary, ...]:
        selected = self._normalize_pids(scoped_pids)
        run_ids = self._store.list_active_task_run_ids_for_pids(selected)
        return tuple(self.get(run_id) for run_id in sorted(run_ids))

    def require_process_epoch(
        self,
        pid: str | None,
        run_id: str,
        epoch: int,
        action: str,
    ) -> None:
        record = self._require_run(run_id)
        if record.runtime_epoch != epoch or epoch != self._runtime_epoch:
            raise TaskRunRevisionConflict(
                f"stale TaskRun epoch refused {action}: {run_id}"
            )
        draining = self._external_dispatch_context.get()
        admitted_settlement = (
            record.status in {TaskRunStatus.PAUSED, TaskRunStatus.CANCELLING}
            and draining is not None
            and draining[0] == run_id
            and draining[1] == pid
        )
        if (
            record.status not in TASK_RUN_DISPATCHABLE_STATUSES
            and not admitted_settlement
        ):
            raise ValidationError(
                f"TaskRun status refuses {action}: {record.status.value}"
            )
        if self._deadline_expired(record):
            if (
                action == "process.spawn"
                and record.status is TaskRunStatus.QUEUED
                and record.started_at is None
                and pid == record.root_pid
            ):
                # Creation pre-validates the absolute deadline.  If the wall
                # clock crosses it inside the root publication transaction,
                # commit the coherent root/Run unit and let the first observer
                # persist cancellation; never violate the root/status CHECK.
                return
            self._reconcile_deadline(record)
            raise ValidationError(f"TaskRun deadline elapsed before {action}")

    def should_skip_pid(self, pid: str) -> bool:
        """Scheduler gate: a Run process executes only in an explicit scope."""

        process = self._store.get_process(pid)
        run_id = getattr(process, "task_run_id", None) if process is not None else None
        if run_id is None:
            return False
        record = self._store.get_task_run(run_id)
        if record is None:
            return True
        scope = getattr(self._dispatch_scope, "admission", None)
        return not (
            scope == (run_id, record.pause_generation)
            and record.runtime_epoch == self._runtime_epoch
            and getattr(process, "task_run_epoch", None) == self._runtime_epoch
            and record.status is TaskRunStatus.RUNNING
            and not self._deadline_expired(record)
        )

    # ------------------------------------------------------------------
    # LLM durable-resume seam

    def prompt_context_for_pid(self, pid: str) -> Mapping[str, Any] | None:
        process = self._store.get_process(pid)
        run_id = getattr(process, "task_run_id", None) if process is not None else None
        if run_id is None:
            return None
        record = self._require_run(run_id)
        if (
            getattr(process, "task_run_epoch", None) != record.runtime_epoch
            or record.runtime_epoch != self._runtime_epoch
        ):
            raise TaskRunRevisionConflict("TaskRun prompt lost its epoch fence")
        goal_payload = self._payload_by_role(run_id, "goal")
        goal_wrapper = self._decode_payload(goal_payload, role="goal")
        if set(goal_wrapper) != {"goal", "data_labels"}:
            self._mark_attention(
                record,
                self._blocker("payload_corrupt", "goal payload shape is invalid"),
            )
            raise ValidationError("TaskRun goal payload shape is invalid")
        labels_to_aggregate = [DataLabels.from_dict(goal_wrapper["data_labels"])]
        resume = self._store.get_task_run_resume_point(pid, complete_only=True)
        if resume is not None:
            self._require_prompt_resume_integrity(record, process, resume)
        context_generation = (
            resume.context_generation
            if resume is not None
            else self._store.get_llm_context_generation(pid)
        )
        (
            requirements,
            requirement_labels,
            requirement_binding,
        ) = self._prompt_requirement_views(
            record,
            pid=pid,
            context_generation=str(context_generation),
        )
        with self._condition:
            self._prompt_requirement_bindings[
                (pid, str(context_generation))
            ] = requirement_binding
        labels_to_aggregate.extend(requirement_labels)
        transcript_messages: list[dict[str, Any]] = []
        compressed_summary: str | None = None
        if resume is not None:
            transcript = self._decode_payload(
                self._store.get_task_run_payload(resume.transcript_payload_id),
                role="transcript",
            )
            raw_messages = transcript.get("transcript_messages", [])
            if not isinstance(raw_messages, list) or any(
                not self._valid_replay_message(item) for item in raw_messages
            ):
                self._mark_attention(
                    record,
                    self._blocker(
                        "payload_corrupt", "resume transcript shape is invalid"
                    ),
                )
                raise ValidationError("TaskRun resume transcript is invalid")
            transcript_messages = [dict(item) for item in raw_messages]
            raw_transcript_labels = transcript.get("data_labels")
            if raw_transcript_labels is not None:
                labels_to_aggregate.append(
                    DataLabels.from_dict(raw_transcript_labels)
                )
            context_generation = resume.context_generation
            if resume.summary_payload_id is not None:
                summary = self._decode_payload(
                    self._store.get_task_run_payload(resume.summary_payload_id),
                    role="summary",
                )
                value = summary.get("summary")
                if value is not None and not isinstance(value, (str, Mapping)):
                    raise ValidationError("TaskRun compressed summary is invalid")
                compressed_summary = (
                    value
                    if isinstance(value, str)
                    else (
                        canonical_task_run_json(dict(value))
                        if value is not None
                        else None
                    )
                )
                raw_summary_labels = summary.get("data_labels")
                if raw_summary_labels is not None:
                    labels_to_aggregate.append(
                        DataLabels.from_dict(raw_summary_labels)
                    )
        labels = DataLabels.aggregate(labels_to_aggregate)
        return {
            "schema_version": 1,
            "run_id": run_id,
            "context_generation": str(context_generation),
            "goal_text": self._content_text(goal_wrapper["goal"]),
            "requirements": requirements,
            "transcript_messages": transcript_messages,
            "compressed_summary": compressed_summary,
            "data_labels": labels.to_dict(),
        }

    def _require_prompt_resume_integrity(
        self,
        record: TaskRunRecord,
        process: Any,
        resume: TaskRunResumePoint,
    ) -> None:
        if not self._resume_point_identity_valid(
            resume,
            record=record,
            process=process,
        ) or not self._resume_static_integrity_valid(resume):
            self._mark_attention(
                record,
                self._blocker(
                    "payload_corrupt",
                    "resume point failed integrity before Provider dispatch",
                    pid=resume.pid,
                ),
            )
            raise ValidationError("TaskRun prompt resume point failed integrity")
        if not self._resume_current_binding_valid(
            resume,
            record=record,
            process=process,
        ):
            self._mark_attention(
                record,
                self._blocker(
                    "binding_drift",
                    "resume point binding changed before Provider dispatch",
                    pid=resume.pid,
                ),
            )
            raise ValidationError("TaskRun prompt resume binding changed")

    def _prompt_requirement_views(
        self,
        record: TaskRunRecord,
        *,
        pid: str,
        context_generation: str,
    ) -> tuple[list[dict[str, Any]], list[DataLabels], dict[str, Any]]:
        views: list[dict[str, Any]] = []
        labels: list[DataLabels] = []
        visible_requirements = self._requirements_visible_to_prompt(
            record,
            pid=pid,
            context_generation=context_generation,
        )
        for requirement in visible_requirements:
            payload = self._store.get_task_run_payload(requirement.payload_id)
            expected_role = (
                "goal"
                if requirement.kind is TaskRunRequirementKind.INITIAL
                else "follow_up"
            )
            if (
                payload is None
                or payload.run_id != record.run_id
                or payload.role != expected_role
                or payload.sha256 != requirement.requirement_sha256
            ):
                self._mark_attention(
                    record,
                    self._blocker(
                        "payload_corrupt",
                        "requirement payload lost its Run/role/hash binding",
                    ),
                )
                raise ValidationError("TaskRun requirement payload binding is invalid")
            decoded = self._decode_payload(payload, role=expected_role)
            raw_labels = decoded.get("data_labels")
            if not isinstance(raw_labels, Mapping):
                raise ValidationError("TaskRun requirement data labels are missing")
            labels.append(DataLabels.from_dict(raw_labels))
            content = decoded.get("goal", decoded.get("body", decoded))
            views.append(
                {
                    "requirement_id": requirement.requirement_id,
                    "kind": requirement.kind.value,
                    "content_text": self._content_text(content),
                    "status": requirement.status.value,
                }
            )
        return (
            views,
            labels,
            self._prompt_requirement_binding(
                record,
                pid=pid,
                context_generation=context_generation,
                requirements=visible_requirements,
            ),
        )

    def _requirements_visible_to_prompt(
        self,
        record: TaskRunRecord,
        *,
        pid: str,
        context_generation: str,
    ) -> tuple[TaskRunRequirement, ...]:
        """Bind the exact currently visible requirements to this local turn."""

        # The caller's Run projection can legitimately predate a concurrent
        # atomic follow-up append.  Refresh the projection before validating
        # the exact requirement count; if it changes between the two reads,
        # retry rather than misclassifying a legal append as corruption.
        for attempt in range(4):
            with self._uow.transaction():
                current = self._require_run(record.run_id)
                try:
                    requirements = self._bounded_completion_requirements(current)
                except ValidationError:
                    latest = self._require_run(record.run_id)
                    if (
                        latest.revision != current.revision
                        or latest.runtime_epoch != current.runtime_epoch
                    ):
                        if attempt < 3:
                            continue
                        raise TaskRunRevisionConflict(
                            "TaskRun prompt requirements kept changing"
                        )
                    raise
                if current.status is not TaskRunStatus.RUNNING:
                    return requirements
                return self._mark_prompt_requirements_in_progress(
                    current,
                    requirements=requirements,
                    pid=pid,
                    context_generation=context_generation,
                )
        raise TaskRunRevisionConflict("TaskRun prompt requirements kept changing")

    def _mark_prompt_requirements_in_progress(
        self,
        record: TaskRunRecord,
        *,
        requirements: Iterable[TaskRunRequirement],
        pid: str,
        context_generation: str,
    ) -> tuple[TaskRunRequirement, ...]:
        visible: list[TaskRunRequirement] = []
        now = utc_now()
        for requirement in requirements:
            selected = requirement
            if requirement.status is TaskRunRequirementStatus.PENDING:
                selected = self._store.update_task_run_requirement_cas(
                    requirement.requirement_id,
                    expected_status=TaskRunRequirementStatus.PENDING,
                    status=TaskRunRequirementStatus.IN_PROGRESS,
                    updated_at=now,
                    started_at=now,
                )
                if selected is None:
                    raise TaskRunRevisionConflict(
                        "TaskRun requirement visibility raced another mutation"
                    )
                self._append_ledger(
                    record.run_id,
                    kind=TaskRunLedgerKind.REQUIREMENT,
                    status=TaskRunRequirementStatus.IN_PROGRESS.value,
                    label="requirement entered reconstructed local prompt",
                    requirement_id=requirement.requirement_id,
                    pid=pid,
                    metadata={
                        "context_generation": context_generation,
                        "visibility": "local_prompt",
                    },
                )
            visible.append(selected)
        return tuple(visible)

    def requirement_binding_for_prompt(
        self,
        pid: str,
        *,
        context_generation: str,
    ) -> Mapping[str, Any] | None:
        """Return the exact non-content requirement set frozen for one prompt.

        This short-lived handoff is copied into the durable local LLM-call row
        before a Provider release can wait or dispatch.  It is never completion
        evidence by itself.
        """

        generation = str(context_generation)
        with self._condition:
            binding = self._prompt_requirement_bindings.get((pid, generation))
            if binding is None:
                return None
            return json.loads(canonical_task_run_json(binding))

    def completion_contract_for_pid(
        self,
        pid: str,
    ) -> Mapping[str, Any] | None:
        """Return the integrity-bound root contract used by completion review.

        A Durable Run deliberately stores only a non-secret marker in the root
        Process goal Object.  Completion review must therefore resolve the
        authoritative Run payloads instead of scanning that marker.  Content
        returned here is internal to the review builder: the model-visible
        review receives only hashes, requirement identities, and references.
        """

        process = self._store.get_process(pid)
        run_id = getattr(process, "task_run_id", None) if process is not None else None
        if process is None or run_id is None:
            return None
        record = self._require_run(run_id)
        if record.root_pid != pid or getattr(process, "task_run_role", None) != "root":
            return None
        if (
            record.runtime_epoch != self._runtime_epoch
            or getattr(process, "task_run_epoch", None) != self._runtime_epoch
        ):
            raise TaskRunRevisionConflict(
                "TaskRun completion review lost its epoch fence"
            )

        try:
            requirements = self._bounded_completion_requirements(record)
        except ValidationError as exc:
            raise TaskRunCompletionContractError(
                "TaskRun completion requirement set is invalid",
                run_id=record.run_id,
                blocker=self._blocker(
                    "payload_corrupt",
                    "completion review requirement set is missing or unbounded",
                ),
            ) from exc

        views: list[dict[str, Any]] = []
        goal_text: str | None = None
        goal_payload_sha256: str | None = None
        try:
            for requirement in requirements:
                view = self._completion_requirement_contract_view(
                    record,
                    requirement,
                )
                views.append(view)
                if requirement.kind is TaskRunRequirementKind.INITIAL:
                    goal_text = str(view["content_text"])
                    goal_payload_sha256 = requirement.requirement_sha256
        except (NotFound, TypeError, ValueError, ValidationError) as exc:
            raise TaskRunCompletionContractError(
                "TaskRun completion payload failed integrity validation",
                run_id=record.run_id,
                blocker=self._blocker(
                    "payload_corrupt",
                    "completion review payload failed integrity validation",
                ),
            ) from exc
        if goal_text is None or goal_payload_sha256 is None:
            raise TaskRunCompletionContractError(
                "TaskRun completion goal payload is missing",
                run_id=record.run_id,
                blocker=self._blocker(
                    "payload_corrupt",
                    "completion review has no initial goal payload",
                ),
            )
        receipts = self._completion_causal_tool_receipts(
            pid=pid,
            record=record,
            requirements=requirements,
        )
        receipt_index = self._completion_requirement_receipt_index(receipts)
        for view in views:
            requirement_id = str(view["requirement_id"])
            eligible = [
                {
                    "receipt_id": receipt_id,
                    "tool": tool,
                }
                for tool, receipt_id in sorted(
                    receipt_index.get(requirement_id, {}).items()
                )
            ]
            view["eligible_evidence_receipts"] = eligible
            view["eligible_evidence_tool_calls"] = sorted(
                {str(item["tool"]) for item in eligible}
            )
        return {
            "schema_version": 1,
            "run_id": run_id,
            "goal_text": goal_text,
            "goal_payload_sha256": goal_payload_sha256,
            "requirements": views,
        }

    def persist_completion_contract_error(
        self,
        error: TaskRunCompletionContractError,
    ) -> TaskRunRecord:
        """Persist a completion-integrity blocker outside the caller's Store lock."""

        if not isinstance(error, TaskRunCompletionContractError):
            raise ValidationError("TaskRun completion error has an invalid type")
        while True:
            record = self._require_run(error.run_id)
            if record.runtime_epoch != self._runtime_epoch:
                raise TaskRunRevisionConflict(
                    "stale TaskRun epoch refused completion attention"
                )
            try:
                return self._mark_attention(record, error.blocker)
            except TaskRunRevisionConflict:
                # Moving blocker persistence outside the completion review's
                # Store lock removes the lock inversion with Host controls, but
                # also permits one of those controls or a local settlement to
                # advance the Run revision before the attention CAS. Corruption
                # is not a best-effort signal: under the same Runtime epoch,
                # reread until it is durably projected or the Run is terminal.
                latest = self._require_run(error.run_id)
                if latest.runtime_epoch != self._runtime_epoch:
                    raise

    def _completion_requirement_contract_view(
        self,
        record: TaskRunRecord,
        requirement: TaskRunRequirement,
    ) -> dict[str, Any]:
        expected_role = (
            "goal"
            if requirement.kind is TaskRunRequirementKind.INITIAL
            else "follow_up"
        )
        payload = self._store.get_task_run_payload(requirement.payload_id)
        if (
            payload is None
            or payload.run_id != record.run_id
            or payload.role != expected_role
            or payload.sha256 != requirement.requirement_sha256
        ):
            raise ValidationError(
                "TaskRun completion requirement payload binding is invalid"
            )
        decoded = self._decode_payload(payload, role=expected_role)
        raw_labels = decoded.get("data_labels")
        if not isinstance(raw_labels, Mapping):
            raise ValidationError("TaskRun completion requirement labels are missing")
        DataLabels.from_dict(raw_labels)
        if requirement.kind is TaskRunRequirementKind.INITIAL:
            if requirement.ordinal != 0 or set(decoded) != {"goal", "data_labels"}:
                raise ValidationError(
                    "TaskRun completion goal payload shape is invalid"
                )
            content = decoded["goal"]
        else:
            if set(decoded) != {"body", "kind", "data_labels"}:
                raise ValidationError(
                    "TaskRun completion follow-up payload shape is invalid"
                )
            content = decoded["body"]
        return {
            "requirement_id": requirement.requirement_id,
            "ordinal": requirement.ordinal,
            "kind": requirement.kind.value,
            "status": requirement.status.value,
            "requirement_sha256": requirement.requirement_sha256,
            "created_at": requirement.created_at,
            "started_at": requirement.started_at,
            "content_text": self._content_text(content),
        }

    def _completion_causal_tool_receipts(
        self,
        *,
        pid: str,
        record: TaskRunRecord,
        requirements: Iterable[TaskRunRequirement] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return successful tool receipts causally bound to Run requirements.

        Wall-clock ordering is insufficient here: a required follow-up can be
        appended after an LLM response was selected but before that response's
        tool is dispatched.  A receipt is therefore eligible only when its
        terminal successful ``tool_call`` Operation is a child of the exact
        ``llm_request`` root whose linked durable LLM call froze the matching
        requirement id/hash in ``request_options``.  Direct Host dispatches
        have no such LLM root and deliberately do not qualify.
        """

        hard_limit = self.config.task_runs.recovery_page_hard_limit
        operations = list(
            self._uow.evidence.list_operations(
                pid=pid,
                limit=hard_limit + 1,
            )
        )
        if len(operations) > hard_limit:
            raise ValidationError(
                "TaskRun completion operation evidence exceeds its hard cap"
            )
        selected_requirements = (
            tuple(requirements)
            if requirements is not None
            else self._bounded_completion_requirements(record)
        )
        current_requirements = {
            requirement.requirement_id: requirement
            for requirement in selected_requirements
        }
        llm_operations = {
            operation.operation_id: operation
            for operation in operations
            if self._is_completion_llm_operation(operation, pid=pid)
        }
        bindings = self._completion_llm_operation_bindings(
            record=record,
            pid=pid,
            llm_operations=llm_operations,
            current_requirements=current_requirements,
            hard_limit=hard_limit,
        )
        tool_operations = [
            operation
            for operation in operations
            if self._is_completion_tool_operation(operation, pid=pid)
        ]
        valid_tool_operation_ids = self._completion_tool_result_operation_ids(
            tool_operations,
            hard_limit=hard_limit,
        )
        receipts: list[dict[str, Any]] = []
        expanded_binding_count = 0
        for operation in tool_operations:
            llm_operation = llm_operations.get(operation.parent_operation_id or "")
            requirement_ids = bindings.get(operation.parent_operation_id or "", ())
            if (
                operation.operation_id not in valid_tool_operation_ids
                or llm_operation is None
                or operation.root_operation_id != llm_operation.root_operation_id
                or not requirement_ids
            ):
                continue
            expanded_binding_count += len(requirement_ids)
            if expanded_binding_count > hard_limit:
                raise ValidationError(
                    "TaskRun completion requirement/receipt projection exceeds "
                    "its hard cap"
                )
            receipts.append(
                {
                    "receipt_id": operation.operation_id,
                    "tool": operation.name.removeprefix("tool."),
                    "completed_at": operation.completed_at,
                    "requirement_ids": list(requirement_ids),
                }
            )
        receipts.sort(
            key=lambda item: (
                str(item["completed_at"]),
                str(item["receipt_id"]),
            )
        )
        return tuple(receipts)

    def _bounded_completion_requirements(
        self,
        record: TaskRunRecord,
    ) -> tuple[TaskRunRequirement, ...]:
        hard_limit = self.config.task_runs.recovery_page_hard_limit
        if (
            type(record.requirement_count) is not int
            or record.requirement_count <= 0
            or record.requirement_count > hard_limit
        ):
            raise ValidationError(
                "TaskRun completion requirement count is outside its hard cap"
            )
        requirements = tuple(
            self._store.list_task_run_requirements(
                record.run_id,
                limit=hard_limit + 1,
            )
        )
        if (
            len(requirements) > hard_limit
            or len(requirements) != record.requirement_count
            or any(
                requirement.run_id != record.run_id
                or requirement.ordinal != ordinal
                for ordinal, requirement in enumerate(requirements)
            )
        ):
            raise ValidationError(
                "TaskRun completion requirement projection is inconsistent"
            )
        return requirements

    @staticmethod
    def _is_completion_llm_operation(operation: Any, *, pid: str) -> bool:
        return bool(
            operation.pid == pid
            and operation.actor == pid
            and operation.kind is OperationKind.LLM_REQUEST
            and operation.name == "llm.action_selection"
            and operation.operation_id == operation.root_operation_id
            and operation.state.value == "terminal"
            and operation.outcome is OperationOutcome.SUCCEEDED
            and operation.completed_at is not None
        )

    @staticmethod
    def _is_completion_tool_operation(operation: Any, *, pid: str) -> bool:
        return bool(
            operation.pid == pid
            and operation.actor == pid
            and operation.kind is OperationKind.TOOL_CALL
            and operation.name.startswith("tool.")
            and operation.name != "tool.process_exit"
            and operation.state.value == "terminal"
            and operation.outcome is OperationOutcome.SUCCEEDED
            and operation.completed_at is not None
        )

    def _completion_llm_operation_bindings(
        self,
        *,
        record: TaskRunRecord,
        pid: str,
        llm_operations: Mapping[str, Any],
        current_requirements: Mapping[str, TaskRunRequirement],
        hard_limit: int,
    ) -> dict[str, tuple[str, ...]]:
        if not llm_operations:
            return {}
        links = self._uow.evidence.list_operation_evidence(
            operation_ids=tuple(llm_operations),
            evidence_types=("llm_call",),
            limit=hard_limit + 1,
        )
        if len(links) > hard_limit:
            raise ValidationError(
                "TaskRun completion LLM evidence exceeds its hard cap"
            )
        grouped: dict[str, list[Any]] = {}
        for link in links:
            if link.role == "invocation" and link.metadata.get("status") == "ok":
                grouped.setdefault(link.operation_id, []).append(link)
        selected: dict[str, tuple[str, ...]] = {}
        expanded_binding_count = 0
        for operation_id in llm_operations:
            candidates = grouped.get(operation_id, [])
            if len(candidates) != 1:
                continue
            call = self._store.get_llm_call(candidates[0].evidence_id)
            if (
                call is None
                or call.pid != pid
                or call.status != "ok"
                or call.purpose != "action_selection"
                or not call.completed_at
            ):
                continue
            entries = self._completion_causal_binding_entries(
                record=record,
                pid=pid,
                value=call.request_options.get(_TASK_RUN_REQUIREMENT_BINDING_KEY),
                current_requirements=current_requirements,
            )
            if entries:
                expanded_binding_count += len(entries)
                if expanded_binding_count > hard_limit:
                    raise ValidationError(
                        "TaskRun completion LLM requirement binding projection "
                        "exceeds its hard cap"
                    )
                selected[operation_id] = entries
        return selected

    @staticmethod
    def _completion_causal_binding_entries(
        *,
        record: TaskRunRecord,
        pid: str,
        value: Any,
        current_requirements: Mapping[str, TaskRunRequirement],
    ) -> tuple[str, ...]:
        if (
            not isinstance(value, Mapping)
            or set(value) != _PROMPT_REQUIREMENT_BINDING_KEYS
            or value.get("schema_version") != 1
            or type(value.get("schema_version")) is not int
            or value.get("run_id") != record.run_id
            or value.get("pid") != pid
            or not isinstance(value.get("context_generation"), str)
            or not value.get("context_generation")
            or not isinstance(value.get("requirements"), list)
        ):
            return ()
        selected: list[str] = []
        prior_ordinal = -1
        for raw in value["requirements"]:
            if not isinstance(raw, Mapping) or set(raw) != _PROMPT_REQUIREMENT_ENTRY_KEYS:
                return ()
            requirement_id = raw.get("requirement_id")
            ordinal = raw.get("ordinal")
            requirement = (
                current_requirements.get(requirement_id)
                if isinstance(requirement_id, str)
                else None
            )
            if (
                requirement is None
                or requirement_id in selected
                or type(ordinal) is not int
                or ordinal <= prior_ordinal
                or requirement.ordinal != ordinal
                or requirement.requirement_sha256 != raw.get("requirement_sha256")
            ):
                return ()
            selected.append(requirement_id)
            prior_ordinal = ordinal
        return tuple(selected)

    def _completion_tool_result_operation_ids(
        self,
        operations: Iterable[Any],
        *,
        hard_limit: int,
    ) -> set[str]:
        selected = tuple(operations)
        if not selected:
            return set()
        links = self._uow.evidence.list_operation_evidence(
            operation_ids=(operation.operation_id for operation in selected),
            evidence_types=("tool_call",),
            limit=hard_limit + 1,
        )
        if len(links) > hard_limit:
            raise ValidationError(
                "TaskRun completion tool evidence exceeds its hard cap"
            )
        grouped: dict[str, list[Any]] = {}
        for link in links:
            grouped.setdefault(link.operation_id, []).append(link)
        valid: set[str] = set()
        for operation in selected:
            tool_name = operation.name.removeprefix("tool.")
            candidates = grouped.get(operation.operation_id, [])
            invocations = [
                link
                for link in candidates
                if link.role == "invocation"
                and link.metadata.get("tool") == tool_name
            ]
            results = [
                link
                for link in candidates
                if link.role == "result" and link.metadata.get("ok") is True
            ]
            if (
                len(invocations) == 1
                and len(results) == 1
                and invocations[0].evidence_id == results[0].evidence_id
            ):
                valid.add(operation.operation_id)
        return valid

    def _prompt_requirement_binding(
        self,
        record: TaskRunRecord,
        *,
        pid: str,
        context_generation: str,
        requirements: Iterable[TaskRunRequirement],
    ) -> dict[str, Any]:
        binding = {
            "schema_version": 1,
            "run_id": record.run_id,
            "pid": pid,
            "context_generation": str(context_generation),
            "requirements": [
                {
                    "requirement_id": requirement.requirement_id,
                    "ordinal": requirement.ordinal,
                    "requirement_sha256": requirement.requirement_sha256,
                }
                for requirement in requirements
                if requirement.status is TaskRunRequirementStatus.IN_PROGRESS
            ],
        }
        canonical_task_run_json(binding)
        return binding

    def record_validated_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        action_manifest: Mapping[str, Any],
        context_generation: str | int,
    ) -> None:
        """Persist the complete validated local action before dispatch."""

        process, record = self._bound_process_run(pid)
        if record is None:
            return
        # The Provider may have been admitted immediately before a Host pause
        # persisted its control generation.  Its completed local call and the
        # normalized, still-unstarted action remain safe to commit while the
        # Run is paused; only the subsequent tool admission is refused.
        self._require_settlement_epoch(process, record, "LLM action settlement")
        selected_call_id = self._identifier(call_id, "LLM call_id")
        try:
            manifest = normalize_validated_action_manifest(action_manifest)
            self._require_supported_binding_action_manifest(
                record,
                pid=pid,
                manifest=manifest,
            )
            labels = DataLabels.from_dict(manifest["data_labels"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("TaskRun action manifest is invalid") from exc
        if manifest["call_id"] != selected_call_id:
            raise ValidationError("TaskRun action manifest call_id changed")
        llm_call_sha256 = self._local_llm_call_sha256(process, selected_call_id)
        manifest_sha256 = self._sha256(manifest)
        generation = str(context_generation)
        requirement_binding = self._llm_prompt_requirement_binding(
            process,
            record,
            call_id=selected_call_id,
            context_generation=generation,
        )
        now = utc_now()
        prior = self._store.get_task_run_resume_point(pid, complete_only=True)
        if prior is not None and not self._resume_integrity_valid(prior):
            raise ValidationError("TaskRun prior resume point failed integrity")
        if prior is not None and prior.pending_action_payload_id is not None:
            existing = self._decode_pending_resume_payload(prior)
            if (
                existing.get("kind") == "validated_action"
                and existing.get("call_id") == selected_call_id
                and existing.get("context_generation") == generation
                and existing.get("manifest") == manifest
                and existing.get("manifest_sha256") == manifest_sha256
                and existing.get("llm_call_sha256") == llm_call_sha256
                and existing.get("requirement_binding") == requirement_binding
                and existing.get("state") in {"validated", "dispatching"}
            ):
                return
            raise TaskRunRevisionConflict(
                "TaskRun already has another pending local action"
            )

        transcript_payload: TaskRunPayload
        insert_transcript = False
        if prior is None:
            transcript_payload = TaskRunPayload.plaintext(
                payload_id=new_id("trp"),
                run_id=record.run_id,
                role="transcript",
                label="TaskRun base local transcript",
                value={
                    "schema_version": 1,
                    "call_id": selected_call_id,
                    "transcript_messages": [],
                    "data_labels": labels.to_dict(),
                },
                created_at=now,
            )
            insert_transcript = True
        else:
            transcript_payload = self._store.get_task_run_payload(
                prior.transcript_payload_id
            )
            if transcript_payload is None:
                raise ValidationError("TaskRun prior transcript is missing")
        summary_payload = (
            self._store.get_task_run_payload(prior.summary_payload_id)
            if prior is not None and prior.summary_payload_id is not None
            else None
        )
        image_hash, _tool_hash, provider_hash = self._process_binding_hashes(process)
        action_pre_binding = pre_action_binding(
            process,
            image_binding_hash=image_hash,
            provider_binding_hash=provider_hash,
        )
        pending_payload = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=record.run_id,
            role="pending_action",
            label="Validated local TaskRun action",
            value={
                "schema_version": 1,
                "kind": "validated_action",
                "state": "validated",
                "call_id": selected_call_id,
                "context_generation": generation,
                "manifest": manifest,
                "manifest_sha256": manifest_sha256,
                "llm_call_sha256": llm_call_sha256,
                "pre_action_binding": action_pre_binding,
                "requirement_binding": requirement_binding,
            },
            created_at=now,
        )
        self._require_payload_bound(pending_payload)
        self._commit_validated_action(
            process=process,
            run_id=record.run_id,
            pid=pid,
            call_id=selected_call_id,
            generation=generation,
            manifest_sha256=manifest_sha256,
            llm_call_sha256=llm_call_sha256,
            prior=prior,
            transcript_payload=transcript_payload,
            summary_payload=summary_payload,
            pending_payload=pending_payload,
            action_pre_binding=action_pre_binding,
            insert_transcript=insert_transcript,
            now=now,
        )
        with self._condition:
            self._prompt_requirement_bindings.pop((pid, generation), None)
        self._notify_updated()

    def _require_supported_binding_action_manifest(
        self,
        record: TaskRunRecord,
        *,
        pid: str,
        manifest: Mapping[str, Any],
    ) -> None:
        actions = [dict(action) for action in manifest["actions"]]
        if not any(action.get("action") == "activate_skill" for action in actions):
            return
        supported = (
            len(actions) == 1
            and actions[0].get("action") == "activate_skill"
            and manifest["parallel_tool_calls"] is False
            and manifest["host_auto_wait"] is False
        )
        if supported:
            return
        self._mark_attention(
            record,
            self._blocker(
                "binding_drift",
                "activate_skill must be one non-parallel durable action",
                pid=pid,
            ),
        )
        raise ValidationError(
            "Durable TaskRun activate_skill must be a singleton non-parallel action"
        )

    def _commit_validated_action(
        self,
        *,
        process: Any,
        run_id: str,
        pid: str,
        call_id: str,
        generation: str,
        manifest_sha256: str,
        llm_call_sha256: str,
        prior: TaskRunResumePoint | None,
        transcript_payload: TaskRunPayload,
        summary_payload: TaskRunPayload | None,
        pending_payload: TaskRunPayload,
        action_pre_binding: Mapping[str, Any],
        insert_transcript: bool,
        now: str,
    ) -> None:
        for attempt in range(3):
            current_process = self._store.get_process(pid)
            current = self._require_run(run_id)
            if current_process is None:
                raise NotFound(f"process not found: {pid}")
            self._require_settlement_epoch(
                current_process,
                current,
                "LLM action settlement",
            )
            self._require_pre_action_binding_matches_process(
                action_pre_binding,
                current_process,
            )
            point = self._make_resume_point(
                process=current_process,
                record=current,
                context_generation=generation,
                safe_point_seq=(
                    prior.safe_point_seq + 1 if prior is not None else 1
                ),
                transcript_payload=transcript_payload,
                summary_payload=summary_payload,
                pending_payload=pending_payload,
                last_effect_seq=self._current_effect_seq(),
                created_at=prior.created_at if prior is not None else now,
                updated_at=now,
            )
            try:
                with self._uow.transaction():
                    if insert_transcript:
                        self._store.insert_task_run_payload(transcript_payload)
                    self._store.insert_task_run_payload(pending_payload)
                    self._store.upsert_task_run_resume_point(point)
                    updated = self._store.update_task_run_cas(
                        run_id,
                        current.revision,
                        updates={
                            "step_count": current.step_count + 1,
                            "active_pid": pid,
                            "updated_at": now,
                        },
                        expected_runtime_epoch=self._runtime_epoch,
                    )
                    ledger = self._append_ledger(
                        run_id,
                        kind=TaskRunLedgerKind.LLM_TURN,
                        status="validated",
                        label="LLM action validated before dispatch",
                        pid=pid,
                        llm_call_id=call_id,
                        payload_id=pending_payload.payload_id,
                        metadata={
                            "context_generation": generation,
                            "action_manifest_sha256": manifest_sha256,
                            "llm_call_sha256": llm_call_sha256,
                            "safe_point_seq": point.safe_point_seq,
                            "dispatch_state": "validated",
                        },
                    )
                    self._store.insert_task_run_link(
                        TaskRunLink(
                            link_id=new_id("trlink"),
                            run_id=run_id,
                            ledger_seq=ledger.seq,
                            evidence_type="llm_call",
                            evidence_id=call_id,
                            role="validated",
                            created_at=now,
                            metadata={"revision": updated.revision},
                        )
                    )
                return
            except TaskRunRevisionConflict:
                if attempt == 2:
                    raise

    def pending_validated_action_for_pid(
        self,
        pid: str,
    ) -> Mapping[str, Any] | None:
        """Atomically claim one validated action for a single local dispatch."""

        process, record = self._bound_process_run(pid)
        if record is None:
            return None
        if record.status in {TaskRunStatus.PAUSED, TaskRunStatus.CANCELLING}:
            raise TaskRunDispatchDeferred(
                f"TaskRun {record.status.value} deferred validated tool dispatch"
            )
        self.require_process_epoch(
            pid,
            record.run_id,
            getattr(process, "task_run_epoch", -1),
            "validated local action dispatch",
        )
        point = self._store.get_task_run_resume_point(pid, complete_only=True)
        if point is None or point.pending_action_payload_id is None:
            return None
        if not self._resume_integrity_valid(point):
            self._mark_attention(
                record,
                self._blocker(
                    "payload_corrupt",
                    "pending local action failed resume integrity",
                    pid=pid,
                ),
            )
            raise ValidationError("TaskRun pending local action is corrupt")
        wrapper = self._decode_pending_resume_payload(point)
        if wrapper.get("kind") != "validated_action":
            if wrapper.get("kind") in {"completed_outcome", "durable_wait_action"}:
                return None
            raise ValidationError("TaskRun pending action kind is invalid")
        state = wrapper.get("state")
        if state == "dispatching":
            self._mark_attention(
                record,
                self._blocker(
                    "pending_action_unreplayable",
                    "validated action dispatch was interrupted after its durable claim",
                    pid=pid,
                ),
            )
            raise ValidationError("TaskRun action dispatch outcome is unknown")
        if state != "validated":
            raise ValidationError("TaskRun validated action state is invalid")
        try:
            manifest = normalize_validated_action_manifest(wrapper["manifest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("TaskRun validated action manifest is invalid") from exc
        if (
            wrapper.get("call_id") != manifest["call_id"]
            or wrapper.get("manifest_sha256") != self._sha256(manifest)
            or wrapper.get("llm_call_sha256")
            != self._local_llm_call_sha256(process, str(manifest["call_id"]))
        ):
            raise ValidationError("TaskRun validated action lost its call binding")
        self._validated_action_pre_binding(
            wrapper,
            point,
            require_point_revision=True,
        )
        if self._effects_changed_after_resume((process,)) or self._unsettled_effects(
            record.run_id
        ):
            self._mark_attention(
                record,
                self._blocker(
                    "unknown_effect",
                    "effect evidence changed before validated action claim",
                    pid=pid,
                ),
            )
            raise ValidationError("TaskRun validated action lost its effect fence")
        now = utc_now()
        claimed_payload = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=record.run_id,
            role="pending_action",
            label="Claimed local TaskRun action",
            value={**wrapper, "state": "dispatching"},
            created_at=now,
        )
        self._require_payload_bound(claimed_payload)
        transcript = self._store.get_task_run_payload(point.transcript_payload_id)
        summary = (
            self._store.get_task_run_payload(point.summary_payload_id)
            if point.summary_payload_id is not None
            else None
        )
        if transcript is None:
            raise ValidationError("TaskRun pending transcript is missing")
        claimed_point = self._make_resume_point(
            process=process,
            record=record,
            context_generation=point.context_generation,
            safe_point_seq=point.safe_point_seq + 1,
            transcript_payload=transcript,
            summary_payload=summary,
            pending_payload=claimed_payload,
            last_effect_seq=self._current_effect_seq(),
            created_at=point.created_at,
            updated_at=now,
        )
        with self._uow.transaction():
            self._store.insert_task_run_payload(claimed_payload)
            self._store.upsert_task_run_resume_point(claimed_point)
            self._append_ledger(
                record.run_id,
                kind=TaskRunLedgerKind.LLM_TURN,
                status="dispatching",
                label="validated local action dispatch claimed",
                pid=pid,
                llm_call_id=str(manifest["call_id"]),
                payload_id=claimed_payload.payload_id,
                metadata={"safe_point_seq": claimed_point.safe_point_seq},
            )
        self._notify_updated()
        return manifest

    def expected_tool_id_for_pending_action(
        self,
        pid: str,
        action: Mapping[str, Any],
    ) -> str | None:
        """Resolve one dispatch to the exact identity sealed before the action."""

        process, record = self._bound_process_run(pid)
        if record is None:
            return None
        self.require_process_epoch(
            pid,
            record.run_id,
            getattr(process, "task_run_epoch", -1),
            "exact TaskRun tool dispatch",
        )
        point = self._store.get_task_run_resume_point(pid, complete_only=True)
        if point is None or point.pending_action_payload_id is None:
            return None
        try:
            if not self._resume_integrity_valid(point):
                raise ValidationError(
                    "TaskRun binding changed before exact tool dispatch"
                )
            wrapper = self._decode_pending_resume_payload(point)
            kind = wrapper.get("kind")
            if kind == "validated_action":
                return self._validated_action_tool_id(
                    process,
                    point,
                    wrapper,
                    action,
                )
            if kind == "durable_wait_action":
                return self._durable_wait_tool_id(process, wrapper, action)
            if kind == "completed_outcome":
                return None
            raise ValidationError("TaskRun pending action kind is invalid")
        except ValidationError:
            self._mark_attention(
                record,
                self._blocker(
                    "binding_drift",
                    "exact pre-action tool binding could not be certified",
                    pid=pid,
                ),
            )
            raise

    def request_binding_hash_for_pid(self, pid: str) -> str | None:
        process, record = self._bound_process_run(pid)
        if record is None:
            return None
        self.require_process_epoch(
            pid,
            record.run_id,
            getattr(process, "task_run_epoch", -1),
            "TaskRun LLM request binding",
        )
        return self._binding_hash_for_process(process)

    def settlement_binding_hash_for_pid(self, pid: str) -> str | None:
        """Validate an already-admitted Provider call without reopening dispatch."""

        process, record = self._bound_process_run(pid)
        if record is None:
            return None
        self._require_settlement_epoch(
            process,
            record,
            "TaskRun LLM request settlement binding",
        )
        return self._binding_hash_for_process(process)

    @contextmanager
    def dispatch_scope_for_pid(self, pid: str, kind: str):
        """Atomically admit one external call against persisted control state.

        The condition lock serializes this admission with pause/interrupt
        intent persistence.  It is deliberately released during the call: a
        controller can persist PAUSED immediately, while the active counter
        tells it which already-admitted call still owns local settlement.
        """

        selected_kind = self._identifier(kind, "dispatch kind")
        if selected_kind not in {"provider", "tool"}:
            raise ValidationError("TaskRun dispatch kind is unsupported")
        _, record = self._bound_process_run(pid)
        if record is None:
            yield
            return
        with self._condition:
            current_process, current = self._bound_process_run(pid)
            if current is None:
                raise TaskRunDispatchDeferred(
                    "TaskRun binding disappeared before external dispatch"
                )
            if current.status is not TaskRunStatus.RUNNING:
                raise TaskRunDispatchDeferred(
                    f"TaskRun {current.status.value} refused new {selected_kind} dispatch"
                )
            self.require_process_epoch(
                pid,
                current.run_id,
                getattr(current_process, "task_run_epoch", -1),
                f"{selected_kind} dispatch admission",
            )
            self._active_external_dispatches[current.run_id] = (
                self._active_external_dispatches.get(current.run_id, 0) + 1
            )
        context_token = self._external_dispatch_context.set(
            (current.run_id, pid, selected_kind)
        )
        try:
            yield
        finally:
            self._external_dispatch_context.reset(context_token)
            with self._condition:
                remaining = self._active_external_dispatches.get(record.run_id, 0) - 1
                if remaining > 0:
                    self._active_external_dispatches[record.run_id] = remaining
                else:
                    self._active_external_dispatches.pop(record.run_id, None)
                self._condition.notify_all()

    def defer_unstarted_action_for_pid(self, pid: str) -> None:
        """Rewind a durable claim after the tool admission gate refused it."""

        process, record = self._bound_process_run(pid)
        if record is None:
            return
        if record.status not in {TaskRunStatus.PAUSED, TaskRunStatus.CANCELLING}:
            raise ValidationError("TaskRun action deferral requires persisted control intent")
        point = self._store.get_task_run_resume_point(pid, complete_only=True)
        if point is None or point.pending_action_payload_id is None:
            return
        if not self._resume_integrity_valid(point):
            raise ValidationError("TaskRun deferred action failed resume integrity")
        wrapper = self._decode_pending_resume_payload(point)
        if wrapper.get("kind") != "validated_action":
            return
        if wrapper.get("state") == "validated":
            return
        if wrapper.get("state") != "dispatching":
            raise ValidationError("TaskRun deferred action state is invalid")
        if self._changed_effects_for_pid(pid, point.last_effect_seq):
            self._mark_attention(
                record,
                self._blocker(
                    "unknown_effect",
                    "tool evidence changed before paused action deferral",
                    pid=pid,
                ),
            )
            raise ValidationError("TaskRun deferred action may have been dispatched")
        now = utc_now()
        pending = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=record.run_id,
            role="pending_action",
            label="Control-deferred local TaskRun action",
            value={**dict(wrapper), "state": "validated"},
            created_at=now,
        )
        self._require_payload_bound(pending)
        transcript = self._store.get_task_run_payload(point.transcript_payload_id)
        summary = (
            self._store.get_task_run_payload(point.summary_payload_id)
            if point.summary_payload_id is not None
            else None
        )
        if transcript is None:
            raise ValidationError("TaskRun deferred action transcript is missing")
        rewound = self._make_resume_point(
            process=process,
            record=record,
            context_generation=point.context_generation,
            safe_point_seq=point.safe_point_seq + 1,
            transcript_payload=transcript,
            summary_payload=summary,
            pending_payload=pending,
            last_effect_seq=self._current_effect_seq(),
            created_at=point.created_at,
            updated_at=now,
        )
        with self._uow.transaction():
            self._store.insert_task_run_payload(pending)
            self._store.upsert_task_run_resume_point(rewound)
            self._append_ledger(
                record.run_id,
                kind=TaskRunLedgerKind.LLM_TURN,
                status="validated",
                label="persisted control generation deferred unstarted tool action",
                pid=pid,
                llm_call_id=str(wrapper.get("call_id")),
                payload_id=pending.payload_id,
                metadata={"safe_point_seq": rewound.safe_point_seq},
            )

    def mark_request_scope_drift_for_pid(self, pid: str) -> None:
        process, record = self._bound_process_run(pid)
        if record is None:
            return
        self._require_settlement_epoch(
            process,
            record,
            "TaskRun LLM request binding drift",
        )
        self._mark_attention(
            record,
            self._blocker(
                "binding_drift",
                "LLM request binding changed before its action could commit",
                pid=pid,
            ),
        )

    def _validated_action_tool_id(
        self,
        process: Any,
        point: TaskRunResumePoint,
        wrapper: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> str:
        if wrapper.get("state") != "dispatching":
            raise ValidationError("TaskRun action was not durably claimed")
        try:
            manifest = normalize_validated_action_manifest(wrapper["manifest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("TaskRun validated action manifest is invalid") from exc
        selected_action = dict(action)
        if selected_action not in manifest["actions"]:
            raise ValidationError("TaskRun dispatch changed its validated action")
        pre_binding = self._validated_action_pre_binding(wrapper, point)
        name = self._identifier(selected_action.get("action"), "action name")
        expected_tool_id = pre_binding["tool_table"].get(name)
        model_tool_id = pre_binding["model_tool_table"].get(name)
        if not isinstance(expected_tool_id, str) or not expected_tool_id:
            raise ValidationError("TaskRun validated tool binding is missing")
        if not manifest["host_auto_wait"] and model_tool_id != expected_tool_id:
            raise ValidationError("TaskRun validated model tool binding changed")
        if process.tool_table.get(name) != expected_tool_id:
            raise ValidationError("TaskRun exact tool binding changed before dispatch")
        return expected_tool_id

    def _durable_wait_tool_id(
        self,
        process: Any,
        wrapper: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> str:
        if wrapper.get("state") != "waiting":
            raise ValidationError("TaskRun durable wait state changed")
        snapshot = wrapper.get("wait_snapshot")
        if not isinstance(snapshot, Mapping) or snapshot.get("action") != dict(action):
            raise ValidationError("TaskRun resumed action changed its durable wait")
        name = self._identifier(action.get("action"), "action name")
        expected_tool_id = process.tool_table.get(name)
        if not isinstance(expected_tool_id, str) or not expected_tool_id:
            raise ValidationError("TaskRun durable wait tool binding is missing")
        return expected_tool_id

    def stage_completed_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        outcome_manifest: Mapping[str, Any],
        context_generation: str | int,
    ) -> None:
        """Durably bind a complete local result before advancing the transcript.

        This is deliberately a local-only staging seam.  It never calls a
        Provider and it refuses every unresolved effect.  A successor may
        therefore finish the transcript commit from this payload without
        replaying the LLM or the action that produced the result.
        """

        process, record = self._bound_process_run(pid)
        if record is None:
            return
        self._require_settlement_epoch(process, record, "LLM result staging")
        selected_call_id = self._identifier(call_id, "LLM call_id")
        generation = str(context_generation)
        outcome = self._validated_outcome_manifest(outcome_manifest)
        outcome_sha256 = self._sha256(outcome)
        llm_call_sha256 = self._local_llm_call_sha256(process, selected_call_id)
        prior = self._store.get_task_run_resume_point(pid, complete_only=True)
        if (
            prior is None
            or prior.pending_action_payload_id is None
            or not self._resume_point_identity_valid(
                prior,
                record=record,
                process=process,
            )
            or not self._resume_static_integrity_valid(prior)
        ):
            raise ValidationError(
                "TaskRun completed outcome has no integrity-bound pending action"
            )
        source = self._decode_pending_resume_payload(prior)
        source_binding = self._stage_source_binding(
            pid=pid,
            call_id=selected_call_id,
            generation=generation,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
            llm_call_sha256=llm_call_sha256,
            source=source,
            source_point=prior,
        )
        if source_binding[0]:
            return
        (
            action_manifest_sha256,
            source_wait_snapshot,
            generation_transition,
        ) = source_binding[1:]

        settled_effect_seq, settled_effects = self._settled_effect_bundle(
            pid,
            prior.last_effect_seq,
        )
        binding_transition = self._stage_binding_transition(
            process=process,
            prior=prior,
            source=source,
            action_manifest_sha256=action_manifest_sha256,
            outcome=outcome,
            settled_effect_seq=settled_effect_seq,
        )

        wait_snapshot = self._stage_wait_snapshot(
            pid=pid,
            call_id=selected_call_id,
            outcome=outcome,
            source_wait_snapshot=source_wait_snapshot,
        )
        requirement_completion = self._completion_requirement_evidence(
            process=process,
            record=record,
            source=source,
            call_id=selected_call_id,
            context_generation=str(
                source.get("context_generation") or generation
            ),
            action_manifest_sha256=action_manifest_sha256,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
        )
        self._persist_staged_outcome(
            process=process,
            record=record,
            prior=prior,
            call_id=selected_call_id,
            generation=generation,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
            llm_call_sha256=llm_call_sha256,
            action_manifest_sha256=action_manifest_sha256,
            settled_effect_seq=settled_effect_seq,
            settled_effects=settled_effects,
            wait_snapshot=wait_snapshot,
            source_generation=str(source.get("context_generation") or generation),
            generation_transition=generation_transition,
            binding_transition=binding_transition,
            requirement_completion=requirement_completion,
        )
        self._notify_updated()

    def _stage_source_binding(
        self,
        *,
        pid: str,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        llm_call_sha256: str,
        source: Mapping[str, Any],
        source_point: TaskRunResumePoint,
    ) -> tuple[
        bool,
        str | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        kind = source.get("kind")
        if kind == "completed_outcome":
            self._require_matching_staged_outcome(
                pid=pid,
                call_id=call_id,
                generation=generation,
                outcome=outcome,
                outcome_sha256=outcome_sha256,
                llm_call_sha256=llm_call_sha256,
                source=source,
            )
            return True, None, None, None
        if kind == "validated_action":
            return (
                False,
                self._validated_stage_action_sha256(
                    call_id,
                    generation,
                    llm_call_sha256,
                    source,
                    source_point,
                ),
                None,
                None,
            )
        if kind == "durable_wait_action":
            return self._durable_wait_stage_binding(
                pid=pid,
                call_id=call_id,
                generation=generation,
                outcome=outcome,
                outcome_sha256=outcome_sha256,
                llm_call_sha256=llm_call_sha256,
                source=source,
            )
        raise ValidationError("TaskRun completed outcome source is invalid")

    def _require_matching_staged_outcome(
        self,
        *,
        pid: str,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        llm_call_sha256: str,
        source: Mapping[str, Any],
    ) -> None:
        expected = (
            source.get("state") == "staged"
            and source.get("call_id") == call_id
            and source.get("context_generation") == generation
            and source.get("outcome") == outcome
            and source.get("outcome_sha256") == outcome_sha256
            and source.get("llm_call_sha256") == llm_call_sha256
        )
        if not expected:
            raise TaskRunRevisionConflict(
                "TaskRun already staged another completed outcome"
            )
        self._validate_staged_generation_binding(
            pid,
            source,
            generation=generation,
            outcome=outcome,
        )
        self._validate_effect_settlement_bundle(pid, source)
        self._validate_current_staged_binding_for_pid(pid, source)

    def _validated_stage_action_sha256(
        self,
        call_id: str,
        generation: str,
        llm_call_sha256: str,
        source: Mapping[str, Any],
        source_point: TaskRunResumePoint,
    ) -> str:
        try:
            manifest = normalize_validated_action_manifest(source["manifest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("TaskRun staged action manifest is invalid") from exc
        manifest_sha256 = self._sha256(manifest)
        valid = (
            source.get("state") in {"validated", "dispatching"}
            and source.get("call_id") == call_id
            and source.get("context_generation") == generation
            and source.get("manifest_sha256") == manifest_sha256
            and source.get("llm_call_sha256") == llm_call_sha256
        )
        if not valid:
            raise ValidationError("TaskRun completed outcome lost its action binding")
        self._validated_action_pre_binding(source, source_point)
        return manifest_sha256

    def _completion_requirement_evidence(
        self,
        *,
        process: Any,
        record: TaskRunRecord,
        source: Mapping[str, Any],
        call_id: str,
        context_generation: str,
        action_manifest_sha256: str | None,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
    ) -> dict[str, Any] | None:
        """Certify the one local action allowed to satisfy Run requirements."""

        if process.pid != record.root_pid:
            return None
        selected_action = self._committed_process_exit_action(outcome)
        if selected_action is None:
            return None
        if not self._completion_source_matches_action(
            source,
            selected_action=selected_action,
            action_manifest_sha256=action_manifest_sha256,
        ):
            return None
        with self._uow.transaction():
            current_record = self._require_run(record.run_id)
            self._require_settlement_epoch(
                process,
                current_record,
                "TaskRun requirement evidence staging",
            )
            requirements = self._bounded_completion_requirements(current_record)
            current_requirements = {
                requirement.requirement_id: requirement
                for requirement in requirements
            }
            requirement_binding = self._llm_prompt_requirement_binding(
                process,
                current_record,
                call_id=call_id,
                context_generation=context_generation,
                current_requirements=current_requirements,
                check_cached_binding=False,
            )
            if requirement_binding is None:
                return None
            requirement_outcomes = self._completion_requirement_outcomes(
                selected_action,
                requirement_binding,
                pid=process.pid,
                record=current_record,
                requirements=requirements,
            )
        self._require_cached_prompt_requirement_binding(
            process.pid,
            context_generation,
            requirement_binding,
        )
        evidence = {
            "schema_version": 1,
            "kind": "root_process_exit",
            "run_id": record.run_id,
            "pid": process.pid,
            "call_id": call_id,
            "context_generation": context_generation,
            "action_manifest_sha256": action_manifest_sha256,
            "outcome_sha256": outcome_sha256,
            "requirement_binding": requirement_binding,
            "requirement_binding_sha256": self._sha256(requirement_binding),
            "requirement_outcomes": requirement_outcomes,
        }
        canonical_task_run_json(evidence)
        return evidence

    def _completion_requirement_outcomes(
        self,
        selected_action: Mapping[str, Any],
        requirement_binding: Mapping[str, Any],
        *,
        pid: str,
        record: TaskRunRecord,
        requirements: Iterable[TaskRunRequirement] | None = None,
    ) -> list[dict[str, Any]]:
        """Project model completion statuses onto exact bound requirements.

        Images without cumulative completion review retain the original
        integrity-bound root-exit contract.  When structured completion
        evidence is present, however, a model-reported blocker or cancellation
        can never be promoted to satisfaction.  Cancellation is deliberately
        projected as blocked because only the Host may waive a requirement.
        """

        requirement_ids = TaskRunManager._completion_requirement_ids(
            requirement_binding
        )
        completion = selected_action.get("completion_evidence")
        if completion is None:
            return [
                {
                    "requirement_id": requirement_id,
                    "status": TaskRunRequirementStatus.SATISFIED.value,
                    "reported_status": "implicit_root_exit",
                    "evidence_receipt_ids": [],
                }
                for requirement_id in requirement_ids
            ]
        completion = self._normalized_completion_evidence(completion)
        checks = self._completion_checks_by_requirement(
            completion,
            requirement_ids,
        )
        receipts = self._completion_causal_tool_receipts(
            pid=pid,
            record=record,
            requirements=requirements,
        )
        return self._project_completion_requirement_outcomes(
            requirement_ids,
            checks,
            receipts,
        )

    def _normalized_completion_evidence(self, value: Any) -> Mapping[str, Any]:
        """Detach the same JSON-container shape accepted by ``ProcessExitArgs``.

        Some OpenAI-compatible providers serialize a nested object argument as
        a JSON string.  The process-exit Tool intentionally accepts that
        reversible representation, so TaskRun settlement must interpret the
        already integrity-bound action the same way.  Keep the raw action in
        its validated manifest and hashes; only this semantic projection is
        decoded before the existing per-requirement and causal-receipt checks.
        """

        selected = value
        if isinstance(selected, str):
            try:
                encoded = selected.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValidationError(
                    "TaskRun structured completion evidence is invalid"
                ) from exc
            hard_limit = min(
                self.config.tools.tool_call_args_hard_limit_bytes,
                self.config.task_runs.payload_max_bytes,
            )
            if len(encoded) > hard_limit:
                raise ValidationError(
                    "TaskRun structured completion evidence exceeds its hard cap"
                )
            try:
                selected = json.loads(selected)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    "TaskRun structured completion evidence is invalid"
                ) from exc
        if not isinstance(selected, Mapping):
            raise ValidationError("TaskRun structured completion evidence is invalid")
        try:
            canonical = canonical_task_run_json(dict(selected))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "TaskRun structured completion evidence is invalid"
            ) from exc
        if len(canonical.encode("utf-8")) > min(
            self.config.tools.tool_call_args_hard_limit_bytes,
            self.config.task_runs.payload_max_bytes,
        ):
            raise ValidationError(
                "TaskRun structured completion evidence exceeds its hard cap"
            )
        detached = json.loads(canonical)
        if not isinstance(detached, Mapping):
            raise ValidationError("TaskRun structured completion evidence is invalid")
        return detached

    @staticmethod
    def _completion_requirement_ids(
        requirement_binding: Mapping[str, Any],
    ) -> list[str]:
        entries = requirement_binding.get("requirements")
        if not isinstance(entries, list):
            raise ValidationError("TaskRun completion requirement binding is invalid")
        requirement_ids = [
            str(entry.get("requirement_id") or "")
            for entry in entries
            if isinstance(entry, Mapping)
        ]
        if len(requirement_ids) != len(entries) or any(
            not item for item in requirement_ids
        ):
            raise ValidationError(
                "TaskRun completion requirement identities are invalid"
            )
        return requirement_ids

    @staticmethod
    def _completion_checks_by_requirement(
        completion: Any,
        requirement_ids: Iterable[str],
    ) -> dict[str, list[dict[str, Any]]]:
        if not isinstance(completion, Mapping):
            raise ValidationError("TaskRun structured completion evidence is invalid")
        checks = completion.get("acceptance_checks")
        if not isinstance(checks, list) or not checks:
            raise ValidationError("TaskRun completion acceptance checks are missing")
        checks_by_requirement: dict[str, list[dict[str, Any]]] = {
            requirement_id: [] for requirement_id in requirement_ids
        }
        for check in checks:
            normalized, bound_requirement_id = (
                TaskRunManager._validated_completion_check(
                    check,
                    requirement_ids=checks_by_requirement,
                )
            )
            if bound_requirement_id is not None:
                checks_by_requirement[bound_requirement_id].append(normalized)
        return checks_by_requirement

    @staticmethod
    def _validated_completion_check(
        check: Any,
        *,
        requirement_ids: Iterable[str],
    ) -> tuple[dict[str, Any], str | None]:
        if not isinstance(check, Mapping):
            raise ValidationError("TaskRun completion acceptance check is invalid")
        status = check.get("status")
        if status not in {"completed", "blocked", "cancelled"}:
            raise ValidationError("TaskRun completion status is invalid")
        refs = TaskRunManager._completion_check_string_list(
            check.get("source_refs"),
            label="source references",
        )
        evidence_tools = TaskRunManager._completion_check_string_list(
            check.get("evidence_tool_calls"),
            label="evidence tools",
        )
        bound = [requirement_id for requirement_id in requirement_ids if requirement_id in refs]
        if len(bound) > 1:
            raise ValidationError(
                "one TaskRun completion check cannot cover multiple requirements"
            )
        if status == "completed" and not evidence_tools:
            raise ValidationError(
                "completed TaskRun requirement check has no evidence tool"
            )
        return dict(check), bound[0] if bound else None

    @staticmethod
    def _completion_check_string_list(value: Any, *, label: str) -> list[str]:
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValidationError(f"TaskRun completion {label} are invalid")
        return value

    def _project_completion_requirement_outcomes(
        self,
        requirement_ids: Iterable[str],
        checks_by_requirement: Mapping[str, list[dict[str, Any]]],
        receipts: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        receipt_index = self._completion_requirement_receipt_index(receipts)
        outcomes: list[dict[str, Any]] = []
        for requirement_id in requirement_ids:
            checks = checks_by_requirement[requirement_id]
            if not checks:
                raise ValidationError(
                    "TaskRun completion evidence omits a bound requirement"
                )
            reported = [str(check["status"]) for check in checks]
            unresolved = next(
                (status for status in reported if status != "completed"),
                None,
            )
            evidence_receipt_ids = (
                []
                if unresolved is not None
                else self._completion_check_receipt_ids(
                    requirement_id,
                    checks,
                    receipt_index,
                )
            )
            outcomes.append(
                {
                    "requirement_id": requirement_id,
                    "status": (
                        TaskRunRequirementStatus.SATISFIED.value
                        if unresolved is None
                        else TaskRunRequirementStatus.BLOCKED.value
                    ),
                    "reported_status": unresolved or "completed",
                    "evidence_receipt_ids": evidence_receipt_ids,
                }
            )
        return outcomes

    @staticmethod
    def _completion_requirement_receipt_index(
        receipts: Iterable[Mapping[str, Any]],
    ) -> dict[str, dict[str, str]]:
        """Index the latest exact successful Operation receipt for each tool."""

        selected: dict[str, dict[str, str]] = {}
        for receipt in receipts:
            receipt_id = str(receipt.get("receipt_id") or "")
            tool = str(receipt.get("tool") or "")
            requirement_ids = receipt.get("requirement_ids")
            if not receipt_id or not tool or not isinstance(requirement_ids, list):
                continue
            for requirement_id in requirement_ids:
                if isinstance(requirement_id, str) and requirement_id:
                    # Receipts arrive in chronological order, so overwrite with
                    # the latest causally eligible successful Operation.
                    selected.setdefault(requirement_id, {})[tool] = receipt_id
        return selected

    @staticmethod
    def _completion_check_receipt_ids(
        requirement_id: str,
        checks: Iterable[Mapping[str, Any]],
        receipt_index: Mapping[str, Mapping[str, str]],
    ) -> list[str]:
        by_tool = receipt_index.get(requirement_id, {})
        selected: list[str] = []
        for check in checks:
            for tool in check["evidence_tool_calls"]:
                receipt_id = by_tool.get(str(tool))
                if receipt_id is None:
                    raise ValidationError(
                        "TaskRun completion evidence has no causally bound "
                        f"successful receipt for requirement {requirement_id}"
                    )
                if receipt_id not in selected:
                    selected.append(receipt_id)
        if not selected:
            raise ValidationError(
                "completed TaskRun requirement has no evidence receipt"
            )
        return selected

    @staticmethod
    def _committed_process_exit_action(
        outcome: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        completed = outcome.get("result")
        if outcome.get("state") != "completed" or not isinstance(completed, Mapping):
            return None
        action = completed.get("action")
        tool_result = completed.get("result")
        payload = (
            tool_result.get("payload") if isinstance(tool_result, Mapping) else None
        )
        valid = (
            completed.get("ok") is True
            and isinstance(action, Mapping)
            and action.get("action") == "process_exit"
            and isinstance(tool_result, Mapping)
            and tool_result.get("ok") is True
            and isinstance(payload, Mapping)
            and payload.get("status") == "exited"
            and payload.get("terminal_committed") is True
        )
        return dict(action) if valid else None

    def _completion_source_matches_action(
        self,
        source: Mapping[str, Any],
        *,
        selected_action: Mapping[str, Any],
        action_manifest_sha256: str | None,
    ) -> bool:
        kind = source.get("kind")
        if kind == "validated_action":
            self._require_completion_action_manifest(
                source,
                selected_action=selected_action,
                action_manifest_sha256=action_manifest_sha256,
            )
            return True
        if kind == "durable_wait_action":
            snapshot = source.get("wait_snapshot")
            valid = (
                isinstance(snapshot, Mapping)
                and snapshot.get("action") == dict(selected_action)
                and isinstance(action_manifest_sha256, str)
                and bool(action_manifest_sha256)
            )
            if not valid:
                raise ValidationError(
                    "TaskRun requirement completion changed its resumed process_exit action"
                )
            return True
        return False

    def _require_completion_action_manifest(
        self,
        source: Mapping[str, Any],
        *,
        selected_action: Mapping[str, Any],
        action_manifest_sha256: str | None,
    ) -> None:
        try:
            manifest = normalize_validated_action_manifest(source["manifest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                "TaskRun requirement completion lost its action manifest"
            ) from exc
        valid = (
            len(manifest["actions"]) == 1
            and manifest["actions"][0] == dict(selected_action)
            and action_manifest_sha256 == self._sha256(manifest)
        )
        if not valid:
            raise ValidationError(
                "TaskRun requirement completion changed its process_exit action"
            )

    def _stage_binding_transition(
        self,
        *,
        process: Any,
        prior: TaskRunResumePoint,
        source: Mapping[str, Any],
        action_manifest_sha256: str | None,
        outcome: Mapping[str, Any],
        settled_effect_seq: int,
    ) -> dict[str, Any] | None:
        current_hashes = self._process_binding_hashes(process)
        prior_hashes = (
            prior.image_binding_hash,
            prior.tool_binding_hash,
            prior.provider_binding_hash,
        )
        if current_hashes == prior_hashes:
            return None
        if source.get("kind") != "validated_action":
            raise ValidationError(
                "TaskRun binding changed outside a directly validated action"
            )
        if action_manifest_sha256 is None:
            raise ValidationError("TaskRun binding transition lost its action manifest")
        return self._certified_activate_skill_transition(
            process=process,
            source=source,
            source_payload_id=prior.pending_action_payload_id,
            pre_hashes=prior_hashes,
            action_manifest_sha256=action_manifest_sha256,
            outcome=outcome,
            settled_effect_seq=settled_effect_seq,
        )

    def _certified_activate_skill_transition(
        self,
        *,
        process: Any,
        source: Mapping[str, Any],
        source_payload_id: str | None,
        pre_hashes: tuple[str, str, str],
        action_manifest_sha256: str,
        outcome: Mapping[str, Any],
        settled_effect_seq: int,
    ) -> dict[str, Any]:
        pre_binding = validate_pre_action_binding(
            source.get("pre_action_binding"),
            image_binding_hash=pre_hashes[0],
            tool_binding_hash=pre_hashes[1],
            provider_binding_hash=pre_hashes[2],
        )
        manifest = normalize_validated_action_manifest(source.get("manifest"))
        if (
            source.get("manifest_sha256") != self._sha256(manifest)
            or action_manifest_sha256 != self._sha256(manifest)
        ):
            raise ValidationError(
                "TaskRun activate_skill transition changed its action manifest"
            )
        action, activation = self._validated_activate_skill_outcome(
            process,
            manifest=manifest,
            outcome=outcome,
            pre_binding=pre_binding,
        )
        current_hashes = self._process_binding_hashes(process)
        if current_hashes[0] != pre_hashes[0] or current_hashes[2] != pre_hashes[2]:
            raise ValidationError(
                "TaskRun activate_skill changed its Image or provider binding"
            )
        skill_id = str(action["skill_id"])
        skill_evidence = self._host.skills.validate_activated_skill_result(
            process.pid,
            activation,
        )
        current_loaded_hashes = loaded_skill_hashes(process)
        loaded_sha256 = current_loaded_hashes.get(skill_id)
        if loaded_sha256 is None:
            raise ValidationError("TaskRun activated Skill record is missing")
        expected_projection = expected_activated_process_projection(
            pre_binding,
            skill_id=skill_id,
            tool_ids=activation["tool_ids"],
            jit_tool_ids=activation["jit_tool_ids"],
            loaded_skill_sha256=loaded_sha256,
        )
        post_tool_hash = require_exact_activated_projection(
            process,
            expected_projection,
        )
        if post_tool_hash != current_hashes[1] or post_tool_hash == pre_hashes[1]:
            raise ValidationError(
                "TaskRun activate_skill did not produce its certified binding transition"
            )
        source_payload = self._store.get_task_run_payload(source_payload_id or "")
        if (
            source_payload is None
            or source_payload.role != "pending_action"
            or source_payload.run_id != getattr(process, "task_run_id", None)
            or self._decode_payload(source_payload, role="pending_action") != dict(source)
        ):
            raise ValidationError("TaskRun binding transition source payload changed")
        transition = {
            "schema_version": 1,
            "kind": "certified_activate_skill",
            "source_payload_id": source_payload.payload_id,
            "source_payload_sha256": source_payload.sha256,
            "pre_binding_projection_sha256": task_run_binding_sha256(pre_binding),
            "pre_image_binding_hash": pre_hashes[0],
            "pre_tool_binding_hash": pre_hashes[1],
            "pre_provider_binding_hash": pre_hashes[2],
            "post_image_binding_hash": current_hashes[0],
            "post_tool_binding_hash": current_hashes[1],
            "post_provider_binding_hash": current_hashes[2],
            "action_manifest_sha256": action_manifest_sha256,
            "action_result_sha256": self._sha256(outcome.get("result")),
            "skill_id": skill_id,
            "package_sha256": skill_evidence["package_sha256"],
            "loaded_skill_sha256": loaded_sha256,
            "settled_effect_seq": settled_effect_seq,
        }
        canonical_task_run_json(transition)
        return transition

    def _validated_activate_skill_outcome(
        self,
        process: Any,
        *,
        manifest: Mapping[str, Any],
        outcome: Mapping[str, Any],
        pre_binding: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        action = self._single_activate_skill_action(manifest)
        tool_result = self._completed_activate_skill_tool_result(
            outcome,
            action=action,
            pre_binding=pre_binding,
        )
        activation = self._activation_result(
            process,
            action=action,
            tool_result=tool_result,
        )
        return action, activation

    @staticmethod
    def _single_activate_skill_action(
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        actions = manifest.get("actions")
        if (
            manifest.get("parallel_tool_calls") is not False
            or manifest.get("host_auto_wait") is not False
            or not isinstance(actions, list)
            or len(actions) != 1
            or not isinstance(actions[0], dict)
        ):
            raise ValidationError(
                "TaskRun binding transition requires one validated action"
            )
        action = dict(actions[0])
        if set(action) != {"action", "skill_id", "expected_package_sha256"} or action.get(
            "action"
        ) != "activate_skill":
            raise ValidationError(
                "TaskRun binding transition is not a validated activate_skill"
            )
        return action

    @staticmethod
    def _completed_activate_skill_tool_result(
        outcome: Mapping[str, Any],
        *,
        action: Mapping[str, Any],
        pre_binding: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        completed = outcome.get("result")
        if (
            outcome.get("state") != "completed"
            or not isinstance(completed, Mapping)
            or completed.get("ok") is not True
            or completed.get("action") != action
            or not isinstance(completed.get("result"), Mapping)
        ):
            raise ValidationError(
                "TaskRun activate_skill transition lacks its complete local result"
            )
        tool_result = completed["result"]
        expected_tool_id = pre_binding["tool_table"].get("activate_skill")
        if (
            set(tool_result)
            != {"ok", "tool_id", "result_oid", "payload", "error", "message_notice"}
            or tool_result.get("ok") is not True
            or tool_result.get("error") is not None
            or not isinstance(expected_tool_id, str)
            or pre_binding["model_tool_table"].get("activate_skill")
            != expected_tool_id
            or tool_result.get("tool_id") != expected_tool_id
        ):
            raise ValidationError(
                "TaskRun activate_skill transition has an invalid tool result"
            )
        return tool_result

    @staticmethod
    def _activation_result(
        process: Any,
        *,
        action: Mapping[str, Any],
        tool_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        result_oid = tool_result.get("result_oid")
        payload = tool_result.get("payload")
        if (
            not isinstance(result_oid, str)
            or not result_oid
            or not isinstance(payload, Mapping)
            or set(payload) != {"result"}
            or not isinstance(payload.get("result"), Mapping)
        ):
            raise ValidationError(
                "TaskRun activate_skill transition lost its persisted result"
            )
        activation = dict(payload["result"])
        if (
            activation.get("pid") != process.pid
            or activation.get("skill_id") != action.get("skill_id")
            or activation.get("package_sha256")
            != action.get("expected_package_sha256")
        ):
            raise ValidationError(
                "TaskRun activate_skill result lost its validated action binding"
            )
        return activation

    def _durable_wait_stage_binding(
        self,
        *,
        pid: str,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        llm_call_sha256: str,
        source: Mapping[str, Any],
    ) -> tuple[
        bool,
        str | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        source_generation = source.get("context_generation")
        if (
            source.get("state") != "waiting"
            or not isinstance(source_generation, str)
            or not source_generation
            or source.get("call_id") != call_id
            or source.get("llm_call_sha256") != llm_call_sha256
            or not isinstance(source.get("wait_snapshot"), Mapping)
        ):
            raise ValidationError("TaskRun resumed wait lost its call binding")
        pending_service = getattr(self._host.llm, "pending", None)
        getter = getattr(pending_service, "get", None)
        snapshot = self._durable_wait_snapshot(
            pid,
            getter(pid) if callable(getter) else None,
            call_id=call_id,
        )
        if (
            outcome["state"] == "waiting"
            and source.get("wait_outcome_sha256") == outcome_sha256
            and source.get("wait_snapshot_sha256") == self._sha256(snapshot)
            and source.get("wait_snapshot") == snapshot
        ):
            if source_generation != generation:
                raise ValidationError(
                    "TaskRun waiting outcome changed its context generation"
                )
            return True, source.get("action_manifest_sha256"), snapshot, None
        if outcome["state"] == "completed" and (
            snapshot.get("status") != "completed"
            or self._durable_wait_identity_sha256(snapshot)
            != source.get("wait_identity_sha256")
        ):
            raise ValidationError("TaskRun resumed result does not match its durable wait")
        transition = None
        if source_generation != generation:
            transition = self._certified_compaction_generation_transition(
                pid,
                source_generation=source_generation,
                result_generation=generation,
                outcome=outcome,
                wait_snapshot=snapshot,
            )
        return False, source.get("action_manifest_sha256"), snapshot, transition

    def _certified_compaction_generation_transition(
        self,
        pid: str,
        *,
        source_generation: str,
        result_generation: str,
        outcome: Mapping[str, Any],
        wait_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Certify the one action allowed to advance generation while waiting.

        ``compact_process_context`` replaces the caller's context only after
        its compressor child has completed.  The durable child-wait therefore
        names the pre-compaction generation while the locally completed tool
        result names the post-compaction generation.  No other wait or action
        may use this exception.
        """

        if source_generation == result_generation:
            raise ValidationError(
                "TaskRun context generation changed outside a completed compaction wait"
            )
        output, resume_job = self._compaction_wait_evidence(
            pid,
            outcome=outcome,
            wait_snapshot=wait_snapshot,
        )
        compaction = self._current_compaction_projection(
            pid,
            result_generation=result_generation,
            output=output,
            resume_job=resume_job,
        )
        transition = {
            "schema_version": 1,
            "kind": "certified_context_compaction",
            "source_context_generation": source_generation,
            "result_context_generation": result_generation,
            "compaction_sha256": self._sha256(compaction),
        }
        canonical_task_run_json(transition)
        return transition

    def _compaction_wait_evidence(
        self,
        pid: str,
        *,
        outcome: Mapping[str, Any],
        wait_snapshot: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        wait_state = (
            outcome.get("state"),
            wait_snapshot.get("wait_type"),
            wait_snapshot.get("status"),
        )
        if wait_state != ("completed", "child", "completed"):
            raise ValidationError(
                "TaskRun context generation changed outside a completed compaction wait"
            )
        action = wait_snapshot.get("action")
        if not isinstance(action, Mapping):
            raise ValidationError(
                "TaskRun generation transition lost its compaction action"
            )
        result = outcome.get("result")
        if not isinstance(result, Mapping):
            raise ValidationError(
                "TaskRun generation transition lost its local result"
            )
        action_binding = (
            action.get("action"),
            result.get("ok"),
            result.get("action"),
        )
        if action_binding != ("compact_process_context", True, action):
            raise ValidationError(
                "TaskRun generation transition is not bound to its compaction action"
            )
        tool_result = result.get("result")
        if not isinstance(tool_result, Mapping):
            raise ValidationError("TaskRun compaction tool result is missing")
        output = tool_result.get("payload")
        resume_job = action.get("_resume_job")
        if not isinstance(output, Mapping) or not isinstance(resume_job, Mapping):
            raise ValidationError(
                "TaskRun compaction result lacks its exact local resume job"
            )
        job_binding = (
            tool_result.get("ok"),
            output.get("compacted"),
            resume_job.get("kind"),
            resume_job.get("caller_pid"),
        )
        if job_binding != (True, True, "context_compaction_job", pid):
            raise ValidationError(
                "TaskRun compaction result lacks its exact local resume job"
            )
        return output, resume_job

    def _current_compaction_projection(
        self,
        pid: str,
        *,
        result_generation: str,
        output: Mapping[str, Any],
        resume_job: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        context_memory = getattr(
            getattr(self._host, "llm", None),
            "context_memory",
            None,
        )
        latest = getattr(context_memory, "latest_validated_compaction", None)
        if not callable(latest):
            raise ValidationError(
                "TaskRun compaction generation transition cannot be certified"
            )
        try:
            compaction = latest(pid)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValidationError(
                "TaskRun compaction generation transition is invalid"
            ) from exc
        if not isinstance(compaction, Mapping):
            raise ValidationError(
                "TaskRun current local compaction projection is missing"
            )
        expected = (
            1,
            result_generation,
            result_generation,
            resume_job.get("context_oid"),
            resume_job.get("source_version"),
            compaction.get("context_oid"),
            compaction.get("source_version"),
            compaction.get("context_version"),
        )
        actual = (
            compaction.get("schema_version"),
            compaction.get("context_generation"),
            compaction.get("compacted_at"),
            compaction.get("context_oid"),
            compaction.get("source_version"),
            output.get("context_oid"),
            output.get("old_version"),
            output.get("new_version"),
        )
        if actual != expected:
            raise ValidationError(
                "TaskRun compaction result does not match the current local compaction"
            )
        return compaction

    def _validate_staged_generation_binding(
        self,
        pid: str,
        staged: Mapping[str, Any],
        *,
        generation: str,
        outcome: Mapping[str, Any],
    ) -> None:
        source_generation = staged.get("source_context_generation")
        result_generation = staged.get("result_context_generation")
        transition = staged.get("generation_transition")
        if (
            not isinstance(source_generation, str)
            or not source_generation
            or result_generation != generation
            or staged.get("context_generation") != generation
        ):
            raise ValidationError(
                "TaskRun staged outcome lost its context generation binding"
            )
        if source_generation == result_generation:
            if transition is not None:
                raise ValidationError(
                    "TaskRun staged outcome has an unexpected generation transition"
                )
            return
        wait_snapshot = staged.get("wait_snapshot")
        if not isinstance(wait_snapshot, Mapping):
            raise ValidationError(
                "TaskRun staged generation transition lost its durable wait"
            )
        certified = self._certified_compaction_generation_transition(
            pid,
            source_generation=source_generation,
            result_generation=generation,
            outcome=outcome,
            wait_snapshot=wait_snapshot,
        )
        if transition != certified:
            raise ValidationError(
                "TaskRun staged generation transition evidence changed"
            )

    def _stage_wait_snapshot(
        self,
        *,
        pid: str,
        call_id: str,
        outcome: Mapping[str, Any],
        source_wait_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if outcome["state"] != "waiting":
            return source_wait_snapshot
        pending_service = getattr(self._host.llm, "pending", None)
        getter = getattr(pending_service, "get", None)
        snapshot = self._durable_wait_snapshot(
            pid,
            getter(pid) if callable(getter) else None,
            call_id=call_id,
        )
        expected_type = {
            "human": "human",
            "process": "child",
            "message": "message",
        }[str(outcome["durable_wait"]["wait_type"])]
        if snapshot["status"] != "pending" or snapshot["wait_type"] != expected_type:
            raise ValidationError("TaskRun wait outcome does not match pending action")
        return snapshot

    def _persist_staged_outcome(
        self,
        *,
        process: Any,
        record: TaskRunRecord,
        prior: TaskRunResumePoint,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        llm_call_sha256: str,
        action_manifest_sha256: str | None,
        settled_effect_seq: int,
        settled_effects: list[dict[str, Any]],
        wait_snapshot: dict[str, Any] | None,
        source_generation: str,
        generation_transition: dict[str, Any] | None,
        binding_transition: dict[str, Any] | None,
        requirement_completion: dict[str, Any] | None,
    ) -> None:
        now = utc_now()
        staged = self._staged_outcome_wrapper(
            prior=prior,
            call_id=call_id,
            generation=generation,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
            llm_call_sha256=llm_call_sha256,
            action_manifest_sha256=action_manifest_sha256,
            settled_effect_seq=settled_effect_seq,
            settled_effects=settled_effects,
            wait_snapshot=wait_snapshot,
            source_generation=source_generation,
            generation_transition=generation_transition,
            binding_transition=binding_transition,
            requirement_completion=requirement_completion,
        )
        pending_payload = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=record.run_id,
            role="pending_action",
            label="Complete local TaskRun outcome awaiting settlement",
            value=staged,
            created_at=now,
        )
        self._require_payload_bound(pending_payload)
        transcript = self._store.get_task_run_payload(prior.transcript_payload_id)
        summary = self._optional_resume_payload(prior.summary_payload_id)
        if transcript is None:
            raise ValidationError("TaskRun staged transcript is missing")
        for attempt in range(5):
            current_process = self._store.get_process(process.pid)
            current_record = self._require_run(record.run_id)
            if current_process is None:
                raise NotFound(f"process not found: {process.pid}")
            self._require_settlement_epoch(
                current_process,
                current_record,
                "LLM result staging",
            )
            self._require_current_staged_binding(
                process=current_process,
                prior=prior,
                staged=staged,
            )
            point = self._make_resume_point(
                process=current_process,
                record=current_record,
                context_generation=generation,
                safe_point_seq=prior.safe_point_seq + 1,
                transcript_payload=transcript,
                summary_payload=summary,
                pending_payload=pending_payload,
                last_effect_seq=settled_effect_seq,
                created_at=prior.created_at,
                updated_at=now,
            )
            try:
                self._commit_staged_outcome(
                    point=point,
                    payload=pending_payload,
                    call_id=call_id,
                    outcome_sha256=outcome_sha256,
                    settled_effect_seq=settled_effect_seq,
                )
                return
            except TaskRunRevisionConflict:
                if self._pending_wrapper_is_current(process.pid, staged):
                    return
                if attempt == 4:
                    raise

    def _staged_outcome_wrapper(self, **values: Any) -> dict[str, Any]:
        prior = values["prior"]
        wait_snapshot = values["wait_snapshot"]
        return {
            "schema_version": 1,
            "kind": "completed_outcome",
            "state": "staged",
            "call_id": values["call_id"],
            "context_generation": values["generation"],
            "source_context_generation": values["source_generation"],
            "result_context_generation": values["generation"],
            "generation_transition": values["generation_transition"],
            "binding_transition": values["binding_transition"],
            "requirement_completion": values["requirement_completion"],
            "outcome": values["outcome"],
            "outcome_sha256": values["outcome_sha256"],
            "llm_call_sha256": values["llm_call_sha256"],
            "action_manifest_sha256": values["action_manifest_sha256"],
            "source_safe_point_seq": prior.safe_point_seq,
            "effect_baseline_seq": prior.last_effect_seq,
            "settled_effect_seq": values["settled_effect_seq"],
            "settled_effects": values["settled_effects"],
            "wait_snapshot": wait_snapshot,
            "wait_snapshot_sha256": (
                self._sha256(wait_snapshot) if wait_snapshot is not None else None
            ),
            "wait_identity_sha256": (
                self._durable_wait_identity_sha256(wait_snapshot)
                if wait_snapshot is not None
                else None
            ),
        }

    def _require_current_staged_binding(
        self,
        *,
        process: Any,
        prior: TaskRunResumePoint,
        staged: Mapping[str, Any],
    ) -> None:
        transition = staged.get("binding_transition")
        current_hashes = self._process_binding_hashes(process)
        prior_hashes = (
            prior.image_binding_hash,
            prior.tool_binding_hash,
            prior.provider_binding_hash,
        )
        if transition is None:
            if current_hashes != prior_hashes:
                raise ValidationError(
                    "TaskRun binding changed after result certification"
                )
            return
        selected = self._validated_binding_transition(transition)
        if self._transition_pre_hashes(selected) != prior_hashes:
            raise ValidationError("TaskRun binding transition lost its source point")
        self._require_reproducible_binding_transition(
            process=process,
            staged=staged,
            transition=selected,
        )

    def _validate_staged_binding_transition(
        self,
        *,
        process: Any,
        point: TaskRunResumePoint,
        staged: Mapping[str, Any],
    ) -> None:
        transition = staged.get("binding_transition")
        if transition is None:
            return
        selected = self._validated_binding_transition(transition)
        post_hashes = (
            point.image_binding_hash,
            point.tool_binding_hash,
            point.provider_binding_hash,
        )
        if self._transition_post_hashes(selected) != post_hashes:
            raise ValidationError("TaskRun staged binding transition lost its safe point")
        self._require_reproducible_binding_transition(
            process=process,
            staged=staged,
            transition=selected,
        )

    def _validate_current_staged_binding_for_pid(
        self,
        pid: str,
        staged: Mapping[str, Any],
    ) -> None:
        process = self._store.get_process(pid)
        point = self._store.get_task_run_resume_point(pid, complete_only=True)
        if process is None or point is None or not self._resume_integrity_valid(point):
            raise ValidationError(
                "TaskRun staged binding has no current integrity-bound safe point"
            )
        self._validate_staged_binding_transition(
            process=process,
            point=point,
            staged=staged,
        )

    def _require_reproducible_binding_transition(
        self,
        *,
        process: Any,
        staged: Mapping[str, Any],
        transition: Mapping[str, Any],
    ) -> None:
        source_payload = self._store.get_task_run_payload(
            str(transition["source_payload_id"])
        )
        if (
            source_payload is None
            or source_payload.sha256 != transition["source_payload_sha256"]
            or source_payload.run_id != getattr(process, "task_run_id", None)
            or source_payload.role != "pending_action"
        ):
            raise ValidationError("TaskRun binding transition source is unavailable")
        source = self._decode_payload(source_payload, role="pending_action")
        expected = self._certified_activate_skill_transition(
            process=process,
            source=source,
            source_payload_id=source_payload.payload_id,
            pre_hashes=self._transition_pre_hashes(transition),
            action_manifest_sha256=str(staged.get("action_manifest_sha256") or ""),
            outcome=self._validated_outcome_manifest(staged.get("outcome")),
            settled_effect_seq=int(staged.get("settled_effect_seq", -1)),
        )
        if expected != dict(transition):
            raise ValidationError("TaskRun binding transition evidence changed")

    def _validated_binding_transition(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != _BINDING_TRANSITION_KEYS:
            raise ValidationError("TaskRun binding transition has an invalid shape")
        selected = dict(value)
        if (
            type(selected.get("schema_version")) is not int
            or selected.get("schema_version") != 1
            or selected.get("kind") != "certified_activate_skill"
            or type(selected.get("settled_effect_seq")) is not int
            or selected["settled_effect_seq"] < 0
        ):
            raise ValidationError("TaskRun binding transition is invalid")
        for field in _BINDING_TRANSITION_KEYS - {
            "schema_version",
            "kind",
            "settled_effect_seq",
        }:
            value = selected.get(field)
            if not isinstance(value, str) or not value:
                raise ValidationError(
                    f"TaskRun binding transition has an invalid {field}"
                )
        canonical_task_run_json(selected)
        return selected

    @staticmethod
    def _transition_pre_hashes(
        transition: Mapping[str, Any],
    ) -> tuple[str, str, str]:
        return (
            str(transition["pre_image_binding_hash"]),
            str(transition["pre_tool_binding_hash"]),
            str(transition["pre_provider_binding_hash"]),
        )

    @staticmethod
    def _transition_post_hashes(
        transition: Mapping[str, Any],
    ) -> tuple[str, str, str]:
        return (
            str(transition["post_image_binding_hash"]),
            str(transition["post_tool_binding_hash"]),
            str(transition["post_provider_binding_hash"]),
        )

    def _optional_resume_payload(self, payload_id: str | None) -> TaskRunPayload | None:
        return self._store.get_task_run_payload(payload_id) if payload_id is not None else None

    def _commit_staged_outcome(
        self,
        *,
        point: TaskRunResumePoint,
        payload: TaskRunPayload,
        call_id: str,
        outcome_sha256: str,
        settled_effect_seq: int,
    ) -> None:
        wrapper = self._decode_payload(payload, role="pending_action")
        transition = wrapper.get("binding_transition")
        with self._uow.transaction():
            self._store.insert_task_run_payload(payload)
            self._store.upsert_task_run_resume_point(point)
            self._append_ledger(
                point.run_id,
                kind=TaskRunLedgerKind.LLM_TURN,
                status="result_staged",
                label="complete local action result staged before settlement",
                pid=point.pid,
                llm_call_id=call_id,
                payload_id=payload.payload_id,
                metadata={
                    "safe_point_seq": point.safe_point_seq,
                    "outcome_sha256": outcome_sha256,
                    "settled_effect_seq": settled_effect_seq,
                    "binding_transition_sha256": (
                        self._sha256(transition)
                        if transition is not None
                        else None
                    ),
                },
            )

    def _pending_wrapper_is_current(
        self,
        pid: str,
        expected: Mapping[str, Any],
    ) -> bool:
        point = self._store.get_task_run_resume_point(pid, complete_only=True)
        return bool(
            point is not None
            and point.pending_action_payload_id is not None
            and self._resume_integrity_valid(point)
            and self._decode_pending_resume_payload(point) == expected
        )

    def record_completed_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        outcome_manifest: Mapping[str, Any],
        context_generation: str | int,
    ) -> None:
        """Persist a complete paired outcome as the latest local safe point."""

        process, record = self._bound_process_run(pid)
        if record is None:
            return
        self._require_settlement_epoch(process, record, "LLM settlement")
        selected_call_id = self._identifier(call_id, "LLM call_id")
        generation = str(context_generation)
        outcome = self._validated_outcome_manifest(outcome_manifest)
        outcome_sha256 = self._sha256(outcome)
        llm_call_sha256 = self._local_llm_call_sha256(process, selected_call_id)
        staged_state = self._require_staged_settlement(
            pid=pid,
            record=record,
            call_id=selected_call_id,
            generation=generation,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
            llm_call_sha256=llm_call_sha256,
        )
        if staged_state is None:
            return
        prior, prior_payload, staged = staged_state
        transcript_state = self._build_settled_transcript(
            pid=pid,
            record=record,
            call_id=selected_call_id,
            generation=generation,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
            prior=prior,
            prior_payload=prior_payload,
        )
        if transcript_state is None:
            return
        transcript_payload, summary_payload, effective_summary, now = transcript_state
        next_pending = self._settled_wait_payload(
            pid=pid,
            run_id=record.run_id,
            call_id=selected_call_id,
            generation=generation,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
            llm_call_sha256=llm_call_sha256,
            staged=staged,
            created_at=now,
        )
        self._commit_completed_transcript(
            pid=pid,
            run_id=record.run_id,
            call_id=selected_call_id,
            generation=generation,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
            staged=staged,
            prior=prior,
            transcript_payload=transcript_payload,
            summary_payload=summary_payload,
            effective_summary=effective_summary,
            next_pending=next_pending,
            now=now,
        )
        self._notify_updated()

    def _require_staged_settlement(
        self,
        *,
        pid: str,
        record: TaskRunRecord,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        llm_call_sha256: str,
    ) -> tuple[TaskRunResumePoint, dict[str, Any], dict[str, Any]] | None:
        prior = self._store.get_task_run_resume_point(pid, complete_only=True)
        if prior is None:
            raise ValidationError("TaskRun completed outcome has no resume point")
        transcript = self._decode_payload(
            self._store.get_task_run_payload(prior.transcript_payload_id),
            role="transcript",
        )
        if prior.pending_action_payload_id is None:
            if self._settled_transcript_matches(
                prior,
                transcript,
                call_id=call_id,
                generation=generation,
                outcome=outcome,
                outcome_sha256=outcome_sha256,
            ):
                return None
            raise ValidationError("TaskRun completed outcome was not staged")
        if not self._resume_integrity_valid(prior):
            raise ValidationError("TaskRun staged outcome failed resume integrity")
        staged = self._decode_pending_resume_payload(prior)
        if self._settled_wait_matches(
            staged,
            call_id=call_id,
            generation=generation,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
        ):
            return None
        valid = (
            staged.get("kind") == "completed_outcome"
            and staged.get("state") == "staged"
            and staged.get("call_id") == call_id
            and staged.get("context_generation") == generation
            and staged.get("outcome") == outcome
            and staged.get("outcome_sha256") == outcome_sha256
            and staged.get("llm_call_sha256") == llm_call_sha256
            and staged.get("settled_effect_seq") == prior.last_effect_seq
        )
        if not valid:
            raise ValidationError("TaskRun completed outcome lost its staging binding")
        self._validate_staged_generation_binding(
            pid,
            staged,
            generation=generation,
            outcome=outcome,
        )
        self._validate_effect_settlement_bundle(pid, staged)
        self._validate_current_staged_binding_for_pid(pid, staged)
        return prior, transcript, staged

    @staticmethod
    def _settled_transcript_matches(
        point: TaskRunResumePoint,
        transcript: Mapping[str, Any],
        *,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
    ) -> bool:
        return bool(
            outcome["state"] == "completed"
            and transcript.get("settled_call_id") == call_id
            and transcript.get("settled_outcome_sha256") == outcome_sha256
            and point.context_generation == generation
        )

    @staticmethod
    def _settled_wait_matches(
        staged: Mapping[str, Any],
        *,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
    ) -> bool:
        return bool(
            staged.get("kind") == "durable_wait_action"
            and outcome["state"] == "waiting"
            and staged.get("call_id") == call_id
            and staged.get("context_generation") == generation
            and staged.get("wait_outcome_sha256") == outcome_sha256
        )

    def _build_settled_transcript(
        self,
        *,
        pid: str,
        record: TaskRunRecord,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        prior: TaskRunResumePoint,
        prior_payload: Mapping[str, Any],
    ) -> tuple[TaskRunPayload, TaskRunPayload | None, TaskRunPayload | None, str] | None:
        new_messages = self._outcome_replay_messages(outcome)
        messages = [
            *(dict(item) for item in prior_payload.get("transcript_messages", [])),
            *new_messages,
        ]
        now = utc_now()
        bounded = self._bounded_transcript_projection(
            record.run_id,
            pid,
            messages,
            prior=prior,
            created_at=now,
            context_generation=generation,
            new_message_count=len(new_messages),
        )
        if bounded is None:
            self._mark_attention(
                record,
                self._blocker(
                    "pending_action_unreplayable",
                    "exact transcript exceeds the durable bound without a validated semantic compaction",
                    pid=pid,
                ),
            )
            return None
        messages, summary_payload = bounded
        labels = [DataLabels.from_dict(outcome["data_labels"])]
        if prior_payload.get("data_labels") is not None:
            labels.append(DataLabels.from_dict(prior_payload["data_labels"]))
        transcript_payload = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=record.run_id,
            role="transcript",
            label="Validated provider-independent transcript delta",
            value={
                "schema_version": 1,
                "call_id": call_id,
                "transcript_messages": messages,
                "data_labels": DataLabels.aggregate(labels).to_dict(),
                "latest_outcome_state": outcome["state"],
                "latest_outcome_sha256": outcome_sha256,
                "settled_call_id": call_id if outcome["state"] == "completed" else None,
                "settled_outcome_sha256": (
                    outcome_sha256 if outcome["state"] == "completed" else None
                ),
            },
            created_at=now,
        )
        self._require_payload_bound(transcript_payload)
        effective_summary = summary_payload or self._optional_resume_payload(
            prior.summary_payload_id
        )
        return transcript_payload, summary_payload, effective_summary, now

    def _settled_wait_payload(
        self,
        *,
        pid: str,
        run_id: str,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        llm_call_sha256: str,
        staged: Mapping[str, Any],
        created_at: str,
    ) -> TaskRunPayload | None:
        if outcome["state"] != "waiting":
            return None
        wait_snapshot = staged.get("wait_snapshot")
        if not isinstance(wait_snapshot, Mapping):
            raise ValidationError("TaskRun staged wait snapshot is missing")
        pending_service = getattr(self._host.llm, "pending", None)
        getter = getattr(pending_service, "get", None)
        current = self._durable_wait_snapshot(
            pid,
            getter(pid) if callable(getter) else None,
            call_id=call_id,
        )
        if (
            current != dict(wait_snapshot)
            or staged.get("wait_snapshot_sha256") != self._sha256(current)
            or staged.get("wait_identity_sha256")
            != self._durable_wait_identity_sha256(current)
        ):
            raise ValidationError("TaskRun durable wait changed before settlement")
        wrapper = {
            "schema_version": 1,
            "kind": "durable_wait_action",
            "state": "waiting",
            "call_id": call_id,
            "context_generation": generation,
            "llm_call_sha256": llm_call_sha256,
            "action_manifest_sha256": staged.get("action_manifest_sha256"),
            "wait_snapshot": current,
            "wait_snapshot_sha256": self._sha256(current),
            "wait_identity_sha256": self._durable_wait_identity_sha256(current),
            "wait_outcome_sha256": outcome_sha256,
        }
        payload = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=run_id,
            role="pending_action",
            label="Integrity-bound durable TaskRun wait",
            value=wrapper,
            created_at=created_at,
        )
        self._require_payload_bound(payload)
        return payload

    def _commit_completed_transcript(
        self,
        *,
        pid: str,
        run_id: str,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        staged: Mapping[str, Any],
        prior: TaskRunResumePoint,
        transcript_payload: TaskRunPayload,
        summary_payload: TaskRunPayload | None,
        effective_summary: TaskRunPayload | None,
        next_pending: TaskRunPayload | None,
        now: str,
    ) -> None:
        for attempt in range(5):
            process = self._store.get_process(pid)
            record = self._require_run(run_id)
            if process is None:
                raise NotFound(f"process not found: {pid}")
            self._require_settlement_epoch(process, record, "LLM settlement")
            if (
                outcome["state"] == "completed"
                and record.completed_step_count >= record.step_count
            ):
                if self._completed_settlement_is_current(
                    pid,
                    call_id=call_id,
                    generation=generation,
                    outcome=outcome,
                    outcome_sha256=outcome_sha256,
                ):
                    return
                raise ValidationError("TaskRun completed-step accounting is inconsistent")
            point = self._make_resume_point(
                process=process,
                record=record,
                context_generation=generation,
                safe_point_seq=prior.safe_point_seq + 1,
                transcript_payload=transcript_payload,
                summary_payload=effective_summary,
                pending_payload=next_pending,
                last_effect_seq=int(staged["settled_effect_seq"]),
                created_at=prior.created_at,
                updated_at=now,
            )
            try:
                self._commit_completed_transcript_once(
                    record=record,
                    point=point,
                    staged=staged,
                    outcome=outcome,
                    outcome_sha256=outcome_sha256,
                    call_id=call_id,
                    generation=generation,
                    transcript_payload=transcript_payload,
                    summary_payload=summary_payload,
                    next_pending=next_pending,
                    now=now,
                )
                return
            except TaskRunRevisionConflict:
                if self._completed_settlement_is_current(
                    pid,
                    call_id=call_id,
                    generation=generation,
                    outcome=outcome,
                    outcome_sha256=outcome_sha256,
                ):
                    return
                if attempt == 4:
                    raise

    def _commit_completed_transcript_once(
        self,
        *,
        record: TaskRunRecord,
        point: TaskRunResumePoint,
        staged: Mapping[str, Any],
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        call_id: str,
        generation: str,
        transcript_payload: TaskRunPayload,
        summary_payload: TaskRunPayload | None,
        next_pending: TaskRunPayload | None,
        now: str,
    ) -> None:
        with self._uow.transaction():
            current_record = self._require_run(record.run_id)
            if (
                current_record.revision != record.revision
                or current_record.runtime_epoch != record.runtime_epoch
            ):
                raise TaskRunRevisionConflict(
                    "TaskRun changed before completed transcript settlement"
                )
            record = current_record
            try:
                requirement_snapshot = {
                    requirement.requirement_id: requirement
                    for requirement in self._bounded_completion_requirements(record)
                }
            except ValidationError:
                latest = self._require_run(record.run_id)
                if (
                    latest.revision != record.revision
                    or latest.runtime_epoch != record.runtime_epoch
                ):
                    raise TaskRunRevisionConflict(
                        "TaskRun requirements changed during completed transcript "
                        "settlement"
                    )
                raise
            (
                completion_evidence,
                completed_requirement_outcomes,
                blocked_requirement_outcomes,
            ) = (
                self._validated_requirement_completion(
                    record=record,
                    point=point,
                    staged=staged,
                    outcome=outcome,
                    outcome_sha256=outcome_sha256,
                    call_id=call_id,
                    generation=generation,
                    current_requirements=requirement_snapshot,
                )
            )
            if summary_payload is not None:
                self._store.insert_task_run_payload(summary_payload)
            self._store.insert_task_run_payload(transcript_payload)
            if next_pending is not None:
                self._store.insert_task_run_payload(next_pending)
            self._store.upsert_task_run_resume_point(point)
            for outcome_item in completed_requirement_outcomes:
                requirement_id = outcome_item["requirement_id"]
                prior_requirement = requirement_snapshot.get(requirement_id)
                if prior_requirement is None:
                    raise ValidationError(
                        "TaskRun completed requirement disappeared"
                    )
                requirement = self._store.update_task_run_requirement_cas(
                    requirement_id,
                    expected_status=TaskRunRequirementStatus.IN_PROGRESS,
                    status=TaskRunRequirementStatus.SATISFIED,
                    updated_at=now,
                    started_at=prior_requirement.started_at,
                    completed_at=now,
                )
                if requirement is None:
                    raise TaskRunRevisionConflict(
                        "TaskRun requirement completion raced another mutation"
                    )
                requirement_snapshot[requirement_id] = requirement
                requirement_ledger = self._append_ledger(
                    record.run_id,
                    kind=TaskRunLedgerKind.REQUIREMENT,
                    status=TaskRunRequirementStatus.SATISFIED.value,
                    label="requirement satisfied by integrity-bound root exit",
                    requirement_id=requirement_id,
                    pid=point.pid,
                    llm_call_id=call_id,
                    metadata={
                        "context_generation": generation,
                        "completion_evidence_sha256": self._sha256(
                            completion_evidence
                        ),
                        "outcome_sha256": outcome_sha256,
                        "evidence_receipt_ids": list(
                            outcome_item["evidence_receipt_ids"]
                        ),
                    },
                )
                for receipt_id in outcome_item["evidence_receipt_ids"]:
                    self._store.insert_task_run_link(
                        TaskRunLink(
                            link_id=new_id("trlink"),
                            run_id=record.run_id,
                            ledger_seq=requirement_ledger.seq,
                            evidence_type="operation",
                            evidence_id=receipt_id,
                            role=f"requirement_completion:{requirement_id}",
                            created_at=now,
                            metadata={
                                "requirement_id": requirement_id,
                                "llm_call_id": call_id,
                            },
                        )
                    )
            for outcome_item in blocked_requirement_outcomes:
                requirement_id = outcome_item["requirement_id"]
                prior_requirement = requirement_snapshot.get(requirement_id)
                if prior_requirement is None:
                    raise ValidationError(
                        "TaskRun blocked requirement disappeared"
                    )
                requirement = self._store.update_task_run_requirement_cas(
                    requirement_id,
                    expected_status=TaskRunRequirementStatus.IN_PROGRESS,
                    status=TaskRunRequirementStatus.BLOCKED,
                    updated_at=now,
                    started_at=prior_requirement.started_at,
                    completed_at=None,
                )
                if requirement is None:
                    raise TaskRunRevisionConflict(
                        "TaskRun blocked requirement raced another mutation"
                    )
                requirement_snapshot[requirement_id] = requirement
                self._append_ledger(
                    record.run_id,
                    kind=TaskRunLedgerKind.REQUIREMENT,
                    status=TaskRunRequirementStatus.BLOCKED.value,
                    label="requirement unresolved by integrity-bound root exit",
                    requirement_id=requirement_id,
                    pid=point.pid,
                    llm_call_id=call_id,
                    metadata={
                        "context_generation": generation,
                        "completion_evidence_sha256": self._sha256(
                            completion_evidence
                        ),
                        "outcome_sha256": outcome_sha256,
                        "reported_status": outcome_item["reported_status"],
                        "evidence_receipt_ids": [],
                    },
                )
            updated = record
            if outcome["state"] == "completed":
                satisfied_requirement_count = sum(
                    requirement.status
                    in {
                        TaskRunRequirementStatus.SATISFIED,
                        TaskRunRequirementStatus.WAIVED,
                    }
                    for requirement in requirement_snapshot.values()
                )
                updated = self._store.update_task_run_cas(
                    record.run_id,
                    record.revision,
                    updates={
                        "completed_step_count": record.completed_step_count + 1,
                        "satisfied_requirement_count": satisfied_requirement_count,
                        "active_pid": point.pid,
                        "updated_at": now,
                    },
                    expected_runtime_epoch=self._runtime_epoch,
                )
            self._append_ledger(
                record.run_id,
                kind=TaskRunLedgerKind.LLM_TURN,
                status=str(outcome["state"]),
                label=(
                    "LLM action and paired outputs persisted"
                    if outcome["state"] == "completed"
                    else "LLM durable wait and paired outputs persisted"
                ),
                pid=point.pid,
                llm_call_id=call_id,
                payload_id=transcript_payload.payload_id,
                metadata={
                    "context_generation": generation,
                    "safe_point_seq": point.safe_point_seq,
                    "integrity_sha256": point.integrity_sha256,
                    "outcome_sha256": outcome_sha256,
                    "completed_step_count": updated.completed_step_count,
                },
            )

    def _validated_requirement_completion(
        self,
        *,
        record: TaskRunRecord,
        point: TaskRunResumePoint,
        staged: Mapping[str, Any],
        outcome: Mapping[str, Any],
        outcome_sha256: str,
        call_id: str,
        generation: str,
        current_requirements: Mapping[str, TaskRunRequirement],
    ) -> tuple[
        dict[str, Any] | None,
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
    ]:
        raw = staged.get("requirement_completion")
        if raw is None:
            return None, (), ()
        evidence = self._require_completion_evidence_binding(
            raw,
            record=record,
            point=point,
            staged=staged,
            outcome_sha256=outcome_sha256,
            call_id=call_id,
            generation=generation,
        )
        if self._committed_process_exit_action(outcome) is None:
            raise ValidationError(
                "TaskRun requirement completion lacks a committed root exit"
            )
        binding = self._validated_prompt_requirement_binding(
            record,
            pid=point.pid,
            context_generation=generation,
            value=evidence.get("requirement_binding"),
            allowed_statuses={TaskRunRequirementStatus.IN_PROGRESS},
            current_requirements=current_requirements,
        )
        if evidence.get("requirement_binding_sha256") != self._sha256(binding):
            raise ValidationError(
                "TaskRun requirement completion binding hash changed"
            )
        selected_action = self._committed_process_exit_action(outcome)
        assert selected_action is not None
        expected_outcomes = self._completion_requirement_outcomes(
            selected_action,
            binding,
            pid=point.pid,
            record=record,
            requirements=current_requirements.values(),
        )
        raw_outcomes = evidence.get("requirement_outcomes")
        if raw_outcomes != expected_outcomes:
            raise ValidationError(
                "TaskRun requirement completion outcomes changed"
            )
        for item in expected_outcomes:
            if set(item) != _REQUIREMENT_COMPLETION_OUTCOME_KEYS:
                raise ValidationError(
                    "TaskRun requirement completion outcome shape is invalid"
                )
            self._validate_requirement_completion_outcome(item)
        canonical_task_run_json(evidence)
        return (
            evidence,
            tuple(
                item
                for item in expected_outcomes
                if item["status"] == TaskRunRequirementStatus.SATISFIED.value
            ),
            tuple(
                item
                for item in expected_outcomes
                if item["status"] == TaskRunRequirementStatus.BLOCKED.value
            ),
        )

    @staticmethod
    def _validate_requirement_completion_outcome(item: Mapping[str, Any]) -> None:
        receipt_ids = item.get("evidence_receipt_ids")
        if (
            not isinstance(receipt_ids, list)
            or not all(isinstance(value, str) and value for value in receipt_ids)
            or len(receipt_ids) != len(set(receipt_ids))
        ):
            raise ValidationError(
                "TaskRun requirement completion receipt identities are invalid"
            )
        status = item.get("status")
        reported = item.get("reported_status")
        if status == TaskRunRequirementStatus.BLOCKED.value:
            if receipt_ids or reported not in {"blocked", "cancelled"}:
                raise ValidationError(
                    "blocked TaskRun requirement completion has invalid receipts"
                )
            return
        if status != TaskRunRequirementStatus.SATISFIED.value:
            raise ValidationError("TaskRun requirement completion status is invalid")
        if reported == "implicit_root_exit":
            if receipt_ids:
                raise ValidationError(
                    "implicit TaskRun completion cannot carry evidence receipts"
                )
            return
        if reported != "completed" or not receipt_ids:
            raise ValidationError(
                "completed TaskRun requirement lacks exact evidence receipts"
            )

    def _require_completion_evidence_binding(
        self,
        value: Any,
        *,
        record: TaskRunRecord,
        point: TaskRunResumePoint,
        staged: Mapping[str, Any],
        outcome_sha256: str,
        call_id: str,
        generation: str,
    ) -> dict[str, Any]:
        if (
            not isinstance(value, Mapping)
            or set(value) != _REQUIREMENT_COMPLETION_KEYS
        ):
            raise ValidationError(
                "TaskRun requirement completion evidence has an invalid shape"
            )
        evidence = dict(value)
        process = self._store.get_process(point.pid)
        identity = (
            evidence.get("schema_version") == 1
            and type(evidence.get("schema_version")) is int
            and evidence.get("kind") == "root_process_exit"
            and evidence.get("run_id") == record.run_id
            and evidence.get("pid") == point.pid
            and point.pid == record.root_pid
            and process is not None
            and process.status is ProcessStatus.EXITED
        )
        transcript = (
            evidence.get("call_id") == call_id
            and evidence.get("context_generation") == generation
            and evidence.get("action_manifest_sha256")
            == staged.get("action_manifest_sha256")
            and evidence.get("outcome_sha256") == outcome_sha256
            and evidence.get("outcome_sha256") == staged.get("outcome_sha256")
        )
        if not identity or not transcript:
            raise ValidationError("TaskRun requirement completion evidence changed")
        return evidence

    def _completed_settlement_is_current(
        self,
        pid: str,
        *,
        call_id: str,
        generation: str,
        outcome: Mapping[str, Any],
        outcome_sha256: str,
    ) -> bool:
        point = self._store.get_task_run_resume_point(pid, complete_only=True)
        if point is None or not self._resume_integrity_valid(point):
            return False
        if point.pending_action_payload_id is None:
            transcript = self._decode_payload(
                self._store.get_task_run_payload(point.transcript_payload_id),
                role="transcript",
            )
            return self._settled_transcript_matches(
                point,
                transcript,
                call_id=call_id,
                generation=generation,
                outcome=outcome,
                outcome_sha256=outcome_sha256,
            )
        return self._settled_wait_matches(
            self._decode_pending_resume_payload(point),
            call_id=call_id,
            generation=generation,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
        )

    def run_until_blocked(
        self,
        run_id: str,
        *,
        expected_revision: int,
        command_id: str,
        max_quanta: int | None = None,
    ) -> TaskRunSummary:
        if max_quanta is not None and (
            type(max_quanta) is not int or max_quanta <= 0
        ):
            raise ValidationError(
                "TaskRun max_quanta must be a positive integer when provided"
            )
        request = {
            "expected_revision": expected_revision,
            "max_quanta": max_quanta,
        }
        replay = self._run_command_replay(run_id, command_id, request)
        if replay is not None:
            return replay
        record = self._require_revision(run_id, expected_revision)
        if record.status in {
            TaskRunStatus.PAUSED,
            TaskRunStatus.CANCELLING,
            TaskRunStatus.FINALIZING,
            TaskRunStatus.NEEDS_ATTENTION,
            *TASK_RUN_TERMINAL_STATUSES,
        }:
            raise ValidationError(
                f"TaskRun cannot dispatch from {record.status.value}"
            )
        if self._deadline_expired(record):
            expired = self._expire(
                record,
                command_id=command_id,
                command_kind="run",
                command_request=request,
            )
            return self._complete_command_summary(
                expired, command_id, "run", request
            )
        resume_blocker = self._live_dispatch_resume_blocker(record.run_id)
        if resume_blocker is not None:
            attention = self._mark_attention(
                record,
                resume_blocker,
                command_id=command_id,
                command_kind="run",
                request=request,
            )
            return self._complete_command_summary(
                attention,
                command_id,
                "run",
                request,
            )
        unsettled_before_dispatch = self._unsettled_effects(record.run_id)
        if unsettled_before_dispatch:
            kind = (
                "unknown_effect"
                if any(
                    effect.transaction_state in _UNKNOWN_EFFECT_STATES
                    for effect in unsettled_before_dispatch
                )
                else "effect_unsettled"
            )
            attention = self._mark_attention(
                record,
                self._blocker(
                    kind,
                    "an external effect is not safely settled",
                ),
                command_id=command_id,
                command_kind="run",
                request=request,
            )
            return self._complete_command_summary(
                attention, command_id, "run", request
            )
        now = utc_now()
        with self._uow.transaction():
            running = self._store.update_task_run_cas(
                run_id,
                record.revision,
                updates={
                    "status": TaskRunStatus.RUNNING,
                    "started_at": record.started_at or now,
                    "blockers": (),
                    "updated_at": now,
                },
                expected_runtime_epoch=self._runtime_epoch,
            )
            admission = self._append_control_admission(
                record,
                running,
                command_id=command_id,
                command_kind="run",
                request=request,
                evidence={
                    "kind": "run",
                    "admission_revision": running.revision,
                },
                label="explicit dispatch admitted",
            )
            self._record_command(
                running,
                command_id=command_id,
                command_kind="run",
                request=request,
                result={
                    "settlement_state": "pending",
                    "admission_revision": running.revision,
                    **admission,
                },
            )
        self._notify_updated()
        remaining = max_quanta
        with self._dispatch(
            run_id,
            pause_generation=running.pause_generation,
        ) as admitted:
            while admitted:
                pids = self._member_pids(run_id)
                if not pids:
                    break
                batch = self._host.run_until_idle(
                    max_quanta=remaining,
                    pids=pids,
                    # ``max_quanta`` is an admission boundary for a Durable
                    # Run. Once a Provider/tool quantum has been admitted it
                    # must settle before this command returns; cancelling it
                    # at the budget boundary would manufacture an ambiguous
                    # external effect solely because a remote call took
                    # longer than the scheduler's bounded drain window.
                    cancel_inflight_on_budget_exhaustion=False,
                    # A TaskRun quantum stops at a durable Human boundary.
                    # Consuming the Runtime's interactive stdin queue here
                    # would turn SDK/HTTP ``run`` into a blocking prompt and
                    # hide the required WAITING_HUMAN structured result.
                    process_human_queue=False,
                )
                if remaining is not None:
                    remaining = max(0, remaining - len(batch))
                    if remaining == 0:
                        break
                latest_pids = self._member_pids(run_id)
                if latest_pids == pids or not any(
                    self._store.get_process(pid).status is ProcessStatus.RUNNABLE
                    for pid in latest_pids
                    if self._store.get_process(pid) is not None
                ):
                    break
        projected = self._project(self._require_run(run_id), allow_finalize=True)
        summary = self._complete_command_summary(
            projected,
            command_id,
            "run",
            request,
            extra={
                "settlement_state": "complete",
                "admission_revision": running.revision,
            },
        )
        self._notify_updated()
        return summary

    def _run_command_replay(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary | None:
        selected_id = self._identifier(command_id, "command_id")
        existing = self._store.get_task_run_command(run_id, selected_id)
        if existing is None:
            return None
        self._require_same_command(
            existing,
            "run",
            self._request_hash("run", request),
        )
        if existing.result.get("settlement_state") == "pending":
            if existing.result.get("settlement_kind") == "deadline":
                return self._settle_deadline_command(
                    run_id,
                    selected_id,
                    "run",
                    request,
                )
            return self._settle_run_command(run_id, selected_id, request)
        return self._summary_from_command(existing)

    def _settle_run_command(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        """Finish a committed run command without ever re-dispatching work."""

        existing = self._store.get_task_run_command(run_id, command_id)
        if existing is None:
            raise TaskRunRevisionConflict("TaskRun run receipt is missing")
        self._require_same_command(
            existing,
            "run",
            self._request_hash("run", request),
        )
        if existing.result.get("settlement_state") != "pending":
            return self._summary_from_command(existing)
        admission_revision = existing.result.get("admission_revision")
        if type(admission_revision) is not int or admission_revision <= 0:
            raise ValidationError(
                "TaskRun run receipt has an invalid admission revision"
            )
        current = self._require_run(run_id)
        if current.revision < admission_revision:
            raise ValidationError(
                "TaskRun run receipt lost its admission revision fence"
            )
        if (
            current.status not in TASK_RUN_TERMINAL_STATUSES
            and current.runtime_epoch != self._runtime_epoch
        ):
            raise TaskRunRevisionConflict(f"TaskRun epoch is stale: {run_id}")
        # The original command still owns its admitted scheduler/provider scope.
        # Returning its durable admission receipt is safe; completing or
        # redispatching it concurrently would not be.
        if (
            self._active_run_dispatches.get(run_id, 0) > 0
            or self._has_active_external_dispatch(run_id)
            or self._has_active_quantum(run_id)
        ):
            return self._summary_from_command(existing)
        if (
            current.status not in TASK_RUN_TERMINAL_STATUSES
            and self._deadline_expired(current)
        ):
            current = self._expire(current)
        elif current.status not in TASK_RUN_TERMINAL_STATUSES:
            current = self._project(current, allow_finalize=True)
        summary = self._complete_command_summary(
            current,
            command_id,
            "run",
            request,
            extra={
                "settlement_state": "complete",
                "admission_revision": admission_revision,
            },
        )
        self._notify_updated()
        return summary

    def _settle_deadline_command(
        self,
        run_id: str,
        command_id: str,
        command_kind: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        """Finish a deadline cancellation whose intent/receipt already committed."""

        existing = self._store.get_task_run_command(run_id, command_id)
        if existing is None:
            raise TaskRunRevisionConflict("TaskRun deadline receipt is missing")
        self._require_same_command(
            existing,
            command_kind,
            self._request_hash(command_kind, request),
        )
        if existing.result.get("settlement_state") != "pending":
            return self._summary_from_command(existing)
        if existing.result.get("settlement_kind") != "deadline":
            raise ValidationError("TaskRun deadline receipt has an invalid kind")
        cancel_generation = existing.result.get("cancel_generation")
        if type(cancel_generation) is not int or cancel_generation <= 0:
            raise ValidationError(
                "TaskRun deadline receipt has an invalid cancellation generation"
            )
        current = self._require_run(run_id)
        if current.cancel_generation < cancel_generation:
            raise ValidationError("TaskRun deadline receipt lost its generation fence")
        if (
            current.status not in TASK_RUN_TERMINAL_STATUSES
            and current.runtime_epoch != self._runtime_epoch
        ):
            raise TaskRunRevisionConflict(f"TaskRun epoch is stale: {run_id}")
        if (
            self._active_run_dispatches.get(run_id, 0) > 0
            or self._has_active_external_dispatch(run_id)
            or self._has_active_quantum(run_id)
        ):
            return self._summary_from_command(existing)
        if current.status not in TASK_RUN_TERMINAL_STATUSES:
            current = self._expire(current)
        summary = self._complete_command_summary(
            current,
            command_id,
            command_kind,
            request,
            extra={
                "settlement_state": "complete",
                "settlement_kind": "deadline",
                "cancel_generation": cancel_generation,
            },
        )
        self._notify_updated()
        return summary

    def _live_dispatch_resume_blocker(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:
        record = self._require_run(run_id)
        for process in self._tree_processes(run_id):
            if process.status in _TERMINAL_PROCESS_STATUSES:
                continue
            point = self._store.get_task_run_resume_point(
                process.pid,
                complete_only=True,
            )
            if point is None:
                continue
            if not self._resume_point_identity_valid(
                point,
                record=record,
                process=process,
            ) or not self._resume_static_integrity_valid(point):
                return self._blocker(
                    "payload_corrupt",
                    "resume point failed integrity before dispatch",
                    pid=process.pid,
                )
            if not self._resume_current_binding_valid(
                point,
                record=record,
                process=process,
            ):
                return self._blocker(
                    "binding_drift",
                    "resume point binding changed before dispatch",
                    pid=process.pid,
                )
        return None

    def wait(
        self,
        run_id: str,
        *,
        after_revision: int | None = None,
        timeout: float | None = None,
    ) -> TaskRunSummary:
        if timeout is not None:
            finite_timeout = False
            if not isinstance(timeout, bool) and isinstance(timeout, (int, float)):
                try:
                    finite_timeout = math.isfinite(timeout)
                except OverflowError:
                    finite_timeout = False
            if not finite_timeout or timeout < 0:
                raise ValidationError(
                    "TaskRun wait timeout must be a finite non-negative number"
                )
        if after_revision is not None and (
            type(after_revision) is not int or after_revision < 0
        ):
            raise ValidationError("TaskRun wait revision must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        baseline = self.get(run_id)
        target_revision = (
            baseline.revision if after_revision is None else after_revision
        )
        if baseline.status in _WAIT_BOUNDARY_STATUSES:
            return baseline
        with self._condition:
            while True:
                summary = self.get(run_id)
                if (
                    summary.revision > target_revision
                    or summary.status in _WAIT_BOUNDARY_STATUSES
                ):
                    return summary
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return summary
                self._condition.wait(timeout=min(0.25, remaining) if remaining is not None else 0.25)

    def pause(
        self,
        run_id: str,
        *,
        expected_revision: int,
        command_id: str,
    ) -> TaskRunSummary:
        self._begin_control_mutation(run_id, command_id)
        try:
            return self._pause_controlled(
                run_id,
                expected_revision=expected_revision,
                command_id=command_id,
            )
        finally:
            self._end_control_mutation(run_id, command_id)

    def _pause_controlled(
        self,
        run_id: str,
        *,
        expected_revision: int,
        command_id: str,
    ) -> TaskRunSummary:
        request: dict[str, Any] = {"expected_revision": expected_revision}
        replay = self._command_replay(run_id, command_id, "pause", request)
        if replay is not None:
            return replay
        record = self._require_revision(run_id, expected_revision)
        if record.status in TASK_RUN_TERMINAL_STATUSES or record.status in {
            TaskRunStatus.PAUSED,
            TaskRunStatus.CANCELLING,
            TaskRunStatus.FINALIZING,
            TaskRunStatus.NEEDS_ATTENTION,
        }:
            raise ValidationError(f"TaskRun cannot pause from {record.status.value}")
        # Serialize the durable generation with exact provider/tool admission.
        # Calls admitted before this transaction retain their execution lease;
        # calls arriving afterwards observe PAUSED and cannot start.
        with self._condition:
            with self._uow.transaction():
                paused = self._store.update_task_run_cas(
                    run_id,
                    record.revision,
                    updates={
                        "status": TaskRunStatus.PAUSED,
                        "pause_generation": record.pause_generation + 1,
                        "updated_at": utc_now(),
                    },
                    expected_runtime_epoch=self._runtime_epoch,
                )
                self._record_command(paused, command_id, "pause", request)
                self._status_ledger(record, paused, label="pause intent persisted")
            self._condition.notify_all()
        self._wait_for_dispatch_drain(run_id)
        for process in reversed(self._tree_processes(run_id)):
            if process.status in {ProcessStatus.RUNNABLE, ProcessStatus.RUNNING}:
                latest = self._store.get_process(process.pid)
                if latest is not None and latest.status is ProcessStatus.RUNNING:
                    attention = self._mark_attention(
                        self._require_run(run_id),
                        self._blocker(
                            "manual_recovery_required",
                            "pause drain ended with a live execution lease",
                            pid=process.pid,
                        ),
                    )
                    return self._complete_command_summary(
                        attention,
                        command_id,
                        "pause",
                        request,
                    )
                if latest is not None and latest.status is ProcessStatus.RUNNABLE:
                    self._process.pause_for_host_resume(
                        process.pid,
                        "TaskRun paused",
                    )
        latest = self._require_run(run_id)
        if latest.status not in TASK_RUN_TERMINAL_STATUSES:
            latest = self._reconcile_deadline(latest)
            latest = self._project_paused_terminal_root(latest)
        summary = self._complete_command_summary(
            latest,
            command_id,
            "pause",
            request,
        )
        self._notify_updated()
        return summary

    def resume(
        self,
        run_id: str,
        *,
        expected_revision: int,
        command_id: str,
    ) -> TaskRunSummary:
        request: dict[str, Any] = {"expected_revision": expected_revision}
        replay = self._resume_command_replay(
            run_id,
            command_id,
            request,
        )
        if replay is not None:
            return replay
        record = self._require_revision(run_id, expected_revision)
        if record.status is not TaskRunStatus.PAUSED:
            raise ValidationError("only a paused TaskRun may resume")
        if self._deadline_expired(record):
            expired = self._expire(
                record,
                command_id=command_id,
                command_kind="resume",
                command_request=request,
            )
            return self._complete_command_summary(
                expired, command_id, "resume", request
            )
        if self._unsettled_effects(run_id):
            attention = self._mark_attention(
                record,
                self._blocker("unknown_effect", "external effect is unresolved"),
                command_id=command_id,
                command_kind="resume",
                request=request,
            )
            return self._complete_command_summary(
                attention, command_id, "resume", request
            )
        with self._uow.transaction():
            # A different idempotency key may race an interrupt follow-up.
            # Re-read under the Store transaction and refuse to make the Run
            # dispatchable until the current-generation interrupt has safely
            # superseded its old prompt action and completed local settlement.
            record = self._require_revision(run_id, expected_revision)
            if self._pending_startup_interrupt(record) is not None:
                raise ValidationError(
                    "TaskRun interrupt settlement is pending; retry resume after it converges"
                )
            resumed = self._store.update_task_run_cas(
                run_id,
                record.revision,
                updates={"status": TaskRunStatus.RUNNING, "updated_at": utc_now()},
                expected_runtime_epoch=self._runtime_epoch,
            )
            admission = self._append_control_admission(
                record,
                resumed,
                command_id=command_id,
                command_kind="resume",
                request=request,
                evidence={
                    "kind": "resume",
                    "pause_generation": record.pause_generation,
                },
                label="resume admitted",
            )
            self._record_command(
                resumed,
                command_id,
                "resume",
                request,
                result={
                    "settlement_state": "pending",
                    "pause_generation": record.pause_generation,
                    **admission,
                },
            )
        return self._settle_resume_command(
            run_id,
            command_id,
            request,
        )

    def _resume_command_replay(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary | None:
        selected_id = self._identifier(command_id, "command_id")
        existing = self._store.get_task_run_command(run_id, selected_id)
        if existing is None:
            return None
        self._require_same_command(
            existing,
            "resume",
            self._request_hash("resume", request),
        )
        if existing.result.get("settlement_state") == "pending":
            if existing.result.get("settlement_kind") == "deadline":
                return self._settle_deadline_command(
                    run_id,
                    selected_id,
                    "resume",
                    request,
                )
            return self._settle_resume_command(run_id, selected_id, request)
        return self._summary_from_command(existing)

    def _settle_resume_command(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        """Idempotently finish the local phase of an admitted resume command."""

        existing = self._store.get_task_run_command(run_id, command_id)
        if existing is None:
            raise TaskRunRevisionConflict("TaskRun resume receipt is missing")
        self._require_same_command(
            existing,
            "resume",
            self._request_hash("resume", request),
        )
        if existing.result.get("settlement_state") != "pending":
            return self._summary_from_command(existing)
        pause_generation = existing.result.get("pause_generation")
        if type(pause_generation) is not int or pause_generation < 0:
            raise ValidationError(
                "TaskRun resume receipt has an invalid pause generation"
            )
        current = self._require_run(run_id)
        if current.runtime_epoch != self._runtime_epoch:
            raise TaskRunRevisionConflict(f"TaskRun epoch is stale: {run_id}")
        if current.pause_generation < pause_generation:
            raise ValidationError(
                "TaskRun resume receipt lost its pause generation fence"
            )

        # A later pause/control generation owns the Run.  Exact replay of the
        # older command settles to that newer safe state and never resumes it.
        if (
            current.pause_generation > pause_generation
            or current.status
            in {
                TaskRunStatus.CANCELLING,
                TaskRunStatus.FINALIZING,
                TaskRunStatus.NEEDS_ATTENTION,
                *TASK_RUN_TERMINAL_STATUSES,
            }
        ):
            return self._complete_command_summary(
                current,
                command_id,
                "resume",
                request,
                extra={
                    "settlement_state": "complete",
                    "pause_generation": pause_generation,
                },
            )
        if self._deadline_expired(current):
            current = self._expire(current)
            return self._complete_command_summary(
                current,
                command_id,
                "resume",
                request,
                extra={
                    "settlement_state": "complete",
                    "pause_generation": pause_generation,
                },
            )
        if self._unsettled_effects(run_id):
            current = self._mark_attention(
                current,
                self._blocker("unknown_effect", "external effect is unresolved"),
            )
            return self._complete_command_summary(
                current,
                command_id,
                "resume",
                request,
                extra={
                    "settlement_state": "complete",
                    "pause_generation": pause_generation,
                },
            )
        if current.status is TaskRunStatus.PAUSED:
            before = current
            with self._uow.transaction():
                current = self._store.update_task_run_cas(
                    run_id,
                    before.revision,
                    updates={
                        "status": TaskRunStatus.RUNNING,
                        "updated_at": utc_now(),
                    },
                    expected_runtime_epoch=self._runtime_epoch,
                )
                self._status_ledger(
                    before,
                    current,
                    label="resume settlement continued",
                )
        for process in self._tree_processes(run_id):
            if process.status is ProcessStatus.PAUSED:
                self._process.resume(process.pid)
        current = self._project(
            self._require_run(run_id),
            allow_finalize=True,
        )
        summary = self._complete_command_summary(
            current,
            command_id,
            "resume",
            request,
            extra={
                "settlement_state": "complete",
                "pause_generation": pause_generation,
            },
        )
        self._notify_updated()
        return summary

    def cancel(
        self,
        run_id: str,
        *,
        expected_revision: int,
        command_id: str,
        reason: str | None = None,
    ) -> TaskRunSummary:
        request = {
            "expected_revision": expected_revision,
            "reason": str(reason or "TaskRun cancelled"),
        }
        replay = self._cancel_command_replay(run_id, command_id, request)
        if replay is not None:
            return replay
        record = self._require_revision(run_id, expected_revision)
        if record.status in TASK_RUN_TERMINAL_STATUSES:
            raise ValidationError("terminal TaskRun cannot be cancelled")
        with self._condition:
            with self._uow.transaction():
                cancelling = self._store.update_task_run_cas(
                    run_id,
                    record.revision,
                    updates={
                        "status": TaskRunStatus.CANCELLING,
                        "cancel_generation": record.cancel_generation + 1,
                        "updated_at": utc_now(),
                    },
                    expected_runtime_epoch=self._runtime_epoch,
                )
                admission = self._append_control_admission(
                    record,
                    cancelling,
                    command_id=command_id,
                    command_kind="cancel",
                    request=request,
                    evidence={
                        "kind": "cancel",
                        "cancel_generation": cancelling.cancel_generation,
                    },
                    label="cancel intent persisted",
                )
                self._record_command(
                    cancelling,
                    command_id,
                    "cancel",
                    request,
                    result={
                        "settlement_state": "pending",
                        "cancel_generation": cancelling.cancel_generation,
                        **admission,
                    },
                )
            active_external = self._active_external_dispatches.get(run_id, 0) > 0
            self._condition.notify_all()
        unsafe = self._unsettled_effects(run_id)
        if unsafe:
            attention = self._mark_attention(
                cancelling,
                self._blocker(
                    (
                        "unknown_effect"
                        if any(
                            item.transaction_state in _UNKNOWN_EFFECT_STATES
                            for item in unsafe
                        )
                        else "effect_unsettled"
                    ),
                    "cancellation awaits authoritative effect settlement",
                    effect_ids=[item.effect_id for item in unsafe[:20]],
                ),
            )
            return self._complete_command_summary(
                attention,
                command_id,
                "cancel",
                request,
                extra={
                    "settlement_state": "complete",
                    "cancel_generation": cancelling.cancel_generation,
                },
            )
        if active_external:
            attention = self._mark_attention(
                cancelling,
                self._blocker(
                    "effect_unsettled",
                    "cancellation awaits an already-admitted call's local settlement",
                ),
            )
            return self._complete_command_summary(
                attention,
                command_id,
                "cancel",
                request,
                extra={
                    "settlement_state": "complete",
                    "cancel_generation": cancelling.cancel_generation,
                },
            )
        for process in reversed(self._tree_processes(run_id)):
            if process.status not in _TERMINAL_PROCESS_STATUSES:
                self._process.cancel(process.pid, request["reason"])
        projected = self._project(self._require_run(run_id), allow_finalize=True)
        summary = self._complete_command_summary(
            projected,
            command_id,
            "cancel",
            request,
            extra={
                "settlement_state": "complete",
                "cancel_generation": cancelling.cancel_generation,
            },
        )
        self._notify_updated()
        return summary

    def _cancel_command_replay(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary | None:
        selected_id = self._identifier(command_id, "command_id")
        existing = self._store.get_task_run_command(run_id, selected_id)
        if existing is None:
            return None
        self._require_same_command(
            existing,
            "cancel",
            self._request_hash("cancel", request),
        )
        if existing.result.get("settlement_state") == "pending":
            return self._settle_cancel_command(run_id, selected_id, request)
        return self._summary_from_command(existing)

    def _settle_cancel_command(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        """Idempotently finish local cancellation after an ambiguous response."""

        existing = self._store.get_task_run_command(run_id, command_id)
        if existing is None:
            raise TaskRunRevisionConflict("TaskRun cancel receipt is missing")
        self._require_same_command(
            existing,
            "cancel",
            self._request_hash("cancel", request),
        )
        if existing.result.get("settlement_state") != "pending":
            return self._summary_from_command(existing)
        cancel_generation = existing.result.get("cancel_generation")
        if type(cancel_generation) is not int or cancel_generation <= 0:
            raise ValidationError(
                "TaskRun cancel receipt has an invalid cancellation generation"
            )
        current = self._require_run(run_id)
        if current.cancel_generation < cancel_generation:
            raise ValidationError("TaskRun cancel receipt lost its generation fence")
        if (
            current.status not in TASK_RUN_TERMINAL_STATUSES
            and current.runtime_epoch != self._runtime_epoch
        ):
            raise TaskRunRevisionConflict(f"TaskRun epoch is stale: {run_id}")
        if (
            self._active_run_dispatches.get(run_id, 0) > 0
            or self._has_active_external_dispatch(run_id)
            or self._has_active_quantum(run_id)
        ):
            return self._summary_from_command(existing)
        if current.status not in TASK_RUN_TERMINAL_STATUSES:
            unsafe = self._unsettled_effects(run_id)
            if unsafe:
                current = self._mark_attention(
                    current,
                    self._blocker(
                        (
                            "unknown_effect"
                            if any(
                                item.transaction_state in _UNKNOWN_EFFECT_STATES
                                for item in unsafe
                            )
                            else "effect_unsettled"
                        ),
                        "cancellation awaits authoritative effect settlement",
                        effect_ids=[item.effect_id for item in unsafe[:20]],
                    ),
                )
            else:
                for process in reversed(self._tree_processes(run_id)):
                    if process.status not in _TERMINAL_PROCESS_STATUSES:
                        self._process.cancel(
                            process.pid,
                            str(request.get("reason") or "TaskRun cancelled"),
                        )
                current = self._project(
                    self._require_run(run_id),
                    allow_finalize=True,
                )
        summary = self._complete_command_summary(
            current,
            command_id,
            "cancel",
            request,
            extra={
                "settlement_state": "complete",
                "cancel_generation": cancel_generation,
            },
        )
        self._notify_updated()
        return summary

    def follow_up(
        self,
        run_id: str,
        body: Any,
        *,
        kind: str = "follow_up",
        required: bool = True,
        expected_revision: int,
        command_id: str,
    ) -> TaskRunSummary:
        self._begin_control_mutation(run_id, command_id)
        try:
            return self._follow_up_controlled(
                run_id,
                body,
                kind=kind,
                required=required,
                expected_revision=expected_revision,
                command_id=command_id,
            )
        finally:
            self._end_control_mutation(run_id, command_id)

    def _follow_up_controlled(
        self,
        run_id: str,
        body: Any,
        *,
        kind: str = "follow_up",
        required: bool = True,
        expected_revision: int,
        command_id: str,
    ) -> TaskRunSummary:
        request = {
            "expected_revision": expected_revision,
            "body": body,
            "kind": kind,
            "required": required,
        }
        replay = self._follow_up_command_replay(
            run_id,
            command_id,
            request,
        )
        if replay is not None:
            return replay
        if kind not in {"normal", "interrupt", "follow_up"}:
            raise ValidationError(
                "TaskRun follow-up kind must be normal or interrupt"
            )
        if type(required) is not bool:
            raise ValidationError("TaskRun follow-up required must be boolean")
        record = self._require_revision(run_id, expected_revision)
        if record.status in TASK_RUN_TERMINAL_STATUSES or record.status in {
            TaskRunStatus.CANCELLING,
            TaskRunStatus.FINALIZING,
        }:
            raise ValidationError("TaskRun no longer accepts follow-up requirements")
        if not self._root_accepts_follow_up(record):
            raise ValidationError(
                "TaskRun root process no longer accepts follow-up requirements"
            )
        now = utc_now()
        payload = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=run_id,
            role="follow_up",
            label="TaskRun follow-up",
            value={
                "body": body,
                "kind": str(kind),
                "data_labels": DataLabels(origin="host").to_dict(),
            },
            created_at=now,
        )
        self._require_payload_bound(payload)
        interrupt_control = kind == "interrupt"
        prior, committed, interrupt_resume_fences = self._commit_follow_up(
            run_id=run_id,
            expected_revision=expected_revision,
            payload=payload,
            required=required,
            kind=kind,
            command_id=command_id,
            request=request,
            interrupt_control=interrupt_control,
            now=now,
        )
        latest = (
            self._finish_interrupt_follow_up(
                prior,
                resume_fences=interrupt_resume_fences,
            )
            if interrupt_control
            else committed
        )
        extra = (
            self._interrupt_follow_up_result(
                settlement_state="complete",
                pause_generation=committed.pause_generation,
                cancel_generation=prior.cancel_generation,
                prior_status=prior.status,
            )
            if interrupt_control
            else None
        )
        summary = self._complete_command_summary(
            latest,
            command_id,
            "follow_up",
            request,
            extra=extra,
        )
        self._notify_updated()
        return summary

    def _commit_follow_up(
        self,
        *,
        run_id: str,
        expected_revision: int,
        payload: TaskRunPayload,
        required: bool,
        kind: str,
        command_id: str,
        request: Mapping[str, Any],
        interrupt_control: bool,
        now: str,
    ) -> tuple[
        TaskRunRecord,
        TaskRunRecord,
        tuple[tuple[str, int, int], ...],
    ]:
        # Requirement, durable ProcessMessage, control generation and the
        # idempotency receipt form one atomic unit.  Serializing this commit
        # with exact external admission makes an interrupt a real generation
        # fence rather than a best-effort prompt hint.
        with self._condition:
            with self._uow.transaction():
                # A different command id is not serialized by the idempotency
                # guard. Re-read the CAS target only after acquiring the Store
                # lock so a concurrent loser fails before deriving an ordinal
                # or inserting any child row.
                record = self._require_revision(run_id, expected_revision)
                if record.status in TASK_RUN_TERMINAL_STATUSES or record.status in {
                    TaskRunStatus.CANCELLING,
                    TaskRunStatus.FINALIZING,
                }:
                    raise ValidationError(
                        "TaskRun no longer accepts follow-up requirements"
                    )
                if not self._root_accepts_follow_up(record):
                    raise ValidationError(
                        "TaskRun root process no longer accepts follow-up requirements"
                    )
                requirement = self._new_follow_up_requirement(
                    record,
                    payload=payload,
                    required=required,
                    kind=kind,
                    now=now,
                )
                control_status = (
                    TaskRunStatus.PAUSED
                    if interrupt_control
                    and record.status in TASK_RUN_DISPATCHABLE_STATUSES
                    else record.status
                )
                interrupt_resume_fences = (
                    self._interrupt_resume_fences(record.run_id)
                    if interrupt_control
                    else ()
                )
                self._store.insert_task_run_payload(payload)
                self._store.insert_task_run_requirement(requirement)
                updated = self._store.update_task_run_cas(
                    record.run_id,
                    record.revision,
                    updates={
                        "status": control_status,
                        "pause_generation": (
                            record.pause_generation + (1 if interrupt_control else 0)
                        ),
                        "requirement_count": requirement.ordinal + 1,
                        "satisfied_requirement_count": (
                            record.satisfied_requirement_count
                            + (0 if required else 1)
                        ),
                        "updated_at": now,
                    },
                    expected_runtime_epoch=self._runtime_epoch,
                )
                self._append_ledger(
                    record.run_id,
                    kind=TaskRunLedgerKind.REQUIREMENT,
                    status=requirement.status.value,
                    label="follow-up appended",
                    requirement_id=requirement.requirement_id,
                    payload_id=payload.payload_id,
                    metadata={"required": required, "content_sha256": payload.sha256},
                )
                command_result = (
                    self._interrupt_follow_up_result(
                        settlement_state="pending",
                        pause_generation=updated.pause_generation,
                        cancel_generation=record.cancel_generation,
                        prior_status=record.status,
                        admission_runtime_epoch=record.runtime_epoch,
                        resume_fences=interrupt_resume_fences,
                    )
                    if interrupt_control
                    else None
                )
                if interrupt_control:
                    assert command_result is not None
                    command_result.update(
                        self._append_control_admission(
                            record,
                            updated,
                            command_id=command_id,
                            command_kind="follow_up",
                            request=request,
                            evidence={
                                "kind": "interrupt",
                                "pause_generation": updated.pause_generation,
                                "cancel_generation": record.cancel_generation,
                                "prior_status": record.status.value,
                                "interrupt_provenance_sha256": command_result[
                                    "interrupt_provenance_sha256"
                                ],
                            },
                            label="interrupt generation persisted",
                        )
                    )
                self._record_command(
                    updated,
                    command_id,
                    "follow_up",
                    request,
                    result=command_result,
                )
                if updated.root_pid is not None:
                    self._post_follow_up_message(
                        updated,
                        requirement=requirement,
                        payload=payload,
                        kind=kind,
                    )
            self._condition.notify_all()
        return record, updated, interrupt_resume_fences

    def _interrupt_resume_fences(
        self,
        run_id: str,
    ) -> tuple[tuple[str, int, int], ...]:
        """Freeze only members executing at interrupt admission.

        Startup recovery can write ``stale_execution_recovery`` only for a
        process whose prior Runtime left an execution claim active.  Runnable,
        waiting, queued, and already-paused members therefore have no valid
        interrupt-resume provenance and must not inflate the durable receipt.
        """

        fences = tuple(
            sorted(
                (
                    process.pid,
                    process.state_generation,
                    process.execution_generation,
                )
                for process in self._tree_processes(run_id)
                if process.status is ProcessStatus.RUNNING
            )
        )
        if len(fences) > self.config.task_runs.recovery_page_hard_limit:
            raise ValidationError(
                "TaskRun interrupt resume provenance exceeds its hard cap"
            )
        return fences

    def _new_follow_up_requirement(
        self,
        record: TaskRunRecord,
        *,
        payload: TaskRunPayload,
        required: bool,
        kind: str,
        now: str,
    ) -> TaskRunRequirement:
        """Derive one append position under the caller's Store transaction."""

        hard_limit = self.config.task_runs.recovery_page_hard_limit
        if record.requirement_count >= hard_limit:
            raise ValidationError(
                "TaskRun follow-up would exceed configured requirement "
                "recovery_page_hard_limit"
            )
        requirements = self._store.list_task_run_requirements(
            record.run_id,
            limit=hard_limit,
        )
        actual_count = len(requirements)
        if actual_count >= hard_limit:
            raise ValidationError(
                "TaskRun follow-up would exceed configured requirement "
                "recovery_page_hard_limit"
            )
        if actual_count != record.requirement_count or any(
            item.ordinal != ordinal
            for ordinal, item in enumerate(requirements)
        ):
            raise ValidationError(
                "TaskRun requirement projection is inconsistent with its append order"
            )
        return TaskRunRequirement(
            requirement_id=new_id("trreq"),
            run_id=record.run_id,
            ordinal=actual_count,
            kind=TaskRunRequirementKind.FOLLOW_UP,
            status=(
                TaskRunRequirementStatus.PENDING
                if required
                else TaskRunRequirementStatus.WAIVED
            ),
            payload_id=payload.payload_id,
            requirement_sha256=payload.sha256,
            label=str(kind),
            created_by="host",
            created_at=now,
            updated_at=now,
            waived_by=None if required else "host",
            waiver_reason=None if required else "non-required follow-up",
        )

    def _post_follow_up_message(
        self,
        record: TaskRunRecord,
        *,
        requirement: TaskRunRequirement,
        payload: TaskRunPayload,
        kind: str,
    ) -> None:
        assert record.root_pid is not None
        # Recheck at the durable delivery boundary.  The public preflight keeps
        # the ordinary terminal-root case side-effect free, while this check
        # closes the race with a root that exits before the enclosing
        # requirement/message transaction begins.
        if not self._root_accepts_follow_up(record):
            raise ValidationError(
                "TaskRun root process no longer accepts follow-up requirements"
            )
        self._messages.post(
            sender="host",
            recipient_pid=record.root_pid,
            kind=(
                ProcessMessageKind.INTERRUPT
                if kind == "interrupt"
                else ProcessMessageKind.NORMAL
            ),
            channel="task-run-follow-up",
            correlation_id=requirement.requirement_id,
            subject="Durable TaskRun follow-up",
            body="A durable TaskRun follow-up is available locally.",
            payload={
                "run_id": record.run_id,
                "requirement_id": requirement.requirement_id,
                "payload_sha256": payload.sha256,
            },
            metadata={
                "task_run_id": record.run_id,
                "task_run_payload_ref": payload.payload_id,
            },
        )

    def _finish_interrupt_follow_up(
        self,
        prior: TaskRunRecord,
        *,
        resume_fences: tuple[tuple[str, int, int], ...],
    ) -> TaskRunRecord:
        return self._finish_interrupt_follow_up_state(
            prior.run_id,
            pause_generation=prior.pause_generation + 1,
            cancel_generation=prior.cancel_generation,
            prior_status=prior.status,
            admission_runtime_epoch=prior.runtime_epoch,
            resume_fences=resume_fences,
        )

    def _follow_up_command_replay(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary | None:
        selected_id = self._identifier(command_id, "command_id")
        existing = self._store.get_task_run_command(run_id, selected_id)
        if existing is None:
            return None
        self._require_same_command(
            existing,
            "follow_up",
            self._request_hash("follow_up", request),
        )
        if request.get("kind") == "interrupt":
            parsed = self._interrupt_follow_up_receipt_values(existing)
            if parsed is not None:
                return self._settle_interrupt_follow_up_command(
                    run_id,
                    selected_id,
                    request,
                )
        elif set(existing.result) != _COMMAND_RESULT_BASE_KEYS:
            raise ValidationError(
                "TaskRun non-interrupt follow-up has an invalid result schema"
            )
        return self._summary_from_command(existing)

    def _settle_interrupt_follow_up_command(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        receipt = self._pending_interrupt_follow_up_receipt(
            run_id,
            command_id,
            request,
        )
        if isinstance(receipt, TaskRunSummary):
            return receipt
        (
            pause_generation,
            cancel_generation,
            prior_status,
            admission_runtime_epoch,
            resume_fences,
        ) = receipt
        current = self._finish_interrupt_follow_up_state(
            run_id,
            pause_generation=pause_generation,
            cancel_generation=cancel_generation,
            prior_status=prior_status,
            admission_runtime_epoch=admission_runtime_epoch,
            resume_fences=resume_fences,
        )
        summary = self._complete_command_summary(
            current,
            command_id,
            "follow_up",
            request,
            extra=self._interrupt_follow_up_result(
                settlement_state="complete",
                pause_generation=pause_generation,
                cancel_generation=cancel_generation,
                prior_status=prior_status,
            ),
        )
        self._notify_updated()
        return summary

    def _pending_interrupt_follow_up_receipt(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> tuple[
        int,
        int,
        TaskRunStatus,
        int,
        tuple[tuple[str, int, int], ...],
    ] | TaskRunSummary:
        existing = self._store.get_task_run_command(run_id, command_id)
        if existing is None:
            raise TaskRunRevisionConflict("TaskRun follow-up receipt is missing")
        self._require_same_command(
            existing,
            "follow_up",
            self._request_hash("follow_up", request),
        )
        parsed = self._interrupt_follow_up_receipt_values(existing)
        if parsed is None:
            return self._summary_from_command(existing)
        return parsed

    def _interrupt_follow_up_receipt_values(
        self,
        existing: TaskRunCommand,
    ) -> tuple[
        int,
        int,
        TaskRunStatus,
        int,
        tuple[tuple[str, int, int], ...],
    ] | None:
        (
            settlement_state,
            pause_generation,
            cancel_generation,
            selected_status,
            admission_runtime_epoch,
            selected_fences,
        ) = self._validated_interrupt_receipt(existing)
        if settlement_state == "complete":
            return None
        assert admission_runtime_epoch is not None
        return (
            pause_generation,
            cancel_generation,
            selected_status,
            admission_runtime_epoch,
            selected_fences,
        )

    def _validated_interrupt_receipt(
        self,
        existing: TaskRunCommand,
    ) -> tuple[
        str,
        int,
        int,
        TaskRunStatus,
        int | None,
        tuple[tuple[str, int, int], ...],
    ]:
        self._summary_from_command(existing)
        result = existing.result
        settlement_state = self._settlement_state(
            result,
            "interrupt follow-up receipt",
        )
        expected_keys = (
            _INTERRUPT_PENDING_RESULT_KEYS
            if settlement_state == "pending"
            else _INTERRUPT_COMPLETE_RESULT_KEYS
        )
        self._require_result_keys(
            result,
            expected_keys,
            "interrupt follow-up receipt",
        )
        if result.get("settlement_kind") != "interrupt":
            raise ValidationError("TaskRun follow-up receipt has an invalid kind")
        pause_generation, cancel_generation, selected_status = (
            self._validated_interrupt_receipt_header(result)
        )
        provenance_sha256 = result.get("interrupt_provenance_sha256")
        if (
            type(provenance_sha256) is not str
            or len(provenance_sha256) != 64
            or any(char not in "0123456789abcdef" for char in provenance_sha256)
        ):
            raise ValidationError(
                "TaskRun interrupt receipt has invalid admission provenance"
            )
        self._validate_control_admission(
            existing,
            evidence={
                "kind": "interrupt",
                "pause_generation": pause_generation,
                "cancel_generation": cancel_generation,
                "prior_status": selected_status.value,
                "interrupt_provenance_sha256": provenance_sha256,
            },
            label="interrupt generation persisted",
        )
        if settlement_state == "complete":
            return (
                settlement_state,
                pause_generation,
                cancel_generation,
                selected_status,
                None,
                (),
            )
        admission_runtime_epoch = self._bounded_receipt_integer(
            result.get("admission_runtime_epoch"),
            "interrupt admission Runtime epoch",
            minimum=1,
        )
        selected_fences = self._validated_interrupt_resume_fences(
            result.get("resume_fences")
        )
        if provenance_sha256 != self._sha256(
            {
                "schema_version": 1,
                "admission_runtime_epoch": admission_runtime_epoch,
                "resume_fences": [list(item) for item in selected_fences],
            }
        ):
            raise ValidationError(
                "TaskRun interrupt receipt lost its resume provenance binding"
            )
        return (
            settlement_state,
            pause_generation,
            cancel_generation,
            selected_status,
            admission_runtime_epoch,
            selected_fences,
        )

    def _validated_interrupt_receipt_header(
        self,
        result: Mapping[str, Any],
    ) -> tuple[int, int, TaskRunStatus]:
        pause_generation = self._bounded_receipt_integer(
            result.get("pause_generation"),
            "interrupt pause generation",
            minimum=1,
        )
        cancel_generation = self._bounded_receipt_integer(
            result.get("cancel_generation"),
            "interrupt cancellation generation",
            minimum=0,
        )
        prior_status = result.get("prior_status")
        if type(prior_status) is not str:
            raise ValidationError(
                "TaskRun follow-up receipt has an invalid prior status"
            )
        try:
            selected_status = TaskRunStatus(prior_status)
        except ValueError as exc:
            raise ValidationError(
                "TaskRun follow-up receipt has an invalid prior status"
            ) from exc
        if selected_status in {
            TaskRunStatus.CANCELLING,
            TaskRunStatus.FINALIZING,
            *TASK_RUN_TERMINAL_STATUSES,
        }:
            raise ValidationError(
                "TaskRun follow-up receipt has a forbidden prior status"
            )
        return (
            pause_generation,
            cancel_generation,
            selected_status,
        )

    def _validated_interrupt_resume_fences(
        self,
        raw_fences: Any,
    ) -> tuple[tuple[str, int, int], ...]:
        if (
            not isinstance(raw_fences, list)
            or len(raw_fences) > self.config.task_runs.recovery_page_hard_limit
        ):
            raise ValidationError(
                "TaskRun follow-up receipt has invalid resume provenance"
            )
        fences: list[tuple[str, int, int]] = []
        for value in raw_fences:
            if (
                not isinstance(value, list)
                or len(value) != 3
            ):
                raise ValidationError(
                    "TaskRun follow-up receipt has invalid resume provenance"
                )
            state_generation = self._bounded_receipt_integer(
                value[1],
                "interrupt resume state generation",
                minimum=0,
            )
            execution_generation = self._bounded_receipt_integer(
                value[2],
                "interrupt resume execution generation",
                minimum=1,
            )
            fences.append(
                (
                    self._identifier(value[0], "interrupt resume PID"),
                    state_generation,
                    execution_generation,
                )
            )
        selected_fences = tuple(fences)
        if selected_fences != tuple(sorted(set(selected_fences))):
            raise ValidationError(
                "TaskRun follow-up receipt resume provenance is not canonical"
            )
        if len({pid for pid, _state, _execution in selected_fences}) != len(
            selected_fences
        ):
            raise ValidationError(
                "TaskRun follow-up receipt resume provenance PID is not unique"
            )
        return selected_fences

    def _finish_interrupt_follow_up_state(
        self,
        run_id: str,
        *,
        pause_generation: int,
        cancel_generation: int,
        prior_status: TaskRunStatus,
        admission_runtime_epoch: int,
        resume_fences: tuple[tuple[str, int, int], ...],
    ) -> TaskRunRecord:
        current = self._require_run(run_id)
        if admission_runtime_epoch > current.runtime_epoch:
            raise ValidationError(
                "TaskRun interrupt receipt is ahead of its Runtime epoch fence"
            )
        if not self._interrupt_generation_is_current(
            current,
            pause_generation=pause_generation,
            cancel_generation=cancel_generation,
        ):
            return current
        if current.status is TaskRunStatus.PAUSED:
            self._wait_for_dispatch_drain(run_id)
            current = self._require_run(run_id)
            if not self._interrupt_generation_is_current(
                current,
                pause_generation=pause_generation,
                cancel_generation=cancel_generation,
            ):
                return current
            self._supersede_unstarted_actions_for_interrupt(run_id)
            current = self._restore_interrupt_prior_status(
                self._require_run(run_id),
                prior_status=prior_status,
            )
        current = self._resume_reopened_interrupt_processes(
            current,
            pause_generation=pause_generation,
            cancel_generation=cancel_generation,
            prior_status=prior_status,
            admission_runtime_epoch=admission_runtime_epoch,
            resume_fences=resume_fences,
        )
        if (
            current.status is not TaskRunStatus.QUEUED
            and current.status not in {
                TaskRunStatus.PAUSED,
                TaskRunStatus.NEEDS_ATTENTION,
            }
        ):
            current = self._project(current, allow_finalize=True)
        return current

    def _interrupt_generation_is_current(
        self,
        record: TaskRunRecord,
        *,
        pause_generation: int,
        cancel_generation: int,
    ) -> bool:
        if record.pause_generation < pause_generation:
            raise ValidationError("TaskRun interrupt receipt lost its generation fence")
        if record.cancel_generation < cancel_generation:
            raise ValidationError("TaskRun interrupt receipt lost its cancellation fence")
        if (
            record.status not in TASK_RUN_TERMINAL_STATUSES
            and record.runtime_epoch != self._runtime_epoch
        ):
            raise TaskRunRevisionConflict(f"TaskRun epoch is stale: {record.run_id}")
        return bool(
            record.pause_generation == pause_generation
            and record.cancel_generation == cancel_generation
            and record.status
            not in {
                TaskRunStatus.CANCELLING,
                TaskRunStatus.FINALIZING,
                TaskRunStatus.NEEDS_ATTENTION,
                *TASK_RUN_TERMINAL_STATUSES,
            }
        )

    def _restore_interrupt_prior_status(
        self,
        current: TaskRunRecord,
        *,
        prior_status: TaskRunStatus,
    ) -> TaskRunRecord:
        if (
            current.status is not TaskRunStatus.PAUSED
            or prior_status not in TASK_RUN_DISPATCHABLE_STATUSES
        ):
            return current
        with self._uow.transaction():
            restored = self._store.update_task_run_cas(
                current.run_id,
                current.revision,
                updates={"status": prior_status, "updated_at": utc_now()},
                expected_runtime_epoch=self._runtime_epoch,
            )
            self._status_ledger(
                current,
                restored,
                label="interrupt safe drain completed",
            )
        return restored

    def _resume_reopened_interrupt_processes(
        self,
        current: TaskRunRecord,
        *,
        pause_generation: int,
        cancel_generation: int,
        prior_status: TaskRunStatus,
        admission_runtime_epoch: int,
        resume_fences: tuple[tuple[str, int, int], ...],
    ) -> TaskRunRecord:
        if (
            current.status is not prior_status
            or prior_status not in TASK_RUN_DISPATCHABLE_STATUSES
            or not self._interrupt_generation_is_current(
                current,
                pause_generation=pause_generation,
                cancel_generation=cancel_generation,
            )
        ):
            return current
        with self._condition:
            latest = self._require_run(current.run_id)
            if not self._interrupt_generation_is_current(
                latest,
                pause_generation=pause_generation,
                cancel_generation=cancel_generation,
            ) or latest.status is not prior_status:
                return latest
            stale_processes, blocker = self._interrupt_resume_plan(
                latest,
                admission_runtime_epoch=admission_runtime_epoch,
                resume_fences=resume_fences,
            )
            if blocker is not None:
                return self._mark_attention(
                    latest,
                    blocker,
                )
            for process in stale_processes:
                self._process.resume(process.pid)
            self._condition.notify_all()
        return self._require_run(current.run_id)

    def _interrupt_resume_plan(
        self,
        record: TaskRunRecord,
        *,
        admission_runtime_epoch: int,
        resume_fences: tuple[tuple[str, int, int], ...],
    ) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
        """Prevalidate every fenced member before returning any mutation plan."""

        fence_by_pid = {
            pid: (state_generation, execution_generation)
            for pid, state_generation, execution_generation in resume_fences
        }
        seen_fence_pids: set[str] = set()
        stale_processes: list[Any] = []
        reopened = self._runtime_epoch > admission_runtime_epoch
        for process in self._tree_processes(record.run_id):
            self._require_interrupt_process_membership(record, process)
            fence = fence_by_pid.get(process.pid)
            if fence is not None:
                seen_fence_pids.add(process.pid)
            if isinstance(process.wait_state, StaleExecutionProcessWait):
                blocker = self._stale_interrupt_member_blocker(
                    record,
                    process,
                    fence=fence,
                    reopened=reopened,
                )
                if blocker is None:
                    stale_processes.append(process)
            elif fence is not None and reopened:
                blocker = self._settled_interrupt_member_blocker(
                    record,
                    process,
                    fence=fence,
                )
            elif fence is not None and process.status is ProcessStatus.PAUSED:
                blocker = self._blocker(
                    "manual_recovery_required",
                    "paused interrupt member has no typed stale-execution receipt",
                    pid=process.pid,
                )
            else:
                blocker = None
            if blocker is not None:
                return (), blocker
        if seen_fence_pids != set(fence_by_pid):
            return (), self._blocker(
                "manual_recovery_required",
                "interrupt resume provenance references a missing process",
            )
        return tuple(stale_processes), None

    def _require_interrupt_process_membership(
        self,
        record: TaskRunRecord,
        process: Any,
    ) -> None:
        if (
            process.task_run_id != record.run_id
            or process.task_run_epoch != self._runtime_epoch
        ):
            raise ValidationError(
                "TaskRun interrupt resume provenance crossed its Run fence"
            )

    def _stale_interrupt_member_blocker(
        self,
        record: TaskRunRecord,
        process: Any,
        *,
        fence: tuple[int, int] | None,
        reopened: bool,
    ) -> dict[str, Any] | None:
        receipt = process.wait_state
        if (
            not isinstance(receipt, StaleExecutionProcessWait)
            or not reopened
            or fence is None
            or not self._stale_execution_receipt_is_valid(
                process,
                receipt=receipt,
                admission_state_generation=fence[0],
                admission_execution_generation=fence[1],
            )
        ):
            return self._blocker(
                "manual_recovery_required",
                "stale process pause is not bound to this interrupt generation",
                pid=process.pid,
            )
        point = self._store.get_task_run_resume_point(
            process.pid,
            complete_only=True,
        )
        if (
            point is None
            or point.pending_action_payload_id is not None
            or not self._resume_integrity_valid(
                point,
                record=record,
                process=process,
            )
        ):
            return self._blocker(
                "pending_action_unreplayable",
                "stale process has no complete generation-bound safe point",
                pid=process.pid,
            )
        return None

    def _settled_interrupt_member_blocker(
        self,
        record: TaskRunRecord,
        process: Any,
        *,
        fence: tuple[int, int],
    ) -> dict[str, Any] | None:
        point = self._store.get_task_run_resume_point(
            process.pid,
            complete_only=True,
        )
        if point is None or not self._resume_integrity_valid(
            point,
            record=record,
            process=process,
        ):
            return self._blocker(
                "pending_action_unreplayable",
                "settled interrupt member has no complete generation-bound safe point",
                pid=process.pid,
            )
        if not self._reopened_interrupt_member_is_settled(
            process,
            point=point,
            admission_state_generation=fence[0],
            admission_execution_generation=fence[1],
        ):
            return self._blocker(
                "manual_recovery_required",
                "interrupt member is not in an exact recovered or settled posture",
                pid=process.pid,
            )
        return None

    def _stale_execution_receipt_is_valid(
        self,
        process: Any,
        *,
        receipt: StaleExecutionProcessWait,
        admission_state_generation: int,
        admission_execution_generation: int,
    ) -> bool:
        """Validate Store takeover provenance without compatibility strings."""

        return bool(
            process.status is ProcessStatus.PAUSED
            and process.wait_state is receipt
            and receipt.pid == process.pid
            and receipt.prior_owner_sha256 is not None
            and receipt.prior_lease_sha256 is not None
            and receipt.recovered_by_owner_sha256
            != receipt.prior_owner_sha256
            and receipt.recovered_state_generation == process.state_generation
            and process.state_generation == admission_state_generation + 1
            and receipt.prior_execution_generation
            == admission_execution_generation
            and receipt.recovered_execution_generation
            == receipt.prior_execution_generation + 1
            and receipt.recovered_execution_generation
            == process.execution_generation
            and process.execution_owner_id is None
            and process.execution_lease_id is None
        )

    def _reopened_interrupt_member_is_settled(
        self,
        process: Any,
        *,
        point: TaskRunResumePoint,
        admission_state_generation: int,
        admission_execution_generation: int,
    ) -> bool:
        """Accept only exact, provider-free post-admission process postures."""

        if (
            process.execution_owner_id is not None
            or process.execution_lease_id is not None
        ):
            return False
        if (
            process.execution_generation == admission_execution_generation + 1
            and process.state_generation == admission_state_generation + 2
        ):
            return bool(
                process.status is ProcessStatus.RUNNABLE
                and process.wait_state is None
                and process.outcome is None
                and point.pending_action_payload_id is None
            )
        if process.state_generation != admission_state_generation + 1:
            return False
        if process.status in _TERMINAL_PROCESS_STATUSES:
            return bool(
                process.execution_generation
                == admission_execution_generation + 1
                and process.wait_state is None
                and process.outcome is not None
                and point.pending_action_payload_id is None
            )
        if process.execution_generation != admission_execution_generation:
            return False
        if process.status is ProcessStatus.RUNNABLE:
            return bool(
                process.wait_state is None
                and process.outcome is None
                and point.pending_action_payload_id is None
            )
        if process.status in {
            ProcessStatus.WAITING_EVENT,
            ProcessStatus.WAITING_HUMAN,
            ProcessStatus.WAITING_TOOL,
        }:
            return bool(
                process.wait_state is not None
                and process.outcome is None
                and point.pending_action_payload_id is not None
            )
        return False

    @staticmethod
    def _is_stale_execution_pause(process: Any) -> bool:
        """Identify the typed Store recovery state for fault-injection hooks."""

        return bool(
            process.status is ProcessStatus.PAUSED
            and isinstance(process.wait_state, StaleExecutionProcessWait)
        )

    @staticmethod
    def _interrupt_follow_up_result(
        *,
        settlement_state: str,
        pause_generation: int,
        cancel_generation: int,
        prior_status: TaskRunStatus,
        admission_runtime_epoch: int | None = None,
        resume_fences: tuple[tuple[str, int, int], ...] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "settlement_state": settlement_state,
            "settlement_kind": "interrupt",
            "pause_generation": pause_generation,
            "cancel_generation": cancel_generation,
            "prior_status": prior_status.value,
        }
        if resume_fences is not None:
            if (
                type(admission_runtime_epoch) is not int
                or admission_runtime_epoch <= 0
            ):
                raise ValidationError(
                    "pending TaskRun interrupt requires an admission Runtime epoch"
                )
            result["admission_runtime_epoch"] = admission_runtime_epoch
            selected_fences = [list(item) for item in resume_fences]
            result["resume_fences"] = selected_fences
            result["interrupt_provenance_sha256"] = TaskRunManager._sha256(
                {
                    "schema_version": 1,
                    "admission_runtime_epoch": admission_runtime_epoch,
                    "resume_fences": selected_fences,
                }
            )
        return result

    def purge_payloads(
        self,
        run_id: str,
        *,
        expected_revision: int,
        command_id: str,
    ) -> TaskRunSummary:
        """Host-only audited content purge for a terminal permanent Run."""

        request: dict[str, Any] = {"expected_revision": expected_revision}
        replay = self._command_replay(run_id, command_id, "purge_payloads", request)
        if replay is not None:
            return replay
        now = utc_now()
        with self._uow.transaction():
            record = self._require_expected_revision(run_id, expected_revision)
            if record.status not in TASK_RUN_TERMINAL_STATUSES:
                raise ValidationError(
                    "only a terminal TaskRun may be explicitly purged"
                )
            if record.retention is not TaskRunRetention.PERMANENT:
                raise ValidationError(
                    "explicit purge is reserved for permanent-retention TaskRuns"
                )
            if record.payloads_purged_at is not None:
                raise ValidationError("TaskRun payloads were already purged")
            if record.runtime_epoch != self._runtime_epoch:
                record = self._store.claim_terminal_task_run_epoch(
                    record.run_id,
                    record.revision,
                    self._runtime_epoch,
                )
            self._purge_run_content(record.run_id, purged_at=now)
            updated = self._store.update_task_run_cas(
                record.run_id,
                record.revision,
                updates={"payloads_purged_at": now, "updated_at": now},
                expected_runtime_epoch=self._runtime_epoch,
            )
            ledger = self._append_ledger(
                record.run_id,
                kind=TaskRunLedgerKind.STATUS_TRANSITION,
                status=record.status.value,
                label="permanent TaskRun content explicitly purged",
                metadata={"purged_at": now, "actor": "host"},
            )
            audit = self._audit.record(
                actor="host",
                action="task_run.payloads.purge",
                target=f"task_run:{record.run_id}",
                decision={
                    "retention": record.retention.value,
                    "status": record.status.value,
                    "purged_at": now,
                },
            )
            self._store.insert_task_run_link(
                TaskRunLink(
                    link_id=new_id("trlink"),
                    run_id=record.run_id,
                    ledger_seq=ledger.seq,
                    evidence_type="audit",
                    evidence_id=audit.record_id,
                    role="purge",
                    created_at=now,
                )
            )
            self._record_command(
                updated,
                command_id,
                "purge_payloads",
                request,
            )
        summary = self._complete_command_summary(
            updated,
            command_id,
            "purge_payloads",
            request,
        )
        self._notify_updated()
        return summary

    # ------------------------------------------------------------------
    # Startup recovery and evidence-based manual recovery

    def validate_recoverable_payloads(self) -> None:
        """Integrity-check recoverable Run payloads without writing or dispatching."""

        cursor: TaskRunCursor | None = None
        while True:
            page = self._store.list_recoverable_task_runs(
                after=cursor,
                limit=self.config.task_runs.recovery_page_size,
            )
            for record in page.records:
                try:
                    self._prevalidate_recoverable_record(record)
                except NotFound:
                    self._prevalidated_blockers[record.run_id] = self._blocker(
                        "payload_missing", "recoverable TaskRun payload is missing"
                    )
                except (TypeError, ValueError, ValidationError):
                    self._prevalidated_blockers[record.run_id] = self._blocker(
                        "payload_corrupt", "recoverable TaskRun payload failed integrity"
                    )
            cursor = page.next_cursor
            if cursor is None:
                return

    def _prevalidate_recoverable_record(self, record: TaskRunRecord) -> None:
        goal = self._payload_by_role(record.run_id, "goal")
        decoded = self._decode_payload(goal, role="goal")
        if set(decoded) != {"goal", "data_labels"}:
            raise ValidationError("TaskRun goal payload shape is invalid")
        DataLabels.from_dict(decoded["data_labels"])
        self._prevalidate_recoverable_requirements(record)
        self._prevalidate_recoverable_resume_points(record)

    def _prevalidate_recoverable_requirements(
        self,
        record: TaskRunRecord,
    ) -> None:
        after: tuple[int, str] | None = None
        count = 0
        page_size = self.config.task_runs.recovery_page_size
        while True:
            requirements = self._store.list_task_run_requirements(
                record.run_id,
                after=after,
                limit=page_size,
            )
            for requirement in requirements:
                if (
                    requirement.run_id != record.run_id
                    or requirement.ordinal != count
                ):
                    raise ValidationError(
                        "TaskRun requirement recovery projection is inconsistent"
                    )
                count += 1
                if count > self.config.task_runs.recovery_page_hard_limit:
                    raise ValidationError(
                        "TaskRun requirement recovery exceeds its hard cap"
                    )
                payload = self._store.get_task_run_payload(requirement.payload_id)
                expected_role = (
                    "goal"
                    if requirement.kind is TaskRunRequirementKind.INITIAL
                    else "follow_up"
                )
                decoded = self._decode_payload(payload, role=expected_role)
                if (
                    payload is None
                    or payload.run_id != record.run_id
                    or payload.role != expected_role
                    or payload.sha256 != requirement.requirement_sha256
                    or not isinstance(decoded.get("data_labels"), Mapping)
                ):
                    raise ValidationError(
                        "TaskRun requirement payload hash does not match"
                    )
                DataLabels.from_dict(decoded["data_labels"])
            if len(requirements) < page_size:
                if count != record.requirement_count:
                    raise ValidationError(
                        "TaskRun requirement recovery count is inconsistent"
                    )
                return
            last = requirements[-1]
            after = (last.ordinal, last.requirement_id)

    def _prevalidate_recoverable_resume_points(
        self,
        record: TaskRunRecord,
    ) -> None:
        hard_limit = self.config.task_runs.recovery_page_hard_limit
        points = self._store.list_task_run_resume_points(
            record.run_id,
            limit=hard_limit + 1,
        )
        if len(points) > hard_limit:
            raise ValidationError(
                "TaskRun resume-point recovery exceeds its hard cap"
        )
        for point in points:
            self._prevalidate_recoverable_resume_point(record, point)

    def _prevalidate_recoverable_resume_point(
        self,
        record: TaskRunRecord,
        point: TaskRunResumePoint,
    ) -> None:
        process = self._store.get_process(point.pid)
        transcript = self._store.get_task_run_payload(point.transcript_payload_id)
        decoded = self._decode_payload(transcript, role="transcript")
        if (
            process is None
            or not self._resume_point_identity_valid(
                point,
                record=record,
                process=process,
                require_current_runtime=False,
            )
            or transcript is None
            or transcript.run_id != record.run_id
            or transcript.role != "transcript"
            or not isinstance(decoded.get("transcript_messages"), list)
            or not self._resume_static_integrity_valid(point)
        ):
            raise ValidationError(
                "TaskRun resume bundle failed integrity validation"
            )
        for payload_id, role in (
            (point.summary_payload_id, "summary"),
            (point.pending_action_payload_id, "pending_action"),
        ):
            if payload_id is None:
                continue
            referenced = self._store.get_task_run_payload(payload_id)
            self._decode_payload(referenced, role=role)
            if (
                referenced is None
                or referenced.run_id != record.run_id
                or referenced.role != role
            ):
                raise ValidationError(
                    "TaskRun resume bundle payload role is invalid"
                )

    def recover_startup(self) -> tuple[TaskRunSummary, ...]:
        if self._recovered:
            return ()
        recovered: list[TaskRunSummary] = []
        recovered_total = 0
        cursor: TaskRunCursor | None = None
        while True:
            page = self._store.list_recoverable_task_runs(
                after=cursor,
                limit=self.config.task_runs.recovery_page_size,
            )
            for stale in page.records:
                recovered_total += 1
                claimed = self._store.claim_task_run_epoch(
                    stale.run_id,
                    stale.revision,
                    self._runtime_epoch,
                )
                projected = self._summary(self._recover_one(claimed))
                if len(recovered) < self.config.task_runs.recovery_sample_limit:
                    recovered.append(projected)
            cursor = page.next_cursor
            if cursor is None:
                break
        self._recovered = True
        self._recovered_total_count = recovered_total
        self._notify_updated()
        return tuple(recovered)

    def recovery_options(self, run_id: str) -> tuple[TaskRunRecoveryOption, ...]:
        record = self._require_run(run_id)
        if record.status is not TaskRunStatus.NEEDS_ATTENTION:
            return ()
        kinds = {str(item.get("kind")) for item in record.blockers}
        options: list[TaskRunRecoveryOption] = []
        if "unknown_effect" in kinds or "effect_unsettled" in kinds:
            unsafe_states = {"authorized", "approved", "dispatched", "unknown"}
            for effect in self._unsettled_effects(run_id):
                if effect.transaction_state not in unsafe_states:
                    continue
                provider = self._provider_for_effect(effect.provider)
                if not callable(
                    getattr(provider, "verify_external_effect_receipt", None)
                ):
                    continue
                binding = self._sha256(
                    {
                        "schema_version": 1,
                        "kind": "authoritative_effect_receipt",
                        "run_id": run_id,
                        "effect_id": effect.effect_id,
                        "expected_transaction_state": effect.transaction_state,
                        "runtime_epoch": record.runtime_epoch,
                    }
                )
                options.append(
                    TaskRunRecoveryOption(
                        option_id=f"effect_receipt:{binding}",
                        kind="effect_receipt",
                        label=(
                            "Record an authoritative provider receipt for "
                            f"effect {effect.effect_id}"
                        ),
                        requires_receipt=True,
                        effect_id=effect.effect_id,
                        expected_transaction_state=effect.transaction_state,
                        runtime_epoch=record.runtime_epoch,
                    )
                )
        if (
            kinds
            & {
            "binding_drift",
            "pending_action_unreplayable",
            "active_object_task",
            "authority_revoked",
            }
            and self._payloads_retained(run_id)
        ):
            options.append(
                TaskRunRecoveryOption(
                    option_id="create_linked_run",
                    kind="linked_rerun",
                    label="Create a separately fenced linked Run",
                )
            )
        options.append(
            TaskRunRecoveryOption(
                option_id="terminate_run",
                kind="terminalize",
                label="Stop old Run execution without settling effects",
            )
        )
        return tuple(options)

    def recover(
        self,
        run_id: str,
        *,
        option_id: str,
        receipt: Mapping[str, Any] | None = None,
        expected_revision: int,
        command_id: str,
    ) -> TaskRunSummary:
        request = {
            "expected_revision": expected_revision,
            "option_id": option_id,
            "receipt": dict(receipt or {}),
        }
        replay = self._recover_command_replay(
            run_id,
            command_id,
            request,
        )
        if replay is not None:
            return replay
        record = self._require_revision(run_id, expected_revision)
        options = {
            item.option_id: item for item in self.recovery_options(run_id)
        }
        selected_option = options.get(option_id)
        if selected_option is None:
            raise ValidationError(
                "TaskRun recovery option was not generated from server evidence"
            )
        if selected_option.kind == "effect_receipt":
            return self._recover_effect_receipt(
                record,
                selected_option,
                receipt,
                command_id=command_id,
                request=request,
            )
        if selected_option.kind == "linked_rerun":
            return self._recover_linked_run(
                record,
                command_id=command_id,
                request=request,
            )
        return self._recover_terminate_run(
            record,
            command_id=command_id,
            request=request,
        )

    def _recover_command_replay(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary | None:
        selected_id = self._identifier(command_id, "command_id")
        existing = self._store.get_task_run_command(run_id, selected_id)
        if existing is None:
            return self._recover_linked_command_gap_replay(
                run_id,
                selected_id,
                request,
            )
        self._require_same_command(
            existing,
            "recover",
            self._request_hash("recover", request),
        )
        variant, settlement_state = self._validated_recover_variant(
            existing,
            request,
        )
        if variant == "effect_receipt" and settlement_state == "pending":
            return self._settle_recover_effect_receipt_command(
                run_id,
                selected_id,
                request,
            )
        if variant == "terminalize" and settlement_state == "pending":
            return self._settle_recover_terminate_command(
                run_id,
                selected_id,
                request,
            )
        if variant == "linked":
            linked = self._linked_summary_from_command(existing)
            assert linked is not None
            return linked
        return self._summary_from_command(existing)

    def _validated_recover_variant(
        self,
        command: TaskRunCommand,
        request: Mapping[str, Any],
    ) -> tuple[str, str]:
        option_id = request.get("option_id")
        if option_id == "terminate_run":
            state, _cancel_generation = self._validated_terminalize_receipt(command)
            return "terminalize", state
        if option_id == "create_linked_run":
            self._validate_linked_command_result(command)
            return "linked", "complete"
        if type(option_id) is str and option_id.startswith("effect_receipt:"):
            (
                state,
                effect_id,
                expected_state,
                _cancel_generation,
                admission_runtime_epoch,
                _settlement_transition_seq,
                _settlement_audit_record_id,
            ) = self._validated_effect_receipt(command)
            self._require_effect_receipt_option_binding(
                command.run_id,
                request,
                effect_id=effect_id,
                expected_state=expected_state,
                admission_runtime_epoch=admission_runtime_epoch,
            )
            return "effect_receipt", state
        raise ValidationError("TaskRun recover command has an invalid option binding")

    def _recover_linked_command_gap_replay(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary | None:
        """Recover the one committed nested-rerun/absent-parent crash window.

        A linked recovery historically committed its deterministic nested
        rerun receipt before writing the parent ``recover`` receipt.  The
        nested command is sufficient local evidence only when its hidden
        parent binding, immutable source/target summaries, create receipt, and
        append-only link all match this exact outer request.  No model- or
        caller-supplied assertion can substitute for that chain.
        """

        nested_id = f"{command_id}:rerun"
        if len(nested_id) > 256:
            return None
        nested = self._store.get_task_run_command(run_id, nested_id)
        if nested is None:
            return None
        if (
            set(request) != {"expected_revision", "option_id", "receipt"}
            or request.get("option_id") != "create_linked_run"
            or not isinstance(request.get("receipt"), Mapping)
        ):
            raise TaskRunCommandConflict(
                "TaskRun idempotency key was reused with a different request"
            )
        expected_revision = request.get("expected_revision")
        if type(expected_revision) is not int or expected_revision < 0:
            raise TaskRunCommandConflict(
                "TaskRun idempotency key was reused with a different request"
            )
        recovery_parent = self._linked_recovery_parent(command_id, request)
        nested_request = {
            "expected_revision": expected_revision,
            "client_request_id": f"recover:{command_id}",
            "spec_overrides": {},
            "recovery_parent": recovery_parent,
        }
        self._require_same_command(
            nested,
            "rerun",
            self._request_hash("rerun", nested_request),
        )
        if nested.client_request_id is not None:
            raise ValidationError(
                "TaskRun linked recovery nested receipt has a client identity"
            )
        source_summary = self._summary_from_command(nested)
        if (
            source_summary.status is not TaskRunStatus.NEEDS_ATTENTION
            or source_summary.revision != expected_revision + 1
            or nested.result.get("run_id") != run_id
            or nested.result.get("revision") != source_summary.revision
            or set(nested.result)
            != {
                "schema_version",
                "summary",
                "run_id",
                "revision",
                "new_run_id",
                "new_run_summary",
            }
        ):
            raise ValidationError(
                "TaskRun linked recovery nested source receipt is invalid"
            )
        linked = self._linked_summary_from_command(nested)
        if linked is None:
            raise ValidationError(
                "TaskRun linked recovery nested target receipt is missing"
            )
        self._require_linked_recovery_target(
            source_run_id=run_id,
            nested_command_id=nested_id,
            client_request_id=f"recover:{command_id}",
            linked=linked,
        )
        candidate = TaskRunCommand(
            command_id=command_id,
            client_request_id=None,
            run_id=run_id,
            command_kind="recover",
            request_hash=self._request_hash("recover", request),
            result=dict(nested.result),
            result_revision=nested.result_revision,
            created_at=utc_now(),
        )
        with self._uow.transaction():
            current_nested = self._store.get_task_run_command(run_id, nested_id)
            if current_nested != nested:
                raise ValidationError(
                    "TaskRun linked recovery nested receipt changed"
                )
            completed = self._store.insert_task_run_command(
                candidate,
                expected_runtime_epoch=self._runtime_epoch,
            )
        self._require_same_command(
            completed,
            "recover",
            candidate.request_hash,
        )
        if (
            completed.result_revision != nested.result_revision
            or completed.result != nested.result
        ):
            raise ValidationError(
                "TaskRun linked recovery parent receipt conflicts with its target"
            )
        restored = self._linked_summary_from_command(completed)
        if restored != linked:
            raise ValidationError(
                "TaskRun linked recovery parent target changed"
            )
        return restored

    def _require_linked_recovery_target(
        self,
        *,
        source_run_id: str,
        nested_command_id: str,
        client_request_id: str,
        linked: TaskRunSummary,
    ) -> None:
        target = self._store.get_task_run(linked.run_id)
        create_command = self._store.get_task_run_command_by_client_request_id(
            client_request_id
        )
        if create_command is not None:
            self._validate_command_result_for_kind(create_command)
        if (
            target is None
            or linked.run_id == source_run_id
            or create_command is None
            or create_command.command_id != f"create:{client_request_id}"
            or create_command.client_request_id != client_request_id
            or create_command.run_id != linked.run_id
            or create_command.command_kind != "create"
            or self._summary_from_command(create_command) != linked
        ):
            raise ValidationError(
                "TaskRun linked recovery target create receipt is invalid"
            )
        hard = self.config.task_runs.recovery_page_hard_limit
        links = self._store.list_task_run_links(
            linked.run_id,
            limit=hard + 1,
        )
        if len(links) > hard:
            raise ValidationError(
                "TaskRun linked recovery target links exceed the recovery bound"
            )
        matches = [
            link
            for link in links
            if link.evidence_type == "task_run"
            and link.evidence_id == source_run_id
            and link.role == "rerun_of"
            and link.metadata == {"command_id": nested_command_id}
        ]
        if len(matches) != 1:
            raise ValidationError(
                "TaskRun linked recovery target link is invalid"
            )

    def _linked_recovery_parent(
        self,
        command_id: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "command_id": self._identifier(command_id, "command_id"),
            "request_hash": self._request_hash("recover", request),
        }

    def _recover_effect_receipt(
        self,
        record: TaskRunRecord,
        option: TaskRunRecoveryOption,
        receipt: Mapping[str, Any] | None,
        *,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        if not isinstance(receipt, Mapping) or not receipt:
            raise ValidationError("authoritative effect receipt is required")
        if (
            option.effect_id is None
            or option.expected_transaction_state is None
            or option.runtime_epoch != record.runtime_epoch
        ):
            raise ValidationError(
                "authoritative receipt option lost its server evidence binding"
            )
        effect = self._uow.evidence.get_external_effect(option.effect_id)
        if effect is None:
            raise ValidationError("authoritative receipt effect no longer exists")
        provider = self._provider_for_effect(effect.provider)
        if provider is None:
            raise ValidationError(
                "configured provider does not support authoritative receipt settlement"
            )
        updated, transition_seq, audit_record_id = self._settle_effect_recovery_receipt(
            record,
            option,
            provider,
            receipt,
            command_id=command_id,
            request=request,
        )
        if updated.status is TaskRunStatus.CANCELLING:
            self._continue_cancel_after_effect_settlement(record.run_id)
            updated = self._project(
                self._require_run(record.run_id),
                allow_finalize=True,
            )
        summary = self._complete_command_summary(
            updated,
            command_id,
            "recover",
            request,
            extra=self._effect_receipt_command_result(
                option,
                cancel_generation=record.cancel_generation,
                settlement_state="complete",
                settlement_transition_seq=transition_seq,
                settlement_audit_record_id=audit_record_id,
            ),
        )
        self._notify_updated()
        return summary

    def _settle_effect_recovery_receipt(
        self,
        record: TaskRunRecord,
        option: TaskRunRecoveryOption,
        provider: Any,
        receipt: Mapping[str, Any],
        *,
        command_id: str,
        request: Mapping[str, Any],
    ) -> tuple[TaskRunRecord, int, str]:
        from agent_libos.evidence import (
            settle_external_effect_from_authoritative_receipt,
        )

        assert option.effect_id is not None
        assert option.expected_transaction_state is not None
        assert option.runtime_epoch is not None
        # This is a live Host/admin recovery command, not startup recovery.
        # Hold an ordinary mutation admission across receipt verification and
        # the single fenced settlement transaction so shutdown/recovery fencing
        # can revoke it before commit.
        with self._host.lifecycle.admit():
            with self._uow.transaction():
                self._record_command(
                    record,
                    command_id,
                    "recover",
                    request,
                    result=self._effect_receipt_command_result(
                        option,
                        cancel_generation=record.cancel_generation,
                        settlement_state="pending",
                    ),
                )
                settlement = settle_external_effect_from_authoritative_receipt(
                    self._uow.protected_effects,
                    provider=provider,
                    run_id=record.run_id,
                    effect_id=option.effect_id,
                    expected_transaction_state=option.expected_transaction_state,
                    provider_receipt=dict(receipt),
                    runtime_epoch=record.runtime_epoch,
                    require_recovery_lease=(
                        self._host.lifecycle.revalidate_current_admission_if_present
                    ),
                )
                action_rewound: bool | None = None
                if (
                    settlement.provider_state == "not_started"
                    and record.cancel_generation == 0
                ):
                    action_rewound = (
                        self._rewind_live_not_started_recovery_action(
                            settlement,
                        )
                    )
                updated = self._apply_effect_recovery_settlement(
                    self._require_run(record.run_id),
                    settlement,
                    not_started_action_rewound=action_rewound,
                )
                pending = self._store.get_task_run_command(
                    record.run_id,
                    command_id,
                )
                if pending is None:
                    raise TaskRunRevisionConflict(
                        "TaskRun effect receipt disappeared during settlement"
                    )
                self._store.update_task_run_command_result(
                    record.run_id,
                    command_id,
                    expected_result_revision=pending.result_revision,
                    result=self._command_result(
                        updated,
                        extra=self._effect_receipt_command_result(
                            option,
                            cancel_generation=record.cancel_generation,
                            settlement_state="pending",
                            settlement_transition_seq=settlement.transition_seq,
                            settlement_audit_record_id=settlement.audit_record_id,
                        ),
                    ),
                    result_revision=updated.revision,
                    expected_runtime_epoch=self._runtime_epoch,
                )
                return (
                    updated,
                    settlement.transition_seq,
                    settlement.audit_record_id,
                )

    def _validated_effect_receipt(
        self,
        command: TaskRunCommand,
    ) -> tuple[str, str, str, int, int, int, str]:
        self._summary_from_command(command)
        self._require_result_keys(
            command.result,
            _EFFECT_RECEIPT_RESULT_KEYS,
            "effect receipt",
        )
        state = self._settlement_state(command.result, "effect receipt")
        if command.result.get("settlement_kind") != "effect_receipt":
            raise ValidationError("TaskRun effect receipt has an invalid kind")
        effect_id = self._identifier(
            command.result.get("effect_id"),
            "effect receipt effect_id",
        )
        expected_state = command.result.get("expected_transaction_state")
        if expected_state not in {
            "prepared",
            "authorized",
            "approved",
            "dispatched",
            "unknown",
        } or type(expected_state) is not str:
            raise ValidationError("TaskRun effect receipt has an invalid fence")
        cancel_generation = self._bounded_receipt_integer(
            command.result.get("cancel_generation"),
            "effect receipt cancellation generation",
            minimum=0,
        )
        admission_runtime_epoch = self._bounded_receipt_integer(
            command.result.get("admission_runtime_epoch"),
            "effect receipt admission Runtime epoch",
            minimum=1,
        )
        settlement_transition_seq = self._bounded_receipt_integer(
            command.result.get("settlement_transition_seq"),
            "effect receipt settlement transition sequence",
            minimum=1,
        )
        settlement_audit_record_id = self._identifier(
            command.result.get("settlement_audit_record_id"),
            "effect receipt settlement audit record_id",
        )
        return (
            state,
            effect_id,
            expected_state,
            cancel_generation,
            admission_runtime_epoch,
            settlement_transition_seq,
            settlement_audit_record_id,
        )

    def _validated_terminalize_receipt(
        self,
        command: TaskRunCommand,
    ) -> tuple[str, int]:
        self._summary_from_command(command)
        self._require_result_keys(
            command.result,
            _TERMINALIZE_RESULT_KEYS,
            "terminalization receipt",
        )
        state = self._settlement_state(command.result, "terminalization receipt")
        if command.result.get("settlement_kind") != "terminalize":
            raise ValidationError("TaskRun termination receipt has an invalid kind")
        cancel_generation = self._bounded_receipt_integer(
            command.result.get("cancel_generation"),
            "terminalization cancellation generation",
            minimum=1,
        )
        self._validate_control_admission(
            command,
            evidence={
                "kind": "terminalize",
                "cancel_generation": cancel_generation,
            },
            label="manual recovery termination intent persisted",
        )
        return state, cancel_generation

    @staticmethod
    def _effect_receipt_command_result(
        option: TaskRunRecoveryOption,
        *,
        cancel_generation: int,
        settlement_state: str,
        settlement_transition_seq: int | None = None,
        settlement_audit_record_id: str | None = None,
    ) -> dict[str, Any]:
        assert option.effect_id is not None
        assert option.expected_transaction_state is not None
        assert option.runtime_epoch is not None
        result = {
            "settlement_state": settlement_state,
            "settlement_kind": "effect_receipt",
            "effect_id": option.effect_id,
            "expected_transaction_state": option.expected_transaction_state,
            "cancel_generation": cancel_generation,
            "admission_runtime_epoch": option.runtime_epoch,
        }
        if settlement_transition_seq is not None:
            assert settlement_audit_record_id is not None
            result["settlement_transition_seq"] = settlement_transition_seq
            result["settlement_audit_record_id"] = settlement_audit_record_id
        return result

    def _settle_recover_effect_receipt_command(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        """Finish local receipt settlement without consulting the provider again."""

        (
            _command,
            effect_id,
            expected_state,
            cancel_generation,
            admission_runtime_epoch,
            settlement_transition_seq,
            settlement_audit_record_id,
        ) = (
            self._pending_effect_receipt_command(run_id, command_id, request)
        )
        current = self._require_run(run_id)
        if (
            current.status not in TASK_RUN_TERMINAL_STATUSES
            and current.runtime_epoch != self._runtime_epoch
        ):
            raise TaskRunRevisionConflict(f"TaskRun epoch is stale: {run_id}")
        if current.cancel_generation < cancel_generation:
            raise ValidationError(
                "TaskRun effect receipt lost its cancellation generation fence"
            )
        authoritative = self._effect_receipt_settlement_is_authoritative(
            current,
            effect_id,
            expected_state,
            admission_runtime_epoch,
            settlement_transition_seq,
            settlement_audit_record_id,
        )
        if not authoritative:
            if current.status in {
                TaskRunStatus.FINALIZING,
                *TASK_RUN_TERMINAL_STATUSES,
            }:
                raise ValidationError(
                    "TaskRun effect receipt evidence is invalid for a terminal Run"
                )
            current = self._mark_attention(
                current,
                self._blocker(
                    "unknown_effect",
                    "authoritative external-effect settlement is incomplete",
                    effect_ids=[effect_id],
                ),
            )
            return self._summary(current)
        if current.status not in TASK_RUN_TERMINAL_STATUSES:
            if current.cancel_generation > 0:
                self._continue_cancel_after_effect_settlement(run_id)
                current = self._project(
                    self._require_run(run_id),
                    allow_finalize=True,
                )
        summary = self._complete_command_summary(
            current,
            command_id,
            "recover",
            request,
            extra={
                "settlement_state": "complete",
                "settlement_kind": "effect_receipt",
                "effect_id": effect_id,
                "expected_transaction_state": expected_state,
                "cancel_generation": cancel_generation,
                "admission_runtime_epoch": admission_runtime_epoch,
                "settlement_transition_seq": settlement_transition_seq,
                "settlement_audit_record_id": settlement_audit_record_id,
            },
        )
        self._notify_updated()
        return summary

    def _pending_effect_receipt_command(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> tuple[TaskRunCommand, str, str, int, int, int, str]:
        command = self._store.get_task_run_command(run_id, command_id)
        if command is None:
            raise TaskRunRevisionConflict("TaskRun effect receipt is missing")
        self._require_same_command(
            command,
            "recover",
            self._request_hash("recover", request),
        )
        (
            settlement_state,
            effect_id,
            expected_state,
            cancel_generation,
            admission_runtime_epoch,
            settlement_transition_seq,
            settlement_audit_record_id,
        ) = self._validated_effect_receipt(command)
        if settlement_state != "pending":
            raise ValidationError("TaskRun effect receipt is not pending")
        self._require_effect_receipt_option_binding(
            command.run_id,
            request,
            effect_id=effect_id,
            expected_state=expected_state,
            admission_runtime_epoch=admission_runtime_epoch,
        )
        return (
            command,
            effect_id,
            expected_state,
            cancel_generation,
            admission_runtime_epoch,
            settlement_transition_seq,
            settlement_audit_record_id,
        )

    def _require_effect_receipt_option_binding(
        self,
        run_id: str,
        request: Mapping[str, Any],
        *,
        effect_id: str,
        expected_state: str,
        admission_runtime_epoch: int,
    ) -> None:
        binding = self._sha256(
            {
                "schema_version": 1,
                "kind": "authoritative_effect_receipt",
                "run_id": run_id,
                "effect_id": effect_id,
                "expected_transaction_state": expected_state,
                "runtime_epoch": admission_runtime_epoch,
            }
        )
        if request.get("option_id") != f"effect_receipt:{binding}":
            raise ValidationError(
                "TaskRun effect receipt does not match its recovery option binding"
            )

    def _effect_receipt_settlement_is_authoritative(
        self,
        current: TaskRunRecord,
        effect_id: str,
        expected_state: str,
        admission_runtime_epoch: int,
        settlement_transition_seq: int,
        settlement_audit_record_id: str,
    ) -> bool:
        effect = self._authoritative_effect_receipt_effect(
            current,
            effect_id=effect_id,
            admission_runtime_epoch=admission_runtime_epoch,
            settlement_transition_seq=settlement_transition_seq,
        )
        return bool(
            effect is not None
            and self._effect_receipt_audit_is_authoritative(
                effect,
                expected_state=expected_state,
                settlement_transition_seq=settlement_transition_seq,
                settlement_audit_record_id=settlement_audit_record_id,
            )
        )

    def _authoritative_effect_receipt_effect(
        self,
        current: TaskRunRecord,
        *,
        effect_id: str,
        admission_runtime_epoch: int,
        settlement_transition_seq: int,
    ) -> Any | None:
        if admission_runtime_epoch > current.runtime_epoch:
            return None
        effect = self._uow.evidence.get_external_effect(effect_id)
        if effect is None:
            return None
        process = self._store.get_process(effect.pid)
        if not all(
            (
                process is not None,
                process is not None and process.task_run_id == current.run_id,
                process is not None
                and process.task_run_epoch == current.runtime_epoch,
                effect.effect_state == "finalized",
            )
        ):
            return None
        transition_matches = self._uow.evidence.external_effect_transition_matches(
            settlement_transition_seq,
            effect_id,
            effect_state="finalized",
            transaction_state=effect.transaction_state,
        )
        return effect if transition_matches else None

    def _effect_receipt_audit_is_authoritative(
        self,
        effect: Any,
        *,
        expected_state: str,
        settlement_transition_seq: int,
        settlement_audit_record_id: str,
    ) -> bool:
        audit = self._uow.evidence.get_audit(settlement_audit_record_id)
        if audit is None or not isinstance(audit.decision, Mapping):
            return False
        decision = audit.decision
        if not all(
            (
                audit.action == "external_effect.recovery_settled",
                audit.target == f"external_effect:{effect.effect_id}",
                audit.correlation_id == effect.effect_id,
                decision.get("source") == "host_verified_receipt",
                decision.get("previous_transaction_state") == expected_state,
                decision.get("settled_transaction_state")
                == effect.transaction_state,
                decision.get("transition_seq") == settlement_transition_seq,
            )
        ):
            return False
        provider_state = decision.get("provider_state")
        if provider_state not in {
            "committed",
            "failed",
            "compensated",
            "not_started",
        }:
            return False
        expected_settled_state = (
            "failed" if provider_state == "not_started" else provider_state
        )
        if effect.transaction_state != expected_settled_state:
            return False
        receipt_sha256 = decision.get("provider_receipt_sha256")
        if type(receipt_sha256) is not str or len(receipt_sha256) != 64:
            return False
        return all(
            character in "0123456789abcdef" for character in receipt_sha256
        )

    def _continue_cancel_after_effect_settlement(self, run_id: str) -> None:
        for process in reversed(self._tree_processes(run_id)):
            if process.status not in _TERMINAL_PROCESS_STATUSES:
                self._process.cancel(
                    process.pid,
                    "continuing TaskRun cancellation after effect settlement",
                )

    def _recover_linked_run(
        self,
        record: TaskRunRecord,
        *,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        recovery_parent = self._linked_recovery_parent(command_id, request)
        created = self._rerun(
            record.run_id,
            expected_revision=record.revision,
            command_id=f"{command_id}:rerun",
            client_request_id=f"recover:{command_id}",
            allow_needs_attention=True,
            recovery_parent=recovery_parent,
        )
        latest = self._require_run(record.run_id)
        self._record_command(
            latest,
            command_id,
            "recover",
            request,
            result={
                "run_id": record.run_id,
                "revision": latest.revision,
                "new_run_id": created.run_id,
                "new_run_summary": to_jsonable(created),
            },
        )
        return created

    def _recover_terminate_run(
        self,
        record: TaskRunRecord,
        *,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        # ``terminate_run`` is a Host cancellation decision for the old Run,
        # not evidence that an unresolved requirement was satisfied.  Persist
        # and generation-fence that decision before touching the process tree
        # so an already-terminal root can converge through the ordinary
        # cancellation finalizer instead of cycling back to needs_attention.
        with self._condition:
            with self._uow.transaction():
                terminating = self._store.update_task_run_cas(
                    record.run_id,
                    record.revision,
                    updates={
                        "status": TaskRunStatus.CANCELLING,
                        "cancel_generation": record.cancel_generation + 1,
                        "updated_at": utc_now(),
                    },
                    expected_runtime_epoch=self._runtime_epoch,
                )
                admission = self._append_control_admission(
                    record,
                    terminating,
                    command_id=command_id,
                    command_kind="recover",
                    request=request,
                    evidence={
                        "kind": "terminalize",
                        "cancel_generation": terminating.cancel_generation,
                    },
                    label="manual recovery termination intent persisted",
                )
                self._record_command(
                    terminating,
                    command_id,
                    "recover",
                    request,
                    result={
                        "settlement_state": "pending",
                        "settlement_kind": "terminalize",
                        "cancel_generation": terminating.cancel_generation,
                        **admission,
                    },
                )
            self._condition.notify_all()
        return self._settle_recover_terminate_command(
            record.run_id,
            command_id,
            request,
        )

    def _settle_recover_terminate_command(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> TaskRunSummary:
        """Finish a generation-fenced manual termination without redispatch."""

        receipt = self._pending_recovery_termination_receipt(
            run_id,
            command_id,
            request,
        )
        if isinstance(receipt, TaskRunSummary):
            return receipt
        existing, cancel_generation = receipt
        self._require_current_recovery_termination_run(
            run_id,
            cancel_generation,
        )
        if self._recovery_termination_has_active_dispatch(run_id):
            return self._summary_from_command(existing)
        self._stop_recovery_process_tree(run_id)
        updated = self._project_stopped_recovery_termination(run_id)
        return self._complete_recovery_termination(
            updated,
            command_id,
            request,
            cancel_generation=cancel_generation,
        )

    def _pending_recovery_termination_receipt(
        self,
        run_id: str,
        command_id: str,
        request: Mapping[str, Any],
    ) -> tuple[TaskRunCommand, int] | TaskRunSummary:
        existing = self._store.get_task_run_command(run_id, command_id)
        if existing is None:
            raise TaskRunRevisionConflict("TaskRun termination receipt is missing")
        self._require_same_command(
            existing,
            "recover",
            self._request_hash("recover", request),
        )
        variant, settlement_state = self._validated_recover_variant(
            existing,
            request,
        )
        if variant != "terminalize":
            raise ValidationError("TaskRun termination receipt changed variant")
        if settlement_state != "pending":
            return self._summary_from_command(existing)
        _state, cancel_generation = self._validated_terminalize_receipt(existing)
        return existing, cancel_generation

    def _require_current_recovery_termination_run(
        self,
        run_id: str,
        cancel_generation: int,
    ) -> TaskRunRecord:
        current = self._require_run(run_id)
        if current.cancel_generation < cancel_generation:
            raise ValidationError("TaskRun termination receipt lost its generation fence")
        if (
            current.status not in TASK_RUN_TERMINAL_STATUSES
            and current.runtime_epoch != self._runtime_epoch
        ):
            raise TaskRunRevisionConflict(f"TaskRun epoch is stale: {run_id}")
        return current

    def _recovery_termination_has_active_dispatch(self, run_id: str) -> bool:
        return bool(
            self._active_run_dispatches.get(run_id, 0) > 0
            or self._has_active_external_dispatch(run_id)
            or self._has_active_quantum(run_id)
        )

    def _stop_recovery_process_tree(self, run_id: str) -> None:
        for process in reversed(self._tree_processes(run_id)):
            if process.status not in _TERMINAL_PROCESS_STATUSES:
                try:
                    self._process.cancel(
                        process.pid,
                        "TaskRun terminated during manual recovery",
                    )
                except Exception:
                    pass

    def _project_stopped_recovery_termination(
        self,
        run_id: str,
    ) -> TaskRunRecord:
        latest = self._require_run(run_id)
        unsettled = self._unsettled_effects(run_id)
        if unsettled:
            kind = (
                "unknown_effect"
                if any(
                    effect.transaction_state in _UNKNOWN_EFFECT_STATES
                    for effect in unsettled
                )
                else "effect_unsettled"
            )
            updated = self._mark_attention(
                latest,
                self._blocker(
                    kind,
                    "old Run execution stopped; external effect still requires settlement",
                    effect_ids=[effect.effect_id for effect in unsettled[:20]],
                ),
            )
            return updated
        live = [
            item
            for item in self._tree_processes(run_id)
            if item.status not in _TERMINAL_PROCESS_STATUSES
        ]
        if live:
            updated = self._mark_attention(
                latest,
                self._blocker(
                    "manual_recovery_required",
                    "old Run processes did not terminate cleanly",
                    pid=live[0].pid,
                ),
            )
        else:
            root = self._store.get_process(latest.root_pid) if latest.root_pid else None
            if root is None:
                raise ValidationError("TaskRun root is missing during termination")
            # Re-enter the ordinary projection path so an already-admitted
            # scheduler quantum or external call retains its local-settlement
            # lease.  Manual recovery must not bypass the same terminal/purge
            # barrier used by normal cancellation.
            updated = self._project(latest, allow_finalize=True)
        return updated

    def _complete_recovery_termination(
        self,
        record: TaskRunRecord,
        command_id: str,
        request: Mapping[str, Any],
        *,
        cancel_generation: int,
    ) -> TaskRunSummary:
        summary = self._complete_command_summary(
            record,
            command_id,
            "recover",
            request,
            extra={
                "settlement_state": "complete",
                "settlement_kind": "terminalize",
                "cancel_generation": cancel_generation,
            },
        )
        self._notify_updated()
        return summary

    def rerun(
        self,
        source_run_id: str,
        *,
        expected_revision: int,
        command_id: str,
        client_request_id: str | None = None,
        spec_overrides: Mapping[str, Any] | None = None,
    ) -> TaskRunSummary:
        return self._rerun(
            source_run_id,
            expected_revision=expected_revision,
            command_id=command_id,
            client_request_id=client_request_id,
            spec_overrides=spec_overrides,
            allow_needs_attention=False,
        )

    def _rerun(
        self,
        source_run_id: str,
        *,
        expected_revision: int,
        command_id: str,
        client_request_id: str | None = None,
        spec_overrides: Mapping[str, Any] | None = None,
        allow_needs_attention: bool,
        recovery_parent: Mapping[str, Any] | None = None,
    ) -> TaskRunSummary:
        selected_request_id = (
            self._identifier(client_request_id, "client_request_id")
            if client_request_id is not None
            else f"rerun:{source_run_id}:{self._identifier(command_id, 'command_id')}"
        )
        overrides = dict(spec_overrides or {})
        if "goal" not in overrides and "objective" in overrides:
            overrides["goal"] = overrides.pop("objective")
        request = {
            "expected_revision": expected_revision,
            "client_request_id": selected_request_id,
            "spec_overrides": overrides,
        }
        request.update(self._recovery_parent_request(recovery_parent))
        replay = self._command_replay(source_run_id, command_id, "rerun", request)
        if replay is not None:
            result = self._store.get_task_run_command(source_run_id, command_id)
            target = result.result.get("new_run_id") if result is not None else None
            target_summary = (
                result.result.get("new_run_summary")
                if result is not None
                else None
            )
            if target is not None and isinstance(target_summary, Mapping):
                try:
                    summary = TaskRunSummary(**dict(target_summary))
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        "TaskRun rerun command contains an invalid stored result"
                    ) from exc
                if summary.run_id != str(target):
                    raise ValidationError(
                        "TaskRun rerun command target result is not identity-bound"
                    )
                return summary
            return self.get(str(target)) if target else replay
        with self._uow.transaction(include_object_payloads=True):
            source = self._require_expected_revision(
                source_run_id,
                expected_revision,
            )
            if allow_needs_attention:
                if source.status is not TaskRunStatus.NEEDS_ATTENTION:
                    raise ValidationError(
                        "linked recovery rerun requires a needs-attention source"
                    )
            elif source.status not in TASK_RUN_TERMINAL_STATUSES:
                raise ValidationError(
                    "direct rerun requires a terminal source TaskRun"
                )
            if source.runtime_epoch != self._runtime_epoch:
                if source.status not in TASK_RUN_TERMINAL_STATUSES:
                    raise TaskRunRevisionConflict(
                        f"TaskRun epoch is stale: {source_run_id}"
                    )
                source = self._store.claim_terminal_task_run_epoch(
                    source.run_id,
                    source.revision,
                    self._runtime_epoch,
                )
            else:
                # The source has no other current-row mutation in a rerun.
                # Advance it under CAS so a concurrent purge/follow-up cannot
                # race the goal read and resurrect content from another
                # revision.
                source = self._store.update_task_run_cas(
                    source.run_id,
                    source.revision,
                    updates={"updated_at": utc_now()},
                    expected_runtime_epoch=self._runtime_epoch,
                )
            goal: Any
            if "goal" in overrides:
                goal = overrides["goal"]
            else:
                goal_payload = self._payload_by_role(source_run_id, "goal")
                decoded = self._decode_payload(goal_payload, role="goal")
                goal = decoded["goal"]
            base: dict[str, Any] = {
                "schema_version": 1,
                "goal": goal,
                "display_title": source.display_title,
                "image_id": source.image_id,
                "launch_options": source.launch_options,
                "authority_manifest_id": source.authority_manifest_id,
                "deadline_at": None,
                "retention": source.retention.value,
            }
            forbidden = sorted(set(overrides) - set(base))
            if forbidden:
                raise ValidationError(
                    f"unknown TaskRun rerun overrides: {forbidden}"
                )
            base.update(overrides)
            created = self.create(
                TaskRunSpecV1.from_mapping(base),
                client_request_id=selected_request_id,
                auto_run=False,
            )
            command = self._record_command(
                source,
                command_id,
                "rerun",
                request,
                result={
                    "run_id": source_run_id,
                    "revision": source.revision,
                    "new_run_id": created.run_id,
                    "new_run_summary": to_jsonable(created),
                },
            )
            ledger = self._append_ledger(
                created.run_id,
                kind=TaskRunLedgerKind.STATUS_TRANSITION,
                status=created.status.value,
                label="linked rerun created",
                metadata={
                    "source_run_id": source_run_id,
                    "command_id": command.command_id,
                },
            )
            self._store.insert_task_run_link(
                TaskRunLink(
                    link_id=new_id("trlink"),
                    run_id=created.run_id,
                    ledger_seq=ledger.seq,
                    evidence_type="task_run",
                    evidence_id=source_run_id,
                    role="rerun_of",
                    created_at=utc_now(),
                    metadata={"command_id": command.command_id},
                )
            )
        return created

    def _validated_recovery_parent(
        self,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        selected = dict(value)
        if (
            set(selected) != {"schema_version", "command_id", "request_hash"}
            or selected.get("schema_version") != 1
            or type(selected.get("schema_version")) is not int
            or not _is_lower_sha256(selected.get("request_hash"))
        ):
            raise ValidationError(
                "TaskRun linked recovery parent binding is invalid"
            )
        self._identifier(
            selected.get("command_id"),
            "linked recovery parent command_id",
        )
        return selected

    def _recovery_parent_request(
        self,
        value: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        return {"recovery_parent": self._validated_recovery_parent(value)}

    # ------------------------------------------------------------------
    # Internal state projection

    def _project_paused_terminal_root(
        self,
        record: TaskRunRecord,
    ) -> TaskRunRecord:
        """Persist the control projection before advertising PAUSED actions.

        ``allowed_actions`` is part of the revision-bound public summary and
        therefore cannot consult a Process row that may change independently.
        A normal paused tree remains untouched.  A missing or terminal root is
        projected once through the ordinary convergence/finalization path, so
        later reads at that revision are stable and never advertise Resume or
        Follow-up to a process that cannot receive either operation.
        """

        if record.status is not TaskRunStatus.PAUSED:
            return record
        root = self._store.get_process(record.root_pid) if record.root_pid else None
        if root is not None and root.status not in _TERMINAL_PROCESS_STATUSES:
            return record
        return self._project(record, allow_finalize=True)

    def _recover_one(self, record: TaskRunRecord) -> TaskRunRecord:
        prevalidated = self._prevalidated_blockers.get(record.run_id)
        if prevalidated is not None:
            return self._mark_attention(record, prevalidated)
        blocker = self._recovery_run_binding_blocker(record)
        if blocker is not None:
            return self._mark_attention(record, blocker)
        processes = self._tree_processes(record.run_id)
        local_blockers: list[dict[str, Any]] = []
        for process in processes:
            pending_blocker = self._recover_task_run_resume_state(process)
            if pending_blocker is not None:
                local_blockers.append(dict(pending_blocker))
        record = self._require_run(record.run_id)
        processes = self._tree_processes(record.run_id)
        blocker = self._recovery_evidence_blocker(record, processes)
        if blocker is not None:
            return self._mark_attention(record, blocker)
        # Completed provider results and durable waits above settle before any
        # stop/interrupt projection. Persisted cancellation/deadline intent
        # then outranks recovery of an older interrupt receipt; corrupt command
        # history cannot prevent an already durable stop from converging.
        if (
            record.status
            not in {
                TaskRunStatus.CANCELLING,
                TaskRunStatus.FINALIZING,
                TaskRunStatus.NEEDS_ATTENTION,
            }
            and self._deadline_expired(record)
        ):
            return self._expire(record)
        if record.status is TaskRunStatus.CANCELLING:
            return self._recover_cancelling_run(record, processes)
        try:
            pending_interrupt = self._pending_startup_interrupt(record)
        except (TaskRunRevisionConflict, TypeError, ValueError, ValidationError) as exc:
            return self._mark_attention(
                record,
                self._blocker(
                    "manual_recovery_required",
                    "persisted interrupt receipt failed integrity validation",
                    error_type=type(exc).__name__,
                ),
            )
        for pending_blocker in local_blockers:
            if (
                pending_interrupt is not None
                and self._interrupt_can_supersede_local_blocker(
                    record,
                    pending_blocker,
                )
            ):
                continue
            attention = self._mark_attention(record, pending_blocker)
            if pending_interrupt is None:
                return attention
            return self._complete_startup_interrupt_command(
                attention,
                pending_interrupt,
            )
        if pending_interrupt is not None:
            try:
                return self._settle_startup_interrupt(
                    record,
                    pending_interrupt,
                )
            except (
                TaskRunRevisionConflict,
                TypeError,
                ValueError,
                ValidationError,
            ) as exc:
                return self._mark_attention(
                    self._require_run(record.run_id),
                    self._blocker(
                        "manual_recovery_required",
                        "persisted interrupt settlement did not converge",
                        error_type=type(exc).__name__,
                    ),
                )
        if record.status is TaskRunStatus.PAUSED:
            return self._recover_paused_run(record, processes)
        blocker = self._recovery_resume_blocker(record, processes)
        if blocker is not None:
            return self._mark_attention(record, blocker)
        if record.status is TaskRunStatus.QUEUED:
            # Validation and local settlement above are allowed on reopen;
            # execution is not.  Preserve the explicit-dispatch boundary.
            return self._require_run(record.run_id)
        return self._project(record, allow_finalize=True)

    def _interrupt_can_supersede_local_blocker(
        self,
        record: TaskRunRecord,
        blocker: Mapping[str, Any],
    ) -> bool:
        if blocker.get("kind") != "pending_action_unreplayable":
            return False
        pid = blocker.get("pid")
        if not isinstance(pid, str):
            return False
        process = self._store.get_process(pid)
        point = self._store.get_task_run_resume_point(pid, complete_only=True)
        if (
            process is None
            or process.task_run_id != record.run_id
            or process.task_run_epoch != self._runtime_epoch
            or point is None
            or point.pending_action_payload_id is None
            or not self._resume_integrity_valid(point)
        ):
            return False
        try:
            wrapper = self._decode_pending_resume_payload(point)
        except (KeyError, NotFound, TypeError, ValueError, ValidationError):
            return False
        return bool(
            wrapper.get("kind") == "validated_action"
            and wrapper.get("state") == "dispatching"
            and not self._changed_effects_for_pid(pid, point.last_effect_seq)
        )

    def _recovery_run_binding_blocker(
        self,
        record: TaskRunRecord,
    ) -> dict[str, Any] | None:
        try:
            goal = self._payload_by_role(record.run_id, "goal")
            self._decode_payload(goal, role="goal")
        except (NotFound, ValidationError, ValueError):
            return self._blocker(
                "payload_missing",
                "durable goal payload is unavailable",
            )
        current = self._binding_hash(
            image_id=record.image_id,
            launch_options=record.launch_options,
            authority_manifest_id=record.authority_manifest_id,
        )
        if record.binding_hash != current:
            return self._blocker(
                "binding_drift",
                "Image/tool/provider binding changed",
            )
        return None

    def _recovery_evidence_blocker(
        self,
        record: TaskRunRecord,
        processes: list[Any],
    ) -> dict[str, Any] | None:
        for process in processes:
            blocker = self._authority_recovery_blocker(process)
            if blocker is not None:
                return blocker
        try:
            changed = (
                self._effects_changed_after_resume(processes)
                if record.cancel_generation == 0
                and record.status is not TaskRunStatus.CANCELLING
                else []
            )
        except ValidationError:
            return self._blocker(
                "payload_corrupt",
                "resume point effect baseline failed integrity validation",
            )
        if changed:
            return self._blocker(
                "unknown_effect",
                "external effect changed after the latest complete safe point",
                effect_ids=[item.effect_id for item in changed[:20]],
            )
        if self._unsafe_effects(record.run_id):
            return self._blocker(
                "unknown_effect",
                "external effect outcome is unknown",
            )
        abandoned = self._abandoned_object_tasks(record.run_id)
        if abandoned:
            return self._blocker(
                "active_object_task",
                "ObjectTask was abandoned during Runtime reopen",
                object_task_ids=abandoned[:20],
            )
        return None

    def _recover_cancelling_run(
        self,
        record: TaskRunRecord,
        processes: list[Any],
    ) -> TaskRunRecord:
        for process in reversed(processes):
            if process.status not in _TERMINAL_PROCESS_STATUSES:
                self._process.cancel(
                    process.pid,
                    "resuming persisted TaskRun cancellation",
                )
        return self._project(
            self._require_run(record.run_id),
            allow_finalize=True,
        )

    def _recover_paused_run(
        self,
        record: TaskRunRecord,
        processes: list[Any],
    ) -> TaskRunRecord:
        try:
            for process in processes:
                if process.status in {
                    ProcessStatus.RUNNABLE,
                    ProcessStatus.RUNNING,
                }:
                    self._process.pause_for_host_resume(
                        process.pid,
                        "resuming persisted TaskRun pause",
                    )
        except Exception as exc:
            return self._mark_attention(
                record,
                self._blocker(
                    "manual_recovery_required",
                    "persisted pause intent did not converge",
                    error_type=type(exc).__name__,
                ),
            )
        return self._project(
            self._require_run(record.run_id),
            allow_finalize=True,
        )

    def _pending_startup_interrupt(
        self,
        record: TaskRunRecord,
    ) -> tuple[
        TaskRunCommand,
        int,
        int,
        TaskRunStatus,
        int,
        tuple[tuple[str, int, int], ...],
    ] | None:
        hard_limit = self.config.task_runs.recovery_page_hard_limit
        commands = self._store.list_task_run_commands(
            record.run_id,
            limit=hard_limit + 1,
        )
        if len(commands) > hard_limit:
            raise ValidationError(
                "TaskRun command recovery exceeds its hard cap"
            )
        matches: list[
            tuple[
                TaskRunCommand,
                int,
                int,
                TaskRunStatus,
                int,
                tuple[tuple[str, int, int], ...],
            ]
        ] = []
        for command in commands:
            if command.command_kind != "follow_up":
                continue
            if not (set(command.result) & _INTERRUPT_RECEIPT_FIELDS):
                continue
            parsed = self._interrupt_follow_up_receipt_values(command)
            if parsed is None:
                continue
            (
                pause_generation,
                cancel_generation,
                prior_status,
                admission_runtime_epoch,
                resume_fences,
            ) = parsed
            if (
                pause_generation > record.pause_generation
                or cancel_generation > record.cancel_generation
                or admission_runtime_epoch > record.runtime_epoch
            ):
                raise ValidationError(
                    "TaskRun interrupt receipt is ahead of its generation fence"
                )
            if (
                pause_generation == record.pause_generation
                and cancel_generation == record.cancel_generation
            ):
                matches.append(
                    (
                        command,
                        pause_generation,
                        cancel_generation,
                        prior_status,
                        admission_runtime_epoch,
                        resume_fences,
                    )
                )
        if len(matches) > 1:
            raise ValidationError(
                "TaskRun has multiple pending interrupts for one generation"
            )
        return matches[0] if matches else None

    def _settle_startup_interrupt(
        self,
        record: TaskRunRecord,
        pending: tuple[
            TaskRunCommand,
            int,
            int,
            TaskRunStatus,
            int,
            tuple[tuple[str, int, int], ...],
        ],
    ) -> TaskRunRecord:
        (
            _command,
            pause_generation,
            cancel_generation,
            prior_status,
            admission_runtime_epoch,
            resume_fences,
        ) = pending
        current = self._finish_interrupt_follow_up_state(
            record.run_id,
            pause_generation=pause_generation,
            cancel_generation=cancel_generation,
            prior_status=prior_status,
            admission_runtime_epoch=admission_runtime_epoch,
            resume_fences=resume_fences,
        )
        current = self._converge_after_startup_interrupt(current)
        return self._complete_startup_interrupt_command(current, pending)

    def _converge_after_startup_interrupt(
        self,
        current: TaskRunRecord,
    ) -> TaskRunRecord:
        """Apply the ordinary safe-resume/finalization gates before receipt CAS."""

        if current.status in TASK_RUN_TERMINAL_STATUSES:
            return current
        if current.status is TaskRunStatus.QUEUED:
            # A queued Run has never owned an explicit scheduler admission.
            return current
        processes = self._tree_processes(current.run_id)
        if current.status is TaskRunStatus.PAUSED:
            return self._recover_paused_run(current, processes)
        if current.status is TaskRunStatus.NEEDS_ATTENTION:
            return current
        blocker = self._recovery_resume_blocker(current, processes)
        if blocker is not None:
            return self._mark_attention(current, blocker)
        return self._project(current, allow_finalize=True)

    def _complete_startup_interrupt_command(
        self,
        current: TaskRunRecord,
        pending: tuple[
            TaskRunCommand,
            int,
            int,
            TaskRunStatus,
            int,
            tuple[tuple[str, int, int], ...],
        ],
    ) -> TaskRunRecord:
        (
            command,
            pause_generation,
            cancel_generation,
            prior_status,
            _admission_runtime_epoch,
            _resume_fences,
        ) = pending
        extra = self._interrupt_follow_up_result(
            settlement_state="complete",
            pause_generation=pause_generation,
            cancel_generation=cancel_generation,
            prior_status=prior_status,
        )
        for field in _CONTROL_COMPLETION_PRESERVED_FIELDS:
            if field in command.result:
                extra[field] = command.result[field]
        result = self._command_result(current, extra=extra)
        self._store.update_task_run_command_result(
            current.run_id,
            command.command_id,
            expected_result_revision=command.result_revision,
            result=result,
            result_revision=current.revision,
            expected_runtime_epoch=self._runtime_epoch,
        )
        return current

    def _recovery_resume_blocker(
        self,
        record: TaskRunRecord,
        processes: list[Any],
    ) -> dict[str, Any] | None:
        for process in processes:
            blocker = self._process_resume_blocker(record, process)
            if blocker is not None:
                return blocker
        return None

    def _process_resume_blocker(
        self,
        record: TaskRunRecord,
        process: Any,
    ) -> dict[str, Any] | None:
        if process.status in _TERMINAL_PROCESS_STATUSES:
            return None
        point = self._store.get_task_run_resume_point(
            process.pid,
            complete_only=True,
        )
        pending = self._pending_action_recovery_blocker(process, point)
        if pending is not None or record.status is TaskRunStatus.QUEUED:
            return pending
        if process.status in {
            ProcessStatus.WAITING_EVENT,
            ProcessStatus.WAITING_HUMAN,
            ProcessStatus.WAITING_TOOL,
            ProcessStatus.PAUSED,
        }:
            return None
        if point is None:
            return self._blocker(
                "pending_action_unreplayable",
                "process has no complete local resume point",
                pid=process.pid,
            )
        if not self._resume_point_identity_valid(
            point,
            record=record,
            process=process,
        ) or not self._resume_static_integrity_valid(point):
            return self._blocker(
                "payload_corrupt",
                "resume point failed its integrity binding",
                pid=process.pid,
            )
        if not self._resume_current_binding_valid(
            point,
            record=record,
            process=process,
        ):
            return self._blocker(
                "binding_drift",
                "resume point Image/tool/provider binding changed",
                pid=process.pid,
            )
        return None

    def _project(
        self,
        record: TaskRunRecord,
        *,
        allow_finalize: bool,
    ) -> TaskRunRecord:
        if record.status in TASK_RUN_TERMINAL_STATUSES:
            return record
        if (
            record.status
            not in {
                TaskRunStatus.CANCELLING,
                TaskRunStatus.FINALIZING,
                TaskRunStatus.NEEDS_ATTENTION,
            }
            and self._deadline_expired(record)
        ):
            return self._expire(record)
        processes = self._tree_processes(record.run_id)
        blocker = self._projection_evidence_blocker(record.run_id)
        if blocker is not None:
            return self._mark_attention(record, blocker)
        blocker = self._projection_effect_blocker(record, processes)
        if blocker is not None:
            return self._mark_attention(record, blocker)
        return self._project_process_state(
            record,
            processes,
            allow_finalize=allow_finalize,
        )

    def _projection_evidence_blocker(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:
        try:
            self._project_evidence(run_id)
        except ValidationError as exc:
            return self._blocker(
                "manual_recovery_required",
                "TaskRun evidence projection exceeded its recovery bound",
                error_type=type(exc).__name__,
            )
        return None

    def _projection_effect_blocker(
        self,
        record: TaskRunRecord,
        processes: list[Any],
    ) -> dict[str, Any] | None:
        try:
            changed_effects = (
                self._effects_changed_after_resume(processes)
                if record.cancel_generation == 0
                and record.status is not TaskRunStatus.CANCELLING
                else []
            )
        except ValidationError:
            return self._blocker(
                "payload_corrupt",
                "resume point effect baseline failed integrity validation",
            )
        if changed_effects:
            return self._blocker(
                "unknown_effect",
                "external effect is newer than the latest complete safe point",
                effect_ids=[item.effect_id for item in changed_effects[:20]],
            )
        unsettled = self._unsettled_effects(record.run_id)
        if unsettled:
            kind = (
                "unknown_effect"
                if any(
                    effect.transaction_state in _UNKNOWN_EFFECT_STATES
                    for effect in unsettled
                )
                else "effect_unsettled"
            )
            return self._blocker(
                kind,
                "external effect outcome is unresolved",
                effect_ids=[effect.effect_id for effect in unsettled[:20]],
            )
        return None

    def _project_process_state(
        self,
        record: TaskRunRecord,
        processes: list[Any],
        *,
        allow_finalize: bool,
    ) -> TaskRunRecord:
        if not processes:
            return self._mark_attention(
                record,
                self._blocker("manual_recovery_required", "Run process tree is missing"),
            )
        live = [item for item in processes if item.status not in _TERMINAL_PROCESS_STATUSES]
        root = next((item for item in processes if item.pid == record.root_pid), None)
        if root is None:
            return self._mark_attention(
                record,
                self._blocker("manual_recovery_required", "Run root process is missing"),
            )
        if not live:
            if self._has_active_external_dispatch(
                record.run_id
            ) or self._has_active_quantum(record.run_id):
                # Process terminal state may be published before the worker's
                # external-dispatch scope or scheduler future has released.
                # Preserve control intent until the already-admitted call has
                # durably landed its paired result and cleanup evidence.
                return record
            if not allow_finalize:
                return record
            return self._finalize_terminal(record, root)
        status = self._status_from_processes(record, live)
        if status is record.status:
            return record
        with self._uow.transaction():
            updated = self._store.update_task_run_cas(
                record.run_id,
                record.revision,
                updates={
                    "status": status,
                    "active_pid": self._active_pid(live),
                    "updated_at": utc_now(),
                },
                expected_runtime_epoch=self._runtime_epoch,
            )
            self._status_ledger(
                record,
                updated,
                label="process state projected",
            )
        return updated

    def _finalize_terminal(
        self,
        record: TaskRunRecord,
        root: Any,
    ) -> TaskRunRecord:
        convergence = self._terminal_convergence_blocker(record)
        if convergence is not None:
            return self._mark_attention(record, convergence)
        try:
            requirements = self._bounded_completion_requirements(record)
        except ValidationError:
            return self._mark_attention(
                record,
                self._blocker(
                    "payload_corrupt",
                    "terminal requirement projection failed integrity validation",
                ),
            )
        root_succeeded = root.status is ProcessStatus.EXITED
        cancellation_intent = (
            record.status is TaskRunStatus.CANCELLING
            or record.cancel_generation > 0
        )
        terminal_blockers = self._terminal_blockers_after_convergence(
            record,
            cancellation_intent=cancellation_intent,
        )
        # Requirement satisfaction is committed with an integrity-bound root
        # ``process_exit`` transcript.  A terminal Process row is not completion
        # evidence: Host lifecycle calls, provider failures, and stale workers
        # must all leave the contract unsatisfied.
        satisfied = sum(
            item.status
            in {TaskRunRequirementStatus.SATISFIED, TaskRunRequirementStatus.WAIVED}
            for item in requirements
        )
        if record.satisfied_requirement_count != satisfied:
            record = self._store.update_task_run_cas(
                record.run_id,
                record.revision,
                updates={
                    "satisfied_requirement_count": satisfied,
                    "updated_at": utc_now(),
                },
                expected_runtime_epoch=self._runtime_epoch,
            )
        if (
            not cancellation_intent
            and root_succeeded
            and satisfied != len(requirements)
        ):
            return self._mark_attention(
                record,
                self._blocker(
                    "requirements_unsatisfied",
                    "root exited before every required requirement was satisfied",
                ),
            )
        if cancellation_intent:
            # A tool admitted before control intent may still commit a real root
            # exit while it owns local settlement.  The exit is evidence, but it
            # cannot overtake the already-persisted cancellation generation and
            # convert the Run back into successful completion.
            terminal = TaskRunStatus.CANCELLED
        elif root.status is ProcessStatus.KILLED:
            terminal = TaskRunStatus.FAILED
        elif root_succeeded:
            terminal = TaskRunStatus.SUCCEEDED
        else:
            terminal = TaskRunStatus.FAILED
        outcome = getattr(root, "outcome", None)
        result_ref = getattr(outcome, "result_oid", None)
        now = utc_now()
        with self._uow.transaction():
            finalizing = self._store.update_task_run_cas(
                record.run_id,
                record.revision,
                updates={
                    "status": TaskRunStatus.FINALIZING,
                    "satisfied_requirement_count": satisfied,
                    "result_ref": result_ref,
                    "blockers": terminal_blockers,
                    "updated_at": now,
                },
                expected_runtime_epoch=self._runtime_epoch,
            )
            self._status_ledger(
                record,
                finalizing,
                label="terminal cleanup started",
            )
        try:
            with self._uow.transaction():
                purged_at = None
                if finalizing.retention is TaskRunRetention.PURGE_ON_TERMINAL:
                    self._purge_run_content(record.run_id, purged_at=now)
                    purged_at = now
                terminal_record = self._store.update_task_run_cas(
                    record.run_id,
                    finalizing.revision,
                    updates={
                        "status": terminal,
                        "completed_at": now,
                        "finalized_at": now,
                        "payloads_purged_at": purged_at,
                        "updated_at": now,
                    },
                    expected_runtime_epoch=self._runtime_epoch,
                )
                self._status_ledger(
                    finalizing,
                    terminal_record,
                    label="terminal cleanup completed",
                )
                return terminal_record
        except BaseException as exc:
            current = self._require_run(record.run_id)
            blocker = self._blocker(
                "cleanup_failed",
                "terminal payload cleanup did not converge",
                error_type=type(exc).__name__,
            )
            try:
                self._store.update_task_run_cas(
                    record.run_id,
                    current.revision,
                    updates={
                        "status": TaskRunStatus.FINALIZING,
                        "blockers": (blocker,),
                        "updated_at": utc_now(),
                    },
                    expected_runtime_epoch=self._runtime_epoch,
                )
            finally:
                raise

    def _terminal_blockers_after_convergence(
        self,
        record: TaskRunRecord,
        *,
        cancellation_intent: bool,
    ) -> tuple[dict[str, Any], ...]:
        """Drop only blocker projections disproved by terminal evidence.

        ``_finalize_terminal`` calls this only after
        ``_terminal_convergence_blocker`` has proved every Run effect is in a
        terminal settlement state and no paired local result remains pending.
        The earlier attention transition remains in the append-only ledger,
        while its now-obsolete effect/local-settlement projection is not
        carried into the final ``cancelled`` summary.  All durable non-effect
        blockers (for example ``deadline_reached``) remain part of the terminal
        projection. Effect blockers are cleared only for cancellation; a
        transient local-settlement marker is cleared for every terminal status.
        """

        if self._unsettled_effects(record.run_id):
            raise ValidationError(
                "TaskRun cancellation effect blockers cannot clear before settlement"
            )
        return tuple(
            dict(blocker)
            for blocker in record.blockers
            if not (
                cancellation_intent
                and blocker.get("kind")
                in {"unknown_effect", "effect_unsettled"}
            )
            and blocker.get("transient_local_settlement") is not True
        )

    def _terminal_convergence_blocker(
        self,
        record: TaskRunRecord,
    ) -> dict[str, Any] | None:
        """Return the first durable subsystem that has not terminally settled."""

        run_id = record.run_id
        unsettled = self._unsettled_effects(run_id)
        if unsettled:
            kind = (
                "unknown_effect"
                if any(
                    effect.transaction_state in _UNKNOWN_EFFECT_STATES
                    for effect in unsettled
                )
                else "effect_unsettled"
            )
            return self._blocker(
                kind,
                "external effect has not terminally settled",
                effect_ids=[effect.effect_id for effect in unsettled[:20]],
            )
        processes = self._tree_processes(run_id)
        pending_settlement = self._terminal_local_settlement_blocker(
            processes,
            cancellation_intent=(
                record.status is TaskRunStatus.CANCELLING
                or record.cancel_generation > 0
            ),
        )
        if pending_settlement is not None:
            return pending_settlement
        pids = tuple(process.pid for process in processes)
        for process in processes:
            if process.status not in _TERMINAL_PROCESS_STATUSES:
                return self._blocker(
                    "manual_recovery_required",
                    "TaskRun process tree is not terminal",
                    pid=process.pid,
                )
            try:
                cleanup = self._process.terminal_cleanup_state(process.pid)
            except Exception as exc:
                return self._blocker(
                    "cleanup_failed",
                    "terminal process cleanup evidence is unavailable",
                    pid=process.pid,
                    error_type=type(exc).__name__,
                )
            if cleanup.get("state") != "completed":
                return self._blocker(
                    "cleanup_failed",
                    "terminal process cleanup has not converged",
                    pid=process.pid,
                )

        for pid in pids:
            if self._has_unsettled_runtime_publication(pid):
                return self._blocker(
                    "publication_unsettled",
                    "Runtime publication has not converged",
                    pid=pid,
                )

        for pid in pids:
            active_usage = self._uow.resources.list_resource_usage_reservations(
                pid=pid,
                status="active",
            )
            if active_usage:
                return self._blocker(
                    "reservation_unsettled",
                    "resource-usage reservation remains active",
                    pid=pid,
                )
        process_reservations = list(
            self._uow.resources.list_resource_reservations(parent_pids=pids)
        )
        for pid in pids:
            process_reservations.extend(
                self._uow.resources.list_resource_reservations(child_pid=pid)
            )
        if process_reservations:
            return self._blocker(
                "reservation_unsettled",
                "process resource reservation remains active",
            )

        list_capability_reservations = getattr(
            self._store,
            "list_active_capability_use_reservations_for_pids",
            None,
        )
        if not callable(list_capability_reservations):
            return self._blocker(
                "reservation_unsettled",
                "capability-use reservation evidence is unavailable",
            )
        if list_capability_reservations(pids):
            return self._blocker(
                "reservation_unsettled",
                "capability-use reservation remains active",
            )
        return None

    def _terminal_local_settlement_blocker(
        self,
        processes: Iterable[Any],
        *,
        cancellation_intent: bool,
    ) -> dict[str, Any] | None:
        """Refuse terminal projection while a paired local result is pending."""

        for process in processes:
            point = self._store.get_task_run_resume_point(
                process.pid,
                complete_only=True,
            )
            if point is None or point.pending_action_payload_id is None:
                continue
            if not self._resume_integrity_valid(point):
                return self._blocker(
                    "payload_corrupt",
                    "terminal TaskRun resume point failed integrity",
                    pid=process.pid,
                )
            try:
                wrapper = self._decode_pending_resume_payload(point)
            except (KeyError, NotFound, TypeError, ValueError, ValidationError):
                return self._blocker(
                    "payload_corrupt",
                    "terminal TaskRun pending action is unreadable",
                    pid=process.pid,
                )
            kind = wrapper.get("kind")
            state = wrapper.get("state")
            if kind == "completed_outcome" and state == "staged":
                return self._blocker(
                    "effect_unsettled",
                    "paired local action result has not terminally settled",
                    pid=process.pid,
                    transient_local_settlement=True,
                )
            if kind == "validated_action" and state == "dispatching":
                changed = self._changed_effects_for_pid(
                    process.pid,
                    point.last_effect_seq,
                )
                expected_binding = {
                    "run_id": point.run_id,
                    "pid": process.pid,
                    "call_id": wrapper.get("call_id"),
                    "context_generation": wrapper.get("context_generation"),
                    "action_manifest_sha256": wrapper.get("manifest_sha256"),
                    "source_safe_point_seq": point.safe_point_seq,
                }
                if cancellation_intent and self._settled_effects_match_action(
                    changed,
                    expected_binding=expected_binding,
                ):
                    # Every claimed provider boundary has authoritative
                    # terminal truth. Cancellation does not need future prompt
                    # context or a retryable action it will never dispatch.
                    continue
                return self._blocker(
                    "pending_action_unreplayable",
                    "terminal process has an action without a paired local result",
                    pid=process.pid,
                    transient_local_settlement=True,
                )
            if not (
                (kind == "validated_action" and state == "validated")
                or (kind == "durable_wait_action" and state == "waiting")
            ):
                return self._blocker(
                    "payload_corrupt",
                    "terminal TaskRun pending action has an invalid state",
                    pid=process.pid,
                )
        return None

    @staticmethod
    def _settled_effects_match_action(
        changed: Iterable[Any],
        *,
        expected_binding: Mapping[str, Any],
    ) -> bool:
        selected = list(changed)
        return bool(selected) and all(
            effect.effect_state == "finalized"
            and effect.transaction_state in _SETTLED_EFFECT_STATES
            and effect.provider_metadata.get("task_run_action")
            == dict(expected_binding)
            for effect in selected
        )

    def _has_unsettled_runtime_publication(self, pid: str) -> bool:
        unsettled_states = {
            "planning",
            "applying",
            "reconciliation_pending",
            "rollback_pending",
            "manual",
        }
        return any(
            str(item.get("state")) in unsettled_states
            or item.get("operation_reconciled") is not True
            for item in self._store.list_runtime_publications(pid=pid)
        )

    def _has_active_quantum(self, run_id: str) -> bool:
        scheduler = getattr(self._host, "scheduler", None)
        active_pids = getattr(scheduler, "active_pids", None)
        if not callable(active_pids):
            return False
        members = set(self._member_pids(run_id))
        return any(pid in members for pid in active_pids())

    def _has_active_external_dispatch(self, run_id: str) -> bool:
        with self._condition:
            return self._active_external_dispatches.get(run_id, 0) > 0

    def _purge_run_content(self, run_id: str, *, purged_at: str) -> None:
        """Apply the complete Run-scoped irreversible content purge.

        The caller owns the surrounding transaction.  Every store primitive is
        monotonic/idempotent so a permanent explicit purge and terminal cleanup
        share exactly the same privacy boundary.
        """

        from agent_libos.evidence import (
            redact_task_run_external_effects,
            redact_task_run_human_requests,
            redact_task_run_llm_calls,
        )

        pids = self._member_pids(run_id)
        redact_task_run_human_requests(self._store, run_id, pids)
        redact_task_run_llm_calls(self._store, run_id, pids)
        redact_task_run_external_effects(self._store, run_id, pids)
        self._store.purge_task_run_llm_pending_actions(
            run_id,
            pids,
        )
        purge_messages = getattr(self._store, "purge_task_run_messages", None)
        if not callable(purge_messages):
            raise RuntimeError("Store lacks TaskRun message purge support")
        purge_messages(run_id, pids)
        self._store.purge_task_run_payloads(run_id, purged_at=purged_at)

    def _status_from_processes(
        self,
        record: TaskRunRecord,
        live: list[Any],
    ) -> TaskRunStatus:
        if record.status in {
            TaskRunStatus.PAUSED,
            TaskRunStatus.CANCELLING,
            TaskRunStatus.NEEDS_ATTENTION,
        }:
            return record.status
        statuses = {item.status for item in live}
        if ProcessStatus.WAITING_HUMAN in statuses:
            return TaskRunStatus.WAITING_HUMAN
        if ProcessStatus.WAITING_TOOL in statuses:
            return TaskRunStatus.WAITING_TOOL
        for item in live:
            wait_name = type(getattr(item, "wait_state", None)).__name__
            if wait_name == "MessageProcessWait":
                return TaskRunStatus.WAITING_MESSAGE
            if wait_name == "ChildProcessWait":
                return TaskRunStatus.WAITING_PROCESS
        if statuses == {ProcessStatus.PAUSED}:
            return TaskRunStatus.PAUSED
        return TaskRunStatus.RUNNING

    def _project_evidence(self, run_id: str) -> None:
        """Idempotently materialize bounded evidence into the Run ledger."""

        hard = self.config.task_runs.recovery_page_hard_limit
        record = self._require_run(run_id)
        processes = self._tree_processes(run_id)
        process_by_pid = {process.pid: process for process in processes}
        effects = self._bounded_evidence(
            self._effects_for_run(run_id),
            hard=hard,
            label="effect",
        )
        ignored_effects, ignored_operations = (
            self._post_purge_gui_projection_exclusions(
                record,
                effects,
                process_by_pid=process_by_pid,
            )
        )

        with self._uow.transaction():
            # Projection callers can race (for example an interrupt settling
            # while the admitted quantum finishes). Read the identity set
            # only after entering the Store's serialized transaction; a
            # pre-transaction snapshot can become stale and append a second
            # ledger item before the unique link insert detects the race.
            links = self._store.list_task_run_links(
                run_id,
                limit=hard + 1,
            )
            if len(links) > hard:
                raise ValidationError(
                    "TaskRun evidence links exceed the recovery bound"
                )
            existing = {
                (link.evidence_type, link.evidence_id, link.role)
                for link in links
            }
            for process in processes:
                self._project_process_evidence(
                    run_id,
                    process,
                    existing=existing,
                    ignored_operations=ignored_operations,
                    hard=hard,
                )
            self._project_effect_evidence(
                run_id,
                effects,
                existing=existing,
                ignored_effects=ignored_effects,
            )
            self._repair_purged_effect_projection(
                record,
                process_pids=tuple(sorted(process_by_pid)),
            )

    def _project_evidence_item(
        self,
        run_id: str,
        *,
        existing: set[tuple[str, str, str]],
        kind: TaskRunLedgerKind,
        status: str,
        label: str,
        evidence_type: str,
        evidence_id: str,
        role: str,
        metadata: Mapping[str, Any] | None = None,
        **identities: Any,
    ) -> None:
        key = (evidence_type, evidence_id, role)
        if key in existing:
            return
        ledger = self._append_ledger(
            run_id,
            kind=kind,
            status=status,
            label=label,
            metadata=metadata,
            **identities,
        )
        self._store.insert_task_run_link(
            TaskRunLink(
                link_id=new_id("trlink"),
                run_id=run_id,
                ledger_seq=ledger.seq,
                evidence_type=evidence_type,
                evidence_id=evidence_id,
                role=role,
                created_at=ledger.occurred_at,
            )
        )
        existing.add(key)

    @staticmethod
    def _bounded_evidence(
        items: list[Any],
        *,
        hard: int,
        label: str,
    ) -> list[Any]:
        if len(items) > hard:
            raise ValidationError(
                f"TaskRun {label} evidence exceeds recovery hard cap"
            )
        return items

    def _project_process_evidence(
        self,
        run_id: str,
        process: Any,
        *,
        existing: set[tuple[str, str, str]],
        ignored_operations: set[str],
        hard: int,
    ) -> None:
        self._project_evidence_item(
            run_id,
            existing=existing,
            kind=TaskRunLedgerKind.PROCESS,
            status=process.status.value,
            label="TaskRun process state",
            evidence_type="process",
            evidence_id=process.pid,
            role="process",
            pid=process.pid,
            metadata={
                "task_run_role": process.task_run_role,
                "parent_pid": process.parent_pid,
            },
        )
        self._project_operation_evidence(
            run_id,
            process,
            existing=existing,
            ignored_operations=ignored_operations,
            hard=hard,
        )
        self._project_human_evidence(
            run_id,
            process,
            existing=existing,
            hard=hard,
        )
        self._project_message_evidence(
            run_id,
            process,
            existing=existing,
            hard=hard,
        )
        self._project_checkpoint_evidence(
            run_id,
            process,
            existing=existing,
            hard=hard,
        )
        self._project_object_task_evidence(
            run_id,
            process,
            existing=existing,
            hard=hard,
        )

    def _project_operation_evidence(
        self,
        run_id: str,
        process: Any,
        *,
        existing: set[tuple[str, str, str]],
        ignored_operations: set[str],
        hard: int,
    ) -> None:
        operations = self._bounded_evidence(
            self._uow.evidence.list_operations(
                pid=process.pid,
                limit=hard + 1,
            ),
            hard=hard,
            label="operation",
        )
        for operation in operations:
            if operation.operation_id in ignored_operations:
                continue
            operation_kind = operation.kind.value
            self._project_evidence_item(
                run_id,
                existing=existing,
                kind=self._operation_ledger_kind(operation_kind),
                status=operation.outcome.value,
                label="TaskRun operation evidence",
                evidence_type="operation",
                evidence_id=operation.operation_id,
                role="operation",
                pid=process.pid,
                operation_id=operation.operation_id,
                metadata={
                    "operation_kind": operation_kind,
                    "operation_state": operation.state.value,
                },
            )

    @staticmethod
    def _operation_ledger_kind(operation_kind: str) -> TaskRunLedgerKind:
        if operation_kind == "llm_request":
            return TaskRunLedgerKind.LLM_TURN
        if operation_kind == "tool_call":
            return TaskRunLedgerKind.TOOL_CALL
        return TaskRunLedgerKind.PROCESS

    def _project_human_evidence(
        self,
        run_id: str,
        process: Any,
        *,
        existing: set[tuple[str, str, str]],
        hard: int,
    ) -> None:
        requests = self._bounded_evidence(
            self._store.list_human_requests(process.pid, limit=hard + 1),
            hard=hard,
            label="Human",
        )
        for request in requests:
            self._project_evidence_item(
                run_id,
                existing=existing,
                kind=TaskRunLedgerKind.HUMAN_WAIT,
                status=request.status.value,
                label="TaskRun Human request",
                evidence_type="human_request",
                evidence_id=request.request_id,
                role="wait",
                pid=process.pid,
                human_request_id=request.request_id,
                metadata={"blocking": bool(request.blocking)},
            )

    def _project_message_evidence(
        self,
        run_id: str,
        process: Any,
        *,
        existing: set[tuple[str, str, str]],
        hard: int,
    ) -> None:
        messages = self._bounded_evidence(
            self._store.list_process_messages(process.pid, limit=hard + 1),
            hard=hard,
            label="message",
        )
        for message in messages:
            self._project_evidence_item(
                run_id,
                existing=existing,
                kind=TaskRunLedgerKind.MESSAGE_WAIT,
                status=message.status.value,
                label="TaskRun durable process message",
                evidence_type="process_message",
                evidence_id=message.message_id,
                role="message",
                pid=process.pid,
                metadata={"kind": message.kind.value},
            )

    def _project_checkpoint_evidence(
        self,
        run_id: str,
        process: Any,
        *,
        existing: set[tuple[str, str, str]],
        hard: int,
    ) -> None:
        checkpoints = self._bounded_evidence(
            self._store.list_checkpoints(process.pid, limit=hard + 1),
            hard=hard,
            label="checkpoint",
        )
        for checkpoint in checkpoints:
            self._project_evidence_item(
                run_id,
                existing=existing,
                kind=TaskRunLedgerKind.CHECKPOINT,
                status="created",
                label="TaskRun checkpoint evidence",
                evidence_type="checkpoint",
                evidence_id=checkpoint.checkpoint_id,
                role="checkpoint",
                pid=process.pid,
                checkpoint_id=checkpoint.checkpoint_id,
            )

    def _project_object_task_evidence(
        self,
        run_id: str,
        process: Any,
        *,
        existing: set[tuple[str, str, str]],
        hard: int,
    ) -> None:
        tasks = self._bounded_evidence(
            self._store.list_object_tasks(
                creator_pid=process.pid,
                include_terminal=True,
                limit=hard + 1,
            ),
            hard=hard,
            label="ObjectTask",
        )
        for task in tasks:
            self._project_evidence_item(
                run_id,
                existing=existing,
                kind=TaskRunLedgerKind.PROCESS,
                status=task.status.value,
                label="TaskRun ObjectTask evidence",
                evidence_type="object_task",
                evidence_id=task.task_id,
                role="object_task",
                pid=process.pid,
                object_task_id=task.task_id,
            )

    def _project_effect_evidence(
        self,
        run_id: str,
        effects: list[Any],
        *,
        existing: set[tuple[str, str, str]],
        ignored_effects: set[str],
    ) -> None:
        for effect in effects:
            if effect.effect_id in ignored_effects:
                continue
            self._project_evidence_item(
                run_id,
                existing=existing,
                kind=TaskRunLedgerKind.EFFECT,
                status=f"{effect.effect_state}:{effect.transaction_state}",
                label="TaskRun external effect evidence",
                evidence_type="external_effect",
                evidence_id=effect.effect_id,
                role="effect",
                pid=effect.pid,
                effect_id=effect.effect_id,
                metadata={"provider": effect.provider},
            )

    def _post_purge_gui_projection_exclusions(
        self,
        record: TaskRunRecord,
        effects: list[Any],
        *,
        process_by_pid: Mapping[str, Any],
    ) -> tuple[set[str], set[str]]:
        """Keep later Host GUI observations outside a terminal Run ledger."""

        ignored_effects: set[str] = set()
        ignored_operations: set[str] = set()
        if record.payloads_purged_at is None:
            return ignored_effects, ignored_operations
        for effect in effects:
            operation_id = self._post_purge_gui_presentation_operation_id(
                record,
                effect,
                process=process_by_pid.get(effect.pid),
            )
            if operation_id is None:
                continue
            ignored_effects.add(effect.effect_id)
            ignored_operations.add(operation_id)
        return ignored_effects, ignored_operations

    def _repair_purged_effect_projection(
        self,
        record: TaskRunRecord,
        *,
        process_pids: tuple[str, ...],
    ) -> None:
        if record.payloads_purged_at is None:
            return
        # Preserve append-only legacy links while monotonically reducing every
        # linked terminal effect in the same transaction as new projections.
        # A nonterminal/conflicting effect raises and rolls back the new link.
        from agent_libos.evidence import redact_task_run_external_effects

        redact_task_run_external_effects(
            self._store,
            record.run_id,
            process_pids,
        )

    def _post_purge_gui_presentation_operation_id(
        self,
        record: TaskRunRecord,
        effect: Any,
        *,
        process: Any | None,
    ) -> str | None:
        """Prove one GUI presentation occurred strictly after terminal purge."""

        if not _is_post_purge_gui_presentation_candidate(record, effect, process):
            return None
        request_id = _gui_presentation_request_id(effect)
        if request_id is None:
            return None
        if not _has_exact_gui_presentation_context(effect, request_id):
            return None
        if not self._presentation_request_matches(
            effect,
            request_id=request_id,
            pid=process.pid,
        ):
            return None
        return self._exact_gui_presentation_operation_id(effect, pid=process.pid)

    def _presentation_request_matches(
        self,
        effect: Any,
        *,
        request_id: str,
        pid: str,
    ) -> bool:
        request = self._store.get_human_request(request_id)
        return bool(
            request is not None
            and request.pid == pid
            and effect.target == f"human:{request.human}"
        )

    def _exact_gui_presentation_operation_id(
        self,
        effect: Any,
        *,
        pid: str,
    ) -> str | None:
        links = self._uow.evidence.list_operation_evidence(
            evidence_types=("external_effect",),
            evidence_id=effect.effect_id,
            limit=2,
        )
        if len(links) != 1 or links[0].role != "effect":
            return None
        operation = self._uow.evidence.get_operation(links[0].operation_id)
        if not self._is_exact_gui_presentation_operation(operation, pid=pid):
            return None
        return operation.operation_id

    @staticmethod
    def _is_exact_gui_presentation_operation(
        operation: Any | None,
        *,
        pid: str,
    ) -> bool:
        return all(
            (
                operation is not None,
                operation is not None and operation.pid == pid,
                operation is not None and operation.actor == pid,
                operation is not None
                and operation.kind is OperationKind.PRIMITIVE,
                operation is not None
                and operation.name == "primitive.human.write",
                operation is not None and operation.state.value == "terminal",
                operation is not None
                and operation.outcome is OperationOutcome.SUCCEEDED,
                operation is not None and operation.completed_at is not None,
            )
        )

    # ------------------------------------------------------------------
    # Small integrity, persistence, and projection helpers

    def _claim_runtime_epoch(self) -> int:
        claim = getattr(self._store, "claim_runtime_epoch", None)
        if not callable(claim):
            raise RuntimeError("schema-v4 Store has no Runtime epoch fence")
        epoch = claim(self._host.instance_id)
        if type(epoch) is not int or epoch <= 0:
            raise RuntimeError("Store returned an invalid Runtime epoch")
        return epoch

    def _require_run(self, run_id: str) -> TaskRunRecord:
        selected = self._identifier(run_id, "run_id")
        record = self._store.get_task_run(selected)
        if record is None:
            raise NotFound(f"TaskRun not found: {selected}")
        return record

    def _require_expected_revision(
        self,
        run_id: str,
        expected: int,
    ) -> TaskRunRecord:
        if type(expected) is not int or expected < 0:
            raise ValidationError("TaskRun expected_revision must be non-negative")
        record = self._require_run(run_id)
        if record.revision != expected:
            error = TaskRunRevisionConflict(
                f"TaskRun revision conflict for {run_id}: expected {expected}, "
                f"found {record.revision}"
            )
            for name, value in (
                ("run_id", run_id),
                ("expected_revision", expected),
                ("actual_revision", record.revision),
            ):
                try:
                    setattr(error, name, value)
                except Exception:
                    pass
            raise error
        return record

    def _require_revision(self, run_id: str, expected: int) -> TaskRunRecord:
        record = self._require_expected_revision(run_id, expected)
        if record.runtime_epoch != self._runtime_epoch:
            raise TaskRunRevisionConflict(f"TaskRun epoch is stale: {run_id}")
        return record

    def _summary(self, record: TaskRunRecord) -> TaskRunSummary:
        return TaskRunSummary(
            run_id=record.run_id,
            revision=record.revision,
            status=record.status,
            display_title=record.display_title,
            root_pid=record.root_pid,
            active_pid=record.active_pid,
            step_count=record.step_count,
            completed_step_count=record.completed_step_count,
            requirement_count=record.requirement_count,
            satisfied_requirement_count=record.satisfied_requirement_count,
            blockers=tuple(self._public_blocker(item) for item in record.blockers),
            allowed_actions=self._allowed_actions(record),
            result_ref=record.result_ref,
            retention=record.retention,
            payloads_purged=record.payloads_purged_at is not None,
            created_at=record.created_at,
            updated_at=record.updated_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
        )

    def _allowed_actions(self, record: TaskRunRecord) -> tuple[str, ...]:
        status = record.status
        if status in TASK_RUN_TERMINAL_STATUSES:
            # A purged Run can still be rerun when the Host supplies a
            # replacement goal in ``spec_overrides``.  The rerun method itself
            # continues to fail closed when neither retained goal material nor
            # an explicit replacement is available.
            return ("rerun",)
        if status is TaskRunStatus.NEEDS_ATTENTION:
            return ("recover", "cancel")
        if status is TaskRunStatus.PAUSED:
            return ("resume", "cancel", "follow_up")
        if status in {TaskRunStatus.CANCELLING, TaskRunStatus.FINALIZING}:
            return ("wait",)
        # Dispatchable Run states are fenced by their persisted Run revision;
        # their normal process projection advances that revision before the
        # root can become an actionable terminal boundary.  Keep this return
        # revision-stable rather than deriving it from a concurrently changing
        # Process row.
        return ("run", "pause", "cancel", "follow_up")

    def _root_accepts_follow_up(self, record: TaskRunRecord) -> bool:
        if record.root_pid is None:
            return False
        root = self._store.get_process(record.root_pid)
        return bool(
            root is not None
            and root.task_run_id == record.run_id
            and root.task_run_epoch == record.runtime_epoch
            and root.status not in _TERMINAL_PROCESS_STATUSES
        )

    def _public_blocker(self, value: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(value.get("kind") or "manual_recovery_required")
        if kind not in _PUBLIC_BLOCKER_KINDS:
            kind = "manual_recovery_required"
        selected: dict[str, Any] = {"kind": kind}
        for key in (
            "message",
            "pid",
            "effect_ids",
            "object_task_ids",
            "error_type",
        ):
            item = value.get(key)
            if isinstance(item, (str, list, tuple)):
                selected[key] = list(item) if isinstance(item, tuple) else item
        return selected

    @staticmethod
    def _blocker(kind: str, message: str, **values: Any) -> dict[str, Any]:
        return {"kind": kind, "message": message, **values}

    def _mark_attention(
        self,
        record: TaskRunRecord,
        blocker: Mapping[str, Any],
        *,
        command_id: str | None = None,
        command_kind: str = "recovery",
        request: Mapping[str, Any] | None = None,
    ) -> TaskRunRecord:
        selected_blocker = dict(blocker)
        while True:
            current = self._require_run(record.run_id)
            if current.runtime_epoch != self._runtime_epoch:
                raise TaskRunRevisionConflict(
                    f"stale TaskRun epoch refused attention: {current.run_id}"
                )
            if current.status in {
                TaskRunStatus.FINALIZING,
                *TASK_RUN_TERMINAL_STATUSES,
            }:
                return current
            blockers = tuple(
                [*current.blockers, selected_blocker]
                if selected_blocker not in current.blockers
                else current.blockers
            )
            if (
                current.status is TaskRunStatus.NEEDS_ATTENTION
                and blockers == current.blockers
            ):
                return current
            try:
                updated = self._store.update_task_run_cas(
                    current.run_id,
                    current.revision,
                    updates={
                        "status": TaskRunStatus.NEEDS_ATTENTION,
                        "blockers": blockers,
                        "updated_at": utc_now(),
                    },
                    expected_runtime_epoch=self._runtime_epoch,
                )
                break
            except TaskRunRevisionConflict:
                if current.status not in {
                    TaskRunStatus.CANCELLING,
                    TaskRunStatus.NEEDS_ATTENTION,
                }:
                    raise
                # Persisted cancellation prevents new dispatch. Any competing
                # revisions are a finite set of already-admitted local settlements;
                # reread until attention converges on their revision.
                continue
        self._status_ledger(current, updated, label="manual attention required")
        if command_id is not None:
            self._record_command(
                updated,
                command_id,
                command_kind,
                dict(request or {}),
            )
        self._notify_updated()
        return updated

    def _status_ledger(
        self,
        before: TaskRunRecord,
        after: TaskRunRecord,
        *,
        label: str,
    ) -> TaskRunLedgerItem:
        return self._append_ledger(
            after.run_id,
            kind=TaskRunLedgerKind.STATUS_TRANSITION,
            status=after.status.value,
            label=label,
            metadata={"from": before.status.value, "to": after.status.value},
        )

    def _append_control_admission(
        self,
        before: TaskRunRecord,
        after: TaskRunRecord,
        *,
        command_id: str,
        command_kind: str,
        request: Mapping[str, Any],
        evidence: Mapping[str, Any],
        label: str,
    ) -> dict[str, Any]:
        """Append immutable provenance for a split local-control settlement."""

        selected_id = self._identifier(command_id, "command_id")
        request_hash = self._request_hash(command_kind, request)
        selected_evidence = to_jsonable(dict(evidence))
        evidence_sha256 = self._sha256(
            {
                "schema_version": 1,
                "run_id": after.run_id,
                "command_id": selected_id,
                "command_kind": command_kind,
                "request_hash": request_hash,
                "evidence": selected_evidence,
            }
        )
        item = self._append_ledger(
            after.run_id,
            kind=TaskRunLedgerKind.STATUS_TRANSITION,
            status=after.status.value,
            label=label,
            metadata={
                "schema_version": 1,
                "from": before.status.value,
                "to": after.status.value,
                "command_id": selected_id,
                "command_kind": command_kind,
                "request_hash": request_hash,
                "admission_evidence_sha256": evidence_sha256,
            },
        )
        return {
            "admission_ledger_seq": item.seq,
            "admission_ledger_item_id": item.item_id,
            "admission_evidence_sha256": evidence_sha256,
        }

    def _validate_control_admission(
        self,
        command: TaskRunCommand,
        *,
        evidence: Mapping[str, Any],
        label: str,
    ) -> None:
        """Bind mutable command projection fields to one append-only ledger row."""

        result = command.result
        ledger_seq = self._bounded_receipt_integer(
            result.get("admission_ledger_seq"),
            "control admission ledger sequence",
            minimum=1,
        )
        ledger_item_id = self._identifier(
            result.get("admission_ledger_item_id"),
            "control admission ledger item id",
        )
        evidence_sha256 = result.get("admission_evidence_sha256")
        if (
            type(evidence_sha256) is not str
            or len(evidence_sha256) != 64
            or any(char not in "0123456789abcdef" for char in evidence_sha256)
        ):
            raise ValidationError(
                "TaskRun control admission evidence digest is invalid"
            )
        expected_sha256 = self._sha256(
            {
                "schema_version": 1,
                "run_id": command.run_id,
                "command_id": command.command_id,
                "command_kind": command.command_kind,
                "request_hash": command.request_hash,
                "evidence": to_jsonable(dict(evidence)),
            }
        )
        if evidence_sha256 != expected_sha256:
            raise ValidationError(
                "TaskRun control admission receipt lost its evidence binding"
            )
        item = self._store.get_task_run_ledger_item(
            command.run_id,
            ledger_item_id,
        )
        if (
            item is None
            or item.seq != ledger_seq
            or item.run_id != command.run_id
            or item.kind is not TaskRunLedgerKind.STATUS_TRANSITION
            or item.label != label
            or type(item.metadata.get("from")) is not str
            or type(item.metadata.get("to")) is not str
            or item.status != item.metadata.get("to")
            or item.metadata
            != {
                "schema_version": 1,
                "from": item.metadata.get("from"),
                "to": item.metadata.get("to"),
                "command_id": command.command_id,
                "command_kind": command.command_kind,
                "request_hash": command.request_hash,
                "admission_evidence_sha256": evidence_sha256,
            }
        ):
            raise ValidationError(
                "TaskRun control admission ledger provenance is invalid"
            )

    def _append_ledger(
        self,
        run_id: str,
        *,
        kind: TaskRunLedgerKind,
        status: str,
        label: str,
        requirement_id: str | None = None,
        pid: str | None = None,
        llm_call_id: str | None = None,
        payload_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **links: Any,
    ) -> TaskRunLedgerItem:
        item = TaskRunLedgerItem(
            item_id=new_id("trled"),
            run_id=run_id,
            seq=0,
            kind=kind,
            status=status,
            label=label,
            occurred_at=utc_now(),
            requirement_id=requirement_id,
            pid=pid,
            llm_call_id=llm_call_id,
            payload_id=payload_id,
            metadata=dict(metadata or {}),
            **links,
        )
        return self._store.append_task_run_ledger_item(item)

    def _record_command(
        self,
        record: TaskRunRecord,
        command_id: str,
        command_kind: str,
        request: Mapping[str, Any],
        *,
        result: Mapping[str, Any] | None = None,
    ) -> TaskRunCommand:
        selected_id = self._identifier(command_id, "command_id")
        selected_result = self._command_result(record, extra=result)
        command = TaskRunCommand(
            command_id=selected_id,
            client_request_id=None,
            run_id=record.run_id,
            command_kind=command_kind,
            request_hash=self._request_hash(command_kind, request),
            result=selected_result,
            result_revision=record.revision,
            created_at=utc_now(),
        )
        return self._store.insert_task_run_command(
            command,
            expected_runtime_epoch=self._runtime_epoch,
        )

    def _record_optional_command(
        self,
        record: TaskRunRecord,
        command_id: str | None,
        command_kind: str,
        request: Mapping[str, Any],
        *,
        result: Mapping[str, Any] | None = None,
    ) -> bool:
        if command_id is None:
            return False
        self._record_command(
            record,
            command_id,
            command_kind,
            request,
            result=result,
        )
        return True

    def _command_replay(
        self,
        run_id: str,
        command_id: str,
        command_kind: str,
        request: Mapping[str, Any],
        *,
        prefer_linked_result: bool = False,
    ) -> TaskRunSummary | None:
        selected_id = self._identifier(command_id, "command_id")
        existing = self._store.get_task_run_command(run_id, selected_id)
        if existing is None:
            return None
        self._require_same_command(
            existing,
            command_kind,
            self._request_hash(command_kind, request),
        )
        source = self._summary_from_command(existing)
        if prefer_linked_result:
            linked = self._linked_summary_from_command(existing)
            if linked is not None:
                return linked
        return source

    def _linked_summary_from_command(
        self,
        command: TaskRunCommand,
    ) -> TaskRunSummary | None:
        """Decode the immutable target of a command that created a linked Run."""

        if not (set(command.result) & _LINKED_RESULT_FIELDS):
            return None
        self._validate_linked_command_result(command)
        return self._validated_summary_mapping(
            command.result.get("new_run_summary"),
            "linked target",
        )

    def _complete_command(
        self,
        record: TaskRunRecord,
        command_id: str,
        command_kind: str,
        request: Mapping[str, Any],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> TaskRunCommand:
        """CAS the provisional receipt to the exact synchronous result.

        Mutations that may dispatch or settle work insert a command before the
        effectful phase, so a crash never authorizes a second dispatch.  Once
        that phase converges, this CAS replaces the provisional public summary
        with the result actually returned to the caller.
        """

        selected_id = self._identifier(command_id, "command_id")
        existing = self._store.get_task_run_command(record.run_id, selected_id)
        if existing is None:
            raise TaskRunRevisionConflict("TaskRun command receipt is missing")
        self._require_same_command(
            existing,
            command_kind,
            self._request_hash(command_kind, request),
        )
        selected_extra = dict(extra or {})
        for field in _CONTROL_COMPLETION_PRESERVED_FIELDS:
            if field in existing.result:
                selected_extra[field] = existing.result[field]
        return self._store.update_task_run_command_result(
            record.run_id,
            selected_id,
            expected_result_revision=existing.result_revision,
            result=self._command_result(
                record,
                extra=selected_extra or None,
            ),
            result_revision=record.revision,
            expected_runtime_epoch=self._runtime_epoch,
        )

    def _complete_command_summary(
        self,
        record: TaskRunRecord,
        command_id: str,
        command_kind: str,
        request: Mapping[str, Any],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> TaskRunSummary:
        self._complete_command(
            record,
            command_id,
            command_kind,
            request,
            extra=extra,
        )
        return self._summary(record)

    def _command_result(
        self,
        record: TaskRunRecord,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the bounded, immutable public result of one mutation.

        A command replay must not silently turn into a read of the newest Run
        revision: callers use the stored result to distinguish an exact retry
        from an unrelated later mutation.  Keep the full *public* summary (not
        private payloads or provider material) in ``result_json`` and permit a
        small command-specific envelope such as ``new_run_id``.
        """

        result: dict[str, Any] = {
            "schema_version": 1,
            "summary": to_jsonable(self._summary(record)),
        }
        if extra is not None:
            for key, value in dict(extra).items():
                if key in {"schema_version", "summary"}:
                    raise ValidationError(
                        "TaskRun command result extras use a reserved field"
                    )
                result[str(key)] = to_jsonable(value)
        canonical_task_run_json(result)
        return result

    def _summary_from_command(self, command: TaskRunCommand) -> TaskRunSummary:
        """Decode the exact v1, revision-bound public mutation result."""

        try:
            encoded = canonical_task_run_json(command.result).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "TaskRun command contains an invalid stored result"
            ) from exc
        if len(encoded) > self.config.task_runs.command_result_max_bytes:
            raise ValidationError(
                "TaskRun command result exceeds configured maximum"
            )
        if (
            type(command.result_revision) is not int
            or command.result_revision < 0
            or command.result_revision > _SIGNED_BIGINT_MAX
            or type(command.result.get("schema_version")) is not int
            or command.result.get("schema_version") != 1
        ):
            raise ValidationError(
                "TaskRun command contains an invalid stored result envelope"
            )
        value = command.result.get("summary")
        if not isinstance(value, Mapping):
            raise ValidationError(
                "TaskRun command contains an invalid stored result envelope"
            )
        try:
            summary = TaskRunSummary(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "TaskRun command contains an invalid stored result"
            ) from exc
        if to_jsonable(summary) != dict(value):
            raise ValidationError(
                "TaskRun command contains a non-canonical stored summary"
            )
        if (
            summary.run_id != command.run_id
            or summary.revision != command.result_revision
        ):
            raise ValidationError(
                "TaskRun command stored result is not revision-bound"
            )
        return summary

    @staticmethod
    def _require_result_keys(
        result: Mapping[str, Any],
        expected: frozenset[str],
        label: str,
    ) -> None:
        if set(result) != expected:
            raise ValidationError(f"TaskRun {label} has an invalid result schema")

    @staticmethod
    def _bounded_receipt_integer(
        value: Any,
        label: str,
        *,
        minimum: int,
    ) -> int:
        if (
            type(value) is not int
            or value < minimum
            or value > _SIGNED_BIGINT_MAX
        ):
            raise ValidationError(
                f"TaskRun {label} is outside the signed BIGINT range"
            )
        return value

    @staticmethod
    def _settlement_state(result: Mapping[str, Any], label: str) -> str:
        state = result.get("settlement_state")
        if state not in {"pending", "complete"} or type(state) is not str:
            raise ValidationError(f"TaskRun {label} has an invalid settlement state")
        return state

    def _validate_command_result_for_kind(self, command: TaskRunCommand) -> None:
        self._summary_from_command(command)
        validators: dict[str, Callable[[TaskRunCommand], None]] = {
            "run": self._validate_run_command_result,
            "resume": self._validate_resume_command_result,
            "cancel": self._validate_cancel_command_result,
            "follow_up": self._validate_follow_up_command_result,
            "recover": self._validate_recover_command_result,
            "rerun": self._validate_linked_command_result,
            "deadline": self._validate_deadline_command_result,
        }
        validator = validators.get(command.command_kind)
        if validator is not None:
            validator(command)
            return
        if command.command_kind not in {
            "create",
            "pause",
            "purge_payloads",
            "recovery",
        }:
            raise ValidationError("TaskRun command has an unknown result contract")
        self._require_result_keys(
            command.result,
            _COMMAND_RESULT_BASE_KEYS,
            f"{command.command_kind} command result",
        )

    def _validate_run_command_result(self, command: TaskRunCommand) -> None:
        keys = set(command.result)
        if keys == _COMMAND_RESULT_BASE_KEYS:
            return
        if keys == _RUN_RESULT_KEYS:
            settlement_state = self._settlement_state(command.result, "run receipt")
            admission_revision = self._bounded_receipt_integer(
                command.result.get("admission_revision"),
                "run admission revision",
                minimum=1,
            )
            if (
                settlement_state == "pending"
                and admission_revision != command.result_revision
            ):
                raise ValidationError(
                    "TaskRun pending run receipt is not admission-revision-bound"
                )
            self._validate_control_admission(
                command,
                evidence={
                    "kind": "run",
                    "admission_revision": admission_revision,
                },
                label="explicit dispatch admitted",
            )
            return
        if keys == _DEADLINE_RESULT_KEYS:
            self._validate_deadline_command_result(command)
            return
        raise ValidationError("TaskRun run command has an invalid result schema")

    def _validate_resume_command_result(self, command: TaskRunCommand) -> None:
        keys = set(command.result)
        if keys == _COMMAND_RESULT_BASE_KEYS:
            return
        if keys == _RESUME_RESULT_KEYS:
            self._settlement_state(command.result, "resume receipt")
            pause_generation = self._bounded_receipt_integer(
                command.result.get("pause_generation"),
                "resume pause generation",
                minimum=0,
            )
            self._validate_control_admission(
                command,
                evidence={
                    "kind": "resume",
                    "pause_generation": pause_generation,
                },
                label="resume admitted",
            )
            return
        if keys == _DEADLINE_RESULT_KEYS:
            self._validate_deadline_command_result(command)
            return
        raise ValidationError("TaskRun resume command has an invalid result schema")

    def _validate_cancel_command_result(self, command: TaskRunCommand) -> None:
        self._require_result_keys(
            command.result,
            _CANCEL_RESULT_KEYS,
            "cancel receipt",
        )
        self._settlement_state(command.result, "cancel receipt")
        cancel_generation = self._bounded_receipt_integer(
            command.result.get("cancel_generation"),
            "cancel generation",
            minimum=1,
        )
        self._validate_control_admission(
            command,
            evidence={
                "kind": "cancel",
                "cancel_generation": cancel_generation,
            },
            label="cancel intent persisted",
        )

    def _validate_deadline_command_result(self, command: TaskRunCommand) -> None:
        self._require_result_keys(
            command.result,
            _DEADLINE_RESULT_KEYS,
            "deadline receipt",
        )
        self._settlement_state(command.result, "deadline receipt")
        if command.result.get("settlement_kind") != "deadline":
            raise ValidationError("TaskRun deadline receipt has an invalid kind")
        cancel_generation = self._bounded_receipt_integer(
            command.result.get("cancel_generation"),
            "deadline cancellation generation",
            minimum=1,
        )
        self._validate_control_admission(
            command,
            evidence={
                "kind": "deadline",
                "cancel_generation": cancel_generation,
            },
            label="deadline command admitted",
        )

    def _validate_follow_up_command_result(self, command: TaskRunCommand) -> None:
        if set(command.result) == _COMMAND_RESULT_BASE_KEYS:
            return
        self._validated_interrupt_receipt(command)

    def _validate_recover_command_result(self, command: TaskRunCommand) -> None:
        keys = set(command.result)
        if keys == _EFFECT_RECEIPT_RESULT_KEYS:
            self._validated_effect_receipt(command)
            return
        if keys == _TERMINALIZE_RESULT_KEYS:
            self._validated_terminalize_receipt(command)
            return
        if keys == _LINKED_RESULT_KEYS:
            self._validate_linked_command_result(command)
            return
        raise ValidationError("TaskRun recover command has an invalid result schema")

    def _validate_linked_command_result(self, command: TaskRunCommand) -> None:
        self._require_result_keys(
            command.result,
            _LINKED_RESULT_KEYS,
            "linked command result",
        )
        source = self._summary_from_command(command)
        source_run_id = self._identifier(
            command.result.get("run_id"),
            "linked source run_id",
        )
        source_revision = self._bounded_receipt_integer(
            command.result.get("revision"),
            "linked source revision",
            minimum=0,
        )
        if (
            source_run_id != command.run_id
            or source_revision != command.result_revision
            or source_revision != source.revision
        ):
            raise ValidationError(
                "TaskRun linked command source result is not revision-bound"
            )
        target_run_id = self._identifier(
            command.result.get("new_run_id"),
            "linked target run_id",
        )
        target = self._validated_summary_mapping(
            command.result.get("new_run_summary"),
            "linked target",
        )
        if target_run_id == source_run_id or target.run_id != target_run_id:
            raise ValidationError(
                "TaskRun linked command target result is not identity-bound"
            )

    @staticmethod
    def _validated_summary_mapping(value: Any, label: str) -> TaskRunSummary:
        if not isinstance(value, Mapping):
            raise ValidationError(f"TaskRun {label} summary is missing")
        try:
            summary = TaskRunSummary(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"TaskRun {label} summary is invalid") from exc
        if to_jsonable(summary) != dict(value):
            raise ValidationError(f"TaskRun {label} summary is not canonical")
        return summary

    def _require_same_command(
        self,
        existing: TaskRunCommand,
        command_kind: str,
        request_hash: str,
    ) -> None:
        if (
            existing.command_kind != command_kind
            or existing.request_hash != request_hash
        ):
            raise TaskRunCommandConflict(
                "TaskRun idempotency key was reused with a different request"
            )
        self._validate_command_result_for_kind(existing)

    @staticmethod
    def _request_hash(kind: str, request: Mapping[str, Any]) -> str:
        return TaskRunManager._sha256(
            {"schema_version": 1, "kind": kind, "request": dict(request)}
        )

    @staticmethod
    def _sha256(value: Any) -> str:
        return hashlib.sha256(canonical_task_run_json(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _identifier(value: Any, label: str) -> str:
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or "\x00" in value
            or len(value) > 256
        ):
            raise ValidationError(f"TaskRun {label} must be a canonical identifier")
        return value

    def _require_payload_bound(self, payload: TaskRunPayload) -> None:
        if payload.size_bytes > self.config.task_runs.payload_max_bytes:
            raise ValidationError("TaskRun payload exceeds configured maximum")
        if payload.retention_state is not TaskRunPayloadRetention.PLAINTEXT:
            raise ValidationError("new TaskRun payload must be plaintext")

    def _require_settlement_epoch(
        self,
        process: Any,
        record: TaskRunRecord,
        action: str,
    ) -> None:
        """Fence a result commit without treating it as new dispatch.

        A result already produced by an in-flight quantum may be persisted
        while the Run is PAUSED or CANCELLING.  It may not cross a Runtime
        epoch or mutate a terminal Run.
        """

        epoch = getattr(process, "task_run_epoch", None)
        if (
            getattr(process, "task_run_id", None) != record.run_id
            or epoch != record.runtime_epoch
            or epoch != self._runtime_epoch
        ):
            raise TaskRunRevisionConflict(
                f"stale TaskRun epoch refused {action}: {record.run_id}"
            )
        if record.status in TASK_RUN_TERMINAL_STATUSES or record.status is TaskRunStatus.FINALIZING:
            raise ValidationError(
                f"TaskRun status refuses {action}: {record.status.value}"
            )

    def _local_llm_call_sha256(
        self,
        process: Any,
        call_id: str,
    ) -> str:
        call = self._store.get_llm_call(call_id)
        if (
            call is None
            or call.call_id != call_id
            or call.pid != process.pid
            or call.status != "ok"
            or not call.completed_at
            or (call.image_id is not None and call.image_id != process.image_id)
        ):
            raise ValidationError(
                "TaskRun action is not backed by a completed local LLM call"
            )
        return self._sha256(
            {
                "call_id": call.call_id,
                "pid": call.pid,
                "image_id": call.image_id,
                "purpose": call.purpose,
                "status": call.status,
                "api": call.api,
                "model": call.model,
                "request_id": call.request_id,
                "response_id": call.response_id,
                "request_options": call.request_options,
                "response_content": call.response_content,
                "tool_calls": call.tool_calls,
                "reasoning": call.reasoning,
                "usage": call.usage,
                "completed_at": call.completed_at,
            }
        )

    def _llm_prompt_requirement_binding(
        self,
        process: Any,
        record: TaskRunRecord,
        *,
        call_id: str,
        context_generation: str,
        current_requirements: Mapping[str, TaskRunRequirement] | None = None,
        check_cached_binding: bool = True,
    ) -> dict[str, Any] | None:
        call = self._store.get_llm_call(call_id)
        if call is None or call.pid != process.pid:
            raise ValidationError(
                "TaskRun requirement binding has no local LLM call"
            )
        raw = call.request_options.get(_TASK_RUN_REQUIREMENT_BINDING_KEY)
        if raw is None:
            return None
        if current_requirements is not None:
            binding = self._validated_prompt_requirement_binding(
                record,
                pid=process.pid,
                context_generation=context_generation,
                value=raw,
                allowed_statuses={TaskRunRequirementStatus.IN_PROGRESS},
                current_requirements=current_requirements,
            )
        else:
            binding = self._validated_llm_requirement_binding_from_current_run(
                process,
                record,
                context_generation=context_generation,
                value=raw,
            )
        if check_cached_binding:
            self._require_cached_prompt_requirement_binding(
                process.pid,
                context_generation,
                binding,
            )
        return binding

    def _require_cached_prompt_requirement_binding(
        self,
        pid: str,
        context_generation: str,
        binding: Mapping[str, Any],
    ) -> None:
        with self._condition:
            cached = self._prompt_requirement_bindings.get(
                (pid, context_generation)
            )
        if cached is not None and binding != cached:
            raise ValidationError(
                "TaskRun LLM call changed its frozen requirement binding"
            )

    def _validated_llm_requirement_binding_from_current_run(
        self,
        process: Any,
        record: TaskRunRecord,
        *,
        context_generation: str,
        value: Any,
    ) -> dict[str, Any]:
        for attempt in range(4):
            with self._uow.transaction():
                current = self._require_run(record.run_id)
                self._require_settlement_epoch(
                    process,
                    current,
                    "TaskRun LLM requirement binding",
                )
                try:
                    requirements = self._bounded_completion_requirements(current)
                except ValidationError:
                    latest = self._require_run(record.run_id)
                    if (
                        latest.revision != current.revision
                        or latest.runtime_epoch != current.runtime_epoch
                    ):
                        if attempt < 3:
                            continue
                        raise TaskRunRevisionConflict(
                            "TaskRun LLM requirement binding kept changing"
                        )
                    raise
                return self._validated_prompt_requirement_binding(
                    current,
                    pid=process.pid,
                    context_generation=context_generation,
                    value=value,
                    allowed_statuses={TaskRunRequirementStatus.IN_PROGRESS},
                    current_requirements={
                        requirement.requirement_id: requirement
                        for requirement in requirements
                    },
                )
        raise TaskRunRevisionConflict(
            "TaskRun LLM requirement binding kept changing"
        )

    def _validated_prompt_requirement_binding(
        self,
        record: TaskRunRecord,
        *,
        pid: str,
        context_generation: str,
        value: Any,
        allowed_statuses: set[TaskRunRequirementStatus],
        current_requirements: Mapping[str, TaskRunRequirement] | None = None,
    ) -> dict[str, Any]:
        selected = self._require_prompt_requirement_binding_header(
            record,
            pid=pid,
            context_generation=context_generation,
            value=value,
        )
        current = (
            {
                requirement.requirement_id: requirement
                for requirement in self._bounded_completion_requirements(record)
            }
            if current_requirements is None
            else dict(current_requirements)
        )
        normalized = self._normalized_prompt_requirement_entries(
            selected["requirements"],
            record=record,
            current=current,
            allowed_statuses=allowed_statuses,
        )
        binding = {
            "schema_version": 1,
            "run_id": record.run_id,
            "pid": pid,
            "context_generation": context_generation,
            "requirements": normalized,
        }
        canonical_task_run_json(binding)
        return binding

    @staticmethod
    def _require_prompt_requirement_binding_header(
        record: TaskRunRecord,
        *,
        pid: str,
        context_generation: str,
        value: Any,
    ) -> dict[str, Any]:
        if (
            not isinstance(value, Mapping)
            or set(value) != _PROMPT_REQUIREMENT_BINDING_KEYS
        ):
            raise ValidationError("TaskRun requirement binding has an invalid shape")
        selected = dict(value)
        valid = (
            selected.get("schema_version") == 1
            and type(selected.get("schema_version")) is int
            and selected.get("run_id") == record.run_id
            and selected.get("pid") == pid
            and selected.get("context_generation") == context_generation
            and isinstance(selected.get("requirements"), list)
        )
        if not valid:
            raise ValidationError("TaskRun requirement binding is invalid")
        return selected

    def _normalized_prompt_requirement_entries(
        self,
        entries: Iterable[Any],
        *,
        record: TaskRunRecord,
        current: Mapping[str, TaskRunRequirement],
        allowed_statuses: set[TaskRunRequirementStatus],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        prior_ordinal = -1
        for raw_entry in entries:
            entry = self._validated_prompt_requirement_entry(
                raw_entry,
                record=record,
                current=current,
                seen=seen,
                prior_ordinal=prior_ordinal,
                allowed_statuses=allowed_statuses,
            )
            requirement_id = entry["requirement_id"]
            ordinal = entry["ordinal"]
            seen.add(requirement_id)
            prior_ordinal = ordinal
            normalized.append(entry)
        return normalized

    @staticmethod
    def _validated_prompt_requirement_entry(
        value: Any,
        *,
        record: TaskRunRecord,
        current: Mapping[str, TaskRunRequirement],
        seen: set[str],
        prior_ordinal: int,
        allowed_statuses: set[TaskRunRequirementStatus],
    ) -> dict[str, Any]:
        if (
            not isinstance(value, Mapping)
            or set(value) != _PROMPT_REQUIREMENT_ENTRY_KEYS
        ):
            raise ValidationError(
                "TaskRun requirement binding entry has an invalid shape"
            )
        entry = dict(value)
        requirement_id = entry.get("requirement_id")
        ordinal = entry.get("ordinal")
        requirement_sha256 = entry.get("requirement_sha256")
        requirement = (
            current.get(requirement_id)
            if isinstance(requirement_id, str)
            else None
        )
        valid_identity = (
            isinstance(requirement_id, str)
            and bool(requirement_id)
            and requirement_id not in seen
            and type(ordinal) is int
            and ordinal >= 0
            and ordinal > prior_ordinal
            and isinstance(requirement_sha256, str)
        )
        valid_requirement = (
            requirement is not None
            and requirement.run_id == record.run_id
            and requirement.ordinal == ordinal
            and requirement.requirement_sha256 == requirement_sha256
            and requirement.status in allowed_statuses
        )
        if not valid_identity or not valid_requirement:
            raise ValidationError(
                "TaskRun requirement binding lost its durable requirement"
            )
        return {
            "requirement_id": requirement_id,
            "ordinal": ordinal,
            "requirement_sha256": requirement_sha256,
        }

    def _durable_wait_snapshot(
        self,
        pid: str,
        pending: Mapping[str, Any] | None,
        *,
        call_id: str,
    ) -> dict[str, Any]:
        if not isinstance(pending, Mapping):
            raise ValidationError("TaskRun durable wait row is missing")
        selected = dict(pending)
        try:
            selected_call_id = pending_task_run_transcript_call_id(selected)
            resume_token = pending_resume_token(selected)
            DataFlowContext.from_dict(dict(selected.get("data_flow_context") or {}))
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValidationError("TaskRun durable wait row is invalid") from exc
        if selected.get("pid") != pid or selected_call_id != call_id:
            raise ValidationError("TaskRun durable wait lost its transcript binding")
        wait_type = selected.get("wait_type")
        status = selected.get("status")
        if wait_type not in _DURABLE_WAIT_TYPES or status not in {
            "pending",
            "resuming",
            "completed",
        }:
            raise ValidationError("TaskRun durable wait state is invalid")
        action = selected.get("action")
        filters = selected.get("filters")
        tool_call_count = selected.get("tool_call_count")
        if (
            not isinstance(action, dict)
            or not action
            or not isinstance(filters, dict)
            or type(tool_call_count) is not int
            or tool_call_count < 0
        ):
            raise ValidationError("TaskRun durable wait action is incomplete")
        if wait_type in {"llm_release", "human"} and not selected.get("request_id"):
            raise ValidationError("TaskRun durable wait request identity is missing")
        if wait_type == "child" and not selected.get("child_pid"):
            raise ValidationError("TaskRun durable child wait identity is missing")
        snapshot = {
            "schema_version": 1,
            "pid": pid,
            "call_id": call_id,
            "resume_token": resume_token,
            "llm_operation_id": selected.get("llm_operation_id"),
            "tool_operation_id": selected.get("tool_operation_id"),
            "wait_type": wait_type,
            "request_id": selected.get("request_id"),
            "child_pid": selected.get("child_pid"),
            "response_id": selected.get("response_id"),
            "tool_call_id": selected.get("tool_call_id"),
            "tool_name": selected.get("tool_name"),
            "filters": filters,
            "action": action,
            "data_flow_context": dict(selected.get("data_flow_context") or {}),
            "content_preview": str(selected.get("content_preview") or ""),
            "tool_call_count": tool_call_count,
            "status": status,
            "created_at": str(selected.get("created_at") or ""),
        }
        canonical_task_run_json(snapshot)
        return snapshot

    def _durable_wait_identity_sha256(self, snapshot: Mapping[str, Any]) -> str:
        return self._sha256(
            {key: value for key, value in snapshot.items() if key != "status"}
        )

    def _effect_settlement_projection(self, effect: Any) -> dict[str, Any]:
        return {
            "effect_id": effect.effect_id,
            "pid": effect.pid,
            "effect_state": effect.effect_state,
            "transaction_state": effect.transaction_state,
            "canonical_args_hash": effect.canonical_args_hash,
            "idempotency_key": effect.idempotency_key,
            "provider_receipt_sha256": self._sha256(effect.provider_receipt),
            "provider_metadata_sha256": self._sha256(effect.provider_metadata),
        }

    def _changed_effects_for_pid(self, pid: str, baseline: int) -> list[Any]:
        changed = self._uow.evidence.list_external_effects_changed_after(
            baseline,
            pids=(pid,),
        )
        if len(changed) > self.config.task_runs.recovery_page_hard_limit:
            raise ValidationError("TaskRun effect settlement exceeds recovery bound")
        return changed

    def _settled_effect_bundle(
        self,
        pid: str,
        baseline: int,
    ) -> tuple[int, list[dict[str, Any]]]:
        changed = self._changed_effects_for_pid(pid, baseline)
        if any(
            effect.effect_state != "finalized"
            or effect.transaction_state not in _SETTLED_EFFECT_STATES
            for effect in changed
        ):
            raise ValidationError("TaskRun external effect is not safely settled")
        return (
            self._current_effect_seq(),
            [self._effect_settlement_projection(effect) for effect in changed],
        )

    def _validate_effect_settlement_bundle(
        self,
        pid: str,
        wrapper: Mapping[str, Any],
    ) -> None:
        baseline = wrapper.get("effect_baseline_seq")
        settled_seq = wrapper.get("settled_effect_seq")
        projections = wrapper.get("settled_effects")
        if (
            type(baseline) is not int
            or baseline < 0
            or type(settled_seq) is not int
            or settled_seq < baseline
            or not isinstance(projections, list)
            or len(projections) > self.config.task_runs.recovery_page_hard_limit
        ):
            raise ValidationError("TaskRun staged effect bundle is invalid")
        current = [
            self._effect_settlement_projection(effect)
            for effect in self._changed_effects_for_pid(pid, baseline)
        ]
        if current != projections:
            raise ValidationError("TaskRun staged effect evidence changed")
        if any(
            item.get("pid") != pid
            or item.get("effect_state") != "finalized"
            or item.get("transaction_state") not in _SETTLED_EFFECT_STATES
            for item in current
        ):
            raise ValidationError("TaskRun staged effect evidence is not terminal")
        if self._changed_effects_for_pid(pid, settled_seq):
            raise ValidationError("TaskRun effect changed after local result staging")

    def _payload_by_role(self, run_id: str, role: str) -> TaskRunPayload:
        if role != "goal":
            raise ValidationError(
                "TaskRun role lookup is restricted to the authoritative goal"
            )
        requirements = self._store.list_task_run_requirements(
            run_id,
            after=None,
            limit=1,
        )
        if not requirements:
            raise NotFound(f"TaskRun goal payload not found: {run_id}")
        initial = requirements[0]
        payload = self._store.get_task_run_payload(initial.payload_id)
        if (
            initial.run_id != run_id
            or initial.ordinal != 0
            or initial.kind is not TaskRunRequirementKind.INITIAL
            or payload is None
            or payload.run_id != run_id
            or payload.role != "goal"
            or payload.sha256 != initial.requirement_sha256
        ):
            raise ValidationError("TaskRun goal payload binding is invalid")
        return payload

    def _requirement_view(
        self,
        requirement: TaskRunRequirement,
    ) -> Mapping[str, Any]:
        payload = self._store.get_task_run_payload(requirement.payload_id)
        retention = (
            payload.retention_state.value
            if payload is not None
            else TaskRunPayloadRetention.HASH_ONLY.value
        )
        view: dict[str, Any] = {
            "schema_version": 1,
            "requirement_id": requirement.requirement_id,
            "run_id": requirement.run_id,
            "ordinal": requirement.ordinal,
            "kind": requirement.kind.value,
            "status": requirement.status.value,
            "label": requirement.label,
            "content_retention": retention,
            "content_available": False,
            "content_sha256": requirement.requirement_sha256,
            "requirement_sha256": requirement.requirement_sha256,
            "created_by": requirement.created_by,
            "created_at": requirement.created_at,
            "updated_at": requirement.updated_at,
            "started_at": requirement.started_at,
            "completed_at": requirement.completed_at,
            "waived_by": requirement.waived_by,
            "waiver_reason": requirement.waiver_reason,
        }
        if (
            payload is not None
            and payload.retention_state is TaskRunPayloadRetention.PLAINTEXT
        ):
            decoded = self._decode_payload(payload, role="requirement")
            content = decoded.get("goal", decoded.get("body", decoded))
            view["content_available"] = True
            view["content_text"] = self._content_text(content)
        return view

    def _decode_payload(
        self,
        payload: TaskRunPayload | None,
        *,
        role: str,
    ) -> dict[str, Any]:
        if payload is None:
            raise NotFound(f"TaskRun {role} payload is missing")
        if (
            payload.retention_state is not TaskRunPayloadRetention.PLAINTEXT
            or payload.canonical_json is None
        ):
            raise ValidationError(f"TaskRun {role} payload is hash-only")
        if task_run_payload_sha256(payload.canonical_json) != payload.sha256:
            raise ValidationError(f"TaskRun {role} payload hash mismatch")
        try:
            value = json.loads(payload.canonical_json)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"TaskRun {role} payload is invalid JSON") from exc
        if canonical_task_run_json(value) != payload.canonical_json:
            raise ValidationError(f"TaskRun {role} payload is not canonical")
        if not isinstance(value, dict):
            raise ValidationError(f"TaskRun {role} payload must be an object")
        return value

    @staticmethod
    def _content_text(value: Any) -> str:
        return value if isinstance(value, str) else canonical_task_run_json(value)

    @staticmethod
    def _valid_replay_message(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        role = value.get("role")
        if role not in {"assistant", "tool", "user"}:
            return False
        if role == "system":
            return False
        return set(value) <= {"role", "content", "call_id", "tool_call_id", "name"}

    def _validated_outcome_manifest(
        self,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != _COMPLETED_OUTCOME_KEYS:
            raise ValidationError("TaskRun outcome manifest has an invalid shape")
        selected = dict(value)
        if type(selected.get("schema_version")) is not int or selected.get("schema_version") != 1:
            raise ValidationError("TaskRun outcome manifest schema must be 1")
        if selected.get("state") not in {"completed", "waiting"}:
            raise ValidationError("TaskRun outcome state is invalid")
        if selected.get("paired_outputs_persisted") is not True:
            raise ValidationError("TaskRun paired outputs are not durably persisted")
        try:
            selected["data_labels"] = DataLabels.from_dict(
                selected.get("data_labels")
            ).to_dict()
        except (TypeError, ValueError) as exc:
            raise ValidationError("TaskRun outcome data labels are invalid") from exc
        if selected["state"] == "waiting":
            wait = selected.get("durable_wait")
            if (
                not isinstance(wait, Mapping)
                or wait.get("wait_type") not in {"human", "process", "message"}
            ):
                raise ValidationError("TaskRun waiting outcome lacks durable wait evidence")
        elif selected.get("durable_wait") is not None:
            raise ValidationError("completed TaskRun outcome cannot retain a wait")
        if selected.get("previous_response_id_used") is not False:
            raise ValidationError(
                "provider previous_response_id cannot be the durable resume source"
            )
        canonical_task_run_json(selected)
        return selected

    def _outcome_replay_messages(
        self,
        outcome: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        result = outcome.get("result")
        if isinstance(result, Mapping):
            explicit = result.get("transcript_messages")
            if isinstance(explicit, list) and all(
                self._valid_replay_message(item) for item in explicit
            ):
                return [dict(item) for item in explicit]
        # Keep the replay projection provider-independent and free of the
        # original system/user snapshot.  The executor rebuilds those locally.
        role = "tool" if isinstance(result, Mapping) and result.get("tool_call_id") else "assistant"
        message: dict[str, Any] = {
            "role": role,
            "content": self._content_text(
                result if outcome["state"] == "completed" else outcome["durable_wait"]
            ),
        }
        if role == "tool":
            message["tool_call_id"] = str(result["tool_call_id"])
        return [message]

    def _bounded_transcript_projection(
        self,
        run_id: str,
        pid: str,
        messages: list[dict[str, Any]],
        *,
        prior: TaskRunResumePoint | None,
        created_at: str,
        context_generation: str,
        new_message_count: int,
    ) -> tuple[list[dict[str, Any]], TaskRunPayload | None] | None:
        limit = self.config.task_runs.payload_max_bytes

        def size(selected: list[dict[str, Any]]) -> int:
            return len(
                canonical_task_run_json(
                    {
                        "schema_version": 1,
                        "call_id": "bounded-call-id",
                        "transcript_messages": selected,
                    }
                ).encode("utf-8")
            )

        if size(messages) <= limit:
            return messages, None
        if (
            prior is None
            or type(new_message_count) is not int
            or new_message_count <= 0
            or new_message_count > len(messages)
        ):
            return None
        # Only an already committed semantic summary may replace exact prior
        # turns.  A digest or excerpt is integrity evidence, not semantics, so
        # v1 deliberately fails closed when no such summary exists.
        semantic_compaction = self._semantic_compaction_for_transcript(
            pid,
            context_generation=context_generation,
            prior=prior,
        )
        if semantic_compaction is None:
            return None
        compaction, summary, summary_sha256, summary_labels = semantic_compaction

        # Never discard any message from the just-completed action/result pair.
        retained = list(messages)
        removable = len(retained) - new_message_count
        while removable > 0 and size(retained) > limit:
            retained.pop(0)
            removable -= 1
        if size(retained) > limit:
            return None
        payload = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=run_id,
            role="summary",
            label="Validated semantic TaskRun context summary",
            value={
                "schema_version": 1,
                "summary": summary,
                "summary_sha256": summary_sha256,
                "coverage": {
                    "context_oid": compaction.get("context_oid"),
                    "context_version": compaction.get("context_version"),
                    "context_generation": context_generation,
                    "source_version": compaction.get("source_version"),
                    "source_entry_count": compaction.get("source_entry_count"),
                    "prior_safe_point_seq": prior.safe_point_seq,
                    "dropped_message_count": len(messages) - len(retained),
                },
                "data_labels": summary_labels.to_dict(),
            },
            created_at=created_at,
        )
        try:
            self._require_payload_bound(payload)
        except ValidationError:
            return None
        return retained, payload

    def _semantic_compaction_for_transcript(
        self,
        pid: str,
        *,
        context_generation: str,
        prior: TaskRunResumePoint,
    ) -> tuple[Mapping[str, Any], dict[str, Any], str, DataLabels] | None:
        context_memory = getattr(
            getattr(self._host, "llm", None),
            "context_memory",
            None,
        )
        latest_compaction = getattr(
            context_memory,
            "latest_validated_compaction",
            None,
        )
        if not callable(latest_compaction):
            return None
        try:
            compaction = latest_compaction(pid)
        except (TypeError, ValueError, ValidationError):
            return None
        if (
            not isinstance(compaction, Mapping)
            or compaction.get("schema_version") != 1
            or compaction.get("context_generation") != context_generation
            or compaction.get("compacted_at") != context_generation
            or not isinstance(compaction.get("summary"), Mapping)
            or not isinstance(compaction.get("data_labels"), Mapping)
            or self._summary_already_covers_compaction(
                prior,
                context_generation=context_generation,
            )
        ):
            return None
        summary = dict(compaction["summary"])
        summary_sha256 = hashlib.sha256(
            json.dumps(
                summary,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if compaction.get("summary_sha256") != summary_sha256:
            return None
        try:
            summary_labels = DataLabels.from_dict(compaction["data_labels"])
        except (TypeError, ValueError):
            return None
        return compaction, summary, summary_sha256, summary_labels

    def _summary_already_covers_compaction(
        self,
        prior: TaskRunResumePoint,
        *,
        context_generation: str,
    ) -> bool:
        if prior.summary_payload_id is None:
            return False
        try:
            selected = self._decode_payload(
                self._store.get_task_run_payload(prior.summary_payload_id),
                role="summary",
            )
        except (NotFound, ValidationError, TypeError, ValueError):
            return True
        coverage = selected.get("coverage")
        if not isinstance(coverage, Mapping):
            return True
        return coverage.get("context_generation") == context_generation

    def _make_resume_point(
        self,
        *,
        process: Any,
        record: TaskRunRecord,
        context_generation: str,
        safe_point_seq: int,
        transcript_payload: TaskRunPayload,
        summary_payload: TaskRunPayload | None,
        pending_payload: TaskRunPayload | None,
        last_effect_seq: int,
        created_at: str,
        updated_at: str,
    ) -> TaskRunResumePoint:
        """Build one hash-bound, provider-independent local resume point."""

        self._validate_resume_point_inputs(
            process=process,
            record=record,
            transcript_payload=transcript_payload,
            summary_payload=summary_payload,
            pending_payload=pending_payload,
            safe_point_seq=safe_point_seq,
            last_effect_seq=last_effect_seq,
        )
        image_hash, tool_hash, provider_hash = self._process_binding_hashes(process)
        binding_hash = record.binding_hash or self._binding_hash_for_process(process)
        summary_id = summary_payload.payload_id if summary_payload is not None else None
        pending_id = pending_payload.payload_id if pending_payload is not None else None
        integrity = self._sha256(
            {
                "run_id": record.run_id,
                "pid": process.pid,
                "task_run_epoch": record.runtime_epoch,
                "process_revision": process.revision,
                "context_generation": str(context_generation),
                "safe_point_seq": safe_point_seq,
                "binding_hash": binding_hash,
                "image_binding_hash": image_hash,
                "tool_binding_hash": tool_hash,
                "provider_binding_hash": provider_hash,
                "transcript_sha256": transcript_payload.sha256,
                "summary_payload_id": summary_id,
                "summary_sha256": (
                    summary_payload.sha256 if summary_payload is not None else None
                ),
                "pending_action_payload_id": pending_id,
                "pending_action_sha256": (
                    pending_payload.sha256 if pending_payload is not None else None
                ),
                "last_effect_seq": last_effect_seq,
            }
        )
        return TaskRunResumePoint(
            run_id=record.run_id,
            pid=process.pid,
            task_run_epoch=record.runtime_epoch,
            process_revision=process.revision,
            context_generation=str(context_generation),
            safe_point_seq=safe_point_seq,
            binding_hash=binding_hash,
            image_binding_hash=image_hash,
            tool_binding_hash=tool_hash,
            provider_binding_hash=provider_hash,
            transcript_payload_id=transcript_payload.payload_id,
            summary_payload_id=summary_id,
            pending_action_payload_id=pending_id,
            last_effect_seq=last_effect_seq,
            integrity_sha256=integrity,
            complete=True,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _validate_resume_point_inputs(
        self,
        *,
        process: Any,
        record: TaskRunRecord,
        transcript_payload: TaskRunPayload,
        summary_payload: TaskRunPayload | None,
        pending_payload: TaskRunPayload | None,
        safe_point_seq: int,
        last_effect_seq: int,
    ) -> None:
        if getattr(process, "task_run_id", None) != record.run_id:
            raise TaskRunRevisionConflict("process left its TaskRun before safe point")
        if getattr(process, "task_run_epoch", None) != record.runtime_epoch:
            raise TaskRunRevisionConflict("process lost its TaskRun epoch before safe point")
        if transcript_payload.run_id != record.run_id or transcript_payload.role != "transcript":
            raise ValidationError("TaskRun transcript payload binding is invalid")
        if summary_payload is not None and (
            summary_payload.run_id != record.run_id
            or summary_payload.role != "summary"
        ):
            raise ValidationError("TaskRun summary payload binding is invalid")
        if pending_payload is not None and (
            pending_payload.run_id != record.run_id
            or pending_payload.role != "pending_action"
        ):
            raise ValidationError("TaskRun pending payload binding is invalid")
        for payload in (transcript_payload, summary_payload, pending_payload):
            if payload is not None:
                self._require_payload_bound(payload)
        if type(safe_point_seq) is not int or safe_point_seq < 0:
            raise ValidationError("TaskRun safe-point sequence is invalid")
        if type(last_effect_seq) is not int or last_effect_seq < 0:
            raise ValidationError("TaskRun effect sequence is invalid")

    def _decode_pending_resume_payload(
        self,
        point: TaskRunResumePoint,
    ) -> dict[str, Any]:
        if point.pending_action_payload_id is None:
            raise NotFound("TaskRun pending resume payload is missing")
        payload = self._store.get_task_run_payload(point.pending_action_payload_id)
        if (
            payload is None
            or payload.run_id != point.run_id
            or payload.role != "pending_action"
        ):
            raise ValidationError("TaskRun pending resume payload binding is invalid")
        return self._decode_payload(payload, role="pending_action")

    def _effects_for_run(self, run_id: str) -> list[Any]:
        pids = self._member_pids(run_id)
        return self._uow.evidence.list_external_effects(pids=pids) if pids else []

    def _provider_for_effect(self, provider_name: str) -> Any | None:
        """Resolve only a Host-configured provider for receipt verification."""

        primitive = getattr(self._host, provider_name, None)
        provider = getattr(primitive, "provider", None)
        if provider is not None:
            return provider
        return getattr(self._host.substrate, provider_name, None)

    def _rewind_live_not_started_recovery_action(self, settlement: Any) -> bool:
        """Rewind only the action exactly bound to a settled live receipt.

        The caller owns the outer recovery transaction.  A negative result is
        deliberately non-exceptional: the authoritative effect settlement is
        still useful evidence, but the Run must remain in ``needs_attention``
        because no replayable local action was proven.
        """

        effect = settlement.effect
        if not self._effect_certifies_not_started(effect):
            return False
        process = self._store.get_process(effect.pid)
        if process is None:
            return False
        point = self._store.get_task_run_resume_point(
            process.pid,
            complete_only=True,
        )
        if point is None or point.pending_action_payload_id is None:
            return False
        try:
            wrapper = self._decode_pending_resume_payload(point)
        except (KeyError, NotFound, TypeError, ValueError, ValidationError):
            return False
        return self._rewind_certified_not_started_action(
            process,
            point,
            wrapper,
            required_effect_id=effect.effect_id,
            within_transaction=True,
        )

    def _apply_effect_recovery_settlement(
        self,
        record: TaskRunRecord,
        settlement: Any,
        *,
        not_started_action_rewound: bool | None = None,
    ) -> TaskRunRecord:
        """Clear only the blocker proven resolved by one fenced settlement."""

        settled_effect_id = settlement.effect.effect_id
        unresolved_ids = {
            effect.effect_id for effect in self._unsettled_effects(record.run_id)
        }
        blockers = self._remaining_effect_recovery_blockers(
            record,
            settlement,
            unresolved_ids=unresolved_ids,
            not_started_action_rewound=not_started_action_rewound,
        )
        action_blocker = self._effect_recovery_action_blocker(
            record,
            settlement,
            not_started_action_rewound=not_started_action_rewound,
        )
        if action_blocker is not None:
            blockers.append(action_blocker)

        if unresolved_ids or blockers:
            status = TaskRunStatus.NEEDS_ATTENTION
        elif record.cancel_generation > 0:
            status = TaskRunStatus.CANCELLING
        elif record.started_at is None:
            status = TaskRunStatus.QUEUED
        else:
            # Receipt recovery never authorizes implicit dispatch.  The Host
            # must issue an explicit resume/run after inspecting the new
            # revision and allowed actions.
            status = TaskRunStatus.PAUSED
        now = utc_now()
        updated = self._store.update_task_run_cas(
            record.run_id,
            record.revision,
            updates={
                "status": status,
                "blockers": tuple(blockers),
                "updated_at": now,
            },
            expected_runtime_epoch=self._runtime_epoch,
        )
        self._append_ledger(
            record.run_id,
            kind=TaskRunLedgerKind.EFFECT,
            status=settlement.provider_state,
            label="authoritative external-effect receipt settled",
            pid=settlement.effect.pid,
            effect_id=settled_effect_id,
            metadata={
                "previous_transaction_state": (
                    settlement.previous_transaction_state
                ),
                "transition_seq": settlement.transition_seq,
                "audit_record_id": settlement.audit_record_id,
            },
        )
        if status is not record.status:
            self._status_ledger(
                record,
                updated,
                label="effect recovery blocker reconciled",
            )
        return updated

    def _remaining_effect_recovery_blockers(
        self,
        record: TaskRunRecord,
        settlement: Any,
        *,
        unresolved_ids: set[str],
        not_started_action_rewound: bool | None,
    ) -> list[dict[str, Any]]:
        blockers: list[dict[str, Any]] = []
        for original in record.blockers:
            blocker = dict(original)
            if (
                not_started_action_rewound is True
                and blocker.get("kind") == "pending_action_unreplayable"
                and blocker.get("pid") == settlement.effect.pid
            ):
                # The exact PID-local action blocker is obsolete only after
                # its rewind joined the authoritative settlement transaction.
                continue
            if blocker.get("kind") not in {"unknown_effect", "effect_unsettled"}:
                blockers.append(blocker)
                continue
            identities = blocker.get("effect_ids")
            if not isinstance(identities, (list, tuple)):
                if unresolved_ids:
                    blockers.append(blocker)
                continue
            remaining = [
                str(effect_id)
                for effect_id in identities
                if str(effect_id) in unresolved_ids
            ]
            if remaining:
                blocker["effect_ids"] = remaining
                blockers.append(blocker)
        return blockers

    def _effect_recovery_action_blocker(
        self,
        record: TaskRunRecord,
        settlement: Any,
        *,
        not_started_action_rewound: bool | None,
    ) -> dict[str, Any] | None:
        # Cancellation needs settlement but never future prompt context.
        if record.cancel_generation > 0:
            return None
        effect_id = settlement.effect.effect_id
        if settlement.provider_state != "not_started":
            return self._blocker(
                "pending_action_unreplayable",
                "provider outcome is known but no complete paired local result exists",
                effect_ids=[effect_id],
            )
        if not_started_action_rewound is True:
            return None
        return self._blocker(
            "pending_action_unreplayable",
            "provider proved the effect was not started, but the exact "
            "dispatching action could not be safely rewound",
            pid=settlement.effect.pid,
            effect_ids=[effect_id],
        )

    def _effects_changed_after_resume(
        self,
        processes: Iterable[Any],
    ) -> list[Any]:
        """Find effect transitions not covered by a complete local safe point.

        A terminal provider effect is not by itself replay context.  If it was
        committed after the transcript safe point and the Runtime crashed
        before writing the paired local result, continuing from that transcript
        would dispatch it twice.  The append-only effect ledger is therefore
        compared per PID against that PID's last bound sequence.
        """

        changed_after = getattr(
            self._uow.evidence,
            "list_external_effects_changed_after",
            None,
        )
        if not callable(changed_after):
            raise RuntimeError(
                "schema-v4 Store lacks external-effect ledger resume fencing"
            )
        selected: dict[str, Any] = {}
        current_effect_seq = self._current_effect_seq()
        for process in processes:
            point = self._store.get_task_run_resume_point(
                process.pid,
                complete_only=True,
            )
            if point is not None and not self._resume_integrity_valid(
                point,
                process=process,
                check_current_binding=False,
            ):
                raise ValidationError(
                    "TaskRun effect baseline resume point failed integrity"
                )
            if (
                point is not None
                and point.last_effect_seq > current_effect_seq
            ):
                raise ValidationError(
                    "TaskRun effect baseline is ahead of append-only evidence"
                )
            baseline = point.last_effect_seq if point is not None else 0
            changed = changed_after(baseline, pids=(process.pid,))
            if len(changed) > self.config.task_runs.recovery_page_hard_limit:
                raise ValidationError(
                    "TaskRun effect settlement exceeds recovery bound"
                )
            for effect in changed:
                if self._effect_certifies_not_started(effect):
                    continue
                if (
                    point is not None
                    and self._effect_is_settled_wait_presentation(
                        effect,
                        process=process,
                        point=point,
                    )
                ):
                    continue
                selected[effect.effect_id] = effect
        return sorted(
            selected.values(),
            key=lambda item: (item.created_at, item.effect_id),
        )

    def _effect_is_settled_wait_presentation(
        self,
        effect: Any,
        *,
        process: Any,
        point: TaskRunResumePoint,
    ) -> bool:
        """Recognize a Human presentation paired to one durable wait.

        GUI presentation happens after the action executor has committed the
        Human wait safe point.  Its protected effect is safe only when the
        provider receipt is final, the matching HumanRequest is local, and the
        integrity-bound pending payload names that exact request.  Other
        post-safe-point effects remain fail-closed.
        """

        if not _is_settled_gui_presentation_effect(effect):
            return False
        if point.pending_action_payload_id is None:
            return False
        if not self._resume_integrity_valid(point):
            return False
        request_id = _gui_presentation_request_id(effect)
        if request_id is None:
            return False
        if not self._presentation_request_matches(
            effect,
            request_id=request_id,
            pid=process.pid,
        ):
            return False
        try:
            wrapper = self._decode_pending_resume_payload(point)
        except (KeyError, NotFound, TypeError, ValueError, ValidationError):
            return False
        return _matches_durable_human_wait(
            wrapper,
            request_id=request_id,
            pid=process.pid,
        )

    @staticmethod
    def _effect_certifies_not_started(effect: Any) -> bool:
        # Provider flags are supporting evidence, never a substitute for the
        # durable effect state.  In particular, a committed effect carrying a
        # contradictory legacy/custom ``certified_not_started`` flag must
        # remain behind the changed-effect fence.
        if (
            getattr(effect, "effect_state", None) != "finalized"
            or getattr(effect, "transaction_state", None) != "failed"
        ):
            return False
        receipt = getattr(effect, "provider_receipt", None)
        metadata = getattr(effect, "provider_metadata", None)
        if isinstance(metadata, Mapping) and (
            metadata.get("provider_reconciliation_state") == "not_started"
            and metadata.get("certified_not_started") is True
        ):
            return True
        if not isinstance(receipt, Mapping):
            return False
        return bool(
            (
                receipt.get("dispatch_status") == "not_started"
                and receipt.get("certified") is True
            )
            or receipt.get("certified_not_started") is True
        )

    def _unsafe_effects(self, run_id: str) -> list[Any]:
        return [
            effect
            for effect in self._effects_for_run(run_id)
            if effect.transaction_state in _UNKNOWN_EFFECT_STATES
        ]

    def _unsettled_effects(self, run_id: str) -> list[Any]:
        """Return every effect that still forbids dispatch or terminalization.

        ``dispatched``/``unknown`` are the irreducibly ambiguous subset used by
        manual recovery.  Earlier states are also unsettled, however: allowing
        a Run to dispatch or report a terminal result while an authorized or
        approved intent remains open would break the same at-most-once fence.
        """

        return [
            effect
            for effect in self._effects_for_run(run_id)
            if not (
                effect.effect_state == "finalized"
                and effect.transaction_state in _SETTLED_EFFECT_STATES
            )
        ]

    def _current_effect_seq(self) -> int:
        method = getattr(self._uow.evidence, "current_effect_ledger_seq", None)
        return int(method()) if callable(method) else 0

    def _recover_task_run_resume_state(
        self,
        process: Any,
    ) -> dict[str, Any] | None:
        """Settle or validate the TaskRun-owned pending resume payload.

        This runs before the generic changed-effect fence.  The only write it
        may perform is a provider-free commit of an already staged complete
        result, or a narrowly certified not-started rewind.
        """

        point = self._store.get_task_run_resume_point(
            process.pid,
            complete_only=True,
        )
        if point is None or point.pending_action_payload_id is None:
            return None
        if not self._resume_integrity_valid(point):
            return self._blocker(
                "payload_corrupt",
                "TaskRun pending resume point failed integrity",
                pid=process.pid,
            )
        try:
            wrapper = self._decode_pending_resume_payload(point)
            kind = wrapper.get("kind")
            if kind == "completed_outcome":
                self._recover_staged_completed_outcome(process, wrapper)
                return None
            if kind == "durable_wait_action":
                self._validate_recovered_durable_wait(process, wrapper)
                return None
            if kind != "validated_action":
                raise ValidationError("unsupported TaskRun pending resume kind")
            manifest = normalize_validated_action_manifest(wrapper["manifest"])
            if (
                wrapper.get("call_id") != manifest["call_id"]
                or wrapper.get("manifest_sha256") != self._sha256(manifest)
                or wrapper.get("llm_call_sha256")
                != self._local_llm_call_sha256(process, str(manifest["call_id"]))
            ):
                raise ValidationError("TaskRun validated action binding changed")
            self._validated_action_pre_binding(wrapper, point)
            state = wrapper.get("state")
            if state == "validated":
                return None
            if state == "dispatching" and self._rewind_certified_not_started_action(
                process,
                point,
                wrapper,
            ):
                return None
            if state == "dispatching":
                changed_effects = self._changed_effects_for_pid(
                    process.pid,
                    point.last_effect_seq,
                )
                if changed_effects:
                    return self._blocker(
                        "unknown_effect",
                        "validated action reached a provider boundary without a complete local result",
                        pid=process.pid,
                        effect_ids=[
                            effect.effect_id for effect in changed_effects[:20]
                        ],
                    )
                return self._blocker(
                    "pending_action_unreplayable",
                    "validated action dispatch outcome is unknown after restart",
                    pid=process.pid,
                )
            raise ValidationError("TaskRun validated action state is invalid")
        except (KeyError, NotFound, TypeError, ValueError, RuntimeError, ValidationError):
            return self._blocker(
                "pending_action_unreplayable",
                "TaskRun pending resume payload cannot be recovered locally",
                pid=process.pid,
            )

    def _recover_staged_completed_outcome(
        self,
        process: Any,
        wrapper: Mapping[str, Any],
    ) -> None:
        if wrapper.get("state") != "staged":
            raise ValidationError("TaskRun staged outcome state is invalid")
        call_id = self._identifier(wrapper.get("call_id"), "LLM call_id")
        outcome = self._validated_outcome_manifest(wrapper.get("outcome"))
        if (
            wrapper.get("outcome_sha256") != self._sha256(outcome)
            or wrapper.get("llm_call_sha256")
            != self._local_llm_call_sha256(process, call_id)
        ):
            raise ValidationError("TaskRun staged result binding changed")
        point = self._store.get_task_run_resume_point(
            process.pid,
            complete_only=True,
        )
        if point is None:
            raise ValidationError("TaskRun staged result safe point is missing")
        self._validate_staged_binding_transition(
            process=process,
            point=point,
            staged=wrapper,
        )
        self._validate_effect_settlement_bundle(process.pid, wrapper)
        self.record_completed_transcript(
            pid=process.pid,
            call_id=call_id,
            outcome_manifest=outcome,
            context_generation=str(wrapper.get("context_generation")),
        )

    def _validate_recovered_durable_wait(
        self,
        process: Any,
        wrapper: Mapping[str, Any],
    ) -> None:
        call_id = self._identifier(wrapper.get("call_id"), "LLM call_id")
        pending_service = getattr(getattr(self._host, "llm", None), "pending", None)
        get_pending = getattr(pending_service, "get", None)
        current = self._durable_wait_snapshot(
            process.pid,
            get_pending(process.pid) if callable(get_pending) else None,
            call_id=call_id,
        )
        snapshot = wrapper.get("wait_snapshot")
        if (
            wrapper.get("state") != "waiting"
            or current.get("status") != "pending"
            or not isinstance(snapshot, Mapping)
            or current != dict(snapshot)
            or wrapper.get("wait_snapshot_sha256") != self._sha256(current)
            or wrapper.get("wait_identity_sha256")
            != self._durable_wait_identity_sha256(current)
            or wrapper.get("llm_call_sha256")
            != self._local_llm_call_sha256(process, call_id)
        ):
            raise ValidationError("TaskRun durable wait snapshot changed")

    def _rewind_certified_not_started_action(
        self,
        process: Any,
        point: TaskRunResumePoint,
        wrapper: Mapping[str, Any],
        *,
        required_effect_id: str | None = None,
        within_transaction: bool = False,
    ) -> bool:
        if not self._rewind_action_payload_matches(process, point, wrapper):
            return False
        changed = self._changed_effects_for_pid(process.pid, point.last_effect_seq)
        expected_binding = {
            "run_id": point.run_id,
            "pid": process.pid,
            "call_id": wrapper.get("call_id"),
            "context_generation": wrapper.get("context_generation"),
            "action_manifest_sha256": wrapper.get("manifest_sha256"),
            "source_safe_point_seq": point.safe_point_seq,
        }
        if not self._rewind_effects_match_action(
            changed,
            expected_binding=expected_binding,
            required_effect_id=required_effect_id,
        ):
            return False
        record = self._require_run(point.run_id)
        now = utc_now()
        pending = TaskRunPayload.plaintext(
            payload_id=new_id("trp"),
            run_id=point.run_id,
            role="pending_action",
            label="Provider-certified not-started TaskRun action",
            value={**dict(wrapper), "state": "validated"},
            created_at=now,
        )
        self._require_payload_bound(pending)
        transcript = self._store.get_task_run_payload(point.transcript_payload_id)
        summary = (
            self._store.get_task_run_payload(point.summary_payload_id)
            if point.summary_payload_id is not None
            else None
        )
        if transcript is None:
            raise ValidationError("TaskRun action transcript is missing")
        rewound = self._make_resume_point(
            process=process,
            record=record,
            context_generation=point.context_generation,
            safe_point_seq=point.safe_point_seq + 1,
            transcript_payload=transcript,
            summary_payload=summary,
            pending_payload=pending,
            last_effect_seq=self._current_effect_seq(),
            created_at=point.created_at,
            updated_at=now,
        )
        def persist_rewind() -> None:
            self._store.insert_task_run_payload(pending)
            self._store.upsert_task_run_resume_point(rewound)
            self._append_ledger(
                point.run_id,
                kind=TaskRunLedgerKind.LLM_TURN,
                status="validated",
                label="provider certified pending action was not started",
                pid=process.pid,
                llm_call_id=str(wrapper.get("call_id")),
                payload_id=pending.payload_id,
                metadata={"safe_point_seq": rewound.safe_point_seq},
            )
        if within_transaction:
            persist_rewind()
        else:
            with self._uow.transaction():
                persist_rewind()
        return True

    def _rewind_action_payload_matches(
        self,
        process: Any,
        point: TaskRunResumePoint,
        wrapper: Mapping[str, Any],
    ) -> bool:
        if (
            point.pid != process.pid
            or point.pending_action_payload_id is None
            or not self._resume_integrity_valid(point)
            or wrapper.get("kind") != "validated_action"
            or wrapper.get("state") != "dispatching"
        ):
            return False
        try:
            manifest = normalize_validated_action_manifest(wrapper["manifest"])
            if (
                wrapper.get("call_id") != manifest["call_id"]
                or wrapper.get("manifest_sha256") != self._sha256(manifest)
                or wrapper.get("llm_call_sha256")
                != self._local_llm_call_sha256(
                    process,
                    str(manifest["call_id"]),
                )
            ):
                return False
            self._validated_action_pre_binding(wrapper, point)
        except (
            KeyError,
            NotFound,
            TaskRunRevisionConflict,
            TypeError,
            ValueError,
            ValidationError,
        ):
            return False
        return True

    def _rewind_effects_match_action(
        self,
        changed: list[Any],
        *,
        expected_binding: Mapping[str, Any],
        required_effect_id: str | None,
    ) -> bool:
        if not changed:
            return False
        if required_effect_id is not None and required_effect_id not in {
            effect.effect_id for effect in changed
        }:
            return False
        return all(
            self._effect_certifies_not_started(effect)
            and effect.provider_metadata.get("task_run_action")
            == expected_binding
            for effect in changed
        )

    def _pending_action_recovery_blocker(
        self,
        process: Any,
        point: TaskRunResumePoint | None,
    ) -> dict[str, Any] | None:
        """Validate one durable pending action without claiming or dispatching it."""

        pending_service = getattr(getattr(self._host, "llm", None), "pending", None)
        get_pending = getattr(pending_service, "get", None)
        if not callable(get_pending):
            return self._blocker(
                "pending_action_unreplayable",
                "durable pending-action validator is unavailable",
                pid=process.pid,
            )
        try:
            pending = get_pending(process.pid)
        except (TypeError, ValueError, RuntimeError, ValidationError):
            return self._blocker(
                "pending_action_unreplayable",
                "durable pending action failed persisted-row validation",
                pid=process.pid,
            )
        if pending is None:
            return None
        try:
            if not isinstance(pending, dict):
                raise RuntimeError("pending action is not an object")
            if pending.get("status") not in {"pending", "completed"}:
                raise RuntimeError("pending action is in an unsafe resume state")
            wait_type = pending.get("wait_type")
            if wait_type not in {"llm_release", "human", "child", "message"}:
                raise RuntimeError("pending action wait type is invalid")
            action = pending.get("action")
            filters = pending.get("filters")
            if not isinstance(action, dict) or not action:
                raise RuntimeError("pending action body is missing")
            if not isinstance(filters, dict):
                raise RuntimeError("pending action filters are invalid")
            canonical_task_run_json(action)
            pending_resume_token(pending)
            pending_metadata(pending)
            if wait_type == "message":
                pending_message_filters(pending)
            if wait_type in {"human", "llm_release"} and not pending.get(
                "request_id"
            ):
                raise RuntimeError("pending Human request reference is missing")
            if wait_type == "child" and not pending.get("child_pid"):
                raise RuntimeError("pending child process reference is missing")
            transcript_call_id = pending_task_run_transcript_call_id(pending)
            if transcript_call_id is None:
                raise RuntimeError("TaskRun transcript call reference is missing")
            if point is None or not self._resume_integrity_valid(point):
                raise RuntimeError("pending action has no complete safe point")
            transcript = self._decode_payload(
                self._store.get_task_run_payload(point.transcript_payload_id),
                role="transcript",
            )
            if transcript.get("call_id") != transcript_call_id:
                raise RuntimeError("pending action is bound to another transcript")
        except (TypeError, ValueError, RuntimeError, ValidationError, NotFound):
            return self._blocker(
                "pending_action_unreplayable",
                "durable pending action cannot be replayed from local evidence",
                pid=process.pid,
            )
        return None

    def _authority_recovery_blocker(self, process: Any) -> dict[str, Any] | None:
        """Read-only liveness check for authority needed by a live process."""

        if process.status in _TERMINAL_PROCESS_STATUSES:
            return None
        try:
            manifests = self._host.authority_manifests
            manifest = manifests.get_for_process(process.pid)
            if manifest is None:
                raise CapabilityDenied("process authority manifest is missing")
            # This public helper performs the manifest expiry check without
            # authorizing, reserving, consuming, or creating a Human request.
            manifests.bound_capability_expiry(process.pid, None)
            required_authorized = [
                required
                for required in manifest.required_capabilities
                if any(
                    self._host.capability.spec_covers(declared, required)
                    for declared in manifest.authorized_capabilities
                )
            ]
            if required_authorized:
                capabilities = self._host.capability.capabilities_for(process.pid)
                active = [
                    capability
                    for capability in capabilities
                    if capability.active
                    and capability.effect is CapabilityEffect.ALLOW
                    and not self._host.capability.is_expired(capability)
                    and self._host.capability.parent_chain_active(capability)
                ]
                for required in required_authorized:
                    if not any(
                        self._host.capability.spec_covers(capability, required)
                        for capability in active
                    ):
                        raise CapabilityDenied(
                            "required process capability is no longer active"
                        )
        except (CapabilityDenied, NotFound, TypeError, ValueError, ValidationError):
            return self._blocker(
                "authority_revoked",
                "required TaskRun authority expired, was revoked, or is invalid",
                pid=process.pid,
            )
        return None

    def _bound_process_run(
        self,
        pid: str,
    ) -> tuple[Any | None, TaskRunRecord | None]:
        process = self._store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        run_id = getattr(process, "task_run_id", None)
        return process, self._require_run(run_id) if run_id is not None else None

    def _member_pids(self, run_id: str) -> tuple[str, ...]:
        pids = tuple(self._store.list_task_run_process_ids(run_id))
        if len(pids) > self.config.task_runs.recovery_page_hard_limit:
            raise ValidationError("TaskRun process tree exceeds recovery hard limit")
        return pids

    def _tree_processes(self, run_id: str) -> list[Any]:
        records = list(self._store.list_processes_for_task_run(run_id))
        if len(records) > self.config.task_runs.recovery_page_hard_limit:
            raise ValidationError("TaskRun process tree exceeds recovery hard limit")
        depths: dict[str, int] = {}
        by_id = {item.pid: item for item in records}
        for process in records:
            depth = 0
            parent = process.parent_pid
            seen: set[str] = set()
            while parent in by_id and parent not in seen:
                seen.add(parent)
                depth += 1
                parent = by_id[parent].parent_pid
            depths[process.pid] = depth
        return sorted(records, key=lambda item: (depths[item.pid], item.created_at, item.pid))

    @staticmethod
    def _active_pid(processes: Iterable[Any]) -> str | None:
        selected = list(processes)
        if not selected:
            return None
        selected.sort(key=lambda item: (item.created_at, item.pid))
        return selected[-1].pid

    def _abandoned_object_tasks(self, run_id: str) -> list[str]:
        pids = self._member_pids(run_id)
        if not pids:
            return []
        selected: set[str] = set()
        hard_limit = self.config.task_runs.recovery_page_hard_limit
        for pid in pids:
            tasks = self._store.list_object_tasks(
                creator_pid=pid,
                statuses=(ObjectTaskStatus.ABANDONED,),
                include_terminal=True,
                limit=hard_limit + 1,
            )
            for task in tasks:
                selected.add(task.task_id)
                if len(selected) > hard_limit:
                    # One sentinel keeps the public blocker bounded while still
                    # failing closed on an adversarially large per-Run set.
                    return [*sorted(selected)[:hard_limit], "__truncated__"]
        return sorted(selected)

    def _deadline_expired(self, record: TaskRunRecord) -> bool:
        if record.deadline_at is None:
            return False
        deadline = datetime.fromisoformat(record.deadline_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) >= deadline.astimezone(timezone.utc)

    def _reconcile_deadline(self, record: TaskRunRecord) -> TaskRunRecord:
        return self._expire(record) if self._deadline_expired(record) else record

    def _converge_expired_deadlines(self) -> None:
        """Persist elapsed deadline intent before applying public status filters."""

        cursor: TaskRunCursor | None = None
        while True:
            page = self._store.list_recoverable_task_runs(
                after=cursor,
                limit=self.config.task_runs.recovery_page_size,
            )
            for record in page.records:
                if self._deadline_expired(record):
                    self._expire(record)
            cursor = page.next_cursor
            if cursor is None:
                return

    def _expire(
        self,
        record: TaskRunRecord,
        *,
        command_id: str | None = None,
        command_kind: str = "deadline",
        command_request: Mapping[str, Any] | None = None,
    ) -> TaskRunRecord:
        with self._condition:
            command_recorded = False
            selected_command_request = dict(command_request or {})

            def record_deadline_command(
                before: TaskRunRecord,
                after: TaskRunRecord,
            ) -> bool:
                if command_id is None:
                    return False
                admission = self._append_control_admission(
                    before,
                    after,
                    command_id=command_id,
                    command_kind=command_kind,
                    request=selected_command_request,
                    evidence={
                        "kind": "deadline",
                        "cancel_generation": after.cancel_generation,
                    },
                    label="deadline command admitted",
                )
                self._record_command(
                    after,
                    command_id,
                    command_kind,
                    selected_command_request,
                    result={
                        "settlement_state": "pending",
                        "settlement_kind": "deadline",
                        "cancel_generation": after.cancel_generation,
                        **admission,
                    },
                )
                return True

            blocker = self._blocker(
                "deadline_reached",
                "absolute TaskRun deadline elapsed",
            )
            for _attempt in range(16):
                current = self._require_run(record.run_id)
                if current.runtime_epoch != self._runtime_epoch:
                    raise TaskRunRevisionConflict(
                        f"stale TaskRun epoch refused deadline: {current.run_id}"
                    )
                if current.status in TASK_RUN_TERMINAL_STATUSES:
                    self._record_optional_command(
                        current,
                        command_id,
                        command_kind,
                        selected_command_request,
                    )
                    return current
                if (
                    current.status is TaskRunStatus.CANCELLING
                    or current.cancel_generation > 0
                ):
                    cancelling = current
                    break
                try:
                    with self._uow.transaction():
                        cancelling = self._store.update_task_run_cas(
                            current.run_id,
                            current.revision,
                            updates={
                                "status": TaskRunStatus.CANCELLING,
                                "cancel_generation": current.cancel_generation + 1,
                                "blockers": (blocker,),
                                "updated_at": utc_now(),
                            },
                            expected_runtime_epoch=self._runtime_epoch,
                        )
                        self._status_ledger(
                            current,
                            cancelling,
                            label="deadline cancellation intent persisted",
                        )
                        command_recorded = record_deadline_command(
                            current,
                            cancelling,
                        )
                    break
                except TaskRunRevisionConflict:
                    # An already-admitted local settlement may advance the Run
                    # revision without taking the control condition.  Reread
                    # and converge the single deadline generation instead of
                    # surfacing an internal housekeeping race to observers.
                    continue
            else:
                raise TaskRunRevisionConflict(
                    f"TaskRun deadline intent did not converge for {record.run_id}"
                )
            if not command_recorded:
                # A concurrent control command already persisted cancellation;
                # this command owns no state transition, but still needs its own
                # exact idempotency receipt.
                with self._uow.transaction():
                    command_recorded = record_deadline_command(
                        cancelling,
                        cancelling,
                    )
            active_external = (
                self._active_external_dispatches.get(record.run_id, 0) > 0
            )
            self._condition.notify_all()
        unsettled = self._unsettled_effects(record.run_id)
        if unsettled:
            return self._mark_attention(
                cancelling,
                self._blocker(
                    (
                        "unknown_effect"
                        if any(
                            effect.transaction_state in _UNKNOWN_EFFECT_STATES
                            for effect in unsettled
                        )
                        else "effect_unsettled"
                    ),
                    "deadline elapsed while an effect remained unsettled",
                    effect_ids=[effect.effect_id for effect in unsettled[:20]],
                ),
            )
        if active_external:
            return self._mark_attention(
                cancelling,
                self._blocker(
                    "effect_unsettled",
                    "deadline awaits an already-admitted call's local settlement",
                ),
            )
        for process in reversed(self._tree_processes(record.run_id)):
            if process.status not in _TERMINAL_PROCESS_STATUSES:
                self._process.cancel(process.pid, "TaskRun deadline elapsed")
        return self._project(
            self._require_run(record.run_id),
            allow_finalize=True,
        )

    def _binding_hash(
        self,
        *,
        image_id: str,
        launch_options: Mapping[str, Any],
        authority_manifest_id: str | None,
    ) -> str:
        image = self._host.images.get(image_id)
        if image is None:
            raise NotFound(f"TaskRun image not found: {image_id}")
        image_projection = to_jsonable(image)
        authority_hash = None
        if authority_manifest_id is not None:
            authority_hash = self._host.authority_manifests.get(
                authority_manifest_id
            ).manifest_hash
        profile_id = str(
            launch_options.get("llm_profile_id")
            or getattr(image, "llm_profile_id", None)
            or self.config.llm.default_profile_id
        )
        provider_projection = {
            "llm_profile_id": profile_id,
            "llm_profile_identity_sha256": (
                self._host.llms.profile_identity_sha256(profile_id)
            ),
            "substrate_type": type(self._host.substrate).__qualname__,
        }
        return self._sha256(
            {
                "schema_version": 1,
                "image": image_projection,
                "launch_options": dict(launch_options),
                "authority_manifest_hash": authority_hash,
                "provider_projection": provider_projection,
            }
        )

    def _process_binding_hashes(self, process: Any) -> tuple[str, str, str]:
        image = self._host.images.get(process.image_id)
        image_hash = self._sha256(to_jsonable(image))
        tool_hash = tool_binding_hash(process)
        provider_hash = self._sha256(
            {
                "profile_id": process.llm_profile_id,
                "profile_identity_sha256": (
                    self._host.llms.profile_identity_sha256(
                        process.llm_profile_id
                    )
                ),
                "substrate_type": type(self._host.substrate).__qualname__,
            }
        )
        return image_hash, tool_hash, provider_hash

    def _require_pre_action_binding_matches_process(
        self,
        value: Any,
        process: Any,
    ) -> dict[str, Any]:
        image_hash, current_tool_hash, provider_hash = self._process_binding_hashes(
            process
        )
        selected = validate_pre_action_binding(
            value,
            image_binding_hash=image_hash,
            tool_binding_hash=current_tool_hash,
            provider_binding_hash=provider_hash,
            process_revision=process.revision,
        )
        expected = pre_action_binding(
            process,
            image_binding_hash=image_hash,
            provider_binding_hash=provider_hash,
        )
        if selected != expected:
            raise ValidationError("TaskRun pre-action process binding changed")
        return selected

    def _validated_action_pre_binding(
        self,
        wrapper: Mapping[str, Any],
        point: TaskRunResumePoint,
        *,
        require_point_revision: bool = False,
    ) -> dict[str, Any]:
        return validate_pre_action_binding(
            wrapper.get("pre_action_binding"),
            image_binding_hash=point.image_binding_hash,
            tool_binding_hash=point.tool_binding_hash,
            provider_binding_hash=point.provider_binding_hash,
            process_revision=(
                point.process_revision if require_point_revision else None
            ),
        )

    def _binding_hash_for_process(self, process: Any) -> str:
        image, tools, provider = self._process_binding_hashes(process)
        return self._sha256(
            {"image": image, "tools": tools, "provider": provider}
        )

    def _resume_integrity_valid(
        self,
        point: TaskRunResumePoint,
        *,
        check_current_binding: bool = True,
        record: TaskRunRecord | None = None,
        process: Any | None = None,
    ) -> bool:
        if not self._resume_point_identity_valid(
            point,
            record=record,
            process=process,
        ) or not self._resume_static_integrity_valid(point):
            return False
        if not check_current_binding:
            return True
        return self._resume_current_binding_valid(
            point,
            record=record,
            process=process,
        )

    def _resume_point_identity_valid(
        self,
        point: TaskRunResumePoint,
        *,
        record: TaskRunRecord | None = None,
        process: Any | None = None,
        require_current_runtime: bool = True,
    ) -> bool:
        """Bind a resume point to one Run, process, and monotonic epoch.

        A safe point may legitimately predate several Runtime reopens, but it
        can never come from another Run/process, claim a future epoch, or name
        a process revision that has not happened.  Startup prevalidation runs
        before the current Runtime claims the Run epoch, so that phase opts out
        only from the current-writer equality while retaining every persisted
        identity and monotonicity check.
        """

        selected_record = record or self._store.get_task_run(point.run_id)
        selected_process = process or self._store.get_process(point.pid)
        if selected_record is None or selected_process is None:
            return False
        if (
            point.run_id != selected_record.run_id
            or point.pid != selected_process.pid
            or selected_process.task_run_id != selected_record.run_id
            or selected_process.task_run_epoch != selected_record.runtime_epoch
            or point.task_run_epoch > selected_record.runtime_epoch
            or point.process_revision > selected_process.revision
            or point.binding_hash != selected_record.binding_hash
        ):
            return False
        if (
            require_current_runtime
            and selected_record.runtime_epoch != self._runtime_epoch
        ):
            return False
        return True

    def _resume_static_integrity_valid(
        self,
        point: TaskRunResumePoint,
    ) -> bool:
        """Validate only persisted bytes and their integrity envelope.

        Startup invokes this before Image/provider/tool publication recovery.
        It must consequently remain independent of every live registry and
        provider lookup.  Dynamic binding comparison is a later recovery
        phase performed by :meth:`_resume_current_binding_valid`.
        """

        payload = self._store.get_task_run_payload(point.transcript_payload_id)
        if (
            payload is None
            or payload.run_id != point.run_id
            or payload.role != "transcript"
            or payload.retention_state
            is not TaskRunPayloadRetention.PLAINTEXT
        ):
            return False
        summary = (
            self._store.get_task_run_payload(point.summary_payload_id)
            if point.summary_payload_id is not None
            else None
        )
        if point.summary_payload_id is not None and (
            summary is None
            or summary.run_id != point.run_id
            or summary.role != "summary"
            or summary.retention_state is not TaskRunPayloadRetention.PLAINTEXT
        ):
            return False
        pending_action = (
            self._store.get_task_run_payload(point.pending_action_payload_id)
            if point.pending_action_payload_id is not None
            else None
        )
        if point.pending_action_payload_id is not None and (
            pending_action is None
            or pending_action.run_id != point.run_id
            or pending_action.role != "pending_action"
            or pending_action.retention_state
            is not TaskRunPayloadRetention.PLAINTEXT
        ):
            return False
        expected = self._sha256(
            {
                "run_id": point.run_id,
                "pid": point.pid,
                "task_run_epoch": point.task_run_epoch,
                "process_revision": point.process_revision,
                "context_generation": point.context_generation,
                "safe_point_seq": point.safe_point_seq,
                "binding_hash": point.binding_hash,
                "image_binding_hash": point.image_binding_hash,
                "tool_binding_hash": point.tool_binding_hash,
                "provider_binding_hash": point.provider_binding_hash,
                "transcript_sha256": payload.sha256,
                "summary_payload_id": point.summary_payload_id,
                "summary_sha256": summary.sha256 if summary is not None else None,
                "pending_action_payload_id": point.pending_action_payload_id,
                "pending_action_sha256": (
                    pending_action.sha256 if pending_action is not None else None
                ),
                "last_effect_seq": point.last_effect_seq,
            }
        )
        if expected != point.integrity_sha256:
            return False
        return True

    def _resume_current_binding_valid(
        self,
        point: TaskRunResumePoint,
        *,
        record: TaskRunRecord | None = None,
        process: Any | None = None,
    ) -> bool:
        """Compare a statically valid safe point with recovered live bindings."""

        selected_process = process or self._store.get_process(point.pid)
        if selected_process is None or not self._resume_point_identity_valid(
            point,
            record=record,
            process=selected_process,
        ):
            return False
        current = self._process_binding_hashes(selected_process)
        return current == (
            point.image_binding_hash,
            point.tool_binding_hash,
            point.provider_binding_hash,
        )

    def _launch_kwargs(self, spec: TaskRunSpecV1) -> dict[str, Any]:
        options = dict(spec.launch_options)
        unknown = sorted(set(options) - _LAUNCH_OPTION_FIELDS)
        if unknown:
            raise ValidationError(f"unknown TaskRun launch options: {unknown}")
        # Durable Runs persist only the Host-managed authority reference.  An
        # inline manifest could contain credentials or other authority-bearing
        # material in ``task_runs.launch_options_json``, which is intentionally
        # retained even when payloads are purged.
        options["authority_manifest"] = spec.authority_manifest_id
        return options

    def _spawn_root(
        self,
        *,
        run_id: str,
        epoch: int,
        image_id: str,
        marker: Mapping[str, Any],
        launch: Mapping[str, Any],
        authority_manifest_id: str | None,
        task_run_commit: Callable[[str, str, str, str], None],
    ) -> str:
        kwargs = dict(launch)
        authority = kwargs.pop("authority_manifest", None)
        return self._process.spawn(
            image=image_id,
            goal=dict(marker),
            capabilities=kwargs.pop("capabilities", None),
            resource_budget=kwargs.pop("resource_budget", None),
            working_directory=kwargs.pop("working_directory", None),
            llm_profile_id=kwargs.pop("llm_profile_id", None),
            authority_manifest=authority,
            _task_run_id=run_id,
            _task_run_epoch=epoch,
            _task_run_role="root",
            _task_run_commit=task_run_commit,
        )

    def _payloads_retained(self, run_id: str) -> bool:
        try:
            return (
                self._payload_by_role(run_id, "goal").retention_state
                is TaskRunPayloadRetention.PLAINTEXT
            )
        except (NotFound, TypeError, ValueError, ValidationError):
            # A corrupt durable row is itself recovery evidence.  Summary
            # projection must remain available and must fail closed instead of
            # decoding the same damaged row a second time during startup.
            return False

    def _supersede_unstarted_actions_for_interrupt(self, run_id: str) -> None:
        """Discard only validated actions proven not to have reached a tool.

        An interrupt changes the requirement generation, so an action selected
        from the old prompt must not run on the next explicit dispatch.  The
        prior transcript remains the replay base; only its separate pending
        action projection is removed.  Any changed effect fails closed instead
        of being called "not started" by the Runtime.
        """

        if self._require_run(run_id).status is TaskRunStatus.NEEDS_ATTENTION:
            return
        for process in self._tree_processes(run_id):
            point = self._store.get_task_run_resume_point(
                process.pid,
                complete_only=True,
            )
            if point is None or point.pending_action_payload_id is None:
                continue
            if not self._resume_integrity_valid(point):
                self._mark_attention(
                    self._require_run(run_id),
                    self._blocker(
                        "payload_corrupt",
                        "interrupt could not validate the pending local action",
                        pid=process.pid,
                    ),
                )
                return
            wrapper = self._decode_pending_resume_payload(point)
            if wrapper.get("kind") != "validated_action":
                continue
            if wrapper.get("state") == "dispatching":
                self.defer_unstarted_action_for_pid(process.pid)
                point = self._store.get_task_run_resume_point(
                    process.pid,
                    complete_only=True,
                )
                if point is None or point.pending_action_payload_id is None:
                    continue
                wrapper = self._decode_pending_resume_payload(point)
            if wrapper.get("state") != "validated":
                self._mark_attention(
                    self._require_run(run_id),
                    self._blocker(
                        "pending_action_unreplayable",
                        "interrupt encountered an action with unknown dispatch state",
                        pid=process.pid,
                    ),
                )
                return
            if self._changed_effects_for_pid(process.pid, point.last_effect_seq):
                self._mark_attention(
                    self._require_run(run_id),
                    self._blocker(
                        "unknown_effect",
                        "interrupt found effect evidence after the pending action",
                        pid=process.pid,
                    ),
                )
                return
            current_process = self._store.get_process(process.pid)
            current_record = self._require_run(run_id)
            if current_process is None:
                raise NotFound(f"process not found: {process.pid}")
            transcript = self._store.get_task_run_payload(
                point.transcript_payload_id
            )
            summary = (
                self._store.get_task_run_payload(point.summary_payload_id)
                if point.summary_payload_id is not None
                else None
            )
            if transcript is None:
                raise ValidationError("TaskRun interrupt transcript is missing")
            now = utc_now()
            superseded = self._make_resume_point(
                process=current_process,
                record=current_record,
                context_generation=point.context_generation,
                safe_point_seq=point.safe_point_seq + 1,
                transcript_payload=transcript,
                summary_payload=summary,
                pending_payload=None,
                last_effect_seq=self._current_effect_seq(),
                created_at=point.created_at,
                updated_at=now,
            )
            with self._uow.transaction():
                self._store.upsert_task_run_resume_point(superseded)
                self._append_ledger(
                    run_id,
                    kind=TaskRunLedgerKind.LLM_TURN,
                    status="superseded",
                    label="interrupt superseded an unstarted old-generation action",
                    pid=process.pid,
                    llm_call_id=str(wrapper.get("call_id")),
                    metadata={"safe_point_seq": superseded.safe_point_seq},
                )

    @contextmanager
    def _dispatch(self, run_id: str, *, pause_generation: int):
        if getattr(self._dispatch_scope, "admission", None) is not None:
            raise ValidationError("nested TaskRun dispatch scopes are not allowed")
        with self._condition:
            current = self._store.get_task_run(run_id)
            admitted = bool(
                current is not None
                and current.runtime_epoch == self._runtime_epoch
                and current.status is TaskRunStatus.RUNNING
                and current.pause_generation == pause_generation
            )
            if admitted:
                self._active_run_dispatches[run_id] = (
                    self._active_run_dispatches.get(run_id, 0) + 1
                )
        self._dispatch_scope.admission = (
            (run_id, pause_generation) if admitted else None
        )
        try:
            yield admitted
        finally:
            self._dispatch_scope.admission = None
            if admitted:
                with self._condition:
                    remaining = self._active_run_dispatches.get(run_id, 0) - 1
                    if remaining > 0:
                        self._active_run_dispatches[run_id] = remaining
                    else:
                        self._active_run_dispatches.pop(run_id, None)
                    self._condition.notify_all()

    def _notify_updated(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _wait_for_dispatch_drain(self, run_id: str) -> None:
        """Wait until every pre-control run/call scope has settled locally."""

        with self._condition:
            while (
                self._active_run_dispatches.get(run_id, 0) > 0
                or self._active_external_dispatches.get(run_id, 0) > 0
            ):
                self._condition.wait(timeout=0.25)

    def _begin_control_mutation(self, run_id: str, command_id: str) -> None:
        key = (
            self._identifier(run_id, "run_id"),
            self._identifier(command_id, "command_id"),
        )
        with self._condition:
            while key in self._control_mutations:
                self._condition.wait(timeout=0.25)
            self._control_mutations.add(key)

    def _end_control_mutation(self, run_id: str, command_id: str) -> None:
        key = (run_id, command_id)
        with self._condition:
            self._control_mutations.discard(key)
            self._condition.notify_all()

    @staticmethod
    def _normalize_pids(pids: Iterable[str]) -> tuple[str, ...]:
        if isinstance(pids, (str, bytes, bytearray)):
            raise ValidationError("TaskRun PID scope must be an iterable")
        selected = tuple(dict.fromkeys(str(pid) for pid in pids))
        if any(not pid or pid != pid.strip() or "\x00" in pid for pid in selected):
            raise ValidationError("TaskRun PID scope contains an invalid PID")
        return selected

    @staticmethod
    def _encode_opaque_cursor(value: Mapping[str, Any]) -> str:
        encoded = canonical_task_run_json(dict(value)).encode("utf-8")
        return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_opaque_cursor(value: str | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value or len(value) > 2048:
            raise ValidationError("TaskRun cursor is invalid")
        try:
            padded = value + "=" * (-len(value) % 4)
            decoded = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise ValidationError("TaskRun cursor is invalid") from exc
        if not isinstance(decoded, dict):
            raise ValidationError("TaskRun cursor is invalid")
        return decoded

    def _encode_run_cursor(self, value: TaskRunCursor | None) -> str | None:
        return (
            None
            if value is None
            else self._encode_opaque_cursor(
                {"created_at": value.created_at, "run_id": value.run_id}
            )
        )

    def _decode_run_cursor(self, value: str | None) -> TaskRunCursor | None:
        decoded = self._decode_opaque_cursor(value)
        if decoded is None:
            return None
        if set(decoded) != {"created_at", "run_id"}:
            raise ValidationError("TaskRun list cursor shape is invalid")
        return TaskRunCursor(decoded["created_at"], decoded["run_id"])

    def _decode_ledger_cursor(self, value: str | None) -> TaskRunLedgerCursor | None:
        decoded = self._decode_opaque_cursor(value)
        if decoded is None:
            return None
        if set(decoded) != {"seq", "item_id"}:
            raise ValidationError("TaskRun ledger cursor shape is invalid")
        return TaskRunLedgerCursor(decoded["seq"], decoded["item_id"])

    def _encode_ledger_cursor(
        self,
        value: TaskRunLedgerCursor | None,
    ) -> str | None:
        return (
            None
            if value is None
            else self._encode_opaque_cursor(
                {"seq": value.seq, "item_id": value.item_id}
            )
        )

    def _decode_requirement_cursor(
        self,
        value: str | None,
    ) -> tuple[int, str] | None:
        decoded = self._decode_opaque_cursor(value)
        if decoded is None:
            return None
        if set(decoded) != {"ordinal", "requirement_id"}:
            raise ValidationError("TaskRun requirement cursor shape is invalid")
        ordinal = decoded["ordinal"]
        requirement_id = decoded["requirement_id"]
        if type(ordinal) is not int or ordinal < 0 or not isinstance(requirement_id, str):
            raise ValidationError("TaskRun requirement cursor is invalid")
        return ordinal, requirement_id


__all__ = [
    "TASK_RUN_REFERENCE_KEY",
    "TASK_RUN_REFERENCE_SCHEMA_VERSION",
    "TaskRunListPage",
    "TaskRunLedgerListPage",
    "TaskRunManager",
    "TaskRunRecoveryOption",
    "TaskRunRequirementPage",
    "is_task_run_reference_payload",
]
