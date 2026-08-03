from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import (
    RuntimePublicationPending,
    RuntimeRecoveryRequired,
)
from agent_libos.tools.base import (
    SyncAgentTool,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolPolicy,
)

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_CANCELLED_HUMAN_REQS_KEY = "cancelled_human_re" "quests"
_MODEL_PREVIEW_ITEMS = DEFAULT_CONFIG.checkpoint.diff_preview_items
_MODEL_PREVIEW_TEXT_CHARS = DEFAULT_CONFIG.tools.tool_observability_preview_chars
_CHECKPOINT_REASON_MAX_CHARS = 512
_CHECKPOINT_REASON_MAX_BYTES = 1_024
_RESTORE_EXCEPTION_TREE_MAX_NODES = 1_024
_DIFF_TABLE_NAMES = (
    "processes",
    "objects",
    "capabilities",
    "process_resource_reservations",
    "process_messages",
    "llm_pending_actions",
    "tool_candidates",
    "skills",
)
_PROCESS_FIELDS = (
    "pid",
    "parent_pid",
    "image_id",
    "status",
    "working_directory",
    "goal_oid",
    "wait_state",
    "outcome",
    "state_generation",
)
_MODULE_FIELDS = (
    "module_id",
    "name",
    "version",
    "source_kind",
    "entrypoint",
)
_CHECKPOINT_FIELDS = (
    "checkpoint_id",
    "pid",
    "reason",
    "created_at",
    "created_by",
    "snapshot_version",
)
_EXTERNAL_EFFECT_FIELDS = (
    "provider",
    "operation",
    "target",
    "rollback_class",
    "rollback_status",
    "state_mutation",
    "information_flow",
    "effect_state",
    "transaction_state",
)
_MODEL_PRIVATE_NESTED_FIELDS = {
    "audit_id",
    "canonical_args_hash",
    "created_at",
    "event_id",
    "idempotency_key",
    "metadata",
    "provider_metadata",
    "provider_receipt",
    "receipt",
    "record_id",
    "source_sha256",
    "updated_at",
}
_FORK_RECEIPT_KEYS = {
    "checkpoint_id", "source_pid", "fork_root_pid", "pid_map", "object_map",
    "tool_map", "status", "main_state_committed", "reconciliation_pending",
    "post_commit_failures", "outcome_diagnostic",
}
_FORK_FAILURE_KEYS = {
    "phase", "error_type", "message", "audit_error_type", "audit_error",
    "failure_record_error_type", "failure_record_error",
}
_FORK_DIAGNOSTIC_KEYS = {
    "phase", "interruption_error_type", "interruption",
    "diagnostic_error_type", "diagnostic_error",
    "prepared_runtime_assets_retained", "fork_subtree_quarantined",
    "recovery_signal_record_id", "recovery_signal_error_type",
    "recovery_signal_error", "lifecycle_fence_requested",
    "operation_recovery_signal_recorded",
    "operation_recovery_signal_error_type", "operation_recovery_signal_error",
    "lifecycle_fence_error_type", "lifecycle_fence_error", "lifecycle_fenced",
}


class CheckpointResultPage(BaseModel):
    """Size metadata for one bounded model-facing collection."""

    count: int
    returned_count: int
    truncated: bool
    next_cursor: int | None = None


def _empty_result_page() -> CheckpointResultPage:
    return CheckpointResultPage(
        count=0,
        returned_count=0,
        truncated=False,
    )


class CreateCheckpointArgs(BaseModel):
    reason: str = Field(
        min_length=1,
        max_length=_CHECKPOINT_REASON_MAX_CHARS,
        description=(
            "A concise reason of at most "
            f"{_CHECKPOINT_REASON_MAX_CHARS} characters and "
            f"{_CHECKPOINT_REASON_MAX_BYTES} bytes of UTF-8 text."
        ),
    )
    pid: str | None = Field(
        default=None,
        description=(
            "Target process id. Omit this field to checkpoint the caller; do "
            "not pass null, the text 'None', or the caller pid."
        ),
    )

    @field_validator("reason")
    @classmethod
    def _bound_reason_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > _CHECKPOINT_REASON_MAX_BYTES:
            raise ValueError(
                "checkpoint reason exceeds "
                f"{_CHECKPOINT_REASON_MAX_BYTES} UTF-8 bytes"
            )
        return value

    @field_validator("pid", mode="before")
    @classmethod
    def _normalize_omitted_pid(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().casefold() in {
            "",
            "none",
            "null",
        }:
            return None
        return value


class CreateCheckpointOutput(BaseModel):
    checkpoint_id: str
    pid: str
    reason: str = Field(description="Bounded preview of the persisted reason.")


class ListCheckpointsArgs(BaseModel):
    pid: str | None = Field(default=None, description="Process id to list. Defaults to the caller.")
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum checkpoints to return.",
    )


class CheckpointListItem(BaseModel):
    checkpoint_id: str
    pid: str
    reason: str = Field(description="Bounded preview of the checkpoint reason.")
    created_at: str
    created_by: str | None = None
    snapshot_version: int | None = None


class ListCheckpointsOutput(BaseModel):
    checkpoints: list[CheckpointListItem]
    count: int = Field(
        description="Checkpoint count observed in the Host-bounded list window."
    )
    has_more: bool = Field(
        description=(
            "Whether increasing limit up to the Host-configured maximum can "
            "return more checkpoints from this bounded list window."
        )
    )


class InspectCheckpointArgs(BaseModel):
    checkpoint_id: str = Field(description="Checkpoint id returned by create_checkpoint or list_checkpoints.")
    process_cursor: int = Field(
        default=0,
        ge=0,
        description="Zero-based cursor for the saved process/subtree preview.",
    )
    module_cursor: int = Field(
        default=0,
        ge=0,
        description="Zero-based cursor for the saved module preview.",
    )
    detail_limit: int = Field(
        default=_MODEL_PREVIEW_ITEMS,
        ge=1,
        description="Maximum entries returned per inspect collection.",
    )


class CheckpointProcessInfo(BaseModel):
    pid: str
    parent_pid: str | None = None
    image_id: str
    status: str
    working_directory: str
    goal_oid: str | None = None
    wait_state: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    state_generation: int


class InspectCheckpointOutput(BaseModel):
    checkpoint: dict[str, Any]
    snapshot_version: int | None = None
    subtree_pids: list[str]
    subtree_pids_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    modules: list[dict[str, Any]]
    modules_page: CheckpointResultPage = Field(default_factory=_empty_result_page)
    counts: dict[str, int]
    processes: list[CheckpointProcessInfo]
    processes_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )


class DiffCheckpointArgs(BaseModel):
    checkpoint_id: str = Field(
        description="Checkpoint id whose snapshot is compared with current reconstructable state."
    )
    external_effect_cursor: int = Field(
        default=0,
        ge=0,
        description="Zero-based cursor for the external-effect preview.",
    )
    external_effect_limit: int = Field(
        default=_MODEL_PREVIEW_ITEMS,
        ge=1,
        description="Maximum sanitized external-effect summaries returned.",
    )


class DiffCheckpointOutput(BaseModel):
    checkpoint_id: str
    pid: str
    tables: dict[str, Any]
    external_effects_since_checkpoint: list[dict[str, Any]]
    external_effects_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    external_effect_summary: dict[str, Any] = Field(default_factory=dict)
    restore_external_policy: str = "report_only"


class RestoreCheckpointArgs(BaseModel):
    checkpoint_id: str = Field(description="Checkpoint id whose saved subtree will replace the live subtree.")


class RestoreCheckpointOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    checkpoint_id: str
    publication_id: str
    pid: str
    status: str
    main_state_committed: bool
    reconciliation_pending: bool
    post_commit_failures: list[dict[str, str]]
    restored_pids: list[str]
    previous_pids: list[str]
    cancelled_human_request_ids: list[str] = Field(
        validation_alias=_CANCELLED_HUMAN_REQS_KEY,
        serialization_alias=_CANCELLED_HUMAN_REQS_KEY,
    )
    superseded_messages: list[str]
    superseded_object_tasks: list[str]
    external_effects_since_checkpoint: list[dict[str, Any]]
    restored_pids_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    previous_pids_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    cancelled_human_requests_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    superseded_messages_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    superseded_object_tasks_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    external_effects_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    external_effect_summary: dict[str, Any] = Field(default_factory=dict)
    restore_external_policy: str = "report_only"
    post_commit_failures_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )


def _restore_recovery_signal(
    error: Exception,
) -> RuntimePublicationPending | RuntimeRecoveryRequired | None:
    """Find one unambiguous recovery signal in an ordinary exception tree."""

    signals: list[RuntimePublicationPending | RuntimeRecoveryRequired] = []
    pending: list[Exception] = [error]
    visited = 0
    try:
        while pending:
            current = pending.pop()
            visited += 1
            if visited > _RESTORE_EXCEPTION_TREE_MAX_NODES:
                return None
            if isinstance(
                current,
                (RuntimePublicationPending, RuntimeRecoveryRequired),
            ):
                signals.append(current)
                continue
            if isinstance(current, ExceptionGroup):
                nested = current.exceptions
                remaining = (
                    _RESTORE_EXCEPTION_TREE_MAX_NODES
                    - visited
                    - len(pending)
                )
                if len(nested) > remaining:
                    return None
                pending.extend(reversed(nested))
    except Exception:
        # Receipt extraction is diagnostic-only. Never replace the original
        # restore failure with an extractor traversal error.
        return None
    if not signals:
        return None
    identities = {
        (
            type(signal),
            signal.publication_id,
            signal.operation_id,
            signal.state,
            signal.phase,
        )
        for signal in signals
    }
    return signals[0] if len(identities) == 1 else None


class ForkCheckpointArgs(BaseModel):
    checkpoint_id: str = Field(
        description="Checkpoint id to copy into a new subtree without changing the source subtree."
    )
    parent_pid: str | None = Field(
        default=None,
        description=(
            "Optional existing process that will own the new fork root. "
            "Omit for a detached root; pass the caller pid explicitly to create a direct child."
        ),
    )


class ForkCheckpointOutput(BaseModel):
    checkpoint_id: str
    source_pid: str
    fork_root_pid: str
    pid_map: dict[str, str]
    object_map: dict[str, str]
    tool_map: dict[str, str] = Field(default_factory=dict)
    status: str = "forked"
    main_state_committed: bool | None = True
    reconciliation_pending: bool = False
    post_commit_failures: list[dict[str, str]] = Field(default_factory=list)
    outcome_diagnostic: dict[str, Any] | None = None
    pid_map_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    object_map_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    tool_map_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )
    post_commit_failures_page: CheckpointResultPage = Field(
        default_factory=_empty_result_page
    )


class CreateCheckpointTool(SyncAgentTool[CreateCheckpointArgs]):
    name = "create_checkpoint"
    description = (
        "Snapshot reconstructable internal state for a process subtree. "
        "External provider state and already completed external effects are recorded as evidence but are not "
        "rolled back."
    )
    args_schema = CreateCheckpointArgs
    output_schema = CreateCheckpointOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"checkpoint.write"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["checkpoint", "durable"]

    def run(self, args: CreateCheckpointArgs, ctx: ToolContext) -> CreateCheckpointOutput:
        runtime = _runtime(ctx)
        target_pid = args.pid or ctx.pid
        checkpoint_id = runtime.checkpoint.create(target_pid, args.reason, actor=ctx.pid)
        return CreateCheckpointOutput(
            checkpoint_id=checkpoint_id,
            pid=target_pid,
            reason=_preview_text(args.reason, _model_preview_text_limit(ctx)),
        )


class ListCheckpointsTool(SyncAgentTool[ListCheckpointsArgs]):
    name = "list_checkpoints"
    description = "List durable checkpoints visible to this process."
    args_schema = ListCheckpointsArgs
    output_schema = ListCheckpointsOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.standard_timeout_s)
    tags = ["checkpoint", "inspect"]

    def run(self, args: ListCheckpointsArgs, ctx: ToolContext) -> ListCheckpointsOutput:
        runtime = _runtime(ctx)
        configured_limit = _checkpoint_list_limit(ctx)
        selected_limit = min(args.limit or configured_limit, configured_limit)
        raw_checkpoints = runtime.checkpoint.list(
            args.pid or ctx.pid,
            actor=ctx.pid,
            limit=configured_limit,
        )
        selected = raw_checkpoints[:selected_limit]
        text_limit = _model_preview_text_limit(ctx)
        return ListCheckpointsOutput(
            checkpoints=[
                _checkpoint_list_item(value, text_limit=text_limit)
                for value in selected
                if isinstance(value, Mapping)
            ],
            count=len(raw_checkpoints),
            has_more=len(raw_checkpoints) > selected_limit,
        )


class InspectCheckpointTool(SyncAgentTool[InspectCheckpointArgs]):
    name = "inspect_checkpoint"
    description = (
        "Inspect a checkpoint's saved processes, modules, and counts without changing live state. "
        "Use this before a restore or fork when the snapshot contents are uncertain."
    )
    args_schema = InspectCheckpointArgs
    output_schema = InspectCheckpointOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.standard_timeout_s)
    tags = ["checkpoint", "inspect"]

    def run(self, args: InspectCheckpointArgs, ctx: ToolContext) -> InspectCheckpointOutput:
        data = _runtime(ctx).checkpoint.inspect(args.checkpoint_id, actor=ctx.pid)
        limit = min(args.detail_limit, _model_preview_limit(ctx))
        return _inspect_output(
            data,
            process_cursor=args.process_cursor,
            module_cursor=args.module_cursor,
            limit=limit,
            text_limit=_model_preview_text_limit(ctx),
        )


class DiffCheckpointTool(SyncAgentTool[DiffCheckpointArgs]):
    name = "diff_checkpoint"
    description = (
        "Compare current reconstructable process state with a checkpoint and list external effects since capture. "
        "The comparison is read-only and does not imply those external effects can be reversed."
    )
    args_schema = DiffCheckpointArgs
    output_schema = DiffCheckpointOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.standard_timeout_s)
    tags = ["checkpoint", "inspect"]

    def run(self, args: DiffCheckpointArgs, ctx: ToolContext) -> DiffCheckpointOutput:
        limit = min(args.external_effect_limit, _model_preview_limit(ctx))
        return _diff_output(
            _runtime(ctx).checkpoint.diff(args.checkpoint_id, actor=ctx.pid),
            cursor=args.external_effect_cursor,
            limit=limit,
            text_limit=_model_preview_text_limit(ctx),
        )


class RestoreCheckpointTool(SyncAgentTool[RestoreCheckpointArgs]):
    name = "restore_checkpoint"
    description = (
        "Replace the live process subtree with this checkpoint's reconstructable internal state, superseding "
        "pending messages, object tasks, and human requests as reported in the result. This does not roll back external provider "
        "state. It requires checkpoint admin capability plus exact image admin authority for changed existing images, "
        "or exact image write authority for missing images."
    )
    args_schema = RestoreCheckpointArgs
    output_schema = RestoreCheckpointOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={
            "capability.write",
            "checkpoint.restore",
            "image.admin",
            "image.write",
            "object.write",
            "process.lifecycle",
        },
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["checkpoint", "restore", "high_risk"]

    def run(self, args: RestoreCheckpointArgs, ctx: ToolContext) -> RestoreCheckpointOutput:
        try:
            restored = _runtime(ctx).checkpoint.restore(ctx.pid, args.checkpoint_id)
        except Exception as exc:
            recovery = _restore_recovery_signal(exc)
            if recovery is None:
                raise
            status = (
                "restore_recovery_required"
                if isinstance(recovery, RuntimeRecoveryRequired)
                else "restore_publication_pending"
            )
            raise ToolExecutionError(
                "Checkpoint restore outcome requires Host reconciliation; "
                "inspect the structured receipt and do not retry.",
                code=ToolErrorCode.EXECUTION_ERROR,
                retryable=False,
                details={
                    "checkpoint_restore_receipt": {
                        "checkpoint_id": args.checkpoint_id,
                        "publication_id": recovery.publication_id,
                        "operation_id": recovery.operation_id,
                        "state": recovery.state,
                        "phase": recovery.phase,
                        "status": status,
                        "main_state_committed": None,
                        "reconciliation_pending": True,
                    }
                },
            ) from exc
        return _restore_output(
            restored,
            limit=_model_preview_limit(ctx),
            text_limit=_model_preview_text_limit(ctx),
        )


class ForkCheckpointTool(SyncAgentTool[ForkCheckpointArgs]):
    name = "fork_checkpoint"
    description = (
        "Create a new isolated process subtree and remapped Objects from a checkpoint without replacing live state. "
        "The result returns pid/object/tool maps plus committed status and post-commit warnings; external provider "
        "state is not cloned or isolated. Omit parent_pid for a detached root or pass it explicitly for attachment."
    )
    args_schema = ForkCheckpointArgs
    output_schema = ForkCheckpointOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"capability.write", "checkpoint.execute", "object.write", "process.spawn"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["checkpoint", "fork"]

    def run(self, args: ForkCheckpointArgs, ctx: ToolContext) -> ForkCheckpointOutput:
        limit = _model_preview_limit(ctx)
        text_limit = _model_preview_text_limit(ctx)
        try:
            receipt = _runtime(ctx).checkpoint.fork_from_checkpoint(
                ctx.pid,
                args.checkpoint_id,
                parent_pid=args.parent_pid,
            )
        except Exception as exc:
            raw_receipt = getattr(exc, "checkpoint_fork_receipt", None)
            if not isinstance(raw_receipt, Mapping):
                raise
            bounded_receipt = _fork_output(
                raw_receipt,
                limit=limit,
                text_limit=text_limit,
            )
            raise ToolExecutionError(
                "Checkpoint fork outcome requires reconciliation; inspect the "
                "structured receipt before retrying.",
                code=ToolErrorCode.EXECUTION_ERROR,
                retryable=False,
                details={
                    "error_type": type(exc).__name__,
                    "checkpoint_fork_receipt": bounded_receipt.model_dump(),
                },
            ) from exc
        return _fork_output(
            receipt,
            limit=limit,
            text_limit=text_limit,
        )


def _inspect_output(
    data: Mapping[str, Any],
    *,
    process_cursor: int,
    module_cursor: int,
    limit: int,
    text_limit: int,
) -> InspectCheckpointOutput:
    raw_subtree = _mapping_sequence(data.get("subtree_pids"))
    raw_modules = _mapping_sequence(data.get("modules"))
    raw_processes = _mapping_sequence(data.get("processes"))
    subtree, subtree_page = _sequence_page(
        raw_subtree,
        cursor=process_cursor,
        limit=limit,
        resumable=True,
    )
    modules, modules_page = _sequence_page(
        raw_modules,
        cursor=module_cursor,
        limit=limit,
        resumable=True,
    )
    processes, processes_page = _sequence_page(
        raw_processes,
        cursor=process_cursor,
        limit=limit,
        resumable=True,
    )
    checkpoint = data.get("checkpoint")
    return InspectCheckpointOutput(
        checkpoint=_project_allowlist_mapping(
            checkpoint if isinstance(checkpoint, Mapping) else {},
            _CHECKPOINT_FIELDS,
            text_limit=text_limit,
            nested=False,
        ),
        snapshot_version=_optional_int(data.get("snapshot_version")),
        subtree_pids=[str(value) for value in subtree],
        subtree_pids_page=subtree_page,
        modules=[
            _project_allowlist_mapping(
                value if isinstance(value, Mapping) else {},
                _MODULE_FIELDS,
                text_limit=text_limit,
                nested=False,
            )
            for value in modules
        ],
        modules_page=modules_page,
        counts=_integer_mapping(data.get("counts")),
        processes=[
            CheckpointProcessInfo.model_validate(
                _project_allowlist_mapping(
                    value if isinstance(value, Mapping) else {},
                    _PROCESS_FIELDS,
                    text_limit=text_limit,
                    nested=True,
                    item_limit=limit,
                )
            )
            for value in processes
        ],
        processes_page=processes_page,
    )


def _diff_output(
    data: Mapping[str, Any],
    *,
    cursor: int,
    limit: int,
    text_limit: int,
) -> DiffCheckpointOutput:
    raw_effects = _mapping_sequence(data.get("external_effects_since_checkpoint"))
    effects, effects_page = _sequence_page(
        raw_effects,
        cursor=cursor,
        limit=limit,
        resumable=True,
    )
    return DiffCheckpointOutput(
        checkpoint_id=str(data["checkpoint_id"]),
        pid=str(data["pid"]),
        tables=_project_diff_tables(
            data.get("tables"),
            limit=limit,
            text_limit=text_limit,
        ),
        external_effects_since_checkpoint=[
            _project_external_effect(value, text_limit=text_limit)
            for value in effects
            if isinstance(value, Mapping)
        ],
        external_effects_page=effects_page,
        external_effect_summary=_project_external_effect_summary(
            data.get("external_effect_summary"),
            limit=limit,
            text_limit=text_limit,
        ),
        restore_external_policy=str(
            data.get("restore_external_policy") or "report_only"
        ),
    )


def _restore_output(
    data: Mapping[str, Any],
    *,
    limit: int,
    text_limit: int,
) -> RestoreCheckpointOutput:
    restored, restored_page = _bounded_sequence(data.get("restored_pids"), limit)
    previous, previous_page = _bounded_sequence(data.get("previous_pids"), limit)
    cancelled, cancelled_page = _bounded_sequence(
        data.get("cancelled_human_requests"), limit
    )
    superseded_messages, superseded_messages_page = _bounded_sequence(
        data.get("superseded_messages"), limit
    )
    superseded_tasks, superseded_tasks_page = _bounded_sequence(
        data.get("superseded_object_tasks"), limit
    )
    raw_effects, effects_page = _bounded_sequence(
        data.get("external_effects_since_checkpoint"), limit
    )
    raw_failures, failures_page = _bounded_sequence(
        data.get("post_commit_failures"), limit
    )
    return RestoreCheckpointOutput(
        checkpoint_id=str(data["checkpoint_id"]),
        publication_id=str(data["publication_id"]),
        pid=str(data["pid"]),
        status=str(data["status"]),
        main_state_committed=bool(data["main_state_committed"]),
        reconciliation_pending=bool(data["reconciliation_pending"]),
        post_commit_failures=[
            _string_mapping(value, text_limit=text_limit)
            for value in raw_failures
            if isinstance(value, Mapping)
        ],
        restored_pids=[str(value) for value in restored],
        previous_pids=[str(value) for value in previous],
        cancelled_human_request_ids=[str(value) for value in cancelled],
        superseded_messages=[str(value) for value in superseded_messages],
        superseded_object_tasks=[str(value) for value in superseded_tasks],
        external_effects_since_checkpoint=[
            _project_external_effect(value, text_limit=text_limit)
            for value in raw_effects
            if isinstance(value, Mapping)
        ],
        restored_pids_page=restored_page,
        previous_pids_page=previous_page,
        cancelled_human_requests_page=cancelled_page,
        superseded_messages_page=superseded_messages_page,
        superseded_object_tasks_page=superseded_tasks_page,
        external_effects_page=effects_page,
        external_effect_summary=_project_external_effect_summary(
            data.get("external_effect_summary"),
            limit=limit,
            text_limit=text_limit,
        ),
        restore_external_policy=str(
            data.get("restore_external_policy") or "report_only"
        ),
        post_commit_failures_page=failures_page,
    )


def _fork_map_is_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return not any(
        not isinstance(source, str)
        or not source
        or not isinstance(target, str)
        or not target
        for source, target in value.items()
    )


def _validate_fork_receipt_fields(data: Mapping[str, Any]) -> None:
    if set(data) - _FORK_RECEIPT_KEYS:
        raise ValueError("checkpoint fork receipt contains unsupported fields")
    for key in ("checkpoint_id", "source_pid", "fork_root_pid", "status"):
        if not isinstance(data.get(key), str) or not data.get(key):
            raise ValueError(f"checkpoint fork receipt has invalid {key}")
    for key in ("pid_map", "object_map", "tool_map"):
        if not _fork_map_is_valid(data.get(key)):
            raise ValueError(f"checkpoint fork receipt has invalid {key}")


def _fork_receipt_outcome(
    data: Mapping[str, Any],
) -> tuple[str, bool | None, bool]:
    committed = data.get("main_state_committed", True)
    if committed is not None and not isinstance(committed, bool):
        raise ValueError("checkpoint fork receipt has invalid commit status")
    pending = data.get("reconciliation_pending", False)
    if not isinstance(pending, bool):
        raise ValueError("checkpoint fork receipt has invalid reconciliation status")
    status = data["status"]
    if (status, committed, pending) not in {
        ("forked", True, False),
        ("forked_with_warnings", True, False),
        ("fork_outcome_unknown", None, True),
        ("fork_recovery_required", True, True),
    }:
        raise ValueError("checkpoint fork receipt has inconsistent outcome fields")
    return status, committed, pending


def _fork_failure_is_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) - _FORK_FAILURE_KEYS:
        return False
    return all(
        isinstance(value.get(key), str) and value.get(key)
        for key in ("phase", "error_type", "message")
    )


def _validate_fork_failure_records(data: Mapping[str, Any], status: str) -> None:
    failures = data.get("post_commit_failures")
    if not isinstance(failures, list) or any(
        not _fork_failure_is_valid(value) for value in failures
    ):
        raise ValueError("checkpoint fork receipt has invalid failure records")
    if status == "forked" and failures:
        raise ValueError("checkpoint fork success receipt cannot contain failures")
    if status == "forked_with_warnings" and not failures:
        raise ValueError("checkpoint fork warning receipt requires a failure")
    if status == "fork_recovery_required" and not failures:
        raise ValueError("checkpoint fork recovery receipt requires a failure")


def _project_fork_diagnostic(
    data: Mapping[str, Any],
    *,
    pending: bool,
    text_limit: int,
) -> dict[str, Any] | None:
    raw_diagnostic = data.get("outcome_diagnostic")
    if raw_diagnostic is not None and (
        not isinstance(raw_diagnostic, Mapping)
        or set(raw_diagnostic) - _FORK_DIAGNOSTIC_KEYS
    ):
        raise ValueError("checkpoint fork receipt has invalid outcome diagnostic")
    if pending and not isinstance(raw_diagnostic, Mapping):
        raise ValueError("pending checkpoint fork receipt requires a diagnostic")
    if not isinstance(raw_diagnostic, Mapping):
        return None
    return {
        str(key): (
            value
            if isinstance(value, (bool, int)) or value is None
            else _preview_text(str(value), text_limit)
        )
        for key, value in raw_diagnostic.items()
        if isinstance(value, (str, bool, int)) or value is None
    }


def _fork_output(
    data: Mapping[str, Any],
    *,
    limit: int,
    text_limit: int,
) -> ForkCheckpointOutput:
    _validate_fork_receipt_fields(data)
    pid_map, pid_page = _bounded_mapping(data.get("pid_map"), limit)
    object_map, object_page = _bounded_mapping(data.get("object_map"), limit)
    tool_map, tool_page = _bounded_mapping(data.get("tool_map"), limit)
    raw_failures, failures_page = _bounded_sequence(
        data.get("post_commit_failures"), limit
    )
    status, committed, pending = _fork_receipt_outcome(data)
    _validate_fork_failure_records(data, status)
    diagnostic = _project_fork_diagnostic(
        data,
        pending=pending,
        text_limit=text_limit,
    )
    return ForkCheckpointOutput(
        checkpoint_id=str(data["checkpoint_id"]),
        source_pid=str(data["source_pid"]),
        fork_root_pid=str(data["fork_root_pid"]),
        pid_map={str(key): str(value) for key, value in pid_map.items()},
        object_map={str(key): str(value) for key, value in object_map.items()},
        tool_map={str(key): str(value) for key, value in tool_map.items()},
        status=status,
        main_state_committed=committed,
        reconciliation_pending=pending,
        post_commit_failures=[
            _string_mapping(value, text_limit=text_limit)
            for value in raw_failures
            if isinstance(value, Mapping)
        ],
        outcome_diagnostic=diagnostic,
        pid_map_page=pid_page,
        object_map_page=object_page,
        tool_map_page=tool_page,
        post_commit_failures_page=failures_page,
    )


def _project_external_effect(
    value: Mapping[str, Any],
    *,
    text_limit: int,
) -> dict[str, Any]:
    """Return only effect facts useful for model decisions.

    Durable evidence keeps effect ids, receipts, provider metadata, hashes,
    audit/event links, and timestamps. Those volatile/high-volume fields are
    intentionally absent from the model-facing checkpoint tool result.
    """

    return _project_allowlist_mapping(
        value,
        _EXTERNAL_EFFECT_FIELDS,
        text_limit=text_limit,
        nested=False,
    )


def _checkpoint_list_item(
    value: Mapping[str, Any],
    *,
    text_limit: int,
) -> CheckpointListItem:
    """Project durable checkpoint metadata to a small model-facing allowlist."""

    created_by = value.get("created_by")
    snapshot_version = value.get("snapshot_version")
    return CheckpointListItem(
        checkpoint_id=str(value["checkpoint_id"]),
        pid=str(value["pid"]),
        reason=_preview_text(str(value.get("reason") or ""), text_limit),
        created_at=_preview_text(str(value.get("created_at") or ""), text_limit),
        created_by=(
            _preview_text(str(created_by), text_limit)
            if created_by is not None
            else None
        ),
        snapshot_version=(
            _optional_int(snapshot_version)
            if snapshot_version is not None
            else None
        ),
    )


def _project_external_effect_summary(
    value: Any,
    *,
    limit: int,
    text_limit: int,
) -> dict[str, Any]:
    summary = value if isinstance(value, Mapping) else {}
    projected: dict[str, Any] = {}
    for key in (
        "total",
        "state_mutations",
        "information_flows",
        "pending",
    ):
        projected[key] = _nonnegative_int(summary.get(key))
    for key in ("by_rollback_class", "by_state"):
        selected, page = _bounded_mapping(summary.get(key), limit)
        projected[key] = {
            _preview_text(str(item_key), text_limit): _nonnegative_int(item_value)
            for item_key, item_value in selected.items()
        }
        projected[f"{key}_page"] = page.model_dump()
    selected_operations, operations_page = _bounded_mapping(
        summary.get("by_provider_operation"), limit
    )
    projected["by_provider_operation"] = {
        _preview_text(str(key), text_limit): _nonnegative_int(item)
        for key, item in selected_operations.items()
    }
    projected["by_provider_operation_page"] = operations_page.model_dump()
    return projected


def _project_diff_tables(
    value: Any,
    *,
    limit: int,
    text_limit: int,
) -> dict[str, Any]:
    tables = value if isinstance(value, Mapping) else {}
    projected: dict[str, Any] = {}
    for name in _DIFF_TABLE_NAMES:
        raw = tables.get(name)
        table = raw if isinstance(raw, Mapping) else {}
        projected[name] = {
            collection: [
                _preview_text(str(item), text_limit)
                for item in _mapping_sequence(table.get(collection))[
                    :limit
                ]
            ]
            for collection in ("added", "removed", "changed")
        }
        for collection in ("added", "removed", "changed"):
            projected[name][f"{collection}_count"] = _nonnegative_int(
                table.get(f"{collection}_count")
            )
    return projected


def _project_allowlist_mapping(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    text_limit: int,
    nested: bool,
    item_limit: int = _MODEL_PREVIEW_ITEMS,
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for field in fields:
        if field not in value:
            continue
        item = value[field]
        if isinstance(item, str):
            projected[field] = _preview_text(item, text_limit)
        elif nested:
            projected[field] = _bounded_json_value(
                item,
                text_limit=text_limit,
                remaining_depth=3,
                item_limit=item_limit,
            )
        else:
            projected[field] = item
    return projected


def _bounded_json_value(
    value: Any,
    *,
    text_limit: int,
    remaining_depth: int,
    item_limit: int,
) -> Any:
    if isinstance(value, str):
        return _preview_text(value, text_limit)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if remaining_depth <= 0:
        return "[nested value omitted]"
    if isinstance(value, Mapping):
        public_value = {
            key: item
            for key, item in value.items()
            if str(key) not in _MODEL_PRIVATE_NESTED_FIELDS
        }
        selected, page = _bounded_mapping(public_value, item_limit)
        projected = {
            _preview_text(str(key), text_limit): _bounded_json_value(
                item,
                text_limit=text_limit,
                remaining_depth=remaining_depth - 1,
                item_limit=item_limit,
            )
            for key, item in selected.items()
        }
        if page.truncated:
            projected["_projection"] = page.model_dump()
        return projected
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        selected, page = _bounded_sequence(value, item_limit)
        projected = [
            _bounded_json_value(
                item,
                text_limit=text_limit,
                remaining_depth=remaining_depth - 1,
                item_limit=item_limit,
            )
            for item in selected
        ]
        if page.truncated:
            projected.append({"_projection": page.model_dump()})
        return projected
    return _preview_text(str(value), text_limit)


def _sequence_page(
    value: Sequence[Any],
    *,
    cursor: int,
    limit: int,
    resumable: bool,
) -> tuple[list[Any], CheckpointResultPage]:
    count = len(value)
    selected = list(value[cursor : cursor + limit])
    returned_count = len(selected)
    next_offset = cursor + returned_count
    truncated = next_offset < count
    return selected, CheckpointResultPage(
        count=count,
        returned_count=returned_count,
        truncated=truncated,
        next_cursor=next_offset if resumable and truncated else None,
    )


def _bounded_sequence(
    value: Any,
    limit: int,
) -> tuple[list[Any], CheckpointResultPage]:
    return _sequence_page(
        _mapping_sequence(value),
        cursor=0,
        limit=limit,
        resumable=False,
    )


def _bounded_mapping(
    value: Any,
    limit: int,
) -> tuple[dict[Any, Any], CheckpointResultPage]:
    mapping = value if isinstance(value, Mapping) else {}
    ordered = sorted(mapping.items(), key=lambda item: str(item[0]))
    selected = dict(ordered[:limit])
    return selected, CheckpointResultPage(
        count=len(ordered),
        returned_count=len(selected),
        truncated=len(selected) < len(ordered),
    )


def _mapping_sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return list(value)
    return []


def _string_mapping(value: Mapping[str, Any], *, text_limit: int) -> dict[str, str]:
    return {
        _preview_text(str(key), text_limit): _preview_text(str(item), text_limit)
        for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
    }


def _integer_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _nonnegative_int(item)
        for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
    }


def _preview_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "…[truncated]"
    return f"{value[: max(0, limit - len(marker))]}{marker}"


def _model_preview_limit(ctx: ToolContext) -> int:
    configured = getattr(
        getattr(getattr(ctx.runtime, "config", None), "checkpoint", None),
        "diff_preview_items",
        _MODEL_PREVIEW_ITEMS,
    )
    try:
        selected = int(configured)
    except (TypeError, ValueError):
        selected = _MODEL_PREVIEW_ITEMS
    return max(1, selected)


def _checkpoint_list_limit(ctx: ToolContext) -> int:
    configured = getattr(
        getattr(getattr(ctx.runtime, "config", None), "checkpoint", None),
        "list_limit",
        DEFAULT_CONFIG.checkpoint.list_limit,
    )
    try:
        selected = int(configured)
    except (TypeError, ValueError):
        selected = DEFAULT_CONFIG.checkpoint.list_limit
    return max(1, selected)


def _model_preview_text_limit(ctx: ToolContext) -> int:
    configured = getattr(
        getattr(getattr(ctx.runtime, "config", None), "tools", None),
        "tool_observability_preview_chars",
        _MODEL_PREVIEW_TEXT_CHARS,
    )
    try:
        selected = int(configured)
    except (TypeError, ValueError):
        selected = _MODEL_PREVIEW_TEXT_CHARS
    return max(32, min(selected, _MODEL_PREVIEW_TEXT_CHARS))


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _runtime(ctx: ToolContext) -> Any:
    if ctx.runtime is None:
        raise ToolExecutionError("Runtime is unavailable.")
    return ctx.runtime
