from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolExecutionError, ToolPolicy

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_CANCELLED_HUMAN_REQS_KEY = "cancelled_human_re" "quests"


class CreateCheckpointArgs(BaseModel):
    reason: str = Field(description="Why this checkpoint is being created.")
    pid: str | None = Field(default=None, description="Target process id. Defaults to the caller.")


class CreateCheckpointOutput(BaseModel):
    checkpoint_id: str
    pid: str
    reason: str


class ListCheckpointsArgs(BaseModel):
    pid: str | None = Field(default=None, description="Process id to list. Defaults to the caller.")
    limit: int | None = Field(default=None, description="Maximum checkpoints to return.")


class ListCheckpointsOutput(BaseModel):
    checkpoints: list[dict[str, Any]]


class InspectCheckpointArgs(BaseModel):
    checkpoint_id: str = Field(description="Checkpoint id returned by create_checkpoint or list_checkpoints.")


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
    modules: list[dict[str, Any]]
    counts: dict[str, int]
    processes: list[CheckpointProcessInfo]


class DiffCheckpointArgs(BaseModel):
    checkpoint_id: str = Field(
        description="Checkpoint id whose snapshot is compared with current reconstructable state."
    )


class DiffCheckpointOutput(BaseModel):
    checkpoint_id: str
    pid: str
    tables: dict[str, Any]
    external_effects_since_checkpoint: list[dict[str, Any]]


class RestoreCheckpointArgs(BaseModel):
    checkpoint_id: str = Field(description="Checkpoint id whose saved subtree will replace the live subtree.")


class RestoreCheckpointOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

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


class ForkCheckpointArgs(BaseModel):
    checkpoint_id: str = Field(
        description="Checkpoint id to copy into a new subtree without changing the source subtree."
    )
    parent_pid: str | None = Field(
        default=None,
        description="Optional existing process that will own the new fork root; defaults to the caller.",
    )


class ForkCheckpointOutput(BaseModel):
    checkpoint_id: str
    source_pid: str
    fork_root_pid: str
    pid_map: dict[str, str]
    object_map: dict[str, str]


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
        return CreateCheckpointOutput(checkpoint_id=checkpoint_id, pid=target_pid, reason=args.reason)


class ListCheckpointsTool(SyncAgentTool[ListCheckpointsArgs]):
    name = "list_checkpoints"
    description = "List durable checkpoints visible to this process."
    args_schema = ListCheckpointsArgs
    output_schema = ListCheckpointsOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.standard_timeout_s)
    tags = ["checkpoint", "inspect"]

    def run(self, args: ListCheckpointsArgs, ctx: ToolContext) -> ListCheckpointsOutput:
        runtime = _runtime(ctx)
        return ListCheckpointsOutput(
            checkpoints=runtime.checkpoint.list(args.pid or ctx.pid, actor=ctx.pid, limit=args.limit)
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
        return InspectCheckpointOutput(**data)


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
        return DiffCheckpointOutput(**_runtime(ctx).checkpoint.diff(args.checkpoint_id, actor=ctx.pid))


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

    def run(self, args: RestoreCheckpointArgs, ctx: ToolContext) -> dict[str, Any]:
        return _runtime(ctx).checkpoint.restore(ctx.pid, args.checkpoint_id)


class ForkCheckpointTool(SyncAgentTool[ForkCheckpointArgs]):
    name = "fork_checkpoint"
    description = (
        "Create a new isolated process subtree and remapped Objects from a checkpoint without replacing live state. "
        "The result returns pid/object maps; external provider state is not cloned or isolated."
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
        return ForkCheckpointOutput(
            **_runtime(ctx).checkpoint.fork_from_checkpoint(
                ctx.pid,
                args.checkpoint_id,
                parent_pid=args.parent_pid,
            )
        )


def _runtime(ctx: ToolContext) -> Any:
    if ctx.runtime is None:
        raise ToolExecutionError("Runtime is unavailable.")
    return ctx.runtime
