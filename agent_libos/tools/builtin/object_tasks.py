from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import ObjectHandle, ObjectRight, ObjectTask, ProcessMessageKind
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.utils.serde import to_jsonable

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class ObjectTaskInfo(BaseModel):
    task_id: str
    owner_oid: str
    creator_pid: str
    runner_pid: str | None
    tool: str
    tool_id: str | None
    status: str
    result_oid: str | None = None
    error: str | None = None
    wait: dict[str, Any]
    notification: dict[str, Any]
    owner_watch: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class StartObjectTaskArgs(BaseModel):
    owner_oid: str | None = Field(default=None, description="Object id that owns this task.")
    owner_name: str | None = Field(default=None, description="Namespace-local object name that owns this task.")
    namespace: str | None = Field(default=None, description="Namespace for owner_name. Defaults to this process namespace.")
    tool: str = Field(description="Visible tool to execute in the object task runner.")
    args: dict[str, Any] = Field(default_factory=dict, description="JSON object passed to the tool.")
    notify_pid: str | None = Field(default=None, description="Process to notify; defaults to this process.")
    notify_kind: str = Field(default=ProcessMessageKind.NORMAL.value, description="normal or interrupt.")
    notify_channel: str | None = Field(default=None, description="Process-message channel; defaults to object-task.")
    inherit_capabilities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Explicit capabilities to delegate into the runner child process.",
    )
    grant_result_to_notify: bool = Field(
        default=False,
        description="If true, try to grant result read authority to notify_pid; requires object grant authority.",
    )
    owner_watch: bool = Field(default=False, description="If true, notify the runner when the owner object changes.")
    watch_events: list[str] = Field(
        default_factory=list,
        description="Owner change events to watch: updated and/or linked. Defaults to both when owner_watch is true.",
    )
    watch_channel: str | None = Field(default=None, description="Runner process-message channel for owner watch notices.")
    watch_kind: str = Field(default=ProcessMessageKind.NORMAL.value, description="normal or interrupt.")


class StartObjectTaskOutput(BaseModel):
    task: ObjectTaskInfo


class GetObjectTaskArgs(BaseModel):
    task_id: str


class GetObjectTaskOutput(BaseModel):
    task: ObjectTaskInfo


class ListObjectTasksArgs(BaseModel):
    owner_oid: str | None = None
    include_terminal: bool = True
    limit: int | None = Field(default=None, ge=0, le=1000)


class ListObjectTasksOutput(BaseModel):
    tasks: list[ObjectTaskInfo]


class CancelObjectTaskArgs(BaseModel):
    task_id: str
    reason: str | None = None


class CancelObjectTaskOutput(BaseModel):
    task: ObjectTaskInfo


class WaitObjectTaskArgs(BaseModel):
    task_id: str
    timeout_s: float | None = Field(default=None, ge=0, le=_TOOL_DEFAULTS.max_sleep_seconds)


class WaitObjectTaskOutput(BaseModel):
    task: ObjectTaskInfo


class WatchObjectTaskOwnerArgs(BaseModel):
    task_id: str
    enabled: bool = Field(default=True, description="Enable or disable owner-change notices for this Object task.")
    watch_events: list[str] = Field(
        default_factory=list,
        description="Owner change events to watch: updated and/or linked. Empty keeps the current/default events.",
    )
    watch_channel: str | None = Field(default=None, description="Runner process-message channel for owner watch notices.")
    watch_kind: str | None = Field(default=None, description="normal or interrupt.")


class WatchObjectTaskOwnerOutput(BaseModel):
    task: ObjectTaskInfo


class StartObjectTaskTool(SyncAgentTool[StartObjectTaskArgs]):
    name = "start_object_task"
    description = (
        "Start a background Object-bound task that executes one visible tool through a runner child process. "
        "Completion and wait states notify the selected process via process messages."
    )
    args_schema = StartObjectTaskArgs
    output_schema = StartObjectTaskOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={
            "object.read",
            "object.write",
            "object.link",
            "process.spawn",
            "process.message",
            "tool.call",
        },
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object", "task", "async"]

    def run(self, args: StartObjectTaskArgs, ctx: ToolContext) -> StartObjectTaskOutput:
        runtime = _runtime(ctx)
        owner = _owner_handle(runtime, ctx.pid, owner_oid=args.owner_oid, owner_name=args.owner_name, namespace=args.namespace)
        task = runtime.object_tasks.start(
            ctx.pid,
            owner,
            args.tool,
            args.args,
            notify_pid=args.notify_pid,
            notify_kind=args.notify_kind,
            notify_channel=args.notify_channel,
            inherit_capabilities=args.inherit_capabilities,
            grant_result_to_notify=args.grant_result_to_notify,
            owner_watch=_owner_watch_args(args),
        )
        return StartObjectTaskOutput(task=_task_info(task))


class GetObjectTaskTool(SyncAgentTool[GetObjectTaskArgs]):
    name = "get_object_task"
    description = (
        "Inspect an Object task if this process owns it or can read its owner object. "
        "Inspection may reconcile runner state, retry a notification, or schedule a safe resume."
    )
    args_schema = GetObjectTaskArgs
    output_schema = GetObjectTaskOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.read", "process.message", "tool.call"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object", "task", "inspect"]

    def run(self, args: GetObjectTaskArgs, ctx: ToolContext) -> GetObjectTaskOutput:
        task = _runtime(ctx).object_tasks.get(args.task_id, actor_pid=ctx.pid)
        return GetObjectTaskOutput(task=_task_info(task))


class ListObjectTasksTool(SyncAgentTool[ListObjectTasksArgs]):
    name = "list_object_tasks"
    description = (
        "List Object tasks visible to this process. Candidate refresh may reconcile "
        "runner state and publish a terminal notification."
    )
    args_schema = ListObjectTasksArgs
    output_schema = ListObjectTasksOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.read", "process.message"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object", "task", "inspect"]

    def run(self, args: ListObjectTasksArgs, ctx: ToolContext) -> ListObjectTasksOutput:
        tasks = _runtime(ctx).object_tasks.list(
            actor_pid=ctx.pid,
            owner_oid=args.owner_oid,
            include_terminal=args.include_terminal,
            limit=args.limit,
        )
        return ListObjectTasksOutput(tasks=[_task_info(task) for task in tasks])


class CancelObjectTaskTool(SyncAgentTool[CancelObjectTaskArgs]):
    name = "cancel_object_task"
    description = "Cancel a running or waiting Object task."
    args_schema = CancelObjectTaskArgs
    output_schema = CancelObjectTaskOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write", "process.signal"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object", "task"]

    def run(self, args: CancelObjectTaskArgs, ctx: ToolContext) -> CancelObjectTaskOutput:
        task = _runtime(ctx).object_tasks.cancel(args.task_id, actor_pid=ctx.pid, reason=args.reason)
        return CancelObjectTaskOutput(task=_task_info(task))


class WaitObjectTaskTool(SyncAgentTool[WaitObjectTaskArgs]):
    name = "wait_object_task"
    description = (
        "Wait for an Object task state change. It may return a terminal state, one of "
        "waiting_human/waiting_process/waiting_message, or queued/running when timeout_s expires; "
        "only succeeded is successful completion."
    )
    args_schema = WaitObjectTaskArgs
    output_schema = WaitObjectTaskOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.read", "process.message", "tool.call"},
        timeout_s=_TOOL_DEFAULTS.sleep_tool_timeout_s,
    )
    tags = ["memory", "object", "task", "wait"]

    def run(self, args: WaitObjectTaskArgs, ctx: ToolContext) -> WaitObjectTaskOutput:
        task = _runtime(ctx).object_tasks.wait(args.task_id, actor_pid=ctx.pid, timeout=args.timeout_s)
        return WaitObjectTaskOutput(task=_task_info(task))


class WatchObjectTaskOwnerTool(SyncAgentTool[WatchObjectTaskOwnerArgs]):
    name = "watch_object_task_owner"
    description = "Enable, disable, or update owner-change notices for an active Object task."
    args_schema = WatchObjectTaskOwnerArgs
    output_schema = WatchObjectTaskOwnerOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write", "process.message"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object", "task", "watch"]

    def run(self, args: WatchObjectTaskOwnerArgs, ctx: ToolContext) -> WatchObjectTaskOwnerOutput:
        task = _runtime(ctx).object_tasks.watch_owner(
            args.task_id,
            actor_pid=ctx.pid,
            enabled=args.enabled,
            events=args.watch_events or None,
            channel=args.watch_channel,
            kind=args.watch_kind,
        )
        return WatchObjectTaskOwnerOutput(task=_task_info(task))


def _runtime(ctx: ToolContext):
    runtime = ctx.runtime
    if runtime is None:
        raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
    return runtime


def _owner_handle(runtime: Any, pid: str, *, owner_oid: str | None, owner_name: str | None, namespace: str | None) -> ObjectHandle:
    if owner_oid:
        return runtime.memory.handle_for_oid(
            pid,
            owner_oid,
            required_rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
        )
    if owner_name:
        return runtime.memory.handle_for_name(
            pid,
            owner_name,
            rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
            namespace=namespace,
        )
    raise ToolExecutionError(
        "Either owner_oid or owner_name is required.",
        code=ToolErrorCode.VALIDATION_ERROR,
    )


def _owner_watch_args(args: StartObjectTaskArgs) -> dict[str, Any] | bool:
    enabled = bool(args.owner_watch or args.watch_events or args.watch_channel)
    if not enabled:
        return False
    selected: dict[str, Any] = {"enabled": True, "kind": args.watch_kind}
    if args.watch_events:
        selected["events"] = args.watch_events
    if args.watch_channel:
        selected["channel"] = args.watch_channel
    return selected


def _task_info(task: ObjectTask) -> ObjectTaskInfo:
    return ObjectTaskInfo(**to_jsonable(task))
