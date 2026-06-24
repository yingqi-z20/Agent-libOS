from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import NotFound
from agent_libos.models import (
    AgentProcess,
    CapabilityRight,
    ForkMode,
    MemoryViewSpec,
    MergePolicy,
    ObjectHandle,
    ObjectMetadata,
    ObjectRight,
    ObjectType,
    ProcessSignal,
    ResourceBudget,
    ViewMode,
)
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class ProcessExitArgs(BaseModel):
    payload: dict[str, Any] | None = Field(default=None, description="Optional structured final result.")
    result_oid: str | None = Field(default=None, description="Existing object id to use as process result.")
    message: str | None = Field(default=None, description="Optional status message.")

    @field_validator("payload", mode="before")
    @classmethod
    def parse_json_payload(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return {"content": value}
            if isinstance(decoded, dict):
                return decoded
            return {"value": decoded}
        return value


class ProcessExitOutput(BaseModel):
    status: str
    result_oid: str | None = None


class GetWorkingDirectoryArgs(BaseModel):
    pass


class GetWorkingDirectoryOutput(BaseModel):
    working_directory: str


class SetWorkingDirectoryArgs(BaseModel):
    path: str = Field(description="Workspace-relative directory, resolved from the current process working directory.")


class SetWorkingDirectoryOutput(BaseModel):
    working_directory: str


class ExecProcessArgs(BaseModel):
    image: str = Field(description="Target AgentImage id for the current process.")
    args: dict[str, Any] = Field(default_factory=dict, description="Structured exec arguments recorded in audit.")
    goal: str | dict[str, Any] | None = Field(
        default=None,
        description="Optional replacement goal for the new image.",
    )
    preserve_memory: bool = Field(default=True, description="Keep the current MemoryView unless a replacement is requested.")
    preserve_capabilities: bool = Field(
        default=False,
        description="Keep existing capabilities. Exec never grants capabilities required by the target image.",
    )


class ExecProcessOutput(BaseModel):
    pid: str
    old_image: str
    new_image: str
    status: str
    goal_oid: str | None
    preserve_memory: bool
    preserve_capabilities: bool
    active_tools: list[str]


class ForkChildProcessArgs(BaseModel):
    goal: str | dict[str, Any] = Field(description="Goal for the child AgentProcess.")
    mode: str = Field(default=ForkMode.WORKER.value, description="Fork mode: copy, restricted, speculative, or worker.")
    image: str | None = Field(
        default=None,
        description="Optional image id. MVP tool only allows the current process image.",
    )
    include_parent_roots: bool = Field(
        default=True,
        description="Whether the child view includes the parent's current MemoryView roots.",
    )
    root_oids: list[str] | None = Field(
        default=None,
        description="Optional explicit Object ids to expose to the child instead of all parent roots.",
    )
    inherit_read_files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative files whose read capability should be inherited by the child.",
    )
    inherit_write_files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative files whose write capability should be inherited by the child.",
    )
    inherit_read_dirs: list[str] = Field(
        default_factory=list,
        description="Workspace-relative directories whose read capability should be inherited by the child.",
    )
    inherit_write_dirs: list[str] = Field(
        default_factory=list,
        description="Workspace-relative directories whose write capability should be inherited by the child.",
    )
    inherit_capabilities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Explicit capability specs to inherit, each with resource and rights.",
    )
    working_directory: str | None = Field(
        default=None,
        description="Optional child working directory. Defaults to the parent's current working directory.",
    )
    resource_budget: dict[str, Any] | None = Field(
        default=None,
        description="Optional child ResourceBudget override. It must fit inside the parent process remaining budget.",
    )


class ForkChildProcessOutput(BaseModel):
    child_pid: str
    parent_pid: str
    image: str
    mode: str
    status: str
    goal_oid: str | None
    inherited_capabilities: list[dict[str, Any]]
    working_directory: str


class SpawnChildProcessArgs(BaseModel):
    goal: str | dict[str, Any] = Field(description="Goal for the fresh child AgentProcess.")
    image: str | None = Field(default=None, description="Optional child image id. Defaults to the parent image.")
    inherit_read_files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative files whose read capability should be inherited by the child.",
    )
    inherit_write_files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative files whose write capability should be inherited by the child.",
    )
    inherit_read_dirs: list[str] = Field(
        default_factory=list,
        description="Workspace-relative directories whose read capability should be inherited by the child.",
    )
    inherit_write_dirs: list[str] = Field(
        default_factory=list,
        description="Workspace-relative directories whose write capability should be inherited by the child.",
    )
    inherit_capabilities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Explicit capability specs to inherit, each with resource and rights.",
    )
    working_directory: str | None = Field(
        default=None,
        description="Optional child working directory. Defaults to the parent's current working directory.",
    )
    resource_budget: dict[str, Any] | None = Field(
        default=None,
        description="Optional child ResourceBudget override. It must fit inside the parent process remaining budget.",
    )


class SpawnChildProcessOutput(BaseModel):
    child_pid: str
    parent_pid: str
    image: str
    status: str
    goal_oid: str | None
    inherited_capabilities: list[dict[str, Any]]
    fresh_memory_view: bool
    working_directory: str


class WaitChildProcessArgs(BaseModel):
    child_pid: str = Field(description="Direct child process id to wait for.")
    block: bool = Field(default=True, description="If false, return ready=false when the child is still running.")


class WaitChildProcessOutput(BaseModel):
    child_pid: str
    status: str
    ready: bool
    result_oid: str | None = None
    message: str | None = None


class ChildProcessInfo(BaseModel):
    pid: str
    image: str
    status: str
    working_directory: str
    goal_oid: str | None
    result_oid: str | None = None
    status_message: str | None = None


class ListChildProcessesArgs(BaseModel):
    include_terminal: bool = Field(default=True, description="Whether exited/failed/killed children are included.")


class ListChildProcessesOutput(BaseModel):
    children: list[ChildProcessInfo]


class SignalChildProcessArgs(BaseModel):
    child_pid: str = Field(description="Direct child process id to signal.")
    signal: str = Field(description="Signal to send: pause, resume, cancel, or terminate.")
    reason: str | None = Field(default=None, description="Optional reason stored in the child status message.")


class SignalChildProcessOutput(BaseModel):
    child_pid: str
    signal: str
    status: str


class MergeChildMemoryArgs(BaseModel):
    child_pid: str = Field(description="Direct child process id whose memory should be merged.")
    include_child_created: bool = Field(default=True, description="Include objects created by the child.")


class MergeChildMemoryOutput(BaseModel):
    child_pid: str
    merged_oids: list[str]
    skipped_oids: list[str]


class ProcessExitTool(SyncAgentTool[ProcessExitArgs]):
    name = "process_exit"
    description = (
        "Exit the current Agent Process with an optional final result. "
        "This is a Skills/Tools Layer wrapper over process lifecycle primitives."
    )
    args_schema = ProcessExitArgs
    output_schema = ProcessExitOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.read", "object.write", "process.lifecycle"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "lifecycle"]

    def run(self, args: ProcessExitArgs, ctx: ToolContext) -> ProcessExitOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        result_handle: ObjectHandle | None = None
        if args.result_oid:
            result_handle = runtime.memory.handle_for_oid(
                ctx.pid,
                args.result_oid,
                required_rights={ObjectRight.READ.value},
                optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value, ObjectRight.DIFF.value},
                issued_by="process_exit_tool",
            )
        elif args.payload is not None:
            result_handle = runtime.memory.create_object(
                pid=ctx.pid,
                object_type=ObjectType.SUMMARY,
                payload=args.payload,
                metadata=ObjectMetadata(title="Process final result", tags=["final"]),
            )
        runtime.process.exit(ctx.pid, result=result_handle, message=args.message)
        result_oid = result_handle.oid if result_handle is not None else None
        return ProcessExitOutput(status="exited", result_oid=result_oid)


class GetWorkingDirectoryTool(SyncAgentTool[GetWorkingDirectoryArgs]):
    name = "get_working_directory"
    description = "Return this AgentProcess working directory, relative to the runtime workspace root."
    args_schema = GetWorkingDirectoryArgs
    output_schema = GetWorkingDirectoryOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.standard_timeout_s)
    tags = ["process", "working_directory"]

    def run(self, args: GetWorkingDirectoryArgs, ctx: ToolContext) -> GetWorkingDirectoryOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        return GetWorkingDirectoryOutput(working_directory=runtime.process.working_directory(ctx.pid))


class SetWorkingDirectoryTool(SyncAgentTool[SetWorkingDirectoryArgs]):
    name = "set_working_directory"
    description = (
        "Set this AgentProcess working directory. Relative filesystem and shell tool paths resolve from it."
    )
    args_schema = SetWorkingDirectoryArgs
    output_schema = SetWorkingDirectoryOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"process.cwd"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "working_directory"]

    def run(self, args: SetWorkingDirectoryArgs, ctx: ToolContext) -> SetWorkingDirectoryOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        process = runtime.set_process_working_directory(ctx.pid, args.path)
        return SetWorkingDirectoryOutput(working_directory=process.working_directory)


class ExecProcessTool(SyncAgentTool[ExecProcessArgs]):
    name = "exec_process"
    description = (
        "Replace the current AgentProcess image and tool table without changing pid. "
        "Exec is not a permission escalation path: target image required capabilities are not granted automatically."
    )
    args_schema = ExecProcessArgs
    output_schema = ExecProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write", "process.lifecycle", "tool.table"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "lifecycle", "exec"]

    def run(self, args: ExecProcessArgs, ctx: ToolContext) -> ExecProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        old_image = runtime.process.get(ctx.pid).image_id
        try:
            process = runtime.exec_process(
                ctx.pid,
                args.image,
                args=args.args,
                goal=args.goal,
                preserve_memory=args.preserve_memory,
                preserve_capabilities=args.preserve_capabilities,
            )
        except NotFound as exc:
            raise ToolExecutionError(
                "Target image does not exist.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"image": args.image},
            ) from exc
        return ExecProcessOutput(
            pid=process.pid,
            old_image=old_image,
            new_image=process.image_id,
            status=process.status.value,
            goal_oid=process.goal_oid,
            preserve_memory=args.preserve_memory,
            preserve_capabilities=args.preserve_capabilities,
            active_tools=sorted(process.tool_table),
        )


class ForkChildProcessTool(SyncAgentTool[ForkChildProcessArgs]):
    name = "fork_child_process"
    description = (
        "Fork a direct child AgentProcess with an attenuated MemoryView. "
        "This creates an Agent libOS child process, not a host OS process."
    )
    args_schema = ForkChildProcessArgs
    output_schema = ForkChildProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"capability.write", "object.read", "object.write", "process.spawn"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "fork"]

    def run(self, args: ForkChildProcessArgs, ctx: ToolContext) -> ForkChildProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        parent = runtime.process.get(ctx.pid)
        image = args.image or parent.image_id
        if image != parent.image_id:
            raise ToolExecutionError(
                "Forking into a different image is not exposed to processes yet.",
                code=ToolErrorCode.PERMISSION_DENIED,
                details={"requested_image": image, "parent_image": parent.image_id},
            )
        try:
            fork_mode = ForkMode(args.mode)
        except ValueError as exc:
            raise ToolExecutionError(
                "Invalid fork mode.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"mode": args.mode, "allowed": [mode.value for mode in ForkMode]},
            ) from exc
        roots = self._selected_roots(runtime, ctx.pid, args.root_oids)
        view_spec = MemoryViewSpec(
            roots=roots,
            mode=_view_mode_for_fork(fork_mode),
            include_parent_roots=args.include_parent_roots,
        )
        inherit_specs = self._inheritance_specs(runtime, args, cwd=parent.working_directory)
        child_cwd = (
            runtime.resolve_process_working_directory(ctx.pid, args.working_directory)
            if args.working_directory is not None
            else parent.working_directory
        )
        child_pid = runtime.fork_child_process(
            parent=ctx.pid,
            goal=args.goal,
            memory_view=view_spec,
            inherit_capabilities=inherit_specs,
            resource_budget=_resource_budget_from_spec(args.resource_budget),
            image=image,
            mode=fork_mode,
            working_directory=child_cwd,
        )
        child = runtime.process.get(child_pid)
        return ForkChildProcessOutput(
            child_pid=child.pid,
            parent_pid=ctx.pid,
            image=child.image_id,
            mode=fork_mode.value,
            status=child.status.value,
            goal_oid=child.goal_oid,
            inherited_capabilities=inherit_specs,
            working_directory=child.working_directory,
        )

    def _selected_roots(self, runtime: Any, pid: str, root_oids: list[str] | None) -> list[ObjectHandle] | None:
        if root_oids is None:
            return None
        process = runtime.process.get(pid)
        visible = {handle.oid: handle for handle in (process.memory_view.roots if process.memory_view else [])}
        roots: list[ObjectHandle] = []
        for oid in root_oids:
            if oid in visible:
                roots.append(visible[oid])
                continue
            roots.append(
                runtime.memory.handle_for_oid(
                    pid,
                    oid,
                    required_rights={ObjectRight.READ.value},
                    optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.DIFF.value},
                    issued_by="fork_child_process_tool",
                )
            )
        return roots

    def _inheritance_specs(
        self,
        runtime: Any,
        args: ForkChildProcessArgs,
        *,
        cwd: str,
    ) -> list[dict[str, Any]]:
        specs = [_normalize_capability_spec(spec) for spec in args.inherit_capabilities]
        for path in args.inherit_read_files:
            specs.append({"resource": runtime.filesystem.resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.READ.value]})
        for path in args.inherit_write_files:
            specs.append({"resource": runtime.filesystem.resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.WRITE.value]})
        for path in args.inherit_read_dirs:
            specs.append(
                {"resource": runtime.filesystem.directory_resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.READ.value]}
            )
        for path in args.inherit_write_dirs:
            specs.append(
                {"resource": runtime.filesystem.directory_resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.WRITE.value]}
            )
        return _coalesce_capability_specs(specs)


class SpawnChildProcessTool(SyncAgentTool[SpawnChildProcessArgs]):
    name = "spawn_child_process"
    description = (
        "Create a fresh direct child AgentProcess with a new process namespace and goal-only MemoryView. "
        "Unlike fork_child_process, this does not copy parent memory roots."
    )
    args_schema = SpawnChildProcessArgs
    output_schema = SpawnChildProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"capability.write", "object.write", "process.spawn"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "spawn"]

    def run(self, args: SpawnChildProcessArgs, ctx: ToolContext) -> SpawnChildProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        inherit_specs = self._inheritance_specs(runtime, args, cwd=runtime.process.working_directory(ctx.pid))
        child_pid = runtime.spawn_child_process(
            parent=ctx.pid,
            goal=args.goal,
            image=args.image,
            inherit_capabilities=inherit_specs,
            resource_budget=_resource_budget_from_spec(args.resource_budget),
            working_directory=args.working_directory,
        )
        child = runtime.process.get(child_pid)
        return SpawnChildProcessOutput(
            child_pid=child.pid,
            parent_pid=ctx.pid,
            image=child.image_id,
            status=child.status.value,
            goal_oid=child.goal_oid,
            inherited_capabilities=inherit_specs,
            fresh_memory_view=True,
            working_directory=child.working_directory,
        )

    def _inheritance_specs(
        self,
        runtime: Any,
        args: SpawnChildProcessArgs,
        *,
        cwd: str,
    ) -> list[dict[str, Any]]:
        specs = [_normalize_capability_spec(spec) for spec in args.inherit_capabilities]
        for path in args.inherit_read_files:
            specs.append({"resource": runtime.filesystem.resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.READ.value]})
        for path in args.inherit_write_files:
            specs.append({"resource": runtime.filesystem.resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.WRITE.value]})
        for path in args.inherit_read_dirs:
            specs.append(
                {"resource": runtime.filesystem.directory_resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.READ.value]}
            )
        for path in args.inherit_write_dirs:
            specs.append(
                {"resource": runtime.filesystem.directory_resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.WRITE.value]}
            )
        return _coalesce_capability_specs(specs)


class WaitChildProcessTool(SyncAgentTool[WaitChildProcessArgs]):
    name = "wait_child_process"
    description = (
        "Wait for a direct child AgentProcess to exit, fail, or be killed. "
        "If the child is still running and block=true, the current process is suspended and the same wait resumes later."
    )
    args_schema = WaitChildProcessArgs
    output_schema = WaitChildProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"process.wait"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "wait"]

    def run(self, args: WaitChildProcessArgs, ctx: ToolContext) -> WaitChildProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        try:
            result = runtime.process.wait(
                ctx.pid,
                args.child_pid,
                timeout=None if args.block else 0,
            )
        except TimeoutError:
            child = runtime.process.get(args.child_pid)
            return WaitChildProcessOutput(
                child_pid=args.child_pid,
                status=child.status.value,
                ready=False,
                message=child.status_message,
            )
        return WaitChildProcessOutput(
            child_pid=result.pid,
            status=result.status.value,
            ready=True,
            result_oid=result.result.oid if result.result is not None else None,
            message=result.message,
        )


class ListChildProcessesTool(SyncAgentTool[ListChildProcessesArgs]):
    name = "list_child_processes"
    description = "List direct child AgentProcesses owned by the current process."
    args_schema = ListChildProcessesArgs
    output_schema = ListChildProcessesOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.standard_timeout_s)
    tags = ["process", "child", "inspect"]

    def run(self, args: ListChildProcessesArgs, ctx: ToolContext) -> ListChildProcessesOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        return ListChildProcessesOutput(
            children=[_child_info(child) for child in runtime.process.list_children(ctx.pid, args.include_terminal)]
        )


class SignalChildProcessTool(SyncAgentTool[SignalChildProcessArgs]):
    name = "signal_child_process"
    description = "Pause, resume, cancel, or terminate a direct child AgentProcess."
    args_schema = SignalChildProcessArgs
    output_schema = SignalChildProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"process.signal"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "signal"]

    def run(self, args: SignalChildProcessArgs, ctx: ToolContext) -> SignalChildProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        try:
            signal = ProcessSignal(args.signal)
        except ValueError as exc:
            raise ToolExecutionError(
                "Invalid process signal.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"signal": args.signal, "allowed": ["pause", "resume", "cancel", "terminate"]},
            ) from exc
        if signal not in {ProcessSignal.PAUSE, ProcessSignal.RESUME, ProcessSignal.CANCEL, ProcessSignal.TERMINATE}:
            raise ToolExecutionError(
                "Signal is not exposed through this tool.",
                code=ToolErrorCode.PERMISSION_DENIED,
                details={"signal": signal.value},
            )
        child = runtime.process.signal_child(ctx.pid, args.child_pid, signal, reason=args.reason)
        return SignalChildProcessOutput(child_pid=child.pid, signal=signal.value, status=child.status.value)


class MergeChildMemoryTool(SyncAgentTool[MergeChildMemoryArgs]):
    name = "merge_child_memory"
    description = "Merge result-visible Object Memory from an exited direct child into the parent process view."
    args_schema = MergeChildMemoryArgs
    output_schema = MergeChildMemoryOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.link", "object.write"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "memory"]

    def run(self, args: MergeChildMemoryArgs, ctx: ToolContext) -> MergeChildMemoryOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        result = runtime.process.merge_child_memory(
            ctx.pid,
            args.child_pid,
            policy=MergePolicy(include_child_created=args.include_child_created),
        )
        return MergeChildMemoryOutput(
            child_pid=args.child_pid,
            merged_oids=result.merged_oids,
            skipped_oids=result.skipped_oids,
        )


def _view_mode_for_fork(mode: ForkMode) -> ViewMode:
    if mode == ForkMode.COPY:
        return ViewMode.COPY_ON_WRITE
    if mode == ForkMode.SPECULATIVE:
        return ViewMode.EPHEMERAL
    return ViewMode.READ_ONLY


def _child_info(child: AgentProcess) -> ChildProcessInfo:
    result_oid = None
    if child.status_message and child.status_message.startswith("result_oid:"):
        result_oid = child.status_message.split(":", 1)[1]
    return ChildProcessInfo(
        pid=child.pid,
        image=child.image_id,
        status=child.status.value,
        working_directory=child.working_directory,
        goal_oid=child.goal_oid,
        result_oid=result_oid,
        status_message=child.status_message,
    )


def _normalize_capability_spec(spec: dict[str, Any]) -> dict[str, Any]:
    resource = spec.get("resource")
    if not isinstance(resource, str) or not resource:
        raise ToolExecutionError(
            "Inherited capability spec requires a non-empty resource.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"spec": spec},
        )
    rights = spec.get("rights", [CapabilityRight.READ.value])
    if not isinstance(rights, list) or not rights:
        raise ToolExecutionError(
            "Inherited capability spec requires a non-empty rights list.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"spec": spec},
        )
    normalized: dict[str, Any] = {"resource": resource, "rights": [str(right) for right in rights]}
    constraints = spec.get("constraints")
    if isinstance(constraints, dict):
        normalized["constraints"] = constraints
    return normalized


def _resource_budget_from_spec(spec: dict[str, Any] | None) -> ResourceBudget | None:
    if spec is None:
        return None
    allowed = set(ResourceBudget.__dataclass_fields__)
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ToolExecutionError(
            "Unknown resource_budget fields.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"unknown_fields": unknown},
        )
    try:
        return ResourceBudget(**{key: value for key, value in spec.items() if key in allowed})
    except ValueError as exc:
        raise ToolExecutionError(
            "Invalid resource_budget.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"error": str(exc)},
        ) from exc


def _coalesce_capability_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_resource: dict[str, dict[str, Any]] = {}
    for spec in specs:
        resource = str(spec["resource"])
        current = by_resource.setdefault(resource, {"resource": resource, "rights": []})
        current["rights"] = sorted(set(current["rights"]) | {str(right) for right in spec.get("rights", [])})
        if isinstance(spec.get("constraints"), dict):
            current["constraints"] = dict(spec["constraints"])
    return list(by_resource.values())
