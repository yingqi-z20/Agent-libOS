from __future__ import annotations

import asyncio
import inspect
from typing import Any, TYPE_CHECKING

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    CapabilityEffect,
    CapabilityRight,
    CapabilitySpec,
    ForkMode,
    HumanRequestStatus,
    MemoryViewSpec,
    MergePolicy,
    ObjectHandle,
    ObjectMetadata,
    ObjectRight,
    ObjectType,
    ProcessMessageKind,
    ProcessSignal,
    ProcessStatus,
    ResourceUsage,
    ViewMode,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    ValidationError,
)
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import to_jsonable

if TYPE_CHECKING:
    from agent_libos.runtime.runtime import Runtime


BUILTIN_SYSCALL_NAMES = {
    "ask_human",
    "capability.delegate",
    "capability.inspect",
    "capability.list",
    "capability.request_permission",
    "capability.revoke",
    "checkpoint.create",
    "checkpoint.diff",
    "checkpoint.fork",
    "checkpoint.fork_from_checkpoint",
    "checkpoint.inspect",
    "checkpoint.list",
    "checkpoint.replay",
    "checkpoint.replay_to_event",
    "checkpoint.restore",
    "clock.now",
    "clock.sleep",
    "filesystem.delete_directory",
    "filesystem.delete_file",
    "filesystem.list_directory",
    "filesystem.make_directory",
    "filesystem.read_directory",
    "filesystem.read_text",
    "filesystem.read_text_file",
    "filesystem.write_directory",
    "filesystem.write_text",
    "filesystem.write_text_file",
    "human.ask",
    "human.output",
    "human.request_permission",
    "human_output",
    "image.commit_checkpoint",
    "image.inspect",
    "image.list",
    "image.load_package",
    "jsonrpc.call",
    "jsonrpc.inspect",
    "jsonrpc.list",
    "memory.append_memory_object",
    "memory.append_object",
    "memory.create_namespace",
    "memory.create_object",
    "memory.get_object",
    "memory.list_namespace",
    "memory.read_object",
    "permission.request",
    "process.chdir",
    "process.cwd",
    "process.exec",
    "process.exit",
    "process.fork",
    "process.get_working_directory",
    "process.list_children",
    "process.merge_child_memory",
    "process.read_messages",
    "process.receive_messages",
    "process.send_message",
    "process.set_working_directory",
    "process.signal",
    "process.spawn_child",
    "process.wait",
    "request_permission",
    "shell.run",
    "shell.run_command",
    "skill.activate",
    "skill.discover",
    "skill.inspect",
    "skill.read_resource",
    "skill.register_path",
    "skill.unload",
    "sleep",
    "time.now",
}


class LibOSSyscallSession:
    """Per JIT tool-call syscall session.

    Syscalls are libOS primitive calls made as the AgentProcess pid. They do not
    consult the process tool table; the primitives remain responsible for
    capability checks, human approval, audit, and events.
    """

    TERMINAL_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}

    def __init__(self, runtime: "Runtime", pid: str, config: AgentLibOSConfig | None = None) -> None:
        self.runtime = runtime
        self.pid = pid
        self.config = config or DEFAULT_CONFIG
        self._deferred_exit: dict[str, Any] | None = None
        self._deferred_exec: dict[str, Any] | None = None
        self._tracked_wait_states: set[tuple[ProcessStatus, str]] = set()

    async def handle(self, name: str, args: dict[str, Any]) -> Any:
        normalized = name.strip()
        if not normalized:
            raise ValidationError("syscall name must be non-empty")
        self._charge_syscall(normalized)
        self.runtime.audit.record(
            actor=self.pid,
            action="syscall.request",
            target=normalized,
            decision={"args": sanitize_for_observability(args)},
        )
        try:
            result = await self._with_blocking(lambda: self._dispatch(normalized, args))
        except asyncio.CancelledError:
            preserve_wait = (asyncio.current_task().get_name() if asyncio.current_task() else "").startswith(
                "agent-process:"
            )
            if not preserve_wait:
                self._cleanup_interrupted_wait(normalized)
            self.runtime.audit.record(
                actor=self.pid,
                action="syscall.cancelled",
                target=normalized,
                decision={"wait_state_preserved": preserve_wait},
            )
            raise
        except BaseException:
            self._cleanup_interrupted_wait(normalized)
            raise
        self.runtime.audit.record(
            actor=self.pid,
            action="syscall.result",
            target=normalized,
            decision={"ok": True},
        )
        return to_jsonable(result)

    def _charge_syscall(self, name: str) -> None:
        resources = getattr(self.runtime, "resources", None)
        if resources is None:
            return
        resources.charge(
            self.pid,
            ResourceUsage(deno_syscalls=1),
            source="deno.syscall",
            context={"syscall": name},
            allow_overage=False,
            kill_on_exceed=False,
        )

    async def apply_deferred_lifecycle(self, tool_result: ObjectHandle | None = None) -> None:
        if self._deferred_exec is not None:
            exec_args = self._deferred_exec
            self.runtime.exec_process(
                self.pid,
                str(exec_args["image"]),
                args=exec_args.get("args"),
                goal=exec_args.get("goal"),
                preserve_memory=bool(exec_args.get("preserve_memory", True)),
                preserve_capabilities=bool(exec_args.get("preserve_capabilities", False)),
            )
        if self._deferred_exit is not None:
            exit_args = self._deferred_exit
            result_handle = self._result_handle_for_exit(exit_args, tool_result)
            self.runtime.process.exit(
                self.pid,
                result=result_handle,
                failed=bool(exit_args.get("failed", False)),
                message=exit_args.get("message"),
            )

    async def _with_blocking(self, operation: Any) -> Any:
        while True:
            try:
                result = operation()
                if inspect.isawaitable(result):
                    return await result
                return result
            except HumanApprovalRequired as exc:
                self._remember_wait_state()
                await self._resolve_human_request(exc.request_id)
            except ProcessWaitRequired as exc:
                self._remember_wait_state()
                await self._wait_for_child_terminal(exc.child_pid)
            except ProcessMessageWaitRequired as exc:
                self._remember_wait_state()
                await self._wait_for_process_message(exc.recipient_pid, exc.filters)

    def _remember_wait_state(self) -> None:
        process = self.runtime.store.get_process(self.pid)
        if process is None or process.status not in {ProcessStatus.WAITING_EVENT, ProcessStatus.WAITING_HUMAN}:
            return
        if not process.status_message:
            return
        self._tracked_wait_states.add((process.status, process.status_message))

    def _cleanup_interrupted_wait(self, syscall_name: str) -> None:
        process = self.runtime.store.get_process(self.pid)
        if process is None or not process.status_message:
            return
        if (process.status, process.status_message) not in self._tracked_wait_states:
            return
        process.status = ProcessStatus.RUNNABLE
        process.status_message = None
        process.updated_at = utc_now()
        self.runtime.store.update_process(process)
        self.runtime.audit.record(
            actor=self.pid,
            action="syscall.wait_interrupted",
            target=syscall_name,
            decision={"restored_status": ProcessStatus.RUNNABLE.value},
        )

    async def _resolve_human_request(self, request_id: str, *, allow_rejected: bool = False) -> None:
        while True:
            request = self.runtime.human.get(request_id)
            if request.status == HumanRequestStatus.APPROVED:
                return
            if allow_rejected and request.status == HumanRequestStatus.REJECTED:
                return
            if request.status != HumanRequestStatus.PENDING:
                raise CapabilityDenied(f"human request was not approved: {request_id} status={request.status.value}")
            processed = await self.runtime.human.aprocess_next_terminal(
                human=request.human,
                auto_approve=getattr(self.runtime, "_current_human_auto_approve", None),
                auto_policy=getattr(self.runtime, "_current_human_auto_policy", None),
                auto_answer=getattr(self.runtime, "_current_human_auto_answer", None),
            )
            if processed is None:
                await asyncio.sleep(self.runtime.scheduler.poll_interval_s)

    async def _wait_for_child_terminal(self, child_pid: str) -> None:
        while True:
            child = self.runtime.process.get(child_pid)
            if child.status in self.TERMINAL_STATUSES:
                return
            await asyncio.sleep(self.runtime.scheduler.poll_interval_s)

    async def _wait_for_process_message(self, pid: str, filters: dict[str, Any]) -> None:
        while True:
            messages = self.runtime.messages.unread(
                pid,
                kind=filters.get("kind"),
                sender=filters.get("sender"),
                channel=filters.get("channel"),
                correlation_id=filters.get("correlation_id"),
                reply_to=filters.get("reply_to"),
                message_ids=filters.get("message_ids"),
            )
            if messages:
                return
            await asyncio.sleep(self.runtime.scheduler.poll_interval_s)

    def _dispatch(self, name: str, args: dict[str, Any]) -> Any:
        registered = self.runtime.syscalls.get(name)
        if registered is not None:
            return registered.handler(self, args)
        if name in {"filesystem.read_text", "filesystem.read_text_file"}:
            return self._filesystem_read_text(args)
        if name in {"filesystem.write_text", "filesystem.write_text_file"}:
            return self._filesystem_write_text(args)
        if name in {"filesystem.read_directory", "filesystem.list_directory"}:
            return self._filesystem_read_directory(args)
        if name in {"filesystem.write_directory", "filesystem.make_directory"}:
            return self._filesystem_write_directory(args)
        if name == "filesystem.delete_file":
            return self._filesystem_delete_file(args)
        if name == "filesystem.delete_directory":
            return self._filesystem_delete_directory(args)
        if name == "memory.create_namespace":
            ns = self.runtime.memory.create_namespace(
                self.pid,
                namespace=str(args["namespace"]),
                parent_namespace=args.get("parent_namespace"),
                metadata=dict(args.get("metadata") or {}),
            )
            return {"namespace": ns.namespace, "parent_namespace": ns.parent_namespace, "created": True}
        if name == "memory.list_namespace":
            listing = self.runtime.memory.list_namespace(self.pid, args.get("namespace"))
            return {
                "namespace": listing["namespace"],
                "objects": [
                    {
                        "oid": obj.oid,
                        "namespace": obj.namespace,
                        "name": obj.name,
                        "type": obj.type.value,
                        "version": obj.version,
                    }
                    for obj in listing["objects"]
                ],
                "namespaces": [
                    {"namespace": ns.namespace, "parent_namespace": ns.parent_namespace}
                    for ns in listing["namespaces"]
                ],
            }
        if name == "memory.create_object":
            return self._memory_create_object(args)
        if name in {"memory.read_object", "memory.get_object"}:
            obj = self.runtime.memory.get_object_by_name(self.pid, str(args["name"]), namespace=args.get("namespace"))
            return {
                "oid": obj.oid,
                "namespace": obj.namespace,
                "name": obj.name,
                "type": obj.type.value,
                "version": obj.version,
                "payload": obj.payload,
            }
        if name in {"memory.append_object", "memory.append_memory_object"}:
            return self._memory_append_object(args)
        if name in {"human.output", "human_output"}:
            return self.runtime.human.output(
                pid=self.pid,
                message=str(args.get("message", "")),
                human=str(args.get("human") or self.config.runtime.default_human),
                channel=str(args.get("channel") or self.config.runtime.terminal_channel),
            )
        if name in {"human.ask", "ask_human"}:
            return self._human_ask(args)
        if name in {"human.request_permission", "permission.request", "request_permission"}:
            return self._request_permission(args)
        if name == "capability.list":
            return self._capability_list(args)
        if name == "capability.inspect":
            return self._capability_inspect(args)
        if name == "capability.request_permission":
            return self._request_permission(args)
        if name == "capability.delegate":
            return self._capability_delegate(args)
        if name == "capability.revoke":
            return self._capability_revoke(args)
        if name in {"clock.now", "time.now"}:
            return self.runtime.clock.now(self.pid, tz=str(args.get("timezone") or self.config.tools.clock_timezone))
        if name in {"clock.sleep", "sleep"}:
            return self.runtime.clock.asleep(self.pid, float(args.get("seconds", 0)))
        if name == "jsonrpc.list":
            return {"endpoints": self.runtime.jsonrpc.list_endpoints(actor=self.pid)}
        if name == "jsonrpc.inspect":
            return self.runtime.jsonrpc.inspect_endpoint(str(args["endpoint_id"]), actor=self.pid)
        if name == "jsonrpc.call":
            return self._jsonrpc_call(args)
        if name in {"process.get_working_directory", "process.cwd"}:
            return {"working_directory": self.runtime.process.working_directory(self.pid)}
        if name in {"process.set_working_directory", "process.chdir"}:
            process = self.runtime.set_process_working_directory(self.pid, str(args["path"]))
            return {"working_directory": process.working_directory}
        if name == "process.fork":
            return self._process_fork(args)
        if name == "process.spawn_child":
            return self._process_spawn_child(args)
        if name == "process.wait":
            return self._process_wait(args)
        if name == "process.list_children":
            return {
                "children": [
                    {
                        "pid": child.pid,
                        "image": child.image_id,
                        "status": child.status.value,
                        "working_directory": child.working_directory,
                        "goal_oid": child.goal_oid,
                        "status_message": child.status_message,
                    }
                    for child in self.runtime.process.list_children(
                        self.pid,
                        include_terminal=bool(args.get("include_terminal", True)),
                    )
                ]
            }
        if name == "process.signal":
            child = self.runtime.process.signal_child(
                self.pid,
                str(args["child_pid"]),
                ProcessSignal(str(args["signal"])),
                reason=args.get("reason"),
            )
            return {"child_pid": child.pid, "status": child.status.value, "signal": str(args["signal"])}
        if name == "process.merge_child_memory":
            result = self.runtime.process.merge_child_memory(
                self.pid,
                str(args["child_pid"]),
                policy=MergePolicy(include_child_created=bool(args.get("include_child_created", True))),
            )
            return result
        if name == "process.send_message":
            message = self.runtime.messages.send_from_process(
                self.pid,
                str(args["recipient_pid"]),
                kind=ProcessMessageKind(str(args.get("kind", ProcessMessageKind.NORMAL.value))),
                channel=str(args.get("channel", "default")),
                correlation_id=str(args["correlation_id"]) if args.get("correlation_id") is not None else None,
                reply_to=str(args["reply_to"]) if args.get("reply_to") is not None else None,
                subject=str(args.get("subject", "")),
                body=str(args.get("body", "")),
                payload=dict(args.get("payload") or {}),
            )
            return self._process_message_result(message)
        if name == "process.read_messages":
            return self._process_read_messages(args, default_block=False)
        if name == "process.receive_messages":
            return self._process_read_messages(args, default_block=True)
        if name == "process.exec":
            self._deferred_exec = dict(args)
            return {"deferred": True, "operation": "process.exec", "image": args.get("image")}
        if name == "process.exit":
            self._deferred_exit = dict(args)
            return {"deferred": True, "operation": "process.exit"}
        if name == "checkpoint.create":
            checkpoint_id = self.runtime.checkpoint.create(
                self.pid,
                str(args.get("reason", "checkpoint syscall")),
                actor=self.pid,
                metadata=dict(args.get("metadata") or {}),
            )
            return {"checkpoint_id": checkpoint_id, "pid": self.pid}
        if name == "checkpoint.list":
            return {
                "checkpoints": self.runtime.checkpoint.list(
                    str(args.get("pid") or self.pid),
                    actor=self.pid,
                    limit=int(args["limit"]) if args.get("limit") is not None else None,
                )
            }
        if name == "checkpoint.inspect":
            return self.runtime.checkpoint.inspect(str(args["checkpoint_id"]), actor=self.pid)
        if name == "checkpoint.diff":
            return self.runtime.checkpoint.diff(str(args["checkpoint_id"]), actor=self.pid)
        if name == "checkpoint.restore":
            return self.runtime.checkpoint.restore(self.pid, str(args["checkpoint_id"]))
        if name in {"checkpoint.fork", "checkpoint.fork_from_checkpoint"}:
            return self.runtime.checkpoint.fork_from_checkpoint(
                self.pid,
                str(args["checkpoint_id"]),
                parent_pid=str(args["parent_pid"]) if args.get("parent_pid") is not None else None,
            )
        if name in {"checkpoint.replay", "checkpoint.replay_to_event"}:
            return self.runtime.checkpoint.replay_to_event(
                str(args["checkpoint_id"]),
                str(args["event_id"]),
                actor=self.pid,
            )
        if name == "skill.discover":
            return {
                "skills": self.runtime.skills.discover_skills(
                    text=str(args["text"]) if args.get("text") is not None else None,
                    actor=self.pid,
                    limit=int(args["limit"]) if args.get("limit") is not None else None,
                )
            }
        if name == "skill.inspect":
            return self.runtime.skills.inspect_skill(str(args["skill_id"]), actor=self.pid)
        if name == "skill.register_path":
            return self.runtime.skills.register_skill_from_workspace_path(
                self.pid,
                str(args["path"]),
                replace=bool(args.get("replace", False)),
            )
        if name == "skill.activate":
            return self.runtime.skills.activate_skill(self.pid, str(args["skill_id"]), actor=self.pid)
        if name == "skill.unload":
            return self.runtime.skills.unload_skill(self.pid, str(args["skill_id"]), actor=self.pid)
        if name == "skill.read_resource":
            return self.runtime.skills.read_skill_resource(
                self.pid,
                str(args["skill_id"]),
                str(args["path"]),
                actor=self.pid,
                max_bytes=int(args["max_bytes"]) if args.get("max_bytes") is not None else None,
            )
        if name in {"shell.run", "shell.run_command"}:
            cwd = self.runtime.process.working_directory(self.pid)
            return self.runtime.shell.arun(
                self.pid,
                self._string_list_arg(args, "argv"),
                timeout=float(args.get("timeout_s", self.config.tools.shell_timeout_s)),
                cwd=cwd,
            )
        if name == "image.list":
            self.runtime.capability.require(self.pid, self.runtime.image_registry.registry_resource(), CapabilityRight.READ)
            return {"images": self.runtime.image_registry.list_images()}
        if name == "image.inspect":
            self.runtime.capability.require(
                self.pid,
                self.runtime.image_registry.resource_for(str(args["image_id"])),
                CapabilityRight.READ,
            )
            return self.runtime.image_registry.inspect(str(args["image_id"]))
        if name == "image.commit_checkpoint":
            result = self.runtime.image_registry.commit_from_checkpoint(
                actor=self.pid,
                checkpoint_id=str(args["checkpoint_id"]),
                image_id=str(args["image_id"]),
                name=str(args["name"]),
                version=str(args.get("version") or "v0"),
                replace=bool(args.get("replace", False)),
                metadata=dict(args.get("metadata") or {}),
                require_capability=True,
            )
            return self._image_result(result)
        if name == "image.load_package":
            result = self.runtime.image_registry.register_from_workspace_package(
                self.pid,
                str(args["path"]),
                replace=bool(args.get("replace", False)),
            )
            return self._image_result(result)
        raise NotFound(f"unknown libOS syscall: {name}")

    def _filesystem_read_text(self, args: dict[str, Any]) -> Any:
        cwd = self.runtime.process.working_directory(self.pid)
        return self.runtime.filesystem.read_text(
            pid=self.pid,
            path=str(args["path"]),
            encoding=str(args.get("encoding") or self.config.tools.default_text_encoding),
            max_bytes=int(args.get("max_bytes", self.config.tools.filesystem_read_max_bytes)),
            cwd=cwd,
        )

    def _filesystem_write_text(self, args: dict[str, Any]) -> Any:
        cwd = self.runtime.process.working_directory(self.pid)
        return self.runtime.filesystem.write_text(
            pid=self.pid,
            path=str(args["path"]),
            text=str(args.get("content", args.get("text", ""))),
            encoding=str(args.get("encoding") or self.config.tools.default_text_encoding),
            overwrite=bool(args.get("overwrite", True)),
            cwd=cwd,
        )

    def _filesystem_read_directory(self, args: dict[str, Any]) -> Any:
        cwd = self.runtime.process.working_directory(self.pid)
        return self.runtime.filesystem.read_directory(
            pid=self.pid,
            path=str(args["path"]),
            limit=int(args.get("limit", self.config.tools.directory_entry_limit)),
            cwd=cwd,
        )

    def _filesystem_write_directory(self, args: dict[str, Any]) -> Any:
        cwd = self.runtime.process.working_directory(self.pid)
        return self.runtime.filesystem.write_directory(
            pid=self.pid,
            path=str(args["path"]),
            parents=bool(args.get("parents", True)),
            exist_ok=bool(args.get("exist_ok", True)),
            cwd=cwd,
        )

    def _filesystem_delete_file(self, args: dict[str, Any]) -> Any:
        cwd = self.runtime.process.working_directory(self.pid)
        return self.runtime.filesystem.delete_file(
            pid=self.pid,
            path=str(args["path"]),
            missing_ok=bool(args.get("missing_ok", False)),
            cwd=cwd,
        )

    def _filesystem_delete_directory(self, args: dict[str, Any]) -> Any:
        cwd = self.runtime.process.working_directory(self.pid)
        return self.runtime.filesystem.delete_directory(
            pid=self.pid,
            path=str(args["path"]),
            recursive=bool(args.get("recursive", False)),
            missing_ok=bool(args.get("missing_ok", False)),
            cwd=cwd,
        )

    def _memory_create_object(self, args: dict[str, Any]) -> Any:
        metadata_arg = dict(args.get("metadata") or {})
        metadata = ObjectMetadata(
            title=metadata_arg.get("title"),
            summary=metadata_arg.get("summary"),
            tags=list(metadata_arg.get("tags", [])),
            mime_type=metadata_arg.get("mime_type"),
        )
        handle = self.runtime.memory.create_object(
            pid=self.pid,
            object_type=ObjectType(str(args.get("type", ObjectType.OBSERVATION.value))),
            payload=args.get("payload"),
            metadata=metadata,
            immutable=bool(args.get("immutable", True)),
            name=args.get("name"),
            namespace=args.get("namespace"),
        )
        self._add_handle_to_view(handle)
        obj = self.runtime.memory.get_object(self.pid, handle)
        return {"oid": handle.oid, "namespace": obj.namespace, "name": obj.name, "type": obj.type.value}

    def _memory_append_object(self, args: dict[str, Any]) -> Any:
        updated, list_field, length = self.runtime.memory.append_object_by_name(
            self.pid,
            str(args["name"]),
            args.get("entry"),
            str(args.get("list_field", "entries")),
            namespace=args.get("namespace"),
            issued_by="jit.syscall",
        )
        return {
            "oid": updated.oid,
            "namespace": updated.namespace,
            "name": updated.name,
            "version": updated.version,
            "list_field": list_field,
            "length": length,
        }

    def _human_ask(self, args: dict[str, Any]) -> Any:
        request_id = self.runtime.human.ask(
            pid=self.pid,
            human=str(args.get("human") or self.config.runtime.default_human),
            question=str(args["question"]),
            context=dict(args.get("context") or {}),
            blocking=True,
        )
        return self._answer_human_question(request_id)

    def _request_permission(self, args: dict[str, Any]) -> Any:
        request_id = self.runtime.human.request_permission(
            pid=self.pid,
            human=str(args.get("human") or self.config.runtime.default_human),
            resource=str(args["resource"]),
            rights=[str(right) for right in args.get("rights", [])],
            reason=str(args.get("reason", "")),
            blocking=True,
        )
        return self._permission_request_result(request_id, args)

    async def _jsonrpc_call(self, args: dict[str, Any]) -> dict[str, Any]:
        result = await self.runtime.jsonrpc.acall(
            self.pid,
            endpoint_id=str(args["endpoint_id"]),
            method_id=str(args["method_id"]),
            params=args.get("params"),
        )
        return to_jsonable(result)

    async def _answer_human_question(self, request_id: str) -> dict[str, Any]:
        await self._resolve_human_request(request_id)
        answer = self.runtime.human.answer_for_request(request_id)
        return {"request_id": request_id, "answer": answer, "status": "answered"}

    async def _permission_request_result(self, request_id: str, args: dict[str, Any]) -> dict[str, Any]:
        await self._resolve_human_request(request_id, allow_rejected=True)
        request = self.runtime.human.get(request_id)
        return {
            "request_id": request_id,
            "resource": str(args["resource"]),
            "rights": [str(right) for right in args.get("rights", [])],
            "status": request.status.value,
            "decision": request.decision,
        }

    def _capability_list(self, args: dict[str, Any]) -> dict[str, Any]:
        subject = str(args.get("subject") or self.pid)
        if subject != self.pid:
            raise CapabilityDenied("process syscalls may list only their own capabilities")
        caps = self.runtime.capability.list_subject(
            self.pid,
            include_inactive=bool(args.get("include_inactive", False)),
            limit=int(args["limit"]) if args.get("limit") is not None else None,
        )
        return {"capabilities": [self.runtime.capability.inspect(cap.cap_id) for cap in caps]}

    def _capability_inspect(self, args: dict[str, Any]) -> dict[str, Any]:
        cap = self.runtime.store.get_capability(str(args["capability_id"]))
        if cap is None:
            raise NotFound(f"capability not found: {args['capability_id']}")
        if cap.subject != self.pid:
            raise CapabilityDenied("process syscalls may inspect only their own capabilities")
        return {"capability": self.runtime.capability.inspect(cap.cap_id)}

    def _capability_delegate(self, args: dict[str, Any]) -> dict[str, Any]:
        child_pid = str(args["child_pid"])
        child = self.runtime.process.get(child_pid)
        if child.parent_pid != self.pid:
            raise CapabilityDenied("capability.delegate may target only a direct child process")
        cap = self.runtime.capability.delegate(
            self.pid,
            child_pid,
            CapabilitySpec(
                resource=str(args["resource"]),
                rights={str(right) for right in args.get("rights", [])},
                effect=CapabilityEffect(str(args.get("effect", CapabilityEffect.ALLOW.value))),
                constraints=dict(args.get("constraints") or {}),
                metadata=dict(args.get("metadata") or {}),
                expires_at=str(args["expires_at"]) if args.get("expires_at") is not None else None,
                uses_remaining=int(args["uses_remaining"]) if args.get("uses_remaining") is not None else None,
                delegable=bool(args.get("delegable", False)),
                revocable=bool(args.get("revocable", True)),
            ),
            actor=self.pid,
        )
        return {"capability": self.runtime.capability.inspect(cap.cap_id)}

    def _capability_revoke(self, args: dict[str, Any]) -> dict[str, Any]:
        cap = self.runtime.capability.revoke(
            str(args["capability_id"]),
            revoked_by=self.pid,
            reason=str(args["reason"]) if args.get("reason") is not None else None,
        )
        return {"capability": self.runtime.capability.inspect(cap.cap_id)}

    def _process_fork(self, args: dict[str, Any]) -> Any:
        mode = ForkMode(str(args.get("mode", ForkMode.WORKER.value)))
        child_pid = self.runtime.fork_child_process(
            parent=self.pid,
            goal=args.get("goal", ""),
            memory_view=MemoryViewSpec(
                roots=self._selected_roots(args.get("root_oids")),
                mode=self._view_mode_for_fork(mode),
                include_parent_roots=bool(args.get("include_parent_roots", True)),
            ),
            inherit_capabilities=list(args.get("inherit_capabilities") or []),
            resource_budget=args.get("resource_budget"),
            image=args.get("image"),
            mode=mode,
            working_directory=args.get("working_directory"),
        )
        child = self.runtime.process.get(child_pid)
        return {"child_pid": child.pid, "status": child.status.value, "image": child.image_id, "goal_oid": child.goal_oid}

    def _process_spawn_child(self, args: dict[str, Any]) -> Any:
        child_pid = self.runtime.spawn_child_process(
            parent=self.pid,
            goal=args.get("goal", ""),
            image=args.get("image"),
            inherit_capabilities=list(args.get("inherit_capabilities") or []),
            resource_budget=args.get("resource_budget"),
            working_directory=args.get("working_directory"),
        )
        child = self.runtime.process.get(child_pid)
        return {"child_pid": child.pid, "status": child.status.value, "image": child.image_id, "goal_oid": child.goal_oid}

    def _process_wait(self, args: dict[str, Any]) -> Any:
        try:
            result = self.runtime.process.wait(
                self.pid,
                str(args["child_pid"]),
                timeout=None if bool(args.get("block", True)) else 0,
            )
        except TimeoutError:
            child = self.runtime.process.get(str(args["child_pid"]))
            return {"child_pid": child.pid, "status": child.status.value, "ready": False, "message": child.status_message}
        return {
            "child_pid": result.pid,
            "status": result.status.value,
            "ready": True,
            "result_oid": result.result.oid if result.result is not None else None,
            "message": result.message,
        }

    def _process_read_messages(self, args: dict[str, Any], *, default_block: bool) -> dict[str, Any]:
        kind = ProcessMessageKind(str(args["kind"])) if args.get("kind") is not None else None
        messages = self.runtime.messages.receive(
            self.pid,
            block=bool(args.get("block", default_block)),
            include_acked=bool(args.get("include_acked", False)),
            kind=kind,
            sender=str(args["sender"]) if args.get("sender") is not None else None,
            channel=str(args["channel"]) if args.get("channel") is not None else None,
            correlation_id=str(args["correlation_id"]) if args.get("correlation_id") is not None else None,
            reply_to=str(args["reply_to"]) if args.get("reply_to") is not None else None,
            message_ids=[str(item) for item in args["message_ids"]] if args.get("message_ids") is not None else None,
            limit=int(args["limit"]) if args.get("limit") is not None else None,
        )
        acked = []
        if bool(args.get("ack", True)):
            unread_ids = [message.message_id for message in messages if message.status.value == "unread"]
            if unread_ids:
                acked = self.runtime.messages.ack(self.pid, unread_ids)
                acked_by_id = {message.message_id: message for message in acked}
                messages = [acked_by_id.get(message.message_id, message) for message in messages]
        return {
            "ready": bool(messages),
            "messages": [self._process_message_result(message) for message in messages],
            "acked_message_ids": [message.message_id for message in acked],
        }

    def _process_message_result(self, message: Any) -> dict[str, Any]:
        return {
            "message_id": message.message_id,
            "sender": message.sender,
            "recipient_pid": message.recipient_pid,
            "kind": message.kind.value,
            "channel": message.channel,
            "correlation_id": message.correlation_id,
            "reply_to": message.reply_to,
            "subject": message.subject,
            "body": message.body,
            "payload": message.payload,
            "status": message.status.value,
            "created_at": message.created_at,
            "acked_at": message.acked_at,
        }

    def _image_result(self, result: Any) -> dict[str, Any]:
        image = result.image
        return {
            "image_id": image.image_id,
            "name": image.name,
            "version": image.version,
            "replaced": result.replaced,
            "source": result.source,
            "default_tools": list(image.default_tools),
            "boot_kind": image.boot.get("kind", "fresh"),
            "artifact_id": image.boot.get("artifact_id"),
            "artifact_sha256": image.boot.get("artifact_sha256"),
            "required_capabilities_count": len(image.required_capabilities),
        }

    def _string_list_arg(self, args: dict[str, Any], key: str) -> list[str]:
        value = args.get(key, [])
        if not isinstance(value, list):
            raise ValidationError(f"{key} must be a list of strings")
        if not all(isinstance(item, str) for item in value):
            raise ValidationError(f"{key} must contain only strings")
        return list(value)

    def _result_handle_for_exit(self, args: dict[str, Any], tool_result: ObjectHandle | None) -> ObjectHandle | None:
        if args.get("result_oid"):
            return self.runtime.memory.handle_for_oid(
                self.pid,
                str(args["result_oid"]),
                required_rights={ObjectRight.READ.value},
                optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value, ObjectRight.DIFF.value},
                issued_by="jit.process_exit",
            )
        if "payload" in args:
            return self.runtime.memory.create_object(
                pid=self.pid,
                object_type=ObjectType.SUMMARY,
                payload=args.get("payload"),
                metadata=ObjectMetadata(title="Process final result", tags=["final", "jit"]),
            )
        if bool(args.get("use_tool_result", False)):
            return tool_result
        return None

    def _selected_roots(self, root_oids: Any) -> list[ObjectHandle] | None:
        if root_oids is None:
            return None
        process = self.runtime.process.get(self.pid)
        visible = {handle.oid: handle for handle in (process.memory_view.roots if process.memory_view else [])}
        roots: list[ObjectHandle] = []
        for oid in root_oids:
            oid_text = str(oid)
            if oid_text in visible:
                roots.append(visible[oid_text])
                continue
            roots.append(
                self.runtime.memory.handle_for_oid(
                    self.pid,
                    oid_text,
                    required_rights={ObjectRight.READ.value},
                    optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.DIFF.value},
                    issued_by="jit.syscall.fork",
                )
            )
        return roots

    def _add_handle_to_view(self, handle: ObjectHandle) -> None:
        process = self.runtime.process.get(self.pid)
        if process.memory_view is None:
            process.memory_view = self.runtime.memory.create_view(self.pid, [handle], mode=ViewMode.READ_ONLY)
        elif all(existing.oid != handle.oid for existing in process.memory_view.roots):
            process.memory_view.roots.append(handle)
        self.runtime.store.update_process(process)

    def _view_mode_for_fork(self, mode: ForkMode) -> ViewMode:
        if mode == ForkMode.COPY:
            return ViewMode.COPY_ON_WRITE
        if mode == ForkMode.SPECULATIVE:
            return ViewMode.EPHEMERAL
        return ViewMode.READ_ONLY
