from __future__ import annotations

import asyncio
import builtins
import hashlib
import inspect
import time
from pathlib import Path
from typing import Any

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema import validate as jsonschema_validate
from jsonschema.validators import validator_for as jsonschema_validator_for

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import HumanApprovalRequired, NotFound, ProcessMessageWaitRequired, ProcessWaitRequired, ResourceLimitExceeded, ValidationError
from agent_libos.human.manager import HumanObjectManager
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.models import (
    EventType,
    JIT_MULTIPLEXER_TOOL_NAME,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    OPENAI_TOOL_NAME_MAX_CHARS,
    ObjectMetadata,
    ObjectOwnerKind,
    ObjectType,
    ProcessStatus,
    ResourceUsage,
    ToolCallResult,
    ToolCandidate,
    ToolCandidateStatus,
    ToolHandle,
    ToolSpec,
    ValidationResult,
    is_openai_tool_name,
)
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.storage import SQLiteStore
from agent_libos.substrate import CommandMetrics, SubprocessLimitExceeded, SubprocessLimits, SubprocessTimeoutExpired
from agent_libos.tools.base import BaseAgentTool, SyncAgentTool, ToolContext
from agent_libos.tools.observability import ensure_json_size, sanitize_for_observability
from agent_libos.tools.sandbox import DenoTypescriptSandbox, SandboxBackend, SandboxExecutionResult
from agent_libos.utils.serde import dumps

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_TERMINAL_PROCESS_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
_TOOL_CALLABLE_PROCESS_STATUSES = {ProcessStatus.RUNNABLE, ProcessStatus.RUNNING}
_JIT_DECLARED_PERMISSIONS = (
    "checkpoint.read",
    "checkpoint.write",
    "filesystem.delete",
    "filesystem.read",
    "filesystem.write",
    "human.ask",
    "human.output",
    "image.write",
    "jsonrpc.call",
    "libos.syscall",
    "object.link",
    "object.read",
    "object.write",
    "process.lifecycle",
    "process.message",
    "process.signal",
    "process.spawn",
    "shell.execute",
    "skill.read",
    "skill.write",
)

_JIT_MULTIPLEXER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "description": "Name of a visible process-local JIT tool to execute.",
        },
        "arguments": {
            "type": "object",
            "description": "Arguments for the selected JIT tool.",
            "additionalProperties": True,
        },
    },
    "required": ["tool_name", "arguments"],
    "additionalProperties": False,
}
# This spec is only an LLM protocol surface. It is never inserted into a
# process tool table, so direct runtime calls still resolve to real tools only.
_JIT_MULTIPLEXER_SPEC = ToolSpec(
    name=JIT_MULTIPLEXER_TOOL_NAME,
    description=(
        "Execute one visible process-local Deno/TypeScript JIT tool by name. "
        "The image prompt must describe available JIT tool names and argument shapes."
    ),
    input_schema=_JIT_MULTIPLEXER_INPUT_SCHEMA,
    output_schema={"type": "object"},
    policy={
        "side_effects": True,
        "idempotent": False,
        "declared_permissions": list(_JIT_DECLARED_PERMISSIONS),
    },
    tags=["jit", "tool", "multiplexer", "protocol"],
    side_effects=list(_JIT_DECLARED_PERMISSIONS),
)


class ToolBroker:
    """Registry and dispatch boundary for model-facing tools."""

    def __init__(
        self,
        store: SQLiteStore,
        memory: ObjectMemoryManager,
        capabilities: CapabilityManager,
        human: HumanObjectManager,
        audit: AuditManager,
        events: EventBus,
        sandbox: SandboxBackend | None = None,
        workspace_root: str | Path | None = None,
        config: AgentLibOSConfig | None = None,
        resources: Any | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.store = store
        self.memory = memory
        self.capabilities = capabilities
        self.human = human
        self.audit = audit
        self.events = events
        self.resources = resources
        self.sandbox = sandbox or DenoTypescriptSandbox(
            deno_executable=self.config.tools.deno_executable,
            default_timeout_s=self.config.tools.deno_timeout_s,
            max_rpc_calls=self.config.tools.deno_max_rpc_calls,
            max_stdout_bytes=self.config.tools.deno_max_stdout_bytes,
            max_stderr_bytes=self.config.tools.deno_max_stderr_bytes,
            jsr_allowlist=self.config.tools.deno_jsr_allowlist,
            max_validation_log_chars=self.config.tools.jit_validation_log_max_chars,
            forbidden_executable_roots=[Path(workspace_root or Path.cwd()).resolve()],
        )
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self._tools: dict[str, BaseAgentTool] = {}
        self._tool_ids_by_name: dict[str, str] = {}
        self._handles: dict[str, ToolHandle] = {}
        self._jit_sources: dict[str, str] = {}

    def register_tool(
        self,
        tool: BaseAgentTool,
        registered_by: str = "runtime",
        scope: str = "static",
        ephemeral: bool = False,
    ) -> ToolHandle:
        spec = tool.spec(config=self.config)
        if spec.name in self._tool_ids_by_name:
            raise ValueError(f"tool already registered: {spec.name}")
        tool_id = new_id("tool") if ephemeral else _stable_static_tool_id(
            spec.name,
            digest_chars=self.config.tools.static_tool_id_digest_chars,
        )
        handle = ToolHandle(tool_id=tool_id, name=spec.name, capability_id=None, scope=scope)
        self._tools[tool_id] = tool
        self._tool_ids_by_name[spec.name] = tool_id
        self._handles[tool_id] = handle
        existing = next((row for row in self.store.list_tools() if row["tool_id"] == tool_id), None)
        if existing is not None and existing["name"] != spec.name:
            raise ValueError(f"tool id collision: {tool_id}")
        if existing is None:
            self.store.insert_tool(handle, spec, registered_by=registered_by, created_at=utc_now(), ephemeral=ephemeral)
        else:
            self.store.update_tool(handle, spec, registered_by=registered_by, ephemeral=ephemeral)
        self.audit.record(
            actor=registered_by,
            action="tool.register",
            target=f"tool:{tool_id}",
            decision={
                "name": spec.name,
                "version": spec.version,
                "policy": spec.policy,
                "tags": spec.tags,
            },
        )
        return handle

    def unregister_tool(self, tool: ToolHandle | str, *, registered_by: str | None = None) -> bool:
        handle = self._handle_for_unregistration(tool)
        if handle is None:
            return False
        row = next((item for item in self.store.list_tools() if item["tool_id"] == handle.tool_id), None)
        if registered_by is not None and row is not None and row.get("registered_by") != registered_by:
            return False
        self._tools.pop(handle.tool_id, None)
        self._jit_sources.pop(handle.tool_id, None)
        self._handles.pop(handle.tool_id, None)
        if self._tool_ids_by_name.get(handle.name) == handle.tool_id:
            self._tool_ids_by_name.pop(handle.name, None)
        self.store.delete_tool(handle.tool_id, registered_by=registered_by)
        self.audit.record(
            actor=registered_by or "tool_broker",
            action="tool.unregister",
            target=f"tool:{handle.tool_id}",
            decision={"name": handle.name},
        )
        return True

    def configure_process_tools(
        self,
        pid: str,
        tools: builtins.list[ToolHandle | str],
        assigned_by: str = "tool_broker",
    ) -> dict[str, str]:
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        table: dict[str, str] = {}
        for tool in tools:
            handle = self.resolve(tool)
            table[handle.name] = handle.tool_id
        process.tool_table = table
        process.updated_at = utc_now()
        self.store.update_process(process)
        self.audit.record(
            actor=assigned_by,
            action="process.tools.configure",
            target=f"process:{pid}",
            decision={"tools": sorted(table)},
        )
        return table

    def grant_execute(self, pid: str, tool: ToolHandle | str, issued_by: str = "tool_broker") -> str:
        handle = self.resolve(tool, pid=pid)
        if not self._process_has_tool(pid, handle):
            raise ValidationError(
                "tool execute capabilities are not a resource authority; configure process tools at creation time"
            )
        self.audit.record(
            actor=issued_by,
            action="tool.execute_grant_ignored",
            target=f"tool:{handle.tool_id}",
            decision={
                "tool": handle.name,
                "reason": "tool calls are allowed by process tool table, not execute capability",
            },
        )
        return handle.tool_id

    def process_has_tool(self, pid: str, tool: ToolHandle | str) -> bool:
        handle = self.resolve(tool, pid=pid)
        return self._process_has_tool(pid, handle)

    def is_sync_side_effect_tool(self, tool: ToolHandle | str) -> bool:
        handle = self.resolve(tool)
        implementation = self._tools.get(handle.tool_id)
        if implementation is None:
            return False
        return isinstance(implementation, SyncAgentTool) and bool(implementation.spec().policy.get("side_effects"))

    def propose(
        self,
        pid: str,
        spec: ToolSpec | dict[str, Any],
        source_code: str,
        tests: builtins.list[dict[str, Any]] | None = None,
        requested_capabilities: builtins.list[dict[str, Any]] | None = None,
    ) -> str:
        raw_tool_spec = spec if isinstance(spec, ToolSpec) else ToolSpec(**spec)
        tool_spec = _conservative_jit_tool_spec(raw_tool_spec)
        self._validate_jit_tool_spec(tool_spec)
        if self._jit_exposure_for_process(pid) == JIT_TOOL_EXPOSURE_MULTIPLEXED and tool_spec.name == JIT_MULTIPLEXER_TOOL_NAME:
            raise ValidationError(f"{JIT_MULTIPLEXER_TOOL_NAME} is reserved by multiplexed JIT tool exposure")
        self._validate_jit_source_and_tests(source_code, tests or [])
        now = utc_now()
        candidate = ToolCandidate(
            candidate_id=new_id("tcand"),
            pid=pid,
            spec=tool_spec,
            source_code=source_code,
            tests=tests or [],
            requested_capabilities=requested_capabilities or [],
            status=ToolCandidateStatus.PROPOSED,
            validation=None,
            created_at=now,
            updated_at=now,
        )
        self.store.insert_tool_candidate(candidate)
        candidate_obj = self.memory.create_object(
            pid=pid,
            object_type=ObjectType.TOOL_CANDIDATE,
            payload={
                "candidate_id": candidate.candidate_id,
                "language": self.sandbox.language,
                "spec": {
                    "name": tool_spec.name,
                    "description": tool_spec.description,
                    "input_schema": tool_spec.input_schema,
                    "output_schema": tool_spec.output_schema,
                    "side_effects": tool_spec.side_effects,
                },
                "tests": candidate.tests,
                "requested_capabilities": candidate.requested_capabilities,
            },
            metadata=ObjectMetadata(title=f"Tool candidate: {tool_spec.name}", tags=["tool", "candidate"]),
            immutable=True,
        )
        self.audit.record(
            actor=pid,
            action="tool.propose",
            target=f"tool_candidate:{candidate.candidate_id}",
            output_refs=[candidate_obj.oid],
            decision={"name": tool_spec.name},
        )
        return candidate.candidate_id

    def validate(self, candidate_id: str, *, pid: str | None = None) -> ValidationResult:
        candidate = self._get_candidate(candidate_id)
        owner_pid = pid or candidate.pid
        self._require_candidate_owner(candidate, owner_pid)
        try:
            result = self._run_candidate_tests(candidate, owner_pid)
        except SubprocessLimitExceeded as exc:
            self._charge_subprocess_metrics(
                owner_pid,
                exc.metrics,
                source="tool.validate.deno",
                context={"candidate_id": candidate_id, "tool": candidate.spec.name},
            )
            raise
        except SubprocessTimeoutExpired as exc:
            self._charge_subprocess_metrics(
                owner_pid,
                exc.metrics,
                source="tool.validate.deno",
                context={"candidate_id": candidate_id, "tool": candidate.spec.name},
            )
            raise
        errors = list(result.errors)
        warnings = list(result.warnings)
        if candidate.requested_capabilities:
            errors.append("Deno/TypeScript JIT tools cannot request external capabilities")
        validation = ValidationResult(
            ok=not errors and result.ok,
            errors=errors,
            warnings=warnings,
            logs=result.logs,
            metadata=result.metadata,
        )
        metadata = self.sandbox.metadata_for_source(candidate.source_code)
        candidate.validation = {
            "ok": validation.ok,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "logs": validation.logs,
            "metrics": validation.metadata.get("metrics"),
            **metadata,
        }
        candidate.status = ToolCandidateStatus.VALIDATED if validation.ok else ToolCandidateStatus.REJECTED
        candidate.updated_at = utc_now()
        self.store.update_tool_candidate(candidate)
        self.audit.record(
            actor="tool_broker",
            action="tool.validate",
            target=f"tool_candidate:{candidate_id}",
            decision=candidate.validation,
        )
        return validation

    def register(
        self,
        pid: str,
        candidate_id: str,
        approver: str = "policy:local",
        scope: str = "ephemeral_process",
    ) -> ToolHandle:
        candidate = self._get_candidate(candidate_id)
        self._require_candidate_owner(candidate, pid)
        if candidate.status == ToolCandidateStatus.REGISTERED:
            raise ValidationError(f"tool candidate is already registered: {candidate_id}")
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        if candidate.spec.name in process.tool_table:
            raise ValidationError(f"process already has a tool named: {candidate.spec.name}")
        if self._name_collides_with_static_tool(candidate.spec.name):
            raise ValidationError(f"tool name already exists: {candidate.spec.name}")
        if candidate.status != ToolCandidateStatus.VALIDATED:
            validation = self.validate(candidate_id, pid=pid)
            if not validation.ok:
                raise ValidationError("; ".join(validation.errors))
            candidate = self._get_candidate(candidate_id)
        tool_id = new_id("tool")
        handle = ToolHandle(tool_id=tool_id, name=candidate.spec.name, capability_id=None, scope=scope)
        self._jit_sources[tool_id] = candidate.source_code
        self._handles[tool_id] = handle
        # JIT tool names are process-local through AgentProcess.tool_table. Do
        # not add them to the global name index, otherwise later pid-less
        # resolve(name) calls can turn one process' JIT into a globally
        # referenceable tool.
        self.store.insert_tool(handle, candidate.spec, registered_by=approver, created_at=utc_now(), ephemeral=True)
        candidate.status = ToolCandidateStatus.REGISTERED
        candidate.registered_tool_id = tool_id
        candidate.updated_at = utc_now()
        self.store.update_tool_candidate(candidate)
        process.tool_table[candidate.spec.name] = tool_id
        process.updated_at = utc_now()
        self.store.update_process(process)
        self.audit.record(
            actor=approver,
            action="tool.register",
            target=f"tool:{tool_id}",
            decision={"candidate_id": candidate_id, "scope": scope},
        )
        return handle

    def call(
        self,
        pid: str,
        tool: ToolHandle | str,
        args: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acall(pid, tool, args, context_metadata=context_metadata))
        raise RuntimeError("Cannot call ToolBroker.call() inside a running event loop. Use await acall(...).")

    async def acall(
        self,
        pid: str,
        tool: ToolHandle | str,
        args: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        handle = self.resolve(tool, pid=pid)
        resource = f"tool:{handle.tool_id}"
        process_status_error = self._process_status_error(pid)
        if process_status_error is not None:
            call_id = new_id("tcall")
            self.events.emit(
                EventType.TOOL_FAILED,
                source=resource,
                target=pid,
                payload={"call_id": call_id, "error": process_status_error, "policy_decision": "deny"},
            )
            self.audit.record(
                actor=pid,
                action="tool.call",
                target=resource,
                decision={
                    "ok": False,
                    "tool": handle.name,
                    "policy_decision": "deny",
                    "policy_reason": "process_not_tool_callable",
                    "error": process_status_error,
                },
            )
            return ToolCallResult(
                call_id=call_id,
                tool_id=handle.tool_id,
                result_handle=None,
                payload=None,
                ok=False,
                error=process_status_error,
            )
        if not self._process_has_tool(pid, handle):
            # The process tool table gates only LLM-facing tool visibility.
            # Host resources are checked by the primitive each tool calls into.
            call_id = new_id("tcall")
            error = f"tool is not in process tool table: {handle.name}"
            self.events.emit(
                EventType.TOOL_FAILED,
                source=resource,
                target=pid,
                payload={"call_id": call_id, "error": error, "policy_decision": "deny"},
            )
            self.audit.record(
                actor=pid,
                action="tool.call",
                target=resource,
                decision={
                    "ok": False,
                    "tool": handle.name,
                    "policy_decision": "deny",
                    "policy_reason": "tool_not_in_process_table",
                },
            )
            return ToolCallResult(call_id=call_id, tool_id=handle.tool_id, result_handle=None, payload=None, ok=False, error=error)

        call_id = new_id("tcall")
        try:
            self._charge_tool_call(pid, handle, call_id)
        except ResourceLimitExceeded as exc:
            error = str(exc)
            self.events.emit(
                EventType.TOOL_FAILED,
                source=resource,
                target=pid,
                payload={"call_id": call_id, "error": error, "policy_decision": "resource_limit"},
            )
            self.audit.record(
                actor=pid,
                action="tool.call",
                target=resource,
                decision={
                    "ok": False,
                    "tool": handle.name,
                    "policy_decision": "resource_limit",
                    "error": error,
                },
            )
            return ToolCallResult(call_id=call_id, tool_id=handle.tool_id, result_handle=None, payload=None, ok=False, error=error)
        try:
            ensure_json_size(
                args,
                self.config.tools.tool_call_args_hard_limit_bytes,
                "tool call arguments",
            )
        except ValidationError as exc:
            error = str(exc)
            self.events.emit(
                EventType.TOOL_FAILED,
                source=resource,
                target=pid,
                payload={"call_id": call_id, "error": error, "policy_decision": "validation_error"},
            )
            self.audit.record(
                actor=pid,
                action="tool.call",
                target=resource,
                decision={
                    "ok": False,
                    "tool": handle.name,
                    "policy_decision": "validation_error",
                    "error": error,
                },
            )
            return ToolCallResult(call_id=call_id, tool_id=handle.tool_id, result_handle=None, payload=None, ok=False, error=error)
        self.events.emit(
            EventType.TOOL_CALLED,
            source=pid,
            target=resource,
            payload={"call_id": call_id, "args": sanitize_for_observability(args)},
        )
        started_at = time.perf_counter()
        result_was_omitted = False
        try:
            jit_session: LibOSSyscallSession | None = None
            if handle.tool_id in self._tools:
                tool_result = await self._tools[handle.tool_id].ainvoke(
                    args,
                    self._context(pid, handle, call_id, metadata=context_metadata),
                )
                if not tool_result.ok:
                    error_message = tool_result.error.message if tool_result.error else tool_result.content
                    self.events.emit(
                        EventType.TOOL_FAILED,
                        source=resource,
                        target=pid,
                        payload={
                            "call_id": call_id,
                            "error": error_message,
                            "tool_result": sanitize_for_observability(tool_result.model_dump(mode="json")),
                        },
                    )
                    self.audit.record(
                        actor=pid,
                        action="tool.call",
                        target=resource,
                        decision={
                            "ok": False,
                            "tool": handle.name,
                            "policy_decision": "allow",
                            "tool_result": sanitize_for_observability(tool_result.model_dump(mode="json")),
                            "tool_wall_seconds": self._elapsed(started_at),
                        },
                    )
                    return ToolCallResult(
                        call_id=call_id,
                        tool_id=handle.tool_id,
                        result_handle=None,
                        payload=tool_result.model_dump(mode="json"),
                        ok=False,
                        error=error_message,
                    )
                payload = tool_result.data
                result_payload = {
                    "tool_id": handle.tool_id,
                    "tool_name": handle.name,
                    "result": payload,
                    "content": tool_result.content,
                    "artifacts": [artifact.model_dump(mode="json") for artifact in tool_result.artifacts],
                    "metadata": tool_result.metadata,
                }
            elif handle.tool_id in self._jit_sources:
                runtime = getattr(self, "runtime", None)
                if runtime is None:
                    raise RuntimeError("Runtime is unavailable for Deno JIT syscall execution.")
                self._validate_jit_arguments(handle, args)
                jit_session = LibOSSyscallSession(runtime, pid, config=self.config)
                deno_result = await self._run_sandbox_source(
                    self._jit_sources[handle.tool_id],
                    args,
                    pid=pid,
                    syscall_handler=jit_session.handle,
                )
                if isinstance(deno_result, SandboxExecutionResult):
                    payload = deno_result.value
                    self._charge_subprocess_metrics(
                        pid,
                        deno_result.metrics,
                        source="tool.deno",
                        context={"tool": handle.name, "tool_id": handle.tool_id},
                    )
                else:
                    payload = deno_result
                self._validate_jit_output(handle, payload)
                result_payload = {"tool_id": handle.tool_id, "tool_name": handle.name, "result": payload}
            else:
                raise NotFound(f"tool implementation not loaded: {handle.tool_id}")
        except SubprocessLimitExceeded as exc:
            self._handle_subprocess_limit(pid, handle, resource, call_id, exc)
            return ToolCallResult(
                call_id=call_id,
                tool_id=handle.tool_id,
                result_handle=None,
                payload=None,
                ok=False,
                error=str(exc),
            )
        except SubprocessTimeoutExpired as exc:
            error = self._handle_subprocess_timeout(pid, handle, resource, call_id, exc)
            return ToolCallResult(
                call_id=call_id,
                tool_id=handle.tool_id,
                result_handle=None,
                payload=None,
                ok=False,
                error=error,
            )
        except HumanApprovalRequired as exc:
            # Do not convert this into a ToolCallResult: the LLM quantum has not
            # completed and must be resumed after the human decision.
            self.audit.record(
                actor=pid,
                action="tool.call_waiting_human",
                target=resource,
                decision={
                    "ok": False,
                    "tool": handle.name,
                    "policy_decision": "require_human_approval",
                    "request_id": exc.request_id,
                    "tool_wall_seconds": self._elapsed(started_at),
                },
            )
            raise
        except ProcessWaitRequired as exc:
            self.audit.record(
                actor=pid,
                action="tool.call_waiting_process",
                target=resource,
                decision={
                    "ok": False,
                    "tool": handle.name,
                    "policy_decision": "wait_for_child",
                    "child_pid": exc.child_pid,
                    "tool_wall_seconds": self._elapsed(started_at),
                },
            )
            raise
        except ProcessMessageWaitRequired as exc:
            self.audit.record(
                actor=pid,
                action="tool.call_waiting_message",
                target=resource,
                decision={
                    "ok": False,
                    "tool": handle.name,
                    "policy_decision": "wait_for_process_message",
                    "recipient_pid": exc.recipient_pid,
                    "filters": exc.filters,
                    "tool_wall_seconds": self._elapsed(started_at),
                },
            )
            raise
        except ValueError as exc:
            error = str(exc)
            self.events.emit(
                EventType.TOOL_FAILED,
                source=resource,
                target=pid,
                payload={"call_id": call_id, "error": error, "policy_decision": "validation_error"},
            )
            self.audit.record(
                actor=pid,
                action="tool.call",
                target=resource,
                decision={
                    "ok": False,
                    "tool": handle.name,
                    "policy_decision": "validation_error",
                    "error": error,
                    "tool_wall_seconds": self._elapsed(started_at),
                },
            )
            return ToolCallResult(call_id=call_id, tool_id=handle.tool_id, result_handle=None, payload=None, ok=False, error=error)
        except Exception as exc:
            self.events.emit(
                EventType.TOOL_FAILED,
                source=resource,
                target=pid,
                payload={"call_id": call_id, "error": sanitize_for_observability(str(exc))},
            )
            self.audit.record(
                actor=pid,
                action="tool.call",
                target=resource,
                decision={
                    "ok": False,
                    "tool": handle.name,
                    "policy_decision": "allow",
                    "error": sanitize_for_observability(str(exc)),
                    "tool_wall_seconds": self._elapsed(started_at),
                },
            )
            return ToolCallResult(call_id=call_id, tool_id=handle.tool_id, result_handle=None, payload=None, ok=False, error=str(exc))

        try:
            ensure_json_size(
                result_payload,
                self._tool_result_persistence_limit(),
                "tool result payload",
            )
        except ValidationError as exc:
            error = str(exc)
            if self._tool_has_side_effects(handle):
                # At this point a side-effecting tool has already run. Reporting
                # failure would invite unsafe retries, so persist a bounded
                # success envelope and make the omitted payload explicit.
                result_was_omitted = True
                payload = {
                    "result_omitted": True,
                    "reason": error,
                    "tool_id": handle.tool_id,
                    "tool_name": handle.name,
                }
                result_payload = {
                    "tool_id": handle.tool_id,
                    "tool_name": handle.name,
                    "result": payload,
                    "content": "[tool result omitted after size-limit failure]",
                    "artifacts": [],
                    "metadata": {
                        "result_omitted": True,
                        "omission_reason": error,
                    },
                }
            else:
                self.events.emit(
                    EventType.TOOL_FAILED,
                    source=resource,
                    target=pid,
                    payload={"call_id": call_id, "error": error, "policy_decision": "validation_error"},
                )
                self.audit.record(
                    actor=pid,
                    action="tool.call",
                    target=resource,
                    decision={
                        "ok": False,
                        "tool": handle.name,
                        "policy_decision": "validation_error",
                        "error": error,
                        "result": self._oversize_result_observation(),
                        "tool_wall_seconds": self._elapsed(started_at),
                    },
                )
                return ToolCallResult(call_id=call_id, tool_id=handle.tool_id, result_handle=None, payload=None, ok=False, error=error)

        with self.memory.lifetime_scope(
            actor=resource,
            owner_kind=ObjectOwnerKind.PROCESS,
            owner_id=pid,
            reason="tool_result",
        ) as result_scope:
            result_handle = result_scope.create_object(
                pid=pid,
                object_type=ObjectType.TOOL_RESULT,
                payload=result_payload,
                metadata=ObjectMetadata(title=f"Tool result: {handle.name}", tags=["tool_result", handle.name]),
                immutable=True,
            )
            if jit_session is not None:
                try:
                    await jit_session.apply_deferred_lifecycle(result_handle)
                except Exception as exc:
                    error = str(exc)
                    self.events.emit(
                        EventType.TOOL_FAILED,
                        source=resource,
                        target=pid,
                        payload={"call_id": call_id, "error": error, "policy_decision": "lifecycle_error"},
                    )
                    self.audit.record(
                        actor=pid,
                        action="tool.call",
                        target=resource,
                        decision={
                            "ok": False,
                            "tool": handle.name,
                            "policy_decision": "lifecycle_error",
                            "error": sanitize_for_observability(error),
                            "tool_wall_seconds": self._elapsed(started_at),
                        },
                    )
                    return ToolCallResult(
                        call_id=call_id,
                        tool_id=handle.tool_id,
                        result_handle=None,
                        payload=None,
                        ok=False,
                        error=error,
                    )
            result_scope.commit()
        self.events.emit(
            EventType.TOOL_COMPLETED,
            source=resource,
            target=pid,
            payload={"call_id": call_id, "result_oid": result_handle.oid},
        )
        decision = {
            "ok": True,
            "tool": handle.name,
            "policy_decision": "allow",
            "tool_wall_seconds": self._elapsed(started_at),
        }
        if result_was_omitted:
            decision["result_omitted"] = True
            decision["result"] = self._oversize_result_observation()
        self.audit.record(
            actor=pid,
            action="tool.call",
            target=resource,
            output_refs=[result_handle.oid],
            decision=decision,
        )
        return ToolCallResult(
            call_id=call_id,
            tool_id=handle.tool_id,
            result_handle=result_handle,
            payload=payload,
            ok=True,
        )

    def _charge_tool_call(self, pid: str, handle: ToolHandle, call_id: str) -> None:
        if self.resources is None:
            return
        self.resources.charge(
            pid,
            ResourceUsage(tool_calls=1),
            source="tool.call",
            context={"tool": handle.name, "tool_id": handle.tool_id, "call_id": call_id},
            allow_overage=False,
            kill_on_exceed=False,
        )

    def _tool_has_side_effects(self, handle: ToolHandle) -> bool:
        implementation = self._tools.get(handle.tool_id)
        if implementation is not None:
            return bool(implementation.spec(config=self.config).policy.get("side_effects"))
        spec = self.store.get_tool_spec(handle.tool_id)
        return bool(spec is not None and spec.policy.get("side_effects"))

    def _tool_result_persistence_limit(self) -> int:
        return min(
            self.config.tools.tool_result_payload_hard_limit_bytes,
            self.config.tools.memory_payload_hard_limit_bytes,
        )

    async def _run_sandbox_source(
        self,
        source_code: str,
        args: dict[str, Any],
        *,
        pid: str,
        syscall_handler: Any,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "pid": pid,
            "syscall_handler": syscall_handler,
            "limits": self._subprocess_limits(pid),
            "return_metrics": True,
        }
        if kwargs["limits"] is not None:
            self._require_sandbox_resource_controls()
        supported = self._supported_sandbox_kwargs()
        selected_kwargs = {key: value for key, value in kwargs.items() if key in supported}
        if kwargs["limits"] is not None and "limits" not in selected_kwargs:
            raise ValidationError("sandbox backend must accept SubprocessLimits when resource limits are configured")
        if kwargs["limits"] is not None and "return_metrics" not in selected_kwargs:
            raise ValidationError("sandbox backend must return subprocess metrics")
        return await self.sandbox.arun_source(source_code, args, **selected_kwargs)

    def _supported_sandbox_kwargs(self) -> set[str]:
        signature = inspect.signature(self.sandbox.arun_source)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return {"pid", "syscall_handler", "timeout", "limits", "return_metrics"}
        return set(signature.parameters)

    def _supported_run_tests_kwargs(self) -> set[str]:
        signature = inspect.signature(self.sandbox.run_tests)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return {"timeout", "limits", "return_metrics"}
        return set(signature.parameters)

    def _require_sandbox_resource_controls(self) -> None:
        supported = self._supported_sandbox_kwargs()
        if "limits" not in supported:
            raise ValidationError("sandbox backend must accept SubprocessLimits when resource limits are configured")
        if "return_metrics" not in supported:
            raise ValidationError("sandbox backend must return subprocess metrics")

    def _require_sandbox_validation_resource_controls(self) -> None:
        supported = self._supported_run_tests_kwargs()
        if "limits" not in supported:
            raise ValidationError("sandbox backend must accept SubprocessLimits when validating with resource limits")
        if "return_metrics" not in supported:
            raise ValidationError("sandbox backend must return validation subprocess metrics")

    def _run_candidate_tests(
        self,
        candidate: ToolCandidate,
        owner_pid: str,
    ) -> ValidationResult:
        batches = [[test] for test in candidate.tests] or [[]]
        errors: list[str] = []
        warnings: list[str] = []
        logs: list[str] = []
        metrics: list[CommandMetrics] = []
        ok = True
        for index, tests in enumerate(batches, start=1):
            limits = self._subprocess_limits(owner_pid)
            self._preflight_validation_budget(owner_pid, limits, candidate, index)
            result = self._run_candidate_test_batch(candidate, tests, limits)
            ok = ok and result.ok
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            if result.logs:
                logs.append(result.logs)
            result_metrics = self._metrics_from_validation(result.metadata.get("metrics"))
            self._charge_subprocess_metrics(
                owner_pid,
                result_metrics,
                source="tool.validate.deno",
                context={"candidate_id": candidate.candidate_id, "tool": candidate.spec.name, "test_index": index},
            )
            if result_metrics is not None:
                metrics.append(result_metrics)
        return ValidationResult(
            ok=ok and not errors,
            errors=errors,
            warnings=warnings,
            logs=self._bounded_validation_logs(logs),
            metadata={"metrics": self._aggregate_command_metrics(metrics)},
        )

    def _run_candidate_test_batch(
        self,
        candidate: ToolCandidate,
        tests: list[dict[str, Any]],
        limits: SubprocessLimits | None,
    ) -> ValidationResult:
        kwargs: dict[str, Any] = {
            "timeout": self.config.tools.jit_validation_timeout_s,
            "limits": limits,
            "return_metrics": True,
        }
        if limits is not None:
            self._require_sandbox_validation_resource_controls()
        supported = self._supported_run_tests_kwargs()
        selected_kwargs = {key: value for key, value in kwargs.items() if key in supported}
        if limits is not None and "limits" not in selected_kwargs:
            raise ValidationError("sandbox backend must accept SubprocessLimits when validating with resource limits")
        if limits is not None and "return_metrics" not in selected_kwargs:
            raise ValidationError("sandbox backend must return validation subprocess metrics")
        return self.sandbox.run_tests(candidate.source_code, tests, **selected_kwargs)

    def _preflight_validation_budget(
        self,
        owner_pid: str,
        limits: SubprocessLimits | None,
        candidate: ToolCandidate,
        test_index: int,
    ) -> None:
        if self.resources is None or limits is None:
            return
        usage = ResourceUsage()
        if limits.wall_seconds is not None and limits.wall_seconds <= 0:
            usage = ResourceUsage(subprocess_wall_seconds=1e-9)
        elif limits.cpu_seconds is not None and limits.cpu_seconds <= 0:
            usage = ResourceUsage(subprocess_cpu_seconds=1e-9)
        elif limits.memory_bytes is not None and limits.memory_bytes <= 0:
            usage = ResourceUsage(subprocess_peak_memory_bytes=1)
        else:
            return
        self.resources.preflight(
            owner_pid,
            usage,
            source="tool.validate.deno",
            context={"candidate_id": candidate.candidate_id, "tool": candidate.spec.name, "test_index": test_index},
        )

    def _validate_jit_source_and_tests(self, source_code: str, tests: list[dict[str, Any]]) -> None:
        if len(source_code) > self.config.tools.jit_source_max_chars:
            raise ValidationError(
                f"JIT source exceeds {self.config.tools.jit_source_max_chars} chars"
            )
        if len(tests) > self.config.tools.jit_tests_max_count:
            raise ValidationError(
                f"JIT tests exceed {self.config.tools.jit_tests_max_count} cases"
            )
        for index, test in enumerate(tests, start=1):
            ensure_json_size(test, self.config.tools.jit_test_case_max_bytes, f"JIT test {index}")

    def _validate_jit_tool_spec(self, spec: ToolSpec) -> None:
        # JIT names become OpenAI function names in direct mode and catalog keys
        # in multiplexed mode, so reject names that the model/provider protocol
        # cannot address exactly.
        if not is_openai_tool_name(spec.name):
            raise ValidationError(
                f"JIT tool name must match OpenAI tool name syntax "
                f"[A-Za-z0-9_-]{{1,{OPENAI_TOOL_NAME_MAX_CHARS}}}: {spec.name!r}"
            )
        if not isinstance(spec.description, str) or not spec.description.strip():
            raise ValidationError("JIT tool description must be a non-empty string")
        self._validate_json_schema(spec.input_schema or {"type": "object"}, "input_schema")
        self._validate_json_schema(spec.output_schema or {"type": "object"}, "output_schema")

    def _validate_json_schema(self, schema: dict[str, Any], field: str) -> None:
        if not isinstance(schema, dict):
            raise ValidationError(f"JIT tool {field} must be a JSON schema object")
        ensure_json_size(schema, self.config.tools.jit_test_case_max_bytes, f"JIT tool {field}")
        try:
            jsonschema_validator_for(schema).check_schema(schema)
        except JsonSchemaSchemaError as exc:
            raise ValidationError(f"JIT tool {field} is not a valid JSON schema: {exc.message}") from exc

    def _subprocess_limits(self, pid: str) -> SubprocessLimits | None:
        if self.resources is None:
            return None
        wall = self.resources.remaining_cumulative(
            pid,
            "max_subprocess_wall_seconds",
            "subprocess_wall_seconds",
        )
        cpu = self.resources.remaining_cumulative(
            pid,
            "max_subprocess_cpu_seconds",
            "subprocess_cpu_seconds",
        )
        memory = self.resources.peak_limit(pid, "max_subprocess_memory_bytes")
        if wall is None and cpu is None and memory is None:
            return None
        return SubprocessLimits(wall_seconds=wall, cpu_seconds=cpu, memory_bytes=memory)

    def _charge_subprocess_metrics(
        self,
        pid: str,
        metrics: CommandMetrics | None,
        *,
        source: str,
        context: dict[str, Any],
    ) -> None:
        if self.resources is None or metrics is None:
            return
        self.resources.charge(
            pid,
            ResourceUsage(
                subprocess_wall_seconds=max(0.0, metrics.wall_seconds),
                subprocess_cpu_seconds=max(0.0, metrics.cpu_seconds),
                subprocess_peak_memory_bytes=max(0, metrics.peak_memory_bytes),
            ),
            source=source,
            context={**context, "metrics": self._metrics_json(metrics)},
            allow_overage=True,
            kill_on_exceed=True,
        )

    def _handle_subprocess_limit(
        self,
        pid: str,
        handle: ToolHandle,
        resource: str,
        call_id: str,
        exc: SubprocessLimitExceeded,
    ) -> None:
        charge_error: ResourceLimitExceeded | None = None
        try:
            self._charge_subprocess_metrics(
                pid,
                exc.metrics,
                source="tool.deno",
                context={"tool": handle.name, "tool_id": handle.tool_id},
            )
        except ResourceLimitExceeded as resource_exc:
            charge_error = resource_exc
        reason = str(charge_error or exc)
        if self.resources is not None:
            self.resources.kill_if_exceeded(
                pid,
                reason=reason,
                limit={"kind": exc.metrics.limit_kind, "metrics": self._metrics_json(exc.metrics)},
            )
        self.events.emit(
            EventType.TOOL_FAILED,
            source=resource,
            target=pid,
            payload={"call_id": call_id, "error": reason, "policy_decision": "resource_limit"},
        )
        self.audit.record(
            actor=pid,
            action="tool.call",
            target=resource,
            decision={
                "ok": False,
                "tool": handle.name,
                "policy_decision": "resource_limit",
                "error": reason,
                "metrics": self._metrics_json(exc.metrics),
            },
        )

    def _handle_subprocess_timeout(
        self,
        pid: str,
        handle: ToolHandle,
        resource: str,
        call_id: str,
        exc: SubprocessTimeoutExpired,
    ) -> str:
        charge_error: ResourceLimitExceeded | None = None
        try:
            self._charge_subprocess_metrics(
                pid,
                exc.metrics,
                source="tool.deno",
                context={"tool": handle.name, "tool_id": handle.tool_id},
            )
        except ResourceLimitExceeded as resource_exc:
            charge_error = resource_exc
        if charge_error is not None:
            reason = str(charge_error)
            if self.resources is not None:
                self.resources.kill_if_exceeded(
                    pid,
                    reason=reason,
                    limit={"kind": exc.metrics.limit_kind, "metrics": self._metrics_json(exc.metrics)},
                )
        else:
            reason = str(exc)
        self.events.emit(
            EventType.TOOL_FAILED,
            source=resource,
            target=pid,
            payload={"call_id": call_id, "error": reason, "policy_decision": "timeout"},
        )
        self.audit.record(
            actor=pid,
            action="tool.call",
            target=resource,
            decision={
                "ok": False,
                "tool": handle.name,
                "policy_decision": "timeout" if charge_error is None else "resource_limit",
                "error": reason,
                "metrics": self._metrics_json(exc.metrics),
            },
        )
        return reason

    def _metrics_json(self, metrics: CommandMetrics | None) -> dict[str, Any] | None:
        if metrics is None:
            return None
        return {
            "wall_seconds": metrics.wall_seconds,
            "cpu_seconds": metrics.cpu_seconds,
            "peak_memory_bytes": metrics.peak_memory_bytes,
            "killed": metrics.killed,
            "limit_kind": metrics.limit_kind,
        }

    def _metrics_from_validation(self, value: Any) -> CommandMetrics | None:
        if not isinstance(value, dict):
            return None
        return CommandMetrics(
            wall_seconds=float(value.get("wall_seconds") or 0.0),
            cpu_seconds=float(value.get("cpu_seconds") or 0.0),
            peak_memory_bytes=int(value.get("peak_memory_bytes") or 0),
            killed=bool(value.get("killed", False)),
            limit_kind=str(value["limit_kind"]) if value.get("limit_kind") is not None else None,
        )

    def _aggregate_command_metrics(self, metrics: list[CommandMetrics]) -> dict[str, Any]:
        if not metrics:
            return {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "peak_memory_bytes": 0,
                "killed": False,
                "limit_kind": None,
            }
        return {
            "wall_seconds": sum(item.wall_seconds for item in metrics),
            "cpu_seconds": sum(item.cpu_seconds for item in metrics),
            "peak_memory_bytes": max(item.peak_memory_bytes for item in metrics),
            "killed": any(item.killed for item in metrics),
            "limit_kind": next((item.limit_kind for item in metrics if item.limit_kind), None),
        }

    def _bounded_validation_logs(self, logs: list[str]) -> str:
        text = "\n".join(logs)
        if len(text) <= self.config.tools.jit_validation_log_max_chars:
            return text
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        return (
            text[: self.config.tools.jit_validation_log_max_chars]
            + f"\n[validation logs truncated chars={len(text)} sha256={digest}]"
        )

    def _elapsed(self, started_at: float) -> float:
        return max(0.0, time.perf_counter() - started_at)

    def resolve(self, tool: ToolHandle | str, pid: str | None = None) -> ToolHandle:
        if isinstance(tool, ToolHandle):
            return tool
        process_tool_id: str | None = None
        if pid is not None:
            process = self.store.get_process(pid)
            if process is not None and tool in process.tool_table:
                process_tool_id = process.tool_table[tool]
                if process_tool_id in self._handles:
                    return self._handles[process_tool_id]
        if tool in self._handles:
            handle = self._handles[tool]
            if pid is None and handle.tool_id in self._jit_sources:
                raise NotFound(f"tool not found: {tool}")
            return handle
        if tool in self._tool_ids_by_name:
            return self._handles[self._tool_ids_by_name[tool]]
        for row in self.store.list_tools():
            row_tool_id = str(row["tool_id"])
            is_direct_id = row_tool_id == tool
            is_process_local_name = process_tool_id is not None and row_tool_id == process_tool_id
            is_name_match = row["name"] == tool
            if not (is_direct_id or is_name_match):
                continue
            if bool(row["ephemeral"]) and pid is None:
                continue
            if row_tool_id not in self._tools and row_tool_id not in self._jit_sources:
                raise NotFound(f"tool implementation not loaded: {row_tool_id}")
            handle = ToolHandle(tool_id=row_tool_id, name=row["name"], capability_id=None, scope=row["scope"])
            self._handles[handle.tool_id] = handle
            if not bool(row["ephemeral"]):
                self._tool_ids_by_name.setdefault(handle.name, handle.tool_id)
            return handle
        raise NotFound(f"tool not found: {tool}")

    def list(self) -> builtins.list[dict[str, Any]]:
        return self.store.list_tools()

    def visible_tools(self, pid: str) -> builtins.list[dict[str, Any]]:
        visible_ids = self._visible_tool_ids(pid)
        return [row for row in self.store.list_tools() if row["tool_id"] in visible_ids]

    def model_visible_tools(self, pid: str) -> builtins.list[dict[str, Any]]:
        rows = self.visible_tools(pid)
        if self._jit_exposure_for_process(pid) != JIT_TOOL_EXPOSURE_MULTIPLEXED:
            return rows
        static_rows = [row for row in rows if str(row.get("tool_id")) not in self._jit_sources]
        if any(str(row.get("tool_id")) in self._jit_sources for row in rows):
            static_rows.append(self._jit_multiplexer_row())
        return static_rows

    def model_tool_names(self, pid: str) -> builtins.list[str]:
        names = [str(row.get("name") or "") for row in self.model_visible_tools(pid)]
        return sorted(name for name in names if name)

    def model_tool_table(self, pid: str) -> dict[str, str]:
        return {
            str(row["name"]): str(row["tool_id"])
            for row in self.model_visible_tools(pid)
            if row.get("name") and row.get("tool_id")
        }

    def model_loaded_skills(self, pid: str) -> dict[str, Any]:
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        hidden = self._hidden_jit_tool_names(pid)
        if not hidden:
            return dict(process.loaded_skills)
        model_loaded: dict[str, Any] = {}
        for skill_id, loaded in process.loaded_skills.items():
            if not isinstance(loaded, dict):
                model_loaded[skill_id] = self._redact_hidden_jit_names(loaded, hidden)
                continue
            entry = self._redact_hidden_jit_names(dict(loaded), hidden)
            entry["jit_tool_ids"] = {}
            if isinstance(entry.get("tool_names"), list):
                entry["tool_names"] = [
                    name for name in entry["tool_names"]
                    if name != "<multiplexed_jit_tool>"
                ]
            model_loaded[skill_id] = entry
        return model_loaded

    def redact_model_context(self, pid: str, value: Any) -> Any:
        hidden = self._hidden_jit_tool_names(pid)
        if not hidden:
            return value
        return self._redact_hidden_jit_names(value, hidden)

    def openai_tool_schemas(self, pid: str | None = None) -> builtins.list[dict[str, Any]]:
        tool_ids = self._visible_tool_ids(pid) if pid is not None else set(self._tools)
        multiplex_jit = pid is not None and self._jit_exposure_for_process(pid) == JIT_TOOL_EXPOSURE_MULTIPLEXED
        has_visible_jit = any(tool_id in self._jit_sources for tool_id in tool_ids)
        schemas: builtins.list[dict[str, Any]] = []
        for tool_id in sorted(tool_ids, key=self._tool_sort_key):
            if tool_id in self._tools:
                schemas.append(self._tools[tool_id].to_openai_chat_tool(config=self.config))
                continue
            if tool_id not in self._jit_sources:
                continue
            if multiplex_jit:
                continue
            spec = self.store.get_tool_spec(tool_id)
            if spec is None:
                continue
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema,
                    },
                }
            )
        if multiplex_jit and has_visible_jit:
            schemas.append(self._jit_multiplexer_openai_schema())
        return schemas

    def normalize_model_action(self, pid: str, action: dict[str, Any]) -> dict[str, Any]:
        name = str(action.get("action") or "").strip()
        if name == JIT_MULTIPLEXER_TOOL_NAME:
            if self._jit_exposure_for_process(pid) != JIT_TOOL_EXPOSURE_MULTIPLEXED:
                raise ValueError(f"{JIT_MULTIPLEXER_TOOL_NAME} is not available for this image")
            return self._normalize_multiplexed_jit_action(pid, action)
        if self._jit_exposure_for_process(pid) == JIT_TOOL_EXPOSURE_MULTIPLEXED and self._is_visible_jit_name(pid, name):
            raise ValueError(f"JIT tool {name!r} must be called through {JIT_MULTIPLEXER_TOOL_NAME}")
        if self._is_visible_jit_name(pid, name):
            handle = self.resolve(name, pid=pid)
            args = {key: value for key, value in action.items() if key != "action"}
            self._validate_jit_arguments(handle, args)
        return action

    def _normalize_multiplexed_jit_action(self, pid: str, action: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(action.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError(f"{JIT_MULTIPLEXER_TOOL_NAME} requires a non-empty tool_name")
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"{JIT_MULTIPLEXER_TOOL_NAME}.arguments must be an object")
        try:
            handle = self.resolve(tool_name, pid=pid)
        except NotFound as exc:
            raise ValueError(f"JIT tool is not available in this process: {tool_name}") from exc
        if handle.tool_id not in self._jit_sources:
            raise ValueError(f"{JIT_MULTIPLEXER_TOOL_NAME} can only dispatch process-local JIT tools: {tool_name}")
        if not self._process_has_tool(pid, handle):
            raise ValueError(f"JIT tool is not in process tool table: {tool_name}")
        self._validate_jit_arguments(handle, arguments)
        return {**arguments, "action": tool_name}

    def _validate_jit_arguments(self, handle: ToolHandle, arguments: dict[str, Any]) -> None:
        spec = self.store.get_tool_spec(handle.tool_id)
        schema = spec.input_schema if spec is not None and spec.input_schema else {"type": "object"}
        try:
            jsonschema_validate(instance=arguments, schema=schema)
        except JsonSchemaValidationError as exc:
            path = ".".join(str(part) for part in exc.path)
            location = f" at {path}" if path else ""
            raise ValueError(f"arguments for JIT tool {handle.name!r} do not match input_schema{location}: {exc.message}") from exc
        except JsonSchemaSchemaError as exc:
            raise ValueError(f"JIT tool {handle.name!r} has invalid input_schema: {exc.message}") from exc

    def _validate_jit_output(self, handle: ToolHandle, value: Any) -> None:
        spec = self.store.get_tool_spec(handle.tool_id)
        schema = spec.output_schema if spec is not None and spec.output_schema else {}
        if not schema:
            return
        try:
            jsonschema_validate(instance=value, schema=schema)
        except JsonSchemaValidationError as exc:
            path = ".".join(str(part) for part in exc.path)
            location = f" at {path}" if path else ""
            raise ValueError(f"output for JIT tool {handle.name!r} does not match output_schema{location}: {exc.message}") from exc
        except JsonSchemaSchemaError as exc:
            raise ValueError(f"JIT tool {handle.name!r} has invalid output_schema: {exc.message}") from exc

    def _is_visible_jit_name(self, pid: str, name: str) -> bool:
        if not name:
            return False
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        tool_id = process.tool_table.get(name)
        return tool_id in self._jit_sources if tool_id is not None else False

    def _hidden_jit_tool_names(self, pid: str) -> set[str]:
        if self._jit_exposure_for_process(pid) != JIT_TOOL_EXPOSURE_MULTIPLEXED:
            return set()
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return {
            name
            for name, tool_id in process.tool_table.items()
            if tool_id in self._jit_sources
        }

    def _redact_hidden_jit_names(self, value: Any, hidden: set[str]) -> Any:
        if isinstance(value, str):
            redacted = value
            for name in hidden:
                redacted = redacted.replace(name, "<multiplexed_jit_tool>")
            return redacted
        if isinstance(value, list):
            return [self._redact_hidden_jit_names(item, hidden) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_hidden_jit_names(item, hidden) for item in value)
        if isinstance(value, dict):
            return {
                self._redact_hidden_jit_names(key, hidden): self._redact_hidden_jit_names(item, hidden)
                for key, item in value.items()
            }
        return value

    def _jit_exposure_for_process(self, pid: str | None) -> str:
        if pid is None:
            return ""
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return ""
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        image = getattr(runtime, "images", {}).get(process.image_id)
        return str(getattr(image, "jit_tool_exposure", "") or "")

    def _jit_multiplexer_row(self) -> dict[str, Any]:
        return {
            "tool_id": f"protocol:{JIT_MULTIPLEXER_TOOL_NAME}",
            "name": JIT_MULTIPLEXER_TOOL_NAME,
            "spec_json": dumps(_JIT_MULTIPLEXER_SPEC),
            "scope": "llm_protocol",
            "registered_by": "runtime",
            "created_at": "",
            "ephemeral": 1,
        }

    def _jit_multiplexer_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": _JIT_MULTIPLEXER_SPEC.name,
                "description": _JIT_MULTIPLEXER_SPEC.description,
                "parameters": _JIT_MULTIPLEXER_SPEC.input_schema,
            },
        }

    def _tool_sort_key(self, tool_id: str) -> tuple[str, str]:
        handle = self._handles.get(tool_id)
        if handle is not None:
            return (handle.name, tool_id)
        spec = self.store.get_tool_spec(tool_id)
        if spec is not None:
            return (spec.name, tool_id)
        return (tool_id, tool_id)

    def _context(
        self,
        pid: str,
        handle: ToolHandle,
        call_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolContext:
        runtime = getattr(self, "runtime", None)
        selected_metadata = {
            "tool_id": handle.tool_id,
            "tool_name": handle.name,
        }
        selected_metadata.update(dict(metadata or {}))
        return ToolContext(
            trace_id=call_id,
            call_id=call_id,
            pid=pid,
            workspace_id=str(self.workspace_root),
            runtime=runtime,
            metadata=selected_metadata,
        )

    def _get_candidate(self, candidate_id: str) -> ToolCandidate:
        candidate = self.store.get_tool_candidate(candidate_id)
        if candidate is None:
            raise NotFound(f"tool candidate not found: {candidate_id}")
        return candidate

    def _require_candidate_owner(self, candidate: ToolCandidate, pid: str) -> None:
        if candidate.pid != pid:
            raise ValidationError(
                f"tool candidate {candidate.candidate_id} belongs to process {candidate.pid}, not {pid}"
            )

    def _name_collides_with_static_tool(self, name: str) -> bool:
        mapped = self._tool_ids_by_name.get(name)
        if mapped in self._tools:
            return True
        return any(row["name"] == name and not bool(row["ephemeral"]) for row in self.store.list_tools())

    def _process_has_tool(self, pid: str, handle: ToolHandle) -> bool:
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return process.tool_table.get(handle.name) == handle.tool_id

    def _handle_for_unregistration(self, tool: ToolHandle | str) -> ToolHandle | None:
        if isinstance(tool, ToolHandle):
            return tool
        if tool in self._handles:
            return self._handles[tool]
        tool_id = self._tool_ids_by_name.get(str(tool))
        if tool_id is not None:
            return self._handles.get(tool_id)
        return None

    def _process_status_error(self, pid: str) -> str | None:
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        if process.status in _TERMINAL_PROCESS_STATUSES:
            return f"cannot call tools for terminal process {pid}: {process.status.value}"
        if process.status not in _TOOL_CALLABLE_PROCESS_STATUSES:
            return f"cannot call tools for process {pid}: status={process.status.value} is not runnable"
        return None

    def _visible_tool_ids(self, pid: str) -> set[str]:
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return set(process.tool_table.values())

    def _oversize_result_observation(self) -> dict[str, Any]:
        return {
            "redacted": True,
            "truncated": True,
            "preview": "[tool result omitted after size-limit failure]",
        }


def _stable_static_tool_id(name: str, digest_chars: int = _TOOL_DEFAULTS.static_tool_id_digest_chars) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:digest_chars]
    return f"tool_static_{digest}"


def _conservative_jit_tool_spec(spec: ToolSpec) -> ToolSpec:
    policy = dict(spec.policy)
    declared_permissions = _string_set(policy.get("declared_permissions")) | set(_JIT_DECLARED_PERMISSIONS)
    policy["side_effects"] = True
    policy["idempotent"] = False
    policy["declared_permissions"] = sorted(declared_permissions)
    return ToolSpec(
        name=spec.name,
        description=spec.description,
        version=spec.version,
        input_schema=dict(spec.input_schema),
        output_schema=dict(spec.output_schema),
        policy=policy,
        tags=list(dict.fromkeys([*spec.tags, "jit", "side_effect"])),
        metadata=dict(spec.metadata),
        required_capabilities=[dict(item) for item in spec.required_capabilities],
        side_effects=sorted(set(spec.side_effects) | declared_permissions),
    )


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item) for item in value}
    return {str(value)}
