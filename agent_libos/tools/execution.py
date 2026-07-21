from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema import validate as jsonschema_validate

from agent_libos.config import AgentLibOSConfig
from agent_libos.memory.data_labels import propagate_object_labels
from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.models import (
    DataFlowContext,
    EventType,
    ObjectHandle,
    ObjectMetadata,
    ObjectOwnerKind,
    ObjectType,
    ProcessStatus,
    Provenance,
    ResourceUsage,
    ToolCallResult,
    ToolHandle,
)
from agent_libos.models.exceptions import (
    HumanApprovalRequired,
    NotFound,
    ProcessMessageWaitRequired,
    ProcessRevisionConflict,
    ProcessWaitRequired,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.process_execution import (
    current_process_execution_token,
    trusted_post_exec_completion_mutation,
    trusted_terminal_process_mutation,
)
from agent_libos.ports import AuditPort, DataFlowPort, EventPort, OperationPort
from agent_libos.storage.repositories import (
    EvidenceRepository,
    ExtensionRepository,
    ProcessRepository,
)
from agent_libos.substrate import (
    CommandMetrics,
    SubprocessLimitExceeded,
    SubprocessLimits,
    SubprocessTimeoutExpired,
)
from agent_libos.tools.base import (
    ToolContext,
    attach_wait_data_flow_context,
    wait_data_flow_context,
)
from agent_libos.tools.observability import ensure_json_size, sanitize_for_observability
from agent_libos.tools.registry import ToolRegistry
from agent_libos.tools.sandbox import SandboxBackend, SandboxExecutionResult
from agent_libos.utils.ids import new_id
from agent_libos.utils.public_errors import (
    provider_error_envelope,
    public_exception_message,
)


_TERMINAL_PROCESS_STATUSES = {
    ProcessStatus.EXITED,
    ProcessStatus.FAILED,
    ProcessStatus.KILLED,
}
_TOOL_CALLABLE_PROCESS_STATUSES = {
    ProcessStatus.RUNNABLE,
    ProcessStatus.RUNNING,
}


class JITSyscallSession(Protocol):
    """Narrow syscall session used by one sandboxed JIT invocation."""

    @property
    def observed_context(self) -> DataFlowContext: ...

    async def handle(self, name: str, args: dict[str, Any]) -> Any: ...

    async def apply_deferred_lifecycle(
        self,
        tool_result: ObjectHandle | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _Invocation:
    pid: str
    handle: ToolHandle
    call_id: str
    resource: str
    context_metadata: dict[str, Any] | None


@dataclass(slots=True)
class _InvocationOutput:
    payload: Any
    result_payload: dict[str, Any]
    jit_session: JITSyscallSession | None = None
    result_was_omitted: bool = False


class ToolExecutionService:
    """Own tool invocation, data-flow, resource, and durable operation state."""

    def __init__(
        self,
        *,
        data_flow: DataFlowPort,
        operations: OperationPort,
        evidence: EvidenceRepository,
        processes: ProcessRepository,
        extensions: ExtensionRepository,
        memory: ObjectMemoryManager,
        audit: AuditPort,
        events: EventPort,
        resources: Any | None,
        registry: ToolRegistry,
        sandbox: SandboxBackend,
        config: AgentLibOSConfig,
        jit_session_factory: Callable[[str], JITSyscallSession],
        tool_context_host: Any,
        workspace_root: str | Path,
        registry_lifecycle_lock: threading.RLock,
    ) -> None:
        self._data_flow = data_flow
        self._operations = operations
        self._evidence = evidence
        self._processes = processes
        self._extensions = extensions
        self._memory = memory
        self._audit = audit
        self._events = events
        self._resources = resources
        self._registry = registry
        self._sandbox = sandbox
        self._config = config
        self._jit_session_factory = jit_session_factory
        self._tool_context_host = tool_context_host
        self._workspace_root = Path(workspace_root).resolve()
        self._registry_lifecycle_lock = registry_lifecycle_lock

    @property
    def sandbox(self) -> SandboxBackend:
        return self._sandbox

    @sandbox.setter
    def sandbox(self, value: SandboxBackend) -> None:
        self._sandbox = value

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
            return asyncio.run(
                self.acall(pid, tool, args, context_metadata=context_metadata)
            )
        raise RuntimeError(
            "Cannot call ToolBroker.call() inside a running event loop. "
            "Use await acall(...)."
        )

    async def acall(
        self,
        pid: str,
        tool: ToolHandle | str,
        args: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        context = _trusted_context(context_metadata, self._data_flow.current_context())
        token = self._data_flow.push(context)
        try:
            result = await self._with_operation(
                pid,
                tool,
                args,
                context_metadata=context_metadata,
            )
        except (HumanApprovalRequired, ProcessWaitRequired, ProcessMessageWaitRequired) as exc:
            carried = wait_data_flow_context(exc)
            self._data_flow.reset(token)
            if carried is not None:
                self._data_flow.observe_ingress(carried)
            raise
        except BaseException:
            self._data_flow.reset(token)
            raise
        self._data_flow.reset(token)
        return result

    async def _with_operation(
        self,
        pid: str,
        tool: ToolHandle | str,
        args: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None,
    ) -> ToolCallResult:
        selected_name = tool.name if isinstance(tool, ToolHandle) else str(tool)
        metadata = context_metadata or {}
        resume_id = str(metadata.get("operation_id") or "") or None
        parent_id = str(metadata.get("parent_operation_id") or "") or None
        with self._operations.scope(
            kind="tool_call",
            name=f"tool.{selected_name}",
            actor=pid,
            pid=pid,
            expected_roles=("invocation", "audit", "event"),
            operation_id=resume_id,
            parent_operation_id=parent_id,
            auto_finish=False,
        ) as operation:
            result = await self.execute(
                pid,
                tool,
                args,
                context_metadata=context_metadata,
            )
            self._record_result(operation.operation_id, selected_name, result)
            outcome = self._outcome(operation, result)
            self._operations.finish(outcome, operation_id=operation.operation_id)
            return result

    async def execute(
        self,
        pid: str,
        tool: ToolHandle | str,
        args: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        prepared = self._prepare_invocation(pid, tool, args, context_metadata)
        if isinstance(prepared, ToolCallResult):
            return prepared

        started_at = time.perf_counter()
        try:
            output = await self._invoke(prepared, args, started_at)
        except SubprocessLimitExceeded as exc:
            return self._subprocess_limit_result(prepared, exc)
        except SubprocessTimeoutExpired as exc:
            return self._subprocess_timeout_result(prepared, exc)
        except HumanApprovalRequired as exc:
            self._record_wait(prepared, exc, started_at)
            raise
        except ProcessWaitRequired as exc:
            self._record_wait(prepared, exc, started_at)
            raise
        except ProcessMessageWaitRequired as exc:
            self._record_wait(prepared, exc, started_at)
            raise
        except ValueError as exc:
            return self._invocation_error_result(
                prepared,
                exc,
                started_at,
                policy_decision="validation_error",
            )
        except Exception as exc:
            return self._invocation_error_result(
                prepared,
                exc,
                started_at,
                policy_decision="allow",
            )

        if isinstance(output, ToolCallResult):
            return output
        bounded = self._bound_result(prepared, output, started_at)
        if isinstance(bounded, ToolCallResult):
            return bounded
        try:
            return await self._persist_success(prepared, bounded, started_at)
        except ProcessRevisionConflict as exc:
            # Resource enforcement can terminalize the process after the
            # provider returns but before its ToolResult capability attaches.
            # Preserve that terminal winner and surface one audited failure.
            latest = self._processes.get_process(prepared.pid)
            if latest is None or latest.status not in _TERMINAL_PROCESS_STATUSES:
                raise
            return self._invocation_error_result(
                prepared,
                exc,
                started_at,
                policy_decision="allow",
            )

    def _prepare_invocation(
        self,
        pid: str,
        tool: ToolHandle | str,
        args: dict[str, Any],
        context_metadata: dict[str, Any] | None,
    ) -> _Invocation | ToolCallResult:
        selected_name = tool.name if isinstance(tool, ToolHandle) else str(tool)
        try:
            with self._registry_lifecycle_lock:
                handle = self._registry.resolve(tool, pid=pid)
        except NotFound:
            process = self._processes.get_process(pid)
            if process is None or selected_name in process.tool_table:
                raise
            return self._deny_result(
                pid=pid,
                tool_id=selected_name,
                tool_name=selected_name,
                resource=f"tool:{selected_name}",
                error=f"tool is not in process tool table: {selected_name}",
                reason="tool_not_in_process_table",
            )

        resource = f"tool:{handle.tool_id}"
        status_error = self._process_status_error(pid)
        if status_error is not None:
            return self._deny_result(
                pid=pid,
                tool_id=handle.tool_id,
                tool_name=handle.name,
                resource=resource,
                error=status_error,
                reason="process_not_tool_callable",
            )
        if not self._registry.process_has_tool(pid, handle):
            return self._deny_result(
                pid=pid,
                tool_id=handle.tool_id,
                tool_name=handle.name,
                resource=resource,
                error=f"tool is not in process tool table: {handle.name}",
                reason="tool_not_in_process_table",
            )

        call_id = new_id("tcall")
        try:
            self._charge_tool_call(pid, handle, call_id)
        except ResourceLimitExceeded as exc:
            return self._preflight_error_result(
                pid,
                handle,
                resource,
                call_id,
                str(exc),
                policy_decision="resource_limit",
            )
        try:
            ensure_json_size(
                args,
                self._config.tools.tool_call_args_hard_limit_bytes,
                "tool call arguments",
            )
        except ValidationError as exc:
            return self._preflight_error_result(
                pid,
                handle,
                resource,
                call_id,
                str(exc),
                policy_decision="validation_error",
            )
        self._events.emit(
            EventType.TOOL_CALLED,
            source=pid,
            target=resource,
            payload={"call_id": call_id, "args": sanitize_for_observability(args)},
        )
        return _Invocation(pid, handle, call_id, resource, context_metadata)

    def _deny_result(
        self,
        *,
        pid: str,
        tool_id: str,
        tool_name: str,
        resource: str,
        error: str,
        reason: str,
    ) -> ToolCallResult:
        call_id = new_id("tcall")
        self._events.emit(
            EventType.TOOL_FAILED,
            source=resource,
            target=pid,
            payload={"call_id": call_id, "error": error, "policy_decision": "deny"},
        )
        decision: dict[str, Any] = {
            "ok": False,
            "tool": tool_name,
            "policy_decision": "deny",
            "policy_reason": reason,
        }
        if reason == "process_not_tool_callable":
            decision["error"] = error
        self._audit.record(
            actor=pid,
            action="tool.call",
            target=resource,
            decision=decision,
        )
        return ToolCallResult(
            call_id=call_id,
            tool_id=tool_id,
            result_handle=None,
            payload=None,
            ok=False,
            error=error,
        )

    def _preflight_error_result(
        self,
        pid: str,
        handle: ToolHandle,
        resource: str,
        call_id: str,
        error: str,
        *,
        policy_decision: str,
    ) -> ToolCallResult:
        self._events.emit(
            EventType.TOOL_FAILED,
            source=resource,
            target=pid,
            payload={
                "call_id": call_id,
                "error": error,
                "policy_decision": policy_decision,
            },
        )
        self._audit.record(
            actor=pid,
            action="tool.call",
            target=resource,
            decision={
                "ok": False,
                "tool": handle.name,
                "policy_decision": policy_decision,
                "error": error,
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

    async def _invoke(
        self,
        invocation: _Invocation,
        args: dict[str, Any],
        started_at: float,
    ) -> _InvocationOutput | ToolCallResult:
        implementation = self._registry.implementation(invocation.handle.tool_id)
        if implementation is not None:
            tool_result = await implementation.ainvoke(
                args,
                self._context(invocation),
            )
            tool_metadata = self._merge_returned_context(dict(tool_result.metadata))
            tool_result.metadata = tool_metadata
            if not tool_result.ok:
                return self._structured_failure(invocation, tool_result, started_at)
            payload = tool_result.data
            return _InvocationOutput(
                payload=payload,
                result_payload={
                    "tool_id": invocation.handle.tool_id,
                    "tool_name": invocation.handle.name,
                    "result": payload,
                    "content": tool_result.content,
                    "artifacts": [
                        artifact.model_dump(mode="json")
                        for artifact in tool_result.artifacts
                    ],
                    "metadata": tool_metadata,
                },
            )
        if not self._registry.is_jit(invocation.handle.tool_id):
            raise NotFound(
                f"tool implementation not loaded: {invocation.handle.tool_id}"
            )
        return await self._invoke_jit(invocation, args)

    def _merge_returned_context(self, metadata: dict[str, Any]) -> dict[str, Any]:
        inbound = metadata.get("data_flow_context")
        if inbound is None:
            return metadata
        returned = _trusted_context(
            {"data_flow_context": inbound},
            self._data_flow.current_context(),
        )
        combined = DataFlowContext.aggregate(
            (self._data_flow.current_context(), returned)
        )
        self._data_flow.observe_ingress(combined)
        metadata["data_flow_context"] = combined.to_dict()
        return metadata

    async def _invoke_jit(
        self,
        invocation: _Invocation,
        args: dict[str, Any],
    ) -> _InvocationOutput:
        self.validate_jit_arguments(invocation.handle, args)
        session = self._jit_session_factory(invocation.pid)
        source = self._registry.jit_source(invocation.handle.tool_id)
        if source is None:
            raise NotFound(
                f"tool implementation not loaded: {invocation.handle.tool_id}"
            )
        try:
            sandbox_result = await self._run_sandbox_source(
                source,
                args,
                pid=invocation.pid,
                syscall_handler=session.handle,
            )
        finally:
            self._data_flow.observe_ingress(session.observed_context)
        if isinstance(sandbox_result, SandboxExecutionResult):
            payload = sandbox_result.value
            self._charge_subprocess_metrics(
                invocation.pid,
                sandbox_result.metrics,
                source="tool.deno",
                context={
                    "tool": invocation.handle.name,
                    "tool_id": invocation.handle.tool_id,
                },
            )
        else:
            payload = sandbox_result
        self.validate_jit_output(invocation.handle, payload)
        return _InvocationOutput(
            payload=payload,
            result_payload={
                "tool_id": invocation.handle.tool_id,
                "tool_name": invocation.handle.name,
                "result": payload,
            },
            jit_session=session,
        )

    def _structured_failure(
        self,
        invocation: _Invocation,
        tool_result: Any,
        started_at: float,
    ) -> ToolCallResult:
        error = tool_result.error.message if tool_result.error else tool_result.content
        payload = tool_result.model_dump(mode="json")
        result_handle = self._persist_labeled_tool_failure(
            invocation,
            payload=payload,
        )
        self._events.emit(
            EventType.TOOL_FAILED,
            source=invocation.resource,
            target=invocation.pid,
            payload={
                "call_id": invocation.call_id,
                "error": error,
                "result_oid": result_handle.oid if result_handle else None,
                "tool_result": sanitize_for_observability(payload),
            },
        )
        self._audit.record(
            actor=invocation.pid,
            action="tool.call",
            target=invocation.resource,
            output_refs=[result_handle.oid] if result_handle else [],
            decision={
                "ok": False,
                "tool": invocation.handle.name,
                "policy_decision": "allow",
                "tool_result": sanitize_for_observability(payload),
                "tool_wall_seconds": self._elapsed(started_at),
            },
        )
        return ToolCallResult(
            call_id=invocation.call_id,
            tool_id=invocation.handle.tool_id,
            result_handle=result_handle,
            payload=payload,
            ok=False,
            error=error,
        )

    def _subprocess_limit_result(
        self,
        invocation: _Invocation,
        exc: SubprocessLimitExceeded,
    ) -> ToolCallResult:
        self._handle_subprocess_limit(invocation, exc)
        result_handle = self._persist_exception_tool_failure(
            invocation,
            error=exc,
            policy_decision="resource_limit",
        )
        return ToolCallResult(
            call_id=invocation.call_id,
            tool_id=invocation.handle.tool_id,
            result_handle=result_handle,
            payload=None,
            ok=False,
            error=str(exc),
        )

    def _subprocess_timeout_result(
        self,
        invocation: _Invocation,
        exc: SubprocessTimeoutExpired,
    ) -> ToolCallResult:
        error = self._handle_subprocess_timeout(invocation, exc)
        result_handle = self._persist_exception_tool_failure(
            invocation,
            error=exc,
            policy_decision="timeout",
            message=error,
        )
        return ToolCallResult(
            call_id=invocation.call_id,
            tool_id=invocation.handle.tool_id,
            result_handle=result_handle,
            payload=None,
            ok=False,
            error=error,
        )

    def _record_wait(
        self,
        invocation: _Invocation,
        exc: HumanApprovalRequired | ProcessWaitRequired | ProcessMessageWaitRequired,
        started_at: float,
    ) -> None:
        self._preserve_wait_data_flow_context(exc)
        if isinstance(exc, HumanApprovalRequired):
            action = "tool.call_waiting_human"
            decision = {
                "policy_decision": "require_human_approval",
                "request_id": exc.request_id,
            }
        elif isinstance(exc, ProcessWaitRequired):
            action = "tool.call_waiting_process"
            decision = {
                "policy_decision": "wait_for_child",
                "child_pid": exc.child_pid,
            }
        else:
            action = "tool.call_waiting_message"
            decision = {
                "policy_decision": "wait_for_process_message",
                "recipient_pid": exc.recipient_pid,
                "filters": exc.filters,
            }
        self._audit.record(
            actor=invocation.pid,
            action=action,
            target=invocation.resource,
            decision={
                "ok": False,
                "tool": invocation.handle.name,
                **decision,
                "tool_wall_seconds": self._elapsed(started_at),
            },
        )

    def _invocation_error_result(
        self,
        invocation: _Invocation,
        exc: Exception,
        started_at: float,
        *,
        policy_decision: str,
    ) -> ToolCallResult:
        error = public_exception_message(exc)
        observed = self._error_observation(error)
        result_handle = self._persist_exception_tool_failure(
            invocation,
            error=exc,
            policy_decision=policy_decision,
        )
        event_payload: dict[str, Any] = {
            "call_id": invocation.call_id,
            "error": observed,
            "result_oid": result_handle.oid if result_handle else None,
        }
        if policy_decision == "validation_error":
            event_payload["policy_decision"] = policy_decision
        self._events.emit(
            EventType.TOOL_FAILED,
            source=invocation.resource,
            target=invocation.pid,
            payload=event_payload,
        )
        self._audit.record(
            actor=invocation.pid,
            action="tool.call",
            target=invocation.resource,
            output_refs=[result_handle.oid] if result_handle else [],
            decision={
                "ok": False,
                "tool": invocation.handle.name,
                "policy_decision": policy_decision,
                "error": observed,
                "tool_wall_seconds": self._elapsed(started_at),
            },
        )
        return ToolCallResult(
            call_id=invocation.call_id,
            tool_id=invocation.handle.tool_id,
            result_handle=result_handle,
            payload=None,
            ok=False,
            error=error,
        )

    def _bound_result(
        self,
        invocation: _Invocation,
        output: _InvocationOutput,
        started_at: float,
    ) -> _InvocationOutput | ToolCallResult:
        try:
            ensure_json_size(
                output.result_payload,
                self._tool_result_persistence_limit(),
                "tool result payload",
            )
            return output
        except ValidationError as exc:
            error = str(exc)
        if self.has_side_effects(invocation.handle):
            output.result_was_omitted = True
            output.payload = {
                "result_omitted": True,
                "reason": error,
                "tool_id": invocation.handle.tool_id,
                "tool_name": invocation.handle.name,
            }
            output.result_payload = {
                "tool_id": invocation.handle.tool_id,
                "tool_name": invocation.handle.name,
                "result": output.payload,
                "content": "[tool result omitted after size-limit failure]",
                "artifacts": [],
                "metadata": {
                    "result_omitted": True,
                    "omission_reason": error,
                },
            }
            return output
        self._events.emit(
            EventType.TOOL_FAILED,
            source=invocation.resource,
            target=invocation.pid,
            payload={
                "call_id": invocation.call_id,
                "error": error,
                "policy_decision": "validation_error",
            },
        )
        self._audit.record(
            actor=invocation.pid,
            action="tool.call",
            target=invocation.resource,
            decision={
                "ok": False,
                "tool": invocation.handle.name,
                "policy_decision": "validation_error",
                "error": error,
                "result": self._oversize_result_observation(),
                "tool_wall_seconds": self._elapsed(started_at),
            },
        )
        return ToolCallResult(
            call_id=invocation.call_id,
            tool_id=invocation.handle.tool_id,
            result_handle=None,
            payload=None,
            ok=False,
            error=error,
        )

    async def _persist_success(
        self,
        invocation: _Invocation,
        output: _InvocationOutput,
        started_at: float,
    ) -> ToolCallResult:
        lifecycle_error: Exception | None = None
        with self._memory.lifetime_scope(
            actor=invocation.resource,
            owner_kind=ObjectOwnerKind.PROCESS,
            owner_id=invocation.pid,
            reason="tool_result",
        ) as result_scope:
            flow = self._data_flow.current_context()
            parent_oids, durable_source_refs = self._data_flow.provenance_sources(flow)
            with self._result_process_mutation_scope(invocation):
                result_handle = result_scope.create_object(
                    pid=invocation.pid,
                    object_type=ObjectType.TOOL_RESULT,
                    payload=output.result_payload,
                    metadata=self._tool_result_metadata(invocation.handle),
                    provenance=Provenance(
                        created_from_action=f"tool.{invocation.handle.name}",
                        parent_oids=list(parent_oids),
                        source_refs=list(durable_source_refs),
                    ),
                    immutable=True,
                )
            if output.jit_session is not None:
                try:
                    await output.jit_session.apply_deferred_lifecycle(result_handle)
                except Exception as exc:
                    lifecycle_error = exc
            if lifecycle_error is None:
                result_scope.commit()
        if lifecycle_error is not None:
            return self._lifecycle_error_result(
                invocation,
                lifecycle_error,
                started_at,
            )
        self._events.emit(
            EventType.TOOL_COMPLETED,
            source=invocation.resource,
            target=invocation.pid,
            payload={
                "call_id": invocation.call_id,
                "result_oid": result_handle.oid,
            },
        )
        decision: dict[str, Any] = {
            "ok": True,
            "tool": invocation.handle.name,
            "policy_decision": "allow",
            "tool_wall_seconds": self._elapsed(started_at),
        }
        if output.result_was_omitted:
            decision["result_omitted"] = True
            decision["result"] = self._oversize_result_observation()
        self._audit.record(
            actor=invocation.pid,
            action="tool.call",
            target=invocation.resource,
            output_refs=[result_handle.oid],
            decision=decision,
        )
        return ToolCallResult(
            call_id=invocation.call_id,
            tool_id=invocation.handle.tool_id,
            result_handle=result_handle,
            payload=output.payload,
            ok=True,
        )

    def _result_process_mutation_scope(self, invocation: _Invocation) -> Any:
        if invocation.handle.name == "exec_process":
            return self._post_exec_result_scope(invocation)
        if invocation.handle.name != "process_exit":
            return nullcontext()
        process = self._processes.get_process(invocation.pid)
        if process is None:
            raise ProcessRevisionConflict(
                f"process disappeared before terminal tool-result persistence: {invocation.pid}"
            )
        return trusted_terminal_process_mutation(
            invocation.pid,
            expected_revision=process.revision,
            expected_generation=process.execution_generation,
            allowed_statuses={ProcessStatus.EXITED},
            execution_token=current_process_execution_token(),
            reason="process_exit appends its terminal tool-result capability",
        )

    def _post_exec_result_scope(self, invocation: _Invocation) -> Any:
        token = current_process_execution_token()
        if token is None:
            return nullcontext()
        operation = self._operations.current()
        if (
            operation is None
            or operation.name != "tool.exec_process"
            or operation.actor != invocation.pid
            or operation.pid != invocation.pid
        ):
            raise ProcessRevisionConflict(
                f"committed exec has no exact ToolResult operation binding: {invocation.pid}"
            )
        publications = [
            candidate
            for candidate in self._evidence.list_operations(
                root_operation_id=operation.root_operation_id
            )
            if candidate.parent_operation_id == operation.operation_id
            and candidate.name == "process.exec"
            and candidate.actor == invocation.pid
            and candidate.pid == invocation.pid
            and candidate.metadata.get("runtime_publication_kind") == "process_exec"
            and candidate.metadata.get("runtime_publication_bound") is True
        ]
        if len(publications) != 1:
            raise ProcessRevisionConflict(
                f"committed exec ToolResult binding is not unique: {invocation.pid}"
            )
        publication_operation = publications[0]
        publication_id = str(
            publication_operation.metadata.get("runtime_publication_id") or ""
        )
        if not publication_id:
            raise ProcessRevisionConflict(
                f"committed exec publication id is missing: {invocation.pid}"
            )
        process = self._processes.get_process(invocation.pid)
        if process is None:
            raise ProcessRevisionConflict(
                f"process disappeared before exec ToolResult persistence: {invocation.pid}"
            )
        return trusted_post_exec_completion_mutation(
            invocation.pid,
            publication_id=publication_id,
            operation_id=publication_operation.operation_id,
            expected_revision=process.revision,
            expected_generation=process.execution_generation,
            execution_token=token,
            reason="exec_process appends its committed ToolResult capability",
        )

    def _lifecycle_error_result(
        self,
        invocation: _Invocation,
        error: Exception,
        started_at: float,
    ) -> ToolCallResult:
        outward_error = "JIT tool failed while applying deferred lifecycle."
        failure_payload = {
            "ok": False,
            "error": {
                "type": type(error).__name__,
                "message": outward_error,
            },
            "policy_decision": "lifecycle_error",
        }
        result_handle = self._persist_exception_tool_failure(
            invocation,
            error=error,
            policy_decision="lifecycle_error",
        )
        observed = self._error_observation(outward_error)
        self._events.emit(
            EventType.TOOL_FAILED,
            source=invocation.resource,
            target=invocation.pid,
            payload={
                "call_id": invocation.call_id,
                "error": observed,
                "policy_decision": "lifecycle_error",
                "result_oid": result_handle.oid if result_handle else None,
            },
        )
        self._audit.record(
            actor=invocation.pid,
            action="tool.call",
            target=invocation.resource,
            output_refs=[result_handle.oid] if result_handle else [],
            decision={
                "ok": False,
                "tool": invocation.handle.name,
                "policy_decision": "lifecycle_error",
                "error": observed,
                "tool_wall_seconds": self._elapsed(started_at),
            },
        )
        return ToolCallResult(
            call_id=invocation.call_id,
            tool_id=invocation.handle.tool_id,
            result_handle=result_handle,
            payload=failure_payload,
            ok=False,
            error=outward_error,
        )

    def _persist_labeled_tool_failure(
        self,
        invocation: _Invocation,
        *,
        payload: dict[str, Any],
    ) -> ObjectHandle | None:
        flow = self._data_flow.current_context()
        if flow == DataFlowContext():
            return None
        process = self._processes.get_process(invocation.pid)
        if process is None or process.status in _TERMINAL_PROCESS_STATUSES:
            return None
        carrier_payload: dict[str, Any] = {
            "tool_id": invocation.handle.tool_id,
            "tool_name": invocation.handle.name,
            "ok": False,
            "failure": payload,
        }
        try:
            ensure_json_size(
                carrier_payload,
                self._tool_result_persistence_limit(),
                "tool failure result payload",
            )
        except ValidationError as exc:
            carrier_payload = {
                "tool_id": invocation.handle.tool_id,
                "tool_name": invocation.handle.name,
                "ok": False,
                "failure_omitted": True,
                "reason": str(exc),
                "metadata": {"data_flow_context": flow.to_dict()},
            }
        try:
            with self._memory.lifetime_scope(
                actor=invocation.resource,
                owner_kind=ObjectOwnerKind.PROCESS,
                owner_id=invocation.pid,
                reason="tool_failure_result",
            ) as result_scope:
                parent_oids, durable_source_refs = self._data_flow.provenance_sources(flow)
                result_handle = result_scope.create_object(
                    pid=invocation.pid,
                    object_type=ObjectType.TOOL_RESULT,
                    payload=carrier_payload,
                    metadata=self._tool_result_metadata(invocation.handle),
                    provenance=Provenance(
                        created_from_action=f"tool.{invocation.handle.name}.failure",
                        parent_oids=list(parent_oids),
                        source_refs=list(durable_source_refs),
                    ),
                    immutable=True,
                )
                result_scope.commit()
        except ProcessRevisionConflict:
            latest = self._processes.get_process(invocation.pid)
            if latest is None or latest.status in _TERMINAL_PROCESS_STATUSES:
                return None
            raise
        return result_handle

    def _persist_exception_tool_failure(
        self,
        invocation: _Invocation,
        *,
        error: BaseException,
        policy_decision: str,
        message: str | None = None,
    ) -> ObjectHandle | None:
        public_error = provider_error_envelope(error)
        error_payload: dict[str, Any] = {
            "type": (
                public_error["error_type"]
                if public_error is not None
                else type(error).__name__
            ),
            "message": (
                public_error["message"]
                if public_error is not None
                else message if message is not None else str(error)
            ),
        }
        if public_error is not None:
            error_payload.update(
                {
                    key: public_error[key]
                    for key in ("code", "error_type", "correlation_id")
                }
            )
        return self._persist_labeled_tool_failure(
            invocation,
            payload={
                "ok": False,
                "error": error_payload,
                "policy_decision": policy_decision,
            },
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
            "cached_only": True,
        }
        supported = self._supported_sandbox_kwargs()
        selected = {key: value for key, value in kwargs.items() if key in supported}
        if kwargs["limits"] is not None and "limits" not in selected:
            raise ValidationError(
                "sandbox backend must accept SubprocessLimits when resource limits are configured"
            )
        if kwargs["limits"] is not None and "return_metrics" not in selected:
            raise ValidationError("sandbox backend must return subprocess metrics")
        return await self._sandbox.arun_source(source_code, args, **selected)

    def _supported_sandbox_kwargs(self) -> set[str]:
        signature = inspect.signature(self._sandbox.arun_source)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return {
                "pid",
                "syscall_handler",
                "timeout",
                "limits",
                "return_metrics",
                "cached_only",
            }
        return set(signature.parameters)

    def _charge_tool_call(
        self,
        pid: str,
        handle: ToolHandle,
        call_id: str,
    ) -> None:
        if self._resources is None:
            return
        self._resources.charge(
            pid,
            ResourceUsage(tool_calls=1),
            source="tool.call",
            context={
                "tool": handle.name,
                "tool_id": handle.tool_id,
                "call_id": call_id,
            },
            allow_overage=False,
            kill_on_exceed=False,
        )

    def _subprocess_limits(self, pid: str) -> SubprocessLimits | None:
        if self._resources is None:
            return None
        wall = self._resources.remaining_cumulative(
            pid,
            "max_subprocess_wall_seconds",
            "subprocess_wall_seconds",
        )
        cpu = self._resources.remaining_cumulative(
            pid,
            "max_subprocess_cpu_seconds",
            "subprocess_cpu_seconds",
        )
        memory = self._resources.peak_limit(pid, "max_subprocess_memory_bytes")
        if wall is None and cpu is None and memory is None:
            return None
        return SubprocessLimits(
            wall_seconds=wall,
            cpu_seconds=cpu,
            memory_bytes=memory,
        )

    def _charge_subprocess_metrics(
        self,
        pid: str,
        metrics: CommandMetrics | None,
        *,
        source: str,
        context: dict[str, Any],
    ) -> None:
        if self._resources is None or metrics is None:
            return
        self._resources.charge(
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
        invocation: _Invocation,
        exc: SubprocessLimitExceeded,
    ) -> None:
        charge_error: ResourceLimitExceeded | None = None
        try:
            self._charge_subprocess_metrics(
                invocation.pid,
                exc.metrics,
                source="tool.deno",
                context={
                    "tool": invocation.handle.name,
                    "tool_id": invocation.handle.tool_id,
                },
            )
        except ResourceLimitExceeded as resource_exc:
            charge_error = resource_exc
        reason = str(charge_error or exc)
        if self._resources is not None:
            self._resources.kill_if_exceeded(
                invocation.pid,
                reason=reason,
                limit={
                    "kind": exc.metrics.limit_kind,
                    "metrics": self._metrics_json(exc.metrics),
                },
            )
        self._record_subprocess_failure(
            invocation,
            reason,
            "resource_limit",
            exc.metrics,
        )

    def _handle_subprocess_timeout(
        self,
        invocation: _Invocation,
        exc: SubprocessTimeoutExpired,
    ) -> str:
        charge_error: ResourceLimitExceeded | None = None
        try:
            self._charge_subprocess_metrics(
                invocation.pid,
                exc.metrics,
                source="tool.deno",
                context={
                    "tool": invocation.handle.name,
                    "tool_id": invocation.handle.tool_id,
                },
            )
        except ResourceLimitExceeded as resource_exc:
            charge_error = resource_exc
        reason = str(charge_error or exc)
        if charge_error is not None and self._resources is not None:
            self._resources.kill_if_exceeded(
                invocation.pid,
                reason=reason,
                limit={
                    "kind": exc.metrics.limit_kind,
                    "metrics": self._metrics_json(exc.metrics),
                },
            )
        self._record_subprocess_failure(
            invocation,
            reason,
            "timeout" if charge_error is None else "resource_limit",
            exc.metrics,
        )
        return reason

    def _record_subprocess_failure(
        self,
        invocation: _Invocation,
        reason: str,
        policy_decision: str,
        metrics: CommandMetrics,
    ) -> None:
        self._events.emit(
            EventType.TOOL_FAILED,
            source=invocation.resource,
            target=invocation.pid,
            payload={
                "call_id": invocation.call_id,
                "error": reason,
                "policy_decision": policy_decision,
            },
        )
        self._audit.record(
            actor=invocation.pid,
            action="tool.call",
            target=invocation.resource,
            decision={
                "ok": False,
                "tool": invocation.handle.name,
                "policy_decision": policy_decision,
                "error": reason,
                "metrics": self._metrics_json(metrics),
            },
        )

    def validate_jit_arguments(
        self,
        handle: ToolHandle,
        arguments: dict[str, Any],
    ) -> None:
        spec = self._extensions.get_tool_spec(handle.tool_id)
        schema = (
            spec.input_schema
            if spec is not None and spec.input_schema
            else {"type": "object"}
        )
        self._validate_jit_value(handle, arguments, schema, "arguments", "input_schema")

    def validate_jit_output(self, handle: ToolHandle, value: Any) -> None:
        spec = self._extensions.get_tool_spec(handle.tool_id)
        schema = spec.output_schema if spec is not None and spec.output_schema else {}
        if schema:
            self._validate_jit_value(handle, value, schema, "output", "output_schema")

    @staticmethod
    def _validate_jit_value(
        handle: ToolHandle,
        value: Any,
        schema: dict[str, Any],
        value_name: str,
        schema_name: str,
    ) -> None:
        try:
            jsonschema_validate(instance=value, schema=schema)
        except JsonSchemaValidationError as exc:
            path = ".".join(str(part) for part in exc.path)
            location = f" at {path}" if path else ""
            raise ValueError(
                f"{value_name} for JIT tool {handle.name!r} do not match "
                f"{schema_name}{location}: {exc.message}"
            ) from exc
        except JsonSchemaSchemaError as exc:
            raise ValueError(
                f"JIT tool {handle.name!r} has invalid {schema_name}: {exc.message}"
            ) from exc

    def has_side_effects(self, handle: ToolHandle) -> bool:
        """Return the registered side-effect policy for one exact handle."""

        implementation = self._registry.implementation(handle.tool_id)
        if implementation is not None:
            return bool(
                implementation.spec(config=self._config).policy.get("side_effects")
            )
        spec = self._extensions.get_tool_spec(handle.tool_id)
        return bool(spec is not None and spec.policy.get("side_effects"))

    def _tool_result_metadata(self, handle: ToolHandle) -> ObjectMetadata:
        implementation = self._registry.implementation(handle.tool_id)
        spec = (
            implementation.spec(config=self._config)
            if implementation is not None
            else self._extensions.get_tool_spec(handle.tool_id)
        )
        tags = set(spec.tags if spec is not None else [])
        externally_sourced = bool(
            tags & {"remote", "provider", "jsonrpc", "mcp", "network", "shell"}
        )
        base = ObjectMetadata(
            title=f"Tool result: {handle.name}",
            tags=["tool_result", handle.name],
            origin=f"tool:{handle.name}",
            trust_level="untrusted" if externally_sourced else "unknown",
            integrity="unknown",
        )
        inherited = ObjectMetadata(**self._data_flow.current_context().labels.to_dict())
        return propagate_object_labels(base, [inherited])

    def _context(self, invocation: _Invocation) -> ToolContext:
        metadata = {
            "tool_id": invocation.handle.tool_id,
            "tool_name": invocation.handle.name,
        }
        metadata.update(dict(invocation.context_metadata or {}))
        metadata["data_flow_context"] = self._data_flow.current_context()
        return ToolContext(
            trace_id=invocation.call_id,
            call_id=invocation.call_id,
            pid=invocation.pid,
            workspace_id=str(self._workspace_root),
            runtime=self._tool_context_host,
            metadata=metadata,
        )

    def _process_status_error(self, pid: str) -> str | None:
        process = self._processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        if process.status in _TERMINAL_PROCESS_STATUSES:
            return (
                f"cannot call tools for terminal process {pid}: "
                f"{process.status.value}"
            )
        if process.status not in _TOOL_CALLABLE_PROCESS_STATUSES:
            return (
                f"cannot call tools for process {pid}: "
                f"status={process.status.value} is not runnable"
            )
        return None

    def _preserve_wait_data_flow_context(self, exc: BaseException) -> None:
        carried = wait_data_flow_context(exc)
        contexts = [self._data_flow.current_context()]
        if carried is not None:
            contexts.append(carried)
        attach_wait_data_flow_context(exc, DataFlowContext.aggregate(contexts))

    def _tool_result_persistence_limit(self) -> int:
        return min(
            self._config.tools.tool_result_payload_hard_limit_bytes,
            self._config.tools.memory_payload_hard_limit_bytes,
        )

    def _error_observation(self, text: str) -> dict[str, Any]:
        return sanitize_for_observability(
            text,
            preview_chars=self._config.tools.tool_observability_preview_chars,
        )

    @staticmethod
    def _metrics_json(metrics: CommandMetrics | None) -> dict[str, Any] | None:
        if metrics is None:
            return None
        return {
            "wall_seconds": metrics.wall_seconds,
            "cpu_seconds": metrics.cpu_seconds,
            "peak_memory_bytes": metrics.peak_memory_bytes,
            "killed": metrics.killed,
            "limit_kind": metrics.limit_kind,
        }

    @staticmethod
    def _elapsed(started_at: float) -> float:
        return max(0.0, time.perf_counter() - started_at)

    @staticmethod
    def _oversize_result_observation() -> dict[str, Any]:
        return {
            "redacted": True,
            "truncated": True,
            "preview": "[tool result omitted after size-limit failure]",
        }

    def _record_result(
        self,
        operation_id: str,
        selected_name: str,
        result: ToolCallResult,
    ) -> None:
        self._operations.link_evidence(
            "tool_call",
            result.call_id,
            "invocation",
            operation_id=operation_id,
            metadata={"tool_id": result.tool_id, "tool": selected_name},
        )
        self._operations.link_evidence(
            "tool_call",
            result.call_id,
            "result",
            operation_id=operation_id,
            metadata={
                "ok": result.ok,
                "result_oid": result.result_handle.oid if result.result_handle else None,
            },
        )

    def _outcome(self, operation: Any, result: ToolCallResult) -> str:
        if result.ok:
            return "succeeded"
        descendants = self._evidence.list_operations(
            root_operation_id=operation.root_operation_id
        )
        if any(
            candidate.outcome.value == "unknown"
            and candidate.operation_id != operation.operation_id
            for candidate in descendants
        ):
            return "unknown"
        error = str(result.error or "").lower()
        denial_markers = (
            "denied",
            "capability",
            "permission",
            "lacks ",
            "not in process tool table",
            "resource limit",
            "exceeded max_",
        )
        return "denied" if any(marker in error for marker in denial_markers) else "failed"


def _trusted_context(
    metadata: dict[str, Any] | None,
    fallback: DataFlowContext,
) -> DataFlowContext:
    selected = (metadata or {}).get("data_flow_context")
    if selected is None:
        return fallback
    if isinstance(selected, DataFlowContext):
        return selected
    if not isinstance(selected, dict):
        raise ValidationError("trusted data_flow_context must be an object")
    try:
        return DataFlowContext.from_dict(selected)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid trusted data_flow_context: {exc}") from exc


__all__ = ["JITSyscallSession", "ToolExecutionService"]
