from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.llm.openai_schema import openai_chat_tool_schema
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProcessError,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    ValidationError as LibOSValidationError,
)
from agent_libos.models import DataFlowContext, ToolSpec
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.utils.public_errors import provider_error_envelope

InputT = TypeVar("InputT", bound=BaseModel)

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_WAIT_DATA_FLOW_CONTEXT_ATTR = "_agent_libos_wait_data_flow_context"
_DATA_FLOW_WAIT_EXCEPTIONS = (
    HumanApprovalRequired,
    ProcessWaitRequired,
    ProcessMessageWaitRequired,
)


def attach_wait_data_flow_context(
    exc: BaseException,
    context: DataFlowContext,
) -> None:
    """Attach trusted flow state to a supported wait without changing its text."""

    if isinstance(exc, _DATA_FLOW_WAIT_EXCEPTIONS):
        setattr(exc, _WAIT_DATA_FLOW_CONTEXT_ATTR, context.to_dict())


def wait_data_flow_context(exc: BaseException) -> DataFlowContext | None:
    """Read the Host-private flow carrier from a supported wait exception."""

    if not isinstance(exc, _DATA_FLOW_WAIT_EXCEPTIONS):
        return None
    serialized = getattr(exc, _WAIT_DATA_FLOW_CONTEXT_ATTR, None)
    if not isinstance(serialized, dict):
        return None
    return DataFlowContext.from_dict(serialized)


class ToolErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    PERMISSION_DENIED = "permission_denied"
    CONFIRMATION_REQUIRED = "confirmation_required"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    TRANSIENT_ERROR = "transient_error"
    EXECUTION_ERROR = "execution_error"
    UNSUPPORTED = "unsupported"


class ToolPolicy(BaseModel):
    side_effects: bool = False
    idempotent: bool = True
    declared_confirmation_required: bool = False
    declared_permissions: set[str] = Field(default_factory=set)
    timeout_s: float | None = _TOOL_DEFAULTS.default_timeout_s
    max_retries: int = 0


class ToolContext(BaseModel):
    trace_id: str
    call_id: str
    pid: str
    workspace_id: str | None = None
    runtime: Any | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class ToolArtifact(BaseModel):
    kind: str
    uri: str
    name: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    code: ToolErrorCode
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool
    content: str = ""
    data: Any | None = None
    artifacts: list[ToolArtifact] = Field(default_factory=list)
    error: ToolError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(
        cls,
        *,
        content: str = "",
        data: Any | None = None,
        artifacts: list[ToolArtifact] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(ok=True, content=content, data=data, artifacts=artifacts or [], metadata=metadata or {})

    @classmethod
    def failure(
        cls,
        *,
        code: ToolErrorCode,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=False,
            content=message,
            error=ToolError(code=code, message=message, retryable=retryable, details=details or {}),
            metadata=metadata or {},
        )


def _merge_result_data_flow_context(
    raw_result: Any,
    returned_context: DataFlowContext,
) -> DataFlowContext:
    """Conservatively combine a tool-owned carrier with worker-observed flow."""

    if not isinstance(raw_result, ToolResult):
        return returned_context
    serialized = raw_result.metadata.get("data_flow_context")
    if not isinstance(serialized, Mapping):
        return returned_context
    explicit_context = DataFlowContext.from_dict(dict(serialized))
    return DataFlowContext.aggregate((returned_context, explicit_context))


class ToolExecutionError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode = ToolErrorCode.EXECUTION_ERROR,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


class BaseAgentTool(ABC, Generic[InputT]):
    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[type[InputT]]

    output_schema: ClassVar[type[BaseModel] | None] = None
    version: ClassVar[str] = _TOOL_DEFAULTS.version
    policy: ClassVar[ToolPolicy] = ToolPolicy()
    tags: ClassVar[list[str]] = []
    metadata: ClassVar[dict[str, Any]] = {}
    expose_internal_errors: ClassVar[bool] = False
    enforce_timeout: ClassVar[bool] = True

    def spec(self, *, config: AgentLibOSConfig | None = None) -> ToolSpec:
        self._validate_contract()
        selected_config = config or DEFAULT_CONFIG
        policy = self.policy.model_dump()
        _apply_runtime_policy_overrides(policy, selected_config)
        input_schema = self.args_schema.model_json_schema()
        _strip_internal_schema_fields(input_schema)
        _apply_runtime_schema_overrides(self.name, input_schema, selected_config)
        return ToolSpec(
            name=self.name,
            description=self.description,
            version=self.version,
            input_schema=input_schema,
            output_schema=self.output_schema.model_json_schema() if self.output_schema is not None else {},
            policy=policy,
            tags=list(self.tags),
            metadata=dict(self.metadata),
            required_capabilities=[],
            side_effects=sorted(self.policy.declared_permissions) if self.policy.side_effects else [],
        )

    def to_openai_chat_tool(self, *, config: AgentLibOSConfig | None = None) -> dict[str, Any]:
        spec = self.spec(config=config)
        return openai_chat_tool_schema(spec.name, spec.description, spec.input_schema)

    def to_mcp_tool(self, *, config: AgentLibOSConfig | None = None) -> dict[str, Any]:
        spec = self.spec(config=config)
        return {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": spec.input_schema,
            "_meta": {
                "version": spec.version,
                "tags": spec.tags,
                "policy": spec.policy,
            },
        }

    async def ainvoke(self, raw_args: Mapping[str, Any] | str | InputT, ctx: ToolContext) -> ToolResult:
        started_at = time.perf_counter()
        try:
            args = self.parse_args(self._raw_args_with_runtime_defaults(raw_args, ctx))
        except PydanticValidationError as exc:
            return ToolResult.failure(
                code=ToolErrorCode.VALIDATION_ERROR,
                message=f"Invalid arguments for tool `{self.name}`.",
                details={"errors": exc.errors(include_input=False)},
                metadata=self._base_metadata(ctx, started_at),
            )
        except Exception as exc:
            return ToolResult.failure(
                code=ToolErrorCode.VALIDATION_ERROR,
                message=f"Failed to parse arguments for tool `{self.name}`.",
                details={"error_type": type(exc).__name__},
                metadata=self._base_metadata(ctx, started_at),
            )

        try:
            runtime = ctx.runtime
            manager = getattr(runtime, "data_flow", None) if runtime is not None else None
            cancelled_context = (
                [manager.current_context()] if manager is not None else []
            )

            async def execute_with_flow() -> Any:
                if manager is None:
                    return await self.execute(args, ctx)
                try:
                    raw_result = await self.execute(args, ctx)
                    returned_context = manager.current_context()
                    cancelled_context[:] = [returned_context]
                    return True, raw_result, returned_context
                except asyncio.CancelledError:
                    # ``asyncio.wait_for`` cancels this child Task on a real
                    # deadline. ContextVar mutations are task-local, so export
                    # the post-read context through a parent-visible holder
                    # before the cancellation destroys the child context.
                    cancelled_context[:] = [manager.current_context()]
                    raise
                except BaseException as exc:
                    # ``asyncio.wait_for`` may execute the tool in a child
                    # Task, whose ContextVar mutations do not flow back to the
                    # caller. Return the trusted post-call context alongside
                    # both successful and failed outcomes.
                    returned_context = manager.current_context()
                    cancelled_context[:] = [returned_context]
                    return False, exc, returned_context

            if self.policy.timeout_s is None or not self.enforce_timeout:
                executed = await execute_with_flow()
            else:
                executed = await asyncio.wait_for(
                    execute_with_flow(), timeout=self.policy.timeout_s
                )
            if manager is None:
                raw_result = executed
            else:
                succeeded, raw_result, returned_context = executed
                manager.observe_ingress(returned_context)
                if not succeeded:
                    ctx.metadata["_agent_libos_returned_data_flow_context"] = (
                        returned_context.to_dict()
                    )
                    attach_wait_data_flow_context(raw_result, returned_context)
                    raise raw_result
            result = self._normalize_result(raw_result)
            if manager is not None:
                result.metadata.setdefault(
                    "data_flow_context", returned_context.to_dict()
                )
            result.metadata.update(self._base_metadata(ctx, started_at))
            return result
        except asyncio.TimeoutError:
            if manager is not None and cancelled_context:
                returned_context = cancelled_context[-1]
                manager.observe_ingress(returned_context)
                ctx.metadata["_agent_libos_returned_data_flow_context"] = (
                    returned_context.to_dict()
                )
            return ToolResult.failure(
                code=ToolErrorCode.TIMEOUT,
                message=f"Tool `{self.name}` timed out.",
                retryable=True,
                metadata=self._base_metadata(ctx, started_at),
            )
        except PydanticValidationError as exc:
            return ToolResult.failure(
                code=ToolErrorCode.VALIDATION_ERROR,
                message=f"Invalid output for tool `{self.name}`.",
                details={"errors": exc.errors(include_input=False)},
                metadata=self._base_metadata(ctx, started_at),
            )
        except ToolExecutionError as exc:
            return ToolResult.failure(
                code=exc.code,
                message=str(exc),
                retryable=exc.retryable,
                details=exc.details,
                metadata=self._base_metadata(ctx, started_at),
            )
        except HumanApprovalRequired:
            raise
        except ProcessWaitRequired:
            raise
        except ProcessMessageWaitRequired:
            raise
        except CapabilityDenied as exc:
            return ToolResult.failure(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=str(exc),
                details={"error_type": type(exc).__name__},
                metadata=self._base_metadata(ctx, started_at),
            )
        except NotFound as exc:
            return ToolResult.failure(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=str(exc),
                details={"error_type": type(exc).__name__},
                metadata=self._base_metadata(ctx, started_at),
            )
        except ProcessError as exc:
            return ToolResult.failure(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=str(exc),
                details={"error_type": type(exc).__name__},
                metadata=self._base_metadata(ctx, started_at),
            )
        except LibOSValidationError as exc:
            return ToolResult.failure(
                code=ToolErrorCode.VALIDATION_ERROR,
                message=str(exc),
                details={"error_type": type(exc).__name__},
                metadata=self._base_metadata(ctx, started_at),
            )
        except Exception as exc:
            return self._unexpected_failure_result(exc, ctx, started_at)

    def _unexpected_failure_result(
        self,
        exc: Exception,
        ctx: ToolContext,
        started_at: float,
    ) -> ToolResult:
        public_error = provider_error_envelope(exc)
        if public_error is not None:
            return ToolResult.failure(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=public_error["message"],
                details={
                    key: public_error[key]
                    for key in ("code", "error_type", "correlation_id")
                },
                metadata=self._base_metadata(ctx, started_at),
            )
        details: dict[str, Any] = {"error_type": type(exc).__name__}
        if self.expose_internal_errors:
            details["message"] = str(exc)
        return ToolResult.failure(
            code=ToolErrorCode.EXECUTION_ERROR,
            message=f"Tool `{self.name}` failed during execution.",
            details=details,
            metadata=self._base_metadata(ctx, started_at),
        )

    def invoke(self, raw_args: Mapping[str, Any] | str | InputT, ctx: ToolContext) -> ToolResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(raw_args, ctx))
        raise RuntimeError("Cannot call invoke() inside a running event loop. Use `await ainvoke(...)`.")

    def parse_args(self, raw_args: Mapping[str, Any] | str | InputT) -> InputT:
        if isinstance(raw_args, self.args_schema):
            return raw_args
        if isinstance(raw_args, str):
            return self.args_schema.model_validate_json(raw_args)
        if isinstance(raw_args, Mapping):
            return self.args_schema.model_validate(dict(raw_args))
        raise TypeError(f"Tool arguments must be {self.args_schema.__name__}, dict, or JSON string.")

    def _raw_args_with_runtime_defaults(self, raw_args: Mapping[str, Any] | str | InputT, ctx: ToolContext) -> Mapping[str, Any] | str | InputT:
        if isinstance(raw_args, self.args_schema):
            return raw_args
        config = getattr(ctx.runtime, "config", DEFAULT_CONFIG)
        if isinstance(raw_args, str):
            try:
                decoded = json.loads(raw_args)
            except json.JSONDecodeError:
                return raw_args
            if not isinstance(decoded, dict):
                return raw_args
            return _apply_runtime_arg_defaults(self.name, decoded, config)
        if isinstance(raw_args, Mapping):
            return _apply_runtime_arg_defaults(self.name, dict(raw_args), config)
        return raw_args

    def _normalize_result(self, raw_result: Any) -> ToolResult:
        if isinstance(raw_result, ToolResult):
            if raw_result.ok and self.output_schema is not None and raw_result.data is not None:
                validated = self.output_schema.model_validate(raw_result.data)
                raw_result.data = validated.model_dump()
                raw_result.content = validated.model_dump_json()
            return raw_result
        if self.output_schema is not None:
            validated = self.output_schema.model_validate(
                raw_result.model_dump() if isinstance(raw_result, BaseModel) else raw_result
            )
            return ToolResult.success(content=validated.model_dump_json(), data=validated.model_dump())
        if isinstance(raw_result, BaseModel):
            return ToolResult.success(content=raw_result.model_dump_json(), data=raw_result.model_dump())
        if isinstance(raw_result, (dict, list)):
            return ToolResult.success(content=json.dumps(raw_result, ensure_ascii=False, default=str), data=raw_result)
        if raw_result is None:
            return ToolResult.success()
        return ToolResult.success(content=str(raw_result), data=raw_result)

    def _validate_contract(self) -> None:
        if not getattr(self, "name", None):
            raise TypeError(f"{self.__class__.__name__} must define non-empty `name`.")
        if not getattr(self, "description", None):
            raise TypeError(f"{self.__class__.__name__} must define non-empty `description`.")
        if not getattr(self, "args_schema", None):
            raise TypeError(f"{self.__class__.__name__} must define `args_schema`.")
        if not issubclass(self.args_schema, BaseModel):
            raise TypeError("`args_schema` must be a Pydantic BaseModel subclass.")
        if self.output_schema is not None and not issubclass(self.output_schema, BaseModel):
            raise TypeError("`output_schema` must be a Pydantic BaseModel subclass.")

    def _base_metadata(self, ctx: ToolContext, started_at: float) -> dict[str, Any]:
        metadata = {
            "tool_name": self.name,
            "tool_version": self.version,
            "trace_id": ctx.trace_id,
            "call_id": ctx.call_id,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
        }
        returned_context = ctx.metadata.get(
            "_agent_libos_returned_data_flow_context"
        )
        if isinstance(returned_context, dict):
            metadata["data_flow_context"] = returned_context
        return metadata

    @abstractmethod
    async def execute(self, args: InputT, ctx: ToolContext) -> Any:
        raise NotImplementedError


class SyncAgentTool(BaseAgentTool[InputT], ABC):
    # Python threads cannot be killed safely after asyncio.wait_for() times out.
    # Sync tools therefore rely on their underlying primitive/provider for hard
    # deadlines instead of returning while a background thread may still mutate
    # runtime state.
    enforce_timeout: ClassVar[bool] = False

    async def execute(self, args: InputT, ctx: ToolContext) -> Any:
        runtime = ctx.runtime
        manager = getattr(runtime, "data_flow", None) if runtime is not None else None
        blocking_work = getattr(runtime, "blocking_work", None) if runtime is not None else None
        if manager is None:
            if blocking_work is not None:
                return await blocking_work.run(self.run, args, ctx)
            return await run_blocking_once(self.run, args, ctx)

        # Worker ContextVars do not copy mutations back into the event-loop
        # task. Return the trusted post-call flow explicitly so ToolBroker can
        # label the result Object with every source observed by synchronous
        # primitives.
        def run_with_flow() -> tuple[bool, Any, Any]:
            try:
                return True, self.run(args, ctx), manager.current_context()
            except BaseException as exc:
                # Exceptions are part of the tool output surface too. Capture
                # the worker's post-call ContextVar before it is discarded so
                # an error derived from a labeled source cannot become an
                # unlabeled model-visible result.
                return False, exc, manager.current_context()

        if blocking_work is not None:
            succeeded, raw_result, returned_context = await blocking_work.run(run_with_flow)
        else:
            succeeded, raw_result, returned_context = await run_blocking_once(run_with_flow)
        returned_context = _merge_result_data_flow_context(
            raw_result,
            returned_context,
        )
        if not succeeded:
            manager.observe_ingress(returned_context)
            ctx.metadata["_agent_libos_returned_data_flow_context"] = (
                returned_context.to_dict()
            )
            attach_wait_data_flow_context(raw_result, returned_context)
            raise raw_result
        # Merge the worker context before output validation. Pydantic failures
        # are model-visible tool results too and must retain every source the
        # synchronous implementation observed.
        manager.observe_ingress(returned_context)
        ctx.metadata["_agent_libos_returned_data_flow_context"] = (
            returned_context.to_dict()
        )
        result = self._normalize_result(raw_result)
        result.metadata["data_flow_context"] = returned_context.to_dict()
        return result

    @abstractmethod
    def run(self, args: InputT, ctx: ToolContext) -> Any:
        raise NotImplementedError


def _apply_runtime_policy_overrides(policy: dict[str, Any], config: AgentLibOSConfig) -> None:
    timeout = policy.get("timeout_s")
    defaults = DEFAULT_CONFIG.tools
    runtime = config.tools
    if timeout == defaults.default_timeout_s:
        policy["timeout_s"] = runtime.default_timeout_s
    elif timeout == defaults.standard_timeout_s:
        policy["timeout_s"] = runtime.standard_timeout_s
    elif timeout == defaults.interactive_timeout_s:
        policy["timeout_s"] = runtime.interactive_timeout_s
    elif timeout == defaults.sleep_tool_timeout_s:
        policy["timeout_s"] = runtime.sleep_tool_timeout_s


def _apply_runtime_schema_overrides(name: str, schema: dict[str, Any], config: AgentLibOSConfig) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    tools = config.tools
    shell = config.shell
    runtime = config.runtime

    if name in {"read_text_file", "read_file_bytes"}:
        _set_property_default(properties, "encoding", tools.default_text_encoding)
        _set_number_bounds(
            properties,
            "max_bytes",
            default=tools.filesystem_read_max_bytes,
            maximum=tools.filesystem_read_hard_limit_bytes,
        )
    elif name == "read_directory":
        _set_number_bounds(properties, "limit", default=tools.directory_entry_limit, maximum=tools.directory_entry_hard_limit)
    elif name == "create_object_from_file":
        _set_property_default(properties, "encoding", tools.default_text_encoding)
        _set_number_bounds(properties, "max_bytes", default=tools.object_file_max_bytes, maximum=tools.object_file_hard_limit_bytes)
    elif name == "write_object_to_file":
        _set_property_default(properties, "encoding", tools.default_text_encoding)
    elif name == "read_memory_object":
        _set_number_bounds(
            properties,
            "max_payload_chars",
            default=tools.memory_payload_chars,
            maximum=tools.memory_payload_hard_limit_chars,
        )
    elif name == "list_memory_namespace":
        _set_number_bounds(
            properties,
            "limit",
            maximum=config.memory.query_limit,
        )
    elif name in {"read_process_messages", "receive_process_messages"}:
        _set_number_bounds(properties, "limit", default=tools.message_read_limit, maximum=tools.message_read_hard_limit)
    elif name == "run_shell_command":
        _set_number_bounds(
            properties,
            "timeout_s",
            default=tools.shell_timeout_s,
            maximum=shell.timeout_hard_limit_s,
            exclusive_minimum=0,
        )
        _set_number_bounds(properties, "max_stdout_chars", default=shell.max_stdout_chars, maximum=shell.stdout_hard_limit_chars)
        _set_number_bounds(properties, "max_stderr_chars", default=shell.max_stderr_chars, maximum=shell.stderr_hard_limit_chars)
    elif name == "sleep":
        _set_number_bounds(properties, "seconds", maximum=tools.max_sleep_seconds)
    elif name == "wait_object_task":
        _set_number_bounds(properties, "timeout_s", maximum=tools.max_sleep_seconds)
    elif name == "get_current_time":
        _set_property_default(properties, "timezone", tools.clock_timezone)
    elif name == "ask_human":
        _set_property_default(properties, "human", runtime.default_human)
    elif name == "human_output":
        _set_property_default(properties, "channel", runtime.terminal_channel)
    elif name == "request_permission":
        _set_property_default(properties, "human", runtime.default_human)
    elif name == "list_jsonrpc_endpoints":
        _set_number_bounds(
            properties,
            "limit",
            maximum=config.jsonrpc.list_limit,
        )
    elif name == "list_mcp_servers":
        _set_number_bounds(
            properties,
            "limit",
            maximum=config.mcp.list_limit,
        )


def _apply_runtime_arg_defaults(name: str, args: dict[str, Any], config: AgentLibOSConfig) -> dict[str, Any]:
    tools = config.tools
    shell = config.shell
    runtime = config.runtime

    if name in {"read_text_file", "read_file_bytes"}:
        args.setdefault("encoding", tools.default_text_encoding)
        args.setdefault("max_bytes", tools.filesystem_read_max_bytes)
    elif name == "read_directory":
        args.setdefault("limit", tools.directory_entry_limit)
    elif name == "create_object_from_file":
        args.setdefault("encoding", tools.default_text_encoding)
        args.setdefault("max_bytes", tools.object_file_max_bytes)
    elif name == "write_object_to_file":
        args.setdefault("encoding", tools.default_text_encoding)
    elif name == "read_memory_object":
        args.setdefault("max_payload_chars", tools.memory_payload_chars)
    elif name in {"read_process_messages", "receive_process_messages"}:
        args.setdefault("limit", tools.message_read_limit)
    elif name == "run_shell_command":
        args.setdefault("timeout_s", tools.shell_timeout_s)
        args.setdefault("max_stdout_chars", shell.max_stdout_chars)
        args.setdefault("max_stderr_chars", shell.max_stderr_chars)
    elif name == "get_current_time":
        args.setdefault("timezone", tools.clock_timezone)
    elif name == "ask_human":
        args.setdefault("human", runtime.default_human)
    elif name == "human_output":
        args.setdefault("channel", runtime.terminal_channel)
    elif name == "request_permission":
        args.setdefault("human", runtime.default_human)
    return args


def _set_property_default(properties: dict[str, Any], field: str, value: Any) -> None:
    prop = properties.get(field)
    if isinstance(prop, dict):
        prop["default"] = value


def _strip_internal_schema_fields(schema: dict[str, Any]) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    required = schema.get("required")
    for name, prop in list(properties.items()):
        if isinstance(prop, dict) and prop.get("x-agent-libos-internal"):
            properties.pop(name, None)
            if isinstance(required, list):
                while name in required:
                    required.remove(name)


def _set_number_bounds(
    properties: dict[str, Any],
    field: str,
    *,
    default: int | float | None = None,
    maximum: int | float | None = None,
    exclusive_minimum: int | float | None = None,
) -> None:
    prop = properties.get(field)
    if not isinstance(prop, dict):
        return
    if default is not None:
        prop["default"] = default
    if maximum is not None:
        prop["maximum"] = maximum
    if exclusive_minimum is not None:
        prop["exclusiveMinimum"] = exclusive_minimum
