from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.utils.openai_schema import openai_chat_tool_schema
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
from agent_libos.utils.serde import dumps

InputT = TypeVar("InputT", bound=BaseModel)

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_WAIT_DATA_FLOW_CONTEXT_ATTR = "_agent_libos_wait_data_flow_context"
_DATA_FLOW_WAIT_EXCEPTIONS = (
    HumanApprovalRequired,
    ProcessWaitRequired,
    ProcessMessageWaitRequired,
)
_MODEL_ERROR_MESSAGE_MAX_CHARS = _TOOL_DEFAULTS.tool_observability_preview_chars
_MODEL_ERROR_IDENTIFIER_MAX_CHARS = _TOOL_DEFAULTS.tool_observability_preview_chars
_MODEL_ERROR_DETAIL_LIMIT = max(
    1,
    _TOOL_DEFAULTS.tool_observability_preview_chars // 32,
)
_LOCAL_PATH_PATTERNS = (
    re.compile(r"(?<![:/A-Za-z0-9])/(?:[^/\s,;:]+/)+[^/\s,;:]+"),
    re.compile(r"(?<![A-Za-z0-9])(?:file://)?/(?:Users|home|private|tmp|var|opt)/[^\s,;:]+"),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\(?:[^\\\s,;:]+\\)*[^\\\s,;:]*"),
)
_MODEL_ERROR_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|auth[_-]?token|password|passwd|secret|session[_-]?token|token)\b"
        r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;\"'}]+)"
    ),
)
_MODEL_ERROR_DYNAMIC_METADATA_KEYS = frozenset(
    {
        "call_id",
        "duration_ms",
        "materialization_id",
        "trace_id",
    }
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
    # Optional process-facing projection. ToolExecutionService persists
    # `data`; this excluded field is returned only to the model caller.
    model_data: Any | None = Field(default=None, exclude=True)
    artifacts: list[ToolArtifact] = Field(default_factory=list)
    error: ToolError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(
        cls,
        *,
        content: str = "",
        data: Any | None = None,
        model_data: Any | None = None,
        artifacts: list[ToolArtifact] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=True,
            content=content,
            data=data,
            model_data=model_data,
            artifacts=artifacts or [],
            metadata=metadata or {},
        )

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

    def model_projection(self, *, limit_bytes: int) -> Any:
        """Return a deterministic, bounded payload suitable for an LLM.

        The full ``ToolResult`` remains available to the broker for durable
        evidence.  This projection deliberately excludes per-call telemetry
        and collapses verbose validation failures to bounded, actionable
        summaries.
        """

        if self.ok:
            return _success_model_payload(self)
        error = self.error or ToolError(
            code=ToolErrorCode.EXECUTION_ERROR,
            message=self.content or "Tool execution failed.",
        )
        return bounded_failure_model_projection(
            code=error.code.value,
            error_type=_failure_error_type(error),
            message=error.message or self.content,
            retryable=error.retryable,
            details=error.details,
            metadata=self.metadata,
            limit_bytes=limit_bytes,
        )


def tool_result_content_duplicates_data(content: Any, data: Any) -> bool:
    """Return whether a text representation is exactly the structured data."""

    if not isinstance(content, str):
        return False
    if isinstance(data, str) and content == data:
        return True
    try:
        return json.loads(content) == data
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def model_safe_tool_error_message(
    value: Any,
    *,
    max_chars: int | None = None,
) -> str:
    """Return the standard redacted, single-preview outward error text."""

    return _model_safe_error_message(value, max_chars=max_chars)[0]


def bounded_failure_model_projection(
    *,
    code: str,
    error_type: str,
    message: str,
    retryable: bool,
    details: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    limit_bytes: int,
) -> Any:
    """Build a compact failure carrier without copying raw exception text.

    ``error_hash`` covers the complete pre-projection error so omitted details
    remain correlatable.  At ordinary configured limits every stable field is
    retained.  Extremely small custom limits degrade optional fields first and
    ultimately return ``None`` when no truthful JSON envelope can fit.
    """

    raw_details = dict(details or {})
    raw_error = {
        "code": str(code),
        "type": str(error_type),
        "message": str(message),
        "retryable": bool(retryable),
        "details": raw_details,
    }
    error_hash = hashlib.sha256(_canonical_json_bytes(raw_error)).hexdigest()
    raw_errors = raw_details.get("errors")
    error_items = raw_errors if isinstance(raw_errors, list) else []
    total_errors = len(error_items) if error_items else 1
    projected_errors = [
        _project_validation_error(item)
        for item in error_items[:_MODEL_ERROR_DETAIL_LIMIT]
        if isinstance(item, Mapping)
    ]
    safe_message, message_was_truncated = _model_safe_error_message(message)
    projected_details = _project_stable_error_details(raw_details)
    compact_flow = _project_data_flow_context(metadata)
    error_payload: dict[str, Any] = {
        "code": _bounded_identifier(code, fallback="execution_error"),
        "type": _bounded_identifier(error_type, fallback="ToolError"),
        "retryable": bool(retryable),
        "safe_message": safe_message,
        "total_errors": total_errors,
        "omitted": (
            max(0, total_errors - len(projected_errors))
            if error_items
            else (1 if message_was_truncated else 0)
        ),
        "error_hash": error_hash,
    }
    if projected_errors:
        error_payload["errors"] = projected_errors
    if projected_details:
        error_payload["details"] = projected_details
    projection: dict[str, Any] = {"ok": False, "error": error_payload}
    policy_decision = raw_details.get("policy_decision")
    if isinstance(policy_decision, str) and policy_decision:
        projection["policy_decision"] = _bounded_identifier(
            policy_decision,
            fallback="failure",
        )
    if compact_flow:
        projection["data_flow_context"] = compact_flow

    return _fit_failure_projection(
        projection,
        error_payload=error_payload,
        total_errors=total_errors,
        limit_bytes=limit_bytes,
    )


def _fit_failure_projection(
    projection: dict[str, Any],
    *,
    error_payload: dict[str, Any],
    total_errors: int,
    limit_bytes: int,
) -> Any:
    """Drop optional failure fields until the projection fits its carrier."""

    while _json_size(projection) > limit_bytes and error_payload.get("errors"):
        selected = error_payload["errors"]
        if not isinstance(selected, list):
            break
        selected.pop()
        error_payload["omitted"] = max(
            int(error_payload["omitted"]),
            total_errors - len(selected),
        )
        if not selected:
            error_payload.pop("errors", None)
    for optional_key in ("details",):
        if _json_size(projection) <= limit_bytes:
            break
        error_payload.pop(optional_key, None)
    if _json_size(projection) > limit_bytes:
        projection.pop("data_flow_context", None)
    if _json_size(projection) > limit_bytes:
        projection.pop("policy_decision", None)
    if _json_size(projection) > limit_bytes:
        error_payload["omitted"] = max(int(error_payload["omitted"]), total_errors)
        error_payload["safe_message"] = _fit_text_field(
            projection,
            error_payload,
            "safe_message",
            limit_bytes,
        )
    if _json_size(projection) <= limit_bytes:
        return projection

    # A caller may configure a carrier too small even for the required stable
    # schema.  Preserve the fail-closed signal when possible, otherwise no
    # payload is safer than violating the configured boundary.
    minimal = {"ok": False, "error": {"omitted": True}}
    return minimal if _json_size(minimal) <= limit_bytes else None


def _success_model_payload(result: ToolResult) -> Any:
    has_explicit_projection = result.model_data is not None
    data = result.model_data if has_explicit_projection else result.data
    # `content` is derived from durable `data` by normal tool validation. It
    # must not reintroduce that full record beside an explicit compact view.
    content = "" if has_explicit_projection else result.content
    artifacts = [artifact.model_dump(mode="json") for artifact in result.artifacts]
    if data is None:
        payload: Any = content
    elif not content or tool_result_content_duplicates_data(content, data):
        payload = data
    else:
        payload = {"result": data, "content": content}
    if not artifacts:
        return payload
    if isinstance(payload, dict):
        return {**payload, "artifacts": artifacts}
    return {"result": payload, "artifacts": artifacts}


def _failure_error_type(error: ToolError) -> str:
    selected = error.details.get("error_type")
    if isinstance(selected, str) and selected:
        return selected
    errors = error.details.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, Mapping):
            value = first.get("type")
            if isinstance(value, str) and value:
                return value
    return "ToolError"


def _project_validation_error(value: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    location = value.get("loc")
    if isinstance(location, (list, tuple)):
        projected["loc"] = [
            _bounded_identifier(item, fallback="field") for item in location
        ]
    selected_type = value.get("type")
    if selected_type is not None:
        projected["type"] = _bounded_identifier(selected_type, fallback="validation_error")
    selected_message = value.get("msg") or value.get("message")
    if selected_message is not None:
        projected["safe_message"] = _model_safe_error_message(selected_message)[0]
    return projected


def _project_stable_error_details(details: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    blocked_fragments = (
        "argument",
        "call_id",
        "content",
        "data",
        "duration",
        "input",
        "materialization",
        "path",
        "payload",
        "result",
        "source_ref",
        "stack",
        "stderr",
        "stdout",
        "trace",
    )
    for raw_key in sorted(details, key=str):
        key = str(raw_key)
        normalized = key.strip().lower().replace("-", "_")
        if normalized == "checkpoint_fork_receipt":
            receipt = _project_checkpoint_fork_receipt(details[raw_key])
            if receipt is not None:
                projected[key] = receipt
            continue
        if normalized in {
            "errors",
            "message",
            "policy_decision",
            "safe_message",
        } or any(
            fragment in normalized for fragment in blocked_fragments
        ):
            continue
        value = details[raw_key]
        if isinstance(value, bool) or value is None:
            projected[key] = value
        elif isinstance(value, int):
            projected[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            projected[key] = value
        elif isinstance(value, str) and value:
            if normalized in {"code", "error_type", "correlation_id"}:
                projected[key] = _bounded_identifier(value, fallback="unknown")
            else:
                projected[key] = _model_safe_error_message(value)[0]
    return projected


def _project_checkpoint_fork_identity(
    value: Mapping[Any, Any],
) -> dict[str, str] | None:
    projected: dict[str, str] = {}
    for key in ("checkpoint_id", "fork_root_pid", "status"):
        selected = value.get(key)
        if not isinstance(selected, str) or not selected:
            return None
        projected[key] = _bounded_identifier(selected, fallback="unknown")
    return projected


def _project_checkpoint_fork_outcome(
    value: Mapping[Any, Any],
) -> tuple[bool | None, bool] | None:
    if "main_state_committed" not in value:
        return None
    committed = value.get("main_state_committed")
    if committed is not None and not isinstance(committed, bool):
        return None
    pending = value.get("reconciliation_pending", False)
    if not isinstance(pending, bool):
        return None
    return committed, pending


def _project_checkpoint_fork_failure_phases(
    value: Mapping[Any, Any],
) -> list[str]:
    phases: list[str] = []
    failures = value.get("post_commit_failures")
    if isinstance(failures, (list, tuple)):
        for failure in failures:
            if not isinstance(failure, Mapping):
                continue
            phase = failure.get("phase")
            if isinstance(phase, str) and phase and phase not in phases:
                phases.append(_bounded_identifier(phase, fallback="unknown"))
            if len(phases) >= _MODEL_ERROR_DETAIL_LIMIT:
                break
    diagnostic = value.get("outcome_diagnostic")
    if len(phases) < _MODEL_ERROR_DETAIL_LIMIT and isinstance(diagnostic, Mapping):
        phase = diagnostic.get("phase")
        if isinstance(phase, str) and phase and phase not in phases:
            phases.append(_bounded_identifier(phase, fallback="unknown"))
    return phases


def _project_checkpoint_fork_receipt(value: Any) -> dict[str, Any] | None:
    """Expose only stable retry-safety facts from a fork failure receipt."""

    if not isinstance(value, Mapping):
        return None
    identity = _project_checkpoint_fork_identity(value)
    if identity is None:
        return None
    outcome = _project_checkpoint_fork_outcome(value)
    if outcome is None:
        return None
    committed, pending = outcome
    projected: dict[str, Any] = dict(identity)
    projected["main_state_committed"] = committed
    projected["reconciliation_pending"] = pending
    phases = _project_checkpoint_fork_failure_phases(value)
    if phases:
        projected["failure_phases"] = phases
    return projected


def _project_data_flow_context(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    selected = {
        key: value
        for key, value in metadata.items()
        if key not in _MODEL_ERROR_DYNAMIC_METADATA_KEYS
    }
    flow = selected.get("data_flow_context")
    if isinstance(flow, DataFlowContext):
        if flow == DataFlowContext():
            return None
        flow = flow.to_dict()
    if not isinstance(flow, Mapping):
        return None
    projected: dict[str, Any] = {}
    labels = flow.get("labels")
    if isinstance(labels, Mapping):
        projected["labels"] = dict(labels)
    source_refs = flow.get("source_refs")
    if isinstance(source_refs, (list, tuple)):
        projected["source_ref_count"] = len(source_refs)
    if (
        projected.get("source_ref_count", 0) == 0
        and projected.get("labels") == DataFlowContext().labels.to_dict()
    ):
        return None
    return projected or None


def _model_safe_error_message(
    value: Any,
    *,
    max_chars: int | None = None,
) -> tuple[str, bool]:
    original = str(value).strip()
    selected_lines: list[str] = []
    for line in original.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("traceback (") or lowered.startswith("file \""):
            continue
        if re.match(r"^at\s+\S+\s+\(.+\)$", stripped):
            continue
        selected_lines.append(stripped)
        if len(selected_lines) >= 2:
            break
    safe = " ".join(selected_lines) if selected_lines else "Tool execution failed."
    for pattern in _LOCAL_PATH_PATTERNS:
        safe = pattern.sub("[local-path]", safe)
    for pattern in _MODEL_ERROR_SECRET_PATTERNS:
        safe = pattern.sub("[redacted]", safe)
    selected_max_chars = (
        _MODEL_ERROR_MESSAGE_MAX_CHARS
        if max_chars is None
        else max(0, int(max_chars))
    )
    truncated = len(safe) > selected_max_chars or safe != original
    return safe[:selected_max_chars], truncated


def _bounded_identifier(value: Any, *, fallback: str) -> str:
    selected = str(value).strip()
    if not selected:
        return fallback
    selected = re.sub(r"[^A-Za-z0-9._:-]", "_", selected)
    return selected[:_MODEL_ERROR_IDENTIFIER_MAX_CHARS] or fallback


def _canonical_json_bytes(value: Any) -> bytes:
    return dumps(value).encode("utf-8")


def _json_size(value: Any) -> int:
    return len(_canonical_json_bytes(value))


def _fit_text_field(
    projection: dict[str, Any],
    container: dict[str, Any],
    key: str,
    limit_bytes: int,
) -> str:
    original = str(container.get(key) or "")
    low = 0
    high = len(original)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = original[:middle]
        container[key] = candidate
        if _json_size(projection) <= limit_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    container[key] = best
    return best


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
                details=self._validation_failure_details(exc),
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
                details=self._validation_failure_details(exc),
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

    @staticmethod
    def _validation_failure_details(
        exc: PydanticValidationError,
    ) -> dict[str, Any]:
        # Pydantic context may retain the original, non-serializable ValueError.
        return {
            "error_type": type(exc).__name__,
            "errors": exc.errors(include_input=False, include_context=False),
        }

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

    if _apply_checkpoint_schema_overrides(name, properties, config):
        return

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
        effective_max_bytes = min(
            tools.object_file_hard_limit_bytes,
            tools.filesystem_read_hard_limit_bytes,
        )
        _set_number_bounds(
            properties,
            "max_bytes",
            default=min(tools.object_file_max_bytes, effective_max_bytes),
            maximum=effective_max_bytes,
        )
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


def _apply_checkpoint_schema_overrides(
    name: str,
    properties: dict[str, Any],
    config: AgentLibOSConfig,
) -> bool:
    fields = {
        "list_checkpoints": ("limit", config.checkpoint.list_limit),
        "inspect_checkpoint": ("detail_limit", config.checkpoint.diff_preview_items),
        "diff_checkpoint": ("external_effect_limit", config.checkpoint.diff_preview_items),
    }
    selected = fields.get(name)
    if selected is None:
        return False
    field, value = selected
    _set_number_bounds(properties, field, default=value, maximum=value)
    return True


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
        args.setdefault(
            "max_bytes",
            min(
                tools.object_file_max_bytes,
                tools.object_file_hard_limit_bytes,
                tools.filesystem_read_hard_limit_bytes,
            ),
        )
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
    elif name == "list_checkpoints":
        args.setdefault("limit", config.checkpoint.list_limit)
    elif name == "inspect_checkpoint":
        args.setdefault("detail_limit", config.checkpoint.diff_preview_items)
    elif name == "diff_checkpoint":
        args.setdefault(
            "external_effect_limit",
            config.checkpoint.diff_preview_items,
        )
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
