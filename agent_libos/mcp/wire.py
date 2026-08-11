"""Exact-modern Python SDK wire adapters for Tools, MRTR and Tasks.

These adapters deliberately own no Capability, Human, data-flow, effect or
retry policy.  The Runtime protected-operation facade supplies a governed,
fenced session factory and invokes them only inside the provider phase.  Local
validation happens before the session context is entered; any
``ProviderEffectNotStarted`` raised by that context is allowed to propagate
unchanged so only Runtime-owned evidence can classify a request as not sent.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

from agent_libos.mcp._input import (
    canonical_json_bytes,
    decode_broker_json,
    reject_opaque_secret_reflection,
    sanitize_provider_json,
    sdk_json_mapping,
)
from agent_libos.mcp.app_policy import reject_mcp_app_selector
from agent_libos.mcp.client import (
    McpContinuationSurfaceUnsupported,
    McpSdkV2ResultAdapter,
    McpSdkV2SessionFactory,
    current_mcp_client_binding,
    mcp_transport_spec_from_v3,
)
from agent_libos.mcp.manifest import (
    MCP_TASKS_EXTENSION_ID,
    MCP_V3_PROTOCOL_REVISION,
    McpManifestV3HostPolicy,
    McpServerManifestV3,
    validate_mcp_v3_manifest,
    validate_mcp_v3_tool_arguments,
)
from agent_libos.mcp.types import JsonValue, McpComplete, McpOperationResult
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.models.mcp import McpProtocolMode, McpServerSpec, McpToolSpec
from agent_libos.utils.serde import to_jsonable


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TASK_ID_MAX_BYTES = 8 * 1024
_REQUEST_STATE_MAX_BYTES = 64 * 1024
_TASK_RESULT_ADAPTER = TypeAdapter(dict[str, Any])


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _GetTaskParams(_WireModel):
    taskId: str


class _UpdateTaskParams(_WireModel):
    taskId: str
    inputResponses: dict[str, Any]


class _CancelTaskParams(_WireModel):
    taskId: str


class _GetTaskRequest(_WireModel):
    method: Literal["tasks/get"] = "tasks/get"
    params: _GetTaskParams
    name_param: ClassVar[str | None] = None


class _UpdateTaskRequest(_WireModel):
    method: Literal["tasks/update"] = "tasks/update"
    params: _UpdateTaskParams
    name_param: ClassVar[str | None] = None


class _CancelTaskRequest(_WireModel):
    method: Literal["tasks/cancel"] = "tasks/cancel"
    params: _CancelTaskParams
    name_param: ClassVar[str | None] = None


class McpSdkV3ToolProvider:
    """Manifest-v3-only initial ``tools/call`` provider.

    The logical id is resolved exclusively through the registered Manifest;
    callers cannot inject a raw MCP name, URL, header or transport option.
    Input schema validation is complete before provider/session acquisition.
    """

    mcp_manifest_schema_version: Literal[3] = 3
    mcp_protocol_revision: Literal["2026-07-28"] = "2026-07-28"

    def __init__(
        self,
        session_factory: McpSdkV2SessionFactory,
        *,
        result_adapter: McpSdkV2ResultAdapter | None = None,
        host_policy: McpManifestV3HostPolicy | None = None,
        host_tasks_extension_sha256: str | None = None,
        sensitive_values_resolver: Any | None = None,
    ) -> None:
        if session_factory is None:
            raise TypeError("MCP SDK session factory is required")
        _optional_sha256(host_tasks_extension_sha256, "Tasks extension pin")
        self.session_factory = session_factory
        self.result_adapter = result_adapter or McpSdkV2ResultAdapter()
        self.host_policy = host_policy
        self.host_tasks_extension_sha256 = host_tasks_extension_sha256
        self.sensitive_values_resolver = sensitive_values_resolver

    async def call_tool(
        self,
        manifest: McpServerManifestV3,
        tool_id: str,
        arguments: dict[str, JsonValue],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpOperationResult[dict[str, JsonValue]]:
        selected_deadline = _deadline(deadline)
        validate_mcp_v3_manifest(manifest)
        tool = _manifest_tool(manifest, tool_id)
        detached_arguments = _detached_object(
            arguments,
            label="MCP v3 Tool arguments",
            maximum=manifest.max_request_bytes,
        )
        validate_mcp_v3_tool_arguments(
            tool.input_schema,
            detached_arguments,
            host_policy=self.host_policy,
            deadline=selected_deadline,
        )
        allow_tasks = _require_manifest_tasks_pin(
            manifest,
            self.host_tasks_extension_sha256,
        )
        server = mcp_transport_spec_from_v3(manifest)
        _remaining(selected_deadline, "MCP Tool provider dispatch")
        async with self.session_factory(server, deadline=selected_deadline) as selected:
            session = _exact_modern_tool_session(selected)
            result = await session.call_tool(
                tool.mcp_name,
                detached_arguments,
                read_timeout_seconds=_remaining(
                    selected_deadline, "MCP Tool provider request"
                ),
                allow_input_required=True,
                allow_claimed=allow_tasks,
            )
            operation_sensitive_values = _operation_sensitive_values(
                self.session_factory,
                self.sensitive_values_resolver,
                manifest.server_id,
                sensitive_values,
            )
        _remaining(selected_deadline, "MCP Tool result projection")
        return self.result_adapter.tool_result(
            result,
            server_id=manifest.server_id,
            logical_id=tool.tool_id,
            deadline=selected_deadline,
            sensitive_values=operation_sensitive_values,
        )


class McpSdkV3ContinuationProvider:
    """Dedicated MRTR retry wire path; it has no initial-call API."""

    mcp_manifest_schema_version: Literal[3] = 3
    mcp_protocol_revision: Literal["2026-07-28"] = "2026-07-28"

    def __init__(
        self,
        session_factory: McpSdkV2SessionFactory,
        *,
        result_adapter: McpSdkV2ResultAdapter | None = None,
        host_policy: McpManifestV3HostPolicy | None = None,
        host_tasks_extension_sha256: str | None = None,
        sensitive_values_resolver: Any | None = None,
    ) -> None:
        if session_factory is None:
            raise TypeError("MCP SDK session factory is required")
        _optional_sha256(host_tasks_extension_sha256, "Tasks extension pin")
        self.session_factory = session_factory
        self.result_adapter = result_adapter or McpSdkV2ResultAdapter()
        self.host_policy = host_policy
        self.host_tasks_extension_sha256 = host_tasks_extension_sha256
        self.sensitive_values_resolver = sensitive_values_resolver

    async def continue_tool(
        self,
        server: McpServerSpec,
        mcp_name: str,
        arguments: dict[str, JsonValue],
        input_responses: dict[str, JsonValue],
        request_state: str | None,
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]:
        selected_deadline = _deadline(deadline)
        tool = _server_tool(server, mcp_name)
        detached_arguments = _detached_object(
            arguments,
            label="MCP continuation Tool arguments",
            maximum=server.max_request_bytes,
        )
        validate_mcp_v3_tool_arguments(
            tool.input_schema,
            detached_arguments,
            host_policy=self.host_policy,
            deadline=selected_deadline,
        )
        detached_responses = _detached_object(
            input_responses,
            label="MCP continuation input responses",
            maximum=server.max_request_bytes,
        )
        _request_state(request_state)
        allow_tasks = self._current_manifest_allows_tasks(server)
        _remaining(selected_deadline, "MCP continuation provider dispatch")
        async with self.session_factory(server, deadline=selected_deadline) as selected:
            session = _exact_modern_tool_session(selected)
            result = await session.call_tool(
                tool.mcp_name,
                detached_arguments,
                read_timeout_seconds=_remaining(
                    selected_deadline, "MCP continuation request"
                ),
                input_responses=detached_responses,
                request_state=request_state,
                allow_input_required=True,
                allow_claimed=allow_tasks,
            )
            operation_sensitive_values = _operation_sensitive_values(
                self.session_factory,
                self.sensitive_values_resolver,
                server.server_id,
                (),
            )
        _remaining(selected_deadline, "MCP continuation result validation")
        raw = sdk_json_mapping(result, label="MCP continuation SDK result")
        if raw.get("resultType", "complete") not in {
            "complete",
            "input_required",
            "task",
        }:
            raise ValidationError("MCP continuation resultType is unsupported")
        if raw.get("resultType") == "task" and not allow_tasks:
            raise ValidationError("MCP continuation returned an unpinned Task")
        return _sanitize_polymorphic_result(
            raw,
            sensitive_values=operation_sensitive_values,
            label="MCP continuation SDK result",
        )

    def _current_manifest_allows_tasks(self, server: McpServerSpec) -> bool:
        if self.host_tasks_extension_sha256 is None:
            return False
        binding = current_mcp_client_binding()
        if binding.manifest.server_id != server.server_id:
            raise ValidationError("MCP continuation Tasks binding changed")
        return _require_manifest_tasks_pin(
            binding.manifest,
            self.host_tasks_extension_sha256,
        )

    async def continue_resource(
        self,
        server: McpServerSpec,
        resource_name: str,
        logical_id: str,
        input_responses: dict[str, JsonValue],
        request_state: str | None,
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]:
        """Continue the exact original ``resources/read`` request once.

        ``resource_name`` is the already-expanded, Manifest-resolved remote
        selector.  The adapter never dereferences it locally.  ``logical_id``
        is retained exclusively for the safe public result projection.
        """

        selected_deadline = _deadline(deadline)
        _require_exact_server(server)
        selected_resource = _bounded_wire_text(
            resource_name,
            label="MCP continuation Resource selector",
            maximum=server.max_request_bytes,
        )
        reject_mcp_app_selector(
            selected_resource,
            label="MCP continuation Resource selector",
        )
        selected_logical_id = _bounded_wire_text(
            logical_id,
            label="MCP continuation Resource logical id",
            maximum=server.max_request_bytes,
        )
        detached_responses = _detached_object(
            input_responses,
            label="MCP continuation input responses",
            maximum=server.max_request_bytes,
        )
        _request_state(request_state)
        _bound_continuation_request(
            {
                "resource": selected_resource,
                "inputResponses": detached_responses,
                "requestState": request_state,
            },
            server=server,
            label="MCP Resource continuation request",
        )
        _remaining(selected_deadline, "MCP Resource continuation provider dispatch")
        async with self.session_factory(server, deadline=selected_deadline) as selected:
            session = _exact_modern_session(selected)
            result = await session.read_resource(
                selected_resource,
                input_responses=detached_responses,
                request_state=request_state,
                allow_input_required=True,
            )
            operation_sensitive_values = _operation_sensitive_values(
                self.session_factory,
                self.sensitive_values_resolver,
                server.server_id,
                (),
            )
        _remaining(selected_deadline, "MCP Resource continuation result projection")
        return self._continued_resource_result(
            result,
            server=server,
            logical_id=selected_logical_id,
            deadline=selected_deadline,
            sensitive_values=operation_sensitive_values,
        )

    async def continue_prompt(
        self,
        server: McpServerSpec,
        prompt_name: str,
        logical_id: str,
        arguments: dict[str, str],
        input_responses: dict[str, JsonValue],
        request_state: str | None,
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]:
        """Continue the exact original ``prompts/get`` request once."""

        selected_deadline = _deadline(deadline)
        _require_exact_server(server)
        selected_prompt = _bounded_wire_text(
            prompt_name,
            label="MCP continuation Prompt name",
            maximum=server.max_request_bytes,
        )
        selected_logical_id = _bounded_wire_text(
            logical_id,
            label="MCP continuation Prompt logical id",
            maximum=server.max_request_bytes,
        )
        detached_arguments = _detached_string_object(
            arguments,
            label="MCP continuation Prompt arguments",
            maximum=server.max_request_bytes,
        )
        detached_responses = _detached_object(
            input_responses,
            label="MCP continuation input responses",
            maximum=server.max_request_bytes,
        )
        _request_state(request_state)
        _bound_continuation_request(
            {
                "prompt": selected_prompt,
                "arguments": detached_arguments,
                "inputResponses": detached_responses,
                "requestState": request_state,
            },
            server=server,
            label="MCP Prompt continuation request",
        )
        _remaining(selected_deadline, "MCP Prompt continuation provider dispatch")
        async with self.session_factory(server, deadline=selected_deadline) as selected:
            session = _exact_modern_session(selected)
            result = await session.get_prompt(
                selected_prompt,
                detached_arguments,
                input_responses=detached_responses,
                request_state=request_state,
                allow_input_required=True,
            )
            operation_sensitive_values = _operation_sensitive_values(
                self.session_factory,
                self.sensitive_values_resolver,
                server.server_id,
                (),
            )
        _remaining(selected_deadline, "MCP Prompt continuation result projection")
        return self._continued_prompt_result(
            result,
            server=server,
            logical_id=selected_logical_id,
            deadline=selected_deadline,
            sensitive_values=operation_sensitive_values,
        )

    async def continue_completion(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> Mapping[str, JsonValue]:
        """Fail before dispatch: the official SDK has no Completion MRTR wire."""

        raise McpContinuationSurfaceUnsupported(
            "MCP completion/complete does not support durable input-required continuation"
        )

    def _continued_resource_result(
        self,
        result: Any,
        *,
        server: McpServerSpec,
        logical_id: str,
        deadline: float,
        sensitive_values: tuple[str, ...],
    ) -> Mapping[str, JsonValue]:
        result_type = _sdk_result_type(result)
        if result_type == "input_required":
            return _bounded_polymorphic_result(
                result,
                server=server,
                sensitive_values=sensitive_values,
                label="MCP Resource continuation SDK result",
            )
        if result_type != "complete":
            raise ValidationError(
                "MCP Resource continuation resultType is unsupported"
            )
        projected = self.result_adapter.read_resource_result(
            result,
            server_id=server.server_id,
            logical_id=logical_id,
            deadline=deadline,
            sensitive_values=sensitive_values,
        )
        return _bounded_complete_projection(
            projected,
            server=server,
            label="MCP Resource continuation result",
        )

    def _continued_prompt_result(
        self,
        result: Any,
        *,
        server: McpServerSpec,
        logical_id: str,
        deadline: float,
        sensitive_values: tuple[str, ...],
    ) -> Mapping[str, JsonValue]:
        result_type = _sdk_result_type(result)
        if result_type == "input_required":
            return _bounded_polymorphic_result(
                result,
                server=server,
                sensitive_values=sensitive_values,
                label="MCP Prompt continuation SDK result",
            )
        if result_type != "complete":
            raise ValidationError("MCP Prompt continuation resultType is unsupported")
        projected = self.result_adapter.prompt_result(
            result,
            server_id=server.server_id,
            logical_id=logical_id,
            deadline=deadline,
            sensitive_values=sensitive_values,
        )
        return _bounded_complete_projection(
            projected,
            server=server,
            label="MCP Prompt continuation result",
        )


class McpSdkV3TasksProvider:
    """Fixed-method Tasks extension adapter; intentionally has no list API."""

    mcp_manifest_schema_version: Literal[3] = 3
    mcp_protocol_revision: Literal["2026-07-28"] = "2026-07-28"

    def __init__(
        self,
        session_factory: McpSdkV2SessionFactory,
        *,
        host_tasks_extension_sha256: str,
        sensitive_values_resolver: Any | None = None,
    ) -> None:
        if session_factory is None:
            raise TypeError("MCP SDK session factory is required")
        _require_sha256(host_tasks_extension_sha256, "Tasks extension pin")
        self.session_factory = session_factory
        self.host_tasks_extension_sha256 = host_tasks_extension_sha256
        self.sensitive_values_resolver = sensitive_values_resolver

    async def get_remote_task(
        self,
        server: McpServerSpec,
        remote_task_id: str,
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]:
        request = _GetTaskRequest(
            params=_GetTaskParams(taskId=_remote_task_id(remote_task_id))
        )
        return await self._send(server, request, deadline=deadline)

    async def update_remote_task(
        self,
        server: McpServerSpec,
        remote_task_id: str,
        response: Mapping[str, JsonValue],
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]:
        selected = _detached_object(
            response,
            label="MCP Tasks input responses",
            maximum=server.max_request_bytes,
        )
        if not selected:
            raise ValidationError("MCP tasks/update requires input responses")
        request = _UpdateTaskRequest(
            params=_UpdateTaskParams(
                taskId=_remote_task_id(remote_task_id),
                inputResponses=selected,
            )
        )
        return await self._send(server, request, deadline=deadline)

    async def cancel_remote_task(
        self,
        server: McpServerSpec,
        remote_task_id: str,
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]:
        request = _CancelTaskRequest(
            params=_CancelTaskParams(taskId=_remote_task_id(remote_task_id))
        )
        return await self._send(server, request, deadline=deadline)

    async def _send(
        self,
        server: McpServerSpec,
        request: _GetTaskRequest | _UpdateTaskRequest | _CancelTaskRequest,
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]:
        selected_deadline = _deadline(deadline)
        _require_exact_server(server)
        _remaining(selected_deadline, f"MCP {request.method} provider dispatch")
        async with self.session_factory(server, deadline=selected_deadline) as selected:
            session = _exact_modern_session(selected)
            result = await session.send_request(
                request,
                _TASK_RESULT_ADAPTER,
                request_read_timeout_seconds=_remaining(
                    selected_deadline, f"MCP {request.method} request"
                ),
            )
            operation_sensitive_values = _operation_sensitive_values(
                self.session_factory,
                self.sensitive_values_resolver,
                server.server_id,
                (),
            )
        _remaining(selected_deadline, f"MCP {request.method} result validation")
        raw = sdk_json_mapping(result, label=f"MCP {request.method} SDK result")
        return _sanitize_polymorphic_result(
            raw,
            sensitive_values=operation_sensitive_values,
            label=f"MCP {request.method} SDK result",
        )


def _manifest_tool(manifest: McpServerManifestV3, tool_id: str) -> McpToolSpec:
    if type(tool_id) is not str or not tool_id:
        raise ValidationError("MCP v3 tool_id is invalid")
    selected = next((tool for tool in manifest.tools if tool.tool_id == tool_id), None)
    if selected is None:
        raise NotFound(f"MCP Manifest v3 Tool not found: {tool_id}")
    return selected


def _server_tool(server: McpServerSpec, mcp_name: str) -> McpToolSpec:
    _require_exact_server(server)
    if type(mcp_name) is not str or not mcp_name:
        raise ValidationError("MCP continuation Tool name is invalid")
    matches = [tool for tool in server.tools if tool.mcp_name == mcp_name]
    if len(matches) != 1:
        raise ValidationError("MCP continuation Tool is outside the Manifest allowlist")
    return matches[0]


def _require_exact_server(server: McpServerSpec) -> None:
    if not isinstance(server, McpServerSpec):
        raise TypeError("MCP server spec is required")
    if server.protocol_mode is not McpProtocolMode.REVISION_2026_07_28:
        raise ValidationError("MCP modern provider requires exact protocol 2026-07-28")


def _exact_modern_session(selected: Any) -> Any:
    session = getattr(selected, "session", selected)
    if str(getattr(session, "protocol_version", "")) != MCP_V3_PROTOCOL_REVISION:
        raise ValidationError("MCP modern provider requires exact protocol 2026-07-28")
    return session


def _exact_modern_tool_session(selected: Any) -> Any:
    """Keep Agent libOS's governed Tool adapter instead of unwrapping it.

    Resource/Prompt/Tasks adapters use SDK methods not exposed by the narrow
    Tool adapter and therefore deliberately unwrap the official session.  The
    Tool path must retain the adapter because it owns the audited open-union
    ResultClaim seam and the SDK pre-validation compatibility fix.
    """

    session = (
        selected
        if getattr(selected, "_agent_libos_sdk_v2", False) is True
        else getattr(selected, "session", selected)
    )
    if str(getattr(session, "protocol_version", "")) != MCP_V3_PROTOCOL_REVISION:
        raise ValidationError("MCP modern provider requires exact protocol 2026-07-28")
    return session


def _require_manifest_tasks_pin(
    manifest: McpServerManifestV3,
    host_pin: str | None,
) -> bool:
    extension = manifest.tasks_extension
    if extension is None:
        return False
    if extension.extension_id != MCP_TASKS_EXTENSION_ID:
        raise ValidationError("MCP Manifest uses an unsupported Tasks extension")
    if host_pin is None or extension.spec_sha256 != host_pin:
        raise ValidationError("MCP Tasks extension does not match the Host pin")
    return True


def _detached_object(
    value: Any,
    *,
    label: str,
    maximum: int,
) -> dict[str, JsonValue]:
    if type(value) is not dict:
        raise ValidationError(f"{label} must be a strict JSON object")
    if type(maximum) is not int or maximum <= 0:
        raise ValidationError(f"{label} byte limit is invalid")
    encoded = canonical_json_bytes(value, label=label, max_bytes=maximum)
    return decode_broker_json(encoded, label=label, max_bytes=maximum)


def _detached_string_object(
    value: Any,
    *,
    label: str,
    maximum: int,
) -> dict[str, str]:
    selected = _detached_object(value, label=label, maximum=maximum)
    if any(
        type(key) is not str or type(item) is not str
        for key, item in selected.items()
    ):
        raise ValidationError(f"{label} must contain only string values")
    return cast(dict[str, str], selected)


def _bounded_wire_text(value: Any, *, label: str, maximum: int) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValidationError(f"{label} is invalid")
    return value


def _bound_continuation_request(
    value: dict[str, JsonValue],
    *,
    server: McpServerSpec,
    label: str,
) -> None:
    canonical_json_bytes(value, label=label, max_bytes=server.max_request_bytes)


def _sdk_result_type(result: Any) -> str:
    if isinstance(result, Mapping):
        selected = result.get("resultType", "complete")
    else:
        selected = getattr(
            result,
            "result_type",
            getattr(result, "resultType", "complete"),
        )
    if type(selected) is not str:
        raise ValidationError("MCP continuation resultType is invalid")
    return selected


def _bounded_polymorphic_result(
    result: Any,
    *,
    server: McpServerSpec,
    sensitive_values: tuple[str, ...],
    label: str,
) -> dict[str, JsonValue]:
    raw = sdk_json_mapping(result, label=label)
    selected = _sanitize_polymorphic_result(
        raw,
        sensitive_values=sensitive_values,
        label=label,
    )
    encoded = canonical_json_bytes(
        selected,
        label=label,
        max_bytes=server.max_response_bytes,
    )
    return decode_broker_json(
        encoded,
        label=label,
        max_bytes=server.max_response_bytes,
    )


def _bounded_complete_projection(
    result: McpOperationResult[Any],
    *,
    server: McpServerSpec,
    label: str,
) -> dict[str, JsonValue]:
    if not isinstance(result, McpComplete) or result.value is None:
        raise ValidationError(f"{label} is invalid")
    projected = to_jsonable(result.value)
    if type(projected) is not dict or "resultType" in projected:
        raise ValidationError(f"{label} projection is invalid")
    selected: dict[str, JsonValue] = {"resultType": "complete", **projected}
    encoded = canonical_json_bytes(
        selected,
        label=label,
        max_bytes=server.max_response_bytes,
    )
    return decode_broker_json(
        encoded,
        label=label,
        max_bytes=server.max_response_bytes,
    )


def _remote_task_id(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or len(value.encode("utf-8")) > _TASK_ID_MAX_BYTES
    ):
        raise ValidationError("MCP remote Task identity is invalid")
    return value


def _request_state(value: Any) -> None:
    if value is None:
        return
    if (
        type(value) is not str
        or len(value.encode("utf-8")) > _REQUEST_STATE_MAX_BYTES
    ):
        raise ValidationError("MCP continuation requestState is invalid")


def _sensitive_values(values: Any) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(item) is not str or not item for item in values):
        raise ValidationError("MCP sensitive value snapshot is invalid")
    return values


def _operation_sensitive_values(
    session_factory: Any,
    explicit_resolver: Any,
    server_id: str,
    supplied: tuple[str, ...],
) -> tuple[str, ...]:
    selected = list(_sensitive_values(supplied))
    resolver = explicit_resolver
    if resolver is None:
        resolver = getattr(session_factory, "sensitive_values", None)
    if resolver is not None:
        if not callable(resolver):
            raise ValidationError("MCP sensitive value resolver is invalid")
        resolved = resolver(server_id)
        selected.extend(_sensitive_values(resolved))
    return tuple(dict.fromkeys(selected))


def _sanitize_polymorphic_result(
    raw: dict[str, JsonValue],
    *,
    sensitive_values: tuple[str, ...],
    label: str,
) -> dict[str, JsonValue]:
    result_type = raw.get("resultType", "complete")
    if result_type == "input_required":
        state = raw.get("requestState")
        if state is not None:
            reject_opaque_secret_reflection(
                state,
                sensitive_values=sensitive_values,
                label="MCP requestState",
            )
        input_requests = raw.get("inputRequests")
        if input_requests is not None:
            if type(input_requests) is not dict:
                raise ValidationError("MCP inputRequests must be an object")
            for key in input_requests:
                reject_opaque_secret_reflection(
                    key,
                    sensitive_values=sensitive_values,
                    label="MCP input request key",
                )
    if result_type in {"task", "complete"} and "taskId" in raw:
        reject_opaque_secret_reflection(
            raw["taskId"],
            sensitive_values=sensitive_values,
            label="MCP remote Task identity",
        )
    sanitized = sanitize_provider_json(
        raw,
        sensitive_values=sensitive_values,
        label=label,
    )
    if type(sanitized) is not dict:
        raise ValidationError(f"{label} must be an object")
    return sanitized


def _optional_sha256(value: Any, label: str) -> None:
    if value is not None:
        _require_sha256(value, label)


def _require_sha256(value: Any, label: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"MCP {label} is invalid")


def _deadline(value: Any) -> float:
    if (
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValidationError("MCP absolute deadline is invalid")
    selected = float(value)
    _remaining(selected, "MCP operation")
    return selected


def _remaining(deadline: float, label: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{label} exceeded the absolute deadline")
    return remaining


__all__ = [
    "McpSdkV3ContinuationProvider",
    "McpSdkV3TasksProvider",
    "McpSdkV3ToolProvider",
]
