from __future__ import annotations

import hashlib
import ipaddress
import math
import os
import re
import socket
import threading
import time
from functools import partial
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.validators import validator_for as jsonschema_validator_for

from agent_libos.capability.manager import CapabilityManager
from agent_libos.capability.rules import AUTHORITY_RULES_KEY
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.human.manager import HumanObjectManager
from agent_libos.models import (
    CapabilityEffect,
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    DataSink,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    McpCallResult,
    McpCallStatus,
    McpHeaderSpec,
    McpHttpTransportSpec,
    McpProviderCallResult,
    McpProviderTool,
    McpServerSpec,
    McpStdioTransportSpec,
    McpToolListResult,
    McpToolSpec,
    ResourceUsage,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProviderHostError,
    ValidationError,
)
from agent_libos.ports import AuditPort, EventPort
from agent_libos.storage import UnitOfWork
from agent_libos.substrate import (
    ExecutableSnapshot,
    executable_content_sha256,
    McpProvider,
    ProviderEffectNotStarted,
    snapshot_executable,
)
from agent_libos.sdk import (
    ProviderEffectNotStartedResult,
    ProviderRegistryBinding,
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProtectedOperationSDK,
    ProviderPhase,
    ResourceSettlement,
)
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.public_errors import provider_error_envelope_from_mapping
from agent_libos.utils.serde import dumps, to_jsonable
from agent_libos.utils.yaml_loader import load_yaml_mapping

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*$")
_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_FORBIDDEN_HEADERS = {"connection", "content-length", "host", "transfer-encoding", "upgrade"}
_FORBIDDEN_MCP_HOSTS = {"metadata.google.internal"}
_LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}
_CALL_RIGHTS = {CapabilityRight.READ.value, CapabilityRight.WRITE.value, CapabilityRight.EXECUTE.value}
_ALLOWED_HEADER_PREFIXES = {"", "Bearer ", "Token ", "Basic "}
_ALLOWED_HEADER_SUFFIXES = {""}
_TRANSPORTS = {"stdio", "streamable_http"}
_MCP_PLATFORM_ENV_KEYS = ("SYSTEMROOT", "WINDIR") if os.name == "nt" else ()
_STDIO_EXECUTABLE_IDENTITY_UNSET = object()
_PROVIDER_RESULT_RETURNED_ATTR = "_agent_libos_provider_result_returned"
_SERVER_FIELDS = {
    "schema_version",
    "server_id",
    "transport",
    "stdio",
    "http",
    "tools",
    "timeout_s",
    "max_request_bytes",
    "max_response_bytes",
    "metadata",
}
_STDIO_FIELDS = {"command", "args", "env", "cwd"}
_HTTP_FIELDS = {"url", "headers"}
_TOOL_FIELDS = {
    "tool_id",
    "mcp_name",
    "right",
    "rollback_class",
    "rollback_status",
    "state_mutation",
    "information_flow",
    "input_schema",
    "metadata",
}
_HEADER_FIELDS = {"env", "prefix", "suffix"}


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    *,
    context: str,
) -> None:
    unknown = sorted(str(field) for field in value if field not in allowed)
    if unknown:
        raise ValidationError(f"unknown {context} fields: {unknown}")


def _mark_provider_result_returned(error: ProviderHostError) -> ProviderHostError:
    """Mark a public provider error as occurring after a result was returned."""

    object.__setattr__(error, _PROVIDER_RESULT_RETURNED_ATTR, True)
    return error


def _provider_result_was_returned(error: BaseException) -> bool:
    try:
        attributes = object.__getattribute__(error, "__dict__")
    except Exception:
        return False
    return attributes.get(_PROVIDER_RESULT_RETURNED_ATTR) is True


class _McpLiveToolValidationError(ValidationError):
    def __init__(self, message: str, result: McpToolListResult) -> None:
        super().__init__(message)
        self.result = result


def _strict_provider_json_value(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
) -> Any:
    """Detach an exact JSON tree returned by a Host provider.

    Provider result objects are outside the runtime trust boundary.  In
    particular, attribute access, container iteration, and nested values may
    execute provider-owned code, so normalize the whole tree before any later
    evidence, accounting, or model-visible projection reads it.
    """

    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise TypeError(f"provider JSON contains a non-finite number at {path}")
        return value
    if value_type is dict:
        identity = id(value)
        if identity in active_containers:
            raise TypeError(f"provider JSON contains a cycle at {path}")
        active_containers.add(identity)
        try:
            selected: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(
                        f"provider JSON contains a non-string key at {path}"
                    )
                selected[key] = _strict_provider_json_value(
                    item,
                    path=f"{path}[{key!r}]",
                    active_containers=active_containers,
                )
            return selected
        finally:
            active_containers.remove(identity)
    if value_type is list:
        identity = id(value)
        if identity in active_containers:
            raise TypeError(f"provider JSON contains a cycle at {path}")
        active_containers.add(identity)
        try:
            return [
                _strict_provider_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)
    raise TypeError(f"provider JSON contains a non-JSON value at {path}")


class McpPrimitive:
    """Capability-controlled MCP client primitive for registered external servers."""

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        capabilities: CapabilityManager,
        audit: AuditPort,
        events: EventPort,
        *,
        protected_operations: ProtectedOperationSDK,
        human: HumanObjectManager | None,
        provider: McpProvider,
        config: AgentLibOSConfig | None = None,
        resources: Any | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.unit_of_work = unit_of_work
        self.extensions = unit_of_work.extensions
        self.authority = unit_of_work.authority
        self.processes = unit_of_work.processes
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.protected_operations = protected_operations
        self.human = human
        self.provider = provider
        self.resources = resources
        self._registry_phase_lock = threading.RLock()

    def server_resource(self, server_id: str) -> str:
        return f"mcp_server:{server_id}"

    def tool_resource(self, server_id: str, tool_id: str) -> str:
        return f"mcp:{server_id}:{tool_id}"

    @staticmethod
    def stdio_resource_for_argv(
        command: str,
        args: list[str] | tuple[str, ...],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        payload = dumps(
            {
                "command": command,
                "args": list(args),
                "env": sorted((env or {}).items()),
                "cwd": McpPrimitive._canonical_stdio_cwd(cwd),
            }
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"mcp_stdio:{digest}"

    @classmethod
    def stdio_resource_for_server(cls, server: McpServerSpec) -> str | None:
        if server.transport != "stdio" or server.stdio is None:
            return None
        return cls.stdio_resource_for_argv(
            server.stdio.command,
            list(server.stdio.args),
            env=dict(server.stdio.env),
            cwd=server.stdio.cwd,
        )

    @staticmethod
    def _canonical_stdio_cwd(cwd: str | None) -> str | None:
        if cwd is None:
            return None
        raw = cwd.replace("\\", "/").strip()
        parts: list[str] = []
        for part in raw.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        if not parts:
            return None
        return "/".join(parts)

    def register_server(
        self,
        server: McpServerSpec | dict[str, Any],
        *,
        actor: str = "runtime",
        replace: bool = False,
        require_capability: bool = True,
        source: str | None = None,
    ) -> dict[str, Any]:
        spec = self._coerce_server(server)
        authority_decisions: list[Any] = []
        if require_capability:
            required_right = CapabilityRight.ADMIN if replace else CapabilityRight.WRITE
            authority_decisions.append(
                self.capabilities.require(
                    actor,
                    self.server_resource(spec.server_id),
                    required_right,
                    consume=False,
                )
            )
            authority_decisions.extend(self._require_stdio_process_spawn(actor, spec, consume=False))
        now = utc_now()
        # Registry mutation, composite one-shot authority settlement, and
        # evidence are one store transaction. Local validation or a sink
        # failure rolls every reservation back to its pre-call state.
        with self._registry_phase_lock, self.capabilities.authority_transaction(
            authority_decisions,
            actor=actor,
            operation="MCP server register",
        ):
            existing = self.extensions.get_mcp_server(spec.server_id)
            if existing is not None and not replace:
                raise ValidationError(f"MCP server already exists: {spec.server_id}")
            self.extensions.upsert_mcp_server(spec, registered_by=actor, created_at=now)
            if existing is not None:
                self._disable_replaced_server_tool_capabilities(spec.server_id, actor=actor)
            self.events.emit(
                EventType.EXTERNAL_WRITE,
                source=actor,
                target=self.server_resource(spec.server_id),
                payload={"adapter": "mcp", "operation": "server_register", "server_id": spec.server_id},
            )
            self.audit.record(
                actor=actor,
                action="mcp.server.register" if existing is None else "mcp.server.replace",
                target=self.server_resource(spec.server_id),
                decision={
                    "server_id": spec.server_id,
                    "transport": spec.transport,
                    "tools": [tool.tool_id for tool in spec.tools],
                    "replaced": existing is not None,
                    "source": source,
                },
            )
        return self.inspect_server(spec.server_id, actor=actor, require_capability=False)

    def register_server_from_yaml_text(
        self,
        text: str,
        *,
        actor: str,
        replace: bool = False,
        require_capability: bool = True,
        source: str | None = None,
    ) -> dict[str, Any]:
        if len(text.encode("utf-8")) > self.config.mcp.manifest_max_bytes:
            raise ValidationError(f"MCP manifest exceeds manifest_max_bytes={self.config.mcp.manifest_max_bytes}")
        data = load_yaml_mapping(text)
        if set(data) == {"mcp_server"} and isinstance(data["mcp_server"], dict):
            data = data["mcp_server"]
        if set(data) == {"server"} and isinstance(data["server"], dict):
            data = data["server"]
        return self.register_server(
            data,
            actor=actor,
            replace=replace,
            require_capability=require_capability,
            source=source,
        )

    def list_servers(
        self,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        text: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        servers, _has_more = self.list_servers_window(
            actor=actor,
            require_capability=require_capability,
            text=text,
            limit=limit,
        )
        return servers

    def list_servers_window(
        self,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        text: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return one bounded page plus an exact signal that another row exists."""

        if require_capability and actor is not None:
            self.capabilities.require(actor, self.config.mcp.registry_resource, CapabilityRight.READ)
        selected_limit = self._bounded_list_limit(limit)
        servers: list[dict[str, Any]] = []
        rows = self.extensions.list_mcp_servers(text=text, limit=selected_limit + 1)
        for spec, metadata in rows[:selected_limit]:
            self._validate_server(spec)
            servers.append(self._server_to_json(spec, metadata, include_sensitive_fields=False))
        return servers, len(rows) > selected_limit

    def inspect_server(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        include_sensitive_fields: bool = False,
    ) -> dict[str, Any]:
        if require_capability and actor is not None:
            self.capabilities.require(actor, self.server_resource(server_id), CapabilityRight.READ)
        spec, metadata = self._load_server(server_id)
        return self._server_to_json(spec, metadata, include_sensitive_fields=include_sensitive_fields)

    def list_tools(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        refresh: bool = False,
    ) -> dict[str, Any]:
        try:
            return self._list_tools(
                server_id,
                actor=actor,
                require_capability=require_capability,
                refresh=refresh,
            )
        except ProviderEffectNotStarted as error:
            raise self._safe_not_started_error(error) from None

    def _list_tools(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        refresh: bool = False,
    ) -> dict[str, Any]:
        authority_decisions: list[Any] = []
        if require_capability and actor is not None:
            read_decision = self.capabilities.require(
                actor,
                self.server_resource(server_id),
                CapabilityRight.READ,
                consume=not refresh,
            )
            if refresh:
                authority_decisions.append(read_decision)
        spec, _metadata = self._load_server(server_id)
        live_by_name: dict[str, McpProviderTool] = {}
        live_response_bytes = 0
        if refresh:
            effect_actor = actor or "runtime"
            usage_pid = self._resource_usage_pid(actor)
            if require_capability and actor is not None:
                authority_decisions.append(
                    self.capabilities.require(
                        actor,
                        self.server_resource(server_id),
                        CapabilityRight.EXECUTE,
                        consume=False,
                    )
                )
                authority_decisions.extend(self._require_stdio_process_spawn(actor, spec, consume=False))
            runtime_environment = self._require_runtime_environment(spec)
            request_bytes = len(dumps({"method": "tools/list", "server_id": spec.server_id}).encode("utf-8"))
            if request_bytes > spec.max_request_bytes:
                raise ValidationError(f"MCP list_tools request exceeds max_request_bytes={spec.max_request_bytes}")
            effect_context = self._list_tools_effect_context(spec, request_bytes=request_bytes)
            contract_name = (
                "primitive.mcp.list_tools"
                if authority_decisions
                else "primitive.mcp.list_tools.internal"
            )
            resource_context = {"server_id": server_id, "request_bytes": request_bytes}
            request_flow = (
                self._data_flow().current_context()
                if actor is not None
                else DataFlowContext(
                    labels=DataLabels(
                        sensitivity="public",
                        trust_level="verified",
                        integrity="verified",
                        origin="runtime:mcp-list-tools-metadata",
                    )
                )
            )
            stdio_executable_identity = self._stdio_executable_identity(
                spec,
                runtime_environment=runtime_environment,
            )
            list_sink = DataSink(
                f"mcp:{server_id}:list_tools",
                self._list_tools_identity_sha256(
                    spec,
                    stdio_executable=stdio_executable_identity,
                ),
            )
            invocation = ProtectedOperationInvocation(
                pid=effect_actor,
                actor=effect_actor,
                target=self.server_resource(spec.server_id),
                decisions=tuple(authority_decisions),
                canonical_args=effect_context,
                observation=effect_context,
                reservation_usage=(
                    ResourceUsage(
                        mcp_request_bytes=request_bytes,
                        mcp_response_bytes=spec.max_response_bytes,
                    )
                    if usage_pid is not None
                    else None
                ),
                resource_source="primitive.mcp.list_tools",
                resource_context=resource_context,
                **self._protected_registry_guard(self._registry_binding_for_server_spec(spec), server_id),
                data_sink=list_sink,
                data_sink_revalidator=lambda: self._list_tools_data_sink(
                    server_id, spec, runtime_environment
                ),
                data_flow_context=request_flow,
                data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                    request_flow,
                    origin="external:mcp",
                ),
                data_flow_payload={"method": "tools/list", "server_id": server_id},
                data_flow_operation="mcp.list_tools",
                failure_evidence=lambda error, phase: self._protected_list_failure_evidence(effect_actor, spec, effect_context, error, phase),
            )
            with self._protected().start(contract_name, invocation, provider=self.provider) as protected:
                started = time.monotonic()
                deadline = started + spec.timeout_s
                result, provider_error = self._dispatch_list_tools(
                    protected,
                    spec,
                    deadline=deadline,
                    pid=effect_actor,
                    expected_identity=stdio_executable_identity,
                    sink=list_sink,
                    context=request_flow,
                    payload={"method": "tools/list", "server_id": server_id},
                    runtime_environment=runtime_environment,
                )
                if provider_error is not None:
                    result_payload = self._list_tools_failure_payload(
                        provider_error,
                        duration_s=time.monotonic() - started,
                    )
                    protected.complete(
                        result_payload,
                        self._protected_list_evidence(effect_actor, spec, effect_context, result_payload),
                        classification_context=effect_context,
                        classification_result=result_payload,
                        resource=(
                            ResourceSettlement(
                                usage=ResourceUsage(mcp_request_bytes=request_bytes),
                                source="primitive.mcp.list_tools",
                                context={**resource_context, "response_bytes": 0, "status": result_payload["status"]},
                                charge_reserved_maximum=True,
                            )
                            if usage_pid is not None
                            else None
                        ),
                    )
                    raise provider_error
                live_response_bytes = result.response_bytes
                live_by_name = {tool.name: tool for tool in result.tools}
                result_payload = self._list_tools_success_payload(result)
                protected.complete(
                    result,
                    self._protected_list_evidence(effect_actor, spec, effect_context, result_payload),
                    classification_context=effect_context,
                    classification_result=result_payload,
                    resource=(
                        ResourceSettlement(
                            usage=ResourceUsage(
                                mcp_request_bytes=request_bytes,
                                mcp_response_bytes=live_response_bytes,
                            ),
                            source="primitive.mcp.list_tools",
                            context={**resource_context, "response_bytes": live_response_bytes},
                        )
                        if usage_pid is not None
                        else None
                    ),
                )
        return {
            "server_id": spec.server_id,
            "transport": spec.transport,
            "tools": [
                self._tool_to_json(spec.server_id, tool, live=live_by_name.get(tool.mcp_name) if refresh else None)
                for tool in spec.tools
            ],
            "refreshed": refresh,
            "response_bytes": live_response_bytes,
        }

    async def alist_tools(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Async facade that keeps synchronous providers off the active loop."""

        return await self._data_flow().run_sync_in_worker(
            self.list_tools,
            server_id,
            actor=actor,
            require_capability=require_capability,
            refresh=refresh,
        )

    def unregister_server(
        self,
        server_id: str,
        *,
        actor: str = "runtime",
        require_capability: bool = True,
    ) -> dict[str, Any]:
        authority_decisions: list[Any] = []
        if require_capability:
            authority_decisions.append(
                self.capabilities.require(
                    actor,
                    self.server_resource(server_id),
                    CapabilityRight.ADMIN,
                    consume=False,
                )
            )
        with self._registry_phase_lock, self.capabilities.authority_transaction(
            authority_decisions,
            actor=actor,
            operation="MCP server unregister",
        ):
            self._load_server(server_id)
            self._disable_replaced_server_tool_capabilities(server_id, actor=actor)
            self.extensions.delete_mcp_server(server_id)
            self.events.emit(
                EventType.EXTERNAL_WRITE,
                source=actor,
                target=self.server_resource(server_id),
                payload={"adapter": "mcp", "operation": "server_unregister", "server_id": server_id},
            )
            self.audit.record(
                actor=actor,
                action="mcp.server.unregister",
                target=self.server_resource(server_id),
                decision={"server_id": server_id},
            )
        return {"server_id": server_id, "deleted": True}

    def call_tool(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        arguments: Any = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> McpCallResult:
        try:
            return self._call_tool(
                pid,
                server_id,
                tool_id,
                arguments,
                source_oids=source_oids,
            )
        except ProviderEffectNotStarted as error:
            raise self._safe_not_started_error(error) from None

    def _call_tool(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        arguments: Any = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> McpCallResult:
        resource, selected_args, spec, tool, registry_binding = (
            self._resolve_tool_call_target(
                pid,
                server_id,
                tool_id,
                arguments,
                source_oids=source_oids,
            )
        )
        operation_context = self._operation_context(
            pid,
            spec,
            tool,
            selected_args,
            registry_binding=registry_binding,
        )
        flow_context = self._data_flow().context_from_source_oids(pid, source_oids)
        host_environment, runtime_environment, stdio_identity = self._capture_stdio_identity_environment(spec)
        sink = DataSink(
            f"mcp:{server_id}:{tool_id}",
            self._server_identity_sha256(
                spec,
                tool,
                stdio_executable=stdio_identity,
            ),
        )
        self._data_flow().authorize_egress(
            pid=pid,
            sink=sink,
            context=flow_context,
            payload=selected_args,
            operation="mcp.call_tool",
        )
        decision = self._authorize_call(
            pid,
            resource,
            tool.right,
            operation_context,
            source_oids=source_oids,
        )
        auxiliary_decisions = self._require_stdio_process_spawn(pid, spec, consume=False)
        self._validate_arguments_against_schema(tool, selected_args)
        profile = self.capabilities.profiles.mcp(
            resource=resource,
            effect=decision.effect or CapabilityEffect.DENY,
            server_id=server_id,
            tool_id=tool_id,
        )
        operation_context.update(
            {
                "capability_ids": list(
                    dict.fromkeys(
                        [
                            *decision.matched_capability_ids,
                            *[
                                cap_id
                                for auxiliary in auxiliary_decisions
                                for cap_id in auxiliary.matched_capability_ids
                            ],
                        ]
                    )
                ),
                "selected_capability_id": decision.selected_capability_id,
                "sandbox_profile": self._profile_json(profile),
            }
        )
        request_bytes = len(dumps({"name": tool.mcp_name, "arguments": selected_args}).encode("utf-8"))
        if request_bytes > spec.max_request_bytes:
            raise ValidationError(f"MCP request exceeds max_request_bytes={spec.max_request_bytes}")
        effect_context = self._effect_context(spec, tool, operation_context, request_bytes=request_bytes)
        list_request_bytes = len(
            dumps({"method": "tools/list", "server_id": spec.server_id}).encode("utf-8")
        )
        resource_context = {
            "server_id": server_id,
            "tool_id": tool_id,
            "request_bytes": request_bytes,
            "list_request_bytes": list_request_bytes,
        }
        resource_progress = {"list_response_bytes": 0}

        failure_resource = partial(
            self._mcp_call_failure_resource,
            spec=spec,
            request_bytes=request_bytes,
            list_request_bytes=list_request_bytes,
            resource_context=resource_context,
            resource_progress=resource_progress,
        )
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=tuple([decision, *auxiliary_decisions]),
            canonical_args=operation_context,
            observation=effect_context,
            reservation_usage=ResourceUsage(
                mcp_request_bytes=list_request_bytes + request_bytes,
                mcp_response_bytes=spec.max_response_bytes * 2,
            ),
            resource_source="primitive.mcp.call",
            resource_context=resource_context,
            **self._protected_registry_guard(registry_binding, server_id),
            failure_resource=failure_resource,
            failure_evidence=lambda error, phase: self._protected_call_failure_evidence(pid, resource, tool, operation_context, error, phase),
            data_sink=sink,
            data_sink_revalidator=lambda: self._tool_data_sink(
                server_id, spec, tool, runtime_environment
            ),
            data_flow_context=flow_context,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                flow_context,
                origin="external:mcp",
            ),
            data_flow_payload=selected_args,
            data_flow_operation="mcp.call_tool",
        )
        with self._protected().start("primitive.mcp.call", invocation, provider=self.provider) as protected:
            started = time.monotonic()
            deadline = started + spec.timeout_s
            runtime_environment = self._require_runtime_environment(spec, host_environment=host_environment)
            if spec.transport == "streamable_http":
                observes_host = self._runtime_resolution_observes_host(spec)
                protected.call(
                    ProviderPhase(
                        "dns_resolution",
                        information_flow=observes_host,
                        commits_authority=observes_host,
                    ),
                    self._validate_runtime_resolution,
                    spec,
                )

            validate_and_call = getattr(self.provider, "validate_and_call", None)
            if callable(validate_and_call):
                executable_snapshot = self._stdio_snapshot_for_dispatch(
                    pid=pid,
                    spec=spec,
                    expected_identity=stdio_identity,
                    sink=sink,
                    context=flow_context,
                    payload=selected_args,
                    runtime_environment=runtime_environment,
                )

                def invoke_validated_tool() -> Any:
                    try:
                        provider_kwargs = self._provider_dispatch_kwargs(
                            spec,
                            deadline=deadline,
                            runtime_environment=runtime_environment,
                            executable_snapshot=executable_snapshot,
                        )
                        raw_result = validate_and_call(
                            spec,
                            tool,
                            selected_args,
                            **provider_kwargs,
                        )
                        return self._validated_provider_call_result(raw_result)
                    except ProviderEffectNotStarted as error:
                        return ProviderEffectNotStartedResult(
                            error=error,
                            outcome="validate_and_call_not_started",
                            result=McpProviderCallResult(
                                error="provider call did not start",
                                error_type=type(error).__name__,
                                correlation_id=new_id("corr"),
                                duration_s=time.monotonic() - started,
                                call_started=False,
                            ),
                        )
                    except ProviderHostError:
                        raise
                    except Exception as error:
                        raise ProviderHostError(
                            code="mcp_provider_error",
                            error_type=type(error).__name__,
                            correlation_id=new_id("corr"),
                        ) from None

                try:
                    provider_outcome = protected.call(
                        ProviderPhase(
                            "provider_validate_and_call",
                            information_flow=True,
                        ),
                        invoke_validated_tool,
                    )
                finally:
                    if executable_snapshot is not None:
                        executable_snapshot.close()
                if isinstance(provider_outcome, ProviderEffectNotStartedResult):
                    return self._call_result_from_provider(
                        spec,
                        tool,
                        provider_outcome.result,
                    )
                provider_result = provider_outcome
                result = self._call_result_from_provider(spec, tool, provider_result)
                classification_override = None
                if not provider_result.call_started:
                    classification_override = ExternalEffectClassification(
                        rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                        rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                        state_mutation=False,
                        information_flow=True,
                        metadata={
                            "outcome": "live_validation_failed_before_call",
                            "phase": "live_validation",
                        },
                    )
                return protected.complete(
                    result,
                    self._protected_call_evidence(pid, resource, result, tool, operation_context),
                    classification_context=effect_context,
                    classification_result=self._call_effect_result(result),
                    classification_override=classification_override,
                    resource=self._mcp_exchange_settlement(
                        provider_result,
                        fallback_list_request_bytes=list_request_bytes,
                        fallback_call_request_bytes=request_bytes,
                        server_id=server_id,
                        tool_id=tool_id,
                        status=result.status.value,
                    ),
                )

            live_list_result, validation_error, list_response_bytes = (
                self._dispatch_live_tool_validation(
                    protected,
                    spec,
                    tool,
                    deadline=deadline,
                    pid=pid,
                    expected_identity=stdio_identity,
                    sink=sink,
                    context=flow_context,
                    payload=selected_args,
                    runtime_environment=runtime_environment,
                )
            )
            resource_progress["list_response_bytes"] = list_response_bytes
            if validation_error is not None:
                result = self._live_validation_failure_result(
                    spec,
                    tool,
                    validation_error,
                    duration_s=time.monotonic() - started,
                )
                protected.complete(
                    result,
                    self._protected_call_evidence(pid, resource, result, tool, operation_context),
                    classification_context=effect_context,
                    classification_result=self._call_effect_result(result),
                    resource=self._mcp_live_validation_failure_settlement(
                        spec,
                        request_bytes,
                        result,
                        server_id=server_id,
                        tool_id=tool_id,
                        list_request_bytes=list_request_bytes,
                        live_list_result=live_list_result,
                    ),
                )
                raise validation_error

            deadline_result = self._complete_expired_legacy_exchange(
                protected=protected,
                spec=spec,
                tool=tool,
                operation_context=operation_context,
                effect_context=effect_context,
                live_list_result=live_list_result,
                deadline=deadline,
                started=started,
                list_request_bytes=list_request_bytes,
                request_bytes=request_bytes,
            )
            if deadline_result is not None:
                return deadline_result

            def invoke_tool() -> (
                tuple[McpProviderCallResult, ExternalEffectClassification | None]
                | ProviderEffectNotStartedResult
            ):
                try:
                    provider_kwargs = self._provider_dispatch_kwargs(
                        spec,
                        deadline=deadline,
                        runtime_environment=runtime_environment,
                        executable_snapshot=executable_snapshot,
                    )
                    raw_result = self.provider.call_tool(
                        spec,
                        tool,
                        selected_args,
                        **provider_kwargs,
                    )
                    return self._validated_provider_call_result(raw_result), None
                except ProviderEffectNotStarted as error:
                    return ProviderEffectNotStartedResult(
                        error=error,
                        outcome="call_tool_not_started_after_live_validation",
                        result=McpProviderCallResult(
                            error="provider call did not start",
                            error_type=type(error).__name__,
                            correlation_id=new_id("corr"),
                            duration_s=time.monotonic() - started,
                        ),
                    )
                except ProviderHostError:
                    raise
                except Exception as error:
                    return (
                        McpProviderCallResult(
                            error="provider call failed",
                            error_type=type(error).__name__,
                            correlation_id=new_id("corr"),
                            duration_s=time.monotonic() - started,
                        ),
                        ExternalEffectClassification(
                            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                            state_mutation=tool.state_mutation,
                            information_flow=True,
                            metadata={
                                "outcome": "unknown_provider_exception",
                                "phase": "provider_call",
                                "error_type": type(error).__name__,
                            },
                        ),
                    )

            executable_snapshot = self._stdio_snapshot_for_dispatch(
                pid=pid,
                spec=spec,
                expected_identity=stdio_identity,
                sink=sink,
                context=flow_context,
                payload=selected_args,
                runtime_environment=runtime_environment,
            )
            try:
                provider_outcome = protected.call(
                    ProviderPhase(
                        "provider_call",
                        state_mutation=tool.state_mutation,
                        information_flow=True,
                    ),
                    invoke_tool,
                )
            finally:
                if executable_snapshot is not None:
                    executable_snapshot.close()
            if isinstance(provider_outcome, ProviderEffectNotStartedResult):
                result = self._call_result_from_provider(spec, tool, provider_outcome.result)
                return result
            provider_result, classification_override = provider_outcome
            result = self._call_result_from_provider(spec, tool, provider_result)
            completed = protected.complete(
                result,
                self._protected_call_evidence(pid, resource, result, tool, operation_context),
                classification_context=effect_context,
                classification_result=self._call_effect_result(result),
                classification_override=classification_override,
                resource=self._mcp_legacy_call_completion_settlement(
                    spec,
                    request_bytes,
                    result,
                    provider_result,
                    classification_override,
                    server_id=server_id,
                    tool_id=tool_id,
                    list_request_bytes=list_request_bytes,
                    live_list_result=live_list_result,
                ),
            )
            return completed

    def _resolve_tool_call_target(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        arguments: Any,
        *,
        source_oids: list[str] | tuple[str, ...] | None,
    ) -> tuple[
        str,
        dict[str, Any],
        McpServerSpec,
        McpToolSpec,
        dict[str, Any],
    ]:
        resource = self.tool_resource(server_id, tool_id)
        selected_args = {} if arguments is None else arguments
        if not isinstance(selected_args, dict):
            raise ValidationError("MCP tool arguments must be a JSON object or null")
        self._validate_json_value(selected_args, "arguments")
        visibility_context = self._visibility_operation_context(
            pid,
            server_id,
            tool_id,
            selected_args,
        )
        self._authorize_call_visibility(
            pid,
            resource,
            visibility_context,
            source_oids=source_oids,
        )
        spec, _metadata = self._load_server(server_id)
        tool = spec.tool_by_id(tool_id)
        if tool is None:
            raise NotFound(f"MCP tool not found: {server_id}/{tool_id}")
        return (
            resource,
            selected_args,
            spec,
            tool,
            self._registry_binding_for_server_spec(spec),
        )

    async def acall_tool(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        arguments: Any = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> McpCallResult:
        return await self._data_flow().run_sync_in_worker(
            self.call_tool,
            pid,
            server_id,
            tool_id,
            arguments,
            source_oids=source_oids,
        )

    def grant_tool(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        *,
        right: str | CapabilityRight,
        issued_by: str = "mcp",
        delegable: bool = True,
    ) -> Any:
        return self.capabilities.grant(
            subject=pid,
            resource=self.tool_resource(server_id, tool_id),
            rights=[CapabilityRight(str(right))],
            issued_by=issued_by,
            delegable=delegable,
        )

    def _authorize_call(
        self,
        pid: str,
        resource: str,
        right: str,
        context: dict[str, Any],
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        decision = self.capabilities.authorize(pid, resource, right, context, audit=True)
        if decision.allowed:
            return decision
        if decision.policy == CapabilityManager.ASK_EACH_TIME:
            if self.human is None:
                raise CapabilityDenied(f"{pid} requires human approval for MCP call on {resource}")
            profile = self.capabilities.profiles.mcp(
                resource=resource,
                effect=CapabilityEffect.ASK,
                server_id=str(context["server_id"]),
                tool_id=str(context["tool_id"]),
            )
            approval_context = {**context, "sandbox_profile": self._profile_json(profile)}
            request_id = self.human.query(
                pid=pid,
                human=self.config.runtime.default_human,
                request={
                    "type": "external_operation_approval",
                    "question": f"Allow this process to call MCP tool {resource}?",
                    "requested_once_capability": {
                        "subject": pid,
                        "resource": resource,
                        "rights": [right],
                        "constraints": self._approval_constraints(context),
                    },
                    "context": approval_context,
                },
                blocking=True,
                source_oids=source_oids,
            )
            raise HumanApprovalRequired(
                request_id=request_id,
                message=f"{pid} is waiting for per-use human approval to call {resource}",
            )
        raise CapabilityDenied(decision.reason)

    def _authorize_call_visibility(
        self,
        pid: str,
        resource: str,
        context: dict[str, Any],
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        bound_context: dict[str, Any] | None = None
        for right in (CapabilityRight.READ, CapabilityRight.WRITE, CapabilityRight.EXECUTE):
            selected_context = {**context, "right": str(right)}
            decision = self.capabilities.authorize(pid, resource, right, selected_context)
            if decision.allowed:
                return
            # Keep unprivileged callers ahead of all registry reads. An ASK
            # policy or an existing constrained grant may resolve only the
            # metadata-free digest/generation binding needed for exact use.
            if not (
                decision.policy == CapabilityManager.ASK_EACH_TIME
                or decision.matched_capability_ids
            ):
                continue
            if bound_context is None:
                bound_context = {
                    **context,
                    **self._registry_binding_context(str(context["server_id"])),
                }
            rebound = self.capabilities.authorize(
                pid,
                resource,
                right,
                {**bound_context, "right": str(right)},
            )
            if rebound.allowed:
                return
            if rebound.policy == CapabilityManager.ASK_EACH_TIME:
                self._request_visibility_approval(
                    pid,
                    resource,
                    str(right),
                    bound_context,
                    source_oids=source_oids,
                )
        raise CapabilityDenied(f"{pid} lacks MCP call authority on {resource}")

    def _request_visibility_approval(
        self,
        pid: str,
        resource: str,
        right: str,
        context: dict[str, Any],
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if self.human is None:
            raise CapabilityDenied(f"{pid} requires human approval for MCP call on {resource}")
        profile = self.capabilities.profiles.mcp(
            resource=resource,
            effect=CapabilityEffect.ASK,
            server_id=str(context["server_id"]),
            tool_id=str(context["tool_id"]),
        )
        approval_context = {**context, "right": right, "sandbox_profile": self._profile_json(profile)}
        request_id = self.human.query(
            pid=pid,
            human=self.config.runtime.default_human,
            request={
                "type": "external_operation_approval",
                "question": f"Allow this process to call MCP tool {resource}?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [right],
                    "constraints": self._approval_constraints(context),
                },
                "context": approval_context,
            },
            blocking=True,
            source_oids=source_oids,
        )
        raise HumanApprovalRequired(
            request_id=request_id,
            message=f"{pid} is waiting for per-use human approval to call {resource}",
        )

    def _validate_live_tool(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        *,
        timeout_s: float | None = None,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpToolListResult:
        provider_kwargs: dict[str, Any] = {
            "timeout_s": server.timeout_s if timeout_s is None else timeout_s,
            "max_response_bytes": server.max_response_bytes,
            "runtime_environment": runtime_environment,
        }
        if executable_snapshot is not None:
            provider_kwargs["executable_snapshot"] = executable_snapshot
        try:
            raw_result = self.provider.list_tools(
                server,
                **provider_kwargs,
            )
            result = self._validated_tool_list_result(server, raw_result)
            live = next(
                (item for item in result.tools if item.name == tool.mcp_name),
                None,
            )
            if live is None:
                raise _McpLiveToolValidationError(
                    f"MCP server {server.server_id} no longer exposes tool {tool.mcp_name}",
                    result,
                )
            if tool.input_schema and live.input_schema != tool.input_schema:
                raise _McpLiveToolValidationError(
                    f"MCP tool schema changed for {server.server_id}/{tool.tool_id}",
                    result,
                )
        except ProviderEffectNotStarted:
            raise
        except _McpLiveToolValidationError:
            raise
        except ProviderHostError:
            raise
        except Exception as error:
            raise ProviderHostError(
                code="mcp_provider_error",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            ) from None
        return result

    def _invoke_list_tools_provider(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        executable_snapshot: ExecutableSnapshot | None,
        runtime_environment: Mapping[str, str],
    ) -> tuple[McpToolListResult | None, ProviderHostError | None]:
        try:
            provider_kwargs: dict[str, Any] = {
                "timeout_s": self._remaining_timeout(deadline),
                "max_response_bytes": server.max_response_bytes,
                "runtime_environment": runtime_environment,
            }
            if executable_snapshot is not None:
                provider_kwargs["executable_snapshot"] = executable_snapshot
            raw_result = self.provider.list_tools(server, **provider_kwargs)
            return self._validated_tool_list_result(server, raw_result), None
        except ProviderEffectNotStarted:
            raise
        except ProviderHostError as error:
            return None, error
        except Exception as error:
            return None, ProviderHostError(
                code="mcp_provider_error",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            )

    def _dispatch_list_tools(
        self,
        protected: Any,
        server: McpServerSpec,
        *,
        deadline: float,
        pid: str,
        expected_identity: dict[str, str] | None,
        sink: DataSink,
        context: DataFlowContext,
        payload: Any,
        runtime_environment: Mapping[str, str],
    ) -> tuple[McpToolListResult | None, ProviderHostError | None]:
        if server.transport == "streamable_http":
            observes_host = self._runtime_resolution_observes_host(server)
            protected.call(
                ProviderPhase(
                    "dns_resolution",
                    information_flow=observes_host,
                    commits_authority=observes_host,
                ),
                self._validate_runtime_resolution,
                server,
            )
        executable_snapshot = self._stdio_snapshot_for_dispatch(
            pid=pid,
            spec=server,
            expected_identity=expected_identity,
            sink=sink,
            context=context,
            payload=payload,
            runtime_environment=runtime_environment,
        )
        try:
            return protected.call(
                ProviderPhase(
                    "provider_not_started_after_dns",
                    information_flow=True,
                ),
                self._invoke_list_tools_provider,
                server,
                deadline=deadline,
                executable_snapshot=executable_snapshot,
                runtime_environment=runtime_environment,
            )
        finally:
            if executable_snapshot is not None:
                executable_snapshot.close()

    def _validated_tool_list_result(
        self,
        server: McpServerSpec,
        result: Any,
    ) -> McpToolListResult:
        """Decode every provider-owned tools/list field before it escapes."""

        try:
            if not isinstance(result, McpToolListResult):
                raise TypeError("MCP provider returned an invalid tools/list result")
            server_id = result.server_id
            tools = result.tools
            response_bytes = result.response_bytes
            duration_s = result.duration_s
            self._validate_tool_list_header(
                server,
                server_id=server_id,
                tools=tools,
                response_bytes=response_bytes,
                duration_s=duration_s,
            )
            return McpToolListResult(
                server_id=server_id,
                tools=self._validated_provider_tools(tools),
                response_bytes=response_bytes,
                duration_s=float(duration_s),
            )
        except ProviderHostError as error:
            _mark_provider_result_returned(error)
            raise
        except Exception as error:
            raise _mark_provider_result_returned(
                ProviderHostError(
                    code="mcp_provider_error",
                    error_type=type(error).__name__,
                    correlation_id=new_id("corr"),
                )
            ) from None

    @staticmethod
    def _validate_tool_list_header(
        server: McpServerSpec,
        *,
        server_id: Any,
        tools: Any,
        response_bytes: Any,
        duration_s: Any,
    ) -> None:
        if type(server_id) is not str or server_id != server.server_id:
            raise TypeError("MCP tools/list server identity is invalid")
        if type(tools) is not list:
            raise TypeError("MCP tools/list tools must be a list")
        if (
            type(response_bytes) is not int
            or response_bytes < 0
            or response_bytes > server.max_response_bytes
        ):
            raise TypeError("MCP tools/list response_bytes is invalid")
        if (
            type(duration_s) not in {int, float}
            or not math.isfinite(duration_s)
            or duration_s < 0
        ):
            raise TypeError("MCP tools/list duration_s is invalid")

    @staticmethod
    def _validated_provider_tools(tools: list[Any]) -> list[McpProviderTool]:
        selected_tools: list[McpProviderTool] = []
        names: set[str] = set()
        for item in tools:
            if not isinstance(item, McpProviderTool):
                raise TypeError("MCP tools/list contains an invalid tool")
            name = item.name
            description = item.description
            input_schema = item.input_schema
            metadata = item.metadata
            if type(name) is not str or not name:
                raise TypeError("MCP tools/list tool name is invalid")
            if name in names:
                raise TypeError("MCP tools/list contains duplicate tool names")
            if description is not None and type(description) is not str:
                raise TypeError("MCP tools/list tool description is invalid")
            if type(input_schema) is not dict or type(metadata) is not dict:
                raise TypeError("MCP tools/list tool metadata is invalid")
            selected_tools.append(
                McpProviderTool(
                    name=name,
                    description=description,
                    input_schema=_strict_provider_json_value(
                        input_schema,
                        path="$.tools[].input_schema",
                        active_containers=set(),
                    ),
                    metadata=_strict_provider_json_value(
                        metadata,
                        path="$.tools[].metadata",
                        active_containers=set(),
                    ),
                )
            )
            names.add(name)
        return selected_tools

    @staticmethod
    def _validated_provider_call_result(result: Any) -> McpProviderCallResult:
        """Decode every provider-owned call field into an inert value object."""

        try:
            if not isinstance(result, McpProviderCallResult):
                raise TypeError("MCP provider returned an invalid call result")
            content = result.content
            structured_content = result.structured_content
            is_error = result.is_error
            error = result.error
            response_bytes = result.response_bytes
            duration_s = result.duration_s
            too_large = result.too_large
            error_type = result.error_type
            correlation_id = result.correlation_id
            list_request_bytes = result.list_request_bytes
            list_response_bytes = result.list_response_bytes
            call_request_bytes = result.call_request_bytes
            call_response_bytes = result.call_response_bytes
            call_started = result.call_started
            if type(is_error) is not bool or type(too_large) is not bool:
                raise TypeError("MCP provider call flags are invalid")
            if type(call_started) is not bool:
                raise TypeError("MCP provider call_started is invalid")
            for field_name, selected in (
                ("response_bytes", response_bytes),
                ("list_request_bytes", list_request_bytes),
                ("list_response_bytes", list_response_bytes),
                ("call_request_bytes", call_request_bytes),
                ("call_response_bytes", call_response_bytes),
            ):
                if type(selected) is not int or selected < 0:
                    raise TypeError(f"MCP provider {field_name} is invalid")
            if (
                type(duration_s) not in {int, float}
                or not math.isfinite(duration_s)
                or duration_s < 0
            ):
                raise TypeError("MCP provider duration_s is invalid")
            for field_name, selected in (
                ("error", error),
                ("error_type", error_type),
                ("correlation_id", correlation_id),
            ):
                if selected is not None and type(selected) is not str:
                    raise TypeError(f"MCP provider {field_name} is invalid")
            return McpProviderCallResult(
                content=_strict_provider_json_value(
                    content,
                    path="$.content",
                    active_containers=set(),
                ),
                structured_content=_strict_provider_json_value(
                    structured_content,
                    path="$.structured_content",
                    active_containers=set(),
                ),
                is_error=is_error,
                error=error,
                response_bytes=response_bytes,
                duration_s=float(duration_s),
                too_large=too_large,
                error_type=error_type,
                correlation_id=correlation_id,
                list_request_bytes=list_request_bytes,
                list_response_bytes=list_response_bytes,
                call_request_bytes=call_request_bytes,
                call_response_bytes=call_response_bytes,
                call_started=call_started,
            )
        except ProviderHostError as error:
            _mark_provider_result_returned(error)
            raise
        except Exception as error:
            raise _mark_provider_result_returned(
                ProviderHostError(
                    code="mcp_provider_error",
                    error_type=type(error).__name__,
                    correlation_id=new_id("corr"),
                )
            ) from None

    def _validate_live_tool_for_call(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        *,
        deadline: float,
        executable_snapshot: ExecutableSnapshot | None,
        runtime_environment: Mapping[str, str],
    ) -> tuple[McpToolListResult | None, Exception | None, int]:
        """Retain known list bytes while keeping not-started failures exceptional."""

        try:
            result = self._validate_live_tool(
                server,
                tool,
                timeout_s=self._remaining_timeout(deadline),
                executable_snapshot=executable_snapshot,
                runtime_environment=runtime_environment,
            )
        except _McpLiveToolValidationError as error:
            return error.result, error, error.result.response_bytes
        except ProviderEffectNotStarted:
            raise
        except Exception as error:
            return None, error, 0
        return result, None, result.response_bytes

    def _dispatch_live_tool_validation(
        self,
        protected: Any,
        server: McpServerSpec,
        tool: McpToolSpec,
        *,
        deadline: float,
        pid: str,
        expected_identity: dict[str, str] | None,
        sink: DataSink,
        context: DataFlowContext,
        payload: Any,
        runtime_environment: Mapping[str, str],
    ) -> tuple[McpToolListResult | None, Exception | None, int]:
        executable_snapshot = self._stdio_snapshot_for_dispatch(
            pid=pid,
            spec=server,
            expected_identity=expected_identity,
            sink=sink,
            context=context,
            payload=payload,
            runtime_environment=runtime_environment,
        )
        try:
            return protected.call(
                ProviderPhase(
                    "live_validation_not_started_after_dns",
                    information_flow=True,
                ),
                lambda: self._validate_live_tool_for_call(
                    server,
                    tool,
                    deadline=deadline,
                    executable_snapshot=executable_snapshot,
                    runtime_environment=runtime_environment,
                ),
            )
        finally:
            if executable_snapshot is not None:
                executable_snapshot.close()

    @staticmethod
    def _remaining_timeout(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderEffectNotStarted(
                "MCP deadline exhausted before provider dispatch"
            )
        return remaining

    def _provider_dispatch_kwargs(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        runtime_environment: Mapping[str, str],
        executable_snapshot: ExecutableSnapshot | None,
    ) -> dict[str, Any]:
        selected: dict[str, Any] = {
            "timeout_s": self._remaining_timeout(deadline),
            "max_response_bytes": server.max_response_bytes,
            "runtime_environment": runtime_environment,
        }
        if executable_snapshot is not None:
            selected["executable_snapshot"] = executable_snapshot
        return selected

    @staticmethod
    def _safe_not_started_error(error: ProviderEffectNotStarted) -> ProviderHostError:
        return ProviderHostError(
            code="mcp_provider_not_started",
            error_type=type(error).__name__,
            correlation_id=new_id("corr"),
        )

    def _require_stdio_process_spawn(
        self,
        actor: str | None,
        server: McpServerSpec,
        *,
        consume: bool = True,
    ) -> list[Any]:
        if actor is None or server.transport != "stdio":
            return []
        decisions = [
            self.capabilities.require(
                actor,
                "process:spawn",
                CapabilityRight.WRITE,
                consume=consume,
            )
        ]
        if server.stdio is None:
            raise ValidationError("MCP stdio transport is missing stdio configuration")
        resource = self.stdio_resource_for_server(server)
        if resource is None:
            raise ValidationError("MCP stdio transport is missing stdio authority resource")
        decisions.append(
            self.capabilities.require(
                actor,
                resource,
                CapabilityRight.EXECUTE,
                {
                    "adapter": "mcp",
                    "operation": "mcp.stdio.spawn",
                    "server_id": server.server_id,
                    "stdio_command": server.stdio.command,
                    "stdio_args": list(server.stdio.args),
                    "stdio_env": dict(server.stdio.env),
                    "stdio_cwd": self._canonical_stdio_cwd(server.stdio.cwd),
                },
                consume=consume,
            )
        )
        return decisions

    def _call_result_from_provider(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        provider_result: McpProviderCallResult,
    ) -> McpCallResult:
        provider_result = self._validated_provider_call_result(provider_result)
        if provider_result.error:
            return McpCallResult(
                server_id=server.server_id,
                tool_id=tool.tool_id,
                mcp_name=tool.mcp_name,
                status=McpCallStatus.TRANSPORT_ERROR,
                ok=False,
                error={
                    "code": "mcp_provider_error",
                    "error_type": provider_result.error_type or "TransportError",
                    "correlation_id": provider_result.correlation_id or new_id("corr"),
                },
                response_bytes=provider_result.response_bytes,
                duration_s=provider_result.duration_s,
            )
        if provider_result.too_large:
            return self._failure(
                server,
                tool,
                McpCallStatus.RESPONSE_TOO_LARGE,
                f"response exceeded max_response_bytes={server.max_response_bytes}",
                provider_result,
            )
        if provider_result.is_error:
            return self._failure(
                server,
                tool,
                McpCallStatus.MCP_ERROR,
                "MCP tool returned an error result",
                provider_result,
                extra={"content": provider_result.content},
            )
        return McpCallResult(
            server_id=server.server_id,
            tool_id=tool.tool_id,
            mcp_name=tool.mcp_name,
            status=McpCallStatus.OK,
            ok=True,
            result={
                "content": to_jsonable(provider_result.content),
                "structured_content": to_jsonable(provider_result.structured_content),
            },
            response_bytes=provider_result.response_bytes,
            duration_s=provider_result.duration_s,
        )

    def _failure(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        status: McpCallStatus,
        message: str,
        provider_result: McpProviderCallResult,
        *,
        extra: dict[str, Any] | None = None,
    ) -> McpCallResult:
        code = {
            McpCallStatus.MCP_ERROR: "mcp_tool_error",
            McpCallStatus.INVALID_RESPONSE: "mcp_invalid_response",
            McpCallStatus.RESPONSE_TOO_LARGE: "mcp_response_too_large",
            McpCallStatus.TRANSPORT_ERROR: "mcp_provider_error",
        }.get(status, "mcp_call_failed")
        return McpCallResult(
            server_id=server.server_id,
            tool_id=tool.tool_id,
            mcp_name=tool.mcp_name,
            status=status,
            ok=False,
            error={
                "code": code,
                "error_type": provider_result.error_type or status.value,
                "correlation_id": provider_result.correlation_id or new_id("corr"),
                "message": message,
                **dict(extra or {}),
            },
            response_bytes=provider_result.response_bytes,
            duration_s=provider_result.duration_s,
        )

    def _protected(self) -> Any:
        return self.protected_operations

    def _data_flow(self) -> Any:
        manager = getattr(self, "data_flow", None) or getattr(
            self._protected(),
            "data_flow",
            None,
        )
        if manager is None:
            raise ValidationError("MCP data-flow manager is not attached")
        return manager

    def _server_identity_sha256(
        self,
        spec: McpServerSpec,
        tool: McpToolSpec,
        *,
        stdio_executable: dict[str, str] | None | object = (
            _STDIO_EXECUTABLE_IDENTITY_UNSET
        ),
    ) -> str | None:
        if stdio_executable is _STDIO_EXECUTABLE_IDENTITY_UNSET:
            stdio_executable = self._stdio_executable_identity(spec)
        if spec.transport == "stdio" and stdio_executable is None:
            # A stdio provider that cannot resolve the exact executable may
            # still handle normal data, but it cannot match Host clearance for
            # data above normal sensitivity.
            return None
        return hashlib.sha256(
            dumps(
                to_jsonable(
                    {
                        "schema_version": 1,
                        "server": spec,
                        "tool": tool,
                        "stdio_executable": stdio_executable,
                    }
                )
            ).encode("utf-8")
        ).hexdigest()

    def _list_tools_identity_sha256(
        self,
        spec: McpServerSpec,
        *,
        stdio_executable: dict[str, str] | None | object = (
            _STDIO_EXECUTABLE_IDENTITY_UNSET
        ),
    ) -> str | None:
        if stdio_executable is _STDIO_EXECUTABLE_IDENTITY_UNSET:
            stdio_executable = self._stdio_executable_identity(spec)
        if spec.transport == "stdio" and stdio_executable is None:
            return None
        return hashlib.sha256(
            dumps(
                to_jsonable(
                    {
                        "schema_version": 1,
                        "server": spec,
                        "operation": "tools/list",
                        "stdio_executable": stdio_executable,
                    }
                )
            ).encode("utf-8")
        ).hexdigest()

    def _stdio_executable_identity(
        self,
        spec: McpServerSpec,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> dict[str, str] | None:
        if spec.transport != "stdio" or spec.stdio is None:
            return None
        resolver = getattr(self.provider, "resolve_stdio_executable", None)
        if not callable(resolver):
            return None
        try:
            resolver_kwargs = (
                {"runtime_environment": runtime_environment}
                if runtime_environment is not None
                and bool(
                    getattr(
                        self.provider,
                        "supports_runtime_environment_snapshots",
                        False,
                    )
                )
                else {}
            )
            resolved = Path(resolver(spec, **resolver_kwargs)).resolve(strict=True)
            return {
                "path": resolved.as_posix(),
                "content_sha256": executable_content_sha256(resolved),
            }
        except (OSError, ValidationError):
            # Preserve normal-sensitivity compatibility, but leave the Sink
            # unidentified so any Host rule above normal fails closed.
            return None

    def _capture_stdio_identity_environment(
        self,
        spec: McpServerSpec,
    ) -> tuple[
        Mapping[str, str],
        Mapping[str, str] | None,
        dict[str, str] | None,
    ]:
        host_environment = MappingProxyType(dict(os.environ))
        try:
            runtime_environment: Mapping[str, str] | None = self._runtime_environment_from_host(
                spec,
                host_environment,
            )
        except ValidationError:
            # Keep environment validation behind the existing capability and
            # process-spawn gates.  An unresolved stdio Sink still fails
            # closed for data above normal sensitivity.
            runtime_environment = None
        stdio_identity = (
            self._stdio_executable_identity(
                spec,
                runtime_environment=runtime_environment,
            )
            if runtime_environment is not None
            else None
        )
        return host_environment, runtime_environment, stdio_identity

    def _tool_data_sink(
        self,
        server_id: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
        runtime_environment: Mapping[str, str],
    ) -> DataSink:
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=runtime_environment,
        )
        return DataSink(
            f"mcp:{server_id}:{tool.tool_id}",
            self._server_identity_sha256(
                spec,
                tool,
                stdio_executable=stdio_identity,
            ),
        )

    def _list_tools_data_sink(
        self,
        server_id: str,
        spec: McpServerSpec,
        runtime_environment: Mapping[str, str],
    ) -> DataSink:
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=runtime_environment,
        )
        return DataSink(
            f"mcp:{server_id}:list_tools",
            self._list_tools_identity_sha256(
                spec,
                stdio_executable=stdio_identity,
            ),
        )

    def _stdio_snapshot_for_dispatch(
        self,
        *,
        pid: str,
        spec: McpServerSpec,
        expected_identity: dict[str, str] | None,
        sink: DataSink,
        context: DataFlowContext,
        payload: Any,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> ExecutableSnapshot | None:
        if spec.transport != "stdio" or spec.stdio is None:
            return None
        resolver = getattr(self.provider, "resolve_stdio_executable", None)
        checker = getattr(self.provider, "executable_snapshot_required", None)
        if not callable(resolver) or not callable(checker):
            return None
        resolver_kwargs = (
            {"runtime_environment": runtime_environment}
            if runtime_environment is not None
            and bool(
                getattr(
                    self.provider,
                    "supports_runtime_environment_snapshots",
                    False,
                )
            )
            else {}
        )
        resolved = Path(resolver(spec, **resolver_kwargs)).resolve(strict=True)
        snapshot_required = bool(
            checker(spec, str(resolved), **resolver_kwargs)
        )
        if not snapshot_required:
            if expected_identity is None:
                return None
            actual = {
                "path": resolved.as_posix(),
                "content_sha256": executable_content_sha256(resolved),
            }
            if actual != expected_identity:
                self._data_flow().reject_sink_identity_change(
                    pid=pid,
                    sink=sink,
                    context=context,
                    payload=payload,
                    reason="MCP stdio Sink identity changed before dispatch",
                )
                raise AssertionError("data-flow Sink rejection must raise")
            return None
        if not bool(getattr(self.provider, "supports_executable_snapshots", False)):
            self._data_flow().reject_sink_identity_change(
                pid=pid,
                sink=sink,
                context=context,
                payload=payload,
                reason="MCP provider cannot pin a mutable stdio executable",
            )
            raise AssertionError("data-flow Sink rejection must raise")
        snapshot = snapshot_executable(
            resolved,
            sibling_limit=self.config.tools.executable_snapshot_sibling_limit,
        )
        actual = {
            "path": snapshot.source_path.as_posix(),
            "content_sha256": snapshot.content_sha256,
        }
        if expected_identity is None or actual != expected_identity:
            snapshot.close()
            self._data_flow().reject_sink_identity_change(
                pid=pid,
                sink=sink,
                context=context,
                payload=payload,
                reason=(
                    "MCP stdio Sink identity changed before immutable "
                    "executable dispatch snapshot"
                ),
            )
            raise AssertionError("data-flow Sink rejection must raise")
        return snapshot

    def _protected_list_evidence(
        self,
        pid: str,
        spec: McpServerSpec,
        context: dict[str, Any],
        result_payload: dict[str, Any],
    ) -> ProtectedOperationEvidence:
        event_payload = {
            "adapter": "mcp",
            "operation": "list_tools",
            "server_id": spec.server_id,
            "transport": spec.transport,
            **result_payload,
        }
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_READ,
            event_source=pid,
            event_target=self.server_resource(spec.server_id),
            event_payload=event_payload,
            audit_action="primitive.mcp.list_tools",
            audit_actor=pid,
            audit_target=self.server_resource(spec.server_id),
            audit_decision={
                "server_id": spec.server_id,
                "transport": spec.transport,
                "request_bytes": context["request_bytes"],
                **result_payload,
            },
            effect_metadata=result_payload,
            provider_receipt={
                "response_bytes": int(result_payload.get("response_bytes", 0) or 0),
                "duration_s": float(result_payload.get("duration_s", 0.0) or 0.0),
            },
        )

    def _protected_list_failure_evidence(
        self,
        pid: str,
        spec: McpServerSpec,
        context: dict[str, Any],
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        return self._protected_list_evidence(
            pid,
            spec,
            context,
            {
                "ok": False,
                "status": "transport_error",
                "response_bytes": 0,
                "duration_s": 0.0,
                "tool_count": 0,
                "error_type": type(error).__name__,
                "phase": phase,
            },
        )

    def _protected_call_evidence(
        self,
        pid: str,
        resource: str,
        result: McpCallResult,
        tool: McpToolSpec,
        operation_context: dict[str, Any],
    ) -> ProtectedOperationEvidence:
        result_payload = self._call_effect_result(result)
        return ProtectedOperationEvidence(
            event_type=(
                EventType.EXTERNAL_WRITE
                if tool.state_mutation or tool.right != CapabilityRight.READ.value
                else EventType.EXTERNAL_READ
            ),
            event_source=pid,
            event_target=resource,
            event_payload={
                "adapter": "mcp",
                "server_id": result.server_id,
                "tool_id": result.tool_id,
                "mcp_name": result.mcp_name,
                **result_payload,
            },
            audit_action="primitive.mcp.call",
            audit_actor=pid,
            audit_target=resource,
            audit_decision={
                "server_id": result.server_id,
                "tool_id": result.tool_id,
                "mcp_name": tool.mcp_name,
                "right": tool.right,
                "arguments_sha256": operation_context["arguments_sha256"],
                "arguments_preview": operation_context["arguments_preview"],
                "arguments_observation": operation_context["arguments_observation"],
                "sandbox_profile": operation_context.get("sandbox_profile"),
                **result_payload,
            },
            capability_refs=tuple(operation_context.get("capability_ids") or ()),
            effect_metadata=result_payload,
            provider_receipt={
                "response_bytes": result.response_bytes,
                "duration_s": result.duration_s,
            },
        )

    def _protected_call_failure_evidence(
        self,
        pid: str,
        resource: str,
        tool: McpToolSpec,
        operation_context: dict[str, Any],
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        result = McpCallResult(
            server_id=str(operation_context["server_id"]),
            tool_id=tool.tool_id,
            mcp_name=tool.mcp_name,
            status=McpCallStatus.TRANSPORT_ERROR,
            ok=False,
            error={"message": type(error).__name__, "phase": phase},
        )
        return self._protected_call_evidence(pid, resource, result, tool, operation_context)

    @staticmethod
    def _call_effect_result(result: McpCallResult) -> dict[str, Any]:
        payload = {
            "status": result.status.value,
            "ok": result.ok,
            "response_bytes": result.response_bytes,
            "duration_s": result.duration_s,
        }
        public_error = provider_error_envelope_from_mapping(result.error or {})
        if public_error is not None:
            payload["error"] = {
                key: public_error[key]
                for key in ("code", "error_type", "correlation_id")
            }
        return payload

    def _live_validation_failure_result(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        error: Exception,
        *,
        duration_s: float,
    ) -> McpCallResult:
        if isinstance(error, ProviderHostError):
            return McpCallResult(
                server_id=server.server_id,
                tool_id=tool.tool_id,
                mcp_name=tool.mcp_name,
                status=McpCallStatus.INVALID_RESPONSE,
                ok=False,
                error=error.to_dict(),
                duration_s=duration_s,
            )
        return self._failure(
            server,
            tool,
            McpCallStatus.INVALID_RESPONSE,
            f"MCP live tool metadata validation failed: {error}",
            McpProviderCallResult(
                error=type(error).__name__,
                duration_s=duration_s,
            ),
        )

    @staticmethod
    def _mcp_call_failure_resource(
        error: BaseException,
        phase: str,
        *,
        spec: McpServerSpec,
        request_bytes: int,
        list_request_bytes: int,
        resource_context: dict[str, Any],
        resource_progress: dict[str, int],
    ) -> ResourceSettlement | None:
        list_response_bytes = resource_progress["list_response_bytes"]
        if phase == "provider_call" and _provider_result_was_returned(error):
            return ResourceSettlement(
                usage=ResourceUsage(
                    mcp_request_bytes=list_request_bytes + request_bytes,
                    mcp_response_bytes=(
                        list_response_bytes + spec.max_response_bytes
                    ),
                ),
                source="primitive.mcp.call",
                context={
                    **resource_context,
                    "failure_phase": phase,
                    "list_response_bytes": list_response_bytes,
                    "unknown_response_bytes": spec.max_response_bytes,
                    "call_started": True,
                    "provider_result_returned": True,
                },
            )
        if not isinstance(error, ProviderEffectNotStarted):
            return None
        return ResourceSettlement(
            usage=ResourceUsage(
                mcp_request_bytes=list_request_bytes,
                mcp_response_bytes=list_response_bytes,
            ),
            source="primitive.mcp.call",
            context={
                **resource_context,
                "failure_phase": phase,
                "call_started": False,
                "certified_not_started": True,
            },
        )

    @staticmethod
    def _mcp_call_settlement(
        request_bytes: int,
        result: McpCallResult,
        *,
        server_id: str,
        tool_id: str,
        list_request_bytes: int = 0,
        list_response_bytes: int = 0,
        call_started: bool = True,
        unknown_response_bytes: int = 0,
    ) -> ResourceSettlement:
        call_request_bytes = request_bytes if call_started else 0
        return ResourceSettlement(
            usage=ResourceUsage(
                mcp_request_bytes=list_request_bytes + call_request_bytes,
                mcp_response_bytes=(
                    list_response_bytes
                    + result.response_bytes
                    + unknown_response_bytes
                ),
            ),
            source="primitive.mcp.call",
            context={
                "server_id": server_id,
                "tool_id": tool_id,
                "request_bytes": request_bytes,
                "response_bytes": result.response_bytes,
                "list_request_bytes": list_request_bytes,
                "list_response_bytes": list_response_bytes,
                "unknown_response_bytes": unknown_response_bytes,
                "call_started": call_started,
                "status": result.status.value,
            },
        )

    @classmethod
    def _mcp_live_validation_failure_settlement(
        cls,
        server: McpServerSpec,
        request_bytes: int,
        result: McpCallResult,
        *,
        server_id: str,
        tool_id: str,
        list_request_bytes: int,
        live_list_result: McpToolListResult | None,
    ) -> ResourceSettlement:
        list_response_bytes = (
            live_list_result.response_bytes if live_list_result is not None else 0
        )
        return cls._mcp_call_settlement(
            request_bytes,
            result,
            server_id=server_id,
            tool_id=tool_id,
            list_request_bytes=list_request_bytes,
            list_response_bytes=list_response_bytes,
            call_started=False,
            unknown_response_bytes=(
                server.max_response_bytes if live_list_result is None else 0
            ),
        )

    @classmethod
    def _mcp_legacy_call_completion_settlement(
        cls,
        server: McpServerSpec,
        request_bytes: int,
        result: McpCallResult,
        provider_result: McpProviderCallResult,
        classification_override: ExternalEffectClassification | None,
        *,
        server_id: str,
        tool_id: str,
        list_request_bytes: int,
        live_list_result: McpToolListResult | None,
    ) -> ResourceSettlement:
        list_response_bytes = (
            live_list_result.response_bytes if live_list_result is not None else 0
        )
        provider_outcome_unknown = (
            classification_override is not None and provider_result.error is not None
        )
        return cls._mcp_call_settlement(
            request_bytes,
            result,
            server_id=server_id,
            tool_id=tool_id,
            list_request_bytes=list_request_bytes,
            list_response_bytes=list_response_bytes,
            call_started=True,
            unknown_response_bytes=(
                server.max_response_bytes if provider_outcome_unknown else 0
            ),
        )

    @staticmethod
    def _mcp_exchange_settlement(
        result: McpProviderCallResult,
        *,
        fallback_list_request_bytes: int,
        fallback_call_request_bytes: int,
        server_id: str,
        tool_id: str,
        status: str,
    ) -> ResourceSettlement:
        list_request_bytes = result.list_request_bytes or fallback_list_request_bytes
        call_request_bytes = (
            result.call_request_bytes or fallback_call_request_bytes
            if result.call_started
            else 0
        )
        call_response_bytes = result.call_response_bytes or result.response_bytes
        return ResourceSettlement(
            usage=ResourceUsage(
                mcp_request_bytes=list_request_bytes + call_request_bytes,
                mcp_response_bytes=result.list_response_bytes + call_response_bytes,
            ),
            source="primitive.mcp.call",
            context={
                "server_id": server_id,
                "tool_id": tool_id,
                "list_request_bytes": list_request_bytes,
                "list_response_bytes": result.list_response_bytes,
                "call_request_bytes": call_request_bytes,
                "call_response_bytes": call_response_bytes,
                "call_started": result.call_started,
                "status": status,
            },
        )

    def _complete_expired_legacy_exchange(
        self,
        *,
        protected: Any,
        spec: McpServerSpec,
        tool: McpToolSpec,
        operation_context: dict[str, Any],
        effect_context: dict[str, Any],
        live_list_result: McpToolListResult | None,
        deadline: float,
        started: float,
        list_request_bytes: int,
        request_bytes: int,
    ) -> McpCallResult | None:
        if time.monotonic() < deadline:
            return None
        provider_result = McpProviderCallResult(
            error="MCP exchange deadline exhausted before tool dispatch",
            error_type="McpDeadlineExceeded",
            correlation_id=new_id("corr"),
            duration_s=time.monotonic() - started,
            list_request_bytes=list_request_bytes,
            list_response_bytes=(
                live_list_result.response_bytes
                if live_list_result is not None
                else 0
            ),
            call_request_bytes=request_bytes,
            call_started=False,
        )
        result = self._call_result_from_provider(spec, tool, provider_result)
        pid = str(operation_context["pid"])
        resource = self.tool_resource(spec.server_id, tool.tool_id)
        return protected.complete(
            result,
            self._protected_call_evidence(
                pid,
                resource,
                result,
                tool,
                operation_context,
            ),
            classification_context=effect_context,
            classification_result=self._call_effect_result(result),
            classification_override=ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={
                    "outcome": "deadline_exhausted_before_call",
                    "phase": "live_validation",
                },
            ),
            resource=self._mcp_exchange_settlement(
                provider_result,
                fallback_list_request_bytes=list_request_bytes,
                fallback_call_request_bytes=request_bytes,
                server_id=spec.server_id,
                tool_id=tool.tool_id,
                status=result.status.value,
            ),
        )

    def _list_tools_success_payload(self, result: McpToolListResult) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "ok",
            "response_bytes": result.response_bytes,
            "duration_s": result.duration_s,
            "tool_count": len(result.tools),
        }

    def _list_tools_failure_payload(self, exc: Exception, *, duration_s: float) -> dict[str, Any]:
        safe_error = (
            exc.to_dict()
            if isinstance(exc, ProviderHostError)
            else {
                "code": "mcp_provider_error",
                "error_type": type(exc).__name__,
                "correlation_id": new_id("corr"),
            }
        )
        return {
            "ok": False,
            "status": "transport_error",
            "response_bytes": 0,
            "duration_s": duration_s,
            "tool_count": 0,
            **safe_error,
            "error_observation": sanitize_for_observability(safe_error),
        }

    def _resource_usage_pid(self, actor: str | None) -> str | None:
        if actor is None:
            return None
        return actor if self.processes.get_process(actor) is not None else None

    def _list_tools_effect_context(self, server: McpServerSpec, *, request_bytes: int) -> dict[str, Any]:
        return {
            "server_id": server.server_id,
            "transport": server.transport,
            "rollback_class": ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED.value,
            "rollback_status": ExternalEffectRollbackStatus.NOT_REQUIRED.value,
            "state_mutation": False,
            "information_flow": True,
            "request_bytes": request_bytes,
        }

    def _operation_context(
        self,
        pid: str,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        registry_binding: dict[str, Any],
    ) -> dict[str, Any]:
        arguments_json = dumps(arguments)
        arguments_observation = sanitize_for_observability(
            {"arguments": arguments},
            preview_chars=self.config.mcp.audit_preview_chars,
        )
        return {
            "pid": pid,
            "primitive": "runtime.mcp.call",
            "operation": "mcp.call",
            "authority_operation": "mcp.call",
            "server_id": server.server_id,
            "transport": server.transport,
            "tool_id": tool.tool_id,
            "mcp_name": tool.mcp_name,
            "right": tool.right,
            **registry_binding,
            "arguments_sha256": hashlib.sha256(arguments_json.encode("utf-8")).hexdigest(),
            "arguments_preview": arguments_observation["preview"],
            "arguments_observation": arguments_observation,
        }

    def _visibility_operation_context(self, pid: str, server_id: str, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        arguments_json = dumps(arguments)
        return {
            "pid": pid,
            "primitive": "runtime.mcp.call",
            "operation": "mcp.call",
            "authority_operation": "mcp.call",
            "server_id": server_id,
            "tool_id": tool_id,
            "arguments_sha256": hashlib.sha256(arguments_json.encode("utf-8")).hexdigest(),
        }

    def _approval_constraints(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            AUTHORITY_RULES_KEY: [
                {
                    "rule_id": f"mcp.approval.{context['server_id']}.{context['tool_id']}",
                    "operation": "mcp.call",
                    "effect": CapabilityEffect.ALLOW.value,
                    "risk": "high",
                    "conditions": {
                        "server_id": context["server_id"],
                        "tool_id": context["tool_id"],
                        "registry_spec_sha256": context["registry_spec_sha256"],
                        "registry_generation": context["registry_generation"],
                        "arguments_sha256": context["arguments_sha256"],
                    },
                    "description": "one-shot human approval for exact MCP tool payload",
                }
            ]
        }

    @staticmethod
    def _server_spec_sha256(server: McpServerSpec) -> str:
        return hashlib.sha256(dumps(server).encode("utf-8")).hexdigest()

    def _registry_binding_context(self, server_id: str) -> dict[str, Any]:
        binding = self.extensions.get_mcp_registry_binding(server_id)
        if not isinstance(binding, dict):
            raise ValidationError("MCP registry binding must be an object")
        generation = binding.get("registry_generation")
        digest = binding.get("registry_spec_sha256")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValidationError("MCP registry generation is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValidationError("MCP registry spec digest is invalid")
        return {
            "registry_spec_sha256": digest,
            "registry_generation": generation,
        }

    def _registry_binding_for_server_spec(
        self,
        server: McpServerSpec,
    ) -> dict[str, Any]:
        binding = self._registry_binding_context(server.server_id)
        if binding["registry_spec_sha256"] != self._server_spec_sha256(server):
            raise CapabilityDenied("MCP server registry changed before call authorization")
        return binding

    def _protected_registry_guard(
        self,
        binding: dict[str, Any],
        server_id: str,
    ) -> dict[str, Any]:
        return {
            "provider_registry_binding": ProviderRegistryBinding.from_context(binding),
            "provider_registry_binding_resolver": lambda: ProviderRegistryBinding.from_context(
                self._registry_binding_context(server_id)
            ),
            "provider_registry_phase_guard": lambda: self._registry_phase_lock,
        }

    def _effect_context(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        operation_context: dict[str, Any],
        *,
        request_bytes: int,
    ) -> dict[str, Any]:
        return {
            "server_id": server.server_id,
            "transport": server.transport,
            "tool_id": tool.tool_id,
            "mcp_name": tool.mcp_name,
            "right": tool.right,
            "rollback_class": tool.rollback_class,
            "rollback_status": tool.rollback_status,
            "state_mutation": tool.state_mutation,
            "information_flow": tool.information_flow,
            "arguments_sha256": operation_context["arguments_sha256"],
            "arguments_observation": operation_context["arguments_observation"],
            "request_bytes": request_bytes,
        }

    def _coerce_server(self, value: McpServerSpec | dict[str, Any]) -> McpServerSpec:
        if isinstance(value, McpServerSpec):
            # Normalize typed and mapping/YAML inputs identically.  Runtime
            # dataclass annotations do not coerce ``timeout_s=1`` to ``1.0``;
            # leaving the typed value untouched would make the durable raw
            # spec and its decoded live model hash differently.
            value = to_jsonable(value)
        if isinstance(value, dict):
            _reject_unknown_fields(value, _SERVER_FIELDS, context="MCP server")
            transport = str(value.get("transport", "") or "").strip()
            server_id = self._required(value, "server_id", "MCP server")
            spec = McpServerSpec(
                schema_version=int(value.get("schema_version", 1)),
                server_id=str(server_id),
                transport=transport,
                # Preserve every supplied transport block through canonical
                # coercion. _validate_server owns the strict tagged-union
                # check; dropping the inactive block here would silently turn
                # an invalid typed or mapping manifest into a valid one.
                stdio=(
                    self._stdio_spec(value.get("stdio"))
                    if value.get("stdio") is not None
                    else None
                ),
                http=(
                    self._http_spec(value.get("http"))
                    if value.get("http") is not None
                    else None
                ),
                tools=[self._tool_spec(item) for item in list(value.get("tools") or [])],
                timeout_s=self._coerce_positive_float(value.get("timeout_s", self.config.mcp.timeout_s), "timeout_s"),
                max_request_bytes=self._coerce_positive_int(
                    value.get("max_request_bytes", self.config.mcp.max_request_bytes),
                    "max_request_bytes",
                ),
                max_response_bytes=self._coerce_positive_int(
                    value.get("max_response_bytes", self.config.mcp.max_response_bytes),
                    "max_response_bytes",
                ),
                metadata=dict(value.get("metadata") or {}),
            )
        else:
            raise ValidationError("MCP server must be an object")
        self._validate_server(spec)
        return spec

    def _stdio_spec(self, value: Any) -> McpStdioTransportSpec:
        if not isinstance(value, dict):
            raise ValidationError("MCP stdio transport requires stdio object")
        _reject_unknown_fields(value, _STDIO_FIELDS, context="MCP stdio")
        return McpStdioTransportSpec(
            command=str(value.get("command", "")),
            args=[str(item) for item in list(value.get("args") or [])],
            env={str(name): str(host_name) for name, host_name in dict(value.get("env") or {}).items()},
            cwd=str(value["cwd"]) if value.get("cwd") is not None else None,
        )

    def _http_spec(self, value: Any) -> McpHttpTransportSpec:
        if not isinstance(value, dict):
            raise ValidationError("MCP streamable_http transport requires http object")
        _reject_unknown_fields(value, _HTTP_FIELDS, context="MCP HTTP")
        return McpHttpTransportSpec(
            url=str(value.get("url", "")),
            headers=self._header_specs(value.get("headers") or {}),
        )

    def _tool_spec(self, value: Any) -> McpToolSpec:
        if not isinstance(value, dict):
            raise ValidationError("MCP tools entries must be objects")
        _reject_unknown_fields(value, _TOOL_FIELDS, context="MCP tool")
        return McpToolSpec(
            tool_id=str(self._required(value, "tool_id", "MCP tool")),
            mcp_name=str(self._required(value, "mcp_name", "MCP tool")),
            right=str(self._required(value, "right", "MCP tool")),
            rollback_class=str(self._required(value, "rollback_class", "MCP tool")),
            rollback_status=value.get("rollback_status"),
            state_mutation=self._coerce_bool(
                self._required(value, "state_mutation", "MCP tool"),
                "state_mutation",
            ),
            information_flow=self._coerce_bool(
                self._required(value, "information_flow", "MCP tool"),
                "information_flow",
            ),
            input_schema=dict(value.get("input_schema") or {}),
            metadata=dict(value.get("metadata") or {}),
        )

    def _validate_server(self, server: McpServerSpec) -> None:
        if server.schema_version != 1:
            raise ValidationError("MCP server schema_version must be 1")
        self._validate_identifier(server.server_id, "server_id", self.config.mcp.server_id_max_chars)
        if server.transport not in _TRANSPORTS:
            raise ValidationError("MCP transport must be stdio or streamable_http")
        if server.transport == "stdio":
            self._validate_stdio(server.stdio)
            if server.http is not None:
                raise ValidationError("MCP stdio server cannot include http configuration")
        if server.transport == "streamable_http":
            self._validate_http(server.http)
            if server.stdio is not None:
                raise ValidationError("MCP streamable_http server cannot include stdio configuration")
        if not server.tools:
            raise ValidationError("MCP server must declare at least one allowed tool")
        if server.timeout_s > self.config.mcp.timeout_hard_limit_s:
            raise ValidationError("MCP timeout_s exceeds configured hard limit")
        if server.max_request_bytes > self.config.mcp.max_request_hard_limit_bytes:
            raise ValidationError("MCP max_request_bytes exceeds configured hard limit")
        if server.max_response_bytes > self.config.mcp.max_response_hard_limit_bytes:
            raise ValidationError("MCP max_response_bytes exceeds configured hard limit")
        seen_tool_ids: set[str] = set()
        seen_mcp_names: set[str] = set()
        for tool in server.tools:
            self._validate_tool(tool)
            if tool.tool_id in seen_tool_ids:
                raise ValidationError(f"duplicate MCP tool_id: {tool.tool_id}")
            if tool.mcp_name in seen_mcp_names:
                raise ValidationError(f"duplicate MCP mcp_name: {tool.mcp_name}")
            seen_tool_ids.add(tool.tool_id)
            seen_mcp_names.add(tool.mcp_name)
        self._validate_json_value(server.metadata, "metadata")

    def _validate_stdio(self, stdio: McpStdioTransportSpec | None) -> None:
        if stdio is None:
            raise ValidationError("MCP stdio transport requires stdio configuration")
        command = stdio.command.strip()
        if not command:
            raise ValidationError("MCP stdio command must be non-empty")
        if command != stdio.command or any(char.isspace() for char in command) or any(char in command for char in "\r\n;&|<>"):
            raise ValidationError("MCP stdio command must be a single argv token, not a shell string")
        for arg in stdio.args:
            if not isinstance(arg, str) or "\x00" in arg:
                raise ValidationError("MCP stdio args must be strings without NUL bytes")
        for child_name, host_name in stdio.env.items():
            self._validate_env_name(child_name, "stdio env name")
            self._validate_env_name(host_name, "stdio env source")
            if not self._env_allowed(host_name, self.config.mcp.stdio_env_allowlist):
                raise ValidationError(f"MCP stdio env source is not allowlisted: {host_name}")
        if stdio.cwd is not None:
            raw = stdio.cwd.replace("\\", "/").strip()
            if not raw or PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
                raise ValidationError("MCP stdio cwd must be a non-empty relative path")
            parts: list[str] = []
            for part in raw.split("/"):
                if part in {"", "."}:
                    continue
                if part == "..":
                    if not parts:
                        raise ValidationError("MCP stdio cwd escapes workspace root")
                    parts.pop()
                    continue
                parts.append(part)

    def _validate_http(self, http: McpHttpTransportSpec | None) -> None:
        if http is None:
            raise ValidationError("MCP streamable_http transport requires http configuration")
        self._validate_url(http.url)
        for name, header in http.headers.items():
            self._validate_header_name(name)
            self._validate_env_name(header.env, f"header {name} env")
            if not self._env_allowed(header.env, self.config.mcp.header_env_allowlist):
                raise ValidationError(f"MCP header env is not allowlisted: {header.env}")
            if header.prefix not in _ALLOWED_HEADER_PREFIXES:
                raise ValidationError(f"MCP header {name} prefix is not allowed")
            if header.suffix not in _ALLOWED_HEADER_SUFFIXES:
                raise ValidationError(f"MCP header {name} suffix is not allowed")

    def _validate_tool(self, tool: McpToolSpec) -> None:
        self._validate_identifier(tool.tool_id, "tool_id", self.config.mcp.tool_id_max_chars)
        if not tool.mcp_name or len(tool.mcp_name) > self.config.mcp.mcp_name_max_chars:
            raise ValidationError("MCP mcp_name must be non-empty and within configured length")
        if tool.right not in _CALL_RIGHTS:
            raise ValidationError("MCP tool right must be read, write, or execute")
        try:
            rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
        except ValueError as exc:
            raise ValidationError("MCP rollback_class is invalid") from exc
        if tool.rollback_status is not None:
            try:
                ExternalEffectRollbackStatus(tool.rollback_status)
            except ValueError as exc:
                raise ValidationError("MCP rollback_status is invalid") from exc
        if rollback_class == ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED and tool.rollback_status is None:
            pass
        if rollback_class == ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED and tool.state_mutation:
            raise ValidationError("MCP tool with state_mutation=true cannot use no_rollback_required")
        self._validate_json_schema(tool.input_schema, "input_schema")
        self._validate_json_value(tool.metadata, "tool metadata")

    def _header_specs(self, value: Any) -> dict[str, McpHeaderSpec]:
        if not isinstance(value, dict):
            raise ValidationError("MCP headers must be an object")
        headers: dict[str, McpHeaderSpec] = {}
        for name, spec in value.items():
            if not isinstance(spec, dict):
                raise ValidationError(f"MCP header {name} must be an object")
            _reject_unknown_fields(
                spec,
                _HEADER_FIELDS,
                context=f"MCP header {name}",
            )
            headers[str(name)] = McpHeaderSpec(
                env=str(self._required(spec, "env", f"MCP header {name}")),
                prefix=str(spec.get("prefix", "")),
                suffix=str(spec.get("suffix", "")),
            )
        return headers

    def _required(self, value: dict[str, Any], key: str, context: str) -> Any:
        if key not in value:
            raise ValidationError(f"{context} requires {key}")
        return value[key]

    def _validate_header_name(self, name: str) -> None:
        lowered = name.lower()
        if len(name) > self.config.mcp.header_name_max_chars or not _HEADER_PATTERN.match(name):
            raise ValidationError(f"invalid MCP header name: {name!r}")
        if lowered in _FORBIDDEN_HEADERS:
            raise ValidationError(f"MCP header is forbidden: {name}")

    def _env_allowed(self, name: str, patterns: tuple[str, ...]) -> bool:
        for pattern in patterns:
            if pattern.endswith("*") and name.startswith(pattern[:-1]):
                return True
            if name == pattern:
                return True
        return False

    def _require_runtime_environment(
        self,
        server: McpServerSpec,
        *,
        host_environment: Mapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        selected_host_environment = (
            dict(os.environ)
            if host_environment is None
            else host_environment
        )
        return self._runtime_environment_from_host(
            server,
            selected_host_environment,
        )

    def _runtime_environment_from_host(
        self,
        server: McpServerSpec,
        host_environment: Mapping[str, str],
    ) -> Mapping[str, str]:
        resolved_environment: dict[str, str] = {}
        if server.transport == "stdio" and server.stdio is not None:
            # Windows needs these bootstrap variables to create a child
            # process.  Capture them at the primitive boundary alongside the
            # manifest-declared child environment so provider dispatch never
            # has to consult a newer ambient environment.
            for name in _MCP_PLATFORM_ENV_KEYS:
                resolved = host_environment.get(name)
                if resolved is None:
                    continue
                if "\x00" in resolved:
                    raise ValidationError(f"MCP stdio env {name} contains NUL byte")
                resolved_environment[name] = resolved
            for child_name, host_name in server.stdio.env.items():
                resolved = host_environment.get(host_name)
                if resolved is None:
                    raise ValidationError(f"missing environment variable for MCP stdio env {child_name}: {host_name}")
                if "\x00" in resolved:
                    raise ValidationError(f"MCP stdio env {child_name} contains NUL byte")
                resolved_environment[child_name] = resolved
        if server.transport == "streamable_http" and server.http is not None:
            for name, header in server.http.headers.items():
                resolved = host_environment.get(header.env)
                if resolved is None:
                    raise ValidationError(f"missing environment variable for MCP header {name}: {header.env}")
                header_value = f"{header.prefix}{resolved}{header.suffix}"
                if len(header_value) > self.config.mcp.header_value_max_chars or "\r" in header_value or "\n" in header_value:
                    raise ValidationError(f"MCP header {name} resolved value is invalid")
                resolved_environment[name] = header_value
        # One immutable snapshot spans all provider stages in this operation,
        # including legacy MCP providers that use separate tools/list and
        # call_tool sessions.  The SDK provider must not re-read os.environ.
        return MappingProxyType(resolved_environment)

    def _validate_url(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValidationError("MCP HTTP URL must use http or https")
        if parsed.username or parsed.password:
            raise ValidationError("MCP HTTP URL must not include userinfo")
        if parsed.fragment:
            raise ValidationError("MCP HTTP URL must not include a fragment")
        host = parsed.hostname
        if not host:
            raise ValidationError("MCP HTTP URL must include a host")
        if host.lower() in _FORBIDDEN_MCP_HOSTS:
            raise ValidationError("MCP HTTP host is not allowed")
        if parsed.scheme == "http" and host not in _LOCAL_HTTP_HOSTS:
            raise ValidationError("MCP plain HTTP is allowed only for local development hosts")
        self._validate_host_literal(host, allow_local=host in _LOCAL_HTTP_HOSTS)

    def _validate_runtime_resolution(self, server: McpServerSpec) -> tuple[str, ...]:
        if server.http is None:
            return ()
        parsed = urlsplit(server.http.url)
        host = parsed.hostname
        if not host:
            raise ValidationError("MCP HTTP URL must include a host")
        if host in _LOCAL_HTTP_HOSTS:
            return ()
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValidationError(f"MCP host could not be resolved: {host}") from exc
        addresses = sorted({info[4][0] for info in infos})
        if not addresses:
            raise ValidationError(f"MCP host resolved no addresses: {host}")
        for address in addresses:
            self._validate_host_literal(address, allow_local=False)
        return tuple(addresses)

    def _runtime_resolution_observes_host(self, server: McpServerSpec) -> bool:
        if server.http is None:
            return False
        host = urlsplit(server.http.url).hostname
        return bool(host and host.lower() not in _LOCAL_HTTP_HOSTS)

    def _validate_host_literal(self, host: str, *, allow_local: bool) -> None:
        try:
            ip = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            return
        if allow_local:
            return
        if not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValidationError("MCP HTTP IP address is not allowed")

    def _disable_replaced_server_tool_capabilities(self, server_id: str, *, actor: str) -> None:
        prefix = f"mcp:{server_id}:"
        for cap in self.authority.list_capabilities():
            if not cap.active or cap.revoked:
                continue
            if cap.resource == f"mcp:{server_id}:*" or cap.resource.startswith(prefix):
                self.capabilities.disable_subject_capability(
                    cap.cap_id,
                    actor=actor,
                    reason="MCP server spec replaced; tool authority must be reissued",
                )

    def _validate_identifier(self, value: str, field: str, max_chars: int) -> None:
        if not isinstance(value, str) or not value or len(value) > max_chars or not _ID_PATTERN.match(value):
            raise ValidationError(f"invalid MCP {field}: {value!r}")

    def _validate_env_name(self, value: str, field: str) -> None:
        if not value or not _ENV_PATTERN.match(value):
            raise ValidationError(f"invalid MCP {field}: {value!r}")

    def _coerce_bool(self, value: Any, field: str) -> bool:
        if not isinstance(value, bool):
            raise ValidationError(f"MCP {field} must be a boolean")
        return value

    def _coerce_positive_float(self, value: Any, field: str) -> float:
        try:
            selected = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"MCP {field} must be a number") from exc
        if not math.isfinite(selected) or selected <= 0:
            raise ValidationError(f"MCP {field} must be > 0")
        return selected

    def _coerce_positive_int(self, value: Any, field: str) -> int:
        if isinstance(value, bool):
            raise ValidationError(f"MCP {field} must be an integer")
        try:
            selected = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"MCP {field} must be an integer") from exc
        if selected <= 0:
            raise ValidationError(f"MCP {field} must be > 0")
        return selected

    def _validate_json_value(self, value: Any, field: str) -> None:
        try:
            dumps(value)
        except Exception as exc:
            raise ValidationError(f"MCP {field} must be JSON-serializable") from exc

    def _validate_json_schema(self, schema: dict[str, Any], field: str) -> None:
        if not schema:
            return
        try:
            jsonschema_validator_for(schema).check_schema(schema)
        except JsonSchemaSchemaError as exc:
            raise ValidationError(f"MCP {field} is not a valid JSON Schema") from exc

    def _validate_arguments_against_schema(self, tool: McpToolSpec, arguments: dict[str, Any]) -> None:
        if not tool.input_schema:
            return
        try:
            validator = jsonschema_validator_for(tool.input_schema)
            validator.check_schema(tool.input_schema)
            validator(tool.input_schema).validate(arguments)
        except JsonSchemaValidationError as exc:
            raise ValidationError(f"MCP tool arguments failed schema validation: {exc.message}") from exc
        except JsonSchemaSchemaError as exc:
            raise ValidationError("MCP tool input_schema is invalid") from exc

    def _bounded_list_limit(self, limit: int | None) -> int:
        selected = self.config.mcp.list_limit if limit is None else limit
        if not isinstance(selected, int):
            raise ValidationError("MCP server list limit must be an integer")
        if selected < 1:
            raise ValidationError("MCP server list limit must be >= 1")
        if selected > self.config.mcp.list_limit:
            raise ValidationError(f"MCP server list limit exceeds configured maximum {self.config.mcp.list_limit}")
        return selected

    def _load_server(self, server_id: str) -> tuple[McpServerSpec, dict[str, Any]]:
        self._validate_identifier(server_id, "server_id", self.config.mcp.server_id_max_chars)
        found = self.extensions.get_mcp_server(server_id)
        if found is None:
            raise NotFound(f"MCP server not found: {server_id}")
        spec, metadata = found
        self._validate_server(spec)
        return spec, metadata

    def _server_to_json(
        self,
        server: McpServerSpec,
        metadata: dict[str, Any],
        *,
        include_sensitive_fields: bool,
    ) -> dict[str, Any]:
        transport: dict[str, Any]
        if server.transport == "stdio" and server.stdio is not None:
            transport = {
                "type": "stdio",
                "command": server.stdio.command,
                "args": list(server.stdio.args),
                "env": {name: {"env": host_name} for name, host_name in server.stdio.env.items()},
                "cwd": server.stdio.cwd,
            }
        elif server.http is not None:
            transport = {
                "type": "streamable_http",
                "url": server.http.url if include_sensitive_fields else None,
                "headers": {
                    name: {
                        "env": header.env,
                        "prefix": header.prefix,
                        "suffix": header.suffix,
                    }
                    for name, header in server.http.headers.items()
                },
            }
        else:
            transport = {"type": server.transport}
        return {
            "schema_version": server.schema_version,
            "server_id": server.server_id,
            "transport": transport,
            "stdio_authority_resource": self.stdio_resource_for_server(server),
            "tools": [self._tool_to_json(server.server_id, tool) for tool in server.tools],
            "timeout_s": server.timeout_s,
            "max_request_bytes": server.max_request_bytes,
            "max_response_bytes": server.max_response_bytes,
            "metadata": server.metadata,
            **metadata,
        }

    def _tool_to_json(self, server_id: str, tool: McpToolSpec, *, live: McpProviderTool | None = None) -> dict[str, Any]:
        payload = {
            "tool_id": tool.tool_id,
            "mcp_name": tool.mcp_name,
            "right": tool.right,
            "resource": self.tool_resource(server_id, tool.tool_id),
            "rollback_class": tool.rollback_class,
            "rollback_status": tool.rollback_status,
            "state_mutation": tool.state_mutation,
            "information_flow": tool.information_flow,
            "input_schema": tool.input_schema,
            "metadata": tool.metadata,
        }
        if live is not None:
            payload["live"] = {
                "name": live.name,
                "description": live.description,
                "input_schema": live.input_schema,
                "schema_matches_manifest": not tool.input_schema or live.input_schema == tool.input_schema,
            }
        return payload

    def _profile_json(self, profile: Any) -> dict[str, Any]:
        return to_jsonable(profile)
