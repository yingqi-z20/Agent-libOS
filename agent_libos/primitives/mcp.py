from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import threading
import time
from collections import deque
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
    canonical_mcp_server_spec_json,
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
    McpConnectionInfo,
    McpDiscoveryResult,
    McpExchangePhase,
    McpExchangeReceipt,
    McpHeaderSpec,
    McpHttpTransportSpec,
    McpProtocolEra,
    McpProtocolMode,
    McpProviderCallResult,
    McpProviderDiscoveryResult,
    McpProviderTool,
    McpServerSpec,
    McpStdioTransportSpec,
    McpToolListResult,
    McpToolSpec,
    mcp_server_spec_to_jsonable,
    ResourceUsage,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProviderHostError,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.models.external_effect import default_external_effect_rollback_status
from agent_libos.models.mcp import mcp_runtime_secret_values
from agent_libos.ports import AuditPort, EventPort
from agent_libos.storage import UnitOfWork
from agent_libos.substrate import (
    ExecutableSnapshot,
    executable_content_sha256,
    McpModernProtocolProvider,
    McpProvider,
    McpSubprocessLimitsProvider,
    ProviderEffectNotStarted,
    SubprocessLimits,
    snapshot_executable,
)
from agent_libos.substrate.local import (
    _allowed_mcp_connect_addresses,
    _bounded_mcp_content,
    SdkMcpProvider,
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
from agent_libos.utils.redaction import redact_sensitive_text
from agent_libos.utils.serde import bounded_json_loads, dumps, to_jsonable
from agent_libos.utils.yaml_loader import load_yaml_mapping

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*$")
_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_LEGACY_FORBIDDEN_HEADERS = {
    "connection",
    "content-length",
    "host",
    "transfer-encoding",
    "upgrade",
}
_MODERN_FORBIDDEN_HEADERS = _LEGACY_FORBIDDEN_HEADERS | {
    "accept",
    "accept-charset",
    "accept-encoding",
    "accept-language",
    "baggage",
    "content-encoding",
    "content-language",
    "content-type",
    "last-event-id",
    "mcp-method",
    "mcp-name",
    "mcp-protocol-version",
    "mcp-session-id",
    "traceparent",
    "tracestate",
}
_FORBIDDEN_MCP_HOSTS = {"metadata.google.internal"}
_LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}
_CALL_RIGHTS = {CapabilityRight.READ.value, CapabilityRight.WRITE.value, CapabilityRight.EXECUTE.value}
_ALLOWED_HEADER_PREFIXES = {"", "Bearer ", "Token ", "Basic "}
_ALLOWED_HEADER_SUFFIXES = {""}
_TRANSPORTS = {"stdio", "streamable_http"}
_MCP_WINDOWS = os.name == "nt"
_MCP_WINDOWS_EXECUTABLE_SUFFIXES = {".com", ".exe"}
_MCP_PLATFORM_ENV_KEYS = ("SYSTEMROOT", "WINDIR") if os.name == "nt" else ()
_STDIO_EXECUTABLE_IDENTITY_UNSET = object()
_PROVIDER_RESULT_RETURNED_ATTR = "_agent_libos_provider_result_returned"
_INVALID_MCP_TEXT_JSON = object()
_MCP_PROVIDER_JSON_MAX_DEPTH = 128
_MCP_PROVIDER_JSON_MAX_NODES = 100_000
_MCP_STDIO_PROTOCOL_OUTPUT_MULTIPLIER = 4
_MCP_V2_DYNAMIC_REFERENCE_KEYS = {
    "$dynamicAnchor",
    "$dynamicRef",
    "$recursiveAnchor",
    "$recursiveRef",
}
_MCP_RELEASE_PROTOCOL_REVISIONS = {
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
    "2026-07-28",
}
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
    "protocol_mode",
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


def _wire_failure_evidence(
    error: BaseException,
) -> tuple[bool, tuple[McpExchangeReceipt, ...], McpConnectionInfo | None]:
    """Read inert, operation-local wire evidence from an exception chain."""

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    certified = False
    receipts: tuple[McpExchangeReceipt, ...] = ()
    connection: McpConnectionInfo | None = None
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        try:
            attributes = object.__getattribute__(current, "__dict__")
        except Exception:
            attributes = {}
        if attributes.get("_agent_libos_mcp_wire_evidence") is True:
            certified = True
            candidate = attributes.get("_agent_libos_mcp_receipts", ())
            if (
                type(candidate) is tuple
                and all(isinstance(item, McpExchangeReceipt) for item in candidate)
                and len(candidate) >= len(receipts)
            ):
                receipts = candidate
            candidate_connection = attributes.get("_agent_libos_mcp_connection")
            if isinstance(candidate_connection, McpConnectionInfo):
                connection = candidate_connection
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return certified, receipts, connection


class _McpLiveToolValidationError(ValidationError):
    def __init__(self, message: str, result: McpToolListResult) -> None:
        super().__init__(message)
        self.result = result


class _ProviderJsonBudget:
    """Bound normalization work for one provider-owned response tree."""

    __slots__ = (
        "max_depth",
        "max_nodes",
        "max_string_bytes",
        "nodes",
        "string_bytes",
    )

    def __init__(self, max_response_bytes: int) -> None:
        self.max_depth = _MCP_PROVIDER_JSON_MAX_DEPTH
        self.max_nodes = min(_MCP_PROVIDER_JSON_MAX_NODES, max(1, max_response_bytes))
        self.max_string_bytes = max_response_bytes
        self.nodes = 0
        self.string_bytes = 0

    def consume_node(self, *, path: str, depth: int) -> None:
        if depth > self.max_depth:
            raise TypeError(
                f"provider JSON exceeds maximum depth={self.max_depth} at {path}"
            )
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise TypeError(
                f"provider JSON exceeds maximum nodes={self.max_nodes} at {path}"
            )

    def consume_string(self, value: str, *, path: str) -> None:
        try:
            encoded_bytes = len(value.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise TypeError(f"provider JSON contains invalid UTF-8 at {path}") from error
        self.string_bytes += encoded_bytes
        if self.string_bytes > self.max_string_bytes:
            raise TypeError(
                "provider JSON exceeds maximum aggregate string bytes="
                f"{self.max_string_bytes} at {path}"
            )


def _strict_provider_json_value(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
    budget: _ProviderJsonBudget,
    depth: int = 0,
) -> Any:
    """Detach an exact JSON tree returned by a Host provider.

    Provider result objects are outside the runtime trust boundary.  In
    particular, attribute access, container iteration, and nested values may
    execute provider-owned code, so normalize the whole tree before any later
    evidence, accounting, or model-visible projection reads it.
    """

    budget.consume_node(path=path, depth=depth)
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        if value_type is str:
            budget.consume_string(value, path=path)
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
                budget.consume_string(key, path=f"{path}.<key>")
                selected[key] = _strict_provider_json_value(
                    item,
                    path=f"{path}[{key!r}]",
                    active_containers=active_containers,
                    budget=budget,
                    depth=depth + 1,
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
                    budget=budget,
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)
    raise TypeError(f"provider JSON contains a non-JSON value at {path}")


def _model_facing_mcp_call_payload(
    content: Any,
    structured_content: Any,
) -> dict[str, Any]:
    """Build a compact MCP result view without discarding distinct content.

    MCP servers commonly return the same object twice: once in
    ``structuredContent`` and once as JSON in a text content block.  Only a
    text-only block whose decoded JSON is exactly equivalent is removed.  All
    non-equivalent prose, annotations, and structured values remain visible.
    Binary content is projected to bounded receipts before equivalence checks,
    so base64 never survives merely because it was nested or JSON-encoded.
    """

    projected_structured = _project_mcp_model_value(structured_content)
    projected_content = _project_mcp_model_value(content)
    if structured_content is not None:
        projected_content = _drop_equivalent_mcp_content(
            projected_content,
            projected_structured,
        )
    return {
        "content": projected_content,
        "structured_content": projected_structured,
    }


def _project_mcp_model_value(value: Any) -> Any:
    projected = _bounded_mcp_content(value)
    if isinstance(projected, list):
        return [_project_mcp_text_json(item) for item in projected]
    return _project_mcp_text_json(projected)


def _project_mcp_text_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_project_mcp_text_json(item) for item in value]
    if not isinstance(value, dict):
        return value

    projected = {
        key: _project_mcp_text_json(item)
        for key, item in value.items()
    }
    if value.get("type") != "text" or not isinstance(value.get("text"), str):
        return projected
    decoded = _decode_mcp_text_json(value["text"])
    if decoded is _INVALID_MCP_TEXT_JSON:
        return projected
    decoded_projection = _project_mcp_model_value(decoded)
    if not _json_values_equivalent(decoded, decoded_projection):
        projected["text"] = dumps(decoded_projection)
    return projected


def _drop_equivalent_mcp_content(content: Any, structured_content: Any) -> Any:
    if _json_values_equivalent(content, structured_content):
        return None
    if isinstance(content, list):
        retained = [
            item
            for item in content
            if not _mcp_text_block_duplicates(item, structured_content)
        ]
        return retained
    if _mcp_text_block_duplicates(content, structured_content):
        return None
    return content


def _mcp_text_block_duplicates(value: Any, structured_content: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"type", "text"}:
        return False
    if value.get("type") != "text" or not isinstance(value.get("text"), str):
        return False
    decoded = _decode_mcp_text_json(value["text"])
    if decoded is _INVALID_MCP_TEXT_JSON:
        return False
    return _json_values_equivalent(
        _project_mcp_model_value(decoded),
        structured_content,
    )


def _decode_mcp_text_json(value: str) -> Any:
    try:
        return bounded_json_loads(value)
    except (TypeError, ValueError, RecursionError):
        return _INVALID_MCP_TEXT_JSON


def _json_values_equivalent(left: Any, right: Any) -> bool:
    try:
        return dumps(left).encode("utf-8") == dumps(right).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError):
        return False


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

    def discover(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> McpDiscoveryResult:
        """Perform one governed, operation-local MCP modern discovery."""

        try:
            return self._discover(
                server_id,
                actor=actor,
                require_capability=require_capability,
            )
        except ProviderEffectNotStarted as error:
            raise self._safe_not_started_error(error) from None

    def _discover(
        self,
        server_id: str,
        *,
        actor: str | None,
        require_capability: bool,
    ) -> McpDiscoveryResult:
        authority_decisions: list[Any] = []
        if require_capability and actor is not None:
            authority_decisions.append(
                self.capabilities.require(
                    actor,
                    self.server_resource(server_id),
                    CapabilityRight.READ,
                    consume=False,
                )
            )
        spec, _metadata = self._load_server(server_id)
        mode = self._effective_protocol_mode(spec)
        if spec.schema_version != 2 or mode is McpProtocolMode.LEGACY:
            raise ValidationError(
                "MCP discover requires Manifest v2 protocol_mode auto or 2026-07-28"
            )
        self._require_modern_protocol_provider(spec)
        if require_capability and actor is not None:
            authority_decisions.append(
                self.capabilities.require(
                    actor,
                    self.server_resource(server_id),
                    CapabilityRight.EXECUTE,
                    consume=False,
                )
            )
            authority_decisions.extend(
                self._require_stdio_process_spawn(actor, spec, consume=False)
            )
        effect_actor = actor or "runtime"
        usage_pid = self._resource_usage_pid(actor)
        request_bytes = len(
            dumps(
                {"method": "server/discover", "server_id": spec.server_id}
            ).encode("utf-8")
        )
        if request_bytes > spec.max_request_bytes:
            raise ValidationError(
                "MCP discover request exceeds "
                f"max_request_bytes={spec.max_request_bytes}"
            )
        effect_context = self._discover_effect_context(
            spec,
            request_bytes=request_bytes,
        )
        request_flow = (
            self._data_flow().current_context()
            if actor is not None
            else DataFlowContext(
                labels=DataLabels(
                    sensitivity="public",
                    trust_level="verified",
                    integrity="verified",
                    origin="runtime:mcp-discover-metadata",
                )
            )
        )
        runtime_environment = self._require_runtime_environment(spec)
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=runtime_environment,
        )
        sink = DataSink(
            f"mcp:{server_id}:discover",
            self._discover_identity_sha256(
                spec,
                stdio_executable=stdio_identity,
            ),
        )
        contract_name = (
            "primitive.mcp.discover"
            if authority_decisions
            else "primitive.mcp.discover.internal"
        )
        registry_binding = self._registry_binding_for_server_spec(spec)
        invocation = self._discover_invocation(
            server_id=server_id,
            spec=spec,
            effect_actor=effect_actor,
            authority_decisions=authority_decisions,
            usage_pid=usage_pid,
            request_bytes=request_bytes,
            effect_context=effect_context,
            request_flow=request_flow,
            runtime_environment=runtime_environment,
            sink=sink,
            registry_binding=registry_binding,
        )
        with self._protected().start(
            contract_name,
            invocation,
            provider=self.provider,
        ) as protected:
            deadline = time.monotonic() + spec.timeout_s
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
                    deadline=deadline,
                )
            executable_snapshot = self._stdio_snapshot_for_dispatch(
                pid=effect_actor,
                spec=spec,
                expected_identity=stdio_identity,
                sink=sink,
                context=request_flow,
                payload={"method": "server/discover", "server_id": server_id},
                runtime_environment=runtime_environment,
            )
            try:
                provider_result = protected.call(
                    ProviderPhase(
                        "provider_not_started_after_dns",
                        information_flow=True,
                    ),
                    self._invoke_discover_provider,
                    spec,
                    deadline=deadline,
                    pid=effect_actor,
                    executable_snapshot=executable_snapshot,
                    runtime_environment=runtime_environment,
                )
            finally:
                if executable_snapshot is not None:
                    executable_snapshot.close()
            if self._registry_binding_context(server_id) != registry_binding:
                raise CapabilityDenied(
                    "MCP server registry changed during protocol discovery"
                )
            result = McpDiscoveryResult(
                server_id=spec.server_id,
                connection=provider_result.connection,
                request_bytes=provider_result.request_bytes,
                response_bytes=provider_result.response_bytes,
                duration_s=provider_result.duration_s,
                receipts=provider_result.receipts,
            )
            result_payload = {
                "ok": True,
                "status": "ok",
                "request_bytes": result.request_bytes,
                "response_bytes": result.response_bytes,
                "duration_s": result.duration_s,
                "connection": to_jsonable(result.connection),
                "receipts": to_jsonable(result.receipts),
            }
            return protected.complete(
                result,
                self._protected_discover_evidence(
                    effect_actor,
                    spec,
                    effect_context,
                    result_payload,
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                resource=(
                    ResourceSettlement(
                        usage=ResourceUsage(
                            mcp_request_bytes=result.request_bytes,
                            mcp_response_bytes=result.response_bytes,
                        ),
                        source="primitive.mcp.discover",
                        context={
                            "server_id": server_id,
                            "request_bytes": result.request_bytes,
                            "response_bytes": result.response_bytes,
                            "protocol_revision": result.connection.protocol_revision,
                        },
                    )
                    if usage_pid is not None
                    else None
                ),
            )

    def _discover_invocation(
        self,
        *,
        server_id: str,
        spec: McpServerSpec,
        effect_actor: str,
        authority_decisions: list[Any],
        usage_pid: str | None,
        request_bytes: int,
        effect_context: dict[str, Any],
        request_flow: DataFlowContext,
        runtime_environment: Mapping[str, str],
        sink: DataSink,
        registry_binding: ProviderRegistryBinding,
    ) -> ProtectedOperationInvocation:
        return ProtectedOperationInvocation(
            pid=effect_actor,
            actor=effect_actor,
            target=self.server_resource(server_id),
            decisions=tuple(authority_decisions),
            canonical_args=effect_context,
            observation=effect_context,
            reservation_usage=(
                ResourceUsage(
                    mcp_request_bytes=spec.max_request_bytes,
                    mcp_response_bytes=spec.max_response_bytes,
                )
                if usage_pid is not None
                else None
            ),
            resource_source="primitive.mcp.discover",
            resource_context={
                "server_id": server_id,
                "request_bytes": request_bytes,
            },
            **self._protected_registry_guard(
                registry_binding,
                server_id,
            ),
            data_sink=sink,
            data_sink_revalidator=lambda: self._discover_data_sink(
                server_id,
                spec,
                runtime_environment,
            ),
            data_flow_context=request_flow,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                request_flow,
                origin="external:mcp",
            ),
            data_flow_payload={"method": "server/discover", "server_id": server_id},
            data_flow_operation="mcp.discover",
            failure_evidence=lambda error, phase: self._protected_discover_failure_evidence(
                effect_actor,
                spec,
                effect_context,
                error,
                phase,
            ),
        )

    async def adiscover(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> McpDiscoveryResult:
        return await self._data_flow().run_sync_in_worker(
            self.discover,
            server_id,
            actor=actor,
            require_capability=require_capability,
        )

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
        if refresh:
            # Manifest v2 is an explicit provider-contract boundary, including
            # its legacy mode. Fence an unmarked provider before the refresh
            # path can resolve DNS, snapshot/spawn stdio, or dispatch I/O.
            self._require_modern_protocol_provider(spec)
        live_by_name: dict[str, McpProviderTool] = {}
        live_response_bytes = 0
        live_connection: McpConnectionInfo | None = None
        live_receipts: tuple[McpExchangeReceipt, ...] = ()
        if refresh:
            result = self._refresh_list_tools(
                server_id,
                actor=actor,
                require_capability=require_capability,
                spec=spec,
                authority_decisions=authority_decisions,
            )
            live_response_bytes = result.response_bytes
            live_connection = result.connection
            live_receipts = result.receipts
            live_by_name = {tool.name: tool for tool in result.tools}
        return {
            "server_id": spec.server_id,
            "schema_version": spec.schema_version,
            "transport": spec.transport,
            "protocol_mode": self._effective_protocol_mode(spec).value,
            "tools": [
                self._tool_to_json(spec.server_id, tool, live=live_by_name.get(tool.mcp_name) if refresh else None)
                for tool in spec.tools
            ],
            "refreshed": refresh,
            "response_bytes": live_response_bytes,
            **(
                {
                    "connection": to_jsonable(live_connection),
                    "receipts": to_jsonable(live_receipts),
                }
                if live_connection is not None
                else {}
            ),
        }

    def _refresh_list_tools(
        self,
        server_id: str,
        *,
        actor: str | None,
        require_capability: bool,
        spec: McpServerSpec,
        authority_decisions: list[Any],
    ) -> McpToolListResult:
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
            authority_decisions.extend(
                self._require_stdio_process_spawn(actor, spec, consume=False)
            )
        runtime_environment = self._require_runtime_environment(spec)
        request_payload = {"method": "tools/list", "server_id": spec.server_id}
        request_bytes = len(dumps(request_payload).encode("utf-8"))
        if request_bytes > spec.max_request_bytes:
            raise ValidationError(
                f"MCP list_tools request exceeds max_request_bytes={spec.max_request_bytes}"
            )
        effect_context = self._list_tools_effect_context(
            spec,
            request_bytes=request_bytes,
        )
        resource_context = {"server_id": server_id, "request_bytes": request_bytes}
        request_flow = self._list_tools_request_flow(actor)
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
        invocation = self._list_tools_invocation(
            server_id=server_id,
            effect_actor=effect_actor,
            spec=spec,
            authority_decisions=authority_decisions,
            effect_context=effect_context,
            request_bytes=request_bytes,
            resource_context=resource_context,
            usage_pid=usage_pid,
            runtime_environment=runtime_environment,
            request_flow=request_flow,
            list_sink=list_sink,
            request_payload=request_payload,
        )
        contract_name = (
            "primitive.mcp.list_tools"
            if authority_decisions
            else "primitive.mcp.list_tools.internal"
        )
        with self._protected().start(
            contract_name,
            invocation,
            provider=self.provider,
        ) as protected:
            started = time.monotonic()
            result, provider_error = self._dispatch_list_tools(
                protected,
                spec,
                deadline=started + spec.timeout_s,
                pid=effect_actor,
                expected_identity=stdio_executable_identity,
                sink=list_sink,
                context=request_flow,
                payload=request_payload,
                runtime_environment=runtime_environment,
            )
            if provider_error is not None:
                self._complete_list_tools_failure(
                    protected,
                    provider_error,
                    effect_actor=effect_actor,
                    spec=spec,
                    effect_context=effect_context,
                    resource_context=resource_context,
                    request_bytes=request_bytes,
                    usage_pid=usage_pid,
                    started=started,
                )
                raise provider_error
            self._complete_list_tools_success(
                protected,
                result,
                effect_actor=effect_actor,
                spec=spec,
                effect_context=effect_context,
                resource_context=resource_context,
                request_bytes=request_bytes,
                usage_pid=usage_pid,
            )
            return result

    def _list_tools_request_flow(self, actor: str | None) -> DataFlowContext:
        if actor is not None:
            return self._data_flow().current_context()
        return DataFlowContext(
            labels=DataLabels(
                sensitivity="public",
                trust_level="verified",
                integrity="verified",
                origin="runtime:mcp-list-tools-metadata",
            )
        )

    def _list_tools_invocation(
        self,
        *,
        server_id: str,
        effect_actor: str,
        spec: McpServerSpec,
        authority_decisions: list[Any],
        effect_context: dict[str, Any],
        request_bytes: int,
        resource_context: dict[str, Any],
        usage_pid: str | None,
        runtime_environment: Mapping[str, str],
        request_flow: DataFlowContext,
        list_sink: DataSink,
        request_payload: dict[str, str],
    ) -> ProtectedOperationInvocation:
        reservation_usage = None
        if usage_pid is not None:
            reservation_usage = ResourceUsage(
                mcp_request_bytes=(
                    spec.max_request_bytes
                    if spec.schema_version == 2
                    else request_bytes
                ),
                mcp_response_bytes=spec.max_response_bytes,
            )
        return ProtectedOperationInvocation(
            pid=effect_actor,
            actor=effect_actor,
            target=self.server_resource(spec.server_id),
            decisions=tuple(authority_decisions),
            canonical_args=effect_context,
            observation=effect_context,
            reservation_usage=reservation_usage,
            resource_source="primitive.mcp.list_tools",
            resource_context=resource_context,
            **self._protected_registry_guard(
                self._registry_binding_for_server_spec(spec),
                server_id,
            ),
            data_sink=list_sink,
            data_sink_revalidator=lambda: self._list_tools_data_sink(
                server_id,
                spec,
                runtime_environment,
            ),
            data_flow_context=request_flow,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                request_flow,
                origin="external:mcp",
            ),
            data_flow_payload=request_payload,
            data_flow_operation="mcp.list_tools",
            failure_evidence=lambda error, phase: self._protected_list_failure_evidence(
                effect_actor,
                spec,
                effect_context,
                error,
                phase,
            ),
        )

    def _complete_list_tools_failure(
        self,
        protected: Any,
        provider_error: Exception,
        *,
        effect_actor: str,
        spec: McpServerSpec,
        effect_context: dict[str, Any],
        resource_context: dict[str, Any],
        request_bytes: int,
        usage_pid: str | None,
        started: float,
    ) -> None:
        result_payload = self._list_tools_failure_payload(
            provider_error,
            duration_s=time.monotonic() - started,
        )
        settlement = None
        if usage_pid is not None:
            settlement = ResourceSettlement(
                usage=ResourceUsage(mcp_request_bytes=request_bytes),
                source="primitive.mcp.list_tools",
                context={
                    **resource_context,
                    "response_bytes": 0,
                    "status": result_payload["status"],
                },
                charge_reserved_maximum=True,
            )
        protected.complete(
            result_payload,
            self._protected_list_evidence(
                effect_actor,
                spec,
                effect_context,
                result_payload,
            ),
            classification_context=effect_context,
            classification_result=result_payload,
            resource=settlement,
        )

    def _complete_list_tools_success(
        self,
        protected: Any,
        result: McpToolListResult,
        *,
        effect_actor: str,
        spec: McpServerSpec,
        effect_context: dict[str, Any],
        resource_context: dict[str, Any],
        request_bytes: int,
        usage_pid: str | None,
    ) -> None:
        result_payload = self._list_tools_success_payload(result)
        settlement = None
        if usage_pid is not None:
            settled_request_bytes = request_bytes
            settled_response_bytes = result.response_bytes
            if spec.schema_version == 2 and result.receipts:
                settled_request_bytes = sum(item.request_bytes for item in result.receipts)
                settled_response_bytes = sum(item.response_bytes for item in result.receipts)
            settlement = ResourceSettlement(
                usage=ResourceUsage(
                    mcp_request_bytes=settled_request_bytes,
                    mcp_response_bytes=settled_response_bytes,
                ),
                source="primitive.mcp.list_tools",
                context={
                    **resource_context,
                    "response_bytes": result.response_bytes,
                },
            )
        protected.complete(
            result,
            self._protected_list_evidence(
                effect_actor,
                spec,
                effect_context,
                result_payload,
            ),
            classification_context=effect_context,
            classification_result=result_payload,
            resource=settlement,
        )

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
        (
            resource,
            selected_args,
            spec,
            tool,
            registry_binding,
            operation_context,
            flow_context,
            sink,
            decision,
            auxiliary_decisions,
            stdio_identity,
            stdio_target_environment,
        ) = self._prepare_tool_call(
            pid,
            server_id=server_id,
            tool_id=tool_id,
            arguments=arguments,
            source_oids=source_oids,
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
        runtime_environment: Mapping[str, str] | None = None

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
                mcp_request_bytes=(
                    spec.max_request_bytes
                    if spec.schema_version == 2
                    else list_request_bytes + request_bytes
                ),
                mcp_response_bytes=(
                    spec.max_response_bytes
                    if spec.schema_version == 2
                    else spec.max_response_bytes * 2
                ),
            ),
            resource_source="primitive.mcp.call",
            resource_context=resource_context,
            **self._protected_registry_guard(registry_binding, server_id),
            failure_resource=failure_resource,
            failure_evidence=lambda error, phase: self._protected_call_failure_evidence(pid, resource, tool, operation_context, error, phase),
            data_sink=sink,
            data_sink_revalidator=lambda: self._tool_data_sink_after_runtime_resolution(
                server_id, spec, tool, runtime_environment, expected=sink
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
            runtime_environment = self._require_runtime_environment(
                spec,
                pinned_stdio_environment=stdio_target_environment,
            )
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
                    deadline=deadline,
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

                invoke_validated_tool = partial(
                    self._invoke_validated_provider_tool,
                    validate_and_call,
                    spec,
                    tool,
                    selected_args,
                    deadline=deadline,
                    pid=pid,
                    runtime_environment=runtime_environment,
                    executable_snapshot=executable_snapshot,
                    started=started,
                )
                provider_outcome, wire_bound_combined_provider = self._dispatch_validated_provider_call(
                    protected,
                    invoke_validated_tool,
                    server=spec,
                    tool=tool,
                    executable_snapshot=executable_snapshot,
                )
                if isinstance(provider_outcome, ProviderEffectNotStartedResult):
                    return self._call_result_from_provider(
                        spec,
                        tool,
                        provider_outcome.result,
                    )
                provider_result = provider_outcome
                result = self._call_result_from_provider(spec, tool, provider_result)
                classification_override = self._pre_call_failure_classification_override(
                    spec,
                    tool,
                    provider_result,
                    wire_bound_combined_provider=wire_bound_combined_provider,
                )
                post_call_override = self._post_call_failure_classification_override(
                    tool,
                    provider_result,
                )
                if post_call_override is not None:
                    classification_override = post_call_override
                input_required_override = self._input_required_classification_override(
                    tool,
                    provider_result,
                )
                if input_required_override is not None:
                    classification_override = input_required_override
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
                        pid=pid,
                        runtime_environment=runtime_environment,
                        executable_snapshot=executable_snapshot,
                    )
                    raw_result = self.provider.call_tool(
                        spec,
                        tool,
                        selected_args,
                        **provider_kwargs,
                    )
                    return (
                        self._validated_provider_call_result(
                            spec,
                            raw_result,
                            runtime_environment=runtime_environment,
                        ),
                        None,
                    )
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
            input_required_override = self._input_required_classification_override(
                tool,
                provider_result,
            )
            if input_required_override is not None:
                classification_override = input_required_override
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

    def _prepare_tool_call(
        self,
        pid: str,
        *,
        server_id: str,
        tool_id: str,
        arguments: Any,
        source_oids: list[str] | tuple[str, ...] | None,
    ) -> tuple[
        str,
        dict[str, Any],
        McpServerSpec,
        McpToolSpec,
        dict[str, Any],
        dict[str, Any],
        DataFlowContext,
        DataSink,
        Any,
        list[Any],
        dict[str, str] | None,
        Mapping[str, str],
    ]:
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
        (
            sink,
            decision,
            auxiliary_decisions,
            stdio_identity,
            stdio_target_environment,
        ) = self._authorize_and_resolve_call_sink(
            pid=pid,
            server_id=server_id,
            resource=resource,
            spec=spec,
            tool=tool,
            arguments=selected_args,
            operation_context=operation_context,
            flow_context=flow_context,
            source_oids=source_oids,
        )
        profile = self.capabilities.profiles.mcp(
            resource=resource,
            effect=decision.effect or CapabilityEffect.DENY,
            server_id=server_id,
            tool_id=tool_id,
        )
        self._attach_call_authority_context(
            operation_context,
            decision=decision,
            auxiliary_decisions=auxiliary_decisions,
            profile=profile,
        )
        return (
            resource,
            selected_args,
            spec,
            tool,
            registry_binding,
            operation_context,
            flow_context,
            sink,
            decision,
            auxiliary_decisions,
            stdio_identity,
            stdio_target_environment,
        )

    def _attach_call_authority_context(
        self,
        operation_context: dict[str, Any],
        *,
        decision: Any,
        auxiliary_decisions: list[Any],
        profile: Any,
    ) -> None:
        capability_ids = [
            *decision.matched_capability_ids,
            *[
                cap_id
                for auxiliary in auxiliary_decisions
                for cap_id in auxiliary.matched_capability_ids
            ],
        ]
        operation_context.update(
            {
                "capability_ids": list(dict.fromkeys(capability_ids)),
                "selected_capability_id": decision.selected_capability_id,
                "sandbox_profile": self._profile_json(profile),
            }
        )

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
        # All Manifest v2 modes require the modern provider SPI, even when the
        # configured wire mode is ``legacy``. Keep this check ahead of sink
        # resolution and every DNS, stdio, or provider dispatch path.
        self._require_modern_protocol_provider(spec)
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
            request_id = self.human.query_authority_request(
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
                authority_origin="external_operation",
                source_oids=source_oids,
            )
            raise HumanApprovalRequired(
                request_id=request_id,
                message=f"{pid} is waiting for per-use human approval to call {resource}",
            )
        raise CapabilityDenied(decision.reason)

    def _authorize_and_resolve_call_sink(
        self,
        *,
        pid: str,
        server_id: str,
        resource: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        operation_context: dict[str, Any],
        flow_context: DataFlowContext,
        source_oids: list[str] | tuple[str, ...] | None,
    ) -> tuple[DataSink, Any, list[Any], dict[str, str] | None, Mapping[str, str]]:
        precheck_sink = self._tool_data_sink_for_clearance_precheck(
            server_id,
            spec,
            tool,
        )
        self._data_flow().precheck_egress_clearance(
            pid=pid,
            sink=precheck_sink,
            context=flow_context,
            payload=arguments,
        )
        # A precheck result is never authority: exact ordinary authority and
        # executable-bound authorize_egress below always run independently.
        decision = self._authorize_call(
            pid,
            resource,
            tool.right,
            operation_context,
            source_oids=source_oids,
        )
        auxiliary_decisions = self._require_stdio_process_spawn(
            pid,
            spec,
            consume=False,
        )
        self._validate_arguments_against_schema(spec, tool, arguments)
        stdio_environment = self._stdio_executable_resolution_environment(spec)
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=stdio_environment,
        )
        sink = self._tool_data_sink_from_stdio_identity(
            server_id,
            spec,
            tool,
            stdio_identity,
        )
        self._data_flow().authorize_egress(
            pid=pid,
            sink=sink,
            context=flow_context,
            payload=arguments,
            operation="mcp.call_tool",
        )
        return (
            sink,
            decision,
            auxiliary_decisions,
            stdio_identity,
            stdio_environment,
        )

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
        request_id = self.human.query_authority_request(
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
            authority_origin="external_operation",
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
        pid: str | None = None,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpToolListResult:
        selected_timeout = server.timeout_s if timeout_s is None else timeout_s
        provider_kwargs = self._provider_dispatch_kwargs(
            server,
            deadline=time.monotonic() + selected_timeout,
            pid=pid,
            runtime_environment=runtime_environment,
            executable_snapshot=executable_snapshot,
        )
        try:
            raw_result = self.provider.list_tools(
                server,
                **provider_kwargs,
            )
            result = self._validated_tool_list_result(
                server,
                raw_result,
                runtime_environment=runtime_environment,
            )
            live = next(
                (item for item in result.tools if item.name == tool.mcp_name),
                None,
            )
            if live is None:
                raise _McpLiveToolValidationError(
                    f"MCP server {server.server_id} no longer exposes tool {tool.mcp_name}",
                    result,
                )
            if tool.input_schema and not _json_values_equivalent(
                live.input_schema,
                tool.input_schema,
            ):
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

    def _invoke_discover_provider(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        pid: str,
        executable_snapshot: ExecutableSnapshot | None,
        runtime_environment: Mapping[str, str],
    ) -> McpProviderDiscoveryResult:
        self._require_modern_protocol_provider(server)
        provider_kwargs = self._provider_dispatch_kwargs(
            server,
            deadline=deadline,
            pid=pid,
            runtime_environment=runtime_environment,
            executable_snapshot=executable_snapshot,
        )
        try:
            raw_result = self.provider.discover(server, **provider_kwargs)  # type: ignore[attr-defined]
            return self._validated_discovery_result(
                server,
                raw_result,
                runtime_environment=runtime_environment,
            )
        except ProviderEffectNotStarted:
            raise
        except ProviderHostError:
            raise
        except Exception as error:
            raise ProviderHostError(
                code="mcp_provider_error",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            ) from None

    def _validated_discovery_result(
        self,
        server: McpServerSpec,
        result: Any,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpProviderDiscoveryResult:
        try:
            if not isinstance(result, McpProviderDiscoveryResult):
                raise TypeError("MCP provider returned an invalid discovery result")
            self._validate_discovery_result_header(server, result)
            connection = self._validated_connection_info(
                server,
                result.connection,
                runtime_environment=runtime_environment,
            )
            if connection is None:  # pragma: no cover - schema v2 invariant
                raise TypeError("MCP discovery connection metadata is missing")
            receipts = self._validated_exchange_receipts(server, result.receipts)
            self._validate_discovery_receipts(
                server,
                result,
                connection,
                receipts,
            )
            return McpProviderDiscoveryResult(
                connection=connection,
                request_bytes=result.request_bytes,
                response_bytes=result.response_bytes,
                duration_s=float(result.duration_s),
                receipts=receipts,
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
    def _validate_discovery_result_header(
        server: McpServerSpec,
        result: McpProviderDiscoveryResult,
    ) -> None:
        if type(result.request_bytes) is not int or result.request_bytes < 0:
            raise TypeError("MCP discovery request_bytes is invalid")
        if type(result.response_bytes) is not int or result.response_bytes < 0:
            raise TypeError("MCP discovery response_bytes is invalid")
        if result.request_bytes > server.max_request_bytes:
            raise TypeError("MCP discovery exceeds cumulative request budget")
        if result.response_bytes > server.max_response_bytes:
            raise TypeError("MCP discovery exceeds cumulative response budget")
        if (
            type(result.duration_s) not in {int, float}
            or not math.isfinite(result.duration_s)
            or result.duration_s < 0
        ):
            raise TypeError("MCP discovery duration_s is invalid")

    def _validate_discovery_receipts(
        self,
        server: McpServerSpec,
        result: McpProviderDiscoveryResult,
        connection: McpConnectionInfo,
        receipts: tuple[McpExchangeReceipt, ...],
    ) -> None:
        negotiation_end = self._validated_v2_negotiation_prefix(
            server,
            connection,
            receipts,
        )
        if negotiation_end != len(receipts):
            raise TypeError("MCP discovery returned a non-negotiation phase receipt")
        request_total = sum(item.request_bytes for item in receipts)
        response_total = sum(item.response_bytes for item in receipts)
        if result.request_bytes != request_total:
            raise TypeError("MCP discovery request bytes do not match phase receipts")
        if result.response_bytes != response_total:
            raise TypeError("MCP discovery response bytes do not match phase receipts")

    def _invoke_list_tools_provider(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        pid: str,
        executable_snapshot: ExecutableSnapshot | None,
        runtime_environment: Mapping[str, str],
    ) -> tuple[McpToolListResult | None, ProviderHostError | None]:
        try:
            provider_kwargs = self._provider_dispatch_kwargs(
                server,
                deadline=deadline,
                pid=pid,
                runtime_environment=runtime_environment,
                executable_snapshot=executable_snapshot,
            )
            raw_result = self.provider.list_tools(server, **provider_kwargs)
            return (
                self._validated_tool_list_result(
                    server,
                    raw_result,
                    runtime_environment=runtime_environment,
                ),
                None,
            )
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
                deadline=deadline,
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
                pid=pid,
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
        *,
        runtime_environment: Mapping[str, str] | None = None,
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
            budget = _ProviderJsonBudget(server.max_response_bytes)
            selected_tools = self._validated_provider_tools(
                tools,
                budget=budget,
                max_tools=self.config.mcp.list_limit,
            )
            canonical_response_bytes = self._canonical_provider_json_bytes(
                [to_jsonable(tool) for tool in selected_tools],
                context="MCP tools/list response",
            )
            if canonical_response_bytes > server.max_response_bytes:
                raise TypeError("MCP tools/list canonical response exceeds max_response_bytes")
            if (
                server.schema_version == 1
                and response_bytes < canonical_response_bytes
            ):
                raise TypeError("MCP tools/list response_bytes underreports canonical response")
            connection = self._validated_connection_info(
                server,
                result.connection,
                runtime_environment=runtime_environment,
            )
            receipts = self._validated_exchange_receipts(
                server,
                result.receipts,
            )
            if server.schema_version == 2:
                if connection is None:  # pragma: no cover - required above
                    raise TypeError("MCP Manifest v2 tools/list connection is missing")
                self._validate_v2_list_receipts(
                    server,
                    connection,
                    receipts,
                    response_bytes=response_bytes,
                )
            else:
                if any(
                    receipt.phase
                    not in {
                        McpExchangePhase.SERVER_DISCOVER,
                        McpExchangePhase.INITIALIZE,
                        McpExchangePhase.TOOLS_LIST,
                    }
                    for receipt in receipts
                ):
                    raise TypeError("MCP tools/list returned an unrelated phase receipt")
                if sum(
                    receipt.phase is McpExchangePhase.TOOLS_LIST
                    for receipt in receipts
                ) > getattr(self, "config", DEFAULT_CONFIG).mcp.list_max_pages:
                    raise TypeError(
                        "MCP tools/list exceeds maximum page receipts="
                        f"{getattr(self, 'config', DEFAULT_CONFIG).mcp.list_max_pages}"
                    )
            return McpToolListResult(
                server_id=server_id,
                tools=selected_tools,
                response_bytes=response_bytes,
                duration_s=float(duration_s),
                connection=connection,
                receipts=receipts,
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

    def _validated_connection_info(
        self,
        server: McpServerSpec,
        connection: Any,
        *,
        required: bool | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpConnectionInfo | None:
        if connection is None:
            if required is None:
                required = server.schema_version == 2
            if required:
                raise TypeError(
                    "MCP Manifest v2 provider result is missing connection metadata"
                )
            return None
        if not isinstance(connection, McpConnectionInfo):
            raise TypeError("MCP provider connection metadata is invalid")
        mode, era, revision = self._validated_connection_protocol(server, connection)
        self._validate_connection_server_identity(connection)
        sensitive_values = mcp_runtime_secret_values(server, runtime_environment)
        server_name = (
            redact_sensitive_text(
                connection.server_name,
                sensitive_values=sensitive_values,
            )
            if connection.server_name is not None
            else None
        )
        server_version = (
            redact_sensitive_text(
                connection.server_version,
                sensitive_values=sensitive_values,
            )
            if connection.server_version is not None
            else None
        )
        capabilities = self._validated_connection_names(
            connection.capabilities,
            field="capabilities",
            sensitive_values=sensitive_values,
        )
        unsupported = self._validated_connection_names(
            connection.unsupported_capabilities,
            field="unsupported_capabilities",
            sensitive_values=sensitive_values,
        )
        return McpConnectionInfo(
            protocol_mode=mode,
            protocol_era=era,
            protocol_revision=revision,
            sessionless=connection.sessionless,
            fallback_used=connection.fallback_used,
            server_name=server_name,
            server_version=server_version,
            capabilities=capabilities,
            unsupported_capabilities=unsupported,
        )

    def _validated_connection_protocol(
        self,
        server: McpServerSpec,
        connection: McpConnectionInfo,
    ) -> tuple[McpProtocolMode, McpProtocolEra, str]:
        mode = connection.protocol_mode
        era = connection.protocol_era
        revision = connection.protocol_revision
        if type(mode) is not McpProtocolMode or mode != self._effective_protocol_mode(server):
            raise TypeError("MCP provider protocol mode is invalid")
        if type(era) is not McpProtocolEra:
            raise TypeError("MCP provider protocol era is invalid")
        if type(revision) is not str or revision not in _MCP_RELEASE_PROTOCOL_REVISIONS:
            raise TypeError("MCP provider protocol revision is invalid")
        if (
            type(connection.sessionless) is not bool
            or type(connection.fallback_used) is not bool
        ):
            raise TypeError("MCP provider connection flags are invalid")
        self._validate_connection_protocol_semantics(
            mode,
            era,
            revision,
            sessionless=connection.sessionless,
            fallback_used=connection.fallback_used,
        )
        return mode, era, revision

    def _validate_connection_protocol_semantics(
        self,
        mode: McpProtocolMode,
        era: McpProtocolEra,
        revision: str,
        *,
        sessionless: bool,
        fallback_used: bool,
    ) -> None:
        if era is McpProtocolEra.MODERN:
            self._validate_modern_connection_flags(
                revision,
                sessionless=sessionless,
                fallback_used=fallback_used,
            )
        else:
            self._validate_legacy_connection_flags(
                mode,
                revision,
                sessionless=sessionless,
                fallback_used=fallback_used,
            )
        if mode is McpProtocolMode.LEGACY and era is not McpProtocolEra.LEGACY:
            raise TypeError("MCP forced legacy mode negotiated modern")

    @staticmethod
    def _validate_modern_connection_flags(
        revision: str,
        *,
        sessionless: bool,
        fallback_used: bool,
    ) -> None:
        if revision != McpProtocolMode.REVISION_2026_07_28.value:
            raise TypeError("MCP modern connection revision is invalid")
        if not sessionless or fallback_used:
            raise TypeError("MCP modern connection flags are invalid")

    @staticmethod
    def _validate_legacy_connection_flags(
        mode: McpProtocolMode,
        revision: str,
        *,
        sessionless: bool,
        fallback_used: bool,
    ) -> None:
        if revision == McpProtocolMode.REVISION_2026_07_28.value:
            raise TypeError("MCP legacy connection revision is invalid")
        if sessionless:
            raise TypeError("MCP legacy connection cannot be sessionless")
        if mode is McpProtocolMode.REVISION_2026_07_28:
            raise TypeError("MCP pinned modern mode cannot negotiate legacy")
        if mode is McpProtocolMode.AUTO and not fallback_used:
            raise TypeError("MCP auto legacy connection must report fallback")
        if mode is McpProtocolMode.LEGACY and fallback_used:
            raise TypeError("MCP forced legacy connection cannot report fallback")

    def _validate_connection_server_identity(
        self,
        connection: McpConnectionInfo,
    ) -> None:
        mcp_config = getattr(self, "config", DEFAULT_CONFIG).mcp
        for field_name, value in (
            ("server_name", connection.server_name),
            ("server_version", connection.server_version),
        ):
            if value is not None and (
                type(value) is not str
                or len(value) > mcp_config.header_value_max_chars
            ):
                raise TypeError(f"MCP provider {field_name} is invalid")

    def _validated_connection_names(
        self,
        names: Any,
        *,
        field: str,
        sensitive_values: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        mcp_config = getattr(self, "config", DEFAULT_CONFIG).mcp
        if type(names) is not tuple or len(names) > mcp_config.list_limit:
            raise TypeError(f"MCP provider {field} is invalid")
        selected: list[str] = []
        raw_names: list[str] = []
        for name in names:
            if (
                type(name) is not str
                or not name
                or len(name) > mcp_config.mcp_name_max_chars
            ):
                raise TypeError(f"MCP provider {field} is invalid")
            raw_names.append(name)
            sanitized = redact_sensitive_text(
                name,
                sensitive_values=sensitive_values,
            )
            if sanitized not in selected:
                selected.append(sanitized)
        if len(set(raw_names)) != len(raw_names):
            raise TypeError(f"MCP provider {field} contains duplicates")
        return tuple(selected)

    def _validated_exchange_receipts(
        self,
        server: McpServerSpec,
        receipts: Any,
    ) -> tuple[McpExchangeReceipt, ...]:
        if type(receipts) is not tuple:
            raise TypeError("MCP provider receipts must be a tuple")
        mcp_config = getattr(self, "config", DEFAULT_CONFIG).mcp
        if len(receipts) > mcp_config.list_max_pages + 3:
            raise TypeError("MCP provider returned too many exchange receipts")
        selected: list[McpExchangeReceipt] = []
        request_total = 0
        response_total = 0
        for receipt in receipts:
            if not isinstance(receipt, McpExchangeReceipt):
                raise TypeError("MCP provider exchange receipt is invalid")
            if type(receipt.phase) is not McpExchangePhase:
                raise TypeError("MCP provider receipt phase is invalid")
            if type(receipt.request_bytes) is not int or receipt.request_bytes < 0:
                raise TypeError("MCP provider receipt request_bytes is invalid")
            if type(receipt.response_bytes) is not int or receipt.response_bytes < 0:
                raise TypeError("MCP provider receipt response_bytes is invalid")
            if (
                type(receipt.duration_s) not in {int, float}
                or not math.isfinite(receipt.duration_s)
                or receipt.duration_s < 0
            ):
                raise TypeError("MCP provider receipt duration_s is invalid")
            if type(receipt.call_started) is not bool:
                raise TypeError("MCP provider receipt call_started is invalid")
            selected.append(
                McpExchangeReceipt(
                    phase=receipt.phase,
                    request_bytes=receipt.request_bytes,
                    response_bytes=receipt.response_bytes,
                    duration_s=float(receipt.duration_s),
                    call_started=receipt.call_started,
                )
            )
            request_total += receipt.request_bytes
            response_total += receipt.response_bytes
        if server.schema_version == 2:
            if request_total > server.max_request_bytes:
                raise TypeError("MCP provider receipts exceed cumulative request budget")
            if response_total > server.max_response_bytes:
                raise TypeError("MCP provider receipts exceed cumulative response budget")
        return tuple(selected)

    def _validated_v2_negotiation_prefix(
        self,
        server: McpServerSpec,
        connection: McpConnectionInfo,
        receipts: tuple[McpExchangeReceipt, ...],
    ) -> int:
        """Return the first post-negotiation receipt index for Manifest v2.

        Negotiation evidence is deliberately a small grammar rather than an
        unordered bag.  This prevents a provider from hiding automatic
        retries, reporting fallback without an initialize exchange, or
        attributing a later tools exchange to negotiation.
        """

        if server.schema_version != 2:  # pragma: no cover - caller invariant
            raise TypeError("MCP v2 negotiation receipts require Manifest v2")
        if not receipts:
            raise TypeError("MCP Manifest v2 provider result is missing receipts")

        mode = self._effective_protocol_mode(server)
        if mode is McpProtocolMode.LEGACY:
            receipt = receipts[0]
            if (
                receipt.phase is not McpExchangePhase.INITIALIZE
                or not receipt.call_started
            ):
                raise TypeError("MCP legacy negotiation must begin with initialize")
            return 1

        discover_count = 0
        while (
            discover_count < len(receipts)
            and receipts[discover_count].phase is McpExchangePhase.SERVER_DISCOVER
        ):
            if not receipts[discover_count].call_started:
                raise TypeError("MCP server/discover receipt must prove dispatch")
            discover_count += 1
        if discover_count == 0:
            raise TypeError("MCP modern negotiation must begin with server/discover")
        if discover_count > 2:
            raise TypeError("MCP provider exceeded the bounded server/discover retry")

        if connection.fallback_used:
            # A version-negotiation retry can only converge on the pinned
            # modern revision.  Legacy fallback therefore follows exactly one
            # probe and one initialize exchange.
            if discover_count != 1:
                raise TypeError("MCP legacy fallback cannot follow a discover retry")
            if (
                discover_count >= len(receipts)
                or receipts[discover_count].phase is not McpExchangePhase.INITIALIZE
                or not receipts[discover_count].call_started
            ):
                raise TypeError("MCP legacy fallback must include initialize")
            return discover_count + 1

        if (
            discover_count < len(receipts)
            and receipts[discover_count].phase is McpExchangePhase.INITIALIZE
        ):
            raise TypeError("MCP modern negotiation cannot include initialize")
        return discover_count

    def _validate_v2_list_receipts(
        self,
        server: McpServerSpec,
        connection: McpConnectionInfo,
        receipts: tuple[McpExchangeReceipt, ...],
        *,
        response_bytes: int,
    ) -> None:
        negotiation_end = self._validated_v2_negotiation_prefix(
            server,
            connection,
            receipts,
        )
        list_receipts = receipts[negotiation_end:]
        max_pages = getattr(self, "config", DEFAULT_CONFIG).mcp.list_max_pages
        if not 1 <= len(list_receipts) <= max_pages:
            raise TypeError(
                f"MCP tools/list requires 1..{max_pages} page receipts"
            )
        if any(
            receipt.phase is not McpExchangePhase.TOOLS_LIST
            or not receipt.call_started
            for receipt in list_receipts
        ):
            raise TypeError("MCP tools/list page receipts must be contiguous and dispatched")
        if sum(item.response_bytes for item in list_receipts) != response_bytes:
            raise TypeError("MCP tools/list response bytes do not match page receipts")

    def _validate_v2_call_receipts(
        self,
        server: McpServerSpec,
        result: McpProviderCallResult,
        connection: McpConnectionInfo,
        receipts: tuple[McpExchangeReceipt, ...],
    ) -> None:
        negotiation_end = self._validated_v2_negotiation_prefix(
            server,
            connection,
            receipts,
        )
        remaining = receipts[negotiation_end:]
        list_count = 0
        while (
            list_count < len(remaining)
            and remaining[list_count].phase is McpExchangePhase.TOOLS_LIST
        ):
            if not remaining[list_count].call_started:
                raise TypeError("MCP tools/list receipt must prove dispatch")
            list_count += 1
        max_pages = getattr(self, "config", DEFAULT_CONFIG).mcp.list_max_pages
        if not 1 <= list_count <= max_pages:
            raise TypeError(
                f"MCP tool call requires 1..{max_pages} live tools/list page receipts"
            )

        call_receipts = remaining[list_count:]
        if len(call_receipts) > 1 or any(
            receipt.phase is not McpExchangePhase.TOOLS_CALL
            for receipt in call_receipts
        ):
            raise TypeError("MCP tools/call receipt must be unique and terminal")
        if result.call_started:
            if len(call_receipts) != 1 or not call_receipts[0].call_started:
                raise TypeError("MCP dispatched tools/call is missing its terminal receipt")
        elif any(receipt.call_started for receipt in call_receipts):
            raise TypeError("MCP provider claims tools/call did not start after dispatch")

        list_receipts = remaining[:list_count]
        list_request_bytes = sum(item.request_bytes for item in list_receipts)
        list_response_bytes = sum(item.response_bytes for item in list_receipts)
        call_request_bytes = sum(item.request_bytes for item in call_receipts)
        call_response_bytes = sum(item.response_bytes for item in call_receipts)
        expected_fields = (
            ("list_request_bytes", result.list_request_bytes, list_request_bytes),
            ("list_response_bytes", result.list_response_bytes, list_response_bytes),
            ("call_request_bytes", result.call_request_bytes, call_request_bytes),
            ("call_response_bytes", result.call_response_bytes, call_response_bytes),
            ("response_bytes", result.response_bytes, call_response_bytes),
        )
        for field_name, actual, expected in expected_fields:
            if actual != expected:
                raise TypeError(
                    f"MCP provider {field_name} does not match phase receipts"
                )

    @staticmethod
    def _validated_provider_tools(
        tools: list[Any],
        *,
        budget: _ProviderJsonBudget,
        max_tools: int,
    ) -> list[McpProviderTool]:
        if len(tools) > max_tools:
            raise TypeError(f"MCP tools/list exceeds maximum tool count={max_tools}")
        selected_tools: list[McpProviderTool] = []
        names: set[str] = set()
        for index, item in enumerate(tools):
            budget.consume_node(path=f"$.tools[{index}]", depth=1)
            if not isinstance(item, McpProviderTool):
                raise TypeError("MCP tools/list contains an invalid tool")
            name = item.name
            description = item.description
            input_schema = item.input_schema
            metadata = item.metadata
            if type(name) is not str or not name:
                raise TypeError("MCP tools/list tool name is invalid")
            budget.consume_string(name, path=f"$.tools[{index}].name")
            if name in names:
                raise TypeError("MCP tools/list contains duplicate tool names")
            if description is not None and type(description) is not str:
                raise TypeError("MCP tools/list tool description is invalid")
            if description is not None:
                budget.consume_string(
                    description,
                    path=f"$.tools[{index}].description",
                )
            if type(input_schema) is not dict or type(metadata) is not dict:
                raise TypeError("MCP tools/list tool metadata is invalid")
            selected_tools.append(
                McpProviderTool(
                    name=name,
                    description=description,
                    input_schema=_strict_provider_json_value(
                        input_schema,
                        path=f"$.tools[{index}].input_schema",
                        active_containers=set(),
                        budget=budget,
                        depth=2,
                    ),
                    metadata=_strict_provider_json_value(
                        metadata,
                        path=f"$.tools[{index}].metadata",
                        active_containers=set(),
                        budget=budget,
                        depth=2,
                    ),
                )
            )
            names.add(name)
        return selected_tools

    @staticmethod
    def _canonical_provider_json_bytes(value: Any, *, context: str) -> int:
        try:
            return len(dumps(value).encode("utf-8"))
        except (TypeError, ValueError, RecursionError, UnicodeEncodeError) as error:
            raise TypeError(f"{context} is not canonical JSON") from error

    @staticmethod
    def _validate_provider_call_header(
        server: McpServerSpec,
        result: McpProviderCallResult,
    ) -> None:
        if type(result.is_error) is not bool or type(result.too_large) is not bool:
            raise TypeError("MCP provider call flags are invalid")
        if type(result.call_started) is not bool:
            raise TypeError("MCP provider call_started is invalid")
        byte_fields = (
            ("response_bytes", result.response_bytes),
            ("list_request_bytes", result.list_request_bytes),
            ("list_response_bytes", result.list_response_bytes),
            ("call_request_bytes", result.call_request_bytes),
            ("call_response_bytes", result.call_response_bytes),
        )
        for field_name, selected in byte_fields:
            if type(selected) is not int or selected < 0:
                raise TypeError(f"MCP provider {field_name} is invalid")
        for field_name, selected in (
            ("response_bytes", result.response_bytes),
            ("list_response_bytes", result.list_response_bytes),
            ("call_response_bytes", result.call_response_bytes),
        ):
            if selected > server.max_response_bytes:
                raise TypeError(
                    f"MCP provider {field_name} exceeds max_response_bytes"
                )
        if (
            type(result.duration_s) not in {int, float}
            or not math.isfinite(result.duration_s)
            or result.duration_s < 0
        ):
            raise TypeError("MCP provider duration_s is invalid")
        for field_name, selected in (
            ("error", result.error),
            ("error_type", result.error_type),
            ("correlation_id", result.correlation_id),
        ):
            if selected is not None and type(selected) is not str:
                raise TypeError(f"MCP provider {field_name} is invalid")
        if (
            result.error_type == "mcp_input_required_unsupported"
            and not result.call_started
        ):
            raise TypeError("MCP input_required must follow a dispatched tools/call")

    @staticmethod
    def _validated_provider_call_payloads(
        server: McpServerSpec,
        result: McpProviderCallResult,
    ) -> tuple[Any, Any]:
        budget = _ProviderJsonBudget(server.max_response_bytes)
        selected_content = _strict_provider_json_value(
            result.content,
            path="$.content",
            active_containers=set(),
            budget=budget,
        )
        selected_structured_content = _strict_provider_json_value(
            result.structured_content,
            path="$.structured_content",
            active_containers=set(),
            budget=budget,
        )
        for field_name, selected in (
            ("error", result.error),
            ("error_type", result.error_type),
            ("correlation_id", result.correlation_id),
        ):
            if selected is not None:
                budget.consume_string(selected, path=f"$.{field_name}")
        return selected_content, selected_structured_content

    @staticmethod
    def _validate_provider_call_byte_contract(
        server: McpServerSpec,
        result: McpProviderCallResult,
        *,
        selected_content: Any,
        selected_structured_content: Any,
    ) -> None:
        has_response_payload = (
            result.error is None
            or result.content is not None
            or result.structured_content is not None
        )
        canonical_response_bytes = 0
        if has_response_payload:
            canonical_response_bytes = McpPrimitive._canonical_provider_json_bytes(
                {
                    "content": selected_content,
                    "structured_content": selected_structured_content,
                },
                context="MCP tool response",
            )
            if canonical_response_bytes > server.max_response_bytes:
                raise TypeError("MCP tool canonical response exceeds max_response_bytes")
        # Manifest v1 byte fields predate phase receipts and retain their
        # released canonical-projection lower-bound contract. Manifest v2
        # fields are exact raw-wire measurements and are checked against the
        # phase receipts below; a decoded/canonical JSON projection can be
        # larger than its wire representation without implying undercounting.
        if server.schema_version == 1:
            if result.too_large:
                McpPrimitive._validate_oversized_provider_call_bytes(server, result)
            elif has_response_payload:
                McpPrimitive._validate_canonical_provider_call_bytes(
                    result,
                    canonical_response_bytes=canonical_response_bytes,
                )

    @staticmethod
    def _validate_oversized_provider_call_bytes(
        server: McpServerSpec,
        result: McpProviderCallResult,
    ) -> None:
        if result.response_bytes < server.max_response_bytes:
            raise TypeError(
                "MCP provider response_bytes underreports an oversized response"
            )
        if (
            result.call_response_bytes
            and result.call_response_bytes < server.max_response_bytes
        ):
            raise TypeError(
                "MCP provider call_response_bytes underreports an oversized response"
            )

    @staticmethod
    def _validate_canonical_provider_call_bytes(
        result: McpProviderCallResult,
        *,
        canonical_response_bytes: int,
    ) -> None:
        if result.response_bytes < canonical_response_bytes:
            raise TypeError(
                "MCP provider response_bytes underreports canonical response"
            )
        if (
            result.call_response_bytes
            and result.call_response_bytes < canonical_response_bytes
        ):
            raise TypeError(
                "MCP provider call_response_bytes underreports canonical response"
            )

    def _validated_provider_call_result(
        self,
        server: McpServerSpec,
        result: Any,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpProviderCallResult:
        """Decode every provider-owned call field into an inert value object."""

        try:
            if not isinstance(result, McpProviderCallResult):
                raise TypeError("MCP provider returned an invalid call result")
            McpPrimitive._validate_provider_call_header(server, result)
            selected_content, selected_structured_content = (
                McpPrimitive._validated_provider_call_payloads(server, result)
            )
            McpPrimitive._validate_provider_call_byte_contract(
                server,
                result,
                selected_content=selected_content,
                selected_structured_content=selected_structured_content,
            )
            receipts = self._validated_exchange_receipts(server, result.receipts)
            incomplete_negotiation_failure = (
                server.schema_version == 2
                and result.connection is None
                and self._validate_v2_incomplete_negotiation_failure(
                    server,
                    result,
                    receipts,
                )
            )
            connection = self._validated_connection_info(
                server,
                result.connection,
                required=(
                    server.schema_version == 2
                    and not incomplete_negotiation_failure
                ),
                runtime_environment=runtime_environment,
            )
            if server.schema_version == 2:
                if connection is not None:
                    self._validate_v2_call_receipts(
                        server,
                        result,
                        connection,
                        receipts,
                    )
            else:
                if any(
                    receipt.phase
                    not in {
                        McpExchangePhase.SERVER_DISCOVER,
                        McpExchangePhase.INITIALIZE,
                        McpExchangePhase.TOOLS_LIST,
                        McpExchangePhase.TOOLS_CALL,
                    }
                    for receipt in receipts
                ):
                    raise TypeError("MCP tool call returned an unrelated phase receipt")
                if sum(
                    receipt.phase is McpExchangePhase.TOOLS_LIST
                    for receipt in receipts
                ) > getattr(self, "config", DEFAULT_CONFIG).mcp.list_max_pages:
                    raise TypeError("MCP tool call exceeds maximum tools/list pages")
                if sum(
                    receipt.phase is McpExchangePhase.TOOLS_CALL
                    for receipt in receipts
                ) > 1:
                    raise TypeError("MCP provider automatically retried tools/call")
            return McpProviderCallResult(
                content=selected_content,
                structured_content=selected_structured_content,
                is_error=result.is_error,
                error=result.error,
                response_bytes=result.response_bytes,
                duration_s=float(result.duration_s),
                too_large=result.too_large,
                error_type=result.error_type,
                correlation_id=result.correlation_id,
                list_request_bytes=result.list_request_bytes,
                list_response_bytes=result.list_response_bytes,
                call_request_bytes=result.call_request_bytes,
                call_response_bytes=result.call_response_bytes,
                call_started=result.call_started,
                connection=connection,
                receipts=receipts,
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

    def _validate_v2_incomplete_negotiation_failure(
        self,
        server: McpServerSpec,
        result: McpProviderCallResult,
        receipts: tuple[McpExchangeReceipt, ...],
    ) -> bool:
        """Accept only built-in wire proof that negotiation failed pre-call."""

        if type(self.provider) is not SdkMcpProvider:
            raise TypeError("MCP Manifest v2 tool-call connection is missing")
        if (
            not result.error
            or result.content is not None
            or result.structured_content is not None
            or result.is_error
            or result.too_large
            or result.call_started
            or any(
                value != 0
                for value in (
                    result.response_bytes,
                    result.list_request_bytes,
                    result.list_response_bytes,
                    result.call_request_bytes,
                    result.call_response_bytes,
                )
            )
            or any(not receipt.call_started for receipt in receipts)
        ):
            raise TypeError("MCP incomplete negotiation failure is invalid")

        phases = tuple(receipt.phase for receipt in receipts)
        mode = self._effective_protocol_mode(server)
        if mode is McpProtocolMode.LEGACY:
            valid_phases = phases in {(), (McpExchangePhase.INITIALIZE,)}
        elif mode is McpProtocolMode.REVISION_2026_07_28:
            valid_phases = phases in {
                (),
                (McpExchangePhase.SERVER_DISCOVER,),
                (
                    McpExchangePhase.SERVER_DISCOVER,
                    McpExchangePhase.SERVER_DISCOVER,
                ),
            }
        else:
            valid_phases = phases in {
                (),
                (McpExchangePhase.SERVER_DISCOVER,),
                (
                    McpExchangePhase.SERVER_DISCOVER,
                    McpExchangePhase.SERVER_DISCOVER,
                ),
                (
                    McpExchangePhase.SERVER_DISCOVER,
                    McpExchangePhase.INITIALIZE,
                ),
            }
        if not valid_phases:
            raise TypeError(
                "MCP incomplete negotiation receipts are not a legal prefix"
            )
        return True

    def _validate_live_tool_for_call(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        *,
        deadline: float,
        pid: str,
        executable_snapshot: ExecutableSnapshot | None,
        runtime_environment: Mapping[str, str] | None,
    ) -> tuple[McpToolListResult | None, Exception | None, int]:
        """Retain known list bytes while keeping not-started failures exceptional."""

        try:
            result = self._validate_live_tool(
                server,
                tool,
                timeout_s=self._remaining_timeout(deadline),
                pid=pid,
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
                    pid=pid,
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
        pid: str | None,
        runtime_environment: Mapping[str, str] | None,
        executable_snapshot: ExecutableSnapshot | None,
    ) -> dict[str, Any]:
        selected: dict[str, Any] = {
            "timeout_s": self._remaining_timeout(deadline),
            "max_response_bytes": server.max_response_bytes,
            "runtime_environment": runtime_environment,
        }
        if executable_snapshot is not None:
            selected["executable_snapshot"] = executable_snapshot
        if server.transport == "stdio":
            usage_pid = self._resource_usage_pid(pid)
            limits = (
                self._subprocess_limits(usage_pid)
                if usage_pid is not None
                else None
            )
            if limits is not None:
                if (
                    not isinstance(
                        self.provider,
                        McpSubprocessLimitsProvider,
                    )
                    or self.provider.supports_subprocess_limits is not True
                ):
                    raise ValidationError(
                        "MCP provider must explicitly support SubprocessLimits "
                        "before budgeted stdio execution"
                    )
                selected["limits"] = limits
        return selected

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
        if wall is not None and wall <= 0:
            raise ResourceLimitExceeded(
                f"process {pid} exhausted subprocess wall-time budget"
            )
        if cpu is not None and cpu <= 0:
            raise ResourceLimitExceeded(
                f"process {pid} exhausted subprocess CPU budget"
            )
        if memory is not None and memory <= 0:
            raise ResourceLimitExceeded(
                f"process {pid} exhausted subprocess memory budget"
            )
        if wall is None and cpu is None and memory is None:
            return None
        return SubprocessLimits(
            wall_seconds=wall,
            cpu_seconds=cpu,
            memory_bytes=memory,
        )

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
        provider_result = self._validated_provider_call_result(server, provider_result)
        if provider_result.error_type == "mcp_input_required_unsupported":
            return McpCallResult(
                server_id=server.server_id,
                tool_id=tool.tool_id,
                mcp_name=tool.mcp_name,
                status=McpCallStatus.INPUT_REQUIRED_UNSUPPORTED,
                ok=False,
                error={
                    "code": "mcp_input_required_unsupported",
                    "error_type": "mcp_input_required_unsupported",
                    "correlation_id": provider_result.correlation_id or new_id("corr"),
                    "message": (
                        "MCP server requested multi-round input, which is not "
                        "supported by this release"
                    ),
                    "retryable": False,
                    "automatic_retry_disabled": True,
                    "continuation_present": True,
                },
                response_bytes=provider_result.response_bytes,
                duration_s=provider_result.duration_s,
                connection=provider_result.connection,
                receipts=provider_result.receipts,
            )
        if provider_result.error:
            safe_message = self._safe_transport_error_message(
                server,
                provider_result.error_type,
            )
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
                    **(
                        {"message": safe_message}
                        if safe_message is not None
                        else {}
                    ),
                },
                response_bytes=provider_result.response_bytes,
                duration_s=provider_result.duration_s,
                connection=provider_result.connection,
                receipts=provider_result.receipts,
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
            # Project only the returned/model-facing payload.  Provider byte
            # receipts and settlement fields remain unchanged.
            projected_error = _model_facing_mcp_call_payload(
                provider_result.content,
                provider_result.structured_content,
            )
            return self._failure(
                server,
                tool,
                McpCallStatus.MCP_ERROR,
                "MCP tool returned an error result",
                provider_result,
                extra={
                    key: value
                    for key, value in projected_error.items()
                    if value is not None
                },
            )
        projected = _model_facing_mcp_call_payload(
            provider_result.content,
            provider_result.structured_content,
        )
        return McpCallResult(
            server_id=server.server_id,
            tool_id=tool.tool_id,
            mcp_name=tool.mcp_name,
            status=McpCallStatus.OK,
            ok=True,
            result=to_jsonable(projected),
            response_bytes=provider_result.response_bytes,
            duration_s=provider_result.duration_s,
            connection=provider_result.connection,
            receipts=provider_result.receipts,
        )

    def _invoke_validated_provider_tool(
        self,
        validate_and_call: Any,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        deadline: float,
        pid: str,
        runtime_environment: Mapping[str, str] | None,
        executable_snapshot: ExecutableSnapshot | None,
        started: float,
    ) -> McpProviderCallResult | ProviderEffectNotStartedResult:
        try:
            provider_kwargs = self._provider_dispatch_kwargs(
                server,
                deadline=deadline,
                pid=pid,
                runtime_environment=runtime_environment,
                executable_snapshot=executable_snapshot,
            )
            raw_result = validate_and_call(
                server,
                tool,
                arguments,
                **provider_kwargs,
            )
            return self._validated_provider_call_result(
                server,
                raw_result,
                runtime_environment=runtime_environment,
            )
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
        except Exception as error:
            post_call_failure = self._wire_bound_post_call_failure_result(
                server,
                error,
                started=started,
                runtime_environment=runtime_environment,
            )
            if post_call_failure is not None:
                return post_call_failure
            if isinstance(error, ProviderHostError):
                raise
            raise ProviderHostError(
                code="mcp_provider_error",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            ) from None

    def _wire_bound_post_call_failure_result(
        self,
        server: McpServerSpec,
        error: BaseException,
        *,
        started: float,
        runtime_environment: Mapping[str, str] | None,
    ) -> McpProviderCallResult | None:
        if server.schema_version != 2 or type(self.provider) is not SdkMcpProvider:
            return None
        certified, receipts, connection = _wire_failure_evidence(error)
        call_receipts = tuple(
            receipt
            for receipt in receipts
            if receipt.phase is McpExchangePhase.TOOLS_CALL
        )
        if (
            not certified
            or connection is None
            or len(call_receipts) != 1
            or not call_receipts[0].call_started
        ):
            return None
        list_receipts = tuple(
            receipt
            for receipt in receipts
            if receipt.phase is McpExchangePhase.TOOLS_LIST
        )
        result = McpProviderCallResult(
            error="MCP tools/call dispatch outcome is unknown",
            error_type="McpPostCallFailure",
            correlation_id=new_id("corr"),
            response_bytes=call_receipts[0].response_bytes,
            duration_s=max(0.0, time.monotonic() - started),
            list_request_bytes=sum(item.request_bytes for item in list_receipts),
            list_response_bytes=sum(item.response_bytes for item in list_receipts),
            call_request_bytes=call_receipts[0].request_bytes,
            call_response_bytes=call_receipts[0].response_bytes,
            call_started=True,
            connection=connection,
            receipts=receipts,
        )
        return self._validated_provider_call_result(
            server,
            result,
            runtime_environment=runtime_environment,
        )

    def _pre_call_failure_classification_override(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        provider_result: McpProviderCallResult,
        *,
        wire_bound_combined_provider: bool,
    ) -> ExternalEffectClassification | None:
        if provider_result.call_started:
            return None

        # Manifest v1 retains its released Provider SPI: call_started=False is
        # the provider's pre-dispatch certificate.  Manifest v2 has exact phase
        # receipts, so require the stronger wire-bound certificate before a
        # mutating tool can be narrowed to an external-read failure.
        certified_pre_call = server.schema_version == 1 or (
            wire_bound_combined_provider
            and provider_result.call_request_bytes == 0
            and provider_result.call_response_bytes == 0
            and provider_result.response_bytes == 0
            and all(
                receipt.phase is not McpExchangePhase.TOOLS_CALL
                for receipt in provider_result.receipts
            )
        )
        if certified_pre_call:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={
                    "outcome": "failed",
                    "failure_kind": "live_validation_failed_before_call",
                    "phase": "live_validation",
                    "call_started": False,
                    "tools_call_receipt_present": False,
                },
            )

        # A v2 result which claims no call but carries call bytes or a
        # tools/call receipt is not a pre-dispatch certificate.  Preserve an
        # unknown mutation for mutating/rollback-unclear tools so an ambiguous
        # provider result can never enable replay.
        rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
        if (
            tool.state_mutation
            or rollback_class is not ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        ):
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                state_mutation=tool.state_mutation,
                information_flow=True,
                metadata={
                    "outcome": "unknown_pre_call_dispatch_state",
                    "phase": "tools/call",
                    "call_started": False,
                    "tools_call_receipt_present": any(
                        receipt.phase is McpExchangePhase.TOOLS_CALL
                        for receipt in provider_result.receipts
                    ),
                },
            )
        return None

    @staticmethod
    def _post_call_failure_classification_override(
        tool: McpToolSpec,
        provider_result: McpProviderCallResult,
    ) -> ExternalEffectClassification | None:
        # Any provider failure after tools/call dispatch leaves a mutating
        # tool's external state ambiguous, including bounded transport errors.
        if provider_result.error is None or not provider_result.call_started:
            return None
        rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
        if (
            not tool.state_mutation
            and rollback_class is ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        ):
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={
                    "outcome": "failed",
                    "failure_kind": "mcp_post_call_read_failure",
                    "phase": "tools/call",
                    "call_started": True,
                    "automatic_retry_disabled": True,
                },
            )
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
            state_mutation=tool.state_mutation,
            information_flow=True,
            metadata={
                "outcome": "unknown_mcp_post_call_failure",
                "failure_kind": "mcp_post_call_failure",
                "phase": "tools/call",
                "call_started": True,
                "automatic_retry_disabled": True,
            },
        )

    def _dispatch_validated_provider_call(
        self,
        protected: Any,
        invoke_validated_tool: Any,
        *,
        server: McpServerSpec,
        tool: McpToolSpec,
        executable_snapshot: ExecutableSnapshot | None,
    ) -> tuple[Any, bool]:
        # Manifest v2 combines negotiation, live catalog validation and the
        # eventual tools/call in one provider entry point. Only the exact
        # built-in SDK provider has wire evidence strong enough to narrow a
        # normal pre-call result; custom providers retain the mutation floor.
        wire_bound = (
            server.schema_version == 2
            and type(self.provider) is SdkMcpProvider
        )
        phase = ProviderPhase(
            "provider_validate_and_call",
            state_mutation=tool.state_mutation and not wire_bound,
            information_flow=True,
        )
        try:
            return protected.call(phase, invoke_validated_tool), wire_bound
        finally:
            if executable_snapshot is not None:
                executable_snapshot.close()

    @staticmethod
    def _input_required_classification_override(
        tool: McpToolSpec,
        provider_result: McpProviderCallResult,
    ) -> ExternalEffectClassification | None:
        if provider_result.error_type != "mcp_input_required_unsupported":
            return None
        rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
        if (
            not tool.state_mutation
            and rollback_class is ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        ):
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={
                    "outcome": "failed",
                    "failure_kind": "mcp_input_required_unsupported",
                    "phase": "tools/call",
                    "automatic_retry_disabled": True,
                    "continuation_present": True,
                },
            )
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
            state_mutation=tool.state_mutation,
            information_flow=True,
            metadata={
                "outcome": "unknown_mcp_input_required_unsupported",
                "phase": "tools/call",
                "automatic_retry_disabled": True,
                "continuation_present": True,
            },
        )

    @staticmethod
    def _safe_transport_error_message(
        server: McpServerSpec,
        error_type: str | None,
    ) -> str | None:
        if error_type == "McpStdioFrameTooLarge":
            return (
                "MCP stdio frame exceeded "
                f"max_response_bytes={server.max_response_bytes}"
            )
        if error_type == "McpStdioStdoutTooLarge":
            max_output_bytes = server.max_response_bytes
            if server.schema_version == 1:
                max_output_bytes *= _MCP_STDIO_PROTOCOL_OUTPUT_MULTIPLIER
            return (
                "MCP stdio stdout exceeded max_output_bytes="
                f"{max_output_bytes}"
            )
        if error_type == "McpStdioStderrTooLarge":
            return (
                "MCP stdio stderr exceeded "
                f"max_output_bytes={server.max_response_bytes}"
            )
        if error_type == "McpHttpSseFrameTooLarge":
            return (
                "MCP HTTP SSE frame exceeded "
                f"max_response_bytes={server.max_response_bytes}"
            )
        if error_type == "McpHttpResponseTooLarge":
            return (
                "MCP HTTP response exceeded "
                f"max_response_bytes={server.max_response_bytes}"
            )
        if error_type == "McpHttpOperationTooLarge":
            return (
                "MCP HTTP operation exceeded "
                f"max_response_bytes={server.max_response_bytes}"
            )
        if error_type == "McpHttpContentEncodingDenied":
            return "MCP HTTP response uses unsupported Content-Encoding"
        return None

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
            connection=provider_result.connection,
            receipts=provider_result.receipts,
        )

    def _protected(self) -> Any:
        return self.protected_operations

    @staticmethod
    def _effective_protocol_mode(server: McpServerSpec) -> McpProtocolMode:
        return server.protocol_mode or McpProtocolMode.LEGACY

    def _require_modern_protocol_provider(self, server: McpServerSpec) -> None:
        if server.schema_version != 2:
            return
        if (
            not isinstance(self.provider, McpModernProtocolProvider)
            or self.provider.supports_mcp_modern_protocol is not True
        ):
            raise ValidationError(
                "MCP Manifest v2 requires a provider that explicitly supports "
                "modern protocol negotiation"
            )

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
                        "schema_version": spec.schema_version,
                        "server": mcp_server_spec_to_jsonable(spec),
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
                        "schema_version": spec.schema_version,
                        "server": mcp_server_spec_to_jsonable(spec),
                        "operation": "tools/list",
                        "stdio_executable": stdio_executable,
                    }
                )
            ).encode("utf-8")
        ).hexdigest()

    def _discover_identity_sha256(
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
                        "schema_version": 2,
                        "server": mcp_server_spec_to_jsonable(spec),
                        "operation": "server/discover",
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

    def _tool_data_sink_for_clearance_precheck(
        self,
        server_id: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
    ) -> DataSink:
        identity = f"mcp:{server_id}:{tool.tool_id}"
        if spec.transport != "stdio":
            return self._tool_data_sink_from_stdio_identity(
                server_id,
                spec,
                tool,
                None,
            )

        # Stdio executable identity may depend on a Host-provided PATH. Use
        # the Host trust record's expected identity only for this negative
        # precheck; exact authorization below uses the resolved executable.
        trust = self._data_flow().resolve_sink_trust(DataSink(identity))
        return DataSink(
            identity,
            trust.identity_sha256 if trust is not None else None,
        )

    def _stdio_executable_resolution_environment(
        self,
        spec: McpServerSpec,
    ) -> Mapping[str, str]:
        if spec.transport != "stdio" or spec.stdio is None:
            return MappingProxyType({})
        command = spec.stdio.command
        if Path(command).is_absolute() or "/" in command or "\\" in command:
            return MappingProxyType({})
        child_names = ("PATH", "PATHEXT") if _MCP_WINDOWS else ("PATH",)
        selected: dict[str, str] = {}
        for child_name in child_names:
            host_name = spec.stdio.env.get(child_name)
            if host_name is None:
                if _MCP_WINDOWS:
                    raise ValidationError(
                        "Windows MCP stdio bare commands require "
                        "manifest-mapped child PATH and PATHEXT"
                    )
                continue
            resolved = os.environ.get(host_name)
            if resolved is None:
                raise ValidationError(
                    f"missing environment variable for MCP stdio env {child_name}: "
                    f"{host_name}"
                )
            if "\x00" in resolved:
                raise ValidationError(
                    f"MCP stdio env {child_name} contains NUL byte"
                )
            selected[child_name] = resolved
        return MappingProxyType(selected)

    def _tool_data_sink_from_stdio_identity(
        self,
        server_id: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
        stdio_identity: dict[str, str] | None,
    ) -> DataSink:
        return DataSink(
            f"mcp:{server_id}:{tool.tool_id}",
            self._server_identity_sha256(
                spec,
                tool,
                stdio_executable=stdio_identity,
            ),
        )

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
        return self._tool_data_sink_from_stdio_identity(
            server_id,
            spec,
            tool,
            stdio_identity,
        )

    def _tool_data_sink_after_runtime_resolution(
        self,
        server_id: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
        runtime_environment: Mapping[str, str] | None,
        *,
        expected: DataSink,
    ) -> DataSink:
        if runtime_environment is None:
            return expected
        return self._tool_data_sink(
            server_id,
            spec,
            tool,
            runtime_environment,
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

    def _discover_data_sink(
        self,
        server_id: str,
        spec: McpServerSpec,
        runtime_environment: Mapping[str, str] | None,
    ) -> DataSink:
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=runtime_environment,
        )
        return DataSink(
            f"mcp:{server_id}:discover",
            self._discover_identity_sha256(
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
            sibling_policy="scripts",
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

    def _protected_discover_evidence(
        self,
        pid: str,
        spec: McpServerSpec,
        context: dict[str, Any],
        result_payload: dict[str, Any],
    ) -> ProtectedOperationEvidence:
        event_payload = {
            "adapter": "mcp",
            "operation": "discover",
            "server_id": spec.server_id,
            "transport": spec.transport,
            **result_payload,
        }
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_READ,
            event_source=pid,
            event_target=self.server_resource(spec.server_id),
            event_payload=event_payload,
            audit_action="primitive.mcp.discover",
            audit_actor=pid,
            audit_target=self.server_resource(spec.server_id),
            audit_decision={
                "server_id": spec.server_id,
                "transport": spec.transport,
                "protocol_mode": self._effective_protocol_mode(spec).value,
                "request_bytes": context["request_bytes"],
                **result_payload,
            },
            effect_metadata=result_payload,
            provider_receipt={
                "request_bytes": int(result_payload.get("request_bytes", 0) or 0),
                "response_bytes": int(result_payload.get("response_bytes", 0) or 0),
                "duration_s": float(result_payload.get("duration_s", 0.0) or 0.0),
                "protocol_revision": (
                    result_payload.get("connection", {}).get("protocol_revision")
                    if isinstance(result_payload.get("connection"), dict)
                    else None
                ),
            },
        )

    def _protected_discover_failure_evidence(
        self,
        pid: str,
        spec: McpServerSpec,
        context: dict[str, Any],
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        return self._protected_discover_evidence(
            pid,
            spec,
            context,
            {
                "ok": False,
                "status": "transport_error",
                "request_bytes": 0,
                "response_bytes": 0,
                "duration_s": 0.0,
                "error_type": type(error).__name__,
                "phase": phase,
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
        receipt_request_bytes = sum(
            receipt.request_bytes for receipt in result.receipts
        )
        receipt_response_bytes = sum(
            receipt.response_bytes for receipt in result.receipts
        )
        connection_payload = (
            to_jsonable(result.connection)
            if result.connection is not None
            else None
        )
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
                "request_bytes": receipt_request_bytes,
                "response_bytes": result.response_bytes,
                "operation_response_bytes": receipt_response_bytes,
                "duration_s": result.duration_s,
                **(
                    {"receipts": to_jsonable(result.receipts)}
                    if result.receipts
                    else {}
                ),
                **(
                    {
                        "protocol_revision": result.connection.protocol_revision,
                        "protocol_era": result.connection.protocol_era.value,
                        "fallback_used": result.connection.fallback_used,
                        "connection": connection_payload,
                    }
                    if result.connection is not None
                    else {}
                ),
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
        if result.receipts:
            payload["receipts"] = to_jsonable(result.receipts)
        if result.connection is not None:
            payload.update(
                {
                    "protocol_revision": result.connection.protocol_revision,
                    "connection": to_jsonable(result.connection),
                }
            )
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
        provider_result_failure = _provider_result_was_returned(error)
        if (
            provider_result_failure
            and phase == "provider_validate_and_call"
            and spec.schema_version == 2
        ):
            # A combined v2 provider owns negotiation, live validation and
            # dispatch.  Once it has returned a malformed result, no trusted
            # per-phase receipt remains; settle the same cumulative envelope
            # reserved for that phase instead of the legacy two-response shape.
            return ResourceSettlement(
                usage=ResourceUsage(
                    mcp_request_bytes=spec.max_request_bytes,
                    mcp_response_bytes=spec.max_response_bytes,
                ),
                source="primitive.mcp.call",
                context={
                    **resource_context,
                    "failure_phase": phase,
                    "unknown_request_bytes": spec.max_request_bytes,
                    "unknown_response_bytes": spec.max_response_bytes,
                    "call_started": True,
                    "provider_result_returned": True,
                },
            )
        if phase == "provider_call" and provider_result_failure:
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
        if provider_result.receipts:
            return cls._mcp_exchange_settlement(
                provider_result,
                fallback_list_request_bytes=list_request_bytes,
                fallback_call_request_bytes=request_bytes,
                server_id=server_id,
                tool_id=tool_id,
                status=result.status.value,
            )
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
        if result.receipts:
            request_bytes = sum(item.request_bytes for item in result.receipts)
            response_bytes = sum(item.response_bytes for item in result.receipts)
            return ResourceSettlement(
                usage=ResourceUsage(
                    mcp_request_bytes=request_bytes,
                    mcp_response_bytes=response_bytes,
                ),
                source="primitive.mcp.call",
                context={
                    "server_id": server_id,
                    "tool_id": tool_id,
                    "request_bytes": request_bytes,
                    "response_bytes": response_bytes,
                    "call_started": result.call_started,
                    "status": status,
                    "receipts": to_jsonable(result.receipts),
                },
            )
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
        payload = {
            "ok": True,
            "status": "ok",
            "response_bytes": result.response_bytes,
            "duration_s": result.duration_s,
            "tool_count": len(result.tools),
        }
        if result.connection is not None:
            payload["connection"] = to_jsonable(result.connection)
            payload["receipts"] = to_jsonable(result.receipts)
        return payload

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

    def _discover_effect_context(
        self,
        server: McpServerSpec,
        *,
        request_bytes: int,
    ) -> dict[str, Any]:
        return {
            "server_id": server.server_id,
            "transport": server.transport,
            "protocol_mode": self._effective_protocol_mode(server).value,
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
            **(
                {"protocol_mode": self._effective_protocol_mode(server).value}
                if server.schema_version == 2
                else {}
            ),
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
        return hashlib.sha256(
            canonical_mcp_server_spec_json(server).encode("utf-8")
        ).hexdigest()

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
            **(
                {"protocol_mode": self._effective_protocol_mode(server).value}
                if server.schema_version == 2
                else {}
            ),
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
        typed_input = isinstance(value, McpServerSpec)
        if typed_input:
            # Normalize typed and mapping/YAML inputs identically.  Runtime
            # dataclass annotations do not coerce ``timeout_s=1`` to ``1.0``;
            # leaving the typed value untouched would make the durable raw
            # spec and its decoded live model hash differently.
            value = to_jsonable(value)
        if isinstance(value, dict):
            _reject_unknown_fields(value, _SERVER_FIELDS, context="MCP server")
            schema_version = self._coerce_positive_int(
                value.get("schema_version", 1),
                "schema_version",
            )
            if (
                not typed_input
                and schema_version == 1
                and "protocol_mode" in value
            ):
                raise ValidationError(
                    "MCP server schema_version 1 must omit protocol_mode"
                )
            raw_protocol_mode = value.get("protocol_mode")
            protocol_mode: McpProtocolMode | None = None
            if raw_protocol_mode is not None:
                if type(raw_protocol_mode) is not str:
                    raise ValidationError("MCP protocol_mode must be a string")
                try:
                    protocol_mode = McpProtocolMode(raw_protocol_mode)
                except ValueError as exc:
                    raise ValidationError(
                        "MCP protocol_mode must be legacy, auto, or 2026-07-28"
                    ) from exc
            transport = self._required_string(
                value,
                "transport",
                "MCP server",
            ).strip()
            server_id = self._required_string(
                value,
                "server_id",
                "MCP server",
            )
            spec = McpServerSpec(
                schema_version=schema_version,
                server_id=server_id,
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
                tools=[
                    self._tool_spec(item)
                    for item in self._list_field(value, "tools", "MCP server")
                ],
                timeout_s=self._coerce_positive_float(value.get("timeout_s", self.config.mcp.timeout_s), "timeout_s"),
                max_request_bytes=self._coerce_positive_int(
                    value.get("max_request_bytes", self.config.mcp.max_request_bytes),
                    "max_request_bytes",
                ),
                max_response_bytes=self._coerce_positive_int(
                    value.get("max_response_bytes", self.config.mcp.max_response_bytes),
                    "max_response_bytes",
                ),
                metadata=self._mapping_field(value, "metadata", "MCP server"),
                protocol_mode=protocol_mode,
            )
        else:
            raise ValidationError("MCP server must be an object")
        self._validate_server(spec)
        return spec

    def _stdio_spec(self, value: Any) -> McpStdioTransportSpec:
        if not isinstance(value, dict):
            raise ValidationError("MCP stdio transport requires stdio object")
        _reject_unknown_fields(value, _STDIO_FIELDS, context="MCP stdio")
        args = self._list_field(value, "args", "MCP stdio")
        if any(type(item) is not str for item in args):
            raise ValidationError("MCP stdio args must be a list of strings")
        environment = self._mapping_field(value, "env", "MCP stdio")
        if any(
            type(name) is not str or type(host_name) is not str
            for name, host_name in environment.items()
        ):
            raise ValidationError("MCP stdio env must map strings to strings")
        cwd = value.get("cwd")
        if cwd is not None and type(cwd) is not str:
            raise ValidationError("MCP stdio cwd must be a string or null")
        return McpStdioTransportSpec(
            command=self._required_string(value, "command", "MCP stdio"),
            args=list(args),
            env=dict(environment),
            cwd=cwd,
        )

    def _http_spec(self, value: Any) -> McpHttpTransportSpec:
        if not isinstance(value, dict):
            raise ValidationError("MCP streamable_http transport requires http object")
        _reject_unknown_fields(value, _HTTP_FIELDS, context="MCP HTTP")
        return McpHttpTransportSpec(
            url=self._required_string(value, "url", "MCP HTTP"),
            headers=self._header_specs(
                self._mapping_field(value, "headers", "MCP HTTP")
            ),
        )

    def _tool_spec(self, value: Any) -> McpToolSpec:
        if not isinstance(value, dict):
            raise ValidationError("MCP tools entries must be objects")
        _reject_unknown_fields(value, _TOOL_FIELDS, context="MCP tool")
        rollback_status = value.get("rollback_status")
        if rollback_status is not None and type(rollback_status) is not str:
            raise ValidationError("MCP tool rollback_status must be a string or null")
        return McpToolSpec(
            tool_id=self._required_string(value, "tool_id", "MCP tool"),
            mcp_name=self._required_string(value, "mcp_name", "MCP tool"),
            right=self._required_string(value, "right", "MCP tool"),
            rollback_class=self._required_string(
                value,
                "rollback_class",
                "MCP tool",
            ),
            rollback_status=rollback_status,
            state_mutation=self._coerce_bool(
                self._required(value, "state_mutation", "MCP tool"),
                "state_mutation",
            ),
            information_flow=self._coerce_bool(
                self._required(value, "information_flow", "MCP tool"),
                "information_flow",
            ),
            input_schema=self._mapping_field(value, "input_schema", "MCP tool"),
            metadata=self._mapping_field(value, "metadata", "MCP tool"),
        )

    def _validate_server(self, server: McpServerSpec) -> None:
        self._validate_server_protocol(server)
        self._validate_identifier(server.server_id, "server_id", self.config.mcp.server_id_max_chars)
        self._validate_server_transport(server)
        self._validate_server_limits(server)
        self._validate_server_tools(server)
        self._validate_json_value(server.metadata, "metadata")
        if server.schema_version == 2:
            self._validate_no_reserved_mcp_meta(server.metadata, "metadata")

    @staticmethod
    def _validate_server_protocol(server: McpServerSpec) -> None:
        if server.schema_version not in {1, 2}:
            raise ValidationError("MCP server schema_version must be 1 or 2")
        if server.schema_version == 1 and server.protocol_mode is not None:
            raise ValidationError(
                "MCP server schema_version 1 must omit protocol_mode"
            )
        if server.schema_version == 2 and server.protocol_mode is None:
            raise ValidationError(
                "MCP server schema_version 2 requires protocol_mode"
            )

    def _validate_server_transport(self, server: McpServerSpec) -> None:
        if server.transport not in _TRANSPORTS:
            raise ValidationError("MCP transport must be stdio or streamable_http")
        if server.transport == "stdio":
            self._validate_stdio(server.stdio)
            if server.http is not None:
                raise ValidationError("MCP stdio server cannot include http configuration")
        if server.transport == "streamable_http":
            self._validate_http(
                server.http,
                schema_version=server.schema_version,
            )
            if server.stdio is not None:
                raise ValidationError("MCP streamable_http server cannot include stdio configuration")

    def _validate_server_limits(self, server: McpServerSpec) -> None:
        if not server.tools:
            raise ValidationError("MCP server must declare at least one allowed tool")
        if (
            server.schema_version == 2
            and len(server.tools) > self.config.mcp.list_limit
        ):
            raise ValidationError(
                "MCP schema_version 2 tool allowlist exceeds "
                f"list_limit={self.config.mcp.list_limit}"
            )
        if server.timeout_s > self.config.mcp.timeout_hard_limit_s:
            raise ValidationError("MCP timeout_s exceeds configured hard limit")
        if server.max_request_bytes > self.config.mcp.max_request_hard_limit_bytes:
            raise ValidationError("MCP max_request_bytes exceeds configured hard limit")
        if server.max_response_bytes > self.config.mcp.max_response_hard_limit_bytes:
            raise ValidationError("MCP max_response_bytes exceeds configured hard limit")

    def _validate_server_tools(self, server: McpServerSpec) -> None:
        seen_tool_ids: set[str] = set()
        seen_mcp_names: set[str] = set()
        for tool in server.tools:
            self._validate_tool(tool, schema_version=server.schema_version)
            if tool.tool_id in seen_tool_ids:
                raise ValidationError(f"duplicate MCP tool_id: {tool.tool_id}")
            if tool.mcp_name in seen_mcp_names:
                raise ValidationError(f"duplicate MCP mcp_name: {tool.mcp_name}")
            seen_tool_ids.add(tool.tool_id)
            seen_mcp_names.add(tool.mcp_name)

    def _validate_stdio(self, stdio: McpStdioTransportSpec | None) -> None:
        if stdio is None:
            raise ValidationError("MCP stdio transport requires stdio configuration")
        self._validate_stdio_command(stdio)
        self._validate_stdio_args(stdio)
        self._validate_stdio_environment(stdio)
        self._validate_stdio_cwd(stdio)

    @staticmethod
    def _validate_stdio_command(stdio: McpStdioTransportSpec) -> None:
        command = stdio.command.strip()
        if not command:
            raise ValidationError("MCP stdio command must be non-empty")
        if (
            command != stdio.command
            or any(char.isspace() for char in command)
            or any(char in command for char in "\r\n;&|<>")
        ):
            raise ValidationError("MCP stdio command must be a single argv token, not a shell string")
        if command.startswith("~"):
            raise ValidationError(
                "MCP stdio command must not use Host home-directory expansion"
            )
        if not _MCP_WINDOWS:
            return
        windows_path = PureWindowsPath(command)
        path_qualified = (
            windows_path.is_absolute() or "/" in command or "\\" in command
        )
        if (
            path_qualified
            and windows_path.suffix.casefold()
            not in _MCP_WINDOWS_EXECUTABLE_SUFFIXES
        ):
            raise ValidationError(
                "Windows MCP stdio executables must end in .exe or .com"
            )
        if not path_qualified and not {"PATH", "PATHEXT"}.issubset(stdio.env):
            raise ValidationError(
                "Windows MCP stdio bare commands require manifest-mapped "
                "child PATH and PATHEXT"
            )

    @staticmethod
    def _validate_stdio_args(stdio: McpStdioTransportSpec) -> None:
        for arg in stdio.args:
            if not isinstance(arg, str) or "\x00" in arg:
                raise ValidationError("MCP stdio args must be strings without NUL bytes")

    def _validate_stdio_environment(self, stdio: McpStdioTransportSpec) -> None:
        for child_name, host_name in stdio.env.items():
            self._validate_env_name(child_name, "stdio env name")
            self._validate_env_name(host_name, "stdio env source")
            if not self._env_allowed(host_name, self.config.mcp.stdio_env_allowlist):
                raise ValidationError(f"MCP stdio env source is not allowlisted: {host_name}")

    @staticmethod
    def _validate_stdio_cwd(stdio: McpStdioTransportSpec) -> None:
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

    def _validate_http(
        self,
        http: McpHttpTransportSpec | None,
        *,
        schema_version: int,
    ) -> None:
        if http is None:
            raise ValidationError("MCP streamable_http transport requires http configuration")
        self._validate_url(http.url)
        for name, header in http.headers.items():
            self._validate_header_name(name, schema_version=schema_version)
            self._validate_env_name(header.env, f"header {name} env")
            if not self._env_allowed(header.env, self.config.mcp.header_env_allowlist):
                raise ValidationError(f"MCP header env is not allowlisted: {header.env}")
            if header.prefix not in _ALLOWED_HEADER_PREFIXES:
                raise ValidationError(f"MCP header {name} prefix is not allowed")
            if header.suffix not in _ALLOWED_HEADER_SUFFIXES:
                raise ValidationError(f"MCP header {name} suffix is not allowed")

    def _validate_tool(self, tool: McpToolSpec, *, schema_version: int) -> None:
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
        self._validate_json_schema(
            tool.input_schema,
            "input_schema",
            modern=schema_version == 2,
        )
        self._validate_json_value(tool.metadata, "tool metadata")
        if schema_version == 2:
            self._validate_no_reserved_mcp_meta(tool.metadata, "tool metadata")

    def _header_specs(self, value: Any) -> dict[str, McpHeaderSpec]:
        if not isinstance(value, dict):
            raise ValidationError("MCP headers must be an object")
        headers: dict[str, McpHeaderSpec] = {}
        for name, spec in value.items():
            if type(name) is not str:
                raise ValidationError("MCP header names must be strings")
            if not isinstance(spec, dict):
                raise ValidationError(f"MCP header {name} must be an object")
            _reject_unknown_fields(
                spec,
                _HEADER_FIELDS,
                context=f"MCP header {name}",
            )
            prefix = spec.get("prefix", "")
            suffix = spec.get("suffix", "")
            if type(prefix) is not str or type(suffix) is not str:
                raise ValidationError(
                    f"MCP header {name} prefix and suffix must be strings"
                )
            headers[name] = McpHeaderSpec(
                env=self._required_string(spec, "env", f"MCP header {name}"),
                prefix=prefix,
                suffix=suffix,
            )
        return headers

    def _required(self, value: dict[str, Any], key: str, context: str) -> Any:
        if key not in value:
            raise ValidationError(f"{context} requires {key}")
        return value[key]

    def _required_string(
        self,
        value: dict[str, Any],
        key: str,
        context: str,
    ) -> str:
        selected = self._required(value, key, context)
        if type(selected) is not str:
            raise ValidationError(f"{context} {key} must be a string")
        return selected

    @staticmethod
    def _mapping_field(
        value: dict[str, Any],
        key: str,
        context: str,
    ) -> dict[str, Any]:
        if key not in value:
            return {}
        selected = value[key]
        if not isinstance(selected, dict):
            raise ValidationError(f"{context} {key} must be an object")
        return dict(selected)

    @staticmethod
    def _list_field(
        value: dict[str, Any],
        key: str,
        context: str,
    ) -> list[Any]:
        if key not in value:
            return []
        selected = value[key]
        if not isinstance(selected, list):
            raise ValidationError(f"{context} {key} must be an array")
        return list(selected)

    def _validate_header_name(self, name: str, *, schema_version: int) -> None:
        lowered = name.lower()
        if len(name) > self.config.mcp.header_name_max_chars or not _HEADER_PATTERN.match(name):
            raise ValidationError(f"invalid MCP header name: {name!r}")
        forbidden_headers = (
            _MODERN_FORBIDDEN_HEADERS
            if schema_version == 2
            else _LEGACY_FORBIDDEN_HEADERS
        )
        if lowered in forbidden_headers or (
            schema_version == 2 and lowered.startswith("mcp-param-")
        ):
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
        pinned_stdio_environment: Mapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        selected_host_environment = self._runtime_environment_input_snapshot(
            server,
            host_environment=host_environment,
            pinned_stdio_environment=pinned_stdio_environment,
        )
        return self._runtime_environment_from_host(
            server,
            selected_host_environment,
            pinned_stdio_environment=pinned_stdio_environment,
        )

    def _runtime_environment_input_snapshot(
        self,
        server: McpServerSpec,
        *,
        host_environment: Mapping[str, str] | None = None,
        pinned_stdio_environment: Mapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        source = os.environ if host_environment is None else host_environment
        pinned_stdio = pinned_stdio_environment or {}
        names: set[str] = set()
        pinned_host: dict[str, str] = {}
        if server.transport == "stdio" and server.stdio is not None:
            names.update(_MCP_PLATFORM_ENV_KEYS)
            pinned_host = {
                server.stdio.env[child_name]: value
                for child_name, value in pinned_stdio.items()
                if child_name in server.stdio.env
            }
            names.update(
                host_name
                for host_name in server.stdio.env.values()
                if host_name not in pinned_host
            )
        elif server.transport == "streamable_http" and server.http is not None:
            names.update(header.env for header in server.http.headers.values())
        names.difference_update(pinned_host)
        selected = dict(pinned_host)
        for name in sorted(names):
            resolved = source.get(name)
            if resolved is not None:
                selected[name] = resolved
        return MappingProxyType(selected)

    def _runtime_environment_from_host(
        self,
        server: McpServerSpec,
        host_environment: Mapping[str, str],
        *,
        pinned_stdio_environment: Mapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        resolved_environment: dict[str, str] = {}
        pinned_stdio = pinned_stdio_environment or {}
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
                resolved = pinned_stdio.get(child_name)
                if resolved is None:
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
        try:
            parsed = urlsplit(url)
        except ValueError as exc:
            raise ValidationError("MCP HTTP URL is invalid") from exc
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValidationError("MCP HTTP URL has invalid port") from exc
        if parsed.scheme not in {"http", "https"}:
            raise ValidationError("MCP HTTP URL must use http or https")
        if parsed.username is not None or parsed.password is not None:
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

    def _validate_runtime_resolution(
        self,
        server: McpServerSpec,
        *,
        deadline: float | None = None,
    ) -> tuple[str, ...]:
        if server.http is None:
            return ()
        parsed = urlsplit(server.http.url)
        host = parsed.hostname
        if not host:
            raise ValidationError("MCP HTTP URL must include a host")
        if host in _LOCAL_HTTP_HOSTS:
            return ()
        selected_deadline = (
            time.monotonic() + server.timeout_s if deadline is None else deadline
        )
        try:
            addresses = _allowed_mcp_connect_addresses(
                host,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                deadline=selected_deadline,
            )
        except TimeoutError as exc:
            raise ProviderHostError(
                code="mcp_dns_timeout",
                error_type=type(exc).__name__,
                correlation_id=new_id("corr"),
            ) from None
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
        if type(value) not in {int, float}:
            raise ValidationError(f"MCP {field} must be a number")
        selected = float(value)
        if not math.isfinite(selected) or selected <= 0:
            raise ValidationError(f"MCP {field} must be > 0")
        return selected

    def _coerce_positive_int(self, value: Any, field: str) -> int:
        if type(value) is not int:
            raise ValidationError(f"MCP {field} must be an integer")
        selected = value
        if selected <= 0:
            raise ValidationError(f"MCP {field} must be > 0")
        return selected

    def _validate_json_value(self, value: Any, field: str) -> None:
        try:
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValidationError(f"MCP {field} must be JSON-serializable") from exc

    def _validate_no_reserved_mcp_meta(self, value: Any, field: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str:
                    raise ValidationError(
                        f"MCP {field} object keys must be strings"
                    )
                if key == "_meta" or key.startswith("io.modelcontextprotocol/"):
                    raise ValidationError(
                        f"MCP {field} must not define protocol-reserved _meta"
                    )
                self._validate_no_reserved_mcp_meta(item, field)
        elif isinstance(value, list):
            for item in value:
                self._validate_no_reserved_mcp_meta(item, field)

    def _validate_json_schema(
        self,
        schema: dict[str, Any],
        field: str,
        *,
        modern: bool = False,
    ) -> None:
        if not schema:
            return
        self._validate_json_value(schema, field)
        if modern:
            self._validate_v2_json_schema_safety(schema, field)
        try:
            jsonschema_validator_for(schema).check_schema(schema)
        except JsonSchemaSchemaError as exc:
            raise ValidationError(f"MCP {field} is not a valid JSON Schema") from exc

    def _validate_arguments_against_schema(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
    ) -> None:
        if not tool.input_schema:
            return
        if server.schema_version == 2:
            self._validate_v2_json_schema_safety(
                tool.input_schema,
                "input_schema",
            )
        try:
            validator = jsonschema_validator_for(tool.input_schema)
            validator.check_schema(tool.input_schema)
            validator(tool.input_schema).validate(arguments)
        except JsonSchemaValidationError as exc:
            raise ValidationError(f"MCP tool arguments failed schema validation: {exc.message}") from exc
        except JsonSchemaSchemaError as exc:
            raise ValidationError("MCP tool input_schema is invalid") from exc

    def _validate_v2_json_schema_safety(
        self,
        schema: dict[str, Any],
        field: str,
    ) -> None:
        """Validate the bounded JSON Schema 2020-12 subset used by Manifest v2."""

        if schema.get("type") != "object":
            raise ValidationError(
                f"MCP {field} schema_version 2 root type must be object"
            )
        node_count = 0
        combinator_expansion = 1
        local_refs: list[tuple[dict[str, Any], str]] = []

        def walk(value: Any, *, depth: int, path: str) -> None:
            nonlocal node_count, combinator_expansion
            if depth > self.config.mcp.schema_max_depth:
                raise ValidationError(
                    f"MCP {field} exceeds schema depth={self.config.mcp.schema_max_depth}"
                )
            node_count += 1
            if node_count > self.config.mcp.schema_max_nodes:
                raise ValidationError(
                    f"MCP {field} exceeds schema nodes={self.config.mcp.schema_max_nodes}"
                )
            if isinstance(value, dict):
                if "if" in value and ("then" in value or "else" in value):
                    combinator_expansion *= 2
                    if (
                        combinator_expansion
                        > self.config.mcp.schema_max_composition_expansions
                    ):
                        raise ValidationError(
                            "MCP "
                            f"{field} exceeds combinator expansion="
                            f"{self.config.mcp.schema_max_composition_expansions}"
                        )
                for key, item in value.items():
                    if key in _MCP_V2_DYNAMIC_REFERENCE_KEYS:
                        raise ValidationError(
                            f"MCP {field} does not allow dynamic or recursive references"
                        )
                    if key == "$ref":
                        if not isinstance(item, str) or not item.startswith("#"):
                            raise ValidationError(
                                f"MCP {field} external $ref is not allowed"
                            )
                        local_refs.append((value, item))
                    if key in {"allOf", "anyOf", "oneOf"}:
                        if not isinstance(item, list):
                            raise ValidationError(
                                f"MCP {field} {key} must be an array"
                            )
                        combinator_expansion *= max(1, len(item))
                        if (
                            combinator_expansion
                            > self.config.mcp.schema_max_composition_expansions
                        ):
                            raise ValidationError(
                                "MCP "
                                f"{field} exceeds combinator expansion="
                                f"{self.config.mcp.schema_max_composition_expansions}"
                            )
                    walk(item, depth=depth + 1, path=f"{path}/{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, depth=depth + 1, path=f"{path}/{index}")

        walk(schema, depth=0, path="#")
        for _source, reference in local_refs:
            self._resolve_local_schema_ref(schema, reference, field)
        self._reject_cyclic_schema_refs(schema, local_refs, field)

    @staticmethod
    def _resolve_local_schema_ref(
        schema: dict[str, Any],
        reference: str,
        field: str,
    ) -> Any:
        if reference == "#":
            return schema
        if not reference.startswith("#/"):
            raise ValidationError(
                f"MCP {field} local $ref must use a JSON Pointer fragment"
            )
        selected: Any = schema
        for raw_part in reference[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if isinstance(selected, dict) and part in selected:
                selected = selected[part]
                continue
            if isinstance(selected, list) and part.isdigit():
                index = int(part)
                if index < len(selected):
                    selected = selected[index]
                    continue
            raise ValidationError(f"MCP {field} contains an unresolved local $ref")
        return selected

    def _reject_cyclic_schema_refs(
        self,
        schema: dict[str, Any],
        references: list[tuple[dict[str, Any], str]],
        field: str,
    ) -> None:
        # Build one bounded graph for both containment and reference edges.
        # The former implementation recursively scanned a reference target for
        # every `$ref`, turning a valid schema with many references to one large
        # shared definition into quadratic work.
        nodes, edges = self._schema_containment_graph(schema)
        self._add_schema_reference_edges(
            schema,
            references,
            field,
            nodes=nodes,
            edges=edges,
        )
        self._validate_schema_reference_graph(nodes, edges, field)

    @staticmethod
    def _schema_containment_graph(
        schema: dict[str, Any],
    ) -> tuple[
        dict[int, dict[str, Any] | list[Any]],
        dict[int, list[tuple[int, int]]],
    ]:
        nodes: dict[int, dict[str, Any] | list[Any]] = {}
        edges: dict[int, list[tuple[int, int]]] = {}
        pending: list[dict[str, Any] | list[Any]] = [schema]
        while pending:
            value = pending.pop()
            identity = id(value)
            if identity in nodes:
                continue
            nodes[identity] = value
            edges[identity] = []
            children = (
                (item for key, item in value.items() if key != "$ref")
                if isinstance(value, dict)
                else iter(value)
            )
            for child in children:
                if not isinstance(child, (dict, list)):
                    continue
                child_identity = id(child)
                edges[identity].append((child_identity, 0))
                pending.append(child)
        return nodes, edges

    def _add_schema_reference_edges(
        self,
        schema: dict[str, Any],
        references: list[tuple[dict[str, Any], str]],
        field: str,
        *,
        nodes: dict[int, dict[str, Any] | list[Any]],
        edges: dict[int, list[tuple[int, int]]],
    ) -> None:
        for source, reference in references:
            target = self._resolve_local_schema_ref(schema, reference, field)
            if not isinstance(target, (dict, list)):
                continue
            source_identity = id(source)
            target_identity = id(target)
            nodes.setdefault(source_identity, source)
            nodes.setdefault(target_identity, target)
            edges.setdefault(source_identity, []).append((target_identity, 1))
            edges.setdefault(target_identity, [])

    def _validate_schema_reference_graph(
        self,
        nodes: dict[int, dict[str, Any] | list[Any]],
        edges: dict[int, list[tuple[int, int]]],
        field: str,
    ) -> None:
        indegree = {identity: 0 for identity in nodes}
        for outgoing in edges.values():
            for target_identity, _ref_edge in outgoing:
                indegree[target_identity] += 1
        ready = deque(
            identity for identity, degree in indegree.items() if degree == 0
        )
        ref_hops = {identity: 0 for identity in nodes}
        processed = 0
        while ready:
            identity = ready.popleft()
            processed += 1
            for target_identity, ref_edge in edges[identity]:
                candidate_hops = ref_hops[identity] + ref_edge
                if candidate_hops > ref_hops[target_identity]:
                    ref_hops[target_identity] = candidate_hops
                if (
                    ref_hops[target_identity]
                    > self.config.mcp.schema_max_ref_hops
                ):
                    raise ValidationError(
                        "MCP "
                        f"{field} exceeds local $ref hops="
                        f"{self.config.mcp.schema_max_ref_hops}"
                    )
                indegree[target_identity] -= 1
                if indegree[target_identity] == 0:
                    ready.append(target_identity)

        if processed != len(nodes):
            raise ValidationError(
                f"MCP {field} recursive local $ref is not allowed"
            )

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
            "protocol_mode": self._effective_protocol_mode(server).value,
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
            "rollback_status": (
                tool.rollback_status
                if tool.rollback_status is not None
                else default_external_effect_rollback_status(
                    ExternalEffectRollbackClass(tool.rollback_class)
                ).value
            ),
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
                "schema_matches_manifest": not tool.input_schema
                or _json_values_equivalent(
                    live.input_schema,
                    tool.input_schema,
                ),
            }
        return payload

    def _profile_json(self, profile: Any) -> dict[str, Any]:
        return to_jsonable(profile)
