from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from agent_libos.models.base import StrEnum
from agent_libos.utils.serde import dumps, to_jsonable


class McpCallStatus(StrEnum):
    OK = "ok"
    MCP_ERROR = "mcp_error"
    TRANSPORT_ERROR = "transport_error"
    INVALID_RESPONSE = "invalid_response"
    RESPONSE_TOO_LARGE = "response_too_large"
    INPUT_REQUIRED_UNSUPPORTED = "input_required_unsupported"


class McpProtocolMode(StrEnum):
    """Host-selected MCP wire-protocol behavior for a registered server."""

    LEGACY = "legacy"
    AUTO = "auto"
    REVISION_2026_07_28 = "2026-07-28"


class McpProtocolEra(StrEnum):
    """Normalized MCP protocol era negotiated for one provider operation."""

    LEGACY = "legacy"
    MODERN = "modern"


class McpExchangePhase(StrEnum):
    """One independently accounted MCP protocol phase."""

    SERVER_DISCOVER = "server/discover"
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"


@dataclass(frozen=True)
class McpExchangeReceipt:
    """Bounded accounting projection for one MCP protocol exchange."""

    phase: McpExchangePhase
    request_bytes: int = 0
    response_bytes: int = 0
    duration_s: float = 0.0
    call_started: bool = False


@dataclass(frozen=True)
class McpConnectionInfo:
    """Sanitized, operation-local MCP negotiation metadata."""

    protocol_mode: McpProtocolMode
    protocol_era: McpProtocolEra
    protocol_revision: str
    sessionless: bool
    fallback_used: bool = False
    server_name: str | None = None
    server_version: str | None = None
    capabilities: tuple[str, ...] = ()
    unsupported_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class McpHeaderSpec:
    env: str
    prefix: str = ""
    suffix: str = ""


@dataclass(frozen=True)
class McpStdioTransportSpec:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass(frozen=True)
class McpHttpTransportSpec:
    url: str
    headers: dict[str, McpHeaderSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class McpToolSpec:
    tool_id: str
    mcp_name: str
    right: str
    rollback_class: str
    state_mutation: bool
    information_flow: bool
    rollback_status: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpServerSpec:
    schema_version: int
    server_id: str
    transport: str
    tools: list[McpToolSpec]
    timeout_s: float
    max_request_bytes: int
    max_response_bytes: int
    stdio: McpStdioTransportSpec | None = None
    http: McpHttpTransportSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    protocol_mode: McpProtocolMode | None = None

    def tool_by_id(self, tool_id: str) -> McpToolSpec | None:
        return next((tool for tool in self.tools if tool.tool_id == tool_id), None)


def mcp_runtime_secret_values(
    server: McpServerSpec,
    runtime_environment: Mapping[str, str] | None,
) -> tuple[str, ...]:
    """Select exact manifest-declared credential values for public redaction.

    The operation snapshot is already resolved to HTTP header names or stdio
    child-environment names. Platform bootstrap variables are intentionally
    excluded, while both a composed HTTP value and its raw env-backed portion
    are retained so a peer cannot leak a credential by reflecting either
    representation through diagnostic identity fields.
    """

    if runtime_environment is None:
        return ()
    selected: set[str] = set()
    if server.transport == "streamable_http" and server.http is not None:
        for header_name, header in server.http.headers.items():
            resolved = runtime_environment.get(header_name)
            if type(resolved) is not str or not resolved:
                continue
            selected.add(resolved)
            start = len(header.prefix)
            end = len(resolved) - len(header.suffix) if header.suffix else len(resolved)
            if (
                resolved.startswith(header.prefix)
                and resolved.endswith(header.suffix)
                and end >= start
            ):
                raw = resolved[start:end]
                if raw:
                    selected.add(raw)
    elif server.transport == "stdio" and server.stdio is not None:
        for child_name in server.stdio.env:
            resolved = runtime_environment.get(child_name)
            if type(resolved) is str and resolved:
                selected.add(resolved)
    return tuple(sorted(selected, key=lambda item: (-len(item), item)))


def mcp_server_spec_to_jsonable(server: McpServerSpec) -> dict[str, Any]:
    """Return the versioned canonical projection of an MCP server spec.

    Manifest v1 predates ``protocol_mode``. Its absent optional field must not
    alter existing registry, approval, or Sink hashes. Manifest v2 retains the
    explicitly selected string-valued protocol mode.
    """

    value = to_jsonable(server)
    if not isinstance(value, dict):  # pragma: no cover - dataclass invariant
        raise TypeError("MCP server spec must encode as an object")
    selected = dict(value)
    if server.schema_version == 1:
        selected.pop("protocol_mode", None)
    elif server.protocol_mode is not None:
        selected["protocol_mode"] = server.protocol_mode.value
    return selected


def canonical_mcp_server_spec_json(server: McpServerSpec) -> str:
    """Return stable JSON for MCP registry identity and persistence."""

    return dumps(mcp_server_spec_to_jsonable(server))


@dataclass(frozen=True)
class McpProviderTool:
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpToolListResult:
    server_id: str
    tools: list[McpProviderTool]
    response_bytes: int
    duration_s: float
    connection: McpConnectionInfo | None = None
    receipts: tuple[McpExchangeReceipt, ...] = ()


@dataclass(frozen=True)
class McpProviderDiscoveryResult:
    """Provider-level result for a governed MCP discovery exchange."""

    connection: McpConnectionInfo
    request_bytes: int = 0
    response_bytes: int = 0
    duration_s: float = 0.0
    receipts: tuple[McpExchangeReceipt, ...] = ()


@dataclass(frozen=True)
class McpDiscoveryResult:
    """Public Runtime projection of an MCP discovery exchange."""

    server_id: str
    connection: McpConnectionInfo
    request_bytes: int = 0
    response_bytes: int = 0
    duration_s: float = 0.0
    receipts: tuple[McpExchangeReceipt, ...] = ()


@dataclass(frozen=True)
class McpProviderCallResult:
    content: Any = None
    structured_content: Any = None
    is_error: bool = False
    error: str | None = None
    response_bytes: int = 0
    duration_s: float = 0.0
    too_large: bool = False
    error_type: str | None = None
    correlation_id: str | None = None
    list_request_bytes: int = 0
    list_response_bytes: int = 0
    call_request_bytes: int = 0
    call_response_bytes: int = 0
    call_started: bool = False
    connection: McpConnectionInfo | None = None
    receipts: tuple[McpExchangeReceipt, ...] = ()


@dataclass(frozen=True)
class McpCallResult:
    server_id: str
    tool_id: str
    mcp_name: str
    status: McpCallStatus
    ok: bool
    result: Any = None
    error: dict[str, Any] | None = None
    response_bytes: int = 0
    duration_s: float = 0.0
    connection: McpConnectionInfo | None = None
    receipts: tuple[McpExchangeReceipt, ...] = ()
