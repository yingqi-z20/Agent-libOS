from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    CapabilityRight,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    McpCallStatus,
    McpConnectionInfo,
    McpExchangePhase,
    McpExchangeReceipt,
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
    TaskRunRetention,
    TaskRunStatus,
    canonical_mcp_server_spec_json,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    ProviderHostError,
    ValidationError,
)
from agent_libos.primitives.mcp import McpPrimitive
from agent_libos.substrate.local import SdkMcpProvider, _mcp_auto_fallback_allowed
from agent_libos.utils.serde import dumps, to_jsonable


_ABSENT = object()


class _FallbackError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(str(code))
        self.code = code


class _HttpFallbackEvidence:
    def __init__(
        self,
        *,
        status: int,
        method: str = "server/discover",
        legacy_signal: bool = True,
    ) -> None:
        self.last_response_status = status
        self.last_request_method = method
        self.last_legacy_400_signal = legacy_signal


def test_auto_fallback_policy_rejects_ambiguous_failures_without_sdk() -> None:
    auto = McpProtocolMode.AUTO

    for status in (401, 403, 500):
        assert not _mcp_auto_fallback_allowed(
            _FallbackError(-32603),
            mode=auto,
            transport="streamable_http",
            http_policy_transport=_HttpFallbackEvidence(status=status),
        )
    for modern_error in (-32020, -32021, -32022):
        assert not _mcp_auto_fallback_allowed(
            _FallbackError(modern_error),
            mode=auto,
            transport="streamable_http",
            http_policy_transport=_HttpFallbackEvidence(status=400),
        )
    assert not _mcp_auto_fallback_allowed(
        _FallbackError(-32603),
        mode=auto,
        transport="streamable_http",
        http_policy_transport=_HttpFallbackEvidence(
            status=400,
            legacy_signal=False,
        ),
    )
    assert _mcp_auto_fallback_allowed(
        _FallbackError(-32603),
        mode=auto,
        transport="streamable_http",
        http_policy_transport=_HttpFallbackEvidence(status=400),
    )
    assert _mcp_auto_fallback_allowed(
        _FallbackError(-32601),
        mode=auto,
        transport="stdio",
        http_policy_transport=None,
    )
    assert not _mcp_auto_fallback_allowed(
        _FallbackError(-32603),
        mode=auto,
        transport="stdio",
        http_policy_transport=None,
    )
    assert not _mcp_auto_fallback_allowed(
        _FallbackError(-32601),
        mode=McpProtocolMode.REVISION_2026_07_28,
        transport="stdio",
        http_policy_transport=None,
    )


def _tool(
    *,
    input_schema: dict[str, Any] | None = None,
    state_mutation: bool = False,
) -> dict[str, Any]:
    return {
        "tool_id": "echo",
        "mcp_name": "demo.echo",
        "right": "write" if state_mutation else "read",
        "rollback_class": (
            "rollbackable" if state_mutation else "no_rollback_required"
        ),
        "state_mutation": state_mutation,
        "information_flow": True,
        "input_schema": input_schema or {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "additionalProperties": False,
        },
        "metadata": {},
    }


def _manifest(
    server_id: str = "modern",
    *,
    schema_version: int = 2,
    protocol_mode: object = "auto",
    headers: dict[str, Any] | None = None,
    input_schema: dict[str, Any] | None = None,
    server_metadata: dict[str, Any] | None = None,
    tool_metadata: dict[str, Any] | None = None,
    state_mutation: bool = False,
) -> dict[str, Any]:
    tool = _tool(input_schema=input_schema, state_mutation=state_mutation)
    if tool_metadata is not None:
        tool["metadata"] = tool_metadata
    result: dict[str, Any] = {
        "schema_version": schema_version,
        "server_id": server_id,
        "transport": "streamable_http",
        "http": {
            "url": "http://127.0.0.1:8765/mcp",
            "headers": dict(headers or {}),
        },
        "tools": [tool],
        "timeout_s": 5.0,
        "max_request_bytes": 65_536,
        "max_response_bytes": 1_048_576,
        "metadata": dict(server_metadata or {}),
    }
    if protocol_mode is not _ABSENT:
        result["protocol_mode"] = protocol_mode
    return result


def _connection(
    *,
    mode: McpProtocolMode = McpProtocolMode.AUTO,
    era: McpProtocolEra = McpProtocolEra.MODERN,
    revision: str = "2026-07-28",
    sessionless: bool = True,
    fallback_used: bool = False,
) -> McpConnectionInfo:
    return McpConnectionInfo(
        protocol_mode=mode,
        protocol_era=era,
        protocol_revision=revision,
        sessionless=sessionless,
        fallback_used=fallback_used,
        server_name="contract-fixture",
        server_version="1.0",
        capabilities=("tools",),
        unsupported_capabilities=("prompts",),
    )


def _provider_tool_bytes(tools: list[McpProviderTool]) -> int:
    return len(dumps([to_jsonable(tool) for tool in tools]).encode("utf-8"))


def _provider_call_bytes(content: Any, structured_content: Any) -> int:
    return len(
        dumps(
            {
                "content": content,
                "structured_content": structured_content,
            }
        ).encode("utf-8")
    )


def _negotiation_receipts(
    connection: McpConnectionInfo,
) -> tuple[McpExchangeReceipt, ...]:
    if connection.protocol_mode is McpProtocolMode.LEGACY:
        return (
            McpExchangeReceipt(
                phase=McpExchangePhase.INITIALIZE,
                call_started=True,
            ),
        )
    discover = McpExchangeReceipt(
        phase=McpExchangePhase.SERVER_DISCOVER,
        call_started=True,
    )
    if not connection.fallback_used:
        return (discover,)
    return (
        discover,
        McpExchangeReceipt(
            phase=McpExchangePhase.INITIALIZE,
            call_started=True,
        ),
    )


def _complete_call_result(
    *,
    connection: McpConnectionInfo,
    response_bytes: int,
    is_error: bool = False,
    error: str | None = None,
    error_type: str | None = None,
    content: Any = None,
    structured_content: Any = None,
    call_request_bytes: int = 32,
) -> McpProviderCallResult:
    list_response_bytes = _provider_tool_bytes(
        [
            McpProviderTool(
                name="demo.echo",
                description="Echo",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "additionalProperties": False,
                },
            )
        ]
    )
    list_receipt = McpExchangeReceipt(
        phase=McpExchangePhase.TOOLS_LIST,
        request_bytes=16,
        response_bytes=list_response_bytes,
        call_started=True,
    )
    call_receipt = McpExchangeReceipt(
        phase=McpExchangePhase.TOOLS_CALL,
        request_bytes=call_request_bytes,
        response_bytes=response_bytes,
        duration_s=0.02,
        call_started=True,
    )
    return McpProviderCallResult(
        content=content,
        structured_content=structured_content,
        is_error=is_error,
        error=error,
        error_type=error_type,
        response_bytes=response_bytes,
        duration_s=0.02,
        list_request_bytes=list_receipt.request_bytes,
        list_response_bytes=list_receipt.response_bytes,
        call_request_bytes=call_request_bytes,
        call_response_bytes=response_bytes,
        call_started=True,
        connection=connection,
        receipts=(*_negotiation_receipts(connection), list_receipt, call_receipt),
    )


def _pre_call_failure_result(*, error_type: str) -> McpProviderCallResult:
    connection = _connection()
    list_receipt = McpExchangeReceipt(
        phase=McpExchangePhase.TOOLS_LIST,
        request_bytes=17,
        response_bytes=29,
        duration_s=0.01,
        call_started=True,
    )
    return McpProviderCallResult(
        error="MCP failed before tools/call dispatch",
        error_type=error_type,
        correlation_id="corr-pre-call",
        duration_s=0.02,
        list_request_bytes=list_receipt.request_bytes,
        list_response_bytes=list_receipt.response_bytes,
        call_request_bytes=0,
        call_response_bytes=0,
        response_bytes=0,
        call_started=False,
        connection=connection,
        receipts=(*_negotiation_receipts(connection), list_receipt),
    )


def _negotiation_failure_result(*, error_type: str) -> McpProviderCallResult:
    discover_receipt = McpExchangeReceipt(
        phase=McpExchangePhase.SERVER_DISCOVER,
        request_bytes=19,
        response_bytes=31,
        duration_s=0.01,
        call_started=True,
    )
    return McpProviderCallResult(
        error="MCP negotiation failed before a connection was established",
        error_type=error_type,
        correlation_id="corr-negotiation",
        duration_s=0.02,
        call_started=False,
        connection=None,
        receipts=(discover_receipt,),
    )


def _post_call_wire_error() -> tuple[
    RuntimeError,
    McpConnectionInfo,
    tuple[McpExchangeReceipt, ...],
]:
    connection = _connection()
    completed = _complete_call_result(
        connection=connection,
        response_bytes=23,
    )
    error = RuntimeError("SSE response ended after tools/call dispatch")
    setattr(error, "_agent_libos_mcp_wire_evidence", True)
    setattr(error, "_agent_libos_mcp_receipts", completed.receipts)
    setattr(error, "_agent_libos_mcp_connection", connection)
    return error, connection, completed.receipts


class _ModernFakeProvider:
    supports_mcp_modern_protocol = True

    def __init__(self) -> None:
        self.discover_calls: list[str] = []
        self.list_calls: list[str] = []
        self.validate_calls = 0
        self.call_calls = 0
        self.discovery_result = McpProviderDiscoveryResult(
            connection=_connection(),
            request_bytes=128,
            response_bytes=512,
            duration_s=0.01,
            receipts=(
                McpExchangeReceipt(
                    phase=McpExchangePhase.SERVER_DISCOVER,
                    request_bytes=128,
                    response_bytes=512,
                    duration_s=0.01,
                    call_started=True,
                ),
            ),
        )
        self.tools = [
            McpProviderTool(
                name="demo.echo",
                description="Echo",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "additionalProperties": False,
                },
            )
        ]
        self.list_receipts: tuple[McpExchangeReceipt, ...] | None = None
        self.call_result: McpProviderCallResult | None = None
        self.on_discover: Any = None

    def discover(self, server: McpServerSpec, **_kwargs: Any) -> McpProviderDiscoveryResult:
        self.discover_calls.append(server.server_id)
        if self.on_discover is not None:
            self.on_discover()
        return self.discovery_result

    def list_tools(self, server: McpServerSpec, **_kwargs: Any) -> McpToolListResult:
        self.list_calls.append(server.server_id)
        response_bytes = _provider_tool_bytes(self.tools)
        connection = _connection(
            mode=server.protocol_mode or McpProtocolMode.LEGACY,
            era=(
                McpProtocolEra.LEGACY
                if server.protocol_mode == McpProtocolMode.LEGACY
                else McpProtocolEra.MODERN
            ),
            revision=(
                "2025-11-25"
                if server.protocol_mode == McpProtocolMode.LEGACY
                else "2026-07-28"
            ),
            sessionless=server.protocol_mode != McpProtocolMode.LEGACY,
        )
        return McpToolListResult(
            server_id=server.server_id,
            tools=list(self.tools),
            response_bytes=response_bytes,
            duration_s=0.01,
            connection=connection,
            receipts=(
                self.list_receipts
                if self.list_receipts is not None
                else (
                    *_negotiation_receipts(connection),
                    McpExchangeReceipt(
                        phase=McpExchangePhase.TOOLS_LIST,
                        response_bytes=response_bytes,
                        call_started=True,
                    ),
                )
            ),
        )

    def validate_and_call(
        self,
        server: McpServerSpec,
        _tool_spec: McpToolSpec,
        arguments: dict[str, Any],
        **_kwargs: Any,
    ) -> McpProviderCallResult:
        self.validate_calls += 1
        if self.call_result is not None:
            return self.call_result
        content = [{"type": "text", "text": "ok"}]
        structured = {"echo": dict(arguments)}
        response_bytes = _provider_call_bytes(content, structured)
        connection = _connection(mode=server.protocol_mode or McpProtocolMode.LEGACY)
        return _complete_call_result(
            content=content,
            structured_content=structured,
            response_bytes=response_bytes,
            connection=connection,
        )

    def call_tool(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> McpProviderCallResult:
        self.call_calls += 1
        return self.validate_and_call(server, tool, arguments, **kwargs)

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        _result: Any,
    ) -> ExternalEffectClassification:
        if operation in {"discover", "list_tools"}:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
            )
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass(
                str(context["rollback_class"])
            ),
            rollback_status=(
                ExternalEffectRollbackStatus.NOT_APPLIED
                if bool(context["state_mutation"])
                else ExternalEffectRollbackStatus.NOT_REQUIRED
            ),
            state_mutation=bool(context["state_mutation"]),
            information_flow=bool(context["information_flow"]),
        )


class _SingleMcpTaskRunClient:
    def __init__(self, server_id: str) -> None:
        self.server_id = server_id
        self.calls = 0

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("an unsafe MCP continuation reached the LLM")
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "mcp-input-required",
                    "name": "call_mcp_tool",
                    "arguments": json.dumps(
                        {
                            "server_id": self.server_id,
                            "tool_id": "echo",
                            "arguments": {"text": "hello"},
                        },
                        sort_keys=True,
                    ),
                }
            ],
        )


def _task_run_config() -> Any:
    return replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )


class _LegacyOnlySpyProvider:
    """Old provider contract with no Manifest v2 opt-in marker."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_tools(self, *_args: Any, **_kwargs: Any) -> McpToolListResult:
        self.calls.append("list_tools")
        raise AssertionError("unmarked provider must not be dispatched")

    def validate_and_call(self, *_args: Any, **_kwargs: Any) -> McpProviderCallResult:
        self.calls.append("validate_and_call")
        raise AssertionError("unmarked provider must not be dispatched")

    def call_tool(self, *_args: Any, **_kwargs: Any) -> McpProviderCallResult:
        self.calls.append("call_tool")
        raise AssertionError("unmarked provider must not be dispatched")


def test_manifest_v1_canonical_registry_and_sink_hashes_are_stable() -> None:
    tool = McpToolSpec(
        tool_id="echo",
        mcp_name="demo.echo",
        right="read",
        rollback_class="no_rollback_required",
        state_mutation=False,
        information_flow=True,
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "additionalProperties": False,
        },
    )
    spec = McpServerSpec(
        schema_version=1,
        server_id="stable-v1",
        transport="streamable_http",
        tools=[tool],
        timeout_s=5.0,
        max_request_bytes=65_536,
        max_response_bytes=1_048_576,
        http=McpHttpTransportSpec(url="https://api.example.test/mcp"),
    )
    canonical = canonical_mcp_server_spec_json(spec)

    assert "protocol_mode" not in canonical
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
        "82396f6d2041ebd6489b129e84d2917def76e04357da72832bcabe4e938bca64"
    )
    assert McpPrimitive._server_spec_sha256(spec) == (
        "82396f6d2041ebd6489b129e84d2917def76e04357da72832bcabe4e938bca64"
    )

    runtime = Runtime.open(":memory:")
    try:
        assert runtime.mcp._server_identity_sha256(
            spec,
            tool,
            stdio_executable=None,
        ) == "942c4207b653419a1c97c7c16aefbb365202dd9ef14fb23676b265d0b1df2c89"
        assert runtime.mcp._list_tools_identity_sha256(
            spec,
            stdio_executable=None,
        ) == "1e6104475f7dc4c7f534d079d76b224693c6f702ae5c733a79fa21b825cebf74"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (_manifest(schema_version=1, protocol_mode="legacy"), "must omit"),
        (_manifest(schema_version=1, protocol_mode=None), "must omit"),
        (_manifest(schema_version=2, protocol_mode=_ABSENT), "requires"),
        (_manifest(schema_version=2, protocol_mode="future"), "protocol_mode"),
        (_manifest(schema_version=3, protocol_mode="auto"), "schema_version"),
    ],
)
def test_manifest_protocol_mode_is_explicit_and_versioned(
    manifest: dict[str, Any],
    message: str,
) -> None:
    runtime = Runtime.open(":memory:")
    try:
        with pytest.raises(ValidationError, match=message):
            runtime.mcp.register_server(
                manifest,
                actor="test",
                require_capability=False,
            )
        assert runtime.store.list_mcp_servers() == []
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["legacy", "auto", "2026-07-28"])
def test_manifest_v2_accepts_only_the_three_locked_modes(mode: str) -> None:
    runtime = Runtime.open(":memory:")
    try:
        registered = runtime.mcp.register_server(
            _manifest(server_id=f"mode-{mode.replace('-', '_')}", protocol_mode=mode),
            actor="test",
            require_capability=False,
        )
        assert registered["schema_version"] == 2
        assert registered["protocol_mode"] == mode
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "header_name",
    [
        "Accept",
        "accept-CHARSET",
        "ACCEPT-ENCODING",
        "Accept-Language",
        "content-ENCODING",
        "Content-Language",
        "CONTENT-TYPE",
        "MCP-Protocol-Version",
        "mCp-MeThOd",
        "Mcp-Name",
        "Mcp-Param-Cursor",
        "MCP-SESSION-ID",
        "Last-Event-ID",
        "TraceParent",
        "traceSTATE",
        "Baggage",
    ],
)
def test_manifest_v2_rejects_host_overrides_of_protocol_headers(
    header_name: str,
) -> None:
    runtime = Runtime.open(":memory:")
    try:
        with pytest.raises(ValidationError, match="forbidden"):
            runtime.mcp.register_server(
                _manifest(
                    headers={
                        header_name: {
                            "env": "AGENT_LIBOS_MCP_TEST_TOKEN",
                            "prefix": "",
                        }
                    }
                ),
                actor="test",
                require_capability=False,
            )
        assert runtime.store.list_mcp_servers() == []
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "location",
    ["server", "tool"],
)
def test_manifest_v2_rejects_reserved_protocol_meta(location: str) -> None:
    kwargs = (
        {"server_metadata": {"_meta": {"traceparent": "secret"}}}
        if location == "server"
        else {"tool_metadata": {"_meta": {"protocolVersion": "future"}}}
    )
    runtime = Runtime.open(":memory:")
    try:
        with pytest.raises(ValidationError, match="_meta"):
            runtime.mcp.register_server(
                _manifest(**kwargs),
                actor="test",
                require_capability=False,
            )
        assert runtime.store.list_mcp_servers() == []
    finally:
        runtime.close()


@pytest.mark.parametrize("location", ["server", "tool"])
@pytest.mark.parametrize("nested", [False, True])
def test_manifest_v2_rejects_non_string_metadata_keys(
    location: str,
    nested: bool,
) -> None:
    metadata = {"nested": {1: "value"}} if nested else {1: "value"}
    overrides = (
        {"server_metadata": metadata}
        if location == "server"
        else {"tool_metadata": metadata}
    )
    runtime = Runtime.open(":memory:")
    try:
        with pytest.raises(ValidationError, match="object keys must be strings"):
            runtime.mcp.register_server(
                _manifest(server_id=f"non-string-{location}-{nested}", **overrides),
                actor="test",
                require_capability=False,
            )
        assert runtime.mcp.list_servers() == []
    finally:
        runtime.close()


def test_manifest_v2_accepts_bounded_local_json_schema_references() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "$defs": {
            "name": {"type": "string", "maxLength": 64},
        },
        "properties": {
            "name": {"$ref": "#/$defs/name"},
            "kind": {"enum": ["person", "service"]},
        },
        "allOf": [
            {
                "if": {"properties": {"kind": {"const": "person"}}},
                "then": {"required": ["name"]},
            }
        ],
        "additionalProperties": False,
    }
    runtime = Runtime.open(":memory:")
    try:
        registered = runtime.mcp.register_server(
            _manifest(input_schema=schema),
            actor="test",
            require_capability=False,
        )
        assert registered["tools"][0]["input_schema"] == schema
    finally:
        runtime.close()


def test_manifest_v1_keeps_legacy_schema_registration_behavior() -> None:
    """The v2 safe subset must not silently reinterpret stored v1 manifests."""

    legacy_schema = {
        "type": "object",
        "properties": {
            "value": {"$ref": "https://schemas.example.test/value.json"},
        },
    }
    runtime = Runtime.open(":memory:")
    try:
        registered = runtime.mcp.register_server(
            _manifest(
                server_id="legacy-schema",
                schema_version=1,
                protocol_mode=_ABSENT,
                input_schema=legacy_schema,
            ),
            actor="test",
            require_capability=False,
        )

        assert registered["tools"][0]["input_schema"] == legacy_schema
    finally:
        runtime.close()


def test_manifest_v1_reserved_envelope_reopens_with_stable_identity(
    tmp_path: Any,
) -> None:
    """New v2 reservations must not rewrite the shipped v1 wire envelope."""

    database = tmp_path / "mcp-v1-envelope.db"
    manifest = _manifest(
        server_id="legacy-envelope",
        schema_version=1,
        protocol_mode=_ABSENT,
        headers={
            "Accept": {
                "env": "AGENT_LIBOS_MCP_TEST_TOKEN",
                "prefix": "",
            },
            "Mcp-Param-Cursor": {
                "env": "AGENT_LIBOS_MCP_TEST_TOKEN",
                "prefix": "",
            },
        },
        server_metadata={
            "_meta": {"legacy": True},
            "io.modelcontextprotocol/legacy": "server",
        },
        tool_metadata={"_meta": {"legacy": "tool"}},
    )
    expected_digest = (
        "a8492a83d2116fb34c5690092cb36a9dddbce8f173b34b505ab30918f905e88b"
    )
    expected_call_identity = (
        "0fd119f7cc9ca8f9a141c0192604bf1f33a0e53e00bd27a4654c49e4bbe95dd4"
    )

    runtime = Runtime.open(database)
    try:
        runtime.mcp.register_server(
            manifest,
            actor="test",
            require_capability=False,
        )
        spec, _metadata = runtime.mcp._load_server("legacy-envelope")
        canonical = canonical_mcp_server_spec_json(spec)
        assert "protocol_mode" not in canonical
        assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == expected_digest
        assert runtime.mcp._registry_binding_context("legacy-envelope") == {
            "registry_spec_sha256": expected_digest,
            "registry_generation": 1,
        }
        assert runtime.mcp._server_identity_sha256(
            spec,
            spec.tools[0],
            stdio_executable=None,
        ) == expected_call_identity
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        spec, _metadata = reopened.mcp._load_server("legacy-envelope")
        assert tuple(spec.http.headers) == ("Accept", "Mcp-Param-Cursor")
        assert spec.metadata["_meta"] == {"legacy": True}
        assert spec.tools[0].metadata["_meta"] == {"legacy": "tool"}
        assert reopened.mcp._server_spec_sha256(spec) == expected_digest
        assert reopened.mcp._registry_binding_context("legacy-envelope") == {
            "registry_spec_sha256": expected_digest,
            "registry_generation": 1,
        }
        assert reopened.mcp._server_identity_sha256(
            spec,
            spec.tools[0],
            stdio_executable=None,
        ) == expected_call_identity
    finally:
        reopened.close()


def _schema_with_ref_chain(hops: int) -> dict[str, Any]:
    definitions: dict[str, Any] = {
        f"n{index}": (
            {"$ref": f"#/$defs/n{index + 1}"}
            if index + 1 < hops
            else {"type": "string"}
        )
        for index in range(hops)
    }
    return {
        "type": "object",
        "$defs": definitions,
        "properties": {"value": {"$ref": "#/$defs/n0"}},
    }


def _deep_schema(depth: int) -> dict[str, Any]:
    current: dict[str, Any] = {"type": "string"}
    for _ in range(depth):
        current = {"type": "object", "properties": {"value": current}}
    return current


def _nested_conditional_schema(count: int) -> dict[str, Any]:
    current: dict[str, Any] = {"type": "object"}
    for index in range(count):
        current = {
            "type": "object",
            "if": {
                "properties": {f"flag_{index}": {"const": True}},
            },
            "then": current,
        }
    return current


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"x": {"$ref": "https://evil.test/s"}}},
        {"type": "object", "$dynamicRef": "#node"},
        {
            "type": "object",
            "$defs": {
                "a": {"$ref": "#/$defs/b"},
                "b": {"$ref": "#/$defs/a"},
            },
            "properties": {"x": {"$ref": "#/$defs/a"}},
        },
        _schema_with_ref_chain(130),
        _deep_schema(70),
        {
            "type": "object",
            "properties": {
                f"value_{index}": {"type": "integer"}
                for index in range(5_100)
            },
        },
        {
            "type": "object",
            "oneOf": [
                {"type": "object", "properties": {"x": {"const": index}}}
                for index in range(1_025)
            ],
        },
        _nested_conditional_schema(11),
    ],
)
def test_manifest_v2_rejects_unsafe_or_unbounded_json_schema(
    schema: dict[str, Any],
) -> None:
    runtime = Runtime.open(":memory:")
    try:
        with pytest.raises(ValidationError):
            runtime.mcp.register_server(
                _manifest(input_schema=schema),
                actor="test",
                require_capability=False,
            )
        assert runtime.store.list_mcp_servers() == []
    finally:
        runtime.close()


def test_manifest_v2_shared_definition_ref_fanout_is_bounded() -> None:
    """Thousands of refs to one large target are validated in one graph pass."""

    width = 2_000
    schema = {
        "type": "object",
        "$defs": {
            "shared": {
                "type": "object",
                "properties": {
                    f"field_{index}": {"type": "integer"}
                    for index in range(width)
                },
            },
        },
        "properties": {
            f"item_{index}": {"$ref": "#/$defs/shared"}
            for index in range(width)
        },
    }
    runtime = Runtime.open(":memory:")
    try:
        registered = runtime.mcp.register_server(
            _manifest(server_id="shared-ref-fanout", input_schema=schema),
            actor="test",
            require_capability=False,
        )
        assert registered["tools"][0]["input_schema"] == schema
    finally:
        runtime.close()


def test_discover_requires_read_and_execute_before_provider_dispatch() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="authority"),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(image="base-agent:v0", goal="discover safely")

        with pytest.raises(CapabilityDenied):
            runtime.mcp.discover("authority", actor=pid)
        assert provider.discover_calls == []
        assert runtime.store.list_external_effects(pid=pid) == []

        runtime.capability.grant(
            pid,
            "mcp_server:authority",
            [CapabilityRight.READ],
            issued_by="test",
        )
        with pytest.raises(CapabilityDenied):
            runtime.mcp.discover("authority", actor=pid)
        assert provider.discover_calls == []
        assert runtime.store.list_external_effects(pid=pid) == []

        runtime.capability.grant(
            pid,
            "mcp_server:authority",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        result = runtime.mcp.discover("authority", actor=pid)

        assert result.server_id == "authority"
        assert result.connection.protocol_revision == "2026-07-28"
        assert provider.discover_calls == ["authority"]
        effects = runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].operation == "discover"
        assert not effects[0].state_mutation
        assert effects[0].information_flow
    finally:
        runtime.close()


def test_reflected_server_identity_never_persists_resolved_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque_secret = "opaque-provider-credential-without-a-known-prefix"
    common_token = "ghp_0123456789abcdefghijklmnop"
    monkeypatch.setenv("AGENT_LIBOS_MCP_AUTH_SECRET", opaque_secret)
    monkeypatch.setenv("AGENT_LIBOS_MCP_PROVIDER_KEY", common_token)
    db_path = tmp_path / "reflected-mcp-identity.db"
    runtime = Runtime.open(str(db_path))
    provider = _ModernFakeProvider()
    provider.discovery_result = replace(
        provider.discovery_result,
        connection=replace(
            provider.discovery_result.connection,
            protocol_mode=McpProtocolMode.REVISION_2026_07_28,
            server_name=f"fixed-server/{opaque_secret}",
            server_version=f"release/{common_token}",
            capabilities=(
                "tools",
                f"extension/{opaque_secret}",
                common_token,
            ),
            unsupported_capabilities=(
                f"extension/{opaque_secret}",
                common_token,
            ),
        ),
    )
    runtime.mcp.provider = provider
    evidence: dict[str, Any] = {}
    try:
        server_id = "identity-redaction"
        runtime.mcp.register_server(
            _manifest(
                server_id=server_id,
                protocol_mode="2026-07-28",
                headers={
                    "Authorization": {
                        "env": "AGENT_LIBOS_MCP_AUTH_SECRET",
                        "prefix": "Token ",
                    },
                    "X-Provider-Key": {
                        "env": "AGENT_LIBOS_MCP_PROVIDER_KEY"
                    },
                },
            ),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="prove reflected MCP identity is public-safe",
        )
        runtime.capability.grant(
            pid,
            f"mcp_server:{server_id}",
            [CapabilityRight.READ, CapabilityRight.EXECUTE],
            issued_by="test",
        )

        result = runtime.mcp.discover(server_id, actor=pid)

        assert result.connection.server_name == "fixed-server/[redacted]"
        assert result.connection.server_version == "release/[redacted]"
        assert set(result.connection.capabilities) == {
            "tools",
            "extension/[redacted]",
            "[redacted]",
        }
        assert set(result.connection.unsupported_capabilities) == {
            "extension/[redacted]",
            "[redacted]",
        }
        evidence = {
            "result": result,
            "audit": runtime.store.list_audit(),
            "events": runtime.store.list_events(),
            "operations": runtime.store.list_operations(),
            "effects": runtime.store.list_external_effects(),
            "servers": runtime.store.list_mcp_servers(),
        }
        serialized = dumps(to_jsonable(evidence))
        assert opaque_secret not in serialized
        assert common_token not in serialized
        assert "fixed-server/[redacted]" in serialized
        assert "release/[redacted]" in serialized
    finally:
        runtime.close()

    persisted = db_path.read_bytes()
    for sidecar in (db_path.with_name(f"{db_path.name}-wal"), db_path.with_name(f"{db_path.name}-shm")):
        if sidecar.exists():
            persisted += sidecar.read_bytes()
    assert opaque_secret.encode("utf-8") not in persisted
    assert common_token.encode("utf-8") not in persisted


def test_stdio_discover_requires_spawn_and_exact_executable_authority() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    runtime.mcp.provider = provider
    stdio = McpStdioTransportSpec(
        command="python3",
        args=["-m", "demo_server"],
    )
    spec = McpServerSpec(
        schema_version=2,
        server_id="stdio-authority",
        transport="stdio",
        tools=[
            McpToolSpec(
                tool_id="echo",
                mcp_name="demo.echo",
                right="read",
                rollback_class="no_rollback_required",
                state_mutation=False,
                information_flow=True,
                input_schema={"type": "object"},
            )
        ],
        timeout_s=5.0,
        max_request_bytes=65_536,
        max_response_bytes=1_048_576,
        stdio=stdio,
        protocol_mode=McpProtocolMode.AUTO,
    )
    try:
        runtime.mcp.register_server(
            spec,
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(image="base-agent:v0", goal="stdio discover")
        runtime.capability.grant(
            pid,
            "mcp_server:stdio-authority",
            [CapabilityRight.READ, CapabilityRight.EXECUTE],
            issued_by="test",
        )

        with pytest.raises(CapabilityDenied, match="process:spawn"):
            runtime.mcp.discover("stdio-authority", actor=pid)
        assert provider.discover_calls == []

        runtime.capability.grant(
            pid,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        with pytest.raises(CapabilityDenied, match="mcp_stdio"):
            runtime.mcp.discover("stdio-authority", actor=pid)
        assert provider.discover_calls == []

        stdio_resource = runtime.mcp.stdio_resource_for_server(spec)
        assert stdio_resource is not None
        runtime.capability.grant(
            pid,
            stdio_resource,
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        result = runtime.mcp.discover("stdio-authority", actor=pid)
        assert result.connection.protocol_era is McpProtocolEra.MODERN
        assert provider.discover_calls == ["stdio-authority"]
    finally:
        runtime.close()


def test_adiscover_uses_the_same_operation_local_contract() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="async-discover"),
            actor="test",
            require_capability=False,
        )

        result = asyncio.run(
            runtime.mcp.adiscover(
                "async-discover",
                actor=None,
                require_capability=False,
            )
        )

        assert result.server_id == "async-discover"
        assert result.connection.protocol_revision == "2026-07-28"
        assert provider.discover_calls == ["async-discover"]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "manifest",
    [
        _manifest(server_id="legacy-v1", schema_version=1, protocol_mode=_ABSENT),
        _manifest(server_id="legacy-v2", protocol_mode="legacy"),
    ],
)
def test_discover_rejects_legacy_mode_without_dispatch(
    manifest: dict[str, Any],
) -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            manifest,
            actor="test",
            require_capability=False,
        )
        with pytest.raises(ValidationError):
            runtime.mcp.discover(
                manifest["server_id"],
                actor=None,
                require_capability=False,
            )
        assert provider.discover_calls == []
        assert runtime.store.list_external_effects() == []
    finally:
        runtime.close()


def test_manifest_v2_rejects_unmarked_provider_before_dispatch() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    provider.supports_mcp_modern_protocol = False  # type: ignore[assignment]
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="unmarked-provider"),
            actor="test",
            require_capability=False,
        )

        with pytest.raises(ValidationError, match="explicitly supports"):
            runtime.mcp.discover(
                "unmarked-provider",
                actor=None,
                require_capability=False,
            )

        assert provider.discover_calls == []
        assert runtime.store.list_external_effects() == []
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["legacy", "auto", "2026-07-28"])
@pytest.mark.parametrize("operation", ["list", "call"])
def test_manifest_v2_list_and_call_fence_unmarked_provider_before_dispatch(
    mode: str,
    operation: str,
) -> None:
    runtime = Runtime.open(":memory:")
    provider = _LegacyOnlySpyProvider()
    runtime.mcp.provider = provider
    server_id = f"unmarked-{mode.replace('-', '_')}-{operation}"
    try:
        runtime.mcp.register_server(
            _manifest(server_id=server_id, protocol_mode=mode),
            actor="test",
            require_capability=False,
        )

        if operation == "list":
            with pytest.raises(ValidationError, match="explicitly supports"):
                runtime.mcp.list_tools(
                    server_id,
                    actor=None,
                    require_capability=False,
                    refresh=True,
                )
        else:
            pid = runtime.process.spawn(image="base-agent:v0", goal="fence old provider")
            runtime.capability.grant(
                pid,
                f"mcp:{server_id}:echo",
                [CapabilityRight.READ],
                issued_by="test",
            )
            with pytest.raises(ValidationError, match="explicitly supports"):
                runtime.mcp.call_tool(pid, server_id, "echo", {"text": "hello"})

        assert provider.calls == []
        assert runtime.store.list_external_effects() == []
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "connection",
    [
        _connection(mode=McpProtocolMode.LEGACY),
        _connection(revision="2099-01-01"),
        _connection(sessionless=False),
        _connection(
            mode=McpProtocolMode.REVISION_2026_07_28,
            fallback_used=True,
        ),
    ],
)
def test_discover_rejects_inconsistent_connection_metadata(
    connection: McpConnectionInfo,
) -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    provider.discovery_result = replace(
        provider.discovery_result,
        connection=connection,
    )
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="bad-connection"),
            actor="test",
            require_capability=False,
        )
        with pytest.raises((ProviderHostError, ValidationError)):
            runtime.mcp.discover(
                "bad-connection",
                actor=None,
                require_capability=False,
            )
        assert provider.discover_calls == ["bad-connection"]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "discovery_result",
    [
        McpProviderDiscoveryResult(
            connection=_connection(),
            request_bytes=0,
            response_bytes=0,
            receipts=(),
        ),
        McpProviderDiscoveryResult(
            connection=_connection(),
            receipts=(
                McpExchangeReceipt(
                    phase=McpExchangePhase.INITIALIZE,
                    call_started=True,
                ),
            ),
        ),
        McpProviderDiscoveryResult(
            connection=_connection(),
            receipts=(
                McpExchangeReceipt(
                    phase=McpExchangePhase.SERVER_DISCOVER,
                    call_started=True,
                ),
                McpExchangeReceipt(
                    phase=McpExchangePhase.TOOLS_LIST,
                    call_started=True,
                ),
            ),
        ),
        McpProviderDiscoveryResult(
            connection=_connection(),
            request_bytes=1,
            response_bytes=1,
            receipts=(
                McpExchangeReceipt(
                    phase=McpExchangePhase.SERVER_DISCOVER,
                    request_bytes=2,
                    response_bytes=2,
                    call_started=True,
                ),
            ),
        ),
        McpProviderDiscoveryResult(
            connection=_connection(),
            receipts=(
                McpExchangeReceipt(
                    phase=McpExchangePhase.SERVER_DISCOVER,
                    call_started=False,
                ),
            ),
        ),
    ],
)
def test_discover_rejects_malformed_or_underreported_receipts(
    discovery_result: McpProviderDiscoveryResult,
) -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    provider.discovery_result = discovery_result
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="bad-receipt"),
            actor="test",
            require_capability=False,
        )
        with pytest.raises(ProviderHostError):
            runtime.mcp.discover(
                "bad-receipt",
                actor=None,
                require_capability=False,
            )
        assert provider.discover_calls == ["bad-receipt"]
    finally:
        runtime.close()


def test_discovery_metadata_is_operation_local_not_persisted() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="ephemeral"),
            actor="test",
            require_capability=False,
        )
        runtime.mcp.discover(
            "ephemeral",
            actor=None,
            require_capability=False,
        )

        inspected = runtime.mcp.inspect_server(
            "ephemeral",
            require_capability=False,
        )
        observed = dumps(inspected)
        assert "contract-fixture" not in observed
        assert "protocol_revision" not in inspected
        assert "connection" not in inspected
    finally:
        runtime.close()


def test_discovered_server_capabilities_never_grant_runtime_authority() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    provider.discovery_result = replace(
        provider.discovery_result,
        connection=replace(
            provider.discovery_result.connection,
            capabilities=("tools", "sampling", "roots"),
            unsupported_capabilities=("sampling", "roots", "elicitation"),
        ),
    )
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="diagnostic-only"),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="diagnostic capabilities are not authority",
        )
        runtime.capability.grant(
            pid,
            "mcp_server:diagnostic-only",
            [CapabilityRight.READ, CapabilityRight.EXECUTE],
            issued_by="test",
        )

        discovered = runtime.mcp.discover("diagnostic-only", actor=pid)
        assert "sampling" in discovered.connection.unsupported_capabilities

        with pytest.raises(CapabilityDenied):
            runtime.mcp.call_tool(
                pid,
                "diagnostic-only",
                "echo",
                {"text": "must remain denied"},
            )
        assert provider.validate_calls == 0
        assert provider.call_calls == 0
    finally:
        runtime.close()


def test_v2_live_tool_list_projects_operation_local_connection_and_receipts() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="list-metadata"),
            actor="test",
            require_capability=False,
        )

        result = runtime.mcp.list_tools(
            "list-metadata",
            actor=None,
            require_capability=False,
            refresh=True,
        )

        assert result["connection"]["protocol_mode"] == "auto"
        assert result["connection"]["protocol_era"] == "modern"
        assert result["connection"]["protocol_revision"] == "2026-07-28"
        assert [item["phase"] for item in result["receipts"]] == [
            "server/discover",
            "tools/list",
        ]
    finally:
        runtime.close()


@pytest.mark.parametrize("tool_count", [101])
def test_v2_live_tool_catalog_rejects_item_limit(tool_count: int) -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    provider.tools = [McpProviderTool(name=f"tool-{index}") for index in range(tool_count)]
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="too-many-tools"),
            actor="test",
            require_capability=False,
        )
        with pytest.raises((ProviderHostError, ValidationError)):
            runtime.mcp.list_tools(
                "too-many-tools",
                actor=None,
                require_capability=False,
                refresh=True,
            )
        assert provider.list_calls == ["too-many-tools"]
    finally:
        runtime.close()


def test_v2_live_tool_catalog_rejects_duplicate_names() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    provider.tools = [McpProviderTool(name="demo.echo"), McpProviderTool(name="demo.echo")]
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="duplicate-tools"),
            actor="test",
            require_capability=False,
        )
        with pytest.raises((ProviderHostError, ValidationError)):
            runtime.mcp.list_tools(
                "duplicate-tools",
                actor=None,
                require_capability=False,
                refresh=True,
            )
        assert provider.list_calls == ["duplicate-tools"]
    finally:
        runtime.close()


def test_v2_live_tool_catalog_rejects_more_than_sixteen_list_receipts() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    provider.list_receipts = tuple(
        McpExchangeReceipt(
            phase=McpExchangePhase.TOOLS_LIST,
            call_started=True,
        )
        for _ in range(17)
    )
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="too-many-pages"),
            actor="test",
            require_capability=False,
        )
        with pytest.raises((ProviderHostError, ValidationError)):
            runtime.mcp.list_tools(
                "too-many-pages",
                actor=None,
                require_capability=False,
                refresh=True,
            )
        assert provider.list_calls == ["too-many-pages"]
    finally:
        runtime.close()


def test_v2_receipt_grammar_accepts_bounded_retry_and_explicit_fallback() -> None:
    runtime = Runtime.open(":memory:")
    try:
        modern_server = runtime.mcp._coerce_server(_manifest(server_id="receipt-modern"))
        modern_connection = _connection()
        discover_receipts = (
            McpExchangeReceipt(
                phase=McpExchangePhase.SERVER_DISCOVER,
                request_bytes=11,
                response_bytes=13,
                call_started=True,
            ),
            McpExchangeReceipt(
                phase=McpExchangePhase.SERVER_DISCOVER,
                request_bytes=17,
                response_bytes=19,
                call_started=True,
            ),
        )
        discovered = runtime.mcp._validated_discovery_result(
            modern_server,
            McpProviderDiscoveryResult(
                connection=modern_connection,
                request_bytes=28,
                response_bytes=32,
                duration_s=0.02,
                receipts=discover_receipts,
            ),
        )
        assert discovered.receipts == discover_receipts

        fallback_server = runtime.mcp._coerce_server(
            _manifest(server_id="receipt-fallback")
        )
        fallback_connection = _connection(
            era=McpProtocolEra.LEGACY,
            revision="2025-11-25",
            sessionless=False,
            fallback_used=True,
        )
        fallback_receipts = (
            McpExchangeReceipt(
                phase=McpExchangePhase.SERVER_DISCOVER,
                request_bytes=7,
                response_bytes=5,
                call_started=True,
            ),
            McpExchangeReceipt(
                phase=McpExchangePhase.INITIALIZE,
                request_bytes=23,
                response_bytes=29,
                call_started=True,
            ),
        )
        fallback = runtime.mcp._validated_discovery_result(
            fallback_server,
            McpProviderDiscoveryResult(
                connection=fallback_connection,
                request_bytes=30,
                response_bytes=34,
                duration_s=0.03,
                receipts=fallback_receipts,
            ),
        )
        assert fallback.connection.fallback_used
        assert [item.phase for item in fallback.receipts] == [
            McpExchangePhase.SERVER_DISCOVER,
            McpExchangePhase.INITIALIZE,
        ]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "receipts",
    [
        (
            McpExchangeReceipt(
                phase=McpExchangePhase.SERVER_DISCOVER,
                call_started=True,
            ),
            McpExchangeReceipt(
                phase=McpExchangePhase.INITIALIZE,
                call_started=True,
            ),
        ),
        tuple(
            McpExchangeReceipt(
                phase=McpExchangePhase.SERVER_DISCOVER,
                call_started=True,
            )
            for _ in range(3)
        ),
        (
            McpExchangeReceipt(
                phase=McpExchangePhase.SERVER_DISCOVER,
                call_started=True,
            ),
            McpExchangeReceipt(
                phase=McpExchangePhase.TOOLS_LIST,
                call_started=True,
            ),
            McpExchangeReceipt(
                phase=McpExchangePhase.TOOLS_LIST,
                call_started=True,
            ),
        ),
        (
            McpExchangeReceipt(
                phase=McpExchangePhase.SERVER_DISCOVER,
                call_started=True,
            ),
            McpExchangeReceipt(
                phase=McpExchangePhase.TOOLS_LIST,
                call_started=False,
            ),
        ),
    ],
)
def test_v2_list_rejects_incomplete_or_unordered_receipt_grammar(
    receipts: tuple[McpExchangeReceipt, ...],
) -> None:
    runtime = Runtime.open(":memory:")
    try:
        server = runtime.mcp._coerce_server(_manifest(server_id="invalid-list-receipts"))
        tools = [McpProviderTool(name="demo.echo")]
        response_bytes = _provider_tool_bytes(tools)
        with pytest.raises(ProviderHostError):
            runtime.mcp._validated_tool_list_result(
                server,
                McpToolListResult(
                    server_id=server.server_id,
                    tools=tools,
                    response_bytes=response_bytes,
                    duration_s=0.01,
                    connection=_connection(),
                    receipts=receipts,
                ),
            )
    finally:
        runtime.close()


def test_v2_list_uses_exact_wire_receipts_not_canonical_projection_size() -> None:
    runtime = Runtime.open(":memory:")
    try:
        server = runtime.mcp._coerce_server(_manifest(server_id="wire-list-bytes"))
        connection = _connection()
        tools = [
            McpProviderTool(
                name="demo.echo",
                description="Echo a value with a schema larger than one wire byte",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            )
        ]
        result = McpToolListResult(
            server_id=server.server_id,
            tools=tools,
            response_bytes=1,
            duration_s=0.01,
            connection=connection,
            receipts=(
                *_negotiation_receipts(connection),
                McpExchangeReceipt(
                    phase=McpExchangePhase.TOOLS_LIST,
                    response_bytes=1,
                    call_started=True,
                ),
            ),
        )

        accepted = runtime.mcp._validated_tool_list_result(server, result)

        assert accepted.response_bytes == 1
        assert accepted.receipts[-1].response_bytes == 1
    finally:
        runtime.close()


def test_v2_call_receipts_bind_phase_order_dispatch_and_aggregate_fields() -> None:
    runtime = Runtime.open(":memory:")
    try:
        server = runtime.mcp._coerce_server(_manifest(server_id="call-receipts"))
        connection = _connection()
        response_bytes = _provider_call_bytes(None, None)
        valid = _complete_call_result(
            connection=connection,
            response_bytes=response_bytes,
            is_error=True,
            error="terminal error",
            error_type="ToolError",
        )
        assert runtime.mcp._validated_provider_call_result(server, valid).call_started

        invalid_results = (
            replace(valid, receipts=valid.receipts[1:]),
            replace(
                valid,
                receipts=(valid.receipts[0], valid.receipts[-1]),
                list_request_bytes=0,
                list_response_bytes=0,
            ),
            replace(valid, call_started=False),
            replace(
                valid,
                receipts=(*valid.receipts, valid.receipts[-1]),
                call_request_bytes=valid.call_request_bytes * 2,
                call_response_bytes=valid.call_response_bytes * 2,
            ),
            replace(valid, list_response_bytes=valid.list_response_bytes + 1),
        )
        for invalid in invalid_results:
            with pytest.raises(ProviderHostError):
                runtime.mcp._validated_provider_call_result(server, invalid)

        pre_dispatch_receipts = valid.receipts[:-1]
        not_started = replace(
            valid,
            content=None,
            structured_content=None,
            error="live validation failed",
            error_type="LiveToolValidationError",
            response_bytes=0,
            call_request_bytes=0,
            call_response_bytes=0,
            call_started=False,
            receipts=pre_dispatch_receipts,
        )
        accepted_not_started = runtime.mcp._validated_provider_call_result(
            server,
            not_started,
        )
        assert not accepted_not_started.call_started
        assert all(
            receipt.phase is not McpExchangePhase.TOOLS_CALL
            for receipt in accepted_not_started.receipts
        )
    finally:
        runtime.close()


def test_v2_call_uses_exact_wire_receipts_not_canonical_projection_size() -> None:
    runtime = Runtime.open(":memory:")
    try:
        server = runtime.mcp._coerce_server(_manifest(server_id="wire-byte-receipts"))
        result = _complete_call_result(
            connection=_connection(),
            response_bytes=1,
            content=[{"type": "text", "text": "5"}],
            structured_content={"result": 5},
        )

        accepted = runtime.mcp._validated_provider_call_result(server, result)

        assert accepted.response_bytes == 1
        assert accepted.call_response_bytes == 1
        assert accepted.receipts[-1].response_bytes == 1
    finally:
        runtime.close()


def test_v2_connectionless_failure_requires_builtin_negotiation_wire_prefix() -> None:
    runtime = Runtime.open(":memory:")
    try:
        server = runtime.mcp._coerce_server(
            _manifest(server_id="connectionless-negotiation")
        )
        valid = _negotiation_failure_result(error_type="McpAuthenticationError")

        accepted = runtime.mcp._validated_provider_call_result(server, valid)
        assert accepted.connection is None
        assert not accepted.call_started

        invalid_results = (
            replace(valid, list_request_bytes=1),
            replace(
                valid,
                receipts=(
                    McpExchangeReceipt(
                        phase=McpExchangePhase.TOOLS_LIST,
                        call_started=True,
                    ),
                ),
            ),
            replace(
                valid,
                receipts=(
                    *valid.receipts,
                    McpExchangeReceipt(
                        phase=McpExchangePhase.TOOLS_CALL,
                        call_started=True,
                    ),
                ),
            ),
        )
        for invalid in invalid_results:
            with pytest.raises(ProviderHostError):
                runtime.mcp._validated_provider_call_result(server, invalid)

        runtime.mcp.provider = _ModernFakeProvider()
        with pytest.raises(ProviderHostError):
            runtime.mcp._validated_provider_call_result(server, valid)
    finally:
        runtime.close()


def test_v2_call_persists_sanitized_negotiation_receipts_in_effect_and_audit() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    runtime.mcp.provider = provider
    try:
        server_id = "receipt-evidence"
        runtime.mcp.register_server(
            _manifest(server_id=server_id),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(image="base-agent:v0", goal="record MCP receipt evidence")
        runtime.capability.grant(
            pid,
            f"mcp:{server_id}:echo",
            [CapabilityRight.READ],
            issued_by="test",
        )

        result = runtime.mcp.call_tool(pid, server_id, "echo", {"text": "hello"})

        assert result.ok
        expected_request_bytes = sum(item.request_bytes for item in result.receipts)
        expected_response_bytes = sum(item.response_bytes for item in result.receipts)
        effect = next(
            item
            for item in runtime.store.list_external_effects(pid=pid)
            if item.provider == "mcp"
        )
        effect_result = effect.provider_metadata["result"]
        assert effect_result["protocol_revision"] == "2026-07-28"
        assert effect_result["connection"]["protocol_era"] == "modern"
        assert [item["phase"] for item in effect_result["receipts"]] == [
            "server/discover",
            "tools/list",
            "tools/call",
        ]
        assert effect.provider_receipt["request_bytes"] == expected_request_bytes
        assert (
            effect.provider_receipt["operation_response_bytes"]
            == expected_response_bytes
        )
        assert effect.provider_receipt["protocol_revision"] == "2026-07-28"
        audit = next(
            item
            for item in runtime.audit.trace()
            if item.action == "primitive.mcp.call" and item.actor == pid
        )
        assert audit.decision["protocol_revision"] == "2026-07-28"
        assert audit.decision["connection"]["fallback_used"] is False
        process = runtime.process.get(pid)
        assert process.resource_usage.mcp_request_bytes == expected_request_bytes
        assert process.resource_usage.mcp_response_bytes == expected_response_bytes
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("error_type", "negotiation_failure"),
    [
        ("McpPreCallFailure", True),
        ("LiveToolValidationError", False),
    ],
)
def test_mutating_v2_builtin_pre_call_failure_is_a_settled_external_read(
    error_type: str,
    negotiation_failure: bool,
    tmp_path: Path,
) -> None:
    """Wire-bound auth/catalog failures must not poison a Durable Run."""

    database = tmp_path / f"pre-call-{error_type}.sqlite"
    runtime = Runtime.open(database, config=_task_run_config())
    provider = SdkMcpProvider()
    provider_result = (
        _negotiation_failure_result(error_type=error_type)
        if negotiation_failure
        else _pre_call_failure_result(error_type=error_type)
    )

    def fail_before_call(*_args: Any, **_kwargs: Any) -> McpProviderCallResult:
        return provider_result

    provider.validate_and_call = fail_before_call  # type: ignore[method-assign]
    runtime.mcp.provider = provider
    server_id = f"pre-call-{error_type.lower()}"
    client = _SingleMcpTaskRunClient(server_id)
    try:
        runtime.mcp.register_server(
            _manifest(server_id=server_id, state_mutation=True),
            actor="test",
            require_capability=False,
        )
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal={"objective": "Call the registered MCP tool once."},
                display_title=f"MCP pre-call {error_type}",
                image_id="base-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id=f"create:{server_id}",
        )
        assert created.root_pid is not None
        runtime.capability.grant(
            created.root_pid,
            f"mcp:{server_id}:echo",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.skills.activate_skill(
            created.root_pid,
            "agent-libos-mcp",
            actor=created.root_pid,
        )
        runtime.llm.client = client

        settled = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id=f"run:{server_id}",
            max_quanta=1,
        )

        effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=created.root_pid)
            if effect.provider == "mcp"
        ]
        assert len(effects) == 1
        effect = effects[0]
        assert effect.effect_state == "finalized"
        assert effect.transaction_state == "failed"
        assert effect.rollback_class is (
            ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        )
        assert effect.rollback_status is ExternalEffectRollbackStatus.NOT_REQUIRED
        assert not effect.state_mutation
        assert effect.information_flow
        assert effect.provider_metadata["outcome"] == "failed"
        assert effect.provider_metadata["failure_kind"] == (
            "live_validation_failed_before_call"
        )
        assert effect.provider_metadata["call_started"] is False
        assert effect.provider_metadata["tools_call_receipt_present"] is False
        assert [
            receipt["phase"] for receipt in effect.provider_metadata["result"]["receipts"]
        ] == [receipt.phase.value for receipt in provider_result.receipts]
        assert [
            receipt["phase"] for receipt in effect.provider_receipt["receipts"]
        ] == [receipt.phase.value for receipt in provider_result.receipts]
        expected_request_bytes = sum(
            receipt.request_bytes for receipt in provider_result.receipts
        )
        expected_response_bytes = sum(
            receipt.response_bytes for receipt in provider_result.receipts
        )
        process = runtime.process.get(created.root_pid)
        assert process.resource_usage.mcp_request_bytes == expected_request_bytes
        assert process.resource_usage.mcp_response_bytes == expected_response_bytes
        assert settled.status is not TaskRunStatus.NEEDS_ATTENTION
        assert "unknown_effect" not in {
            blocker["kind"] for blocker in settled.blockers
        }

        runtime.close()
        runtime = Runtime.open(database, config=_task_run_config())
        reopened = runtime.task_runs.get(created.run_id)
        assert reopened.status is not TaskRunStatus.NEEDS_ATTENTION
        assert "unknown_effect" not in {
            blocker["kind"] for blocker in reopened.blockers
        }
    finally:
        runtime.close()


def test_custom_v2_provider_cannot_self_certify_mutating_pre_call_failure() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    provider.call_result = _pre_call_failure_result(
        error_type="LiveToolValidationError"
    )
    runtime.mcp.provider = provider
    try:
        server_id = "custom-pre-call-claim"
        runtime.mcp.register_server(
            _manifest(server_id=server_id, state_mutation=True),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="do not trust a custom MCP provider's dispatch claim",
        )
        runtime.capability.grant(
            pid,
            f"mcp:{server_id}:echo",
            [CapabilityRight.WRITE],
            issued_by="test",
        )

        runtime.mcp.call_tool(pid, server_id, "echo", {"text": "hello"})

        effect = next(
            item
            for item in runtime.store.list_external_effects(pid=pid)
            if item.provider == "mcp"
        )
        assert effect.transaction_state == "unknown"
        assert effect.rollback_class is ExternalEffectRollbackClass.UNKNOWN
        assert effect.rollback_status is ExternalEffectRollbackStatus.UNKNOWN
        assert effect.state_mutation
    finally:
        runtime.close()


def test_mutating_v2_post_call_wire_failure_fences_durable_run(
    tmp_path: Path,
) -> None:
    database = tmp_path / "post-call-wire-failure.sqlite"
    runtime = Runtime.open(database, config=_task_run_config())
    provider = SdkMcpProvider()
    error, _connection_info, receipts = _post_call_wire_error()
    provider_calls = 0

    def fail_after_call(*_args: Any, **_kwargs: Any) -> McpProviderCallResult:
        nonlocal provider_calls
        provider_calls += 1
        raise error

    provider.validate_and_call = fail_after_call  # type: ignore[method-assign]
    runtime.mcp.provider = provider
    server_id = "post-call-wire-failure"
    client = _SingleMcpTaskRunClient(server_id)
    try:
        runtime.mcp.register_server(
            _manifest(server_id=server_id, state_mutation=True),
            actor="test",
            require_capability=False,
        )
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal={"objective": "Call the registered MCP tool once."},
                display_title="MCP ambiguous post-call failure",
                image_id="base-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id=f"create:{server_id}",
        )
        assert created.root_pid is not None
        runtime.capability.grant(
            created.root_pid,
            f"mcp:{server_id}:echo",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.skills.activate_skill(
            created.root_pid,
            "agent-libos-mcp",
            actor=created.root_pid,
        )
        runtime.llm.client = client

        settled = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id=f"run:{server_id}",
            max_quanta=1,
        )

        assert client.calls == 1
        assert provider_calls == 1
        effect = next(
            item
            for item in runtime.store.list_external_effects(pid=created.root_pid)
            if item.provider == "mcp"
        )
        assert effect.effect_state == "finalized"
        assert effect.transaction_state == "unknown"
        assert effect.rollback_class is ExternalEffectRollbackClass.UNKNOWN
        assert effect.rollback_status is ExternalEffectRollbackStatus.UNKNOWN
        assert effect.state_mutation
        assert effect.provider_metadata["failure_kind"] == "mcp_post_call_failure"
        assert effect.provider_metadata["call_started"] is True
        assert [
            item["phase"] for item in effect.provider_metadata["result"]["receipts"]
        ][-1] == McpExchangePhase.TOOLS_CALL.value
        assert [
            item["phase"] for item in effect.provider_receipt["receipts"]
        ][-1] == McpExchangePhase.TOOLS_CALL.value
        expected_request_bytes = sum(item.request_bytes for item in receipts)
        expected_response_bytes = sum(item.response_bytes for item in receipts)
        process = runtime.process.get(created.root_pid)
        assert process.resource_usage.mcp_request_bytes == expected_request_bytes
        assert process.resource_usage.mcp_response_bytes == expected_response_bytes
        assert settled.status is TaskRunStatus.NEEDS_ATTENTION
        assert {item["kind"] for item in settled.blockers} == {"unknown_effect"}
        assert "run" not in settled.allowed_actions
        assert "resume" not in settled.allowed_actions

        runtime.close()
        runtime = Runtime.open(database, config=_task_run_config())
        reopened = runtime.task_runs.get(created.run_id)
        assert reopened.status is TaskRunStatus.NEEDS_ATTENTION
        assert {item["kind"] for item in reopened.blockers} == {
            "unknown_effect"
        }
        assert provider_calls == 1
    finally:
        runtime.close()


def test_builtin_v2_malformed_dispatched_result_remains_unknown() -> None:
    runtime = Runtime.open(":memory:")
    provider = SdkMcpProvider()
    malformed = McpProviderCallResult(
        call_started=True,
        connection=_connection(),
        receipts=(),
    )

    def return_malformed(*_args: Any, **_kwargs: Any) -> McpProviderCallResult:
        return malformed

    provider.validate_and_call = return_malformed  # type: ignore[method-assign]
    runtime.mcp.provider = provider
    try:
        server_id = "builtin-malformed-dispatched"
        runtime.mcp.register_server(
            _manifest(server_id=server_id, state_mutation=True),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="retain unknown MCP dispatch state",
        )
        runtime.capability.grant(
            pid,
            f"mcp:{server_id}:echo",
            [CapabilityRight.WRITE],
            issued_by="test",
        )

        with pytest.raises(ProviderHostError):
            runtime.mcp.call_tool(pid, server_id, "echo", {"text": "hello"})

        effect = next(
            item
            for item in runtime.store.list_external_effects(pid=pid)
            if item.provider == "mcp"
        )
        assert effect.transaction_state == "unknown"
        assert effect.rollback_class is ExternalEffectRollbackClass.UNKNOWN
        assert effect.rollback_status is ExternalEffectRollbackStatus.UNKNOWN
        assert effect.state_mutation
    finally:
        runtime.close()


def test_v2_malformed_combined_provider_result_settles_cumulative_phase_budget() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    response_bytes = _provider_call_bytes(None, None)
    provider.call_result = McpProviderCallResult(
        response_bytes=response_bytes,
        call_response_bytes=response_bytes,
        call_started=True,
        connection=_connection(),
        receipts=(),
    )
    runtime.mcp.provider = provider
    try:
        server_id = "malformed-combined-receipts"
        runtime.mcp.register_server(
            _manifest(server_id=server_id),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(image="base-agent:v0", goal="charge malformed MCP result")
        runtime.capability.grant(
            pid,
            f"mcp:{server_id}:echo",
            [CapabilityRight.READ],
            issued_by="test",
        )

        with pytest.raises(ProviderHostError):
            runtime.mcp.call_tool(pid, server_id, "echo", {"text": "hello"})

        process = runtime.process.get(pid)
        assert process.resource_usage.mcp_request_bytes == 65_536
        assert process.resource_usage.mcp_response_bytes == 1_048_576
        reservation = runtime.store.list_resource_usage_reservations(pid=pid)[0]
        assert reservation["status"] == "settled"
        assert reservation["settled_usage"].mcp_request_bytes == 65_536
        assert reservation["settled_usage"].mcp_response_bytes == 1_048_576
        effect = next(
            item
            for item in runtime.store.list_external_effects(pid=pid)
            if item.provider == "mcp"
        )
        assert effect.provider_metadata["phase"] == "provider_validate_and_call"
    finally:
        runtime.close()


@pytest.mark.parametrize("state_mutation", [False, True])
def test_input_required_is_terminal_never_retried_and_mutation_is_unknown(
    state_mutation: bool,
) -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    connection = _connection()
    response_bytes = _provider_call_bytes(None, None)
    provider.call_result = _complete_call_result(
        connection=connection,
        response_bytes=response_bytes,
        is_error=True,
        error="MCP server requested unsupported multi-round input",
        error_type="mcp_input_required_unsupported",
    )
    runtime.mcp.provider = provider
    try:
        server_id = f"input-required-{'write' if state_mutation else 'read'}"
        runtime.mcp.register_server(
            _manifest(server_id=server_id, state_mutation=state_mutation),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(image="base-agent:v0", goal="no MCP replay")
        runtime.capability.grant(
            pid,
            f"mcp:{server_id}:echo",
            [CapabilityRight.WRITE if state_mutation else CapabilityRight.READ],
            issued_by="test",
        )

        result = runtime.mcp.call_tool(pid, server_id, "echo", {"text": "hello"})

        assert result.status == McpCallStatus.INPUT_REQUIRED_UNSUPPORTED
        assert not result.ok
        assert result.error is not None
        assert result.error["code"] == "mcp_input_required_unsupported"
        assert result.error["retryable"] is False
        assert result.error["automatic_retry_disabled"] is True
        assert provider.validate_calls == 1
        assert provider.call_calls == 0
        effects = runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        if state_mutation:
            assert effects[0].transaction_state == "unknown"
            assert effects[0].rollback_class == ExternalEffectRollbackClass.UNKNOWN
            assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
            assert effects[0].state_mutation
        else:
            assert effects[0].transaction_state == "failed"
            assert effects[0].rollback_class == (
                ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
            )
            assert effects[0].rollback_status == (
                ExternalEffectRollbackStatus.NOT_REQUIRED
            )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("case", "state_mutation", "rollback_class", "expect_unknown"),
    [
        ("read", False, "no_rollback_required", False),
        ("write", True, "rollbackable", True),
        ("rollback-unclear", False, "unknown", True),
    ],
)
def test_task_run_input_required_fences_unknown_mutation_without_replay(
    case: str,
    state_mutation: bool,
    rollback_class: str,
    expect_unknown: bool,
    tmp_path: Path,
) -> None:
    """Durable Runs must inherit MCP's no-replay effect classification."""

    database = tmp_path / f"input-required-{case}.sqlite"
    runtime = Runtime.open(database, config=_task_run_config())
    provider = _ModernFakeProvider()
    connection = _connection()
    response_bytes = _provider_call_bytes(None, None)
    provider.call_result = _complete_call_result(
        connection=connection,
        response_bytes=response_bytes,
        is_error=True,
        error="MCP server requested unsupported multi-round input",
        error_type="mcp_input_required_unsupported",
    )
    runtime.mcp.provider = provider
    server_id = f"task-run-input-required-{case}"
    client = _SingleMcpTaskRunClient(server_id)
    try:
        manifest = _manifest(server_id=server_id, state_mutation=state_mutation)
        manifest["tools"][0]["rollback_class"] = rollback_class
        runtime.mcp.register_server(
            manifest,
            actor="test",
            require_capability=False,
        )
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal={"objective": "Call the registered MCP tool exactly once."},
                display_title=f"MCP input required {state_mutation}",
                image_id="base-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id=f"create:{server_id}",
        )
        assert created.root_pid is not None
        runtime.capability.grant(
            created.root_pid,
            f"mcp:{server_id}:echo",
            [CapabilityRight.WRITE if state_mutation else CapabilityRight.READ],
            issued_by="test",
        )
        runtime.skills.activate_skill(
            created.root_pid,
            "agent-libos-mcp",
            actor=created.root_pid,
        )
        runtime.llm.client = client

        settled = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id=f"run:{server_id}",
            max_quanta=1,
        )

        assert client.calls == 1
        assert provider.validate_calls == 1
        assert provider.call_calls == 0
        effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=created.root_pid)
            if effect.provider == "mcp"
        ]
        assert len(effects) == 1
        effect = effects[0]
        assert effect.effect_state == "finalized"
        if expect_unknown:
            assert effect.transaction_state == "unknown"
            assert effect.rollback_class is ExternalEffectRollbackClass.UNKNOWN
            assert effect.rollback_status is ExternalEffectRollbackStatus.UNKNOWN
            assert settled.status is TaskRunStatus.NEEDS_ATTENTION
            assert {item["kind"] for item in settled.blockers} == {"unknown_effect"}
            assert "run" not in settled.allowed_actions
            assert "resume" not in settled.allowed_actions

            runtime.close()
            runtime = Runtime.open(database, config=_task_run_config())
            runtime.mcp.provider = provider
            settled = runtime.task_runs.get(created.run_id)
            assert settled.status is TaskRunStatus.NEEDS_ATTENTION
            assert {item["kind"] for item in settled.blockers} == {
                "unknown_effect"
            }

            with pytest.raises(ValidationError, match="cannot dispatch"):
                runtime.task_runs.run_until_blocked(
                    created.run_id,
                    expected_revision=settled.revision,
                    command_id=f"retry:{server_id}",
                    max_quanta=1,
                )
            with pytest.raises(ValidationError, match="only a paused"):
                runtime.task_runs.resume(
                    created.run_id,
                    expected_revision=settled.revision,
                    command_id=f"resume:{server_id}",
                )
            assert runtime.run_next_process_once() is None
            assert client.calls == 1
            assert provider.validate_calls == 1
        else:
            assert effect.transaction_state == "failed"
            assert effect.rollback_class is (
                ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
            )
            assert effect.rollback_status is (
                ExternalEffectRollbackStatus.NOT_REQUIRED
            )
            assert settled.status is not TaskRunStatus.NEEDS_ATTENTION
            assert "unknown_effect" not in {
                item["kind"] for item in settled.blockers
            }
    finally:
        runtime.close()


def test_protocol_mode_replacement_changes_registry_binding() -> None:
    runtime = Runtime.open(":memory:")
    try:
        runtime.mcp.register_server(
            _manifest(server_id="binding", protocol_mode="auto"),
            actor="test",
            require_capability=False,
        )
        first = runtime.store.get_mcp_registry_binding("binding")
        assert first is not None

        runtime.mcp.register_server(
            _manifest(server_id="binding", protocol_mode="2026-07-28"),
            actor="test",
            replace=True,
            require_capability=False,
        )
        second = runtime.store.get_mcp_registry_binding("binding")
        assert second is not None

        assert second["registry_generation"] > first["registry_generation"]
        assert second["registry_spec_sha256"] != first["registry_spec_sha256"]
    finally:
        runtime.close()


def test_discovery_result_is_fenced_if_registry_changes_during_dispatch() -> None:
    runtime = Runtime.open(":memory:")
    provider = _ModernFakeProvider()
    runtime.mcp.provider = provider
    try:
        runtime.mcp.register_server(
            _manifest(server_id="fenced", protocol_mode="auto"),
            actor="test",
            require_capability=False,
        )

        def replace_registration() -> None:
            runtime.mcp.register_server(
                _manifest(server_id="fenced", protocol_mode="2026-07-28"),
                actor="test",
                replace=True,
                require_capability=False,
            )

        provider.on_discover = replace_registration
        with pytest.raises((CapabilityDenied, ProviderHostError), match="registry"):
            runtime.mcp.discover(
                "fenced",
                actor=None,
                require_capability=False,
            )

        assert provider.discover_calls == ["fenced"]
    finally:
        runtime.close()
