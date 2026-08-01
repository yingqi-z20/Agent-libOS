from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
from inspect import signature

import pytest
from pydantic import ValidationError as PydanticValidationError

from agent_libos.config import (
    AgentLibOSConfig,
    DEFAULT_CONFIG,
    MCP_PROTOCOL_PROBE_TIMEOUT_MAX_S,
)
from agent_libos.models import (
    canonical_mcp_server_spec_json,
    McpCallResult,
    McpCallStatus,
    McpConnectionInfo,
    McpDiscoveryResult,
    McpExchangePhase,
    McpExchangeReceipt,
    McpProviderCallResult,
    McpProviderDiscoveryResult,
    McpProtocolEra,
    McpProtocolMode,
    McpServerSpec,
    McpToolListResult,
    McpToolSpec,
    mcp_server_spec_to_jsonable,
)
from agent_libos.substrate import McpModernProtocolProvider


def test_mcp_v2_public_types_are_exported_from_top_level_package() -> None:
    import agent_libos

    expected = {
        "McpCallResult",
        "McpConnectionInfo",
        "McpDiscoveryResult",
        "McpProtocolEra",
        "McpProtocolMode",
        "McpProviderCallResult",
        "McpToolListResult",
    }

    assert expected <= set(agent_libos.__all__)
    assert all(getattr(agent_libos, name) is not None for name in expected)


def _server(*, protocol_mode: McpProtocolMode | None = None) -> McpServerSpec:
    return McpServerSpec(
        schema_version=1,
        server_id="demo",
        transport="stdio",
        tools=[
            McpToolSpec(
                tool_id="lookup",
                mcp_name="lookup",
                right="read",
                rollback_class="none",
                state_mutation=False,
                information_flow=False,
            )
        ],
        timeout_s=10.0,
        max_request_bytes=1_024,
        max_response_bytes=4_096,
        protocol_mode=protocol_mode,
    )


def _connection() -> McpConnectionInfo:
    return McpConnectionInfo(
        protocol_mode=McpProtocolMode.AUTO,
        protocol_era=McpProtocolEra.MODERN,
        protocol_revision="2026-07-28",
        sessionless=True,
        capabilities=("tools",),
        unsupported_capabilities=("prompts",),
    )


def test_mcp_v1_model_construction_remains_compatible() -> None:
    server = _server()
    tool_list = McpToolListResult("demo", [], 0, 0.0)
    provider_call = McpProviderCallResult(content={"ok": True})
    call = McpCallResult(
        "demo",
        "lookup",
        "lookup",
        McpCallStatus.OK,
        True,
    )

    assert server.protocol_mode is None
    assert tool_list.connection is None
    assert tool_list.receipts == ()
    assert provider_call.connection is None
    assert provider_call.receipts == ()
    assert call.connection is None
    assert call.receipts == ()
    assert "protocol_mode" not in mcp_server_spec_to_jsonable(server)
    assert "protocol_mode" not in canonical_mcp_server_spec_json(server)


def test_mcp_v2_canonical_projection_binds_explicit_protocol_mode() -> None:
    server = replace(
        _server(protocol_mode=McpProtocolMode.REVISION_2026_07_28),
        schema_version=2,
    )

    assert mcp_server_spec_to_jsonable(server)["protocol_mode"] == "2026-07-28"
    assert '"protocol_mode": "2026-07-28"' in canonical_mcp_server_spec_json(
        server
    )


def test_mcp_v2_models_are_immutable_and_json_friendly() -> None:
    receipt = McpExchangeReceipt(
        phase=McpExchangePhase.SERVER_DISCOVER,
        request_bytes=17,
        response_bytes=31,
        duration_s=0.25,
        call_started=True,
    )
    connection = _connection()
    discovery = McpDiscoveryResult(
        server_id="demo",
        connection=connection,
        request_bytes=17,
        response_bytes=31,
        duration_s=0.25,
        receipts=(receipt,),
    )

    assert asdict(discovery) == {
        "server_id": "demo",
        "connection": {
            "protocol_mode": McpProtocolMode.AUTO,
            "protocol_era": McpProtocolEra.MODERN,
            "protocol_revision": "2026-07-28",
            "sessionless": True,
            "fallback_used": False,
            "server_name": None,
            "server_version": None,
            "capabilities": ("tools",),
            "unsupported_capabilities": ("prompts",),
        },
        "request_bytes": 17,
        "response_bytes": 31,
        "duration_s": 0.25,
        "receipts": (
            {
                "phase": McpExchangePhase.SERVER_DISCOVER,
                "request_bytes": 17,
                "response_bytes": 31,
                "duration_s": 0.25,
                "call_started": True,
            },
        ),
    }
    with pytest.raises(FrozenInstanceError):
        discovery.server_id = "changed"  # type: ignore[misc]


def test_provider_discovery_projection_preserves_receipts() -> None:
    receipt = McpExchangeReceipt(McpExchangePhase.SERVER_DISCOVER, call_started=True)
    result = McpProviderDiscoveryResult(
        connection=_connection(),
        receipts=(receipt,),
    )

    assert result.receipts == (receipt,)
    assert result.connection.protocol_revision == "2026-07-28"
    assert McpCallStatus.INPUT_REQUIRED_UNSUPPORTED.value == (
        "input_required_unsupported"
    )


def test_modern_provider_is_an_optional_runtime_checkable_spi() -> None:
    class ModernProvider:
        supports_mcp_modern_protocol = True

        def discover(
            self,
            server: McpServerSpec,
            *,
            timeout_s: float,
            max_response_bytes: int,
            executable_snapshot: object | None = None,
            runtime_environment: object | None = None,
            limits: object | None = None,
        ) -> McpProviderDiscoveryResult:
            del server, timeout_s, max_response_bytes
            del executable_snapshot, runtime_environment, limits
            return McpProviderDiscoveryResult(connection=_connection())

    class LegacyProvider:
        pass

    assert isinstance(ModernProvider(), McpModernProtocolProvider)
    assert not isinstance(LegacyProvider(), McpModernProtocolProvider)
    assert signature(McpModernProtocolProvider.discover).parameters[
        "limits"
    ].default is None


def test_mcp_v2_safety_defaults_are_bounded_and_validated() -> None:
    defaults = DEFAULT_CONFIG.mcp

    assert MCP_PROTOCOL_PROBE_TIMEOUT_MAX_S == 5.0
    assert defaults.protocol_probe_timeout_s == MCP_PROTOCOL_PROBE_TIMEOUT_MAX_S
    assert defaults.list_max_pages == 16
    assert defaults.schema_max_depth == 64
    assert defaults.schema_max_nodes == 10_000
    assert defaults.schema_max_ref_hops == 128
    assert defaults.schema_max_composition_expansions == 1_024

    with pytest.raises(PydanticValidationError, match="must be > 0"):
        AgentLibOSConfig(
            mcp=replace(defaults, protocol_probe_timeout_s=0.0),
        )

    with pytest.raises(PydanticValidationError, match="release maximum 5.0"):
        AgentLibOSConfig(
            mcp=replace(defaults, protocol_probe_timeout_s=5.001),
        )

    at_release_maximum = AgentLibOSConfig(
        mcp=replace(defaults, protocol_probe_timeout_s=5.0),
    )
    shorter_probe = AgentLibOSConfig(
        mcp=replace(defaults, protocol_probe_timeout_s=0.125),
    )
    assert at_release_maximum.mcp.protocol_probe_timeout_s == 5.0
    assert shorter_probe.mcp.protocol_probe_timeout_s == 0.125

    shorter_operation = AgentLibOSConfig(
        mcp=replace(defaults, timeout_s=1.0),
    )
    assert shorter_operation.mcp.timeout_s == 1.0
