from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from agent_libos.mcp import (
    McpAtomicToolProvider,
    McpModernContinuationProvider,
    McpModernProviderIdentity,
    McpModernToolProvider,
    McpPromptProvider,
    McpResourceProvider,
    McpSdkV2SubscriptionProvider,
    McpSdkV3ContinuationProvider,
    McpSdkV3TasksProvider,
    McpSdkV3ToolProvider,
    McpSubscriptionProvider,
    McpTasksExtensionProvider,
    McpToolProvider,
)
from agent_libos.mcp.client import McpSdkV2SessionProvider
from agent_libos.mcp.types import JsonValue


class _LegacyToolProvider:
    def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not dispatched")

    def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not dispatched")


class _ModernToolProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not dispatched")


class _ModernContinuationProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    async def continue_tool(self, *args: Any, **kwargs: Any) -> Mapping[str, JsonValue]:
        raise AssertionError("not dispatched")

    async def continue_resource(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, JsonValue]:
        raise AssertionError("not dispatched")

    async def continue_prompt(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, JsonValue]:
        raise AssertionError("not dispatched")


class _ToolOnlyContinuationProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    async def continue_tool(self, *args: Any, **kwargs: Any) -> Mapping[str, JsonValue]:
        raise AssertionError("not dispatched")


def test_legacy_tool_protocols_remain_sync_and_distinct_from_modern_v3() -> None:
    expected = {
        McpToolProvider.list_tools: (
            "self",
            "server",
            "deadline",
            "max_response_bytes",
            "executable_snapshot",
            "runtime_environment",
        ),
        McpToolProvider.call_tool: (
            "self",
            "server",
            "tool",
            "arguments",
            "deadline",
            "max_response_bytes",
            "executable_snapshot",
            "runtime_environment",
        ),
        McpAtomicToolProvider.validate_and_call: (
            "self",
            "server",
            "tool",
            "arguments",
            "deadline",
            "max_response_bytes",
            "executable_snapshot",
            "runtime_environment",
        ),
    }
    for method, parameters in expected.items():
        assert tuple(inspect.signature(method).parameters) == parameters
        assert not inspect.iscoroutinefunction(method)

    legacy = _LegacyToolProvider()
    assert isinstance(legacy, McpToolProvider)
    assert not isinstance(legacy, McpModernProviderIdentity)
    assert not isinstance(legacy, McpModernToolProvider)


def test_modern_v3_protocols_have_explicit_identity_and_async_shapes() -> None:
    tool = _ModernToolProvider()
    continuation = _ModernContinuationProvider()

    assert isinstance(tool, McpModernProviderIdentity)
    assert isinstance(tool, McpModernToolProvider)
    assert not isinstance(tool, McpToolProvider)
    assert isinstance(continuation, McpModernContinuationProvider)
    assert not isinstance(
        _ToolOnlyContinuationProvider(),
        McpModernContinuationProvider,
    )

    assert inspect.iscoroutinefunction(McpModernToolProvider.call_tool)
    for method_name in ("continue_tool", "continue_resource", "continue_prompt"):
        assert inspect.iscoroutinefunction(
            getattr(McpModernContinuationProvider, method_name)
        )


def test_modern_custom_provider_deadline_contract_is_explicitly_cooperative() -> None:
    contract = " ".join((inspect.getdoc(McpModernProviderIdentity) or "").split())

    assert "trusted Host code" in contract
    assert "must not block the event-loop thread" in contract
    assert "must stop at the supplied absolute deadline" in contract
    assert "cannot safely preempt arbitrary in-process code" in contract


def test_builtin_modern_providers_publish_exact_v3_identity_markers() -> None:
    providers_and_contracts = (
        (object.__new__(McpSdkV2SessionProvider), McpResourceProvider),
        (object.__new__(McpSdkV2SessionProvider), McpPromptProvider),
        (object.__new__(McpSdkV2SubscriptionProvider), McpSubscriptionProvider),
        (object.__new__(McpSdkV3ToolProvider), McpModernToolProvider),
        (
            object.__new__(McpSdkV3ContinuationProvider),
            McpModernContinuationProvider,
        ),
        (object.__new__(McpSdkV3TasksProvider), McpTasksExtensionProvider),
    )
    for provider, contract in providers_and_contracts:
        assert provider.mcp_manifest_schema_version == 3
        assert provider.mcp_protocol_revision == "2026-07-28"
        assert isinstance(provider, McpModernProviderIdentity)
        assert isinstance(provider, contract)
