from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    CapabilityRight,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    McpProviderCallResult,
    McpProviderTool,
    McpServerSpec,
    McpToolListResult,
    McpToolSpec,
    ResourceBudget,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.substrate import (
    ExecutableSnapshot,
    McpProvider,
    McpSubprocessLimitsProvider,
    SdkMcpProvider,
)
from agent_libos.utils.serde import dumps, to_jsonable


def test_mcp_provider_keeps_legacy_method_signatures() -> None:
    expected_parameters = {
        "validate_and_call": (
            "self",
            "server",
            "tool",
            "arguments",
            "timeout_s",
            "max_response_bytes",
            "executable_snapshot",
            "runtime_environment",
        ),
        "list_tools": (
            "self",
            "server",
            "timeout_s",
            "max_response_bytes",
            "executable_snapshot",
            "runtime_environment",
        ),
        "call_tool": (
            "self",
            "server",
            "tool",
            "arguments",
            "timeout_s",
            "max_response_bytes",
            "executable_snapshot",
            "runtime_environment",
        ),
    }

    for method_name, expected in expected_parameters.items():
        parameters = inspect.signature(
            getattr(McpProvider, method_name)
        ).parameters
        assert tuple(parameters) == expected
        assert "limits" not in parameters
        assert parameters["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
        assert (
            parameters["max_response_bytes"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )

    assert isinstance(SdkMcpProvider(), McpSubprocessLimitsProvider)
    assert not isinstance(_LegacyMcpProvider(), McpSubprocessLimitsProvider)
    for method_name in expected_parameters:
        assert "limits" in inspect.signature(
            getattr(SdkMcpProvider, method_name)
        ).parameters


def test_unbudgeted_stdio_runtime_accepts_exact_legacy_provider() -> None:
    runtime = Runtime.open(":memory:")
    provider = _LegacyMcpProvider()
    runtime.mcp.provider = provider
    try:
        pid = _prepare_stdio_runtime(runtime, "legacy-provider")

        refreshed = runtime.mcp.list_tools(
            "legacy-provider",
            actor=None,
            require_capability=False,
            refresh=True,
        )
        result = runtime.mcp.call_tool(
            pid,
            "legacy-provider",
            "echo",
            {"text": "hello"},
        )

        assert refreshed["refreshed"] is True
        assert result.ok
        assert provider.calls == ["list_tools", "validate_and_call"]

        fallback = _LegacyFallbackMcpProvider()
        runtime.mcp.provider = fallback
        fallback_result = runtime.mcp.call_tool(
            pid,
            "legacy-provider",
            "echo",
            {"text": "fallback"},
        )

        assert fallback_result.ok
        assert fallback.calls == ["list_tools", "call_tool"]
    finally:
        runtime.close()


def test_budgeted_stdio_runtime_rejects_legacy_provider_before_dispatch() -> None:
    runtime = Runtime.open(":memory:")
    provider = _LegacyFallbackMcpProvider()
    runtime.mcp.provider = provider
    try:
        pid = _prepare_stdio_runtime(
            runtime,
            "budgeted-legacy-provider",
            resource_budget=ResourceBudget(max_subprocess_wall_seconds=1.0),
        )

        with pytest.raises(
            ValidationError,
            match="explicitly support SubprocessLimits",
        ):
            runtime.mcp.call_tool(
                pid,
                "budgeted-legacy-provider",
                "echo",
                {"text": "blocked"},
            )

        assert provider.calls == []
    finally:
        runtime.close()


def _prepare_stdio_runtime(
    runtime: Runtime,
    server_id: str,
    *,
    resource_budget: ResourceBudget | None = None,
) -> str:
    pid = runtime.process.spawn(
        image="base-agent:v0",
        goal="MCP provider compatibility",
        resource_budget=resource_budget,
    )
    runtime.mcp.register_server_from_yaml_text(
        _stdio_manifest(server_id),
        actor="cli",
        require_capability=False,
    )
    runtime.capability.grant(
        pid,
        f"mcp:{server_id}:echo",
        [CapabilityRight.READ],
        issued_by="test",
    )
    runtime.capability.grant(
        pid,
        "process:spawn",
        [CapabilityRight.WRITE],
        issued_by="test",
    )
    runtime.capability.grant(
        pid,
        runtime.mcp.stdio_resource_for_argv(
            "python3",
            ["-m", "demo_server"],
        ),
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )
    return pid


def _stdio_manifest(server_id: str) -> str:
    return f"""
schema_version: 1
server_id: {server_id}
transport: stdio
stdio:
  command: python3
  args: ["-m", "demo_server"]
tools:
  - tool_id: echo
    mcp_name: demo.echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
    input_schema:
      type: object
      properties:
        text:
          type: string
      additionalProperties: false
timeout_s: 5
max_request_bytes: 65536
max_response_bytes: 1048576
""".strip()


class _LegacyMcpProvider:
    """Provider implementing exactly the 0.3.4 MCP dispatch signatures."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def validate_and_call(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpProviderCallResult:
        del server, tool, timeout_s, max_response_bytes
        del executable_snapshot, runtime_environment
        self.calls.append("validate_and_call")
        return self._call_result(arguments)

    def list_tools(
        self,
        server: McpServerSpec,
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpToolListResult:
        del timeout_s, max_response_bytes
        del executable_snapshot, runtime_environment
        self.calls.append("list_tools")
        tools = [
            McpProviderTool(
                name="demo.echo",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "additionalProperties": False,
                },
            )
        ]
        return McpToolListResult(
            server_id=server.server_id,
            tools=tools,
            response_bytes=len(
                dumps([to_jsonable(tool) for tool in tools]).encode("utf-8")
            ),
            duration_s=0.01,
        )

    def call_tool(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpProviderCallResult:
        del server, tool, timeout_s, max_response_bytes
        del executable_snapshot, runtime_environment
        self.calls.append("call_tool")
        return self._call_result(arguments)

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        del operation, context, result
        return ExternalEffectClassification(
            rollback_class=(
                ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
            ),
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=False,
            information_flow=True,
        )

    @staticmethod
    def _call_result(arguments: dict[str, Any]) -> McpProviderCallResult:
        content = [{"type": "text", "text": "ok"}]
        structured_content = {"echo": dict(arguments)}
        response_bytes = len(
            dumps(
                {
                    "content": content,
                    "structured_content": structured_content,
                }
            ).encode("utf-8")
        )
        return McpProviderCallResult(
            content=content,
            structured_content=structured_content,
            response_bytes=response_bytes,
            duration_s=0.01,
            call_response_bytes=response_bytes,
            call_started=True,
        )


class _LegacyFallbackMcpProvider(_LegacyMcpProvider):
    validate_and_call = None
