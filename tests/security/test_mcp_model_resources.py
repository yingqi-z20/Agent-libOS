from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.mcp import (
    McpArtifactReceipt,
    McpBlobContent,
    McpCacheHint,
    McpComplete,
    McpInputRequest,
    McpInputRequestKind,
    McpInputRequired,
    McpPage,
    McpRemoteTask,
    McpRemoteTaskStatus,
    McpResource,
    McpResourceContents,
    McpResourceLinkContent,
    McpResourceTemplate,
    McpResourceTemplateSpec,
    McpResourceSpec,
    McpServerManifestV3,
    McpTextContent,
)
from agent_libos.models import (
    CapabilityRight,
    McpCallResult,
    McpCallStatus,
    McpDispatchState,
    McpRetryClass,
    McpToolSpec,
)
from agent_libos.models.mcp import (
    McpHeaderSpec,
    McpHttpTransportSpec,
    McpProtocolMode,
)
from agent_libos.runtime.syscall_descriptors import MCP_SYSCALLS
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.tools.base import BaseAgentTool, ToolContext, ToolErrorCode
from agent_libos.tools.builtin import mcp as builtin_mcp
from agent_libos.tools.builtin.mcp import (
    CallMcpToolArgs,
    CallMcpToolOutput,
    CallMcpToolResultOutput,
    CallMcpToolTool,
    ListMcpResourcesArgs,
    ListMcpResourcesTool,
    ListMcpServersTool,
    ReadMcpResourceArgs,
    ReadMcpResourceTool,
)
from agent_libos.utils.serde import dumps, to_jsonable


_CURSOR = "mcpcur_abcdefghijklmnopqrstuvwxyz012345"


class _ProtectedResourceFacade:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def alist_resources(
        self,
        server_id: str,
        *,
        cursor: str | None,
        actor: str,
        model_visible_only: bool,
    ) -> McpPage[McpResource]:
        self.calls.append(
            (
                "resources",
                {
                    "server_id": server_id,
                    "cursor": cursor,
                    "actor": actor,
                    "model_visible_only": model_visible_only,
                },
            )
        )
        return McpPage(
            items=(McpResource(resource_id="status", name="Status"),),
            next_cursor=_CURSOR,
            cache_hint=McpCacheHint(ttl_ms=1_000),
        )

    async def alist_resource_templates(
        self,
        server_id: str,
        *,
        cursor: str | None,
        actor: str,
        model_visible_only: bool,
    ) -> McpPage[McpResourceTemplate]:
        self.calls.append(
            (
                "templates",
                {
                    "server_id": server_id,
                    "cursor": cursor,
                    "actor": actor,
                    "model_visible_only": model_visible_only,
                },
            )
        )
        return McpPage(
            items=(
                McpResourceTemplate(template_id="greeting", name="Greeting"),
            )
        )

    async def aread_resource(
        self,
        server_id: str,
        resource_id: str,
        *,
        variables: dict[str, str],
        actor: str,
        for_model: bool,
    ) -> McpComplete[McpResourceContents]:
        self.calls.append(
            (
                "read",
                {
                    "server_id": server_id,
                    "resource_id": resource_id,
                    "variables": variables,
                    "actor": actor,
                    "for_model": for_model,
                },
            )
        )
        return McpComplete(
            value=McpResourceContents(
                resource_id=resource_id,
                contents=(
                    McpTextContent(text="credential [redacted]"),
                    McpBlobContent(
                        artifact=McpArtifactReceipt(
                            artifact_id="artifact-safe",
                            byte_length=4,
                            sha256="0" * 64,
                            mime_type="application/octet-stream",
                        )
                    ),
                    McpResourceLinkContent(
                        resource_handle="mcp-link:0123456789abcdef",
                        name="Related",
                    ),
                ),
            )
        )


class _RuntimeResourceProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    async def list_resources(
        self,
        _server: Any,
        cursor: str | None,
        *,
        deadline: float,
    ) -> McpPage[McpResource]:
        assert deadline > 0
        self.calls.append(("list", cursor or "", {}))
        return McpPage(
            items=(
                McpResource(
                    resource_id="opaque://provider/status",
                    name="Status",
                    description=f"visible {self.secret}",
                ),
                McpResource(
                    resource_id="opaque://provider/private",
                    name="Private",
                ),
                McpResource(
                    resource_id="opaque://provider/live-only",
                    name="Live only",
                ),
            )
        )

    async def list_resource_templates(
        self,
        _server: Any,
        cursor: str | None,
        *,
        deadline: float,
    ) -> McpPage[McpResourceTemplate]:
        assert deadline > 0
        self.calls.append(("templates", cursor or "", {}))
        return McpPage(
            items=(
                McpResourceTemplate(
                    template_id="notes://greet/{name}",
                    name="Greeting",
                    description=f"visible {self.secret}",
                ),
                McpResourceTemplate(
                    template_id="notes://private/{name}",
                    name="Private",
                ),
                McpResourceTemplate(
                    template_id="https://live-only.invalid/{name}",
                    name="Live only",
                ),
            )
        )

    async def read_resource(
        self,
        _server: Any,
        selector: str,
        variables: dict[str, str] | None,
        *,
        deadline: float,
    ) -> McpComplete[McpResourceContents]:
        assert deadline > 0
        self.calls.append(("read", selector, dict(variables or {})))
        return McpComplete(
            value=McpResourceContents(
                resource_id="provider-controlled",
                contents=(
                    McpTextContent(text=f"reflected {self.secret}"),
                    McpBlobContent(
                        artifact=McpArtifactReceipt(
                            artifact_id="artifact-safe",
                            byte_length=4,
                            sha256="0" * 64,
                            mime_type="application/octet-stream",
                        )
                    ),
                    McpResourceLinkContent(
                        resource_handle="https://provider.invalid/private",
                        name=f"related {self.secret}",
                    ),
                ),
            )
        )


def _context(facade: Any) -> ToolContext:
    return ToolContext(
        trace_id="trace-mcp-resource",
        call_id="call-mcp-resource",
        pid="pid-mcp-resource",
        runtime=SimpleNamespace(config=DEFAULT_CONFIG, mcp=facade),
    )


def test_model_resource_tools_publish_closed_logical_only_schemas() -> None:
    list_schema = ListMcpResourcesTool().spec().input_schema
    read_schema = ReadMcpResourceTool().spec().input_schema

    assert list_schema["additionalProperties"] is False
    assert set(list_schema["properties"]) == {"server_id", "kind", "cursor"}
    assert list_schema["properties"]["kind"]["enum"] == [
        "resource",
        "template",
    ]
    assert list_schema["properties"]["server_id"]["maxLength"] == 96
    assert read_schema["additionalProperties"] is False
    assert set(read_schema["properties"]) == {
        "server_id",
        "resource_id",
        "variables",
    }
    assert read_schema["properties"]["variables"]["additionalProperties"] is False
    assert read_schema["properties"]["variables"]["maxProperties"] == 256
    assert "patternProperties" in read_schema["properties"]["variables"]
    serialized = repr({"list": list_schema, "read": read_schema}).casefold()
    for forbidden in ("url", "uri", "header", "actor"):
        assert f"'{forbidden}'" not in serialized

    invalid_list = (
        {"server_id": "https://attacker.invalid"},
        {"server_id": "registered", "cursor": "provider-raw-cursor"},
        {"server_id": "registered", "kind": "prompt"},
    )
    for value in invalid_list:
        with pytest.raises(PydanticValidationError):
            ListMcpResourcesArgs.model_validate(value)
    invalid_read = (
        {"server_id": "registered", "resource_id": "file:///secret"},
        {
            "server_id": "registered",
            "resource_id": "status",
            "variables": {"name": 7},
        },
        {
            "server_id": "registered",
            "resource_id": "status",
            "variables": {"bad/key": "value"},
        },
        {
            "server_id": "registered",
            "resource_id": "status",
            "variables": {"name": "é" * 32_769},
        },
        {
            "server_id": "registered",
            "resource_id": "status",
            "variables": {f"name{index}": "value" for index in range(257)},
        },
    )
    for value in invalid_read:
        with pytest.raises(PydanticValidationError):
            ReadMcpResourceArgs.model_validate(value)


def test_model_call_tool_output_schema_is_a_closed_versioned_union() -> None:
    schema = CallMcpToolTool().spec().output_schema
    definitions = schema["$defs"]

    assert schema["title"] == "CallMcpToolResultOutput"
    assert schema["anyOf"][0] == {"$ref": "#/$defs/CallMcpToolOutput"}
    modern = schema["anyOf"][1]
    assert modern["discriminator"] == {
        "mapping": {
            "complete": "#/$defs/CallMcpToolCompleteOutput",
            "input_required": "#/$defs/CallMcpToolInputRequiredOutput",
            "remote_task": "#/$defs/CallMcpToolRemoteTaskOutput",
        },
        "propertyName": "kind",
    }
    assert {item["$ref"] for item in modern["oneOf"]} == {
        "#/$defs/CallMcpToolCompleteOutput",
        "#/$defs/CallMcpToolInputRequiredOutput",
        "#/$defs/CallMcpToolRemoteTaskOutput",
    }
    expected_properties = {
        "CallMcpToolOutput": {
            "server_id",
            "tool_id",
            "mcp_name",
            "status",
            "ok",
            "result",
            "error",
            "response_bytes",
            "duration_s",
            "dispatch_state",
            "retry_class",
            "automatic_retry_disabled",
        },
        "CallMcpToolCompleteOutput": {"kind", "value"},
        "CallMcpToolInputRequiredOutput": {
            "kind",
            "continuation_id",
            "human_receipt",
        },
        "CallMcpToolRemoteTaskOutput": {
            "kind",
            "task_ref",
            "status",
            "result",
            "human_receipt",
        },
        "CallMcpHumanReceiptOutput": {
            "request_id",
            "revision",
            "preview_sha256",
        },
    }
    for name, properties in expected_properties.items():
        variant = definitions[name]
        assert variant["additionalProperties"] is False
        assert set(variant["properties"]) == properties
        expected_required = (
            properties - {"error", "result"}
            if name == "CallMcpToolOutput"
            else properties
        )
        assert set(variant["required"]) == expected_required

    serialized = repr(schema)
    for forbidden in (
        "requestState",
        "remote_task_id",
        "input_requests",
        "expires_at",
        "status_message",
        "created_at",
        "updated_at",
        "ttl_ms",
        "poll_interval_ms",
    ):
        assert forbidden not in serialized


class _ProtectedToolFacade:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[Any, ...]] = []

    async def acall_tool(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.result


def _invoke_model_call(result: Any) -> Any:
    facade = _ProtectedToolFacade(result)
    projected = CallMcpToolTool().invoke(
        {
            "server_id": "modern",
            "tool_id": "lookup",
            "arguments": {"query": "safe"},
        },
        _context(facade),
    )
    assert facade.calls == [
        ("pid-mcp-resource", "modern", "lookup", {"query": "safe"})
    ]
    return projected


def test_model_call_tool_preserves_legacy_payload_and_minimizes_v3_outcomes() -> None:
    legacy = McpCallResult(
        server_id="legacy",
        tool_id="lookup",
        mcp_name="lookup",
        status=McpCallStatus.OK,
        ok=True,
        result={"content": [{"type": "text", "text": "done"}]},
        response_bytes=31,
        duration_s=0.25,
        dispatch_state=McpDispatchState.STARTED,
        retry_class=McpRetryClass.NOT_APPLICABLE,
    )
    serialized_legacy = to_jsonable(legacy)
    expected_legacy = CallMcpToolOutput(
        **{
            field: serialized_legacy[field]
            for field in CallMcpToolOutput.model_fields
        }
    ).model_dump()
    projected_legacy = _invoke_model_call(legacy)
    assert projected_legacy.ok is True
    assert projected_legacy.data == expected_legacy
    assert CallMcpToolResultOutput.model_validate(
        projected_legacy.data
    ).model_dump() == expected_legacy

    completed = _invoke_model_call(
        McpComplete(value={"content": [{"type": "text", "text": "safe"}]})
    )
    assert completed.ok is True
    assert completed.data == {
        "kind": "complete",
        "value": {"content": [{"type": "text", "text": "safe"}]},
    }

    provider_only = "provider requestState must remain Host-only"
    human_receipt = {
        "request_id": "hreq_0123456789abcdef",
        "revision": 2,
        "preview_sha256": "a" * 64,
    }
    pending = _invoke_model_call(
        McpInputRequired(
            continuation_id="mcpcont_abcdefghijklmnopqrstuvwxyz012345",
            input_requests=(
                McpInputRequest(
                    request_id="provider-input",
                    kind=McpInputRequestKind.ELICITATION,
                    prompt=provider_only,
                    schema={"requestState": "remote-bearer"},
                ),
            ),
            expires_at="2030-01-01T00:00:00Z",
            revision=19,
            human_request_id=human_receipt["request_id"],
            human_revision=human_receipt["revision"],
            human_preview_sha256=human_receipt["preview_sha256"],
        )
    )
    assert pending.ok is True
    assert pending.data == {
        "kind": "input_required",
        "continuation_id": "mcpcont_abcdefghijklmnopqrstuvwxyz012345",
        "human_receipt": human_receipt,
    }
    assert provider_only not in repr(pending)
    assert "remote-bearer" not in repr(pending)

    working = _invoke_model_call(
        McpRemoteTask(
            task_ref="mcptask_abcdefghijklmnopqrstuvwxyz012345",
            status=McpRemoteTaskStatus.WORKING,
            status_message=provider_only,
            result={"requestState": "remote-bearer"},
            input_requests=(
                McpInputRequest(
                    request_id="provider-input",
                    kind=McpInputRequestKind.ELICITATION,
                    prompt=provider_only,
                ),
            ),
            created_at="2030-01-01T00:00:00Z",
            updated_at="2030-01-01T00:00:01Z",
            ttl_ms=60_000,
            poll_interval_ms=1_000,
            revision=7,
            human_request_id=human_receipt["request_id"],
            human_revision=human_receipt["revision"],
            human_preview_sha256=human_receipt["preview_sha256"],
        )
    )
    assert working.ok is True
    assert working.data == {
        "kind": "remote_task",
        "task_ref": "mcptask_abcdefghijklmnopqrstuvwxyz012345",
        "status": "working",
        "result": None,
        "human_receipt": human_receipt,
    }
    assert provider_only not in repr(working)
    assert "remote-bearer" not in repr(working)

    completed_task = _invoke_model_call(
        McpRemoteTask(
            task_ref="mcptask_abcdefghijklmnopqrstuvwxyz012345",
            status=McpRemoteTaskStatus.COMPLETED,
            result={"content": [{"type": "text", "text": "safe final"}]},
            revision=8,
        )
    )
    assert completed_task.ok is True
    assert completed_task.data == {
        "kind": "remote_task",
        "task_ref": "mcptask_abcdefghijklmnopqrstuvwxyz012345",
        "status": "completed",
        "result": {"content": [{"type": "text", "text": "safe final"}]},
        "human_receipt": None,
    }


def test_model_call_tool_rejects_raw_or_incomplete_v3_control_state() -> None:
    raw = _invoke_model_call(
        {
            "kind": "input_required",
            "continuation_id": "remote-continuation-id",
            "requestState": "remote-bearer",
            "remote_task_id": "remote-task-bearer",
        }
    )
    assert raw.ok is False
    assert raw.error is not None
    assert raw.error.code is ToolErrorCode.EXECUTION_ERROR
    assert "remote-bearer" not in repr(raw)
    assert "remote-task-bearer" not in repr(raw)

    partial_human = _invoke_model_call(
        McpInputRequired(
            continuation_id="mcpcont_abcdefghijklmnopqrstuvwxyz012345",
            human_request_id="hreq_0123456789abcdef",
            human_revision=None,
            human_preview_sha256="a" * 64,
        )
    )
    assert partial_human.ok is False
    assert partial_human.error is not None
    assert partial_human.error.code is ToolErrorCode.EXECUTION_ERROR

    with pytest.raises(PydanticValidationError):
        CallMcpToolResultOutput.model_validate(
            {
                "kind": "remote_task",
                "task_ref": "remote-task-bearer",
                "status": "working",
                "result": None,
                "human_receipt": None,
            }
        )


def test_runtime_v3_tool_facade_returns_only_sanitized_model_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "modern-tool-credential-7bd4"
    monkeypatch.setenv("AGENT_LIBOS_MCP_MODEL_TOOL_TOKEN", secret)

    class ModernToolProvider:
        calls = 0

        async def call_tool(
            self,
            manifest: McpServerManifestV3,
            tool_id: str,
            arguments: dict[str, Any],
            *,
            deadline: float,
            sensitive_values: tuple[str, ...] = (),
        ) -> McpComplete[dict[str, Any]]:
            assert secret in sensitive_values
            assert deadline > 0
            assert manifest.server_id == "model-modern-tool"
            assert tool_id == "echo"
            assert arguments == {"text": "hello"}
            self.calls += 1
            return McpComplete(
                value={
                    "content": [
                        {
                            "type": "text",
                            "text": f"reflected Bearer {secret}",
                        }
                    ]
                }
            )

    runtime = Runtime.open(tmp_path / "mcp-v3-model-tool.sqlite")
    provider = ModernToolProvider()
    try:
        runtime.mcp.register_server(
            McpServerManifestV3(
                schema_version=3,
                server_id="model-modern-tool",
                transport="streamable_http",
                http=McpHttpTransportSpec(
                    url="http://127.0.0.1:8765/mcp",
                    headers={
                        "Authorization": McpHeaderSpec(
                            env="AGENT_LIBOS_MCP_MODEL_TOOL_TOKEN",
                            prefix="Bearer ",
                        )
                    },
                ),
                timeout_s=1.0,
                max_request_bytes=4_096,
                max_response_bytes=4_096,
                protocol_mode=McpProtocolMode.REVISION_2026_07_28,
                tools=(
                    McpToolSpec(
                        tool_id="echo",
                        mcp_name="provider.echo",
                        right="read",
                        rollback_class="no_rollback_required",
                        rollback_status="not_required",
                        state_mutation=False,
                        information_flow=True,
                        input_schema={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    ),
                ),
            ),
            actor="runtime",
            require_capability=False,
        )
        runtime.mcp._modern_tool_provider = provider  # noqa: SLF001
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="exercise the model-facing modern MCP Tool union",
        )
        runtime.capability.grant(
            pid,
            "mcp:model-modern-tool:echo",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            "mcp_server:model-modern-tool",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        context = ToolContext(
            trace_id="trace-modern-tool",
            call_id="call-modern-tool",
            pid=pid,
            runtime=runtime,
        )

        result = CallMcpToolTool().invoke(
            {
                "server_id": "model-modern-tool",
                "tool_id": "echo",
                "arguments": {"text": "hello"},
            },
            context,
        )

        assert result.ok is True
        assert result.data == {
            "kind": "complete",
            "value": {
                "content": [
                    {"type": "text", "text": "reflected [redacted]"}
                ]
            },
        }
        assert provider.calls == 1
        assert secret not in dumps(
            {
                "result": result.data,
                "audit": runtime.store.list_audit(),
                "events": runtime.store.list_events(),
                "effects": runtime.store.list_external_effects(pid=pid),
                "operations": runtime.store.list_operations(pid=pid),
            }
        )
    finally:
        runtime.close()


def test_model_resource_tools_route_only_through_protected_async_facade() -> None:
    facade = _ProtectedResourceFacade()
    context = _context(facade)

    resources = ListMcpResourcesTool().invoke(
        {"server_id": "modern", "kind": "resource"},
        context,
    )
    templates = ListMcpResourcesTool().invoke(
        {"server_id": "modern", "kind": "template"},
        context,
    )
    read = ReadMcpResourceTool().invoke(
        {
            "server_id": "modern",
            "resource_id": "greeting",
            "variables": {"name": "Ada"},
        },
        context,
    )

    assert resources.ok and resources.data is not None
    assert resources.data["has_more"] is True
    assert resources.data["next_cursor"] == _CURSOR
    assert resources.data["items"][0]["resource_id"] == "status"
    assert templates.ok and templates.data is not None
    assert templates.data["items"][0]["template_id"] == "greeting"
    assert read.ok and read.data is not None
    projected = repr(read.data)
    assert "artifact-safe" in projected
    assert "mcp-link:0123456789abcdef" in projected
    assert "[redacted]" in projected
    assert "https://" not in projected
    assert facade.calls == [
        (
            "resources",
            {
                "server_id": "modern",
                "cursor": None,
                "actor": "pid-mcp-resource",
                "model_visible_only": True,
            },
        ),
        (
            "templates",
            {
                "server_id": "modern",
                "cursor": None,
                "actor": "pid-mcp-resource",
                "model_visible_only": True,
            },
        ),
        (
            "read",
            {
                "server_id": "modern",
                "resource_id": "greeting",
                "variables": {"name": "Ada"},
                "actor": "pid-mcp-resource",
                "for_model": True,
            },
        ),
    ]


@pytest.mark.parametrize(
    "tool",
    [ListMcpResourcesTool(), ReadMcpResourceTool()],
    ids=("list", "read"),
)
def test_model_resource_tools_fail_closed_without_protected_facade(
    tool: BaseAgentTool,
) -> None:
    arguments = (
        {"server_id": "modern"}
        if isinstance(tool, ListMcpResourcesTool)
        else {"server_id": "modern", "resource_id": "status"}
    )
    result = tool.invoke(arguments, _context(SimpleNamespace()))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_ERROR


def test_model_server_list_preserves_bounded_registry_truncation_signal() -> None:
    class RegistryFacade:
        @staticmethod
        def list_servers_window(**kwargs: Any) -> tuple[list[dict[str, str]], bool]:
            assert kwargs == {
                "actor": "pid-mcp-resource",
                "text": None,
                "limit": None,
            }
            return ([{"server_id": "visible"}], True)

    result = ListMcpServersTool().invoke({}, _context(RegistryFacade()))

    assert result.ok is True
    assert result.data == {
        "servers": [{"server_id": "visible"}],
        "has_more": True,
    }


def test_runtime_model_resources_enforce_authority_allowlist_and_safe_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "opaque-resource-credential-92f8"
    monkeypatch.setenv("AGENT_LIBOS_MCP_RESOURCE_TOKEN", secret)
    provider = _RuntimeResourceProvider(secret)
    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate.mcp_resource_provider = provider
    database = tmp_path / "mcp-model-resource.sqlite"
    runtime = Runtime.open(database, substrate=substrate)
    try:
        assert {
            str(row["name"])
            for row in runtime.tools.list()
            if "mcp" in str(row["name"])
        } == {
            "list_mcp_servers",
            "inspect_mcp_server",
            "list_mcp_tools",
            "call_mcp_tool",
            "list_mcp_resources",
            "read_mcp_resource",
        }
        runtime.mcp.register_server(
            McpServerManifestV3(
                schema_version=3,
                server_id="modern",
                transport="streamable_http",
                http=McpHttpTransportSpec(
                    url="https://example.com/mcp",
                    headers={
                        "Authorization": McpHeaderSpec(
                            env="AGENT_LIBOS_MCP_RESOURCE_TOKEN",
                            prefix="Bearer ",
                        )
                    },
                ),
                timeout_s=1.0,
                max_request_bytes=65_536,
                max_response_bytes=262_144,
                protocol_mode=McpProtocolMode.REVISION_2026_07_28,
                resources=(
                    McpResourceSpec(
                        resource_id="status",
                        remote_uri="opaque://provider/status",
                        model_visible=True,
                    ),
                    McpResourceSpec(
                        resource_id="host-only",
                        remote_uri="opaque://provider/private",
                        model_visible=False,
                    ),
                ),
                resource_templates=(
                    McpResourceTemplateSpec(
                        template_id="greeting",
                        remote_uri_template="notes://greet/{name}",
                        variables=("name",),
                        model_visible=True,
                    ),
                    McpResourceTemplateSpec(
                        template_id="host-template",
                        remote_uri_template="notes://private/{name}",
                        variables=("name",),
                        model_visible=False,
                    ),
                ),
            ),
            actor="test.host",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="read one model-visible MCP Resource",
        )
        arguments = {"server_id": "modern", "resource_id": "status"}

        missing_resource = runtime.tools.call(pid, "read_mcp_resource", arguments)
        assert missing_resource.ok is False
        assert missing_resource.error is not None
        assert missing_resource.error.startswith("permission_denied")
        assert provider.calls == []
        assert runtime.store.list_external_effects(pid=pid) == []

        runtime.capability.grant(
            pid,
            "mcp:modern:resource:status",
            [CapabilityRight.READ],
            issued_by="test",
        )
        missing_server_execute = runtime.tools.call(
            pid,
            "read_mcp_resource",
            arguments,
        )
        assert missing_server_execute.ok is False
        assert missing_server_execute.error is not None
        assert missing_server_execute.error.startswith("permission_denied")
        assert provider.calls == []
        assert runtime.store.list_external_effects(pid=pid) == []

        missing_catalog_read = runtime.tools.call(
            pid,
            "list_mcp_resources",
            {"server_id": "modern", "kind": "resource"},
        )
        assert missing_catalog_read.ok is False
        assert missing_catalog_read.error is not None
        assert missing_catalog_read.error.startswith("permission_denied")
        assert provider.calls == []

        runtime.capability.grant(
            pid,
            "mcp_server:modern",
            [CapabilityRight.READ],
            issued_by="test",
        )
        catalog_without_execute = runtime.tools.call(
            pid,
            "list_mcp_resources",
            {"server_id": "modern", "kind": "resource"},
        )
        assert catalog_without_execute.ok is False
        assert catalog_without_execute.error is not None
        assert catalog_without_execute.error.startswith("permission_denied")
        assert provider.calls == []

        runtime.capability.grant(
            pid,
            "mcp_server:modern",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        listed = runtime.tools.call(
            pid,
            "list_mcp_resources",
            {"server_id": "modern", "kind": "resource"},
        )
        assert listed.ok is True
        listed_projection = dumps(listed.payload)
        assert secret not in listed_projection
        assert "[redacted]" in listed_projection
        assert [
            item["resource_id"] for item in listed.payload["items"]
        ] == ["status"]
        assert "opaque://" not in listed_projection

        listed_templates = runtime.tools.call(
            pid,
            "list_mcp_resources",
            {"server_id": "modern", "kind": "template"},
        )
        assert listed_templates.ok is True
        template_projection = dumps(listed_templates.payload)
        assert secret not in template_projection
        assert "[redacted]" in template_projection
        assert [
            item["template_id"] for item in listed_templates.payload["items"]
        ] == ["greeting"]
        assert "notes://" not in template_projection

        completed = runtime.tools.call(pid, "read_mcp_resource", arguments)
        assert completed.ok is True
        projected = dumps(completed.payload)
        assert secret not in projected
        assert "[redacted]" in projected
        assert "artifact-safe" in projected
        assert "mcp-link:" in projected
        assert "https://provider.invalid/private" not in projected
        assert provider.calls == [
            ("list", "", {}),
            ("templates", "", {}),
            ("read", "opaque://provider/status", {}),
        ]

        runtime.capability.grant(
            pid,
            "mcp:modern:resource:greeting",
            [CapabilityRight.READ],
            issued_by="test",
        )
        wrong_variables = runtime.tools.call(
            pid,
            "read_mcp_resource",
            {
                "server_id": "modern",
                "resource_id": "greeting",
                "variables": {"wrong": "Ada"},
            },
        )
        assert wrong_variables.ok is False
        assert len(provider.calls) == 3
        completed_template = runtime.tools.call(
            pid,
            "read_mcp_resource",
            {
                "server_id": "modern",
                "resource_id": "greeting",
                "variables": {"name": "Ada Lovelace"},
            },
        )
        assert completed_template.ok is True
        assert completed_template.payload["resource_id"] == "greeting"
        assert provider.calls[-1] == (
            "read",
            "notes://greet/Ada%20Lovelace",
            {},
        )

        runtime.capability.grant(
            pid,
            "mcp:modern:resource:host-only",
            [CapabilityRight.READ],
            issued_by="test",
        )
        hidden = runtime.tools.call(
            pid,
            "read_mcp_resource",
            {"server_id": "modern", "resource_id": "host-only"},
        )
        assert hidden.ok is False
        assert provider.calls == [
            ("list", "", {}),
            ("templates", "", {}),
            ("read", "opaque://provider/status", {}),
            ("read", "notes://greet/Ada%20Lovelace", {}),
        ]

        effects = runtime.store.list_external_effects(pid=pid)
        assert any(
            effect.target == "mcp:modern:resource:status"
            and effect.information_flow
            and not effect.state_mutation
            for effect in effects
        )
        persisted_projection = dumps(
            {
                "list_result": listed.payload,
                "template_list_result": listed_templates.payload,
                "read_result": completed.payload,
                "template_read_result": completed_template.payload,
                "audit": [
                    to_jsonable(record) for record in runtime.audit.trace(actor=pid)
                ],
                "events": [
                    to_jsonable(event) for event in runtime.events.list()
                ],
                "effects": [to_jsonable(effect) for effect in effects],
                "operations": [
                    to_jsonable(operation)
                    for operation in runtime.store.list_operations(pid=pid)
                ],
            }
        )
        assert secret not in persisted_projection
    finally:
        runtime.close()

    secret_bytes = secret.encode("utf-8")
    for suffix in ("", "-wal", "-shm"):
        selected = Path(f"{database}{suffix}")
        if selected.exists():
            assert secret_bytes not in selected.read_bytes()


def test_mcp_builtin_module_exposes_no_other_modern_model_tools() -> None:
    names = {
        value.name
        for value in vars(builtin_mcp).values()
        if inspect.isclass(value)
        and issubclass(value, BaseAgentTool)
        and value is not BaseAgentTool
        and getattr(value, "name", None)
    }
    assert names == {
        "list_mcp_servers",
        "inspect_mcp_server",
        "list_mcp_tools",
        "call_mcp_tool",
        "list_mcp_resources",
        "read_mcp_resource",
    }
    assert not any(
        forbidden in name
        for name in names
        for forbidden in (
            "prompt",
            "oauth",
            "human",
            "subscription",
            "task",
        )
    )
    assert {descriptor.name for descriptor in MCP_SYSCALLS} == {
        "mcp.list",
        "mcp.inspect",
        "mcp.tools",
        "mcp.call",
        "mcp.resources",
        "mcp.resource_read",
    }
