from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.api.cli import cli as mcp_cli
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.mcp import (
    McpComplete,
    McpClientBinding,
    McpCompletionResult,
    McpContinuationSurfaceUnsupported,
    McpInputRequest,
    McpInputRequestKind,
    McpInputRequired,
    McpPage,
    McpPrompt,
    McpPromptResult,
    McpPromptSpec,
    McpRemoteTask,
    McpResource,
    McpResourceContents,
    McpResourceSpec,
    McpResourceTemplate,
    McpServerManifestV3,
    McpTasksExtensionSpec,
)
from agent_libos.mcp.manifest import MCP_TASKS_EXTENSION_ID
from agent_libos.models import (
    CapabilityRight,
    McpHeaderSpec,
    McpHttpTransportSpec,
    McpProtocolMode,
    McpToolSpec,
    ResourceBudget,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.primitives.mcp import McpPrimitive
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.serde import dumps, to_jsonable


_TASKS_PIN = "9" * 64


class _ResourceProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self, result: Any) -> None:
        self.result = result

    async def list_resources(
        self, _server: Any, _cursor: str | None, *, deadline: float
    ) -> McpPage[McpResource]:
        assert deadline > 0
        return McpPage(items=())

    async def list_resource_templates(
        self, _server: Any, _cursor: str | None, *, deadline: float
    ) -> McpPage[McpResourceTemplate]:
        assert deadline > 0
        return McpPage(items=())

    async def read_resource(
        self,
        _server: Any,
        _resource_name: str,
        _variables: Any,
        *,
        deadline: float,
    ) -> Any:
        assert deadline > 0
        return self.result


class _PromptProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self, *, prompt_result: Any, completion_result: Any) -> None:
        self.prompt_result = prompt_result
        self.completion_result = completion_result

    async def list_prompts(
        self, _server: Any, _cursor: str | None, *, deadline: float
    ) -> McpPage[McpPrompt]:
        assert deadline > 0
        return McpPage(items=())

    async def get_prompt(
        self,
        _server: Any,
        _prompt_name: str,
        _arguments: Any,
        *,
        deadline: float,
    ) -> Any:
        assert deadline > 0
        return self.prompt_result

    async def complete(
        self,
        _server: Any,
        _reference: Any,
        _argument: Any,
        _context: Any,
        *,
        deadline: float,
    ) -> Any:
        assert deadline > 0
        return self.completion_result


class _ToolProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self, result: Any) -> None:
        self.result = result
        self.sensitive_values: tuple[str, ...] | None = None

    async def call_tool(
        self,
        _manifest: Any,
        _tool_id: str,
        _arguments: dict[str, Any],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> Any:
        assert deadline > 0
        self.sensitive_values = sensitive_values
        return self.result


class _RaisingToolProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self) -> None:
        self.sensitive_values: tuple[str, ...] | None = None

    async def call_tool(
        self,
        _manifest: Any,
        _tool_id: str,
        _arguments: dict[str, Any],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> Any:
        assert deadline > 0
        assert sensitive_values
        self.sensitive_values = sensitive_values
        raise RuntimeError(f"custom Provider reflected {sensitive_values[0]}")


def _manifest(
    *,
    tasks: bool = False,
    header_env: str | None = None,
) -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id="custom-modern",
        transport="streamable_http",
        http=McpHttpTransportSpec(
            url="http://127.0.0.1:8765/mcp",
            headers=(
                {"Authorization": McpHeaderSpec(env=header_env, prefix="Bearer ")}
                if header_env is not None
                else {}
            ),
        ),
        # These tests exercise result postconditions, not short-deadline
        # behavior.  Leave enough time for protected setup on loaded runners.
        timeout_s=10.0,
        max_request_bytes=16_384,
        max_response_bytes=16_384,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        tools=(
            McpToolSpec(
                tool_id="review_tool",
                mcp_name="provider.review_tool",
                right="read",
                rollback_class="no_rollback_required",
                rollback_status="not_required",
                state_mutation=False,
                information_flow=True,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                },
            ),
        ),
        resources=(
            McpResourceSpec(
                resource_id="document",
                remote_uri="opaque://provider/document",
            ),
        ),
        prompts=(
            McpPromptSpec(
                prompt_id="review",
                mcp_name="provider.review",
                argument_names=("subject",),
            ),
        ),
        tasks_extension=(
            McpTasksExtensionSpec(
                extension_id=MCP_TASKS_EXTENSION_ID,
                spec_sha256=_TASKS_PIN,
            )
            if tasks
            else None
        ),
    )


def _tasks_config() -> AgentLibOSConfig:
    from dataclasses import replace

    return AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=_TASKS_PIN,
        )
    )


def _assert_no_public_or_durable_ref(
    runtime: Runtime,
    database: Path,
    ref: str,
) -> None:
    assert runtime.uow.mcp_continuations.list() == ()
    assert runtime.uow.mcp_remote_tasks.list() == ()
    audit = dumps([to_jsonable(row) for row in runtime.audit.trace()])
    assert ref not in audit
    assert ref.encode("utf-8") not in database.read_bytes()


def test_custom_resource_cannot_publish_unbacked_input_required(
    tmp_path: Path,
) -> None:
    forged_ref = "forged-resource-continuation"
    database = tmp_path / "custom-resource.sqlite"
    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate.mcp_resource_provider = _ResourceProvider(
        McpInputRequired(continuation_id=forged_ref)
    )
    runtime = Runtime.open(database, substrate=substrate)
    try:
        runtime.mcp.register_server(
            _manifest(), actor="runtime", require_capability=False
        )

        with pytest.raises(ValidationError) as captured:
            runtime.mcp.read_resource(
                "custom-modern",
                "document",
                actor="runtime",
            )

        assert forged_ref not in str(captured.value)
        _assert_no_public_or_durable_ref(runtime, database, forged_ref)
    finally:
        runtime.close()


def test_custom_tool_cannot_publish_unbacked_input_required(
    tmp_path: Path,
) -> None:
    forged_ref = "forged-tool-continuation"
    database = tmp_path / "custom-tool.sqlite"
    substrate = LocalResourceProviderSubstrate(tmp_path)
    provider = _ToolProvider(McpInputRequired(continuation_id=forged_ref))
    substrate.mcp_v3_tool_provider = provider
    runtime = Runtime.open(database, substrate=substrate)
    try:
        runtime.mcp.register_server(
            _manifest(), actor="runtime", require_capability=False
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject forged MCP continuation",
            resource_budget=ResourceBudget(max_mcp_bytes=64_000),
        )
        runtime.capability.grant(
            pid,
            "mcp:custom-modern:review_tool",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            "mcp_server:custom-modern",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )

        with pytest.raises(
            ValidationError,
            match="exact durable provenance",
        ) as captured:
            runtime.mcp.call_tool(pid, "custom-modern", "review_tool", {})

        assert provider.sensitive_values == ()
        assert forged_ref not in str(captured.value)
        _assert_no_public_or_durable_ref(runtime, database, forged_ref)
    finally:
        runtime.close()


def test_typed_sampling_unsupported_is_public_but_never_durable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "custom-tool-sampling-unsupported.sqlite"
    unsupported = McpInputRequired(
        input_requests=(
            McpInputRequest(
                request_id="input-1",
                kind=McpInputRequestKind.SAMPLING_UNSUPPORTED,
            ),
        ),
        respondable=False,
    )
    substrate = LocalResourceProviderSubstrate(tmp_path)
    provider = _ToolProvider(unsupported)
    substrate.mcp_v3_tool_provider = provider
    runtime = Runtime.open(database, substrate=substrate)
    try:
        runtime.mcp.register_server(
            _manifest(), actor="runtime", require_capability=False
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="observe typed unsupported MCP input",
            resource_budget=ResourceBudget(max_mcp_bytes=64_000),
        )
        runtime.capability.grant(
            pid,
            "mcp:custom-modern:review_tool",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            "mcp_server:custom-modern",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )

        assert runtime.mcp.call_tool(
            pid, "custom-modern", "review_tool", {}
        ) == unsupported
        assert runtime.uow.mcp_continuations.list() == ()
        assert runtime.uow.mcp_side_effects.list(
            operation_kind="continuation"
        ) == ()
    finally:
        runtime.close()


def test_forged_nonrespondable_elicitation_is_rejected() -> None:
    primitive = object.__new__(McpPrimitive)
    forged = McpInputRequired(
        input_requests=(
            McpInputRequest(
                request_id="input-1",
                kind=McpInputRequestKind.ELICITATION,
                mode="form",
            ),
        ),
        respondable=False,
    )

    with pytest.raises(ValidationError, match="nonrespondable"):
        primitive._require_typed_unsupported_input_result(  # noqa: SLF001
            forged,
            "effect-forged",
        )


def test_custom_tool_exception_redacts_active_header_secret_for_runtime_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "CUSTOM_MODERN_TOOL_HEADER_SECRET_SENTINEL"
    env_name = "AGENT_LIBOS_MCP_CUSTOM_MODERN_SECRET"
    monkeypatch.setenv(env_name, secret)
    database = tmp_path / "custom-tool-error.sqlite"
    substrate = LocalResourceProviderSubstrate(tmp_path)
    provider = _RaisingToolProvider()
    substrate.mcp_v3_tool_provider = provider
    runtime = Runtime.open(database, substrate=substrate)
    try:
        runtime.mcp.register_server(
            _manifest(header_env=env_name),
            actor="runtime",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject reflected modern MCP Provider credentials",
            resource_budget=ResourceBudget(max_mcp_bytes=1_000_000),
        )
        runtime.capability.grant(
            pid,
            "mcp:custom-modern:review_tool",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            "mcp_server:custom-modern",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )

        with pytest.raises(ValidationError) as runtime_error:
            runtime.mcp.call_tool(pid, "custom-modern", "review_tool", {})
        assert "MCP provider operation failed" in str(runtime_error.value)
        assert secret not in str(runtime_error.value)
        assert "Bearer " + secret not in str(runtime_error.value)
        assert provider.sensitive_values is not None
        assert secret in provider.sensitive_values

        monkeypatch.setattr(
            "agent_libos.api.cli.Runtime.open",
            lambda *_args, **_kwargs: runtime,
        )
        monkeypatch.setattr(
            "agent_libos.api.cli._shutdown_runtime_before_exit",
            lambda _runtime: None,
        )
        capsys.readouterr()
        with pytest.raises(SystemExit) as cli_exit:
            mcp_cli(
                [
                    "--db",
                    ":memory:",
                    "mcp",
                    "call",
                    pid,
                    "custom-modern",
                    "review_tool",
                ]
            )
        assert cli_exit.value.code == 1
        captured = capsys.readouterr()
        assert captured.err == ""
        cli_error = json.loads(captured.out)
        assert cli_error["error"]["type"] == "ValidationError"
        assert "MCP provider operation failed" in cli_error["error"]["message"]
        assert secret not in captured.out

        evidence = dumps(
            {
                "audit": [to_jsonable(row) for row in runtime.audit.trace()],
                "effects": [
                    to_jsonable(row)
                    for row in runtime.store.list_external_effects(pid=pid)
                ],
            }
        )
        assert secret not in evidence
        assert secret.encode("utf-8") not in database.read_bytes()
    finally:
        runtime.close()


def test_custom_prompt_cannot_publish_unbacked_remote_task_even_when_pinned(
    tmp_path: Path,
) -> None:
    forged_ref = "forged-prompt-task"
    database = tmp_path / "custom-prompt.sqlite"
    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate.mcp_prompt_provider = _PromptProvider(
        prompt_result=McpRemoteTask(task_ref=forged_ref),
        completion_result=McpComplete(value=McpCompletionResult(values=("ok",))),
    )
    runtime = Runtime.open(database, substrate=substrate, config=_tasks_config())
    try:
        runtime.mcp.register_server(
            _manifest(tasks=True), actor="runtime", require_capability=False
        )

        with pytest.raises(ValidationError) as captured:
            runtime.mcp.get_prompt(
                "custom-modern",
                "review",
                arguments={"subject": "release"},
                actor="runtime",
            )

        assert forged_ref not in str(captured.value)
        _assert_no_public_or_durable_ref(runtime, database, forged_ref)
    finally:
        runtime.close()


def test_custom_completion_cannot_bypass_nonrespondable_surface(
    tmp_path: Path,
) -> None:
    forged_ref = "forged-completion-continuation"
    database = tmp_path / "custom-completion.sqlite"
    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate.mcp_prompt_provider = _PromptProvider(
        prompt_result=McpComplete(
            value=McpPromptResult(prompt_id="review", messages=())
        ),
        completion_result=McpInputRequired(continuation_id=forged_ref),
    )
    runtime = Runtime.open(database, substrate=substrate)
    try:
        runtime.mcp.register_server(
            _manifest(), actor="runtime", require_capability=False
        )

        with pytest.raises(McpContinuationSurfaceUnsupported) as captured:
            runtime.mcp.complete_prompt(
                "custom-modern",
                "prompt",
                "review",
                {"name": "subject", "value": "release"},
                actor="runtime",
            )

        assert forged_ref not in str(captured.value)
        _assert_no_public_or_durable_ref(runtime, database, forged_ref)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "apps_value",
    (
        "ui://custom-completion-operation-secret",
        'TEXT/HTML ; PROFILE = "MCP-APP"',
    ),
    ids=("selector-with-secret", "apps-mime"),
)
def test_custom_completion_rejects_apps_values_before_public_sinks(
    tmp_path: Path,
    apps_value: str,
) -> None:
    database = tmp_path / "custom-completion-apps.sqlite"
    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate.mcp_prompt_provider = _PromptProvider(
        prompt_result=McpComplete(
            value=McpPromptResult(prompt_id="review", messages=())
        ),
        completion_result=McpComplete(
            value=McpCompletionResult(values=(apps_value,))
        ),
    )
    runtime = Runtime.open(database, substrate=substrate)
    try:
        runtime.mcp.register_server(
            _manifest(), actor="runtime", require_capability=False
        )

        with pytest.raises(ValidationError, match="Apps") as captured:
            runtime.mcp.complete_prompt(
                "custom-modern",
                "prompt",
                "review",
                {"name": "subject", "value": "release"},
                actor="runtime",
            )

        assert apps_value not in str(captured.value)
        _assert_no_public_or_durable_ref(runtime, database, apps_value)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "provider_result",
    (
        {
            "resultType": "complete",
            "content": [{"type": "text", "text": "sealed-state"}],
        },
        {
            "resultType": "input_required",
            "requestState": "sealed-state",
            "inputRequests": {
                "next": {
                    "method": "elicitation/create",
                    "params": {
                        "message": "sealed-state",
                        "requestedSchema": {"type": "object"},
                    },
                }
            },
        },
        {
            "resultType": "task",
            "taskId": "provider-task",
            "status": "working",
            "statusMessage": "sealed-state",
        },
    ),
    ids=("complete", "input-required", "remote-task"),
)
def test_custom_continuation_result_is_sanitized_before_durable_settlement(
    provider_result: dict[str, Any],
) -> None:
    secret = "custom-continuation-operation-secret"
    binding = McpClientBinding(
        manifest=_manifest(tasks=True),
        registry_generation=1,
        owner_id="runtime",
        sensitive_values=(secret,),
    )
    primitive = object.__new__(McpPrimitive)
    primitive.config = _tasks_config()

    async def continue_resource(
        server: Any,
        resource_name: str,
        logical_id: str,
        input_responses: dict[str, Any],
        request_state: str | None,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        assert server.server_id == "custom-modern"
        assert resource_name == "opaque://provider/document"
        assert logical_id == "document"
        assert input_responses == {"input": {"action": "accept"}}
        assert request_state == "sealed-state"
        assert deadline > 0
        return {
            **provider_result,
            "providerSecret": secret,
            "_meta": {"UI/ResourceUri": "ui://must-drop"},
        }

    result = asyncio.run(
        primitive._continue_v3_provider_call(  # noqa: SLF001
            continue_resource,
            binding,
            {"method": "resources/read"},
            {
                "remote_name": "opaque://provider/document",
                "logical_id": "document",
            },
            {"input": {"action": "accept"}},
            "sealed-state",
            10**12,
        )
    )

    public = dumps(result)
    assert secret not in public
    assert "sealed-state" not in public
    assert "ui://must-drop" not in public
    assert "UI/ResourceUri" not in public


def test_custom_continuation_exception_redacts_binding_and_request_state() -> None:
    secret = "custom-continuation-operation-secret"
    request_state = "sealed-continuation-request-state"
    binding = McpClientBinding(
        manifest=_manifest(),
        registry_generation=1,
        owner_id="runtime",
        sensitive_values=(secret,),
    )
    primitive = object.__new__(McpPrimitive)

    async def continue_resource(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(f"peer reflected {secret} and {request_state}")

    with pytest.raises(ValidationError) as captured:
        asyncio.run(
            primitive._continue_v3_provider_call(  # noqa: SLF001
                continue_resource,
                binding,
                {"method": "resources/read"},
                {
                    "remote_name": "opaque://provider/document",
                    "logical_id": "document",
                },
                {"input": {"action": "accept"}},
                request_state,
                10**12,
            )
        )

    assert "MCP provider operation failed" in str(captured.value)
    assert secret not in str(captured.value)
    assert request_state not in str(captured.value)
