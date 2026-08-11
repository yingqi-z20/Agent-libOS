from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Callable

import pytest

from agent_libos import Runtime
from agent_libos.mcp.manifest import (
    McpPromptSpec,
    McpResourceSpec,
    McpServerManifestV3,
)
from agent_libos.mcp.types import (
    McpComplete,
    McpPage,
    McpPromptMessage,
    McpPromptResult,
    McpResource,
    McpTextContent,
)
from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    McpHttpTransportSpec,
    McpProtocolMode,
    McpToolSpec,
    ResourceBudget,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.substrate import ProviderEffectNotStarted
from agent_libos.utils.ids import utc_now


def _manifest(*, timeout_s: float = 1.0) -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id="modern-protected",
        transport="streamable_http",
        http=McpHttpTransportSpec(url="http://127.0.0.1:8765/mcp"),
        timeout_s=timeout_s,
        max_request_bytes=4096,
        max_response_bytes=4096,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        resources=(
            McpResourceSpec(
                resource_id="status",
                remote_uri="file:///provider/status",
                model_visible=True,
            ),
        ),
        prompts=(
            McpPromptSpec(
                prompt_id="review",
                mcp_name="provider.review",
                argument_names=("topic",),
            ),
        ),
    )


def _tool_manifest(*, state_mutation: bool = False) -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id="modern-tool",
        transport="streamable_http",
        http=McpHttpTransportSpec(url="http://127.0.0.1:8765/mcp"),
        timeout_s=1.0,
        max_request_bytes=4096,
        max_response_bytes=4096,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        tools=(
            McpToolSpec(
                tool_id="echo",
                mcp_name="provider.echo",
                right="write" if state_mutation else "read",
                rollback_class="irreversible" if state_mutation else "no_rollback_required",
                rollback_status="not_supported" if state_mutation else "not_required",
                state_mutation=state_mutation,
                information_flow=True,
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
        ),
    )


class _ResourcePromptProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.on_call: Callable[[], None] | None = None
        self.delay_s = 0.0
        self.prompt_text = "review this"

    async def list_resources(
        self,
        _server: Any,
        _cursor: str | None,
        *,
        deadline: float,
    ) -> McpPage[McpResource]:
        del deadline
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return McpPage(
            items=(
                McpResource(
                    resource_id="file:///provider/status",
                    name="Status",
                ),
            )
        )

    async def get_prompt(
        self,
        _server: Any,
        prompt_name: str,
        arguments: dict[str, str],
        *,
        deadline: float,
    ) -> McpComplete[McpPromptResult]:
        del deadline
        assert prompt_name == "provider.review"
        assert arguments == {"topic": "MCP"}
        self.calls += 1
        return McpComplete(
            value=McpPromptResult(
                prompt_id="provider-controlled",
                messages=(
                    McpPromptMessage(
                        role="user",
                        content=McpTextContent(text=self.prompt_text),
                    ),
                ),
                user_confirmation_required=False,
            )
        )


class _ModernToolProvider:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.calls = 0

    async def call_tool(
        self,
        manifest: McpServerManifestV3,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpComplete[dict[str, Any]]:
        assert sensitive_values == ()
        assert deadline > 0
        assert manifest.server_id == "modern-tool"
        assert tool_id == "echo"
        assert arguments == {"text": "hello"}
        effects = [
            item
            for item in self.runtime.store.list_external_effects()
            if item.provider == "mcp" and item.operation == "call_tool"
        ]
        assert effects and effects[-1].effect_state == "pending"
        self.calls += 1
        return McpComplete(value={"content": [{"type": "text", "text": "hello"}]})

@pytest.fixture
def modern_runtime() -> tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3]:
    runtime = Runtime.open(":memory:")
    manifest = _manifest()
    runtime.mcp.register_server(
        manifest,
        actor="runtime",
        require_capability=False,
    )
    provider = _ResourcePromptProvider()
    runtime.mcp._modern_client.resource_provider = provider  # noqa: SLF001
    runtime.mcp._modern_client.prompt_provider = provider  # noqa: SLF001
    try:
        yield runtime, provider, manifest
    finally:
        runtime.close()


def _spawn_reader(runtime: Runtime, *, max_mcp_bytes: int = 16_384) -> str:
    return runtime.process.spawn(
        image="base-agent:v0",
        goal="read governed MCP v3 surface",
        resource_budget=ResourceBudget(max_mcp_bytes=max_mcp_bytes),
    )


def _grant_catalog(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(
        pid,
        "mcp_server:modern-protected",
        [CapabilityRight.READ, CapabilityRight.EXECUTE],
        issued_by="test",
    )


def test_modern_process_denial_precedes_registry_environment_and_provider(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider, _manifest_value = modern_runtime
    pid = _spawn_reader(runtime)
    monkeypatch.setattr(
        runtime.uow.extensions,
        "get_mcp_v3_server",
        lambda _server_id: (_ for _ in ()).throw(
            AssertionError("denied MCP caller must not read the registry")
        ),
    )
    monkeypatch.setattr(
        runtime.mcp,
        "_require_runtime_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("denied MCP caller must not resolve Host environment")
        ),
    )

    with pytest.raises(CapabilityDenied):
        runtime.mcp.list_resources(
            "modern-protected",
            actor=pid,
            model_visible_only=True,
        )

    assert provider.calls == 0


def test_modern_ask_creates_authority_request_before_provider(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
) -> None:
    runtime, provider, _manifest_value = modern_runtime
    pid = _spawn_reader(runtime)
    runtime.capability.set_permission_policy(
        pid,
        "mcp_server:modern-protected",
        [CapabilityRight.READ],
        runtime.capability.ASK_EACH_TIME,
        issued_by="test",
    )
    runtime.capability.grant(
        pid,
        "mcp_server:modern-protected",
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )

    with pytest.raises(HumanApprovalRequired) as caught:
        runtime.mcp.list_resources("modern-protected", actor=pid)

    assert provider.calls == 0
    pending = runtime.human.pending()
    assert [item.request_id for item in pending] == [caught.value.request_id]
    assert pending[0].payload["type"] == "external_operation_approval"
    conditions = pending[0].payload["requested_once_capability"]["constraints"]
    assert "authority_rules" in conditions


def test_modern_success_is_pending_first_and_settles_budget_audit_event_and_effect(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
) -> None:
    runtime, provider, _manifest_value = modern_runtime
    pid = _spawn_reader(runtime)
    _grant_catalog(runtime, pid)

    def assert_pending() -> None:
        effects = [
            item
            for item in runtime.store.list_external_effects()
            if item.provider == "mcp" and item.operation == "resources.list"
        ]
        assert effects
        assert effects[-1].effect_state == "pending"
        assert effects[-1].transaction_state in {
            "prepared",
            "authorized",
            "approved",
            "dispatched",
        }

    provider.on_call = assert_pending
    page = runtime.mcp.list_resources(
        "modern-protected",
        actor=pid,
        model_visible_only=True,
    )

    assert [item.resource_id for item in page.items] == ["status"]
    process = runtime.process.get(pid)
    assert process.resource_usage.mcp_request_bytes > 0
    assert process.resource_usage.mcp_response_bytes > 0
    effect = [
        item
        for item in runtime.store.list_external_effects()
        if item.provider == "mcp" and item.operation == "resources.list"
    ][-1]
    assert effect.effect_state == "finalized"
    assert effect.transaction_state == "committed"
    assert effect.rollback_class.value == "no_rollback_required"
    assert effect.provider_metadata["data_flow"]["sink"].startswith(
        "mcp:modern-protected:resources.list"
    )
    assert any(
        item.action == "primitive.mcp.resources.list"
        for item in runtime.audit.trace()
    )
    event = runtime.events.list(target="mcp_server:modern-protected")[-1]
    assert event.payload["operation"] == "resources.list"


def test_modern_host_internal_path_still_records_effect_and_evidence(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
) -> None:
    runtime, _provider, _manifest_value = modern_runtime

    page = runtime.mcp.list_resources("modern-protected", actor="gui")

    assert page.items[0].resource_id == "status"
    effect = [
        item
        for item in runtime.store.list_external_effects()
        if item.provider == "mcp" and item.operation == "resources.list"
    ][-1]
    assert effect.pid == "gui"
    assert effect.transaction_state == "committed"
    protected = effect.provider_metadata["protected_operation"]
    assert protected["contract_name"] == "primitive.mcp.resources.list.internal"


def test_modern_data_flow_denial_and_budget_denial_precede_provider(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
) -> None:
    runtime, provider, _manifest_value = modern_runtime
    secret_pid = _spawn_reader(runtime)
    _grant_catalog(runtime, secret_pid)
    secret_context = DataFlowContext(
        labels=DataLabels(
            sensitivity="secret",
            trust_level="verified",
            integrity="verified",
            origin="object:test-secret",
        )
    )
    with runtime.data_flow.activate(secret_context):
        with pytest.raises(CapabilityDenied):
            runtime.mcp.list_resources("modern-protected", actor=secret_pid)
    assert provider.calls == 0

    budget_pid = _spawn_reader(runtime, max_mcp_bytes=1024)
    _grant_catalog(runtime, budget_pid)
    with pytest.raises(ResourceLimitExceeded):
        runtime.mcp.list_resources("modern-protected", actor=budget_pid)
    assert provider.calls == 0


def test_modern_registry_race_discards_result_and_deadline_is_absolute(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
) -> None:
    runtime, provider, manifest = modern_runtime

    def replace_registry() -> None:
        replacement = replace(manifest, metadata={"revision": 2})
        runtime.uow.extensions.upsert_mcp_v3_server(
            replacement,
            registered_by="race",
            created_at=utc_now(),
        )

    provider.on_call = replace_registry
    with pytest.raises(ValidationError, match="fence changed|registry"):
        runtime.mcp.list_resources("modern-protected", actor="gui")
    raced = [
        item
        for item in runtime.store.list_external_effects()
        if item.provider == "mcp" and item.operation == "resources.list"
    ][-1]
    assert raced.transaction_state == "unknown"
    assert raced.provider_metadata["outcome"] == "unknown_after_provider_exception"

    runtime.mcp.register_server(
        _manifest(timeout_s=0.02),
        actor="runtime",
        replace=True,
        require_capability=False,
    )
    provider.on_call = None
    provider.delay_s = 0.2
    with pytest.raises(TimeoutError, match="deadline"):
        runtime.mcp.list_resources("modern-protected", actor="gui")


def test_modern_not_started_and_provider_failure_have_distinct_settlement(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, _manifest_value = modern_runtime
    client = runtime.mcp._modern_client
    before = tuple(runtime.store.list_external_effects())

    def certified_not_started(*_args: Any, **_kwargs: Any) -> Any:
        raise ProviderEffectNotStarted("transport was never entered")

    monkeypatch.setattr(client, "list_resources", certified_not_started)
    with pytest.raises(ProviderEffectNotStarted):
        runtime.mcp.list_resources("modern-protected", actor="gui")
    # A certified pre-dispatch failure removes the pending intent instead of
    # manufacturing an UNKNOWN external effect.
    assert tuple(runtime.store.list_external_effects()) == before

    reflected = "provider-secret-diagnostic"

    def provider_failed(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(reflected)

    monkeypatch.setattr(client, "list_resources", provider_failed)
    with pytest.raises(RuntimeError) as raised:
        runtime.mcp.list_resources("modern-protected", actor="gui")
    failed = runtime.store.list_external_effects()[-1]
    assert failed.transaction_state == "unknown"
    assert failed.provider_metadata["outcome"] == "unknown_after_provider_exception"
    assert reflected not in repr(failed)
    assert reflected in str(raised.value)


def test_modern_post_provider_oversize_is_unknown_not_local_preflight(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, _manifest_value = modern_runtime

    def oversized(*_args: Any, **_kwargs: Any) -> McpPage[McpResource]:
        effect = runtime.store.list_external_effects()[-1]
        assert effect.effect_state == "pending"
        return McpPage(
            items=(
                McpResource(
                    resource_id="status",
                    name="x" * 10_000,
                ),
            )
        )

    monkeypatch.setattr(runtime.mcp._modern_client, "list_resources", oversized)
    with pytest.raises(ValidationError, match="max_response_bytes"):
        runtime.mcp.list_resources("modern-protected", actor="gui")
    effect = runtime.store.list_external_effects()[-1]
    assert effect.transaction_state == "unknown"
    assert effect.provider_metadata["outcome"] == "unknown_after_provider_success"
    assert effect.provider_metadata["phase"] == "caller_failed_after_provider"


def test_modern_auth_fence_rotation_discards_provider_result(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider, _manifest_value = modern_runtime
    resolver = runtime.mcp._modern_client.binding_resolver
    original_resolve = resolver.resolve
    rotated = False

    def resolve(server_id: str, *, owner_id: str | None) -> Any:
        binding = original_resolve(server_id, owner_id=owner_id)
        return replace(binding, auth_generation=1 if rotated else 0)

    def rotate_auth_fence() -> None:
        nonlocal rotated
        rotated = True

    monkeypatch.setattr(resolver, "resolve", resolve)
    provider.on_call = rotate_auth_fence
    with pytest.raises(ValidationError, match="authentication fence"):
        runtime.mcp.list_resources("modern-protected", actor="gui")
    effect = runtime.store.list_external_effects()[-1]
    assert effect.transaction_state == "unknown"
    assert effect.provider_metadata["outcome"] == "unknown_after_provider_exception"


def test_prompt_confirmation_is_bound_to_current_public_projection(
    modern_runtime: tuple[Runtime, _ResourcePromptProvider, McpServerManifestV3],
) -> None:
    runtime, provider, _manifest_value = modern_runtime
    preview = runtime.mcp.get_prompt(
        "modern-protected",
        "review",
        arguments={"topic": "MCP"},
        actor="gui",
    )
    assert isinstance(preview, McpComplete)
    assert preview.preview_sha256 is not None
    assert preview.value is not None
    assert preview.value.user_confirmation_required is True

    provider.prompt_text = "changed after preview"
    with pytest.raises(CapabilityDenied, match="preview changed"):
        runtime.mcp.get_prompt(
            "modern-protected",
            "review",
            arguments={"topic": "MCP"},
            confirmed=True,
            expected_preview_sha256=preview.preview_sha256,
            actor="gui",
        )


def test_v3_tool_uses_exact_modern_provider_once_inside_existing_call_boundary() -> None:
    runtime = Runtime.open(":memory:")
    try:
        manifest = _tool_manifest()
        runtime.mcp.register_server(
            manifest,
            actor="runtime",
            require_capability=False,
        )
        provider = _ModernToolProvider(runtime)
        runtime.mcp._modern_tool_provider = provider  # noqa: SLF001
        pid = _spawn_reader(runtime)
        runtime.capability.grant(
            pid,
            "mcp:modern-tool:echo",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            "mcp_server:modern-tool",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )

        result = runtime.mcp.call_tool(
            pid,
            "modern-tool",
            "echo",
            {"text": "hello"},
        )

        assert result == McpComplete(
            value={"content": [{"type": "text", "text": "hello"}]}
        )
        assert provider.calls == 1
        effects = [
            item
            for item in runtime.store.list_external_effects()
            if item.provider == "mcp" and item.operation == "call_tool"
        ]
        assert effects[-1].transaction_state == "committed"
        assert effects[-1].provider_metadata["protected_operation"]["contract_name"] == (
            "primitive.mcp.call"
        )
    finally:
        runtime.close()


def test_v3_tool_schema_and_authority_fail_before_modern_provider() -> None:
    runtime = Runtime.open(":memory:")
    try:
        runtime.mcp.register_server(
            _tool_manifest(),
            actor="runtime",
            require_capability=False,
        )
        provider = _ModernToolProvider(runtime)
        runtime.mcp._modern_tool_provider = provider  # noqa: SLF001
        denied = _spawn_reader(runtime)
        with pytest.raises(CapabilityDenied):
            runtime.mcp.call_tool(
                denied,
                "modern-tool",
                "echo",
                {"text": "hello"},
            )
        assert provider.calls == 0

        pid = _spawn_reader(runtime)
        runtime.capability.grant(
            pid,
            "mcp:modern-tool:echo",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            "mcp_server:modern-tool",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        with pytest.raises(ValidationError, match="schema|valid"):
            runtime.mcp.call_tool(
                pid,
                "modern-tool",
                "echo",
                {"unexpected": True},
            )
        assert provider.calls == 0
    finally:
        runtime.close()
