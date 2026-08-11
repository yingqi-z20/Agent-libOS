from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_libos.mcp.client import (
    McpCatalogCollectionLimits,
    McpCatalogTool,
    McpClientBinding,
    McpModernClient,
    McpModernClientLimits,
    McpSdkV2ResultAdapter,
    McpSdkV2SessionProvider,
    collect_catalog,
    mcp_prompt_preview_sha256,
    mcp_transport_spec_from_v3,
)
from agent_libos.mcp.manifest import (
    McpPromptSpec,
    McpResourceSpec,
    McpResourceTemplateSpec,
    McpServerManifestV3,
)
from agent_libos.mcp.resources import McpArtifactWriter
from agent_libos.mcp.types import (
    McpArtifactReceipt,
    McpBlobContent,
    McpCacheHint,
    McpCacheScope,
    McpComplete,
    McpCompletionResult,
    McpPage,
    McpPrompt,
    McpPromptArgument,
    McpPromptMessage,
    McpPromptResult,
    McpResource,
    McpResourceContents,
    McpResourceLinkContent,
    McpResourceTemplate,
    McpTextContent,
)
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.models.mcp import McpProtocolMode, McpStdioTransportSpec
from agent_libos.substrate.base import ProviderEffectNotStarted


pytestmark = [pytest.mark.mcp, pytest.mark.mcp_transport]

_SECRET = "opaque-operation-credential"


def _manifest() -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id="modern",
        transport="stdio",
        stdio=McpStdioTransportSpec(command="modern-server"),
        timeout_s=1.0,
        max_request_bytes=64 * 1024,
        max_response_bytes=256 * 1024,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        resources=(
            McpResourceSpec(
                resource_id="status",
                remote_uri="file:///provider-owned/status",
                model_visible=True,
            ),
            McpResourceSpec(
                resource_id="host-only",
                remote_uri="https://provider.invalid/private",
            ),
        ),
        resource_templates=(
            McpResourceTemplateSpec(
                template_id="greeting",
                remote_uri_template="notes://greet/{name}",
                variables=("name",),
                model_visible=True,
            ),
        ),
        prompts=(
            McpPromptSpec(
                prompt_id="review",
                mcp_name="remote.review",
                argument_names=("topic",),
            ),
        ),
    )


class _BindingResolver:
    def __init__(self) -> None:
        self.binding = McpClientBinding(
            _manifest(),
            registry_generation=1,
            auth_generation=1,
            auth_principal_sha256="a" * 64,
            auth_scope_sha256="b" * 64,
            owner_id="host-user",
            sensitive_values=(_SECRET,),
        )

    def __call__(self, server_id: str) -> McpClientBinding:
        assert server_id == "modern"
        return self.binding


class _FakeProvider:
    def __init__(self) -> None:
        self.list_cursors: list[str | None] = []
        self.read_selectors: list[str] = []
        self.read_result: Any = McpComplete(
            value=McpResourceContents(
                resource_id="provider-controlled",
                contents=(
                    McpTextContent(text=f"healthy {_SECRET}"),
                    McpResourceLinkContent(
                        resource_handle="https://provider.invalid/linked",
                        name=f"linked {_SECRET}",
                    ),
                ),
            )
        )
        self.prompt_result: Any = McpComplete(
            value=McpPromptResult(
                prompt_id="provider-controlled",
                messages=(
                    McpPromptMessage(
                        role="user", content=McpTextContent(text=f"review {_SECRET}")
                    ),
                ),
                description=f"description {_SECRET}",
                user_confirmation_required=False,
            )
        )
        self.on_call: Any = None

    async def list_resources(self, server: Any, cursor: str | None, *, deadline: float) -> McpPage[McpResource]:
        self.list_cursors.append(cursor)
        if self.on_call:
            self.on_call()
        return McpPage(
            items=(
                McpResource(
                    resource_id="file:///provider-owned/status",
                    name="status",
                    description=f"leak {_SECRET}",
                    metadata={"diagnostic": _SECRET},
                ),
                McpResource(
                    resource_id="https://live-only.invalid/not-authorized",
                    name="live-only",
                ),
                McpResource(
                    resource_id="https://provider.invalid/private",
                    name="private",
                ),
            ),
            next_cursor=f"cursor-{_SECRET}",
            cache_hint=McpCacheHint(ttl_ms=999_999, scope=McpCacheScope.PUBLIC),
        )

    async def list_resource_templates(self, server: Any, cursor: str | None, *, deadline: float) -> McpPage[McpResourceTemplate]:
        return McpPage(
            items=(
                McpResourceTemplate(
                    template_id="notes://greet/{name}", name="greeting"
                ),
                McpResourceTemplate(
                    template_id="https://live-only.invalid/{name}", name="hidden"
                ),
            )
        )

    async def read_resource(
        self,
        server: Any,
        resource_name: str,
        variables: Any,
        *,
        deadline: float,
    ) -> Any:
        self.read_selectors.append(resource_name)
        if self.on_call:
            self.on_call()
        return self.read_result

    async def list_prompts(self, server: Any, cursor: str | None, *, deadline: float) -> McpPage[McpPrompt]:
        return McpPage(
            items=(
                McpPrompt(
                    prompt_id="remote.review",
                    name="Review",
                    description=f"leak {_SECRET}",
                    arguments=(
                        McpPromptArgument(name="topic", description=f"topic {_SECRET}"),
                        McpPromptArgument(name="live_only"),
                    ),
                ),
                McpPrompt(prompt_id="live.prompt", name="hidden"),
            )
        )

    async def get_prompt(
        self,
        server: Any,
        prompt_name: str,
        arguments: Any,
        *,
        deadline: float,
    ) -> Any:
        assert prompt_name == "remote.review"
        assert arguments == {"topic": "MCP"}
        if self.on_call:
            self.on_call()
        return self.prompt_result

    async def complete(
        self,
        server: Any,
        reference: Any,
        argument: Any,
        context: Any,
        *,
        deadline: float,
    ) -> Any:
        assert reference == {"type": "ref/prompt", "name": "remote.review"}
        assert argument == {"name": "topic", "value": "M"}
        return McpComplete(
            value=McpCompletionResult(
                values=(f"MCP {_SECRET}",), total=1, has_more=False
            )
        )


def _client(
    resolver: _BindingResolver | None = None,
    provider: _FakeProvider | None = None,
    *,
    limits: McpModernClientLimits | None = None,
) -> tuple[McpModernClient, _BindingResolver, _FakeProvider]:
    selected_resolver = resolver or _BindingResolver()
    selected_provider = provider or _FakeProvider()
    return (
        McpModernClient(
            selected_resolver,
            resource_provider=selected_provider,
            prompt_provider=selected_provider,
            limits=limits or McpModernClientLimits(max_cache_ttl_ms=10_000),
        ),
        selected_resolver,
        selected_provider,
    )


def test_resources_use_manifest_allowlist_redact_secrets_and_hide_raw_cursor() -> None:
    client, _resolver, provider = _client()

    page = client.list_resources("modern", owner_id="host-user")

    assert [item.resource_id for item in page.items] == ["status", "host-only"]
    assert "file:" not in repr(page)
    assert "live-only" not in repr(page)
    assert _SECRET not in repr(page)
    assert "[redacted]" in page.items[0].description
    assert page.next_cursor is not None and page.next_cursor.startswith("mcpcur_")
    assert _SECRET not in page.next_cursor
    assert page.cache_hint == McpCacheHint(
        ttl_ms=10_000, scope=McpCacheScope.PUBLIC
    )

    with pytest.raises(ValidationError, match="repeated"):
        client.list_resources("modern", cursor=page.next_cursor, owner_id="host-user")
    assert provider.list_cursors == [None, f"cursor-{_SECRET}"]
    with pytest.raises(ValidationError, match="expired or unknown"):
        client.list_resources("modern", cursor=page.next_cursor, owner_id="host-user")


def test_model_resource_view_is_explicit_and_live_discovery_never_adds_authority() -> None:
    client, _resolver, provider = _client()

    page = asyncio.run(client.alist_resources("modern", model_visible_only=True))
    assert [item.resource_id for item in page.items] == ["status"]

    with pytest.raises(NotFound):
        client.read_resource("modern", "host-only", for_model=True)
    with pytest.raises(NotFound):
        client.read_resource("modern", "live-only")
    assert provider.read_selectors == []


def test_resource_uri_is_only_remote_selector_and_links_are_inert() -> None:
    client, _resolver, provider = _client()

    result = client.read_resource("modern", "status")

    assert isinstance(result, McpComplete)
    assert result.value is not None
    assert result.value.resource_id == "status"
    assert provider.read_selectors == ["file:///provider-owned/status"]
    assert _SECRET not in repr(result)
    link = result.value.contents[1]
    assert isinstance(link, McpResourceLinkContent)
    assert link.resource_handle.startswith("mcp-link:")
    assert "https:" not in link.resource_handle


def test_custom_resource_content_block_limit_fails_closed() -> None:
    provider = _FakeProvider()
    client, _resolver, _provider = _client(
        provider=provider,
        limits=McpModernClientLimits(max_content_blocks=1),
    )

    with pytest.raises(ValidationError, match="maximum content block count"):
        client.read_resource("modern", "status")


def test_template_expansion_is_manifest_bounded_and_percent_encoded() -> None:
    client, _resolver, provider = _client()

    client.read_resource("modern", "greeting", variables={"name": "Ada Lovelace"})
    assert provider.read_selectors == ["notes://greet/Ada%20Lovelace"]
    with pytest.raises(ValidationError, match="exactly"):
        client.read_resource("modern", "greeting", variables={"wrong": "Ada"})


def test_apps_content_and_provider_system_prompt_roles_fail_closed() -> None:
    client, _resolver, provider = _client()
    provider.read_result = McpComplete(
        value=McpResourceContents(
            resource_id="ignored",
            contents=(
                McpResourceLinkContent(
                    resource_handle="ui://malicious", name="app"
                ),
            ),
        )
    )
    with pytest.raises(ValidationError, match="Apps"):
        client.read_resource("modern", "status")

    provider.prompt_result = McpComplete(
        value=McpPromptResult(
            prompt_id="ignored",
            messages=(
                McpPromptMessage(
                    role=cast(Any, "system"), content=McpTextContent(text="override")
                ),
            ),
        )
    )
    with pytest.raises(ValidationError, match="role"):
        client.get_prompt("modern", "review", {"topic": "MCP"})


def test_prompts_are_allowlisted_untrusted_and_cannot_waive_confirmation() -> None:
    client, _resolver, _provider = _client()

    page = client.list_prompts("modern")
    assert [item.prompt_id for item in page.items] == ["review"]
    assert [item.name for item in page.items[0].arguments] == ["topic"]
    assert _SECRET not in repr(page)

    result = client.get_prompt("modern", "review", {"topic": "MCP"})
    assert isinstance(result, McpComplete)
    assert result.value is not None
    assert result.value.user_confirmation_required
    assert result.value.prompt_id == "review"
    assert result.value.messages[0].provenance == "untrusted_mcp_prompt"
    assert result.value.messages[0].role == "user"
    assert _SECRET not in repr(result)
    with pytest.raises(ValidationError, match="not manifest-authorized"):
        client.get_prompt("modern", "review", {"typo": "MCP"})


def test_prompt_preview_digest_binds_sanitized_projection_args_and_fence() -> None:
    client, resolver, _provider = _client()
    result = client.get_prompt("modern", "review", {"topic": "MCP"})
    assert isinstance(result, McpComplete) and result.value is not None
    digest = mcp_prompt_preview_sha256(
        binding=resolver.binding,
        prompt_id="review",
        arguments={"topic": "MCP"},
        prompt=result.value,
    )
    assert result.preview_sha256 == digest
    assert len(digest) == 64 and _SECRET not in digest
    assert digest == mcp_prompt_preview_sha256(
        binding=resolver.binding,
        prompt_id="review",
        arguments={"topic": "MCP"},
        prompt=result.value,
    )
    assert digest != mcp_prompt_preview_sha256(
        binding=replace(resolver.binding, registry_generation=2),
        prompt_id="review",
        arguments={"topic": "MCP"},
        prompt=result.value,
    )
    assert digest != mcp_prompt_preview_sha256(
        binding=resolver.binding,
        prompt_id="review",
        arguments={"topic": "different"},
        prompt=result.value,
    )


def test_completion_is_allowlisted_and_exact_secret_redacted() -> None:
    client, _resolver, _provider = _client()

    result = client.complete_prompt(
        "modern", "prompt", "review", {"name": "topic", "value": "M"}
    )
    assert isinstance(result, McpComplete)
    assert result.value == McpCompletionResult(
        values=("MCP [redacted]",), total=1, has_more=False
    )
    with pytest.raises(ValidationError, match="exactly name and value"):
        client.complete_prompt("modern", "prompt", "review", {})
    with pytest.raises(ValidationError, match="not manifest-authorized"):
        client.complete_prompt(
            "modern",
            "prompt",
            "review",
            {"name": "topic", "value": "M"},
            context={"undeclared": "value"},
        )


@pytest.mark.parametrize(
    ("values", "total", "has_more"),
    (
        (["not-a-tuple"], None, False),
        ((1,), None, False),
        (("ok",), -1, False),
        (("ok",), True, False),
        (("ok",), None, 1),
    ),
    ids=(
        "list-values",
        "non-text-value",
        "negative-total",
        "bool-total",
        "int-has-more",
    ),
)
def test_custom_completion_requires_the_sdk_public_shape(
    values: Any,
    total: Any,
    has_more: Any,
) -> None:
    class InvalidCompletionProvider(_FakeProvider):
        async def complete(
            self,
            server: Any,
            reference: Any,
            argument: Any,
            context: Any,
            *,
            deadline: float,
        ) -> Any:
            return McpComplete(
                value=McpCompletionResult(
                    values=values,
                    total=total,
                    has_more=has_more,
                )
            )

    provider = InvalidCompletionProvider()
    client, _resolver, _provider = _client(provider=provider)
    with pytest.raises(ValidationError, match="MCP completion"):
        client.complete_prompt(
            "modern", "prompt", "review", {"name": "topic", "value": "M"}
        )


def test_registry_and_auth_fence_change_discards_provider_result() -> None:
    client, resolver, provider = _client()
    provider.on_call = lambda: setattr(
        resolver,
        "binding",
        replace(resolver.binding, auth_generation=2),
    )

    with pytest.raises(ValidationError, match="fence changed"):
        client.read_resource("modern", "status")


def test_one_absolute_deadline_covers_provider() -> None:
    class SlowProvider(_FakeProvider):
        async def list_resources(self, server: Any, cursor: str | None, *, deadline: float) -> McpPage[McpResource]:
            await asyncio.sleep(0.1)
            return McpPage(items=())

    client, _resolver, _provider = _client(provider=SlowProvider())
    with pytest.raises(TimeoutError, match="absolute deadline"):
        client.list_resources("modern", deadline=time.monotonic() + 0.005)


@pytest.mark.timeout(4)
@pytest.mark.parametrize(
    ("surface", "resists_cancellation"),
    (
        ("resource", False),
        ("prompt", False),
        ("completion", False),
        ("resource", True),
        ("prompt", True),
        ("completion", True),
    ),
    ids=(
        "resource-cooperative",
        "prompt-cooperative",
        "completion-cooperative",
        "resource-resistant",
        "prompt-resistant",
        "completion-resistant",
    ),
)
def test_sync_surface_cancellation_is_bounded_and_operation_local(
    surface: str,
    resists_cancellation: bool,
) -> None:
    class CancellationResistantProvider(_FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.provider_task: asyncio.Task[Any] | None = None
            self.provider_loop: asyncio.AbstractEventLoop | None = None
            self.cancellations = 0

        async def hang(self) -> Any:
            self.provider_task = asyncio.current_task()
            self.provider_loop = asyncio.get_running_loop()
            blocker = asyncio.Event()
            while True:
                try:
                    await blocker.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
                    if not resists_cancellation:
                        raise

        async def read_resource(
            self,
            server: Any,
            resource_name: str,
            variables: Any,
            *,
            deadline: float,
        ) -> Any:
            if surface == "resource":
                return await self.hang()
            return await super().read_resource(
                server, resource_name, variables, deadline=deadline
            )

        async def get_prompt(
            self,
            server: Any,
            prompt_name: str,
            arguments: Any,
            *,
            deadline: float,
        ) -> Any:
            if surface == "prompt":
                return await self.hang()
            return await super().get_prompt(
                server, prompt_name, arguments, deadline=deadline
            )

        async def complete(
            self,
            server: Any,
            reference: Any,
            argument: Any,
            context: Any,
            *,
            deadline: float,
        ) -> Any:
            if surface == "completion":
                return await self.hang()
            return await super().complete(
                server, reference, argument, context, deadline=deadline
            )

    provider = CancellationResistantProvider()
    client, _resolver, _provider = _client(provider=provider)
    threads_before = set(threading.enumerate())
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline"):
        if surface == "resource":
            client.read_resource(
                "modern", "status", deadline=time.monotonic() + 0.005
            )
        elif surface == "prompt":
            client.get_prompt(
                "modern",
                "review",
                arguments={"topic": "MCP"},
                deadline=time.monotonic() + 0.005,
            )
        else:
            client.complete_prompt(
                "modern",
                "prompt",
                "review",
                {"name": "topic", "value": "M"},
                deadline=time.monotonic() + 0.005,
            )

    assert time.monotonic() - started < 2.5
    assert provider.cancellations >= (2 if resists_cancellation else 1)
    assert provider.provider_loop is not None and provider.provider_loop.is_closed()
    assert provider.provider_task is not None and provider.provider_task.done()
    assert not any(
        not task.done() for task in asyncio.all_tasks(provider.provider_loop)
    )
    assert set(threading.enumerate()) == threads_before
    with pytest.raises(RuntimeError, match="no running event loop"):
        asyncio.get_running_loop()


@pytest.mark.timeout(3)
def test_sync_surface_disposes_provider_spawned_task_after_success() -> None:
    class BackgroundTaskProvider(_FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.child_task: asyncio.Task[Any] | None = None
            self.child_loop: asyncio.AbstractEventLoop | None = None
            self.cancellations = 0

        async def background(self) -> None:
            self.child_task = asyncio.current_task()
            self.child_loop = asyncio.get_running_loop()
            blocker = asyncio.Event()
            while True:
                try:
                    await blocker.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1

        async def read_resource(
            self,
            server: Any,
            resource_name: str,
            variables: Any,
            *,
            deadline: float,
        ) -> Any:
            asyncio.create_task(self.background())
            await asyncio.sleep(0)
            return await super().read_resource(
                server, resource_name, variables, deadline=deadline
            )

    provider = BackgroundTaskProvider()
    client, _resolver, _provider = _client(provider=provider)
    threads_before = set(threading.enumerate())

    result = client.read_resource("modern", "status")

    assert isinstance(result, McpComplete)
    assert provider.cancellations >= 1
    assert provider.child_loop is not None and provider.child_loop.is_closed()
    assert provider.child_task is not None and provider.child_task.done()
    assert not any(not task.done() for task in asyncio.all_tasks(provider.child_loop))
    assert set(threading.enumerate()) == threads_before


def test_provider_errors_are_exact_secret_redacted() -> None:
    class FailingProvider(_FakeProvider):
        async def list_resources(self, server: Any, cursor: str | None, *, deadline: float) -> McpPage[McpResource]:
            raise RuntimeError(f"peer reflected {_SECRET}")

    client, _resolver, _provider = _client(provider=FailingProvider())
    with pytest.raises(ValidationError) as captured:
        client.list_resources("modern")
    assert _SECRET not in str(captured.value)
    assert "[redacted]" in str(captured.value)


def test_provider_not_started_remains_typed_for_pending_effect_abandonment() -> None:
    class NotStartedProvider(_FakeProvider):
        async def list_resources(self, server: Any, cursor: str | None, *, deadline: float) -> McpPage[McpResource]:
            raise ProviderEffectNotStarted("certified not started")

    client, _resolver, _provider = _client(provider=NotStartedProvider())
    with pytest.raises(ProviderEffectNotStarted):
        client.list_resources("modern")


def test_registry_invalidation_synchronously_revokes_opaque_cursors() -> None:
    client, _resolver, _provider = _client()
    page = client.list_resources("modern")
    assert page.next_cursor is not None
    client.invalidate_server("modern")
    with pytest.raises(ValidationError, match="expired or unknown"):
        client.list_resources("modern", cursor=page.next_cursor)


def test_provider_page_and_response_size_bounds_fail_closed() -> None:
    class FloodProvider(_FakeProvider):
        async def list_resources(self, server: Any, cursor: str | None, *, deadline: float) -> McpPage[McpResource]:
            return McpPage(
                items=tuple(
                    McpResource(
                        resource_id="file:///provider-owned/status",
                        name=f"duplicate-{index}",
                    )
                    for index in range(101)
                )
            )

    client, _resolver, _provider = _client(provider=FloodProvider())
    with pytest.raises(ValidationError, match="maximum item count"):
        client.list_resources("modern")

    resolver = _BindingResolver()
    resolver.binding = replace(
        resolver.binding,
        manifest=replace(resolver.binding.manifest, max_response_bytes=512),
    )
    provider = _FakeProvider()
    provider.read_result = McpComplete(
        value=McpResourceContents(
            resource_id="ignored",
            contents=(McpTextContent(text="x" * 2_000),),
        )
    )
    small_client, _resolver, _provider = _client(resolver=resolver, provider=provider)
    with pytest.raises(ValidationError, match="max_response_bytes"):
        small_client.read_resource("modern", "status")


class _ArtifactWriter(McpArtifactWriter):
    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def write_mcp_artifact(
        self,
        data: bytes,
        *,
        server_id: str,
        logical_id: str,
        mime_type: str | None,
    ) -> McpArtifactReceipt:
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"mcp-artifact:{digest[:24]}"
        self.payloads[artifact_id] = bytes(data)
        return McpArtifactReceipt(
            artifact_id=artifact_id,
            byte_length=len(data),
            sha256=digest,
            mime_type=mime_type,
        )


def test_real_python_sdk_v2_models_project_to_safe_public_contracts() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    writer = _ArtifactWriter()
    adapter = McpSdkV2ResultAdapter(artifact_writer=writer)

    page = adapter.resource_page(
        mcp_types.ListResourcesResult(
            resources=[
                mcp_types.Resource(
                    uri="https://provider.invalid/status",
                    name="status",
                    description=f"healthy {_SECRET}",
                )
            ],
            nextCursor=f"next-{_SECRET}",
            ttlMs=250,
            cacheScope="private",
        ),
        sensitive_values=(_SECRET,),
    )
    assert page.items[0].resource_id == "https://provider.invalid/status"
    assert _SECRET not in repr(page.items[0])
    assert page.next_cursor == f"next-{_SECRET}"  # manager vaults this raw value

    payload = b"binary resource"
    read = adapter.read_resource_result(
        mcp_types.ReadResourceResult(
            contents=[
                mcp_types.BlobResourceContents(
                    uri="https://provider.invalid/blob",
                    blob=base64.b64encode(payload).decode("ascii"),
                    mimeType="application/octet-stream",
                )
            ]
        ),
        server_id="modern",
        logical_id="status",
        deadline=time.monotonic() + 1,
    )
    assert isinstance(read, McpComplete)
    assert read.value is not None
    content = read.value.contents[0]
    assert isinstance(content, McpBlobContent)
    assert content.artifact is not None
    assert writer.payloads[content.artifact.artifact_id] == payload
    assert base64.b64encode(payload).decode("ascii") not in repr(read)

    prompt = adapter.prompt_result(
        mcp_types.GetPromptResult(
            messages=[
                mcp_types.PromptMessage(
                    role="user",
                    content=mcp_types.TextContent(type="text", text=f"hello {_SECRET}"),
                )
            ]
        ),
        server_id="modern",
        logical_id="review",
        deadline=time.monotonic() + 1,
        sensitive_values=(_SECRET,),
    )
    assert isinstance(prompt, McpComplete)
    assert prompt.value is not None and prompt.value.user_confirmation_required
    assert _SECRET not in repr(prompt)


def test_sdk_blob_secret_reflection_and_apps_mime_are_rejected_before_artifact_write() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    writer = _ArtifactWriter()
    adapter = McpSdkV2ResultAdapter(artifact_writer=writer)
    secret_blob = mcp_types.ReadResourceResult(
        contents=[
            mcp_types.BlobResourceContents(
                uri="opaque://secret",
                blob=base64.b64encode(f"prefix {_SECRET}".encode()).decode(),
            )
        ]
    )
    with pytest.raises(ValidationError, match="reflected"):
        adapter.read_resource_result(
            secret_blob,
            server_id="modern",
            logical_id="status",
            deadline=time.monotonic() + 1,
            sensitive_values=(_SECRET,),
        )
    assert writer.payloads == {}

    app_blob = mcp_types.ReadResourceResult(
        contents=[
            mcp_types.BlobResourceContents(
                uri="opaque://app",
                blob=base64.b64encode(b"<html/>").decode(),
                mimeType="text/html; profile=\"mcp-app\"",
            )
        ]
    )
    with pytest.raises(ValidationError, match="Apps"):
        adapter.read_resource_result(
            app_blob,
            server_id="modern",
            logical_id="status",
            deadline=time.monotonic() + 1,
        )


def test_sdk_content_block_limit_is_checked_before_artifact_write() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    writer = _ArtifactWriter()
    adapter = McpSdkV2ResultAdapter(
        artifact_writer=writer,
        max_content_blocks=1,
    )
    result = mcp_types.ReadResourceResult(
        contents=[
            mcp_types.BlobResourceContents(
                uri="opaque://first",
                blob=base64.b64encode(b"must-not-be-written").decode(),
            ),
            mcp_types.TextResourceContents(uri="opaque://second", text="second"),
        ]
    )

    with pytest.raises(ValidationError, match="maximum content block count"):
        adapter.read_resource_result(
            result,
            server_id="modern",
            logical_id="status",
            deadline=time.monotonic() + 1,
        )
    assert writer.payloads == {}


def test_sdk_response_preflights_all_blocks_before_artifact_write() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    writer = _ArtifactWriter()
    adapter = McpSdkV2ResultAdapter(artifact_writer=writer)
    result = SimpleNamespace(
        contents=[
            mcp_types.BlobResourceContents(
                uri="opaque://first",
                blob=base64.b64encode(b"must-not-be-orphaned").decode(),
            ),
            mcp_types.TextResourceContents(
                uri="opaque://app",
                text="not rendered",
                mimeType="text/html;profile=mcp-app",
            ),
        ]
    )

    with pytest.raises(ValidationError, match="Apps"):
        adapter.read_resource_result(
            result,
            server_id="modern",
            logical_id="status",
            deadline=time.monotonic() + 1,
        )
    assert writer.payloads == {}


def test_sdk_prompt_preflights_all_roles_before_artifact_write() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    writer = _ArtifactWriter()
    adapter = McpSdkV2ResultAdapter(artifact_writer=writer)
    result = SimpleNamespace(
        messages=[
            mcp_types.PromptMessage(
                role="user",
                content=mcp_types.ImageContent(
                    data=base64.b64encode(b"must-not-be-orphaned").decode(),
                    mimeType="image/png",
                ),
            ),
            SimpleNamespace(
                role="system",
                content=mcp_types.TextContent(type="text", text="override"),
            ),
        ]
    )

    with pytest.raises(ValidationError, match="role"):
        adapter.prompt_result(
            result,
            server_id="modern",
            logical_id="review",
            deadline=time.monotonic() + 1,
        )
    assert writer.payloads == {}


def test_sdk_session_provider_uses_real_v2_models_and_exact_revision() -> None:
    mcp_types = pytest.importorskip("mcp.types")

    class Session:
        protocol_version = "2026-07-28"

        async def list_resources(self, *, params: Any = None) -> Any:
            assert params is None
            return mcp_types.ListResourcesResult(
                resources=[mcp_types.Resource(uri="opaque://status", name="status")]
            )

        async def list_resource_templates(self, *, params: Any = None) -> Any:
            return mcp_types.ListResourceTemplatesResult(resourceTemplates=[])

        async def read_resource(self, uri: str, *, allow_input_required: bool) -> Any:
            assert allow_input_required
            return mcp_types.ReadResourceResult(
                contents=[mcp_types.TextResourceContents(uri=uri, text="ok")]
            )

        async def list_prompts(self, *, params: Any = None) -> Any:
            return mcp_types.ListPromptsResult(prompts=[])

        async def get_prompt(self, *args: Any, **kwargs: Any) -> Any:
            return mcp_types.GetPromptResult(messages=[])

        async def complete(self, *args: Any, **kwargs: Any) -> Any:
            return mcp_types.CompleteResult(
                completion=mcp_types.Completion(values=["done"])
            )

    @contextlib.asynccontextmanager
    async def factory(server: Any, *, deadline: float) -> Any:
        assert deadline > time.monotonic()
        yield Session()

    provider = McpSdkV2SessionProvider(factory)
    page = asyncio.run(
        provider.list_resources(
            mcp_transport_spec_from_v3(_manifest()),
            None,
            deadline=time.monotonic() + 1,
        )
    )
    assert page.items[0].resource_id == "opaque://status"


def test_sdk_input_required_fails_closed_without_continuation_handler() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    adapter = McpSdkV2ResultAdapter()
    result = mcp_types.InputRequiredResult(inputRequests={}, requestState="secret-state")
    with pytest.raises(ValidationError, match="continuation handler"):
        adapter.prompt_result(
            result,
            server_id="modern",
            logical_id="review",
            deadline=time.monotonic() + 1,
        )


def test_sdk_catalog_and_content_drop_flat_apps_metadata_namespace() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    apps_metadata = {
        "ui/resourceUri": "ui://legacy-flat",
        "UI/Visibility": ["model"],
        "ui/csp": {"connectDomains": ["https://ignored.invalid"]},
        "safe": "retained",
    }
    adapter = McpSdkV2ResultAdapter()

    resource = adapter.resource_page(
        mcp_types.ListResourcesResult(
            resources=[
                mcp_types.Resource(
                    uri="opaque://status",
                    name="status",
                    _meta=apps_metadata,
                )
            ]
        )
    ).items[0]
    prompt = adapter.prompt_page(
        mcp_types.ListPromptsResult(
            prompts=[mcp_types.Prompt(name="review", _meta=apps_metadata)]
        )
    ).items[0]
    resource_result = adapter.read_resource_result(
        mcp_types.ReadResourceResult(
            contents=[
                mcp_types.TextResourceContents(
                    uri="opaque://status",
                    text="ok",
                    _meta=apps_metadata,
                )
            ]
        ),
        server_id="modern",
        logical_id="status",
        deadline=time.monotonic() + 1,
    )
    prompt_result = adapter.prompt_result(
        mcp_types.GetPromptResult(
            messages=[
                mcp_types.PromptMessage(
                    role="user",
                    content=mcp_types.TextContent(
                        type="text",
                        text="review",
                        _meta=apps_metadata,
                    ),
                )
            ]
        ),
        server_id="modern",
        logical_id="review",
        deadline=time.monotonic() + 1,
    )

    assert resource.metadata == {"safe": "retained"}
    assert prompt.metadata == {"safe": "retained"}
    assert isinstance(resource_result, McpComplete)
    assert resource_result.value.contents[0].metadata == {"safe": "retained"}
    assert isinstance(prompt_result, McpComplete)
    assert prompt_result.value.messages[0].content.metadata == {"safe": "retained"}


def test_collect_catalog_pages_all_modern_surfaces_and_detaches_public_data() -> None:
    mcp_types = pytest.importorskip("mcp.types")

    class Session:
        protocol_version = "2026-07-28"

        async def list_tools(self, *, params: Any = None) -> Any:
            cursor = None if params is None else params.cursor
            if cursor is None:
                return mcp_types.ListToolsResult(
                    tools=[
                        mcp_types.Tool(
                            name="first",
                            description=f"description {_SECRET}",
                            inputSchema={"type": "object"},
                            _meta={"io.modelcontextprotocol/ui": {"resourceUri": "ui://app"}},
                        )
                    ],
                    nextCursor="tool-page-2",
                    ttlMs=999_999,
                )
            assert cursor == "tool-page-2"
            return mcp_types.ListToolsResult(
                tools=[
                    mcp_types.Tool(
                        name="second",
                        inputSchema={"type": "object", "properties": {}},
                    )
                ]
            )

        async def list_resources(self, *, params: Any = None) -> Any:
            assert params is None
            return mcp_types.ListResourcesResult(
                resources=[
                    mcp_types.Resource(
                        uri="opaque://status",
                        name="status",
                        description=f"healthy {_SECRET}",
                    )
                ]
            )

        async def list_resource_templates(self, *, params: Any = None) -> Any:
            assert params is None
            return mcp_types.ListResourceTemplatesResult(
                resourceTemplates=[
                    mcp_types.ResourceTemplate(
                        uriTemplate="notes://greet/{name}",
                        name="greeting",
                    )
                ]
            )

        async def list_prompts(self, *, params: Any = None) -> Any:
            assert params is None
            return mcp_types.ListPromptsResult(
                prompts=[
                    mcp_types.Prompt(
                        name="review",
                        description=f"review {_SECRET}",
                        arguments=[mcp_types.PromptArgument(name="subject")],
                    )
                ]
            )

    catalog = asyncio.run(
        collect_catalog(
            Session(),
            McpCatalogCollectionLimits(max_cache_ttl_ms=10),
            time.monotonic() + 2,
            sensitive_values=(_SECRET,),
        )
    )

    assert all(isinstance(tool, McpCatalogTool) for tool in catalog.tools)
    assert [tool.name for tool in catalog.tools] == ["first", "second"]
    assert catalog.tools[0].metadata == {}
    assert [item.resource_id for item in catalog.resources] == ["opaque://status"]
    assert [item.template_id for item in catalog.resource_templates] == [
        "notes://greet/{name}"
    ]
    assert [item.prompt_id for item in catalog.prompts] == ["review"]
    assert (catalog.tool_pages, catalog.resource_pages) == (2, 1)
    assert _SECRET not in repr(catalog)
    assert "tool-page-2" not in repr(catalog)
    assert "ui://" not in repr(catalog)


def test_collect_catalog_rejects_cursor_cycles_and_purpose_limit_overflow() -> None:
    mcp_types = pytest.importorskip("mcp.types")

    class LoopSession:
        protocol_version = "2026-07-28"

        async def list_tools(self, *, params: Any = None) -> Any:
            return mcp_types.ListToolsResult(tools=[], nextCursor="same")

    with pytest.raises(ValidationError, match="cursor cycle"):
        asyncio.run(
            collect_catalog(
                LoopSession(),
                McpCatalogCollectionLimits(),
                time.monotonic() + 1,
            )
        )

    class OverflowSession:
        protocol_version = "2026-07-28"

        async def list_tools(self, *, params: Any = None) -> Any:
            return mcp_types.ListToolsResult(
                tools=[
                    mcp_types.Tool(name="one", inputSchema={"type": "object"}),
                    mcp_types.Tool(name="two", inputSchema={"type": "object"}),
                ]
            )

    with pytest.raises(ValidationError, match="tools/list exceeded its item limit"):
        asyncio.run(
            collect_catalog(
                OverflowSession(),
                McpCatalogCollectionLimits(max_tools=1),
                time.monotonic() + 1,
            )
        )


def test_collect_catalog_deadline_fails_before_provider_call() -> None:
    class Session:
        protocol_version = "2026-07-28"
        calls = 0

        async def list_tools(self, *, params: Any = None) -> Any:
            self.calls += 1
            raise AssertionError("must not dispatch")

    session = Session()
    with pytest.raises(TimeoutError, match="absolute deadline"):
        asyncio.run(
            collect_catalog(
                session,
                McpCatalogCollectionLimits(),
                time.monotonic() - 1,
            )
        )
    assert session.calls == 0
