from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.mcp import (
    McpOAuthError,
    McpOAuthProfile,
    McpOAuthRegistrationMode,
    McpOAuthTokenEndpointAuthMethod,
    SystemKeyringMcpCredentialBroker,
)
from agent_libos.substrate import LocalResourceProviderSubstrate


class _ResourceProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    async def list_resources(self, server: Any, cursor: Any, *, deadline: float) -> Any:
        raise AssertionError("not dispatched")

    async def list_resource_templates(
        self, server: Any, cursor: Any, *, deadline: float
    ) -> Any:
        raise AssertionError("not dispatched")

    async def read_resource(
        self,
        server: Any,
        resource_name: str,
        variables: Any,
        *,
        deadline: float,
    ) -> Any:
        raise AssertionError("not dispatched")


class _PromptProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    async def list_prompts(self, server: Any, cursor: Any, *, deadline: float) -> Any:
        raise AssertionError("not dispatched")

    async def get_prompt(
        self, server: Any, prompt_name: str, arguments: Any, *, deadline: float
    ) -> Any:
        raise AssertionError("not dispatched")

    async def complete(
        self,
        server: Any,
        reference: Any,
        argument: Any,
        context: Any,
        *,
        deadline: float,
    ) -> Any:
        raise AssertionError("not dispatched")


class _ArtifactWriter:
    def write_mcp_artifact(self, data: bytes, **kwargs: Any) -> Any:
        raise AssertionError("not dispatched")


class _OAuthTransport:
    def request(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not dispatched")


class _UnavailableBroker:
    def available(self) -> bool:
        return False

    def reserve_secret_ref(self, namespace: str) -> str:
        raise AssertionError("unavailable broker must not reserve a secret")

    def put_secret_at(
        self,
        secret_ref: str,
        namespace: str,
        value: bytes,
        *,
        expires_at: str | None,
    ) -> None:
        raise AssertionError("unavailable broker must not be written")

    def put_secret(self, namespace: str, value: bytes, *, expires_at: str | None) -> str:
        raise AssertionError("unavailable broker must not be written")

    def get_secret(self, secret_ref: str) -> bytes:
        raise AssertionError("unavailable broker must not be read")

    def delete_secret(self, secret_ref: str) -> None:
        raise AssertionError("unavailable broker must not be deleted")


def test_builder_preserves_explicit_modern_spi_identity(tmp_path: Any) -> None:
    substrate = LocalResourceProviderSubstrate(tmp_path)
    resource_provider = _ResourceProvider()
    prompt_provider = _PromptProvider()
    artifact_writer = _ArtifactWriter()
    oauth_transport = _OAuthTransport()
    broker = _UnavailableBroker()
    substrate.mcp_resource_provider = resource_provider
    substrate.mcp_prompt_provider = prompt_provider
    substrate.mcp_artifact_writer = artifact_writer
    substrate.mcp_credential_broker = broker
    substrate.mcp_oauth_transport = oauth_transport

    runtime = Runtime.open("local", substrate=substrate)
    try:
        assert runtime._mcp_resource_provider is resource_provider
        assert runtime._mcp_prompt_provider is prompt_provider
        assert runtime._mcp_artifact_writer is artifact_writer
        assert runtime._mcp_credential_broker is broker
        assert runtime._mcp_oauth_transport is oauth_transport
        assert runtime._mcp_oauth_manager._transport is oauth_transport
        assert runtime.mcp._modern_client.resource_provider is resource_provider
        assert runtime.mcp._modern_client.prompt_provider is prompt_provider
        invalidator = runtime._mcp_oauth_manager._connection_invalidator
        assert invalidator.__self__ is runtime._mcp_connection_supervisor
        assert invalidator.__func__ is type(
            runtime._mcp_connection_supervisor
        ).invalidate_server_nowait
    finally:
        runtime.shutdown()


def test_builder_rejects_modern_spi_without_exact_identity(tmp_path: Any) -> None:
    class WrongRevision(_ResourceProvider):
        mcp_protocol_revision = "2025-11-25"

    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate.mcp_resource_provider = WrongRevision()

    with pytest.raises(TypeError, match="exact MCP v3 identity"):
        Runtime.open("local", substrate=substrate)


def test_builder_rejects_sync_or_open_ended_modern_spi(tmp_path: Any) -> None:
    class SyncResource(_ResourceProvider):
        def list_resources(
            self, server: Any, cursor: Any, *, deadline: float
        ) -> Any:
            raise AssertionError("not dispatched")

    class OpenEndedTool:
        mcp_manifest_schema_version = 3
        mcp_protocol_revision = "2026-07-28"

        async def call_tool(
            self,
            manifest: Any,
            tool_id: str,
            arguments: Any,
            *,
            deadline: float,
            **kwargs: Any,
        ) -> Any:
            raise AssertionError("not dispatched")

    sync_substrate = LocalResourceProviderSubstrate(tmp_path)
    sync_substrate.mcp_resource_provider = SyncResource()
    with pytest.raises(TypeError, match="list_resources must be declared with async def"):
        Runtime.open("local", substrate=sync_substrate)

    open_substrate = LocalResourceProviderSubstrate(tmp_path)
    open_substrate.mcp_v3_tool_provider = OpenEndedTool()
    with pytest.raises(TypeError, match="call_tool has an incompatible keyword signature"):
        Runtime.open("local", substrate=open_substrate)


def test_builder_requires_every_respondable_continuation_surface(
    tmp_path: Any,
) -> None:
    class ToolOnlyContinuation:
        mcp_manifest_schema_version = 3
        mcp_protocol_revision = "2026-07-28"

        async def continue_tool(
            self,
            server: Any,
            mcp_name: str,
            arguments: Any,
            input_responses: Any,
            request_state: str | None,
            *,
            deadline: float,
        ) -> Any:
            raise AssertionError("not dispatched")

    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate.mcp_continuation_provider = ToolOnlyContinuation()

    with pytest.raises(TypeError, match="continue_resource must be declared with async def"):
        Runtime.open("local", substrate=substrate)


def test_builder_default_keyring_is_lazy_for_nonauth_runtime(tmp_path: Any) -> None:
    substrate = LocalResourceProviderSubstrate(tmp_path)

    runtime = Runtime.open("local", substrate=substrate)
    try:
        assert isinstance(
            runtime._mcp_credential_broker,
            SystemKeyringMcpCredentialBroker,
        )
    finally:
        runtime.shutdown()


def test_unavailable_broker_fails_closed_only_when_oauth_is_configured(
    tmp_path: Any,
) -> None:
    substrate = LocalResourceProviderSubstrate(tmp_path)
    broker = _UnavailableBroker()
    substrate.mcp_credential_broker = broker
    runtime = Runtime.open("local", substrate=substrate)
    try:
        profile = McpOAuthProfile(
            profile_id="demo-oauth",
            server_id="demo-http",
            resource_uri="https://mcp.example.test/",
            expected_issuer="https://issuer.example.test/",
            redirect_uri="http://127.0.0.1:8765/callback",
            client_id="agent-libos-test",
            registration_mode=McpOAuthRegistrationMode.PREREGISTERED,
            token_endpoint_auth_method=McpOAuthTokenEndpointAuthMethod.NONE,
            allowed_endpoint_origins=("https://issuer.example.test",),
        )
        with pytest.raises(McpOAuthError, match="secure credential backend unavailable"):
            runtime._mcp_oauth_manager.add_profile(profile)
    finally:
        runtime.shutdown()


def test_builder_applies_purpose_specific_modern_limits(tmp_path: Any) -> None:
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            max_content_blocks=1,
            cursor_handle_limit=1,
            subscription_terminal_records=1,
            cache_hint_ttl_cap_ms=1,
            mrtr_max_rounds=2,
            mrtr_max_input_requests=3,
            mrtr_request_state_max_bytes=128,
            continuation_ttl_s=4.0,
        )
    )
    runtime = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(tmp_path),
        config=config,
    )
    try:
        limits = runtime.mcp._modern_client.limits
        assert limits.max_content_blocks == 1
        assert limits.max_cursor_handles == 1
        assert limits.max_cache_ttl_ms == 1
        assert runtime._mcp_subscription_manager._policy.terminal_status_records == 1
        adapter = runtime._mcp_resource_provider.result_adapter
        assert adapter.max_content_blocks == 1
        assert adapter.max_cache_ttl_ms == 1
        continuation = runtime._mcp_continuation_manager
        assert continuation is runtime.mcp._modern_continuations
        assert continuation._max_rounds == 2
        assert continuation._max_input_requests == 3
        assert continuation._request_state_max_bytes == 128
        assert continuation._continuation_ttl.total_seconds() == 4.0
    finally:
        runtime.shutdown()


def test_builder_binds_exact_tasks_wire_manager_and_reopens_with_host_limits(
    tmp_path: Any,
) -> None:
    digest = "a" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
            mrtr_max_input_requests=2,
            remote_task_poll_min_interval_s=1.5,
            remote_task_max_wait_s=9.0,
            remote_task_max_records=7,
        )
    )
    runtime = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(tmp_path),
        config=config,
    )
    try:
        from agent_libos.mcp import McpSdkV3TasksProvider

        assert isinstance(runtime._mcp_tasks_provider, McpSdkV3TasksProvider)
        task_manager = runtime._mcp_remote_task_manager
        assert task_manager is runtime.mcp._modern_remote_tasks
        assert task_manager._max_input_requests == 2
        assert task_manager._poll_min_interval_s == 1.5
        assert task_manager._max_wait.total_seconds() == 9.0
        assert task_manager._max_records == 7
        assert runtime._mcp_tasks_provider.host_tasks_extension_sha256 == digest
        assert runtime._mcp_subscription_manager._task_event_projector is task_manager
        assert (
            runtime._mcp_subscription_provider.task_subscriptions_resolver.__self__
            is task_manager
        )
    finally:
        runtime.shutdown()

    reopened = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(tmp_path),
        config=config,
    )
    try:
        from agent_libos.mcp import McpSdkV3TasksProvider

        assert isinstance(reopened._mcp_tasks_provider, McpSdkV3TasksProvider)
        assert reopened._mcp_remote_task_manager is reopened.mcp._modern_remote_tasks
        assert reopened._mcp_tasks_provider.host_tasks_extension_sha256 == digest
        assert reopened._mcp_subscription_manager is reopened.mcp._modern_subscriptions
        assert (
            reopened._mcp_subscription_manager._task_event_projector
            is reopened._mcp_remote_task_manager
        )
        assert (
            reopened._mcp_subscription_provider.task_subscriptions_resolver.__self__
            is reopened._mcp_remote_task_manager
        )
    finally:
        reopened.shutdown()
