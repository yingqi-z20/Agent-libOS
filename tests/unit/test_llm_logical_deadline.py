from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG, LLMDefaults, LLMProfile
from agent_libos.llm import client as client_module
from agent_libos.llm.client import LLMClient, LLMError, LLMTransientError
from agent_libos.llm.profiles import LLMProfileRegistry
from agent_libos.llm.provider_service import LLMProviderService
from agent_libos.llm.provider_trace import provider_trace_from_error
from agent_libos.models.exceptions import ValidationError


class _SDKError(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status_code = status


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        id="chat_complete",
        model="gpt-test",
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="done", tool_calls=[]),
        )],
    )


def _client(create: Any, **kwargs: Any) -> LLMClient:
    client = LLMClient(
        model="gpt-test", api_key="test-key", api_mode="chat",
        inherit_ambient_openai_sdk_config=False, **kwargs,
    )
    client._async_client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    return client


async def _complete(client: LLMClient, action: bool = False) -> Any:
    messages = [{"role": "user", "content": "finish"}]
    if action:
        return await client.acomplete_action(messages, [])
    return await client.acomplete_with_metadata(messages, json_mode=False)


@pytest.mark.parametrize("action", [False, True])
def test_logical_deadline_cancels_provider_and_retains_terminal_trace(action: bool) -> None:
    calls = 0
    cancelled = False

    async def create(**_kwargs: Any) -> Any:
        nonlocal calls, cancelled
        calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    client = _client(create, logical_call_timeout_s=0.04, max_retries=5, timeout=10.0)
    started = time.monotonic()
    with pytest.raises(LLMTransientError) as raised:
        asyncio.run(_complete(client, action))

    assert time.monotonic() - started < 1.0
    assert cancelled and calls == 1
    assert client.timeout == 10.0
    assert isinstance(raised.value.__cause__, TimeoutError)
    trace = provider_trace_from_error(raised.value)
    assert trace is not None and trace["selected_attempt"] is None
    attempt, = trace["attempts"]
    assert attempt["status"] == "error"
    assert attempt["error"]["error_type"] == "_LogicalCallTimeoutError"
    assert attempt["duration_ms"] > 0
    assert attempt["completed_at"] != attempt["started_at"]
    assert attempt["usage"] == {}  # Unknown usage; never a fabricated zero total.


def test_one_deadline_covers_transport_and_compatibility_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    deadlines: list[float | None] = []

    async def create(**_kwargs: Any) -> Any:
        timeout = client_module._ACTIVE_LOGICAL_CALL_TIMEOUT.get()
        assert timeout is not None
        deadlines.append(timeout.when())
        await asyncio.sleep(0.01)
        if len(deadlines) == 1:
            raise _SDKError("rate limited", 429)
        if len(deadlines) == 2:
            raise _SDKError("unknown parameter temperature", 400)
        await asyncio.Event().wait()

    monkeypatch.setattr(client_module, "_is_openai_sdk_error", lambda exc: isinstance(exc, _SDKError))
    monkeypatch.setattr(client_module, "_openai_retry_delay", lambda *_args: 0.0)
    client = _client(create, logical_call_timeout_s=0.12, max_retries=4)

    with pytest.raises(LLMTransientError) as raised:
        asyncio.run(_complete(client))

    assert len(deadlines) == 3 and len(set(deadlines)) == 1
    trace = provider_trace_from_error(raised.value)
    assert trace is not None
    assert [attempt["kind"] for attempt in trace["attempts"]] == [
        "initial", "transport_retry", "compatibility_retry",
    ]
    assert all(attempt["error"] is not None for attempt in trace["attempts"])
    assert trace["selected_attempt"] is None


@pytest.mark.parametrize("action", [False, True])
def test_responses_to_chat_fallback_keeps_the_same_deadline(
    monkeypatch: pytest.MonkeyPatch, action: bool,
) -> None:
    deadlines: list[float | None] = []

    async def create(**_kwargs: Any) -> Any:
        timeout = client_module._ACTIVE_LOGICAL_CALL_TIMEOUT.get()
        assert timeout is not None
        deadlines.append(timeout.when())
        if len(deadlines) == 1:
            await asyncio.sleep(0.01)
            raise _SDKError("Responses endpoint unavailable", 404)
        await asyncio.Event().wait()

    monkeypatch.setattr(client_module, "_is_openai_sdk_error", lambda exc: isinstance(exc, _SDKError))
    client = _client(create, logical_call_timeout_s=0.08)
    client.api_mode = "auto"

    with pytest.raises(LLMTransientError) as raised:
        asyncio.run(_complete(client, action))

    assert len(deadlines) == 2 and len(set(deadlines)) == 1
    trace = provider_trace_from_error(raised.value)
    assert trace is not None
    assert [attempt["kind"] for attempt in trace["attempts"]] == ["initial", "responses_to_chat"]
    assert [attempt["api"] for attempt in trace["attempts"]] == ["responses", "chat"]


def test_deadline_cancels_retry_backoff_without_another_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def create(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise _SDKError("retry later", 429)

    monkeypatch.setattr(client_module, "_is_openai_sdk_error", lambda exc: isinstance(exc, _SDKError))
    monkeypatch.setattr(client_module, "_openai_retry_delay", lambda *_args: 5.0)
    client = _client(create, logical_call_timeout_s=0.04)
    with pytest.raises(LLMTransientError) as raised:
        asyncio.run(_complete(client))
    assert calls == 1
    trace = provider_trace_from_error(raised.value)
    assert trace is not None and len(trace["attempts"]) == 1
    assert trace["attempts"][0]["error"]["status_code"] == 429


@pytest.mark.parametrize("enabled", [False, True])
def test_external_cancellation_propagates_with_finished_trace(enabled: bool) -> None:
    calls = 0

    async def exercise() -> None:
        started = asyncio.Event()

        async def create(**_kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            started.set()
            await asyncio.Event().wait()

        client = _client(create, logical_call_timeout_s=5.0 if enabled else None)
        # The runtime's provider service must preserve cancellation and its
        # trace rather than converting it to an ordinary retryable exception.
        task = asyncio.create_task(LLMProviderService().complete_action(
            client, [{"role": "user", "content": "finish"}], [],
            temperature=0.2, max_tokens=64, parallel_tool_calls=False,
        ))
        await started.wait()
        await asyncio.sleep(0.01)
        task.cancel("host cancelled")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("host cancelled",)
        trace = provider_trace_from_error(raised.value)
        assert trace is not None and trace["selected_attempt"] is None
        attempt, = trace["attempts"]
        assert attempt["error"]["error_type"] == "CancelledError"
        assert attempt["duration_ms"] > 0 and attempt["status"] == "error"
        assert client_module._ACTIVE_LOGICAL_CALL_TIMEOUT.get() is None

    asyncio.run(exercise())
    assert calls == 1


def test_sync_entrypoint_deadline_runs_in_provider_event_loop() -> None:
    provider_thread: int | None = None
    cancelled_thread: int | None = None
    host_thread = threading.get_ident()

    async def create(**_kwargs: Any) -> Any:
        nonlocal provider_thread, cancelled_thread
        provider_thread = threading.get_ident()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_thread = threading.get_ident()
            raise

    client = _client(create, logical_call_timeout_s=0.04)

    async def exercise() -> None:
        # Legacy callers can invoke the sync facade from a worker bridge. Its
        # deadline must cancel the SDK coroutine in that worker's event loop.
        with pytest.raises(LLMTransientError):
            await asyncio.to_thread(
                client.complete_action, [{"role": "user", "content": "finish"}], [],
            )

    asyncio.run(exercise())
    assert provider_thread == cancelled_thread and provider_thread != host_thread


def test_disabled_deadline_retains_io_timeout_and_success() -> None:
    async def create(**_kwargs: Any) -> Any:
        assert client_module._ACTIVE_LOGICAL_CALL_TIMEOUT.get() is None
        await asyncio.sleep(0.03)
        return _response()

    client = _client(create, timeout=0.01)
    result = asyncio.run(_complete(client))
    assert client.logical_call_timeout_s is None
    assert client._client_kwargs()["timeout"] == 0.01
    assert "logical_call_timeout_s" not in client._client_kwargs()
    assert result.content == "done" and result.provider_trace["selected_attempt"] == 1


def test_late_response_retains_received_usage_without_becoming_selected() -> None:
    async def create(**_kwargs: Any) -> Any:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return _response()

    client = _client(create, logical_call_timeout_s=0.03)
    with pytest.raises(LLMTransientError) as raised:
        asyncio.run(_complete(client))
    trace = provider_trace_from_error(raised.value)
    assert trace is not None and trace["selected_attempt"] is None
    attempt, = trace["attempts"]
    assert attempt["status"] == "ok"  # The remote response actually arrived.
    assert attempt["usage"]["total_tokens"] == 18


def test_transport_translated_cancellation_cannot_start_fresh_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def create(**_kwargs: Any) -> Any:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise _SDKError("transport closed after cancellation", 429)

    def retry_delay(*_args: Any) -> float:
        raise AssertionError("deadline exhaustion must not schedule another retry")

    monkeypatch.setattr(client_module, "_is_openai_sdk_error", lambda exc: isinstance(exc, _SDKError))
    monkeypatch.setattr(client_module, "_openai_retry_delay", retry_delay)
    client = _client(create, logical_call_timeout_s=0.03)
    with pytest.raises(LLMTransientError) as raised:
        asyncio.run(_complete(client))
    trace = provider_trace_from_error(raised.value)
    assert trace is not None and len(trace["attempts"]) == 1
    assert trace["attempts"][0]["error"]["status_code"] == 429


def test_concurrent_calls_have_independent_deadlines() -> None:
    async def exercise() -> None:
        deadlines: list[float | None] = []
        first_started = asyncio.Event()

        async def create(**_kwargs: Any) -> Any:
            timeout = client_module._ACTIVE_LOGICAL_CALL_TIMEOUT.get()
            assert timeout is not None
            deadlines.append(timeout.when())
            first_started.set()
            await asyncio.Event().wait()

        client = _client(create, logical_call_timeout_s=0.12)
        first = asyncio.create_task(_complete(client))
        await first_started.wait()
        await asyncio.sleep(0.04)
        second = asyncio.create_task(_complete(client))
        with pytest.raises(LLMTransientError):
            await first
        assert not second.done()
        with pytest.raises(LLMTransientError):
            await second
        assert len(deadlines) == 2 and deadlines[1] > deadlines[0]
        assert client_module._ACTIVE_LOGICAL_CALL_TIMEOUT.get() is None

    asyncio.run(exercise())


def test_unrelated_timeout_is_not_reclassified_as_logical_deadline() -> None:
    unrelated = TimeoutError("local helper timeout")

    async def create(**_kwargs: Any) -> Any:
        raise unrelated

    client = _client(create, logical_call_timeout_s=5.0)
    with pytest.raises(TimeoutError) as raised:
        asyncio.run(_complete(client))
    assert raised.value is unrelated


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_owned_client_cleanup_uses_remaining_deadline_and_preserves_timeout(
    monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool,
) -> None:
    cleanup_started = False
    cleanup_cancelled = False

    async def create(**_kwargs: Any) -> Any:
        await asyncio.Event().wait()

    async def close() -> None:
        nonlocal cleanup_started, cleanup_cancelled
        cleanup_started = True
        if cleanup_fails:
            raise RuntimeError("cleanup failed")
        try:
            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            cleanup_cancelled = True
            raise

    client = _client(create, logical_call_timeout_s=0.04)
    owned = client._async_client
    owned.aclose = close
    client._async_client = None
    monkeypatch.setattr(client, "_async_client_or_raise", lambda: owned)
    started = time.monotonic()
    with pytest.raises(LLMTransientError) as raised:
        asyncio.run(_complete(client))
    assert time.monotonic() - started < 0.5
    assert cleanup_started and (cleanup_fails or cleanup_cancelled)
    trace = provider_trace_from_error(raised.value)
    assert trace is not None
    assert trace["attempts"][0]["error"]["error_type"] == "_LogicalCallTimeoutError"


@pytest.mark.parametrize("enabled", [False, True])
def test_owned_client_cleanup_failure_preserves_external_cancellation(
    monkeypatch: pytest.MonkeyPatch, enabled: bool,
) -> None:
    async def exercise() -> None:
        started = asyncio.Event()

        async def create(**_kwargs: Any) -> Any:
            started.set()
            await asyncio.Event().wait()

        async def close() -> None:
            raise RuntimeError("cleanup failed")

        client = _client(create, logical_call_timeout_s=5.0 if enabled else None)
        owned = client._async_client
        owned.aclose = close
        client._async_client = None
        monkeypatch.setattr(client, "_async_client_or_raise", lambda: owned)
        task = asyncio.create_task(_complete(client))
        await started.wait()
        task.cancel("host cancelled")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("host cancelled",)
        trace = provider_trace_from_error(raised.value)
        assert trace is not None
        assert trace["attempts"][0]["error"]["error_type"] == "CancelledError"

    asyncio.run(exercise())


@pytest.mark.parametrize("external_cancel", [False, True])
def test_owned_cleanup_deadline_preserves_cancellation_or_received_usage(
    monkeypatch: pytest.MonkeyPatch, external_cancel: bool,
) -> None:
    async def exercise() -> None:
        started = asyncio.Event()

        async def create(**_kwargs: Any) -> Any:
            started.set()
            if external_cancel:
                await asyncio.Event().wait()
            return _response()

        async def close() -> None:
            await asyncio.Event().wait()

        client = _client(create, logical_call_timeout_s=0.04)
        owned = client._async_client
        owned.aclose = close
        client._async_client = None
        monkeypatch.setattr(client, "_async_client_or_raise", lambda: owned)
        task = asyncio.create_task(_complete(client))
        await started.wait()
        if external_cancel:
            task.cancel("host cancelled")
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert raised.value.args == ("host cancelled",)
            assert task.cancelling() == 1
        else:
            with pytest.raises(LLMTransientError) as raised:
                await task
            assert task.cancelling() == 0
            trace = provider_trace_from_error(raised.value)
            assert trace is not None and trace["selected_attempt"] is None
            attempt, = trace["attempts"]
            assert attempt["status"] == "ok"
            assert attempt["usage"]["total_tokens"] == 18

    asyncio.run(exercise())


def test_new_deadline_field_preserves_existing_positional_constructor_order() -> None:
    client = LLMClient(None, "gpt-test", "test-key", "OPENAI_API_KEY", 9.0, 0, "chat")
    profile = LLMProfile("openai_compatible", None, "gpt-test", "OPENAI_API_KEY", "chat", 9.0, 0)
    defaults = LLMDefaults(
        "default", "gpt-test", "medium", "all_turns", "30m", {"default": profile},
        0.2, 16_384, 245_760, 262_144, 262_144, 9.0, 0, "chat",
    )
    assert client.timeout == profile.timeout_s == defaults.timeout_s == 9.0
    assert client.max_retries == profile.max_retries == defaults.max_retries == 0
    assert client.api_mode == profile.api_mode == defaults.api_mode == "chat"
    assert client.logical_call_timeout_s is profile.logical_call_timeout_s is defaults.logical_call_timeout_s is None


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_logical_timeout_is_rejected(value: Any) -> None:
    with pytest.raises(LLMError, match="finite positive"):
        _client(None, logical_call_timeout_s=value)
    with pytest.raises(ValueError):
        AgentLibOSConfig(llm=LLMDefaults(logical_call_timeout_s=value))
    with pytest.raises(ValueError):
        AgentLibOSConfig(llm=LLMDefaults(profiles={"default": LLMProfile(logical_call_timeout_s=value)}))


def test_dynamic_profile_rejects_nonpositive_logical_timeout() -> None:
    registry = LLMProfileRegistry(SimpleNamespace())
    with pytest.raises(ValidationError, match="logical_call_timeout_s"):
        registry.register_profile("test", LLMProfile(logical_call_timeout_s=-1.0))


def test_environment_and_profile_precedence_preserve_io_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in client_module.os.environ:
        if key.startswith("OPENAI_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_TIMEOUT", "7")
    defaults = LLMDefaults(logical_call_timeout_s=3.0, profiles={
        "default": LLMProfile(), "isolated": LLMProfile(model="gpt-test"),
    })
    assert LLMClient.from_env(config=defaults).logical_call_timeout_s == 3.0
    monkeypatch.setenv("OPENAI_LOGICAL_CALL_TIMEOUT", "2.5")
    env_client = LLMClient.from_env(config=defaults)
    assert env_client.logical_call_timeout_s == 2.5 and env_client.timeout == 7.0
    registry = LLMProfileRegistry(SimpleNamespace(), config=AgentLibOSConfig(llm=defaults))
    original = registry.resolve("default")
    assert original.client.logical_call_timeout_s == 2.5
    assert registry.resolve("isolated").client.logical_call_timeout_s == 3.0
    monkeypatch.setenv("OPENAI_LOGICAL_CALL_TIMEOUT", "1.5")
    changed = registry.resolve("default")
    assert changed.client is not original.client
    assert changed.client.logical_call_timeout_s == 1.5
    assert changed.identity_sha256 == original.identity_sha256
    registry.register_profile("default", LLMProfile(logical_call_timeout_s=1.0))
    explicit = registry.resolve("default")
    assert explicit.client.logical_call_timeout_s == 1.0 and explicit.client.timeout == 7.0
    assert explicit.identity_sha256 == original.identity_sha256
    monkeypatch.setenv("OPENAI_LOGICAL_CALL_TIMEOUT", "nan")
    with pytest.raises(LLMError, match="logical_call_timeout_s"):
        LLMClient.from_env(config=defaults)


def test_default_deadline_is_disabled_for_existing_configuration() -> None:
    assert DEFAULT_CONFIG.llm.logical_call_timeout_s is None
    assert LLMProfile().logical_call_timeout_s is None
    assert replace(DEFAULT_CONFIG.llm, timeout_s=0.01).logical_call_timeout_s is None
