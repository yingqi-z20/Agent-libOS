from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, LLMDefaults, LLMProfile
from agent_libos.llm.user_profiles import UserLLMProfileStore, default_user_llm_profiles_path
from agent_libos.llm.executor import LLMProcessExecutor
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    DataSink,
    ProcessStatus,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.runtime.builder import RuntimeBuilder
from agent_libos.storage import SQLiteStore
from tests.support.fakes import RecordingActionClient

_AMBIENT_ACCOUNT_POLICY_ENV = (
    "OPENAI_ENABLE_THINKING",
    "OPENAI_ORGANIZATION",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
)


def _profile_config() -> AgentLibOSConfig:
    return AgentLibOSConfig(
        llm=LLMDefaults(
            default_profile_id="default",
            profiles={
                "default": LLMProfile(model="default-model"),
                "fast": LLMProfile(model="fast-model", temperature=0.0, max_tokens=128),
                "slow": LLMProfile(model="slow-model", temperature=0.4, max_tokens=256),
                "image-default": LLMProfile(model="image-model"),
                "override": LLMProfile(model="override-model"),
                "parallel": LLMProfile(model="parallel-model", parallel_tool_calls=True),
                "auto-wait": LLMProfile(model="auto-wait-model", auto_wait_on_empty_tool_calls=True),
                "json-fallback": LLMProfile(model="fallback-model", fallback_json_actions=True),
            },
        )
    )


class TestLLMProfiles:
    @pytest.mark.parametrize(
        (
            "env_name",
            "equivalent_env_value",
            "env_value",
            "client_attribute",
            "expected",
            "isolated_expected",
        ),
        [
            ("OPENAI_STORE", "0", "1", "store", True, False),
            (
                "OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID",
                "false",
                "true",
                "responses_previous_response_id",
                True,
                False,
            ),
            (
                "OPENAI_PROMPT_CACHE_RETENTION",
                "",
                "24h",
                "prompt_cache_retention",
                "24h",
                None,
            ),
            (
                "OPENAI_FALLBACK_JSON_ACTIONS",
                "false",
                "true",
                "fallback_json_actions",
                True,
                False,
            ),
        ],
    )
    def test_default_profile_identity_tracks_effective_legacy_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
        equivalent_env_value: str,
        env_value: str,
        client_attribute: str,
        expected: object,
        isolated_expected: object,
    ) -> None:
        monkeypatch.delenv(env_name, raising=False)
        config = AgentLibOSConfig(
            llm=LLMDefaults(
                default_profile_id="default",
                profiles={
                    "default": LLMProfile(),
                    "isolated": LLMProfile(),
                },
            )
        )
        runtime = Runtime(SQLiteStore(":memory:"), config=config)
        try:
            baseline = runtime.llms.profile_identity_sha256("default")
            isolated = runtime.llms.profile_identity_sha256("isolated")
            assert runtime.llms.profile_identity_sha256("default") == baseline

            monkeypatch.setenv(env_name, equivalent_env_value)
            assert runtime.llms.profile_identity_sha256("default") == baseline

            monkeypatch.setenv(env_name, env_value)

            changed = runtime.llms.profile_identity_sha256("default")
            assert changed != baseline
            assert runtime.llms.profile_identity_sha256("default") == changed
            assert getattr(runtime.llms.resolve("default").client, client_attribute) == expected
            assert runtime.llms.profile_identity_sha256("isolated") == isolated
            assert getattr(runtime.llms.resolve("isolated").client, client_attribute) == isolated_expected
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ("env_name", "profile_kwargs", "env_value", "client_attribute", "expected"),
        [
            ("OPENAI_STORE", {"store": False}, "1", "store", False),
            (
                "OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID",
                {"responses_previous_response_id": False},
                "true",
                "responses_previous_response_id",
                False,
            ),
            (
                "OPENAI_PROMPT_CACHE_RETENTION",
                {"prompt_cache_retention": "in-memory"},
                "24h",
                "prompt_cache_retention",
                "in_memory",
            ),
            (
                "OPENAI_FALLBACK_JSON_ACTIONS",
                {"fallback_json_actions": False},
                "true",
                "fallback_json_actions",
                False,
            ),
        ],
    )
    def test_explicit_default_profile_policy_precedes_legacy_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
        profile_kwargs: dict[str, object],
        env_value: str,
        client_attribute: str,
        expected: object,
    ) -> None:
        monkeypatch.delenv(env_name, raising=False)
        config = AgentLibOSConfig(
            llm=LLMDefaults(
                profiles={"default": LLMProfile(**profile_kwargs)},
            )
        )
        runtime = Runtime(SQLiteStore(":memory:"), config=config)
        try:
            baseline = runtime.llms.profile_identity_sha256("default")
            monkeypatch.setenv(env_name, env_value)

            assert runtime.llms.profile_identity_sha256("default") == baseline
            assert getattr(runtime.llms.resolve("default").client, client_attribute) == expected
        finally:
            runtime.close()

    def test_legacy_prompt_cache_retention_env_is_normalized_before_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_PROMPT_CACHE_RETENTION", "in-memory")
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            legacy_identity = runtime.llms.profile_identity_sha256("default")
            legacy_client = runtime.llms.resolve("default").client

            assert legacy_client.prompt_cache_retention == "in_memory"

            monkeypatch.setenv("OPENAI_PROMPT_CACHE_RETENTION", "in_memory")

            assert runtime.llms.profile_identity_sha256("default") == legacy_identity
            assert runtime.llms.resolve("default").client is legacy_client
        finally:
            runtime.close()

    def test_profile_identity_excludes_api_key_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "first-secret")
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            first = runtime.llms.profile_identity_sha256("default")

            monkeypatch.setenv("OPENAI_API_KEY", "second-secret")

            assert runtime.llms.profile_identity_sha256("default") == first
        finally:
            runtime.close()

    def test_profile_context_window_override_does_not_change_provider_identity(self) -> None:
        first = Runtime(
            SQLiteStore(":memory:"),
            config=AgentLibOSConfig(
                llm=LLMDefaults(
                    profiles={
                        "default": LLMProfile(context_window_tokens=100_000),
                    }
                )
            ),
        )
        second = Runtime(
            SQLiteStore(":memory:"),
            config=AgentLibOSConfig(
                llm=LLMDefaults(
                    profiles={
                        "default": LLMProfile(context_window_tokens=200_000),
                    }
                )
            ),
        )
        try:
            assert (
                first.llms.profile_identity_sha256("default")
                == second.llms.profile_identity_sha256("default")
            )
            assert first.llms.resolve("default").context_window_tokens == 100_000
            assert second.llms.resolve("default").context_window_tokens == 200_000
        finally:
            first.close()
            second.close()

    def test_cached_default_client_is_rebuilt_for_new_effective_release_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_STORE", "1")
        monkeypatch.setenv("OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID", "true")
        monkeypatch.setenv("OPENAI_PROMPT_CACHE_RETENTION", "24h")
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            permissive_identity = runtime.llms.profile_identity_sha256("default")
            permissive_client = runtime.llms.resolve("default").client
            assert permissive_client.store is True
            assert permissive_client.responses_previous_response_id is True
            assert permissive_client.prompt_cache_retention == "24h"

            monkeypatch.setenv("OPENAI_STORE", "0")
            monkeypatch.setenv("OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID", "false")
            monkeypatch.setenv("OPENAI_PROMPT_CACHE_RETENTION", "in-memory")
            strict_identity = runtime.llms.profile_identity_sha256("default")
            assert strict_identity != permissive_identity
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern="llm:default",
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="secret",
                    identity_sha256=strict_identity,
                ),
                actor="test.host",
                require_capability=False,
            )
            pid = runtime.process.spawn(image="base-agent:v0", goal="strict LLM release")
            runtime.data_flow.precheck_egress_clearance(
                pid=pid,
                sink=DataSink("llm:default", strict_identity),
                context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
                payload={"messages": [{"role": "user", "content": "secret"}]},
            )

            strict_client = runtime.llms.resolve("default").client

            assert strict_client is not permissive_client
            assert strict_client.store is False
            assert strict_client.responses_previous_response_id is False
            assert strict_client.prompt_cache_retention == "in_memory"
            assert runtime.llms.profile_identity_sha256("default") == strict_identity
        finally:
            runtime.close()

    def test_profile_snapshot_binds_identity_and_client_policy_across_environment_change(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_STORE", "1")
        monkeypatch.setenv("OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID", "true")
        monkeypatch.setenv("OPENAI_PROMPT_CACHE_RETENTION", "24h")
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            snapshot = runtime.llms.profile_snapshot("default")

            monkeypatch.setenv("OPENAI_STORE", "0")
            monkeypatch.setenv("OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID", "false")
            monkeypatch.setenv("OPENAI_PROMPT_CACHE_RETENTION", "in-memory")
            assert runtime.llms.profile_identity_sha256("default") != snapshot.identity_sha256

            resolved = runtime.llms.resolve("default", snapshot=snapshot)

            assert resolved.identity_sha256 == snapshot.identity_sha256
            assert resolved.client.store is True
            assert resolved.client.responses_previous_response_id is True
            assert resolved.client.prompt_cache_retention == "24h"
        finally:
            runtime.close()

    def test_concurrent_policy_cache_invalidation_rebuilds_once_and_closes_stale_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_STORE", "1")
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            stale_client = runtime.llms.resolve("default").client
            monkeypatch.setenv("OPENAI_STORE", "0")
            original_create = runtime.llms._create_client
            original_shutdown = runtime.llms._shutdown_client
            created = 0
            closed: list[object] = []
            observations_lock = threading.Lock()

            def delayed_create(profile_id, profile, *, snapshot):
                nonlocal created
                with observations_lock:
                    created += 1
                time.sleep(0.02)
                return original_create(profile_id, profile, snapshot=snapshot)

            def recording_shutdown(client) -> None:
                with observations_lock:
                    closed.append(client)
                original_shutdown(client)

            monkeypatch.setattr(runtime.llms, "_create_client", delayed_create)
            monkeypatch.setattr(runtime.llms, "_shutdown_client", recording_shutdown)
            worker_count = 8
            barrier = threading.Barrier(worker_count)

            def resolve_after_barrier():
                barrier.wait()
                return runtime.llms.resolve("default").client

            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                clients = list(pool.map(lambda _: resolve_after_barrier(), range(worker_count)))

            assert len({id(client) for client in clients}) == 1
            assert clients[0] is runtime.llms.resolve("default").client
            assert clients[0] is not stale_client
            assert clients[0].store is False
            assert created == 1
            assert closed == [stale_client]
        finally:
            runtime.close()

    def test_policy_change_does_not_close_or_replace_host_test_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_STORE", "1")
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            stale_real_client = runtime.llms.resolve("default").client
            test_client = CloseCountingClient()
            runtime.llms.set_test_client("default", test_client)
            monkeypatch.setenv("OPENAI_STORE", "0")
            strict_identity = runtime.llms.profile_identity_sha256("default")

            first = runtime.llms.resolve("default")
            second = runtime.llms.resolve("default")

            assert first.client is test_client
            assert second.client is test_client
            assert first.identity_sha256 == strict_identity
            assert second.identity_sha256 == strict_identity
            assert test_client.close_calls == 0

            runtime.llms.clear_test_client("default")
            strict_real_client = runtime.llms.resolve("default").client

            assert test_client.close_calls == 1
            assert strict_real_client is not stale_real_client
            assert strict_real_client.store is False
        finally:
            runtime.close()

    def test_executor_uses_snapshot_and_resolved_identity_without_rehashing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            client = RecordingActionClient(
                [{"action": "process_exit", "payload": {"identity": "resolved"}}]
            )
            runtime.llms.set_test_client("default", client)

            def unexpected_rehash(_profile_id: str) -> str:
                raise AssertionError("executor must use the frozen/resolved identity")

            monkeypatch.setattr(
                runtime.llms,
                "profile_identity_sha256",
                unexpected_rehash,
            )
            pid = runtime.process.spawn(image="base-agent:v0", goal="resolved identity")

            result = runtime.run_process_once(pid)

            assert result["ok"] is True
            assert len(client.user_prompts) == 1
        finally:
            runtime.close()

    def test_old_profile_trust_is_denied_after_effective_policy_change(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_STORE", raising=False)
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="identity-bound LLM egress")
            old_identity = runtime.llms.profile_identity_sha256("default")
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern="llm:default",
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="secret",
                    identity_sha256=old_identity,
                ),
                actor="test.host",
                require_capability=False,
            )

            monkeypatch.setenv("OPENAI_STORE", "1")
            changed_identity = runtime.llms.profile_identity_sha256("default")

            assert changed_identity != old_identity
            with pytest.raises(CapabilityDenied, match="identity hash does not match"):
                runtime.data_flow.precheck_egress_clearance(
                    pid=pid,
                    sink=DataSink("llm:default", changed_identity),
                    context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
                    payload={"messages": [{"role": "user", "content": "secret"}]},
                )
        finally:
            runtime.close()

    def test_different_processes_use_different_profile_clients(self) -> None:
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            fast = RecordingActionClient([{"action": "process_exit", "payload": {"profile": "fast"}}])
            slow = RecordingActionClient([{"action": "process_exit", "payload": {"profile": "slow"}}])
            runtime.llms.set_test_client("fast", fast)
            runtime.llms.set_test_client("slow", slow)

            fast_pid = runtime.process.spawn(image="base-agent:v0", goal="fast", llm_profile_id="fast")
            slow_pid = runtime.process.spawn(image="base-agent:v0", goal="slow", llm_profile_id="slow")

            runtime.run_process_once(fast_pid)
            runtime.run_process_once(slow_pid)

            assert len(fast.user_prompts) == 1
            assert len(slow.user_prompts) == 1
            calls = {call.pid: call for call in runtime.store.list_llm_calls(limit=10)}
            assert calls[fast_pid].request_options["llm_profile_id"] == "fast"
            assert calls[fast_pid].request_options["client_class"] == "RecordingActionClient"
            assert calls[slow_pid].request_options["llm_profile_id"] == "slow"
        finally:
            runtime.close()

    def test_spawn_child_fork_and_exec_profile_selection_rules(self) -> None:
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            runtime.register_image(
                AgentImage(
                    image_id="profile-image:v0",
                    name="profile-image",
                    default_tools=["process_exit"],
                    llm_profile_id="image-default",
                ),
                actor="cli",
            )
            runtime.register_image(
                AgentImage(
                    image_id="next-profile-image:v0",
                    name="next-profile-image",
                    default_tools=["process_exit"],
                    llm_profile_id="image-default",
                ),
                actor="cli",
            )

            from_image = runtime.process.spawn(image="profile-image:v0", goal="image default")
            explicit = runtime.process.spawn(image="profile-image:v0", goal="explicit", llm_profile_id="fast")
            runtime.capability.grant(explicit, "process:spawn", [CapabilityRight.WRITE], issued_by="test")
            runtime.capability.grant(explicit, "image:next-profile-image:v0", [CapabilityRight.READ], issued_by="test")
            runtime.capability.grant(explicit, "image:base-agent:v0", [CapabilityRight.READ], issued_by="test")
            forked = runtime.process.fork(parent=explicit, goal="fork inherits")
            spawned = runtime.spawn_child_process(explicit, "fresh child inherits")

            assert runtime.process.get(from_image).llm_profile_id == "image-default"
            assert runtime.process.get(explicit).llm_profile_id == "fast"
            assert runtime.process.get(forked).llm_profile_id == "fast"
            assert runtime.process.get(spawned).llm_profile_id == "fast"

            runtime.exec_process(explicit, "next-profile-image:v0", goal="exec keeps profile")
            assert runtime.process.get(explicit).llm_profile_id == "fast"

            runtime.capability.grant(explicit, "image:base-agent:v0", [CapabilityRight.READ], issued_by="test")
            runtime.exec_process(explicit, "base-agent:v0", goal="exec override", llm_profile_id="override")
            assert runtime.process.get(explicit).llm_profile_id == "override"
        finally:
            runtime.close()

    def test_unknown_profile_fails_closed_when_llm_quantum_runs(self) -> None:
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="unknown profile", llm_profile_id="missing")

            result = runtime.run_process_once(pid)

            assert result["ok"] is False
            assert "unknown LLM profile" in result["error"]
            assert runtime.process.get(pid).status == ProcessStatus.FAILED
            calls = runtime.store.list_llm_calls(pid=pid)
            assert len(calls) == 1
            assert calls[0].status == "error"
            assert calls[0].request_options["llm_profile_id"] == "missing"
        finally:
            runtime.close()

    def test_process_llm_profile_persists_across_reopen(self) -> None:
        config = _profile_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f"{temp_dir}/runtime.sqlite"
            runtime = Runtime.open(db, config=config)
            try:
                pid = runtime.process.spawn(image="base-agent:v0", goal="persist profile", llm_profile_id="slow")
            finally:
                runtime.close()

            reopened = Runtime.open(db, config=config)
            try:
                assert reopened.process.get(pid).llm_profile_id == "slow"
            finally:
                reopened.close()

    def test_llm_profile_can_override_parallel_tool_calls(self) -> None:
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            default = runtime.llms.resolve("default")
            parallel = runtime.llms.resolve("parallel")

            assert default.parallel_tool_calls is False
            assert parallel.parallel_tool_calls is True
            assert parallel.client.parallel_tool_calls is True
        finally:
            runtime.close()

    def test_llm_profile_can_override_auto_wait_on_empty_tool_calls(self) -> None:
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            default = runtime.llms.resolve("default")
            auto_wait = runtime.llms.resolve("auto-wait")

            assert default.auto_wait_on_empty_tool_calls is False
            assert auto_wait.auto_wait_on_empty_tool_calls is True
        finally:
            runtime.close()

    def test_llm_profile_can_opt_in_to_json_action_fallback(self) -> None:
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            default = runtime.llms.resolve("default")
            fallback = runtime.llms.resolve("json-fallback")

            assert default.fallback_json_actions is False
            assert default.client.fallback_json_actions is False
            assert fallback.fallback_json_actions is True
            assert fallback.client.fallback_json_actions is True
        finally:
            runtime.close()

    def test_dynamic_llm_profile_can_be_unregistered(self) -> None:
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            runtime.llms.register_profile("temporary", LLMProfile(model="temporary-model"))
            assert runtime.llms.resolve("temporary").profile.model == "temporary-model"

            runtime.llms.unregister_profile("temporary")

            try:
                runtime.llms.resolve("temporary")
            except ValidationError as exc:
                assert "unknown LLM profile" in str(exc)
            else:
                raise AssertionError("temporary profile should be removed")
        finally:
            runtime.close()

    def test_only_default_profile_inherits_legacy_openai_environment(self, monkeypatch) -> None:
        for env_name in _AMBIENT_ACCOUNT_POLICY_ENV:
            monkeypatch.delenv(env_name, raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
        monkeypatch.setenv("OPENAI_MODEL", "ambient-model")
        monkeypatch.setenv("OPENAI_API_MODE", "chat")
        monkeypatch.setenv("OPENAI_TIMEOUT", "7")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "5")
        monkeypatch.setenv("OPENAI_STORE", "1")
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
        monkeypatch.setenv("OPENAI_VERBOSITY", "high")
        monkeypatch.setenv("OPENAI_PARALLEL_TOOL_CALLS", "1")
        monkeypatch.setenv("OPENAI_ENABLE_THINKING", "1")
        monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
        monkeypatch.setenv("AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
        monkeypatch.setenv("PROFILE_API_KEY", "profile-key")
        config = AgentLibOSConfig(
            llm=LLMDefaults(
                default_profile_id="default",
                profiles={
                    "default": LLMProfile(),
                    "isolated": LLMProfile(model="isolated-model", api_key_env="PROFILE_API_KEY"),
                },
            )
        )
        runtime = Runtime(SQLiteStore(":memory:"), config=config)
        try:
            default_resolved = runtime.llms.resolve("default")
            default_client = default_resolved.client
            isolated_client = runtime.llms.resolve("isolated").client

            assert default_client.base_url == "https://ambient.example/v1"
            assert default_client.model == "ambient-model"
            assert default_client.api_mode == "chat"
            assert default_client.timeout == 7.0
            assert default_client.max_retries == 5
            assert default_client.store is True
            assert default_client.reasoning_effort == "medium"
            assert default_client.verbosity == "high"
            assert default_client.parallel_tool_calls is True
            assert default_client._extra_body() == {"enable_thinking": True}
            assert default_client.organization == "ambient-org"
            assert default_client.project == "ambient-project"
            assert default_resolved.parallel_tool_calls is True
            assert isolated_client.base_url is None
            assert isolated_client.model == "isolated-model"
            assert isolated_client.api_mode == config.llm.api_mode
            assert isolated_client.timeout == config.llm.timeout_s
            assert isolated_client.max_retries == config.llm.max_retries
            assert isolated_client.store == config.llm.store
            assert isolated_client.reasoning_effort is None
            assert isolated_client.verbosity is None
            assert isolated_client.parallel_tool_calls == config.llm.parallel_tool_calls
            assert isolated_client._extra_body() == {}
            assert isolated_client.organization is None
            assert isolated_client.project is None
            assert isolated_client.inherit_ambient_openai_sdk_config is False
            isolated_kwargs = isolated_client._client_kwargs()
            assert isolated_kwargs["base_url"] == "https://api.openai.com/v1"
            assert isolated_kwargs["organization"] == ""
            assert isolated_kwargs["project"] == ""
            assert isolated_client.api_key == "profile-key"
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        "env_name",
        (
            pytest.param("OPENAI_ENABLE_THINKING", id="thinking"),
            pytest.param("OPENAI_ORGANIZATION", id="organization"),
            pytest.param("OPENAI_ORG_ID", id="org-id"),
            pytest.param("OPENAI_PROJECT", id="project"),
            pytest.param("OPENAI_PROJECT_ID", id="project-id"),
        ),
    )
    def test_named_profile_chain_identity_ignores_ambient_account_routing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
    ) -> None:
        for selected_env_name in _AMBIENT_ACCOUNT_POLICY_ENV:
            monkeypatch.delenv(selected_env_name, raising=False)
        monkeypatch.setenv("PROFILE_API_KEY", "profile-key")
        config = AgentLibOSConfig(
            llm=LLMDefaults(
                profiles={
                    "default": LLMProfile(model="default-model"),
                    "isolated": LLMProfile(model="isolated-model", api_key_env="PROFILE_API_KEY"),
                }
            )
        )
        runtime = Runtime(SQLiteStore(":memory:"), config=config)
        try:
            client = runtime.llms.resolve("isolated").client
            baseline_profile = runtime.llms.profile_identity_sha256("isolated")
            baseline_chain = LLMProcessExecutor._openai_provider_chain_fingerprint(client)

            assert len(baseline_profile) == 64
            assert baseline_chain is not None
            assert len(baseline_chain) == 64

            monkeypatch.setenv(env_name, "1" if env_name == "OPENAI_ENABLE_THINKING" else "changed")

            assert runtime.llms.profile_identity_sha256("isolated") == baseline_profile
            assert LLMProcessExecutor._openai_provider_chain_fingerprint(client) == baseline_chain
            assert client._extra_body() == {}
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ("env_name", "before", "after", "client_attribute"),
        (
            pytest.param("OPENAI_ENABLE_THINKING", "1", "0", "enable_thinking", id="thinking"),
            pytest.param("OPENAI_ORGANIZATION", "org-a", "org-b", "organization", id="organization"),
            pytest.param("OPENAI_ORG_ID", "org-a", "org-b", "organization", id="org-id"),
            pytest.param("OPENAI_PROJECT", "project-a", "project-b", "project", id="project"),
            pytest.param("OPENAI_PROJECT_ID", "project-a", "project-b", "project", id="project-id"),
        ),
    )
    def test_default_profile_snapshot_identity_tracks_each_account_and_thinking_field(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
        before: str,
        after: str,
        client_attribute: str,
    ) -> None:
        for selected_env_name in _AMBIENT_ACCOUNT_POLICY_ENV:
            monkeypatch.delenv(selected_env_name, raising=False)
        monkeypatch.setenv(env_name, before)
        runtime = Runtime(SQLiteStore(":memory:"), config=_profile_config())
        try:
            snapshot = runtime.llms.profile_snapshot("default")
            monkeypatch.setenv(env_name, after)

            current_identity = runtime.llms.profile_identity_sha256("default")
            assert len(snapshot.identity_sha256) == 64
            assert len(current_identity) == 64
            assert current_identity != snapshot.identity_sha256

            frozen = runtime.llms.resolve("default", snapshot=snapshot).client
            current = runtime.llms.resolve("default").client
            expected_before: object = before == "1" if client_attribute == "enable_thinking" else before
            expected_after: object = after == "1" if client_attribute == "enable_thinking" else after
            assert getattr(frozen, client_attribute) == expected_before
            assert getattr(current, client_attribute) == expected_after
            assert current is not frozen
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ("env_name", "before", "after"),
        (
            pytest.param("OPENAI_ENABLE_THINKING", "1", "0", id="thinking"),
            pytest.param("OPENAI_ORGANIZATION", "org-a", "org-b", id="organization"),
            pytest.param("OPENAI_ORG_ID", "org-a", "org-b", id="org-id"),
            pytest.param("OPENAI_PROJECT", "project-a", "project-b", id="project"),
            pytest.param("OPENAI_PROJECT_ID", "project-a", "project-b", id="project-id"),
        ),
    )
    def test_default_profile_responses_chain_resets_when_one_account_or_thinking_field_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
        before: str,
        after: str,
    ) -> None:
        for selected_env_name in _AMBIENT_ACCOUNT_POLICY_ENV:
            monkeypatch.delenv(selected_env_name, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "profile-chain-key")
        monkeypatch.setenv(env_name, before)
        config = AgentLibOSConfig(
            llm=LLMDefaults(
                profiles={
                    "default": LLMProfile(
                        model="gpt-test",
                        api_mode="responses",
                        store=True,
                        responses_previous_response_id=True,
                    )
                }
            )
        )
        runtime = Runtime(SQLiteStore(":memory:"), config=config)
        try:
            first_client = runtime.llms.resolve("default").client
            first_fake = _ProfileFakeAsyncOpenAIResponses(
                [
                    _profile_response_tool_call(
                        "resp_profile_a",
                        "discover_skills",
                        {"text": "memory", "limit": 4},
                    )
                ]
            )
            first_client._async_client = first_fake
            pid = runtime.process.spawn(image="base-agent:v0", goal="reset changed profile chain")

            first = runtime.run_next_process_once()
            assert first["action"]["action"] == "discover_skills"

            monkeypatch.setenv(env_name, after)
            second_client = runtime.llms.resolve("default").client
            second_fake = _ProfileFakeAsyncOpenAIResponses(
                [
                    _profile_response_tool_call(
                        "resp_profile_b",
                        "process_exit",
                        {"payload": {"done": True}},
                    )
                ]
            )
            second_client._async_client = second_fake

            assert second_client is not first_client
            second = runtime.run_next_process_once()

            assert second["action"]["action"] == "process_exit"
            assert "previous_response_id" not in second_fake.responses.payloads[0]
            assert not any(
                item.get("type") == "function_call_output"
                for item in second_fake.responses.payloads[0]["input"]
            )
            calls = runtime.store.list_llm_calls(pid)
            first_chain = calls[0].request_options["openai_provider_chain_fingerprint"]
            second_chain = calls[1].request_options["openai_provider_chain_fingerprint"]
            assert first_chain is not None and len(first_chain) == 64
            assert second_chain is not None and len(second_chain) == 64
            assert second_chain != first_chain
            assert calls[1].request_options["openai_previous_response_id"] is None
        finally:
            runtime.close()

    def test_runtime_shutdown_closes_async_llm_clients_inside_running_loop(self) -> None:
        async def run() -> bool:
            runtime = await RuntimeBuilder.configured(
                Runtime,
                config=_profile_config(),
            ).afrom_store(SQLiteStore(":memory:"))
            client = AsyncCloseOnlyClient()
            runtime.llms.set_test_client("default", client)

            result = runtime.shutdown(actor="test", reason="event-loop-shutdown")

            assert result["ok"] is True
            return client.closed

        assert asyncio.run(run()) is True


class TestUserLLMProfileStore:
    def test_default_user_llm_profile_paths_follow_platform_conventions(self) -> None:
        home = Path("/home/example")
        assert default_user_llm_profiles_path(platform="win32", env={"APPDATA": "C:/Users/example/AppData/Roaming"}, home=home) == Path("C:/Users/example/AppData/Roaming") / "Agent libOS" / "llm-profiles.json"
        assert default_user_llm_profiles_path(platform="darwin", env={}, home=home) == home / "Library" / "Application Support" / "Agent libOS" / "llm-profiles.json"
        assert default_user_llm_profiles_path(platform="linux", env={"XDG_CONFIG_HOME": "/tmp/config"}, home=home) == Path("/tmp/config") / "agent-libos" / "llm-profiles.json"

    def test_user_llm_profile_store_round_trips_non_secret_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "llm-profiles.json"
            store = UserLLMProfileStore(path)

            saved = store.upsert(
                "qwen3.7-max",
                {
                    "profile_id": "qwen3.7-max",
                    "model": "qwen3.7-max",
                    "base_url": "https://qwen.example/v1/",
                    "api_key_env": "QWEN_API_KEY",
                    "api_mode": "chat",
                    "temperature": 0.1,
                    "max_tokens": 8192,
                    "context_window_tokens": 200000,
                    "auto_wait_on_empty_tool_calls": True,
                    "fallback_json_actions": True,
                    "prompt_cache_retention": "in-memory",
                    "allow_custom_base_url": False,
                },
            )
            implicit = store.upsert(
                "compat-without-opt-in",
                {
                    "model": "compat-without-opt-in",
                    "base_url": "https://compat.example/v1",
                    "api_key_env": "COMPAT_API_KEY",
                },
            )
            loaded = UserLLMProfileStore(path).load()

            assert saved.model == "qwen3.7-max"
            assert implicit.allow_custom_base_url is False
            assert loaded["qwen3.7-max"].base_url == "https://qwen.example/v1"
            assert loaded["qwen3.7-max"].api_key_env == "QWEN_API_KEY"
            assert loaded["qwen3.7-max"].allow_custom_base_url is False
            assert loaded["compat-without-opt-in"].allow_custom_base_url is False
            assert loaded["qwen3.7-max"].auto_wait_on_empty_tool_calls is True
            assert loaded["qwen3.7-max"].fallback_json_actions is True
            assert saved.prompt_cache_retention == "in_memory"
            assert loaded["qwen3.7-max"].prompt_cache_retention == "in_memory"
            assert loaded["qwen3.7-max"].context_window_tokens == 200000
            persisted = json.loads(path.read_text(encoding="utf-8"))["profiles"]["qwen3.7-max"]
            assert persisted["allow_custom_base_url"] is False
            assert persisted["prompt_cache_retention"] == "in_memory"
            assert "secret" not in path.read_text(encoding="utf-8")
            assert "api_key" not in persisted

    def test_user_llm_profile_store_normalizes_legacy_retention_on_load_and_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "llm-profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profiles": {
                            "legacy": {
                                "model": "legacy-model",
                                "api_key_env": "LEGACY_API_KEY",
                                "prompt_cache_retention": "in-memory",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = UserLLMProfileStore(path)

            loaded = store.load()

            assert loaded["legacy"].prompt_cache_retention == "in_memory"

            store.save(loaded)

            persisted = json.loads(path.read_text(encoding="utf-8"))
            assert persisted["profiles"]["legacy"]["prompt_cache_retention"] == "in_memory"
            assert "in-memory" not in path.read_text(encoding="utf-8")

    def test_user_llm_profile_store_rejects_invalid_json_and_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "llm-profiles.json"
            path.write_text("{bad", encoding="utf-8")
            store = UserLLMProfileStore(path)

            try:
                store.load()
            except ValidationError as exc:
                assert "invalid LLM profiles JSON" in str(exc)
            else:
                raise AssertionError("bad JSON should fail closed")

            path.unlink()
            try:
                store.upsert("bad", {"model": "bad", "api_key_env": "BAD_API_KEY", "api_key": "secret"})
            except ValidationError as exc:
                assert "API keys are not accepted" in str(exc)
            else:
                raise AssertionError("raw API keys should be rejected")

            with pytest.raises(ValidationError, match="max_tokens must be less"):
                store.upsert(
                    "window-too-small",
                    {
                        "model": "window-too-small",
                        "api_key_env": "SMALL_API_KEY",
                        "max_tokens": 32_768,
                        "context_window_tokens": 32_768,
                    },
                )

            with pytest.raises(ValidationError, match="temperature must be non-negative"):
                store.upsert(
                    "negative-temperature",
                    {
                        "model": "negative-temperature",
                        "api_key_env": "NEGATIVE_API_KEY",
                        "temperature": -0.1,
                    },
                )

            with pytest.raises(ValidationError, match="max_retries must be an integer"):
                store.upsert(
                    "fractional-retries",
                    {
                        "model": "fractional-retries",
                        "api_key_env": "FRACTIONAL_API_KEY",
                        "max_retries": 1.5,
                    },
                )

    def test_runtime_ashutdown_closes_async_llm_clients(self) -> None:
        async def run() -> bool:
            runtime = await RuntimeBuilder.configured(
                Runtime,
                config=_profile_config(),
            ).afrom_store(SQLiteStore(":memory:"))
            client = AsyncCloseOnlyClient()
            runtime.llms.set_test_client("default", client)

            result = await runtime.ashutdown(actor="test", reason="async-shutdown")

            assert result["ok"] is True
            return client.closed

        assert asyncio.run(run()) is True


class _ProfileFakeAsyncOpenAIResponses:
    def __init__(self, responses: list[object]) -> None:
        self.responses = _ProfileSequencedResponses(responses)


class _ProfileSequencedResponses:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.payloads: list[dict[str, object]] = []

    async def create(self, **payload: object) -> object:
        self.payloads.append(payload)
        return self._responses.pop(0)


def _profile_response_tool_call(
    response_id: str,
    name: str,
    arguments: dict[str, object],
) -> object:
    return SimpleNamespace(
        id=response_id,
        _request_id=f"req_{response_id}",
        model="gpt-test",
        usage=SimpleNamespace(input_tokens=5, output_tokens=2, total_tokens=7),
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                id=f"fc_{response_id}",
                call_id=f"call_{response_id}",
                name=name,
                arguments=json.dumps(arguments),
            )
        ],
    )


class AsyncCloseOnlyClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class CloseCountingClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
