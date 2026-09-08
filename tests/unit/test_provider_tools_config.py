from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_libos.config import (
    DEFAULT_CONFIG,
    AgentLibOSConfig,
    LLMDefaults,
    LLMProfile,
    ProviderToolsConfig,
    SemanticDefaults,
    normalize_provider_tools,
)
from agent_libos.llm.profiles import LLMProfileRegistry
from agent_libos.llm.provider_policy import resolve_provider_policy
from agent_libos.llm.user_profiles import UserLLMProfileStore, summarize_llm_profile
from agent_libos.models.exceptions import ValidationError


def test_provider_tools_are_typed_immutable_and_detached() -> None:
    file_ids = ["file-input"]
    profile = LLMProfile(provider_tools={
        "provider": "openai", "code_interpreter": True, "file_ids": file_ids,
    })
    file_ids.append("file-later")
    assert profile.provider_tools is not None
    assert profile.provider_tools.enabled is True
    assert profile.provider_tools.file_ids == ("file-input",)
    with pytest.raises(FrozenInstanceError):
        profile.provider_tools.web_search = True


@pytest.mark.parametrize("payload", [
    {"provider": "unknown", "web_search": True},
    {"provider": "openai", "web_search": "true"},
    {"provider": "openai", "code_interpreter": 1},
    {"provider": "openai", "arbitrary_wire_options": {}},
    {"provider": "openai", "web_extractor": True, "web_search": True},
    {"provider": "aliyun", "web_extractor": True},
    {"provider": "aliyun", "code_interpreter": True, "file_ids": ["file-input"]},
    {"provider": "openai", "file_ids": ["file-input"]},
    {"provider": "openai", "code_interpreter": True, "file_ids": [7]},
    {"provider": "openai", "code_interpreter": True, "file_ids": ["file-input", "file-input"]},
    {"provider": "openai", "code_interpreter": True, "file_ids": ["https://example.com/file"]},
    {"provider": "openai", "code_interpreter": True, "file_ids": ["/tmp/file"]},
    {"provider": "openai", "code_interpreter": True, "file_ids": ["f" * 257]},
    {"provider": "openai", "code_interpreter": True, "file_ids": [f"file-{i}" for i in range(101)]},
])
def test_provider_tools_reject_unsupported_or_malformed_configuration(payload: dict) -> None:
    with pytest.raises(ValueError):
        LLMProfile(provider_tools=payload)


@pytest.mark.parametrize("provider", ["openai", "aliyun"])
def test_disabled_provider_tools_canonicalize_to_none(provider: str) -> None:
    assert normalize_provider_tools({"provider": provider}) is None
    assert LLMProfile(provider_tools={"provider": provider}).provider_tools is None
    assert normalize_provider_tools(None) is None


@pytest.mark.parametrize("provider", ["openai", "aliyun"])
def test_enabled_provider_tools_select_responses_for_auto(provider: str) -> None:
    policy = resolve_provider_policy(
        defaults=DEFAULT_CONFIG.llm,
        base_url="https://configured.example/v1",
        model="configured-model",
        provider_tools={"provider": provider, "web_search": True},
    )
    assert policy.api_mode == "responses"
    assert policy.provider_tools == ProviderToolsConfig(provider=provider, web_search=True)


def test_aliyun_chat_supports_only_search() -> None:
    profile = LLMProfile(api_mode="chat", provider_tools={"provider": "aliyun", "web_search": True})
    policy = resolve_provider_policy(
        defaults=DEFAULT_CONFIG.llm, base_url="https://configured.example/v1", model="qwen",
        api_mode="chat", provider_tools=profile.provider_tools, responses_replay=True,
    )
    assert policy.api_mode == "chat"
    assert policy.responses_replay is False
    for tools in (
        {"provider": "openai", "web_search": True},
        {"provider": "aliyun", "web_search": True, "web_extractor": True},
        {"provider": "aliyun", "code_interpreter": True},
    ):
        with pytest.raises(ValueError, match="Chat provider tools"):
            LLMProfile(api_mode="chat", provider_tools=tools)
        with pytest.raises(ValueError, match="Chat provider tools"):
            resolve_provider_policy(
                defaults=DEFAULT_CONFIG.llm, base_url=None, model="test",
                api_mode="chat", provider_tools=tools,
            )


def test_global_chat_policy_rejects_incompatible_inherited_tools() -> None:
    with pytest.raises(ValueError, match="Chat provider tools"):
        AgentLibOSConfig(llm=LLMDefaults(api_mode="chat", profiles={
            "default": LLMProfile(),
            "tools": LLMProfile(provider_tools={"provider": "openai", "web_search": True}),
        }))


@pytest.mark.parametrize("provider", ["openai", "aliyun"])
def test_independent_code_execution_disables_replay_and_chaining_with_configuration_preserved(provider: str) -> None:
    profile = LLMProfile(
        model="configured-model", api_mode="responses", responses_replay=True,
        responses_previous_response_id=True,
        provider_tools={"provider": provider, "code_interpreter": True},
    )
    registry = LLMProfileRegistry(SimpleNamespace())
    registry.register_profile("code", profile)
    registry.set_test_client("code", SimpleNamespace())
    snapshot = registry.profile_snapshot("code")
    resolved = registry.resolve("code", snapshot=snapshot)
    assert snapshot.policy.responses_replay is False
    assert snapshot.policy.responses_previous_response_id is False
    assert resolved.responses_replay_configured is True
    assert resolved.responses_previous_response_id_configured is True
    assert resolved.profile.responses_replay is True
    assert resolved.profile.responses_previous_response_id is True
    assert resolved.provider_tools == profile.provider_tools
    options = registry._client_options(
        profile, profile_id="code", legacy_env=snapshot.legacy_env,
        client_env=snapshot.client_env, policy=snapshot.policy,
    )
    assert options["provider_tools"] == profile.provider_tools
    assert options["responses_replay"] is False
    assert options["responses_previous_response_id"] is False


def test_search_preserves_opted_in_replay() -> None:
    policy = resolve_provider_policy(
        defaults=DEFAULT_CONFIG.llm, base_url=None, model=None,
        provider_tools={"provider": "openai", "web_search": True},
    )
    assert policy.responses_replay is True


def test_tools_change_profile_and_client_identity_but_disabled_tools_preserve_legacy_identity() -> None:
    registry = LLMProfileRegistry(SimpleNamespace())
    profile = LLMProfile(
        base_url="https://compatible.example/v1", model="legacy-custom",
        api_key_env="CUSTOM_KEY", api_mode="chat", allow_custom_base_url=True,
    )
    registry.register_profile("compat", profile)
    before = registry.profile_snapshot("compat")
    assert before.identity_sha256 == "1b350b7ffddfb97da15352883b6385bccef01f8dfe0741fdf0ec57ae10728d68"
    registry.register_profile("compat", replace(profile, provider_tools={"provider": "aliyun"}))
    assert registry.profile_snapshot("compat").identity_sha256 == before.identity_sha256
    assert registry.profile_snapshot("compat").client_cache_sha256 == before.client_cache_sha256
    registry.register_profile("compat", replace(profile, provider_tools={"provider": "aliyun", "web_search": True}))
    enabled = registry.profile_snapshot("compat")
    assert enabled.identity_sha256 != before.identity_sha256
    assert enabled.client_cache_sha256 != before.client_cache_sha256


def test_remote_file_ids_bind_provider_identity() -> None:
    registry = LLMProfileRegistry(SimpleNamespace())
    profile = LLMProfile(model="configured-model", provider_tools={
        "provider": "openai", "code_interpreter": True, "file_ids": ["file-one"],
    })
    registry.register_profile("code", profile)
    before = registry.profile_snapshot("code")
    registry.register_profile("code", replace(profile, provider_tools={
        "provider": "openai", "code_interpreter": True, "file_ids": ["file-two"],
    }))
    assert registry.profile_snapshot("code").identity_sha256 != before.identity_sha256
    assert registry.profile_snapshot("code").client_cache_sha256 != before.client_cache_sha256


def test_user_profile_store_round_trips_provider_tools(tmp_path: Path) -> None:
    store = UserLLMProfileStore(tmp_path / "profiles.json")
    tools = {
        "provider": "openai", "web_search": True, "web_extractor": False,
        "code_interpreter": True, "file_ids": ["file-input"],
    }
    profile = store.upsert("research", {
        "model": "configured-model", "api_key_env": "RESEARCH_API_KEY", "provider_tools": tools,
    })
    assert store.load()["research"] == profile
    summary = summarize_llm_profile("research", profile, source="user", editable=True, default_profile_id="default")
    assert summary["provider_tools"] == tools
    store.upsert("research", {"model": "configured-model", "api_key_env": "RESEARCH_API_KEY", "provider_tools": None})
    assert store.load()["research"].provider_tools is None
    with pytest.raises(ValidationError, match="provider_tools"):
        store.upsert("bad", {
            "model": "configured-model", "api_key_env": "RESEARCH_API_KEY",
            "provider_tools": {"provider": "openai", "tools": []},
        })


def test_semantic_classifier_rejects_configured_and_dynamically_registered_tools() -> None:
    profile = LLMProfile(
        model="classifier", api_mode="responses", store=False, max_retries=0, timeout_s=5.0,
        responses_previous_response_id=False, fallback_json_actions=False,
    )
    config = AgentLibOSConfig(
        llm=LLMDefaults(profiles={"default": LLMProfile(), "semantic": profile}),
        semantic=SemanticDefaults(mode="shadow", adapter="external", external_profile_id="semantic"),
    )
    tools_profile = replace(profile, provider_tools={"provider": "openai", "web_search": True})
    with pytest.raises(ValueError, match="disable provider tools"):
        replace(config, llm=replace(config.llm, profiles={"default": LLMProfile(), "semantic": tools_profile}))
    registry = LLMProfileRegistry(SimpleNamespace(), config=config)
    with pytest.raises(ValidationError, match="disable provider tools"):
        registry.register_profile("semantic", tools_profile)
