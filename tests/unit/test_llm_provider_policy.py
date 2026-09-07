from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG, LLMDefaults, LLMProfile, SemanticDefaults
from agent_libos.llm.profiles import LLMProfileRegistry
from agent_libos.llm.provider_policy import resolve_provider_policy


@pytest.mark.parametrize("endpoint", [None, "https://api.openai.com/v1", "https://api.openai.com/v1/"])
def test_official_unset_model_uses_astra_defaults(endpoint: str | None) -> None:
    policy = resolve_provider_policy(defaults=DEFAULT_CONFIG.llm, base_url=endpoint, model=None)
    assert policy.model == "gpt-6-astra"
    assert policy.reasoning_effort == "medium"
    assert policy.reasoning_context == "all_turns"
    assert policy.responses_replay is True
    assert policy.prompt_cache_mode == "provider_default"
    assert policy.prompt_layout == "legacy_v1"


@pytest.mark.parametrize("model", [None, "gpt-6-astra", "third-party-model"])
def test_custom_endpoint_has_no_implicit_openai_policy(model: str | None) -> None:
    policy = resolve_provider_policy(
        defaults=DEFAULT_CONFIG.llm, base_url="https://compatible.example/v1", model=model,
        prompt_cache_mode="auto", prompt_layout="auto",
    )
    assert policy.model == model
    assert policy.reasoning_effort is None
    assert policy.reasoning_context is None
    assert policy.responses_replay is False
    assert policy.prompt_cache_mode == "provider_default"
    assert policy.prompt_cache_ttl is None
    assert policy.prompt_layout == "legacy_v1"


def test_auto_candidate_is_endpoint_scoped_and_keeps_explicit_legacy_retention() -> None:
    policy = resolve_provider_policy(
        defaults=DEFAULT_CONFIG.llm, base_url=None, model=None,
        prompt_layout="auto", prompt_cache_mode="auto",
    )
    assert (policy.prompt_layout, policy.prompt_cache_mode, policy.prompt_cache_ttl) == (
        "cache_optimized_v2", "implicit", "30m",
    )
    retained = resolve_provider_policy(
        defaults=DEFAULT_CONFIG.llm, base_url=None, model=None,
        prompt_cache_mode="auto", prompt_cache_retention="24h",
    )
    assert (retained.prompt_cache_mode, retained.prompt_cache_ttl, retained.prompt_cache_retention) == (
        "provider_default", None, "24h",
    )


def test_explicit_custom_responses_replay_and_reasoning_policy_is_preserved() -> None:
    policy = resolve_provider_policy(
        defaults=DEFAULT_CONFIG.llm, base_url="https://compatible.example/v1", model="custom",
        api_mode="responses", reasoning_effort="high", reasoning_context="current_turn",
        responses_replay=True, prompt_cache_mode="explicit", prompt_cache_ttl="30m",
    )
    assert policy.responses_replay is True
    assert policy.reasoning_effort == "high"
    assert policy.reasoning_context == "current_turn"
    assert policy.prompt_cache_mode == "explicit"
    chat = resolve_provider_policy(
        defaults=DEFAULT_CONFIG.llm, base_url=None, model="gpt-6-astra",
        api_mode="chat", responses_replay=True,
    )
    assert chat.responses_replay is False


def test_profile_and_environment_precedence_is_frozen_without_sdk_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "ambient-model")
    monkeypatch.setenv("OPENAI_REASONING_CONTEXT", "current_turn")
    monkeypatch.setenv("OPENAI_RESPONSES_REPLAY", "false")
    monkeypatch.setenv("OPENAI_PROMPT_LAYOUT", "cache_optimized_v2")
    config = AgentLibOSConfig(llm=LLMDefaults(profiles={
        "default": LLMProfile(),
        "isolated": LLMProfile(model="gpt-6-astra"),
    }))
    registry = LLMProfileRegistry(SimpleNamespace(), config=config)
    legacy = registry.profile_snapshot("default")
    isolated = registry.profile_snapshot("isolated")
    assert legacy.policy.model == "ambient-model"
    assert legacy.policy.reasoning_context == "current_turn"
    assert legacy.policy.responses_replay is False
    assert legacy.policy.prompt_layout == "cache_optimized_v2"
    assert isolated.policy.model == "gpt-6-astra"
    assert isolated.policy.reasoning_context == "all_turns"
    assert isolated.policy.responses_replay is True
    assert isolated.policy.prompt_layout == "legacy_v1"
    assert registry._clients == {}
    registry.register_profile("default", LLMProfile(
        model="explicit-model", reasoning_context="all_turns", responses_replay=True,
        prompt_layout="legacy_v1", api_mode="responses",
    ))
    explicit = registry.profile_snapshot("default")
    assert explicit.policy.model == "explicit-model"
    assert explicit.policy.reasoning_context == "all_turns"
    assert explicit.policy.responses_replay is True
    assert explicit.policy.prompt_layout == "legacy_v1"


def test_auto_cache_domains_are_private_profile_scoped_and_stable() -> None:
    config = AgentLibOSConfig(llm=LLMDefaults(prompt_cache_mode="auto", profiles={
        "default": LLMProfile(), "second": LLMProfile(),
    }))
    registry = LLMProfileRegistry(SimpleNamespace(), config=config)
    first = registry.profile_snapshot("default")
    registry.profile_snapshot("second")
    assert registry.profile_snapshot("default").client_cache_sha256 == first.client_cache_sha256
    assert len(registry._cache_privacy_domains) == 2
    assert registry._cache_privacy_domains["default"] != registry._cache_privacy_domains["second"]
    assert registry._cache_privacy_domains["default"] not in repr(first)
    other = LLMProfileRegistry(SimpleNamespace(), config=config)
    other_snapshot = other.profile_snapshot("default")
    assert first.identity_sha256 == other_snapshot.identity_sha256
    assert first.client_cache_sha256 != other_snapshot.client_cache_sha256


def test_unchanged_custom_profile_keeps_legacy_sink_identity() -> None:
    config = AgentLibOSConfig(llm=LLMDefaults(profiles={
        "default": LLMProfile(),
        "compat": LLMProfile(
            base_url="https://compatible.example/v1", model="legacy-custom",
            api_key_env="CUSTOM_KEY", api_mode="chat", allow_custom_base_url=True,
        ),
    }))
    registry = LLMProfileRegistry(SimpleNamespace(), config=config)
    # This is the schema-v1 hash before the optional reasoning/layout fields
    # existed. Existing custom-provider trust bindings must remain valid.
    assert registry.profile_identity_sha256("compat") == (
        "1b350b7ffddfb97da15352883b6385bccef01f8dfe0741fdf0ec57ae10728d68"
    )


def test_semantic_classifier_isolated_from_auto_cache_and_replay_defaults() -> None:
    profile = LLMProfile(
        model="gpt-6-astra", api_mode="responses", store=False, max_retries=0,
        timeout_s=5.0, responses_previous_response_id=False, fallback_json_actions=False,
    )
    config = AgentLibOSConfig(
        llm=LLMDefaults(prompt_cache_mode="auto", responses_replay=True, profiles={
            "default": LLMProfile(), "semantic": profile,
        }),
        semantic=SemanticDefaults(mode="shadow", adapter="external", external_profile_id="semantic"),
    )
    registry = LLMProfileRegistry(SimpleNamespace(), config=config)
    policy = registry.profile_snapshot("semantic").policy
    assert policy.responses_replay is False
    assert policy.prompt_cache_mode == "provider_default"
    assert policy.prompt_cache_ttl is None
    assert "semantic" not in registry._cache_privacy_domains
    with pytest.raises(ValueError, match="disable Responses replay"):
        replace(config, llm=replace(config.llm, profiles={
            **config.llm.profiles, "semantic": replace(profile, responses_replay=True),
        }))


@pytest.mark.parametrize("field", ["responses_replay_max_bytes", "responses_replay_max_turns"])
def test_replay_bounds_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        AgentLibOSConfig(llm=replace(DEFAULT_CONFIG.llm, **{field: 0}))
