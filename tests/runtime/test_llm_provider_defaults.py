from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, LLMDefaults, LLMProfile
from agent_libos.llm.client import LLMClient
from agent_libos.llm.user_profiles import UserLLMProfileStore, summarize_llm_profile
from agent_libos.models.exceptions import ValidationError


def test_profile_layout_controls_model_tool_schemas_without_constructing_clients() -> None:
    config = AgentLibOSConfig(llm=LLMDefaults(profiles={
        "default": LLMProfile(model="legacy-model"),
        "compact": LLMProfile(model="gpt-6-astra", prompt_layout="cache_optimized_v2"),
    }))
    runtime = Runtime.open("local", config=config)
    try:
        for profile_id, expected, absent in (
            ("default", "ProcessCompletionEvidence", "CompactProcessCompletionEvidence"),
            ("compact", "CompactProcessCompletionEvidence", "ProcessCompletionEvidence"),
        ):
            pid = runtime.process.spawn(
                image="base-agent:v0", goal="project matching completion schema",
                llm_profile_id=profile_id,
            )
            rows = runtime.tools.model_visible_tools(pid)
            spec = next(json.loads(row["spec_json"]) for row in rows if row["name"] == "process_exit")
            definitions = spec["input_schema"]["$defs"]
            assert expected in definitions
            assert absent not in definitions
            schemas = runtime.tools.openai_tool_schemas(pid)
            assert any(row["function"]["name"] == "process_exit" for row in schemas)
        assert runtime.llms._clients == {}
    finally:
        runtime.close()


@pytest.mark.parametrize("field,value", [("reasoning_context", "current_turn"), ("responses_replay", False)])
def test_effective_reasoning_policy_changes_provider_identity(field: str, value: object) -> None:
    config = AgentLibOSConfig(llm=LLMDefaults(profiles={"default": LLMProfile(model="gpt-6-astra")}))
    first = Runtime.open("local", config=config)
    second = Runtime.open("local", config=replace(config, llm=replace(config.llm, **{field: value})))
    try:
        assert first.llms.profile_identity_sha256("default") != second.llms.profile_identity_sha256("default")
    finally:
        second.close()
        first.close()


def test_saved_profile_preserves_new_host_policy_when_gui_editor_omits_it(tmp_path: Path) -> None:
    store = UserLLMProfileStore(tmp_path / "profiles.json")
    payload = {
        "model": "gpt-6-astra", "api_key_env": "OPENAI_API_KEY", "api_mode": "responses",
        "reasoning_context": "all_turns", "responses_replay": True,
        "prompt_layout": "auto", "prompt_cache_mode": "auto", "prompt_cache_ttl": "30m",
    }
    store.upsert("modern", payload)
    updated = store.upsert("modern", {"model": "gpt-6-astra", "api_key_env": "OPENAI_API_KEY"})
    loaded = store.load()["modern"]
    for key in ("reasoning_context", "responses_replay", "prompt_layout", "prompt_cache_mode", "prompt_cache_ttl"):
        assert getattr(updated, key) == payload[key]
        assert getattr(loaded, key) == payload[key]
    summary = summarize_llm_profile("modern", loaded, source="user", editable=True, default_profile_id="default", env={})
    assert summary["responses_replay"] is True
    assert summary["prompt_cache_mode"] == "auto"
    cleared = store.upsert("modern", {
        "model": "gpt-6-astra", "api_key_env": "OPENAI_API_KEY",
        "reasoning_context": None, "responses_replay": None, "prompt_layout": None,
        "prompt_cache_mode": None, "prompt_cache_ttl": None,
    })
    assert cleared.responses_replay is None
    assert cleared.prompt_cache_mode is None


def test_from_env_model_fallback_keeps_explicit_legacy_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_LANGUAGE_MODEL", "OPENAI_API_MODE",
        "OPENAI_REASONING_EFFORT", "OPENAI_REASONING_CONTEXT", "OPENAI_RESPONSES_REPLAY",
    ):
        monkeypatch.delenv(key, raising=False)
    implicit = LLMClient.from_env()
    assert implicit.model == "gpt-6-astra"
    assert implicit.api_mode == "auto"
    assert implicit.responses_replay is True
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6")
    monkeypatch.setenv("OPENAI_API_MODE", "chat")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "none")
    monkeypatch.setenv("OPENAI_REASONING_CONTEXT", "current_turn")
    monkeypatch.setenv("OPENAI_RESPONSES_REPLAY", "false")
    explicit = LLMClient.from_env()
    assert explicit.model == "gpt-5.6"
    assert explicit.api_mode == "chat"
    assert explicit.reasoning_effort == "none"
    assert explicit.reasoning_context == "current_turn"
    assert explicit.responses_replay is False
    assert implicit.model == "gpt-6-astra"


@pytest.mark.parametrize("field,value", [
    ("reasoning_context", "hidden"), ("responses_replay", "true"),
    ("prompt_layout", "automatic"), ("prompt_cache_mode", "automatic"),
])
def test_saved_profile_rejects_invalid_new_policy(tmp_path: Path, field: str, value: object) -> None:
    store = UserLLMProfileStore(tmp_path / "profiles.json")
    with pytest.raises(ValidationError, match=field):
        store.upsert("invalid", {"model": "gpt-6-astra", "api_key_env": "OPENAI_API_KEY", field: value})
    assert not store.path.exists()
