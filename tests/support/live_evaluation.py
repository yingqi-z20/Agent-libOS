from __future__ import annotations

import json
from typing import Any

from benchmarks.live_evaluation_provenance import _safe_llm_config_digest


def stable_source_provenance(
    *,
    digest: str = "b" * 64,
    dirty: bool = False,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "available": True,
        "commit": "a" * 40,
        "dirty": dirty,
        "working_tree_sha256": digest,
    }
    return {
        "schema_version": 1,
        "start": identity,
        "end": dict(identity),
        "stable": True,
    }


def stable_evaluation_provenance(
    *,
    digest: str = "b" * 64,
    dirty: bool = False,
    model: str = "publication-test-model",
    credential_present: bool = True,
    prompt_layout: str = "legacy_v1",
) -> dict[str, Any]:
    source = stable_source_provenance(digest=digest, dirty=dirty)["end"]
    llm: dict[str, Any] = {
        "schema_version": 1,
        "available": True,
        "provider_kind": "openai_compatible",
        "profile_id": "default",
        "model": model,
        "endpoint": {
            "classification": "official_openai",
            "normalized_sha256": "e" * 64,
            "custom_endpoint_allowed": False,
        },
        "api_mode": "responses",
        "credential_present": credential_present,
        "request": {
            "temperature": 0.2,
            "max_tokens": 16_384,
            "max_input_tokens_per_call": 114_688,
            "max_total_tokens_per_call": 131_072,
            "context_window_tokens": 131_072,
            "timeout_s": 180.0,
            "max_retries": 2,
            "compatibility_retry_attempts": 8,
            "action_repair_attempts": 2,
            "parallel_tool_calls": False,
            "auto_wait_on_empty_tool_calls": False,
            "enable_thinking": None,
        },
        "prompt": {
            "layout": prompt_layout,
            "store": False,
            "cache_mode": "provider_default",
            "cache_retention": None,
            "cache_ttl": None,
            "cache_key_configured": False,
            "responses_previous_response_id": False,
            "fallback_json_actions": False,
            "reasoning_effort": None,
            "verbosity": None,
        },
    }
    llm["config_sha256"] = _safe_llm_config_digest(llm)
    snapshot = {"schema_version": 1, "source": source, "llm": llm}
    return {
        "schema_version": 1,
        "start": snapshot,
        "end": json.loads(json.dumps(snapshot)),
        "stable": True,
    }
