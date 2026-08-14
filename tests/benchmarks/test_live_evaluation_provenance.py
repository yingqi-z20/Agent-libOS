from __future__ import annotations

import copy
import json

import pytest

from agent_libos.config import DEFAULT_CONFIG
from benchmarks.live_evaluation_provenance import (
    _safe_llm_config_digest,
    build_evaluation_provenance,
    capture_evaluation_provenance,
    evaluation_provenance_identity,
    live_evaluation_provenance_ready,
    valid_evaluation_provenance,
)
from tests.support.live_evaluation import stable_evaluation_provenance


def test_capture_records_effective_config_without_provider_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "OPENAI_API_KEY": "credential-canary-never-report",
        "OPENAI_BASE_URL": "https://gateway.example/v1?token=endpoint-canary",
        "OPENAI_LANGUAGE_MODEL": "publication-model",
        "OPENAI_PROMPT_CACHE_KEY": "cache-key-canary",
        "OPENAI_SAFETY_IDENTIFIER": "safety-id-canary",
        "OPENAI_ORGANIZATION": "organization-canary",
        "OPENAI_PROJECT": "project-canary",
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL", "true")

    snapshot = capture_evaluation_provenance(DEFAULT_CONFIG)
    provenance = build_evaluation_provenance(snapshot, copy.deepcopy(snapshot))
    rendered = json.dumps(provenance, sort_keys=True)

    assert valid_evaluation_provenance(provenance) is True
    assert live_evaluation_provenance_ready(provenance) is True
    identity = evaluation_provenance_identity(provenance)
    assert identity is not None
    assert identity["llm"]["model"] == "publication-model"
    assert identity["llm"]["endpoint"]["classification"] == "custom"
    assert identity["llm"]["endpoint"]["custom_endpoint_allowed"] is True
    assert identity["llm"]["credential_present"] is True
    assert identity["llm"]["prompt"]["cache_key_configured"] is True
    assert "gateway.example" not in rendered
    for secret in secrets.values():
        if secret == "publication-model":
            continue
        assert secret not in rendered


def test_custom_endpoint_authorization_is_part_of_config_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_LANGUAGE_MODEL", "publication-model")
    monkeypatch.setenv("OPENAI_API_KEY", "credential-canary")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.delenv("AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL", raising=False)
    denied = capture_evaluation_provenance(DEFAULT_CONFIG)

    monkeypatch.setenv("AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL", "true")
    allowed = capture_evaluation_provenance(DEFAULT_CONFIG)

    assert denied["llm"]["endpoint"]["custom_endpoint_allowed"] is False
    assert allowed["llm"]["endpoint"]["custom_endpoint_allowed"] is True
    assert denied["llm"]["config_sha256"] != allowed["llm"]["config_sha256"]


def test_validator_recomputes_config_digest_and_rejects_unknown_fields() -> None:
    provenance = stable_evaluation_provenance()
    provenance["start"]["llm"]["request"]["max_retries"] = 9
    provenance["end"] = copy.deepcopy(provenance["start"])

    assert valid_evaluation_provenance(provenance) is False

    provenance = stable_evaluation_provenance()
    provenance["start"]["llm"]["base_url"] = "https://must-not-survive.test"
    provenance["end"] = copy.deepcopy(provenance["start"])

    assert valid_evaluation_provenance(provenance) is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model", None),
        ("credential_present", False),
    ),
)
def test_live_gate_requires_exact_model_and_present_credential(
    field: str,
    value: object,
) -> None:
    provenance = stable_evaluation_provenance()
    for boundary in ("start", "end"):
        llm = provenance[boundary]["llm"]
        llm[field] = value
        safe = dict(llm)
        safe.pop("config_sha256")
        llm["config_sha256"] = _safe_llm_config_digest(safe)

    assert valid_evaluation_provenance(provenance) is True
    assert live_evaluation_provenance_ready(provenance) is False
