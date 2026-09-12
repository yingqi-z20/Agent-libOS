from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_libos.llm.provider_trace import (
    PROVIDER_TRACE_MAX_BYTES,
    PROVIDER_TRACE_TEXT_MAX_CHARS,
    ProviderTraceBuilder,
    custom_provider_trace,
    project_provider_tools,
    provider_tools_summary,
)
from agent_libos.utils.serde import dumps


def _evidence() -> dict:
    return {
        "provider": "openai",
        "configured": ["web_search", "code_interpreter"],
        "effective": ["web_search", "code_interpreter"],
        "observed": "returned",
        "replay": "stateless",
        "file_count": 1,
        "activities": [
            {
                "type": "code_interpreter_call",
                "id": "activity-1",
                "status": "completed",
                "code": "PRIVATE_CODE",
                "outputs": [{"type": "logs", "text": "PRIVATE_RESULT"}],
            }
        ],
        "citations": [
            {"type": "url_citation", "url": "https://example.com/PRIVATE_QUERY", "title": "PRIVATE_TITLE"}
        ],
        "artifacts": [{"type": "file", "file_id": "PRIVATE_FILE", "filename": "result.txt"}],
        "usage": {"code_interpreter": {"calls": 1}},
    }


def test_managed_evidence_stays_separate_from_local_tool_calls() -> None:
    builder = ProviderTraceBuilder()
    sequence = builder.start_attempt(api="responses", kind="initial", provider_tools=_evidence())
    builder.finish_response(sequence, SimpleNamespace(usage={}))
    builder.enrich_response(
        sequence, reasoning=None, output="answer", tool_calls=[], usage={},
        provider_tools=_evidence(),
    )
    attempt = builder.to_dict()["attempts"][0]
    assert attempt["tool_calls"] == []
    assert attempt["usage"] == {}
    assert attempt["provider_tools"]["usage"] == {"code_interpreter": {"calls": 1}}
    summary = provider_tools_summary(attempt["provider_tools"])
    assert summary["activity_count"] == summary["citation_count"] == summary["artifact_count"] == 1
    assert summary["replay"] == "stateless"
    assert summary["file_count"] == 1
    assert "PRIVATE" not in dumps(summary)


def test_configured_search_without_execution_evidence_is_unknown() -> None:
    builder = ProviderTraceBuilder()
    sequence = builder.start_attempt(
        api="chat", kind="initial",
        provider_tools={"provider": "aliyun", "configured": ["web_search"], "effective": ["web_search"]},
    )
    builder.finish_response(sequence, SimpleNamespace(usage={}))
    value = builder.to_dict()["attempts"][0]["provider_tools"]
    assert value["observed"] == "unknown"
    assert value["usage"] is None
    assert value["activities"] == []
    suppressed = project_provider_tools({**value, "effective": [], "observed": "not_returned"})
    assert suppressed["configured"] == ["web_search"]
    assert suppressed["effective"] == []
    assert suppressed["observed"] == "not_returned"


def test_extractor_output_and_file_reference_coordinates_are_preserved() -> None:
    activity = {
        "type": "web_extractor_call", "id": "extractor-1", "status": "completed",
        "goal": "PRIVATE_GOAL", "urls": ["https://example.com/PRIVATE_QUERY"],
        "output": "PRIVATE_EXTRACTED_OUTPUT",
    }
    artifact = {
        "type": "file_citation", "file_id": "PRIVATE_FILE", "container_id": "PRIVATE_CONTAINER",
        "index": 3,
    }
    projected = project_provider_tools({**_evidence(), "activities": [activity], "artifacts": [artifact]})
    assert projected["activities"] == [activity]
    assert projected["artifacts"] == [artifact]
    assert "PRIVATE" not in dumps(provider_tools_summary(projected))


def test_rejected_response_removes_all_managed_result_bodies() -> None:
    builder = ProviderTraceBuilder()
    sequence = builder.start_attempt(api="responses", kind="initial", provider_tools=_evidence())
    builder.reject_response(sequence, ValueError("PRIVATE_ERROR"))
    attempt = builder.to_dict()["attempts"][0]
    assert "PRIVATE" not in dumps(attempt)
    assert attempt["provider_tools"]["configured"] == ["code_interpreter", "web_search"]
    assert attempt["provider_tools"]["observed"] == "unknown"
    assert attempt["provider_tools"]["usage"] is None


def test_managed_projection_redacts_credentials_and_bounds_text_structure_and_counts() -> None:
    evidence = _evidence()
    evidence["activities"][0]["action"] = {
        "authorization": "PRIVATE_AUTH",
        "headers": [{"name": "api-key", "value": "PRIVATE_API_KEY"}],
        "query": "allowed query",
    }
    evidence["activities"][0]["code"] = "x" * (PROVIDER_TRACE_TEXT_MAX_CHARS + 1)
    evidence["citations"] *= 300
    evidence["artifacts"][0]["unknown_secret"] = "PRIVATE_UNKNOWN"
    projected = project_provider_tools(evidence)
    encoded = dumps(projected)
    assert projected["limited"] is True
    assert len(projected["citations"]) == 256
    assert projected["activities"][0]["code"]["type"] == "omitted"
    assert "allowed query" in encoded
    for private in ("PRIVATE_AUTH", "PRIVATE_API_KEY", "PRIVATE_UNKNOWN"):
        assert private not in encoded
    assert len(encoded.encode()) <= PROVIDER_TRACE_MAX_BYTES


def test_managed_evidence_obeys_total_trace_byte_limit() -> None:
    builder = ProviderTraceBuilder()
    for _ in range(30):
        evidence = _evidence()
        evidence["activities"][0]["code"] = "x" * PROVIDER_TRACE_TEXT_MAX_CHARS
        builder.start_attempt(api="responses", kind="initial", provider_tools=evidence)
    value = builder.to_dict()
    assert value["limited"] is True
    assert len(dumps(value).encode()) <= PROVIDER_TRACE_MAX_BYTES


@pytest.mark.parametrize("value", [None, [], "invalid", 1])
def test_legacy_absent_managed_evidence_remains_absent(value: object) -> None:
    assert project_provider_tools(value) is None
    assert provider_tools_summary(value) is None
    builder = ProviderTraceBuilder()
    builder.start_attempt(api="chat", kind="initial")
    assert "provider_tools" not in builder.to_dict()["attempts"][0]


def test_projection_rejects_untrusted_metadata_shapes_without_throwing() -> None:
    projected = project_provider_tools({
        "provider": {}, "replay": [], "observed": {},
        "configured": [{}, "web_search", "not_supported"],
        "file_count": True, "activities": [None],
    })
    assert projected["provider"] is None
    assert projected["configured"] == ["web_search"]
    assert projected["limited"] is True
    assert "file_count" not in projected


def test_custom_completion_preserves_managed_evidence() -> None:
    evidence = _evidence()
    completion = SimpleNamespace(
        content="answer", tool_calls=[], reasoning=None, usage={},
        provider_request_options={"provider_tools": evidence},
        provider_tool_activities=evidence["activities"],
        citations=evidence["citations"], artifacts=evidence["artifacts"],
    )
    trace = custom_provider_trace(completion)
    assert trace["coverage"] == "custom_client_incomplete"
    assert trace["attempts"][0]["provider_tools"]["activities"][0]["code"] == "PRIVATE_CODE"
