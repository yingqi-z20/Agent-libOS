from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_libos.llm.context_management import (
    ContextPressureAssessment,
    assess_context_pressure,
    context_management_policy,
    context_pressure_prompt,
    estimate_multilingual_tokens,
    estimate_request_input_tokens,
    provider_usage_lower_bound,
)
from agent_libos.models.exceptions import ValidationError


def test_pressure_triggers_at_exact_threshold_with_output_reservation() -> None:
    messages = [{"role": "user", "content": "hello"}]
    local = estimate_request_input_tokens(messages, [])
    reserved = 800 - local

    at_boundary = assess_context_pressure(
        messages=messages,
        tools=[],
        context_window_tokens=1_000,
        reserved_output_tokens=reserved,
        threshold_ratio=0.8,
        profile_id="default",
        context_generation="generation-1",
    )
    below_boundary = assess_context_pressure(
        messages=messages,
        tools=[],
        context_window_tokens=1_000,
        reserved_output_tokens=reserved - 1,
        threshold_ratio=0.8,
        profile_id="default",
        context_generation="generation-1",
    )

    assert at_boundary.projected_tokens == 800
    assert at_boundary.triggered is True
    assert below_boundary.triggered is False


def test_multilingual_estimator_counts_non_ascii_conservatively() -> None:
    assert estimate_multilingual_tokens("你好世界") == 4
    assert estimate_multilingual_tokens("abcdefghijkl") == 4
    assert estimate_multilingual_tokens("你好 abc") >= 4


def test_request_estimator_includes_tool_outputs_and_openai_schemas() -> None:
    messages = [{"role": "user", "content": "continue"}]
    base = estimate_request_input_tokens(messages, [])
    with_tool_output = estimate_request_input_tokens(
        [
            {"role": "tool", "tool_call_id": "call-1", "content": "restored result"},
            *messages,
        ],
        [],
    )
    with_schema = estimate_request_input_tokens(
        messages,
        [
            {
                "type": "function",
                "function": {
                    "name": "example_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
    )

    assert with_tool_output > base
    assert with_schema > base


def test_provider_usage_is_ignored_without_an_active_responses_chain() -> None:
    call = SimpleNamespace(
        api="chat",
        response_id="response-1",
        request_options={
            "llm_profile_id": "coding",
            "llm_context_generation": "generation-1",
        },
        usage={"prompt_tokens": 100, "input_tokens": 900, "total_tokens": 1_000},
    )

    assert provider_usage_lower_bound(
        call,
        profile_id="coding",
        context_generation="generation-1",
        previous_response_id=None,
    ) == 0
    call.api = "responses"
    assert provider_usage_lower_bound(
        call,
        profile_id="coding",
        context_generation="generation-1",
        previous_response_id=None,
    ) == 0
    assert provider_usage_lower_bound(
        call,
        profile_id="coding",
        context_generation="generation-2",
        previous_response_id=None,
    ) == 0
    assert provider_usage_lower_bound(
        call,
        profile_id="review",
        context_generation="generation-1",
        previous_response_id=None,
    ) == 0


def test_responses_chain_uses_previous_total_usage_as_lower_bound() -> None:
    call = SimpleNamespace(
        api="responses",
        response_id="response-1",
        request_options={
            "llm_profile_id": "coding",
            "llm_context_generation": "generation-1",
        },
        usage={"input_tokens": 600, "total_tokens": 950},
    )

    assert provider_usage_lower_bound(
        call,
        profile_id="coding",
        context_generation="generation-1",
        previous_response_id="response-1",
    ) == 950


def test_responses_chain_adds_retained_history_to_new_request() -> None:
    messages = [{"role": "user", "content": "new chained input"}]
    local = estimate_request_input_tokens(messages, [])

    assessment = assess_context_pressure(
        messages=messages,
        tools=[],
        context_window_tokens=2_000,
        reserved_output_tokens=100,
        threshold_ratio=0.8,
        profile_id="coding",
        context_generation="generation-1",
        provider_lower_bound_tokens=950,
    )

    assert assessment.local_input_estimate_tokens == local
    assert assessment.provider_usage_lower_bound_tokens == 950
    assert assessment.estimated_input_tokens == local + 950
    assert assessment.projected_tokens == local + 1_050


def test_context_management_defaults_to_builtin_auto_compaction() -> None:
    policy = context_management_policy({"unrelated_planner_key": True})

    assert policy.mode == "auto_compact"
    assert policy.threshold_ratio == 0.8
    assert policy.tool_action() == {"action": "compact_process_context"}


@pytest.mark.parametrize(
    "context_management,match",
    [
        ({"unknown": True}, "unknown"),
        ({"mode": "ambient"}, "mode"),
        ({"threshold_ratio": 0}, "threshold_ratio"),
        ({"threshold_ratio": 1.1}, "threshold_ratio"),
        ({"tool": {"name": "not a tool"}}, "tool.name"),
        ({"tool": {"arguments": []}}, "tool.arguments"),
        ({"tool": {"arguments": {"action": "other_tool"}}}, "reserved action"),
        ({"tool": {"arguments": {"raw": b"not-json"}}}, "JSON-serializable"),
        ({"tool": {"extra": True}}, "tool fields"),
        ({1: "not-a-field"}, "field names must be strings"),
        ({"tool": {1: "not-a-field"}}, "tool field names must be strings"),
    ],
)
def test_context_management_rejects_invalid_nested_policy(
    context_management: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        context_management_policy({"context_management": context_management})


def test_prompt_is_literal_and_contains_pressure_diagnostics() -> None:
    policy = context_management_policy(
        {
            "context_management": {
                "mode": "prompt",
                "prompt": "Keep {projected_tokens} literally.",
            }
        }
    )
    assessment = ContextPressureAssessment(
        context_window_tokens=100,
        local_input_estimate_tokens=70,
        provider_usage_lower_bound_tokens=0,
        estimated_input_tokens=70,
        reserved_output_tokens=10,
        projected_tokens=80,
        utilization_ratio=0.8,
        threshold_ratio=0.8,
        triggered=True,
        profile_id="default",
        context_generation="initial",
    )

    prompt = context_pressure_prompt(policy, assessment)

    assert "{projected_tokens}" in prompt
    assert "100 tokens" in prompt
    assert "70 tokens" in prompt
    assert "10 tokens" in prompt
    assert "80 tokens" in prompt
    assert "80.00%" in prompt
