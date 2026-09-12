from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_libos.llm.usage import (
    LLM_USAGE_COUNTER_MAX,
    aggregate_cache_usage,
    canonicalize_llm_usage,
)


def test_chat_cache_usage_prefers_formal_details_and_preserves_zero() -> None:
    usage, invalid = canonicalize_llm_usage(
        {
            "prompt_tokens": 100,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 40,
            },
            "cached_tokens": 99,
            "cache_creation_input_tokens": 88,
        },
        api="chat",
    )

    assert invalid == set()
    assert usage == {
        "prompt_tokens": 100,
        "cache_read_tokens": 0,
        "cache_write_tokens": 40,
    }


def test_responses_cache_usage_and_compatibility_aliases_are_normalized() -> None:
    formal, formal_invalid = canonicalize_llm_usage(
        {
            "input_tokens": 120,
            "input_tokens_details": {"cached_tokens": 72},
            "cache_creation_input_tokens": 11,
        },
        api="responses",
    )
    compatible, compatible_invalid = canonicalize_llm_usage(
        {
            "prompt_tokens": 80,
            "cache_read_input_tokens": 48,
            "cache_creation_input_tokens": 16,
        },
        api="chat",
    )

    assert formal_invalid == set()
    assert formal == {
        "input_tokens": 120,
        "cache_read_tokens": 72,
        "cache_write_tokens": 11,
    }
    assert compatible_invalid == set()
    assert compatible == {
        "prompt_tokens": 80,
        "cache_read_tokens": 48,
        "cache_write_tokens": 16,
    }


def test_invalid_cache_telemetry_is_omitted_without_coercion() -> None:
    usage, invalid = canonicalize_llm_usage(
        {
            "prompt_tokens": 50,
            "prompt_tokens_details": {
                "cached_tokens": True,
                "cache_write_tokens": -1,
            },
            "cached_tokens": 20,
        },
        api="chat",
    )

    assert usage == {"prompt_tokens": 50}
    assert invalid == {"cache_read_tokens", "cache_write_tokens"}


def test_cache_aggregate_reads_retained_raw_usage_only_when_needed() -> None:
    calls = [
        SimpleNamespace(
            api="chat",
            usage={"prompt_tokens": 100},
            raw_response={
                "usage": {
                    "prompt_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 60},
                }
            },
        ),
        SimpleNamespace(
            api="chat",
            usage={
                "prompt_tokens": 50,
                "cache_read_tokens": 0,
                "cache_write_tokens": 10,
            },
            raw_response=None,
        ),
    ]

    assert aggregate_cache_usage(calls) == {
        "cache_read_tokens": 60,
        "cache_write_tokens": None,
        "cache_total_calls": 2,
        "cache_reported_calls": 2,
        "cache_read_reported_calls": 2,
        "cache_write_reported_calls": 1,
        "cache_metric_reported_calls": 2,
        "cache_metric_input_tokens": 150,
        "uncached_input_tokens": 90,
        "cache_hit_rate": 0.4,
    }


def test_cache_hit_rate_is_null_when_provider_reports_no_cache_metrics() -> None:
    metrics = aggregate_cache_usage(
        [SimpleNamespace(api="chat", usage={"prompt_tokens": 100}, raw_response=None)]
    )

    assert metrics["cache_reported_calls"] == 0
    assert metrics["cache_total_calls"] == 1
    assert metrics["cache_hit_rate"] is None


def test_cache_hit_rate_uses_only_calls_with_reported_input_tokens() -> None:
    metrics = aggregate_cache_usage(
        [
            SimpleNamespace(
                api="chat",
                usage={"cache_read_tokens": 90},
                raw_response=None,
            ),
            SimpleNamespace(
                api="chat",
                usage={"prompt_tokens": 100, "cache_read_tokens": 40},
                raw_response=None,
            ),
        ]
    )

    assert metrics["cache_read_tokens"] == 130
    assert metrics["cache_total_calls"] == 2
    assert metrics["cache_reported_calls"] == 2
    assert metrics["cache_metric_input_tokens"] == 100
    assert metrics["uncached_input_tokens"] == 60
    assert metrics["cache_hit_rate"] == 0.4


def test_usage_counters_reject_values_outside_finite_accounting_range() -> None:
    usage, invalid = canonicalize_llm_usage(
        {
            "prompt_tokens": LLM_USAGE_COUNTER_MAX,
            "completion_tokens": LLM_USAGE_COUNTER_MAX + 1,
            "total_tokens": 10**400,
        },
        api="chat",
    )

    assert usage == {"prompt_tokens": LLM_USAGE_COUNTER_MAX}
    assert invalid == {"completion_tokens", "total_tokens"}


@pytest.mark.parametrize(
    ("api", "output_key", "detail_key", "other_detail_key"),
    [
        ("responses", "output_tokens", "output_tokens_details", "completion_tokens_details"),
        ("chat", "completion_tokens", "completion_tokens_details", "output_tokens_details"),
    ],
)
def test_reasoning_usage_prefers_formal_api_details_without_changing_billable_usage(
    api: str, output_key: str, detail_key: str, other_detail_key: str,
) -> None:
    usage, invalid = canonicalize_llm_usage(
        {
            "input_tokens": 20,
            output_key: 10,
            "total_tokens": 30,
            detail_key: {"reasoning_tokens": 6},
            other_detail_key: {"reasoning_tokens": 7},
            "reasoning_tokens": 8,
        },
        api=api,
    )

    assert invalid == set()
    assert usage == {
        "input_tokens": 20,
        output_key: 10,
        "total_tokens": 30,
        "reasoning_tokens": 6,
    }
    assert canonicalize_llm_usage(usage, api=api) == (usage, set())


@pytest.mark.parametrize("api", ["responses", "chat", None])
def test_reasoning_usage_preserves_reported_zero_and_missing_counter(api: str | None) -> None:
    assert canonicalize_llm_usage({"reasoning_tokens": 0}, api=api) == (
        {"reasoning_tokens": 0}, set(),
    )
    assert canonicalize_llm_usage({"output_tokens": 0}, api=api) == (
        {"output_tokens": 0}, set(),
    )
    assert canonicalize_llm_usage({"reasoning_tokens": None}, api=api) == ({}, set())


@pytest.mark.parametrize(
    ("formal", "expected", "invalid"),
    [
        (0, {"reasoning_tokens": 0}, set()),
        (None, {}, set()),
        (False, {}, {"reasoning_tokens"}),
        (-1, {}, {"reasoning_tokens"}),
        (1.0, {}, {"reasoning_tokens"}),
        ("1", {}, {"reasoning_tokens"}),
        (float("inf"), {}, {"reasoning_tokens"}),
        (LLM_USAGE_COUNTER_MAX + 1, {}, {"reasoning_tokens"}),
    ],
)
def test_formal_reasoning_counter_never_falls_back_to_alias(
    formal: object, expected: dict[str, int], invalid: set[str],
) -> None:
    assert canonicalize_llm_usage(
        {
            "output_tokens_details": {"reasoning_tokens": formal},
            "completion_tokens_details": {"reasoning_tokens": 2},
            "reasoning_tokens": 3,
        },
        api="responses",
    ) == (expected, invalid)


@pytest.mark.parametrize("reasoning", [True, -1, 1.0, "1", LLM_USAGE_COUNTER_MAX + 1])
def test_retained_reasoning_counter_is_validated_without_coercion(reasoning: object) -> None:
    assert canonicalize_llm_usage({"reasoning_tokens": reasoning}) == (
        {}, {"reasoning_tokens"},
    )


@pytest.mark.parametrize(
    ("api", "output_key", "detail_key"),
    [
        ("responses", "output_tokens", "output_tokens_details"),
        ("chat", "completion_tokens", "completion_tokens_details"),
    ],
)
def test_reasoning_counter_cannot_exceed_output_token_subset(
    api: str, output_key: str, detail_key: str,
) -> None:
    usage, invalid = canonicalize_llm_usage(
        {
            output_key: 2,
            "total_tokens": 7,
            detail_key: {"reasoning_tokens": 3},
        },
        api=api,
    )

    assert usage == {output_key: 2, "total_tokens": 7}
    assert invalid == {"reasoning_tokens"}


def test_reasoning_counter_supports_compatible_details_and_safe_integer_boundary() -> None:
    assert canonicalize_llm_usage(
        {
            "output_tokens": LLM_USAGE_COUNTER_MAX,
            "completion_tokens_details": {"reasoning_tokens": LLM_USAGE_COUNTER_MAX},
        },
        api="responses",
    ) == (
        {"output_tokens": LLM_USAGE_COUNTER_MAX, "reasoning_tokens": LLM_USAGE_COUNTER_MAX},
        set(),
    )
