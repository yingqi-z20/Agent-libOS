from __future__ import annotations

from types import SimpleNamespace

from agent_libos.llm.usage import aggregate_cache_usage, canonicalize_llm_usage


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
        "cache_write_tokens": 10,
        "cache_reported_calls": 2,
        "cache_metric_input_tokens": 150,
        "uncached_input_tokens": 90,
        "cache_hit_rate": 0.4,
    }


def test_cache_hit_rate_is_null_when_provider_reports_no_cache_metrics() -> None:
    metrics = aggregate_cache_usage(
        [SimpleNamespace(api="chat", usage={"prompt_tokens": 100}, raw_response=None)]
    )

    assert metrics["cache_reported_calls"] == 0
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
    assert metrics["cache_reported_calls"] == 2
    assert metrics["cache_metric_input_tokens"] == 100
    assert metrics["uncached_input_tokens"] == 60
    assert metrics["cache_hit_rate"] == 0.4
