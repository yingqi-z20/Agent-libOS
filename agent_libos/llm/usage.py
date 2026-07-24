from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
)


def canonicalize_llm_usage(
    raw_usage: Any,
    *,
    api: str | None = None,
) -> tuple[dict[str, int], set[str]]:
    """Normalize billable and prompt-cache counters without coercing telemetry."""

    if not isinstance(raw_usage, Mapping):
        return {}, set()
    usage: dict[str, int] = {}
    invalid_fields: set[str] = set()
    for key in _TOKEN_FIELDS:
        if key not in raw_usage or raw_usage[key] is None:
            continue
        _store_counter(usage, invalid_fields, key, raw_usage[key])

    detail_maps = _usage_detail_maps(raw_usage, api=api)
    _store_cache_counter(
        usage,
        invalid_fields,
        normalized_key="cache_read_tokens",
        detail_maps=detail_maps,
        formal_key="cached_tokens",
        raw_usage=raw_usage,
        aliases=("cache_read_tokens", "cached_tokens", "cache_read_input_tokens"),
    )
    _store_cache_counter(
        usage,
        invalid_fields,
        normalized_key="cache_write_tokens",
        detail_maps=detail_maps,
        formal_key="cache_write_tokens",
        raw_usage=raw_usage,
        aliases=("cache_write_tokens", "cache_creation_input_tokens"),
    )
    return usage, invalid_fields


def aggregate_cache_usage(records: Iterable[Any]) -> dict[str, Any]:
    """Aggregate persisted calls, reading retained raw usage only as a fallback."""

    cache_read_tokens = 0
    cache_write_tokens = 0
    cache_reported_calls = 0
    cache_metric_input_tokens = 0
    uncached_input_tokens = 0
    calculation_cache_read_tokens = 0
    calculation_calls = 0

    for record in records:
        api = _record_value(record, "api")
        persisted = _record_value(record, "usage")
        canonical, _invalid = canonicalize_llm_usage(persisted, api=api)
        raw_canonical: dict[str, int] = {}
        if not _has_all_cache_fields(canonical):
            raw_usage = _raw_response_usage(_record_value(record, "raw_response"))
            raw_canonical, _raw_invalid = canonicalize_llm_usage(raw_usage, api=api)
            for key in (
                *_TOKEN_FIELDS,
                "cache_read_tokens",
                "cache_write_tokens",
            ):
                if key not in canonical and key in raw_canonical:
                    canonical[key] = raw_canonical[key]

        read_reported = "cache_read_tokens" in canonical
        write_reported = "cache_write_tokens" in canonical
        if not (read_reported or write_reported):
            continue
        cache_reported_calls += 1
        cache_read_tokens += canonical.get("cache_read_tokens", 0)
        cache_write_tokens += canonical.get("cache_write_tokens", 0)

        input_tokens = _input_tokens(canonical, api=api)
        if not read_reported or input_tokens is None:
            continue
        calculation_calls += 1
        calculation_cache_read_tokens += min(
            canonical["cache_read_tokens"],
            input_tokens,
        )
        cache_metric_input_tokens += input_tokens
        uncached_input_tokens += max(
            input_tokens - canonical["cache_read_tokens"],
            0,
        )

    cache_hit_rate: float | None = None
    if calculation_calls and cache_metric_input_tokens > 0:
        cache_hit_rate = calculation_cache_read_tokens / cache_metric_input_tokens
    return {
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_reported_calls": cache_reported_calls,
        "cache_metric_input_tokens": cache_metric_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "cache_hit_rate": cache_hit_rate,
    }


def _usage_detail_maps(
    raw_usage: Mapping[str, Any],
    *,
    api: str | None,
) -> tuple[Mapping[str, Any], ...]:
    ordered_keys = (
        ("input_tokens_details", "prompt_tokens_details")
        if api == "responses"
        else ("prompt_tokens_details", "input_tokens_details")
    )
    details: list[Mapping[str, Any]] = []
    for key in ordered_keys:
        candidate = raw_usage.get(key)
        if isinstance(candidate, Mapping):
            details.append(candidate)
    return tuple(details)


def _store_cache_counter(
    usage: dict[str, int],
    invalid_fields: set[str],
    *,
    normalized_key: str,
    detail_maps: tuple[Mapping[str, Any], ...],
    formal_key: str,
    raw_usage: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> None:
    for details in detail_maps:
        if formal_key in details:
            value = details[formal_key]
            if value is not None:
                _store_counter(usage, invalid_fields, normalized_key, value)
            return
    for alias in aliases:
        if alias not in raw_usage:
            continue
        value = raw_usage[alias]
        if value is not None:
            _store_counter(usage, invalid_fields, normalized_key, value)
        return


def _store_counter(
    usage: dict[str, int],
    invalid_fields: set[str],
    key: str,
    value: Any,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        invalid_fields.add(key)
        return
    usage[key] = value


def _record_value(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _raw_response_usage(raw_response: Any) -> Any:
    if not isinstance(raw_response, Mapping):
        return None
    return raw_response.get("usage")


def _has_all_cache_fields(usage: Mapping[str, Any]) -> bool:
    return "cache_read_tokens" in usage and "cache_write_tokens" in usage


def _input_tokens(usage: Mapping[str, int], *, api: Any) -> int | None:
    ordered_keys = (
        ("input_tokens", "prompt_tokens")
        if api == "responses"
        else ("prompt_tokens", "input_tokens")
    )
    for key in ordered_keys:
        value = usage.get(key)
        if value is not None:
            return value
    return None
