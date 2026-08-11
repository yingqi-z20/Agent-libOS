from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptCacheGateThresholds:
    minimum_uncached_input_reduction: float = 0.20
    minimum_total_input_reduction: float = 0.10


@dataclass(frozen=True)
class PromptCachePricing:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    cache_write_input_per_million: float | None = None


def calculate_prompt_cache_cost(
    metrics: Mapping[str, Any],
    *,
    successful_tasks: int,
    pricing: PromptCachePricing,
) -> dict[str, float]:
    """Calculate provider cost while keeping unreported write tokens unknown."""

    if (
        isinstance(successful_tasks, bool)
        or not isinstance(successful_tasks, int)
        or successful_tasks < 1
    ):
        raise ValueError("successful_tasks must be a positive integer")
    rates = {
        "input": pricing.input_per_million,
        "cached_input": pricing.cached_input_per_million,
        "output": pricing.output_per_million,
    }
    if pricing.cache_write_input_per_million is not None:
        rates["cache_write_input"] = pricing.cache_write_input_per_million
    if not all(_is_nonnegative_number(value) for value in rates.values()):
        raise ValueError("prompt cache pricing rates must be non-negative")

    total_input = _nonnegative_metric(metrics, "total_input_tokens")
    cache_read = _nonnegative_metric(metrics, "cache_read_tokens")
    uncached = _nonnegative_metric(metrics, "uncached_input_tokens")
    output = _nonnegative_metric(metrics, "total_output_tokens")
    if cache_read > total_input or uncached > total_input:
        raise ValueError("cache token components cannot exceed total input tokens")
    if abs((cache_read + uncached) - total_input) > 1e-9:
        raise ValueError(
            "cache read and uncached tokens must reconcile to total input tokens"
        )
    write_value = metrics.get("cache_write_tokens")
    if pricing.cache_write_input_per_million is not None:
        if not _is_nonnegative_number(write_value):
            raise ValueError(
                "cache_write_tokens must be reported when write pricing differs"
            )
        cache_write = float(write_value)
        if cache_write > uncached:
            raise ValueError("cache write tokens cannot exceed uncached input tokens")
    else:
        cache_write = 0.0

    standard_input = max(uncached - cache_write, 0.0)
    input_cost = standard_input * pricing.input_per_million / 1_000_000
    cache_read_cost = cache_read * pricing.cached_input_per_million / 1_000_000
    cache_write_rate = (
        pricing.cache_write_input_per_million
        if pricing.cache_write_input_per_million is not None
        else pricing.input_per_million
    )
    cache_write_cost = cache_write * cache_write_rate / 1_000_000
    output_cost = output * pricing.output_per_million / 1_000_000
    net_cost = input_cost + cache_read_cost + cache_write_cost + output_cost
    return {
        "input_cost": input_cost,
        "cache_read_cost": cache_read_cost,
        "cache_write_cost": cache_write_cost,
        "output_cost": output_cost,
        "net_cost": net_cost,
        "net_cost_per_successful_task": net_cost / successful_tasks,
    }


def evaluate_prompt_cache_release_gate(
    legacy_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    thresholds: PromptCacheGateThresholds = PromptCacheGateThresholds(),
    strict_release_evidence: bool = True,
) -> dict[str, Any]:
    """Compare paired v1/v2 evidence without reading prompt plaintext."""

    legacy = _metrics(legacy_report)
    candidate = _metrics(candidate_report)
    legacy_uncached = _positive_metric(legacy, "uncached_input_tokens")
    candidate_uncached = _nonnegative_metric(candidate, "uncached_input_tokens")
    legacy_total = _total_input_tokens(legacy_report, legacy)
    candidate_total = _total_input_tokens(candidate_report, candidate)
    legacy_hit_rate = _rate_metric(legacy, "cache_hit_rate")
    candidate_hit_rate = _rate_metric(candidate, "cache_hit_rate")

    uncached_reduction = 1.0 - candidate_uncached / legacy_uncached
    total_reduction = 1.0 - candidate_total / legacy_total
    checks: dict[str, bool] = {
        "uncached_input_reduction": (
            uncached_reduction >= thresholds.minimum_uncached_input_reduction
        ),
        "total_input_reduction": (
            total_reduction >= thresholds.minimum_total_input_reduction
        ),
        "cache_hit_rate_not_lower": candidate_hit_rate >= legacy_hit_rate,
        "legacy_cache_telemetry_complete": _cache_telemetry_complete(
            legacy,
            total_input=legacy_total,
            hit_rate=legacy_hit_rate,
        ),
        "candidate_cache_telemetry_complete": _cache_telemetry_complete(
            candidate,
            total_input=candidate_total,
            hit_rate=candidate_hit_rate,
        ),
        "legacy_success_rate": _success_rate(legacy) == 1.0,
        "candidate_success_rate": _success_rate(candidate) == 1.0,
        "forbidden_internal_id_leaks": (
            _nonnegative_metric(candidate, "forbidden_internal_id_leaks")
            == 0
        ),
    }
    release_evidence = candidate_report.get("release_gates")
    if strict_release_evidence:
        checks.update(
            _release_evidence_checks(
                legacy_report,
                candidate_report,
                release_evidence,
            )
        )

    pricing_known = bool(candidate_report.get("pricing_known"))
    cost = candidate_report.get("cost")
    checks["known_price_cost_accounted"] = (
        not pricing_known
        or (
            isinstance(cost, Mapping)
            and _is_nonnegative_number(cost.get("net_cost"))
            and _is_nonnegative_number(cost.get("net_cost_per_successful_task"))
        )
    )
    if strict_release_evidence:
        checks["known_provider_prices_accounted"] = (
            _known_provider_prices_accounted(legacy_report)
            and _known_provider_prices_accounted(candidate_report)
        )
        checks["known_provider_cost_not_higher"] = (
            _known_provider_cost_not_higher(legacy_report, candidate_report)
        )
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "passed": passed,
        "checks": checks,
        "metrics": {
            "legacy_uncached_input_tokens": legacy_uncached,
            "candidate_uncached_input_tokens": candidate_uncached,
            "uncached_input_reduction": uncached_reduction,
            "legacy_total_input_tokens": legacy_total,
            "candidate_total_input_tokens": candidate_total,
            "total_input_reduction": total_reduction,
            "legacy_cache_hit_rate": legacy_hit_rate,
            "candidate_cache_hit_rate": candidate_hit_rate,
        },
    }


def _release_evidence_checks(
    legacy_report: Mapping[str, Any],
    report: Mapping[str, Any],
    value: Any,
) -> dict[str, bool]:
    gates = value if isinstance(value, Mapping) else {}
    provider_checks = _provider_evidence_checks(legacy_report, report)
    return {
        "all_oracles_passed": gates.get("all_oracles_passed") is True,
        "completion_evidence_passed": (
            gates.get("completion_evidence_passed") is True
        ),
        "security_invariants_passed": (
            gates.get("security_invariants_passed") is True
        ),
        **provider_checks,
        "workflow_coverage": _nonnegative_metric(
            gates,
            "workflow_count",
            default=0,
        )
        >= 6,
    }


def _provider_evidence_checks(
    legacy_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> dict[str, bool]:
    legacy = _provider_rows(legacy_report)
    candidate = _provider_rows(candidate_report)
    legacy_ids = {_provider_pair_id(row) for row in legacy}
    candidate_ids = {_provider_pair_id(row) for row in candidate}
    well_formed = (
        len(legacy) >= 2
        and len(candidate) >= 2
        and None not in legacy_ids
        and None not in candidate_ids
        and len(legacy_ids) == len(legacy)
        and len(candidate_ids) == len(candidate)
    )
    rows = (*legacy, *candidate)
    return {
        "provider_coverage": well_formed,
        "paired_provider_coverage": well_formed and legacy_ids == candidate_ids,
        "provider_repetitions": well_formed
        and all(_provider_integer(row, "repetitions") >= 3 for row in rows),
        "provider_workflow_coverage": well_formed
        and all(_provider_integer(row, "workflow_count") >= 6 for row in rows),
        "provider_oracles_passed": well_formed
        and all(row.get("all_oracles_passed") is True for row in candidate),
        "provider_completion_evidence_passed": well_formed
        and all(
            row.get("completion_evidence_passed") is True for row in candidate
        ),
        "provider_forbidden_internal_id_leaks": well_formed
        and all(
            _provider_integer(row, "forbidden_internal_id_leaks") == 0
            for row in candidate
        ),
    }


def _provider_rows(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    value = report.get("providers")
    if not isinstance(value, list):
        return ()
    return tuple(row for row in value if isinstance(row, Mapping))


def _provider_pair_id(row: Mapping[str, Any]) -> tuple[str, str] | None:
    provider_id = row.get("provider_id")
    model_id = row.get("model_id")
    if (
        not isinstance(provider_id, str)
        or not provider_id.strip()
        or not isinstance(model_id, str)
        or not model_id.strip()
    ):
        return None
    return provider_id.strip(), model_id.strip()


def _provider_integer(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return -1


def _known_provider_prices_accounted(report: Mapping[str, Any]) -> bool:
    for provider in _provider_rows(report):
        if provider.get("pricing_known") is not True:
            continue
        cost = provider.get("cost")
        if not (
            isinstance(cost, Mapping)
            and _is_nonnegative_number(cost.get("net_cost"))
            and _is_nonnegative_number(cost.get("net_cost_per_successful_task"))
        ):
            return False
    return True


def _known_provider_cost_not_higher(
    legacy_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> bool:
    legacy = {
        _provider_pair_id(row): row
        for row in _provider_rows(legacy_report)
        if _provider_pair_id(row) is not None
    }
    candidate = {
        _provider_pair_id(row): row
        for row in _provider_rows(candidate_report)
        if _provider_pair_id(row) is not None
    }
    for pair_id in legacy.keys() | candidate.keys():
        before = legacy.get(pair_id)
        after = candidate.get(pair_id)
        if before is None or after is None:
            return False
        if not (before.get("pricing_known") is True or after.get("pricing_known") is True):
            continue
        if not (before.get("pricing_known") is True and after.get("pricing_known") is True):
            return False
        before_cost = before.get("cost")
        after_cost = after.get("cost")
        if not isinstance(before_cost, Mapping) or not isinstance(after_cost, Mapping):
            return False
        before_per_task = before_cost.get("net_cost_per_successful_task")
        after_per_task = after_cost.get("net_cost_per_successful_task")
        if not (
            _is_nonnegative_number(before_per_task)
            and _is_nonnegative_number(after_per_task)
            and float(after_per_task) <= float(before_per_task)
        ):
            return False
    return True


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    value = report.get("metrics")
    if not isinstance(value, Mapping):
        raise ValueError("prompt cache report must contain a metrics object")
    return value


def _total_input_tokens(
    report: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> float:
    for key in (
        "total_input_tokens",
        "input_tokens",
        "prompt_tokens",
        "cache_metric_input_tokens",
    ):
        value = metrics.get(key)
        if _is_positive_number(value):
            return float(value)
    runs = metrics.get("runs", report.get("repetitions"))
    mean = metrics.get("mean_prompt_tokens")
    if _is_positive_number(runs) and _is_positive_number(mean):
        return float(runs) * float(mean)
    raise ValueError("prompt cache report has no positive total input token metric")


def _success_rate(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("success_rate")
    if _is_rate(value):
        return float(value)
    runs = metrics.get("runs")
    successes = metrics.get("successful_runs")
    if _is_positive_number(runs) and _is_nonnegative_number(successes):
        return float(successes) / float(runs)
    raise ValueError("prompt cache report has no valid task success metric")


def _cache_telemetry_complete(
    metrics: Mapping[str, Any],
    *,
    total_input: float,
    hit_rate: float,
) -> bool:
    """Require cache counters for every compared call and reconcile tokens."""

    total_calls = metrics.get("cache_total_calls")
    read_reported_calls = metrics.get("cache_read_reported_calls")
    metric_reported_calls = metrics.get("cache_metric_reported_calls")
    cache_metric_input = metrics.get("cache_metric_input_tokens")
    cache_read = metrics.get("cache_read_tokens")
    uncached = metrics.get("uncached_input_tokens")
    if not (
        _is_positive_number(total_calls)
        and _is_nonnegative_number(read_reported_calls)
        and float(read_reported_calls) == float(total_calls)
        and _is_nonnegative_number(metric_reported_calls)
        and float(metric_reported_calls) == float(total_calls)
        and _is_nonnegative_number(cache_metric_input)
        and _is_nonnegative_number(cache_read)
        and _is_nonnegative_number(uncached)
    ):
        return False
    return (
        abs(float(cache_metric_input) - total_input) <= 1e-9
        and abs((float(cache_read) + float(uncached)) - total_input) <= 1e-9
        and abs((float(cache_read) / total_input) - hit_rate) <= 1e-12
    )


def _positive_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not _is_positive_number(value):
        raise ValueError(f"prompt cache metric {key} must be positive")
    return float(value)


def _nonnegative_metric(
    metrics: Mapping[str, Any],
    key: str,
    *,
    default: float | None = None,
) -> float:
    value = metrics.get(key, default)
    if not _is_nonnegative_number(value):
        raise ValueError(f"prompt cache metric {key} must be non-negative")
    return float(value)


def _rate_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not _is_rate(value):
        raise ValueError(f"prompt cache metric {key} must be between zero and one")
    return float(value)


def _is_positive_number(value: Any) -> bool:
    return _is_nonnegative_number(value) and float(value) > 0


def _is_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) >= 0
    )


def _is_rate(value: Any) -> bool:
    return _is_nonnegative_number(value) and float(value) <= 1


__all__ = [
    "PromptCacheGateThresholds",
    "PromptCachePricing",
    "calculate_prompt_cache_cost",
    "evaluate_prompt_cache_release_gate",
]
