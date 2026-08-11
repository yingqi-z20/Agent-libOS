from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from agent_libos.llm.prompt_cache_gate import (
    PromptCachePricing,
    calculate_prompt_cache_cost,
)
from benchmarks.prompt_cache_evidence import aggregate_prompt_cache_run_evidence


@dataclass(frozen=True)
class ProviderPromptCacheArmInput:
    provider_id: str
    model_id: str
    repetitions: int
    report: Mapping[str, Any]
    pricing: PromptCachePricing | None = None


def build_prompt_cache_arm_report(
    providers: list[ProviderPromptCacheArmInput],
    *,
    security_invariants_passed: bool,
) -> dict[str, Any]:
    """Build one redacted v1/v2 arm from per-provider workflow reports."""

    if not providers:
        raise ValueError("at least one provider report is required")
    if type(security_invariants_passed) is not bool:
        raise ValueError("security_invariants_passed must be boolean")
    pair_ids = {(item.provider_id.strip(), item.model_id.strip()) for item in providers}
    if (
        any(not provider_id or not model_id for provider_id, model_id in pair_ids)
        or len(pair_ids) != len(providers)
    ):
        raise ValueError("provider/model pairs must be non-empty and unique")
    layouts = {str(item.report.get("prompt_layout") or "") for item in providers}
    if len(layouts) != 1 or "" in layouts:
        raise ValueError("provider reports must use one explicit prompt_layout")
    prompt_layout = next(iter(layouts))

    provider_rows = [_provider_row(item) for item in providers]
    metrics = aggregate_prompt_cache_run_evidence(
        _report_metrics(item.report) for item in providers
    )
    runs = sum(int(row["workflow_count"]) for row in provider_rows)
    successful_runs = sum(int(row["successful_tasks"]) for row in provider_rows)
    metrics.update(
        {
            "runs": runs,
            "successful_runs": successful_runs,
            "success_rate": successful_runs / runs if runs else 0.0,
        }
    )
    all_prices_known = all(item.pricing is not None for item in providers)
    report: dict[str, Any] = {
        "schema_version": 1,
        "prompt_layout": prompt_layout,
        "repetitions": min(item.repetitions for item in providers),
        "providers": provider_rows,
        "pricing_known": all_prices_known,
        "metrics": metrics,
        "release_gates": {
            "all_oracles_passed": all(
                row["all_oracles_passed"] is True for row in provider_rows
            ),
            "completion_evidence_passed": all(
                row["completion_evidence_passed"] is True
                for row in provider_rows
            ),
            "security_invariants_passed": security_invariants_passed,
            "workflow_count": runs,
        },
    }
    if all_prices_known:
        costs = [row["cost"] for row in provider_rows]
        net_cost = sum(float(cost["net_cost"]) for cost in costs)
        report["cost"] = {
            "net_cost": net_cost,
            "net_cost_per_successful_task": (
                net_cost / successful_runs if successful_runs else 0.0
            ),
        }
    return report


def _provider_row(item: ProviderPromptCacheArmInput) -> dict[str, Any]:
    if (
        isinstance(item.repetitions, bool)
        or not isinstance(item.repetitions, int)
        or item.repetitions < 1
    ):
        raise ValueError("provider repetitions must be a positive integer")
    metrics = _report_metrics(item.report)
    workflow_count = _nonnegative_int(metrics.get("runs"))
    if workflow_count < 1:
        raise ValueError("provider report must contain at least one workflow")
    successful_tasks = _successful_tasks(metrics)
    completion_successes = _nonnegative_int(
        metrics.get("completion_evidence_successful_runs")
    )
    row: dict[str, Any] = {
        "provider_id": item.provider_id.strip(),
        "model_id": item.model_id.strip(),
        "repetitions": item.repetitions,
        "workflow_count": workflow_count,
        "successful_tasks": successful_tasks,
        "all_oracles_passed": (
            successful_tasks == workflow_count and _report_gate_passed(item.report)
        ),
        "completion_evidence_passed": completion_successes == workflow_count,
        "forbidden_internal_id_leaks": _nonnegative_int(
            metrics.get("forbidden_internal_id_leaks")
        ),
        "pricing_known": item.pricing is not None,
    }
    if item.pricing is not None:
        row["pricing"] = asdict(item.pricing)
        row["cost"] = calculate_prompt_cache_cost(
            metrics,
            successful_tasks=successful_tasks,
            pricing=item.pricing,
        )
    return row


def _report_metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("provider report must contain a metrics object")
    return metrics


def _successful_tasks(metrics: Mapping[str, Any]) -> int:
    selected = metrics.get("successful_runs")
    if isinstance(selected, int) and not isinstance(selected, bool) and selected >= 0:
        return selected
    safety = _nonnegative_int(metrics.get("safety_successful_runs"))
    utility = _nonnegative_int(metrics.get("utility_successful_runs"))
    return min(safety, utility)


def _report_gate_passed(report: Mapping[str, Any]) -> bool:
    gate = report.get("release_gate")
    if not isinstance(gate, Mapping):
        return False
    return gate.get("passed") is True


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = ["ProviderPromptCacheArmInput", "build_prompt_cache_arm_report"]
