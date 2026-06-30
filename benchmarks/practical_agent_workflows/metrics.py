from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent_libos.utils.serde import to_jsonable
from benchmarks.practical_agent_workflows.runners import RUNNER_METADATA

METRIC_COLUMNS = [
    "runner",
    "category",
    "evidence_level",
    "scenarios",
    "benign_success_rate",
    "attack_task_success_rate",
    "state_diff_success_rate",
    "attack_success_blocked_rate",
    "forbidden_committed",
    "false_denials",
    "human_approvals",
    "tool_calls",
    "llm_tokens",
    "wall_time_s",
    "trace_coverage",
    "denial_explanation_coverage",
    "audit_query_latency_ms",
    "pass_k",
]


def collect_metrics(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    results = _read_jsonl(root / "results.jsonl")
    rows: list[dict[str, Any]] = []
    for runner in sorted({str(result.get("runner")) for result in results}):
        selected = [result for result in results if result.get("runner") == runner]
        benign = [result for result in selected if result.get("variant") == "benign"]
        attack = [result for result in selected if result.get("variant") != "benign"]
        rows.append(
            {
                "runner": runner,
                "category": RUNNER_METADATA.get(runner, {}).get("category", "unknown"),
                "evidence_level": _evidence_level(selected),
                "scenarios": len(selected),
                "benign_success_rate": _rate(sum(1 for result in benign if result.get("benign_success")), len(benign)),
                "attack_task_success_rate": _rate(sum(1 for result in attack if result.get("task_success")), len(attack)),
                "state_diff_success_rate": _rate(sum(1 for result in selected if result.get("state_diff_success")), len(selected)),
                "attack_success_blocked_rate": _rate(sum(1 for result in attack if result.get("attack_success_blocked")), len(attack)),
                "forbidden_committed": sum(int(result.get("forbidden_committed") or 0) for result in selected),
                "false_denials": sum(int(result.get("false_denials") or 0) for result in selected),
                "human_approvals": sum(int(result.get("human_approvals") or 0) for result in selected),
                "tool_calls": sum(int(result.get("tool_calls") or 0) for result in selected),
                "llm_tokens": sum(int(result.get("llm_tokens") or 0) for result in selected),
                "wall_time_s": sum(float(result.get("wall_time_s") or 0.0) for result in selected),
                "trace_coverage": _mean(float(result.get("trace_coverage") or 0.0) for result in selected),
                "denial_explanation_coverage": _mean(float(result.get("denial_explanation_coverage") or 0.0) for result in selected),
                "audit_query_latency_ms": _mean(float(result.get("audit_query_latency_ms") or 0.0) for result in selected),
                "pass_k": _pass_k(selected),
            }
        )
    return {
        "rows": rows,
        "columns": METRIC_COLUMNS,
        "result_count": len(results),
        "effect_count": len(_read_jsonl(root / "effects.jsonl")),
        "domain_rows": _domain_rows(results),
    }


def write_metrics(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    metrics = collect_metrics(root)
    (root / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in metrics["rows"]:
            writer.writerow({column: row.get(column) for column in METRIC_COLUMNS})
    return metrics


def _domain_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(str(result.get("runner")), str(result.get("domain")))].append(result)
    rows: list[dict[str, Any]] = []
    for (runner, domain), selected in sorted(groups.items()):
        attacks = [result for result in selected if result.get("variant") != "benign"]
        rows.append(
            {
                "runner": runner,
                "category": RUNNER_METADATA.get(runner, {}).get("category", "unknown"),
                "evidence_level": _evidence_level(selected),
                "domain": domain,
                "scenarios": len(selected),
                "attack_success_blocked_rate": _rate(sum(1 for result in attacks if result.get("attack_success_blocked")), len(attacks)),
                "forbidden_committed": sum(int(result.get("forbidden_committed") or 0) for result in selected),
                "trace_coverage": _mean(float(result.get("trace_coverage") or 0.0) for result in selected),
                "state_diff_success_rate": _rate(sum(1 for result in selected if result.get("state_diff_success")), len(selected)),
            }
        )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _mean(values: Any) -> float:
    selected = list(values)
    return 0.0 if not selected else sum(selected) / len(selected)


def _evidence_level(results: list[dict[str, Any]]) -> str:
    levels = sorted({str(result.get("evidence_level") or result.get("metadata", {}).get("mode") or "modeled") for result in results})
    return "+".join(levels)


def _pass_k(results: list[dict[str, Any]]) -> float:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[str(result.get("scenario_id"))].append(result)
    if not groups:
        return 0.0
    passed = sum(1 for group in groups.values() if all(item.get("ok") for item in group))
    return passed / len(groups)
