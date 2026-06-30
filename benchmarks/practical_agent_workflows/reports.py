from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmarks.practical_agent_workflows.metrics import collect_metrics
from benchmarks.practical_agent_workflows.runners import RUNNER_METADATA


def write_reports(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir)
    metrics = collect_metrics(root)
    results = _read_jsonl(root / "results.jsonl")
    effects = _read_jsonl(root / "effects.jsonl")
    audit = _read_jsonl(root / "audit_trace.jsonl")
    paths = {
        "summary": root / "practical_eval_summary.md",
        "live_runtime": root / "live_runtime_summary.md",
        "case_studies": root / "case_studies.md",
        "failure_taxonomy": root / "failure_taxonomy.md",
    }
    paths["summary"].write_text(_summary_report(metrics), encoding="utf-8")
    paths["live_runtime"].write_text(_live_runtime_summary(results, effects, audit), encoding="utf-8")
    paths["case_studies"].write_text(_case_studies(results, effects, audit), encoding="utf-8")
    paths["failure_taxonomy"].write_text(_failure_taxonomy(results, effects), encoding="utf-8")
    return paths


def _summary_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Practical Agent Workflow Evaluation",
        "",
        "This report is generated from practical end-to-end scenarios, not the legacy runtime-safety microbenchmarks.",
        "",
        "Runner categories are separated because external baselines test competing deployment patterns, while ablations test which Agent libOS mechanisms are necessary.",
        "",
        "## Primary System and External Baselines",
        "",
        "| System | Category | Evidence | Scenarios | Benign Success | State Diff | Attack Blocked | Forbidden Effects | False Denials | Human Approvals | Trace Coverage | pass^k |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _rows_for_categories(metrics["rows"], {"primary_system", "external_baseline"}):
        lines.append(_metric_row(row))
    lines.extend(
        [
            "",
            "## Agent LibOS Ablations",
            "",
            "| System | Category | Evidence | Scenarios | Benign Success | State Diff | Attack Blocked | Forbidden Effects | False Denials | Human Approvals | Trace Coverage | pass^k |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _rows_for_categories(metrics["rows"], {"ablation"}):
        lines.append(_metric_row(row))
    lines.extend(
        [
            "",
            "## Runner Semantics",
            "",
            "| Runner | Category | Interpretation |",
            "|---|---|---|",
        ]
    )
    for runner, meta in RUNNER_METADATA.items():
        lines.append(f"| `{runner}` | {meta['category']} | {meta['claim']} |")
    lines.extend(["", "## Domain Breakdown", ""])
    lines.append("| System | Category | Evidence | Domain | Scenarios | State Diff | Attack Blocked | Forbidden Effects | Trace Coverage |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in metrics["domain_rows"]:
        lines.append(
            f"| {row['runner']} | {row.get('category', 'unknown')} | {row.get('evidence_level', 'modeled')} | {row['domain']} | {row['scenarios']} | "
            f"{float(row.get('state_diff_success_rate') or 0.0) * 100:.1f}% | "
            f"{float(row['attack_success_blocked_rate']) * 100:.1f}% | {row['forbidden_committed']} | "
            f"{float(row['trace_coverage']) * 100:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def _metric_row(row: dict[str, Any]) -> str:
    return (
        "| {runner} | {category} | {evidence} | {scenarios} | {benign:.1f}% | {state:.1f}% | {blocked:.1f}% | {forbidden} | "
        "{false_denials} | {human} | {trace:.1f}% | {pass_k:.1f}% |"
    ).format(
        runner=row["runner"],
        category=row.get("category", "unknown"),
        evidence=row.get("evidence_level", "modeled"),
        scenarios=row["scenarios"],
        benign=float(row["benign_success_rate"]) * 100,
        state=float(row.get("state_diff_success_rate") or 0.0) * 100,
        blocked=float(row["attack_success_blocked_rate"]) * 100,
        forbidden=row["forbidden_committed"],
        false_denials=row["false_denials"],
        human=row["human_approvals"],
        trace=float(row["trace_coverage"]) * 100,
        pass_k=float(row.get("pass_k") or 0.0) * 100,
    )


def _rows_for_categories(rows: list[dict[str, Any]], categories: set[str]) -> list[dict[str, Any]]:
    category_order = {"primary_system": 0, "external_baseline": 1, "ablation": 2}
    return sorted(
        [row for row in rows if row.get("category") in categories],
        key=lambda row: (category_order.get(str(row.get("category")), 99), str(row.get("runner"))),
    )


def _case_studies(results: list[dict[str, Any]], effects: list[dict[str, Any]], audit: list[dict[str, Any]]) -> str:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for effect in effects:
        by_scenario[str(effect.get("scenario_id"))].append(effect)
    audit_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit:
        audit_by_scenario[str(row.get("scenario_id"))].append(row)
    interesting = [
        result for result in results
        if result.get("runner") == "agent_libos_live" and result.get("variant") != "benign"
    ][:3]
    interesting.extend([
        result for result in results
        if result.get("runner") == "agent_libos" and result.get("variant") != "benign"
    ][:2])
    lines = ["# Practical Evaluation Case Studies", ""]
    for result in interesting:
        scenario_id = str(result["scenario_id"])
        lines.extend(
            [
                f"## {scenario_id}",
                "",
                f"- domain: {result.get('domain')}",
                f"- attack blocked: {result.get('attack_success_blocked')}",
                f"- forbidden committed: {result.get('forbidden_committed')}",
                f"- trace coverage: {float(result.get('trace_coverage') or 0.0) * 100:.1f}%",
                "",
                "Effects:",
            ]
        )
        for effect in by_scenario.get(scenario_id, [])[:8]:
            lines.append(
                f"- {effect.get('type')} classification={effect.get('classification')} "
                f"performed={effect.get('performed')} denied={effect.get('denied')} "
                f"target={effect.get('path') or effect.get('target') or effect.get('endpoint') or effect.get('operation')}"
            )
        lines.append("")
        lines.append("Audit sample:")
        for row in audit_by_scenario.get(scenario_id, [])[:4]:
            lines.append(f"- {row.get('action')} target={row.get('target')} decision={row.get('decision')}")
        lines.append("")
    return "\n".join(lines)


def _failure_taxonomy(results: list[dict[str, Any]], effects: list[dict[str, Any]]) -> str:
    counters = Counter()
    for result in results:
        if int(result.get("forbidden_committed") or 0):
            category = RUNNER_METADATA.get(str(result.get("runner")), {}).get("category", "unknown")
            if category == "ablation":
                counters["ablation_regression"] += 1
            elif category == "external_baseline":
                counters["baseline_compromise"] += 1
            else:
                counters["primary_system_violation"] += 1
        if int(result.get("false_denials") or 0):
            counters["false_denial"] += 1
        if int(result.get("unknown_effects") or 0):
            counters["oracle_unknown"] += 1
        if not result.get("task_success"):
            counters["planner_or_utility_failure"] += 1
        if not result.get("state_diff_success", True):
            counters["state_diff_failure"] += 1
        if float(result.get("trace_coverage") or 0.0) < 1.0:
            counters["trace_gap"] += 1
    requested_forbidden = sum(1 for effect in effects if effect.get("classification") == "forbidden" and effect.get("requested"))
    denied_forbidden = sum(1 for effect in effects if effect.get("classification") == "forbidden" and effect.get("denied"))
    committed_forbidden = sum(1 for effect in effects if effect.get("classification") == "forbidden" and effect.get("performed") and not effect.get("denied"))
    lines = [
        "# Practical Evaluation Failure Taxonomy",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for key in sorted(counters):
        lines.append(f"| {key} | {counters[key]} |")
    lines.extend(
        [
            "",
            f"Model/requested forbidden effect attempts: {requested_forbidden}",
            f"Runtime-denied forbidden effect attempts: {denied_forbidden}",
            f"Committed forbidden effects: {committed_forbidden}",
            "",
        ]
    )
    return "\n".join(lines)


def _live_runtime_summary(results: list[dict[str, Any]], effects: list[dict[str, Any]], audit: list[dict[str, Any]]) -> str:
    live = [result for result in results if result.get("runner") == "agent_libos_live"]
    if not live:
        return "# Live Runtime Summary\n\nNo `agent_libos_live` results were produced.\n"
    effect_count = sum(1 for effect in effects if effect.get("runner") == "agent_libos_live")
    audit_count = sum(1 for row in audit if row.get("runner") == "agent_libos_live")
    forbidden = sum(int(result.get("forbidden_committed") or 0) for result in live)
    trace = sum(float(result.get("trace_coverage") or 0.0) for result in live) / len(live)
    lines = [
        "# Live Runtime Summary",
        "",
        f"Scenarios: {len(live)}",
        f"Effects: {effect_count}",
        f"Audit rows linked to effects: {audit_count}",
        f"Forbidden committed effects: {forbidden}",
        f"Mean trace coverage: {trace * 100:.1f}%",
        "",
        "| Scenario | Track | Variant | OK | State Diff | Forbidden | Trace | DB |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in live:
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        lines.append(
            f"| {result.get('scenario_id')} | {metadata.get('track', result.get('domain'))} | {result.get('variant')} | "
            f"{result.get('ok')} | {result.get('state_diff_success')} | {result.get('forbidden_committed')} | "
            f"{float(result.get('trace_coverage') or 0.0) * 100:.1f}% | {metadata.get('db', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
