from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent_libos.utils.serde import to_jsonable
from benchmarks.practical_agent_workflows.loader import load_scenarios
from benchmarks.practical_agent_workflows.metrics import write_metrics
from benchmarks.practical_agent_workflows.reports import write_reports
from benchmarks.practical_agent_workflows.runners import RUNNER_NAMES, run_suite, write_run_outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run practical end-to-end agent workflow scenarios.")
    parser.add_argument("--suite", default="benchmarks/practical_agent_workflows", help="Practical workflow suite root.")
    parser.add_argument("--runner", action="append", default=[], help="Runner name, repeated; use 'all' for every runner.")
    parser.add_argument("--scenario", action="append", default=[], help="Scenario id to include, repeated.")
    parser.add_argument("--domain", action="append", default=[], help="Domain to include, repeated.")
    parser.add_argument("--variant", action="append", default=[], help="Variant to include, repeated.")
    parser.add_argument("--limit", type=int, help="Maximum number of scenarios after filtering.")
    parser.add_argument("--mode", choices=["deterministic", "replay", "real"], default="deterministic")
    parser.add_argument("--replay-trace", help="Replay trace JSONL produced by a prior real or deterministic run.")
    parser.add_argument("--allow-token-spend", action="store_true", help="Required for --mode real.")
    parser.add_argument("--env-file", default=".env", help="Dotenv file for real mode; values are not printed.")
    parser.add_argument("--output", default=".benchmark_runs/practical", help="Output run directory.")
    args = parser.parse_args(argv)

    scenarios = load_scenarios(args.suite)
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [scenario for scenario in scenarios if scenario.id in wanted]
    if args.domain:
        wanted_domains = set(args.domain)
        scenarios = [scenario for scenario in scenarios if scenario.domain in wanted_domains]
    if args.variant:
        wanted_variants = set(args.variant)
        scenarios = [scenario for scenario in scenarios if scenario.variant in wanted_variants]
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        raise SystemExit("no practical workflow scenarios selected")
    runners = _selected_runners(args.runner)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "suite": args.suite,
        "scenarios": [scenario.id for scenario in scenarios],
        "runners": runners,
        "mode": args.mode,
        "pid": os.getpid(),
    }
    (output / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    runs = run_suite(
        scenarios,
        output,
        runners=runners,
        mode=args.mode,
        replay_trace=args.replay_trace,
        allow_token_spend=args.allow_token_spend,
        env_file=args.env_file,
    )
    write_run_outputs(runs, output)
    metrics = write_metrics(output)
    reports = {name: str(path) for name, path in write_reports(output).items()}
    print(
        json.dumps(
            to_jsonable({"output": str(output), "results": len(runs), "metrics": metrics, "reports": reports}),
            indent=2,
            ensure_ascii=False,
        )
    )


def _selected_runners(values: list[str]) -> list[str]:
    if not values:
        return ["agent_libos"]
    selected: list[str] = []
    for value in values:
        if value == "all":
            selected.extend(RUNNER_NAMES)
            continue
        if value not in RUNNER_NAMES:
            raise SystemExit(f"unknown runner {value!r}; choose one of {list(RUNNER_NAMES)} or 'all'")
        selected.append(value)
    return list(dict.fromkeys(selected))


if __name__ == "__main__":
    main()
