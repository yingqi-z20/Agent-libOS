from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent_libos.utils.serde import to_jsonable
from benchmarks.practical_agent_workflows.real_completion import (
    build_real_completion_scenarios,
    run_completion_suite,
    write_completion_outputs,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run free-tool real LLM practical completion scenarios.")
    parser.add_argument("--scenario-set", default="real_completion_8", choices=["real_completion_8"])
    parser.add_argument("--scenario", action="append", default=[], help="Scenario id to include, repeated.")
    parser.add_argument("--track", action="append", default=[], help="Track/domain to include, repeated.")
    parser.add_argument("--variant", action="append", default=[], help="Variant to include, repeated.")
    parser.add_argument("--limit", type=int, help="Maximum number of scenarios after filtering.")
    parser.add_argument("--mode", choices=["real", "deterministic", "replay"], default="real")
    parser.add_argument("--replay-trace", help="Replay trace JSONL produced by a prior completion run.")
    parser.add_argument("--allow-token-spend", action="store_true", help="Required for --mode real.")
    parser.add_argument("--env-file", default=".env", help="Dotenv file for real mode; values are not printed.")
    parser.add_argument("--max-quanta", type=int, default=8, help="Maximum process quanta per scenario.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeated runs per selected scenario.")
    parser.add_argument("--output", default=".benchmark_runs/practical_eval_v2_real_completion", help="Output run directory.")
    args = parser.parse_args(argv)

    scenarios = build_real_completion_scenarios()
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [scenario for scenario in scenarios if scenario.id in wanted]
    if args.track:
        wanted_tracks = set(args.track)
        scenarios = [scenario for scenario in scenarios if scenario.track in wanted_tracks]
    if args.variant:
        wanted_variants = set(args.variant)
        scenarios = [scenario for scenario in scenarios if scenario.variant in wanted_variants]
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        raise SystemExit("no real completion scenarios selected")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "scenario_set": args.scenario_set,
        "scenarios": [scenario.id for scenario in scenarios],
        "mode": args.mode,
        "max_quanta": args.max_quanta,
        "repeats": args.repeats,
        "pid": os.getpid(),
    }
    (output / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    runs = run_completion_suite(
        scenarios,
        output,
        mode=args.mode,
        allow_token_spend=args.allow_token_spend,
        env_file=args.env_file,
        replay_trace=args.replay_trace,
        max_quanta=args.max_quanta,
        repeats=args.repeats,
    )
    reports = {name: str(path) for name, path in write_completion_outputs(runs, output).items()}
    print(
        json.dumps(
            to_jsonable({"output": str(output), "results": len(runs), "reports": reports}),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
