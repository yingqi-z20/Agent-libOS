from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from benchmarks.builtin_tool_skills import (
    EVALUATION_REPETITIONS,
    EVALUATION_VARIANTS,
    HELD_OUT_SCENARIOS,
    evaluation_pair_plan,
    report_all_correct,
    report_publication_ready,
    run_evaluation,
)
from experiments.evaluation_cli import has_real_llm_environment
from experiments.evaluation_output import AtomicJsonOutput


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the opt-in, 15-pair real-LLM evaluation comparing built-in "
            "Tool Skill routing with a no-Skills full-projection baseline."
        )
    )
    parser.add_argument(
        "--output",
        help="JSON report path (required unless --list-scenarios or --dry-run is used).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.scenario_id for scenario in HELD_OUT_SCENARIOS],
        help="Select a held-out scenario; repeat to select several. Defaults to all.",
    )
    parser.add_argument(
        "--confirm-real-llm",
        action="store_true",
        help="Acknowledge that this evaluation makes paid provider calls.",
    )
    parser.add_argument(
        "--require-all-correct",
        action="store_true",
        help=(
            "Exit non-zero unless both arms of every pair choose the correct route, "
            "return a successful probe result, satisfy the task-state oracle, and exit."
        ),
    )
    parser.add_argument(
        "--require-publication-gate",
        action="store_true",
        help=(
            "Exit non-zero unless this is a complete schema-v3 15-pair/30-run "
            "report with clean stable source/model provenance, counterbalanced "
            "order, and complete, decidable paired evidence."
        ),
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print scenario ids without making provider calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected paired evaluation plan without provider calls.",
    )
    args = parser.parse_args(argv)

    if args.list_scenarios:
        print("\n".join(scenario.scenario_id for scenario in HELD_OUT_SCENARIOS))
        return
    if args.dry_run:
        selected = [
            scenario
            for scenario in HELD_OUT_SCENARIOS
            if not args.scenario or scenario.scenario_id in set(args.scenario)
        ]
        print(
            json.dumps(
                {
                    "evaluation": "builtin_tool_skill_routing",
                    "real_llm_calls": False,
                    "repetitions_per_scenario": EVALUATION_REPETITIONS,
                    "variants": list(EVALUATION_VARIANTS),
                    "scenarios": [scenario.scenario_id for scenario in selected],
                    "pair_plan": evaluation_pair_plan(
                        scenario.scenario_id for scenario in selected
                    ),
                    "planned_pairs": len(selected) * EVALUATION_REPETITIONS,
                    "planned_runs": (
                        len(selected)
                        * EVALUATION_REPETITIONS
                        * len(EVALUATION_VARIANTS)
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.require_publication_gate and args.scenario:
        parser.error(
            "--require-publication-gate requires the complete scenario catalog; "
            "omit --scenario"
        )
    if not args.output:
        parser.error("--output is required")
    if not args.confirm_real_llm:
        parser.error("--confirm-real-llm is required to spend real LLM tokens")
    if not has_real_llm_environment():
        parser.error(
            "OPENAI_API_KEY and OPENAI_LANGUAGE_MODEL or OPENAI_MODEL are required"
        )

    output = Path(args.output).resolve()
    with AtomicJsonOutput(output) as artifact:
        with tempfile.TemporaryDirectory(
            prefix="agent-libos-builtin-skill-eval-"
        ) as temp_dir:
            report = run_evaluation(temp_dir, scenario_ids=args.scenario)
        rendered = artifact.commit(report)
    print(rendered, end="")

    if args.require_all_correct and not report_all_correct(report):
        raise SystemExit(1)
    if args.require_publication_gate and not report_publication_ready(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
