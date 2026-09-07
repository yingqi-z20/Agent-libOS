from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from agent_libos.config import DEFAULT_CONFIG
from benchmarks.long_horizon_agent import report_all_successful, run_evaluation
from benchmarks.long_horizon_agent.runner import SCENARIO_ID, SCENARIOS
from experiments.evaluation_cli import (
    has_real_llm_environment,
    paths_overlap,
    positive_int,
)
from experiments.evaluation_output import AtomicJsonOutput


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an opt-in real-LLM repository-maintenance task across a human "
            "follow-up and durable Runtime restart."
        )
    )
    parser.add_argument(
        "--output",
        help="JSON report path (required unless --list-scenarios is given).",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default=SCENARIO_ID,
        help="Registered long-horizon scenario to run.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print the registered scenario ids, one per line, and exit.",
    )
    parser.add_argument("--repetitions", type=positive_int, default=1)
    parser.add_argument(
        "--phase-one-quanta",
        type=positive_int,
        default=None,
        help=(
            "Scheduler quanta before the follow-up message and Runtime restart "
            "(default: the selected scenario's value)."
        ),
    )
    parser.add_argument(
        "--max-quanta",
        type=positive_int,
        default=None,
        help=(
            "Total scheduler quanta per run; must exceed --phase-one-quanta "
            "(default: the selected scenario's value)."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help=(
            "Print one stderr summary line per completed phase (quanta, tool "
            "names, status, and seconds only)."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        help=(
            "Optional new directory that retains the synthetic workspace and "
            "Runtime database for diagnosis. Omit to use a temporary directory."
        ),
    )
    parser.add_argument(
        "--confirm-real-llm",
        action="store_true",
        help="Acknowledge that the evaluation makes paid provider calls.",
    )
    parser.add_argument(
        "--require-all-successful",
        action="store_true",
        help="Exit non-zero unless every durable task-state oracle passes.",
    )
    parser.add_argument(
        "--prompt-layout",
        choices=("legacy_v1", "cache_optimized_v2"),
        default=DEFAULT_CONFIG.llm.prompt_layout,
        help="Model prompt layout used for this paired evaluation arm.",
    )
    args = parser.parse_args(argv)
    if args.list_scenarios:
        for scenario_id in sorted(SCENARIOS):
            print(scenario_id)
        return
    if not args.output:
        parser.error("--output is required")
    if not args.confirm_real_llm:
        parser.error("--confirm-real-llm is required to spend real LLM tokens")
    if not has_real_llm_environment():
        parser.error(
            "OPENAI_API_KEY and OPENAI_LANGUAGE_MODEL or OPENAI_MODEL are required"
        )
    output = Path(args.output).resolve()
    scenario = SCENARIOS[args.scenario]
    phase_one_quanta = (
        scenario.default_phase_one_quanta
        if args.phase_one_quanta is None
        else args.phase_one_quanta
    )
    max_quanta = (
        scenario.default_max_quanta if args.max_quanta is None else args.max_quanta
    )
    if max_quanta <= phase_one_quanta:
        parser.error("--max-quanta must be greater than --phase-one-quanta")
    progress = _stderr_progress if args.progress else None
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(DEFAULT_CONFIG.llm, prompt_layout=args.prompt_layout),
    )
    artifacts_root = (
        Path(args.artifacts_root).resolve() if args.artifacts_root else None
    )
    if artifacts_root is not None:
        if paths_overlap(output, artifacts_root):
            parser.error("--output and --artifacts-root must not overlap")
        if artifacts_root.exists() and not artifacts_root.is_dir():
            parser.error("--artifacts-root must name a directory")
        if artifacts_root.exists() and any(artifacts_root.iterdir()):
            parser.error("--artifacts-root must be absent or empty")

    with AtomicJsonOutput(output) as artifact:
        if artifacts_root is not None:
            report = run_evaluation(
                artifacts_root,
                repetitions=args.repetitions,
                phase_one_quanta=phase_one_quanta,
                max_quanta=max_quanta,
                config=config,
                scenario_id=scenario.scenario_id,
                progress=progress,
            )
            report["artifacts_root"] = str(artifacts_root)
        else:
            with tempfile.TemporaryDirectory(
                prefix="agent-libos-long-horizon-"
            ) as root:
                report = run_evaluation(
                    root,
                    repetitions=args.repetitions,
                    phase_one_quanta=phase_one_quanta,
                    max_quanta=max_quanta,
                    config=config,
                    scenario_id=scenario.scenario_id,
                    progress=progress,
                )
        rendered = artifact.commit(report)
    print(rendered, end="")
    if args.require_all_successful and not report_all_successful(report):
        raise SystemExit(1)


def _stderr_progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
