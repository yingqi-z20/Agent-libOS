from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from benchmarks.knowledge_workflows import (
    RELEASE_REPETITIONS,
    report_release_gate_passed,
    run_evaluation,
)
from benchmarks.knowledge_workflows.evaluation import (
    DEFAULT_MAX_QUANTA,
    DEFAULT_PHASE_ONE_QUANTA,
)
from experiments.evaluation_cli import (
    has_real_llm_environment,
    paths_overlap,
    positive_int,
)
from experiments.evaluation_output import AtomicJsonOutput


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the opt-in real-LLM research and data-analysis Durable "
            "TaskRun release gate with Runtime restart and strict oracles."
        )
    )
    parser.add_argument("--output", required=True, help="Redacted JSON report path.")
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=RELEASE_REPETITIONS,
        help="Repetitions per scenario.",
    )
    parser.add_argument(
        "--phase-one-quanta",
        type=positive_int,
        default=DEFAULT_PHASE_ONE_QUANTA,
    )
    parser.add_argument("--max-quanta", type=positive_int, default=DEFAULT_MAX_QUANTA)
    parser.add_argument(
        "--artifacts-root",
        help=(
            "Optional new directory retaining synthetic workspaces and v4 "
            "Runtime databases. Omit it for automatic temporary cleanup."
        ),
    )
    parser.add_argument(
        "--confirm-real-llm",
        action="store_true",
        help="Acknowledge that this command makes paid provider calls.",
    )
    parser.add_argument(
        "--require-release-gate",
        action="store_true",
        help=(
            "Exit non-zero unless each scenario has exactly three llm-live "
            "runs, safety 3/3, and utility at least 2/3."
        ),
    )
    args = parser.parse_args(argv)
    if not args.confirm_real_llm:
        parser.error("--confirm-real-llm is required to spend real LLM tokens")
    if not has_real_llm_environment():
        parser.error(
            "OPENAI_API_KEY and OPENAI_LANGUAGE_MODEL or OPENAI_MODEL are required"
        )
    if args.require_release_gate and args.repetitions != RELEASE_REPETITIONS:
        parser.error(
            f"--require-release-gate requires --repetitions {RELEASE_REPETITIONS}"
        )
    if args.max_quanta <= args.phase_one_quanta:
        parser.error("--max-quanta must be greater than --phase-one-quanta")

    output = Path(args.output).resolve()
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
                phase_one_quanta=args.phase_one_quanta,
                max_quanta=args.max_quanta,
                confirm_real_llm=True,
            )
            report["artifacts_root"] = str(artifacts_root)
        else:
            with tempfile.TemporaryDirectory(
                prefix="agent-libos-knowledge-workflows-live-"
            ) as root:
                report = run_evaluation(
                    root,
                    repetitions=args.repetitions,
                    phase_one_quanta=args.phase_one_quanta,
                    max_quanta=args.max_quanta,
                    confirm_real_llm=True,
                )
        rendered = artifact.commit(report)
    print(rendered, end="")
    if args.require_release_gate and not report_release_gate_passed(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
