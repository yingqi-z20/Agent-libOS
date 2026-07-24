from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from benchmarks.long_horizon_agent import report_all_successful, run_evaluation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an opt-in real-LLM repository-maintenance task across a human "
            "follow-up and durable Runtime restart."
        )
    )
    parser.add_argument("--output", required=True, help="JSON report path.")
    parser.add_argument("--repetitions", type=_positive_int, default=1)
    parser.add_argument("--phase-one-quanta", type=_positive_int, default=6)
    parser.add_argument("--max-quanta", type=_positive_int, default=48)
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
    args = parser.parse_args(argv)
    if not args.confirm_real_llm:
        parser.error("--confirm-real-llm is required to spend real LLM tokens")
    if not _has_real_llm_environment():
        parser.error(
            "OPENAI_API_KEY and OPENAI_LANGUAGE_MODEL or OPENAI_MODEL are required"
        )
    if args.artifacts_root:
        artifacts_root = Path(args.artifacts_root).resolve()
        if artifacts_root.exists() and any(artifacts_root.iterdir()):
            parser.error("--artifacts-root must be absent or empty")
        report = run_evaluation(
            artifacts_root,
            repetitions=args.repetitions,
            phase_one_quanta=args.phase_one_quanta,
            max_quanta=args.max_quanta,
        )
        report["artifacts_root"] = str(artifacts_root)
    else:
        with tempfile.TemporaryDirectory(prefix="agent-libos-long-horizon-") as root:
            report = run_evaluation(
                root,
                repetitions=args.repetitions,
                phase_one_quanta=args.phase_one_quanta,
                max_quanta=args.max_quanta,
            )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_all_successful and not report_all_successful(report):
        raise SystemExit(1)


def _positive_int(value: str) -> int:
    selected = int(value)
    if selected < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return selected


def _has_real_llm_environment() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        and (os.getenv("OPENAI_LANGUAGE_MODEL") or os.getenv("OPENAI_MODEL"))
    )


if __name__ == "__main__":
    main()
