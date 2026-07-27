from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

from benchmarks.practical_agent_workflows import (
    run_practical_evaluation,
    validate_practical_report,
    validate_practical_report_schema,
)
from experiments.evaluation_output import AtomicJsonOutput


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the strict practical-workflow evidence gate and emit report schema v1."
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path; the complete report is always printed to stdout.",
    )
    args = parser.parse_args(argv)
    reservation = (
        AtomicJsonOutput(Path(args.output))
        if args.output
        else contextlib.nullcontext(None)
    )
    with reservation as artifact:
        report = run_practical_evaluation().to_dict()
        schema_errors = validate_practical_report_schema(report)
        if schema_errors:
            raise RuntimeError(
                "refusing to emit a practical report that violates report.schema.json: "
                + "; ".join(schema_errors)
            )
        invariant_errors = validate_practical_report(report)
        if invariant_errors:
            raise RuntimeError(
                "refusing to emit an internally inconsistent practical report: "
                + "; ".join(invariant_errors)
            )
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        if artifact is not None:
            artifact.commit_text(rendered + "\n")
    print(rendered)
    if (
        not report["native_live_ok"]
        or not report["modeled_suite_ok"]
        or report["modeled_fallback"] != 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
