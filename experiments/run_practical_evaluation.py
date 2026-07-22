from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.practical_agent_workflows import run_practical_evaluation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the strict practical-workflow evidence gate and emit report schema v1."
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path; the complete report is always printed to stdout.",
    )
    args = parser.parse_args(argv)
    report = run_practical_evaluation().to_dict()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if (
        not report["native_live_ok"]
        or not report["modeled_suite_ok"]
        or report["modeled_fallback"] != 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
