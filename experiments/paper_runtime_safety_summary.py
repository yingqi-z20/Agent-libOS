from __future__ import annotations

import argparse
import json

from agent_libos.utils.serde import to_jsonable
from benchmarks.runtime_safety.paper_summary import write_paper_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-facing runtime-safety summary JSON and LaTeX tables."
    )
    parser.add_argument("run_dir", help="Benchmark run directory containing results.jsonl and effects.jsonl.")
    parser.add_argument("--json-out", help="Output JSON path. Defaults to <run_dir>/paper_summary.json.")
    parser.add_argument("--tex-out", help="Output LaTeX path. Defaults to <run_dir>/paper_tables.tex.")
    args = parser.parse_args(argv)
    summary = write_paper_summary(args.run_dir, json_out=args.json_out, tex_out=args.tex_out)
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
