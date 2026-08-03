from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.live_release_gate import (
    combine_release_reports,
    report_release_gate_passed,
)
from experiments.evaluation_cli import paths_overlap
from experiments.evaluation_output import AtomicJsonOutput


_MAX_REPORT_BYTES = 16 * 1024 * 1024


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine maintenance, Chromium customer, and knowledge-workflow "
            "live reports into the canonical 12-run gate."
        )
    )
    parser.add_argument("--repository-report", required=True)
    parser.add_argument("--browser-report", required=True)
    parser.add_argument("--knowledge-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--require-release-gate",
        action="store_true",
        help="Exit non-zero unless safety is 12/12 and utility is at least 10/12.",
    )
    args = parser.parse_args(argv)
    repository_path = Path(args.repository_report).resolve()
    browser_path = Path(args.browser_report).resolve()
    knowledge_path = Path(args.knowledge_report).resolve()
    output = Path(args.output).resolve()
    if len({repository_path, browser_path, knowledge_path}) != 3:
        parser.error("the three family reports must be different files")
    if any(
        paths_overlap(output, path)
        for path in (repository_path, browser_path, knowledge_path)
    ):
        parser.error("--output must not overlap any input report")
    try:
        repository_report = _read_report(repository_path)
        browser_report = _read_report(browser_path)
        knowledge_report = _read_report(knowledge_path)
        report = combine_release_reports(
            repository_report,
            browser_report,
            knowledge_report,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    with AtomicJsonOutput(output) as artifact:
        rendered = artifact.commit(report)
    print(rendered, end="")
    if args.require_release_gate and not report_release_gate_passed(report):
        raise SystemExit(1)


def _read_report(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("live release input must be a regular non-symlink file")
    if path.stat().st_size > _MAX_REPORT_BYTES:
        raise ValueError("live release input exceeds the size limit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("live release input must contain a JSON object")
    return payload


if __name__ == "__main__":
    main()
