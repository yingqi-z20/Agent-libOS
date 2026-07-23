from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.runtime_publication_recovery import (
    PUBLICATION_SCALE_PROFILES,
    run_publication_scale_benchmark,
)
from experiments.recovery_artifact_metadata import (
    build_recovery_artifact_metadata,
    new_recovery_run_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic runtime-publication reopen scale benchmark."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PUBLICATION_SCALE_PROFILES),
        default="ci",
        help="ci seeds 10k terminal publications with a 1001-row repair backlog.",
    )
    parser.add_argument("--total-records", type=int)
    parser.add_argument("--unreconciled-records", type=int)
    parser.add_argument("--page-size", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmark_runs/publication-reconciliation-scale.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = PUBLICATION_SCALE_PROFILES[args.profile]
    run_id, started_at = new_recovery_run_identity()
    effective_parameters = {
        "total_records": (
            args.total_records
            if args.total_records is not None
            else profile.total_records
        ),
        "unreconciled_records": (
            args.unreconciled_records
            if args.unreconciled_records is not None
            else profile.unreconciled_records
        ),
        "page_size": (
            args.page_size if args.page_size is not None else profile.page_size
        ),
    }
    result = run_publication_scale_benchmark(
        total_records=effective_parameters["total_records"],
        unreconciled_records=effective_parameters["unreconciled_records"],
        page_size=effective_parameters["page_size"],
    )
    payload = result.as_dict()
    payload["artifact_metadata"] = build_recovery_artifact_metadata(
        benchmark_id="runtime-publication-reconciliation-scale",
        run_id=run_id,
        started_at=started_at,
        selected_profile=args.profile,
        profile_defaults={
            "total_records": profile.total_records,
            "unreconciled_records": profile.unreconciled_records,
            "page_size": profile.page_size,
        },
        explicit_overrides={
            name: value
            for name, value in (
                ("total_records", args.total_records),
                ("unreconciled_records", args.unreconciled_records),
                ("page_size", args.page_size),
            )
            if value is not None
        },
        effective_parameters=effective_parameters,
        source_paths=(
            Path(__file__),
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "runtime_publication_recovery"
            / "runner.py",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
