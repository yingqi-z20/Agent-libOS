from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.external_effect_recovery import (
    BENCHMARK_PROFILES,
    run_recovery_scale_benchmark,
)
from experiments.recovery_artifact_metadata import (
    build_recovery_artifact_metadata,
    new_recovery_run_identity,
)
from experiments.evaluation_output import AtomicJsonOutput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic external-effect recovery scale benchmark."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(BENCHMARK_PROFILES),
        default="ci",
        help="ci seeds 100k records; million seeds 1m records.",
    )
    parser.add_argument("--total-records", type=int)
    parser.add_argument("--pending-records", type=int)
    parser.add_argument("--page-size", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmark_runs/external-effect-recovery-scale.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with AtomicJsonOutput(args.output) as artifact:
        profile = BENCHMARK_PROFILES[args.profile]
        source_paths = (
            Path(__file__),
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "external_effect_recovery"
            / "runner.py",
        )
        identity = new_recovery_run_identity(source_paths=source_paths)
        effective_parameters = {
            "total_records": (
                args.total_records
                if args.total_records is not None
                else profile.total_records
            ),
            "pending_records": (
                args.pending_records
                if args.pending_records is not None
                else profile.pending_records
            ),
            "page_size": (
                args.page_size if args.page_size is not None else profile.page_size
            ),
            "transaction_states": ["prepared"],
        }
        result = run_recovery_scale_benchmark(
            total_records=effective_parameters["total_records"],
            pending_records=effective_parameters["pending_records"],
            page_size=effective_parameters["page_size"],
        )
        payload = result.as_dict()
        payload["artifact_metadata"] = build_recovery_artifact_metadata(
            benchmark_id="external-effect-recovery-scale",
            identity=identity,
            selected_profile=args.profile,
            profile_defaults={
                "total_records": profile.total_records,
                "pending_records": profile.pending_records,
                "page_size": profile.page_size,
            },
            explicit_overrides={
                name: value
                for name, value in (
                    ("total_records", args.total_records),
                    ("pending_records", args.pending_records),
                    ("page_size", args.page_size),
                )
                if value is not None
            },
            effective_parameters=effective_parameters,
        )
        artifact.commit(payload, sort_keys=True)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
