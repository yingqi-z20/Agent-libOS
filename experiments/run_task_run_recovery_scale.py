from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.durable_task_runs import (
    BENCHMARK_PROFILES,
    run_task_run_recovery_scale_benchmark,
)


def run(profile: str, output: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        selected = BENCHMARK_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown TaskRun recovery profile: {profile}") from exc
    result = run_task_run_recovery_scale_benchmark(
        total_runs=selected.total_runs,
        recoverable_runs=selected.recoverable_runs,
        page_size=selected.page_size,
    )
    payload = result.as_dict()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded Durable TaskRun recovery scale gate."
    )
    parser.add_argument("--profile", choices=sorted(BENCHMARK_PROFILES), default="ci")
    parser.add_argument(
        "--output",
        default=".benchmark_runs/task-run-recovery-scale.json",
    )
    args = parser.parse_args(argv)
    payload = run(args.profile, args.output)
    print(json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
