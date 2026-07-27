#!/usr/bin/env python3
"""Measure latency of the small core Runtime lifecycle surface."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-median-ratio", type=float, default=1.10)
    parser.add_argument("--max-p95-ratio", type=float, default=1.20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    return args


def _timed(operation: Callable[[], None]) -> float:
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        started = time.perf_counter_ns()
        operation()
        return (time.perf_counter_ns() - started) / 1_000_000
    finally:
        if was_enabled:
            gc.enable()


def _measure_open_close(runtime_type: Any, iterations: int) -> list[float]:
    def sample() -> None:
        runtime = runtime_type.open("local")
        runtime.close()

    return [_timed(sample) for _ in range(iterations)]


def _measure_runtime_operation(
    runtime_type: Any,
    iterations: int,
    prepare: Callable[[Any, int], Callable[[], None]],
) -> list[float]:
    samples: list[float] = []
    for index in range(iterations):
        runtime = runtime_type.open("local")
        try:
            operation = prepare(runtime, index)
            samples.append(_timed(operation))
        finally:
            runtime.close()
    return samples


def _spawn(runtime: Any, index: int) -> Callable[[], None]:
    return lambda: runtime.process.spawn(
        image="base-agent:v0",
        goal=f"latency spawn {index}",
    )


def _tool_call(runtime: Any, index: int) -> Callable[[], None]:
    pid = runtime.process.spawn(
        image="base-agent:v0",
        goal=f"latency tool {index}",
    )

    def invoke() -> None:
        result = runtime.tools.call(pid, "get_working_directory", {})
        if not result.ok:
            raise RuntimeError(f"latency tool call failed: {result.error}")

    return invoke


def _checkpoint_create(runtime: Any, index: int) -> Callable[[], None]:
    pid = runtime.process.spawn(
        image="base-agent:v0",
        goal=f"latency checkpoint create {index}",
    )
    return lambda: runtime.checkpoint.create(
        pid,
        f"latency create {index}",
        actor=pid,
        require_capability=False,
    )


def _checkpoint_restore(runtime: Any, index: int) -> Callable[[], None]:
    pid = runtime.process.spawn(
        image="base-agent:v0",
        goal=f"latency checkpoint restore {index}",
    )
    checkpoint_id = runtime.checkpoint.create(
        pid,
        f"latency restore {index}",
        actor=pid,
        require_capability=False,
    )
    return lambda: runtime.checkpoint.restore(
        "latency",
        checkpoint_id,
        require_capability=False,
    )


def _summarize(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "iterations": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _compare(
    operations: dict[str, dict[str, Any]],
    baseline_path: Path,
    *,
    max_median_ratio: float,
    max_p95_ratio: float,
) -> dict[str, Any]:
    _require_finite_positive(max_median_ratio, "max_median_ratio")
    _require_finite_positive(max_p95_ratio, "max_p95_ratio")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_operations = baseline.get("operations", {})
    if not isinstance(baseline_operations, dict):
        raise ValueError("baseline operations must be an object")
    compared: dict[str, Any] = {}
    for name, current in operations.items():
        expected = baseline_operations.get(name)
        if not isinstance(expected, dict):
            raise ValueError(f"baseline is missing operation {name!r}")
        current_median = _require_finite_nonnegative(
            current.get("median_ms"),
            f"current {name!r} median_ms",
        )
        current_p95 = _require_finite_nonnegative(
            current.get("p95_ms"),
            f"current {name!r} p95_ms",
        )
        expected_median = _require_finite_positive(
            expected.get("median_ms"),
            f"baseline {name!r} median_ms",
        )
        expected_p95 = _require_finite_positive(
            expected.get("p95_ms"),
            f"baseline {name!r} p95_ms",
        )
        median_ratio = current_median / expected_median
        p95_ratio = current_p95 / expected_p95
        compared[name] = {
            "median_ratio": median_ratio,
            "p95_ratio": p95_ratio,
            "passed": (
                median_ratio <= max_median_ratio
                and p95_ratio <= max_p95_ratio
            ),
        }
    return {
        "max_median_ratio": max_median_ratio,
        "max_p95_ratio": max_p95_ratio,
        "operations": compared,
        "passed": all(item["passed"] for item in compared.values()),
    }


def _require_finite_positive(value: object, field: str) -> float:
    selected = _require_finite_number(value, field)
    if selected <= 0:
        raise ValueError(f"{field} must be positive")
    return selected


def _require_finite_nonnegative(value: object, field: str) -> float:
    selected = _require_finite_number(value, field)
    if selected < 0:
        raise ValueError(f"{field} must be non-negative")
    return selected


def _require_finite_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{field} must be a finite number")
    return selected


def main() -> int:
    args = _parse_args()
    if args.source_root is not None:
        sys.path.insert(0, str(args.source_root.resolve()))

    from agent_libos import Runtime

    measurements = {
        "open_close": _measure_open_close(Runtime, args.iterations),
        "spawn": _measure_runtime_operation(Runtime, args.iterations, _spawn),
        "tool_call": _measure_runtime_operation(Runtime, args.iterations, _tool_call),
        "checkpoint_create": _measure_runtime_operation(
            Runtime,
            args.iterations,
            _checkpoint_create,
        ),
        "checkpoint_restore": _measure_runtime_operation(
            Runtime,
            args.iterations,
            _checkpoint_restore,
        ),
    }
    operations = {
        name: _summarize(samples)
        for name, samples in measurements.items()
    }
    report = {
        "iterations": args.iterations,
        "operations": operations,
    }
    if args.baseline is not None:
        report["comparison"] = _compare(
            operations,
            args.baseline,
            max_median_ratio=args.max_median_ratio,
            max_p95_ratio=args.max_p95_ratio,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    comparison = report.get("comparison")
    return 0 if comparison is None or comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
