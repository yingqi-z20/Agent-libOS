from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.durable_task_runs import (
    CrashMatrixResult,
    run_crash_matrix,
    run_unpaired_committed_result_scenario,
)


def run(output: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task-run-crash-") as work:
        results = run_crash_matrix(work)
        unpaired = run_unpaired_committed_result_scenario(work)
    payload = {
        "schema_version": 1,
        "barrier_count": len(results),
        "passed": all(result.passed for result in results) and unpaired.passed,
        "results": [_result_payload(result) for result in results],
        "unpaired_committed_after_safe_point": _result_payload(unpaired),
        "timing_is_informational_only": True,
    }
    if not payload["passed"]:
        raise RuntimeError("durable TaskRun crash matrix did not converge safely")
    _atomic_write_json(target, payload)
    return payload


def _result_payload(result: CrashMatrixResult) -> dict[str, Any]:
    return {
        "barrier": result.barrier.value,
        "recovery_class": result.recovery_class.value,
        "process_returncode": result.process_returncode,
        "provider_outcome": result.provider_outcome.value,
        "dispatch_count": result.dispatch_count,
        "receipt_count": result.receipt_count,
        "runtime_reopened": result.runtime_reopened,
        "recovered_status": result.recovered_status,
        "blocker_kinds": list(result.blocker_kinds),
        "local_effect_transaction_state": result.local_effect_transaction_state,
        "root_present": result.root_present,
        "validated_action_present": result.validated_action_present,
        "tool_call_present": result.tool_call_present,
        "effect_link_present": result.effect_link_present,
        "resume_point_present": result.resume_point_present,
        "pending_action_present": result.pending_action_present,
        "local_llm_call_count": result.local_llm_call_count,
        "completed_step_count": result.completed_step_count,
        "settlement_reopen_stable": result.settlement_reopen_stable,
        "idempotency_dedupe_verified": result.idempotency_dedupe_verified,
        "reopen_evidence_fingerprint": result.reopen_evidence_fingerprint,
        "passed": result.passed,
    }


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic six-barrier Durable TaskRun crash gate."
    )
    parser.add_argument(
        "--output",
        default=".benchmark_runs/task-run-crash-matrix.json",
    )
    args = parser.parse_args(argv)
    payload = run(args.output)
    print(json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
