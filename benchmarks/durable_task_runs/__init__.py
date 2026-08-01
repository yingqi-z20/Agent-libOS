"""Deterministic durability gates for first-class Task Runs."""

from benchmarks.durable_task_runs.crash_harness import (
    CRASH_EXIT_CODE,
    CrashMatrixResult,
    DurabilityBarrier,
    FsyncIdempotentJsonRpcProvider,
    FsyncProviderLedger,
    ProviderOutcome,
    RecoveryClass,
    run_crash_matrix,
    run_unpaired_committed_result_scenario,
)
from benchmarks.durable_task_runs.live_evaluation import (
    EVALUATION_ID as LIVE_EVALUATION_ID,
    RELEASE_REPETITIONS as LIVE_RELEASE_REPETITIONS,
    RELEASE_UTILITY_MINIMUM as LIVE_RELEASE_UTILITY_MINIMUM,
    report_release_gate_passed,
    run_evaluation as run_live_evaluation,
)
from benchmarks.durable_task_runs.recovery_scale import (
    BENCHMARK_PROFILES,
    TaskRunRecoveryScaleProfile,
    TaskRunRecoveryScaleResult,
    run_task_run_recovery_scale_benchmark,
)

__all__ = [
    "CRASH_EXIT_CODE",
    "BENCHMARK_PROFILES",
    "CrashMatrixResult",
    "DurabilityBarrier",
    "FsyncIdempotentJsonRpcProvider",
    "FsyncProviderLedger",
    "ProviderOutcome",
    "RecoveryClass",
    "LIVE_EVALUATION_ID",
    "LIVE_RELEASE_REPETITIONS",
    "LIVE_RELEASE_UTILITY_MINIMUM",
    "TaskRunRecoveryScaleProfile",
    "TaskRunRecoveryScaleResult",
    "run_crash_matrix",
    "run_live_evaluation",
    "report_release_gate_passed",
    "run_unpaired_committed_result_scenario",
    "run_task_run_recovery_scale_benchmark",
]
