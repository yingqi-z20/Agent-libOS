import type { RuntimeProcess, RuntimeSnapshot, TaskRunSummary } from "./api/types";
import { upsertTaskRunSummary } from "./api/types";

export function reconcileSelectedRunId(
  snapshot: RuntimeSnapshot,
  current: string | null,
  { preserveExisting = true }: { preserveExisting?: boolean } = {}
): string | null {
  if (preserveExisting && current && snapshot.task_runs.some((run) => run.run_id === current)) {
    return current;
  }
  const attention = snapshot.task_runs.find((run) => run.status === "needs_attention" || run.status === "waiting_human");
  const active = snapshot.task_runs.find((run) => !["succeeded", "failed", "cancelled"].includes(run.status));
  return attention?.run_id ?? active?.run_id ?? snapshot.task_runs[0]?.run_id ?? null;
}

export function upsertRuntimeTaskRun(
  snapshot: RuntimeSnapshot,
  run: TaskRunSummary
): RuntimeSnapshot {
  return { ...snapshot, task_runs: upsertTaskRunSummary(snapshot.task_runs, run) };
}

/** Preserve locally observed higher TaskRun revisions when a full snapshot is stale. */
export function mergeRuntimeTaskRuns(
  snapshot: RuntimeSnapshot,
  current: RuntimeSnapshot
): RuntimeSnapshot {
  return current.task_runs.reduce(
    (merged, run) => upsertRuntimeTaskRun(merged, run),
    snapshot
  );
}

export function reconcileSelectedPid(
  snapshot: RuntimeSnapshot,
  current: string | null,
  { preserveExisting = true }: { preserveExisting?: boolean } = {}
): string | null {
  if (preserveExisting && current && snapshot.processes.some((process) => process.pid === current)) {
    return current;
  }
  const pendingHumanPids = new Set(
    snapshot.human_requests
      .filter((request) => request.status === "pending")
      .map((request) => request.pid)
  );
  const pendingHumanProcess = snapshot.processes.find(
    (process) => pendingHumanPids.has(process.pid)
      && !process.terminal
      && !["exited", "failed", "killed"].includes(process.status)
  );
  const activeProcess = snapshot.processes.find(
    (process) => !process.terminal && !["exited", "failed", "killed"].includes(process.status)
  );
  return pendingHumanProcess?.pid ?? activeProcess?.pid ?? snapshot.processes[0]?.pid ?? null;
}

export function upsertRuntimeProcess(
  snapshot: RuntimeSnapshot,
  process: RuntimeProcess
): RuntimeSnapshot {
  const existing = snapshot.processes.find((item) => item.pid === process.pid);
  if (existing && existing.state_generation > process.state_generation) return snapshot;
  return {
    ...snapshot,
    processes: [process, ...snapshot.processes.filter((item) => item.pid !== process.pid)]
  };
}

export function processFromMutationResult(result: unknown): RuntimeProcess | null {
  if (!result || typeof result !== "object") return null;
  const wrapped = (result as { process?: unknown }).process;
  const candidate = wrapped && typeof wrapped === "object" ? wrapped : result;
  if (
    typeof (candidate as { pid?: unknown }).pid !== "string"
    || typeof (candidate as { status?: unknown }).status !== "string"
    || typeof (candidate as { image_id?: unknown }).image_id !== "string"
  ) {
    return null;
  }
  return candidate as RuntimeProcess;
}
