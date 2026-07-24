import type { RuntimeProcess, RuntimeSnapshot } from "./api/types";

export function reconcileSelectedPid(
  snapshot: RuntimeSnapshot,
  current: string | null,
  { preserveExisting = true }: { preserveExisting?: boolean } = {}
): string | null {
  if (preserveExisting && current && snapshot.processes.some((process) => process.pid === current)) {
    return current;
  }
  const activeProcess = snapshot.processes.find(
    (process) => !process.terminal && !["exited", "failed", "killed"].includes(process.status)
  );
  return activeProcess?.pid ?? snapshot.processes[0]?.pid ?? null;
}

export function upsertRuntimeProcess(
  snapshot: RuntimeSnapshot,
  process: RuntimeProcess
): RuntimeSnapshot {
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
