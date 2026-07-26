import type { LibOSClient } from "./api/client";
import type { RuntimeProcess } from "./api/types";
import type { OptionalQuanta } from "./quanta";

type RunControlClient = Pick<LibOSClient, "run" | "resumeProcess">;
type RunControlProcess = Pick<RuntimeProcess, "pid" | "status">;

export async function runOrResumeProcess(
  client: RunControlClient,
  process: RunControlProcess,
  maxQuanta: OptionalQuanta
): Promise<unknown> {
  if (process.status === "paused") {
    // Keep one click scoped to the selected process even when global auto-run
    // is disabled: first clear the pause fence, then start the requested run.
    await client.resumeProcess(process.pid, false);
    return client.run(process.pid, maxQuanta);
  }
  return client.run(process.pid, maxQuanta);
}
