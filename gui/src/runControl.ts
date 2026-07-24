import type { LibOSClient } from "./api/client";
import type { RuntimeProcess } from "./api/types";
import type { OptionalQuanta } from "./quanta";

type RunControlClient = Pick<LibOSClient, "run" | "resumeProcess">;
type RunControlProcess = Pick<RuntimeProcess, "pid" | "status">;

export async function runOrResumeProcess(
  client: RunControlClient,
  process: RunControlProcess,
  maxQuanta: OptionalQuanta
): Promise<void> {
  if (process.status === "paused") {
    // "Run" means continue execution, including after stale-execution
    // recovery or an explicit process pause.
    await client.resumeProcess(process.pid, true);
    return;
  }
  await client.run(process.pid, maxQuanta);
}
