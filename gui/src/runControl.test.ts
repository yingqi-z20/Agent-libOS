import { describe, expect, it, vi } from "vitest";
import { runOrResumeProcess } from "./runControl";

describe("runOrResumeProcess", () => {
  it("resumes a paused process and explicitly restarts scheduling", async () => {
    const client = {
      run: vi.fn(),
      resumeProcess: vi.fn().mockResolvedValue(undefined)
    };

    await runOrResumeProcess(client, { pid: "pid_paused", status: "paused" }, 12);

    expect(client.resumeProcess).toHaveBeenCalledWith("pid_paused", true);
    expect(client.run).not.toHaveBeenCalled();
  });

  it("runs a non-paused process with the selected quantum budget", async () => {
    const client = {
      run: vi.fn().mockResolvedValue(undefined),
      resumeProcess: vi.fn()
    };

    await runOrResumeProcess(client, { pid: "pid_ready", status: "runnable" }, 12);

    expect(client.run).toHaveBeenCalledWith("pid_ready", 12);
    expect(client.resumeProcess).not.toHaveBeenCalled();
  });
});
