import { describe, expect, it, vi } from "vitest";
import { runOrResumeProcess } from "./runControl";

describe("runOrResumeProcess", () => {
  it("resumes and runs a paused process without enabling global auto-run", async () => {
    const client = {
      run: vi.fn().mockResolvedValue(undefined),
      resumeProcess: vi.fn().mockResolvedValue(undefined)
    };

    await runOrResumeProcess(client, { pid: "pid_paused", status: "paused" }, 12);

    expect(client.resumeProcess).toHaveBeenCalledWith("pid_paused", false);
    expect(client.run).toHaveBeenCalledWith("pid_paused", 12);
    expect(client.resumeProcess.mock.invocationCallOrder[0]).toBeLessThan(
      client.run.mock.invocationCallOrder[0]
    );
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
