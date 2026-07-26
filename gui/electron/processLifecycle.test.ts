import { describe, expect, it, vi } from "vitest";
import { appendStartupOutput, cleanupBeforeExit, consumeStartupOutput, isChildAlive, withStartupFailureCleanup } from "./processLifecycle.js";

describe("child process lifecycle", () => {
  it("uses exitCode and signalCode instead of killed to decide liveness", () => {
    expect(isChildAlive({ exitCode: null, signalCode: null, killed: true })).toBe(true);
    expect(isChildAlive({ exitCode: 0, signalCode: null, killed: false })).toBe(false);
    expect(isChildAlive({ exitCode: null, signalCode: "SIGTERM", killed: false })).toBe(false);
  });

  it("fails closed before either startup stream can exceed the configured byte limit", () => {
    const first = appendStartupOutput("", Buffer.from("prefix"), 11, "stdout");

    expect(appendStartupOutput(first, Buffer.from("-tail"), 11, "stdout")).toBe("prefix-tail");
    expect(() => appendStartupOutput(first, Buffer.from("-tails"), 11, "stdout")).toThrow(/stdout exceeded 11 bytes/);
    expect(() => appendStartupOutput("", Buffer.alloc(12), 11, "stderr")).toThrow(/stderr exceeded 11 bytes/);
    expect(() => appendStartupOutput("", Buffer.alloc(1), 0, "stdout")).toThrow(/positive safe integer/);
  });

  it("runs process cleanup before propagating a startup output overflow", async () => {
    for (const streamName of ["stdout", "stderr"] as const) {
      const cleanup = vi.fn(async () => undefined);

      await expect(withStartupFailureCleanup(async () => {
        appendStartupOutput("", Buffer.alloc(65_537), 65_536, streamName);
        return "unreachable";
      }, cleanup)).rejects.toThrow(new RegExp(`${streamName} exceeded 65536 bytes`));

      expect(cleanup).toHaveBeenCalledOnce();
    }
  });

  it("waits for a complete newline-delimited startup frame and ignores unrelated JSON logs", () => {
    let result = consumeStartupOutput(
      { text: "", scanOffset: 0 },
      Buffer.from('{"stage":"boot"}\n{"url":"http://127.0.0.1:'),
      4096
    );
    expect(result.connection).toBeNull();

    result = consumeStartupOutput(result.state, Buffer.from('4321","token":"secret","db":"local"}\n'), 4096);

    expect(result.connection).toEqual({ url: "http://127.0.0.1:4321", token: "secret", db: "local" });
  });

  it("rejects a complete malformed connection candidate", () => {
    expect(() => consumeStartupOutput(
      { text: "", scanOffset: 0 },
      Buffer.from('{"url":"https://example.com","token":"secret","db":"local"}\n'),
      4096
    )).toThrow(/loopback HTTP origin/);
  });

  it("settles startup cleanup before exiting", async () => {
    const order: string[] = [];
    await cleanupBeforeExit(async () => {
      await Promise.resolve();
      order.push("cleanup");
    }, () => order.push("exit"));
    expect(order).toEqual(["cleanup", "exit"]);
  });
});
