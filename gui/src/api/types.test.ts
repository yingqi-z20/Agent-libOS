import { describe, expect, it } from "vitest";
import { assertRuntimeSnapshot, runtimeSnapshotFromSseData } from "./types";

describe("assertRuntimeSnapshot", () => {
  it("accepts the minimum same-build snapshot shape", () => {
    const value = snapshot();
    expect(() => assertRuntimeSnapshot(value)).not.toThrow();
  });

  it("rejects malformed scheduler and collection fields before rendering", () => {
    expect(() => assertRuntimeSnapshot({ ...snapshot(), scheduler: { auto_run: "yes" } })).toThrow(/scheduler/);
    expect(() => assertRuntimeSnapshot({ ...snapshot(), events: {} })).toThrow(/events/);
  });

  it("rejects process rows without a valid pid", () => {
    expect(() => assertRuntimeSnapshot({ ...snapshot(), processes: [{ pid: "" }] })).toThrow(/pid/);
  });

  it("validates streamed snapshots before exposing them to React", () => {
    expect(runtimeSnapshotFromSseData({ snapshot: snapshot() })).toMatchObject({ db: "local" });
    expect(() => runtimeSnapshotFromSseData({ snapshot: { db: "local" } })).toThrow(/scheduler/);
    expect(() => runtimeSnapshotFromSseData({})).toThrow(/payload/);
  });
});

function snapshot(): Record<string, unknown> {
  return {
    db: "local",
    scheduler: { auto_run: true, running: false, paused: false },
    processes: [],
    human_requests: [],
    events: [],
    audit: [],
    llm_calls: [],
    object_tasks: [],
    tools: [],
    llm_profiles: [],
    images: [],
    skills: [],
    jsonrpc_endpoints: [],
    mcp_servers: [],
    modules: []
  };
}
