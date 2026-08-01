import { describe, expect, it } from "vitest";
import { allowedTaskRunActions, assertRuntimeSnapshot, assertTaskRunDetail, runtimeSnapshotFromSseData, taskRunSummaryFromSseData, upsertTaskRunSummary } from "./types";

describe("assertRuntimeSnapshot", () => {
  it("accepts the minimum same-build snapshot shape", () => {
    const value = snapshot();
    expect(() => assertRuntimeSnapshot(value)).not.toThrow();
  });

  it("rejects malformed scheduler and collection fields before rendering", () => {
    expect(() => assertRuntimeSnapshot({ ...snapshot(), scheduler: { auto_run: "yes" } })).toThrow(/scheduler/);
    expect(() => assertRuntimeSnapshot({ ...snapshot(), events: {} })).toThrow(/events/);
  });

  it("requires schema v2 and validates durable run controls", () => {
    expect(() => assertRuntimeSnapshot({ ...snapshot(), schema_version: 1 })).toThrow(/schema_version/);
    expect(() => assertRuntimeSnapshot({ ...snapshot(), task_runs: [{ ...run(), allowed_actions: ["retry"] }] })).toThrow(/allowed_actions/);
    expect(() => assertRuntimeSnapshot({ ...snapshot(), task_runs: [{ ...run(), allowed_actions: ["purge_payloads"] }] })).toThrow(/allowed_actions/);
  });

  it("rejects process rows without a valid pid", () => {
    expect(() => assertRuntimeSnapshot({ ...snapshot(), processes: [{ pid: "" }] })).toThrow(/pid/);
  });

  it("accepts only hash-redacted typed stale-execution wait receipts", () => {
    const receipt = {
      schema_version: 1,
      kind: "stale_execution",
      pid: "pid_stale",
      recovered_by_owner_sha256: "a".repeat(64),
      prior_owner_sha256: "b".repeat(64),
      prior_lease_sha256: "c".repeat(64),
      prior_execution_generation: 4,
      recovered_execution_generation: 5,
      recovered_state_generation: 6
    };
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      processes: [{ pid: "pid_stale", wait_state: receipt }]
    })).not.toThrow();
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      processes: [{
        pid: "pid_stale",
        wait_state: { ...receipt, prior_execution_lease_id: "raw-token" }
      }]
    })).toThrow(/wait state/);
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      processes: [{
        pid: "pid_stale",
        wait_state: { ...receipt, prior_lease_sha256: "C".repeat(64) }
      }]
    })).toThrow(/wait state/);
  });

  it("validates streamed snapshots before exposing them to React", () => {
    expect(runtimeSnapshotFromSseData({ snapshot: snapshot() })).toMatchObject({ db: "local" });
    expect(() => runtimeSnapshotFromSseData({ snapshot: { schema_version: 2, db: "local" } })).toThrow(/scheduler/);
    expect(() => runtimeSnapshotFromSseData({})).toThrow(/payload/);
  });
});

describe("task run SSE reconciliation", () => {
  it("ignores lower and equal revisions", () => {
    const current = run(3);
    expect(upsertTaskRunSummary([current], run(2))[0].revision).toBe(3);
    expect(upsertTaskRunSummary([current], { ...run(3), status: "running" })[0].status).toBe("paused");
    expect(upsertTaskRunSummary([current], run(4))[0].revision).toBe(4);
  });

  it("fails closed and never exposes ordinary resume for unknown effects", () => {
    const summary = {
      ...run(),
      status: "needs_attention" as const,
      blockers: [{ kind: "unknown_effect" }],
      allowed_actions: ["resume", "recover", "cancel"] as const
    };
    expect([...allowedTaskRunActions({ ...summary, allowed_actions: ["retry"] })]).toEqual([]);
    expect([...allowedTaskRunActions({ ...summary, allowed_actions: ["resume", "recover", "cancel"] })]).toEqual(["recover", "cancel"]);
    expect(taskRunSummaryFromSseData({ summary: { ...summary, allowed_actions: ["recover", "cancel"] } }).revision).toBe(1);
  });

  it("rejects private payload material in summaries", () => {
    expect(() => taskRunSummaryFromSseData({ ...run(), goal: "secret" })).toThrow(/private field/);
    expect(() => taskRunSummaryFromSseData({ ...run(), blockers: [{ kind: "unknown_effect", payload: { token: "x" } }] })).toThrow(/private field/);
    expect(() => taskRunSummaryFromSseData({ ...run(), payloads_purged: "false" })).toThrow(/purge state/);
  });
});

describe("task run recovery option projection", () => {
  it("accepts only the safe evidence binding used by localized recovery templates", () => {
    const detail = {
      summary: run(),
      requirements: { items: [], next_cursor: null, has_more: false },
      recovery_options: [{
        schema_version: 1,
        option_id: "effect_receipt:binding",
        kind: "effect_receipt",
        requires_receipt: true,
        effect_id: "effect_1",
        expected_transaction_state: "unknown",
        runtime_epoch: 12
      }]
    };

    expect(() => assertTaskRunDetail(detail)).not.toThrow();
    expect(() => assertTaskRunDetail({
      ...detail,
      recovery_options: [{ ...detail.recovery_options[0], provider_secret: "never-render" }]
    })).toThrow(/recovery options/);
    expect(() => assertTaskRunDetail({
      ...detail,
      recovery_options: [{ ...detail.recovery_options[0], runtime_epoch: -1 }]
    })).toThrow(/recovery options/);
  });
});

function snapshot(): Record<string, unknown> {
  return {
    schema_version: 2,
    db: "local",
    scheduler: { auto_run: true, running: false, paused: false },
    processes: [],
    human_requests: [],
    events: [],
    audit: [],
    llm_calls: [],
    object_tasks: [],
    task_runs: [],
    tools: [],
    llm_profiles: [],
    images: [],
    skills: [],
    jsonrpc_endpoints: [],
    mcp_servers: [],
    modules: []
  };
}

function run(revision = 1) {
  return {
    schema_version: 1 as const,
    run_id: "run_1",
    revision,
    status: "paused" as const,
    display_title: "Durable task",
    root_pid: "pid_1",
    active_pid: "pid_1",
    allowed_actions: ["resume", "cancel"] as Array<"resume" | "cancel">,
    blockers: [],
    retention: "purge_on_terminal" as const,
    payloads_purged: false
  };
}
