import { describe, expect, it } from "vitest";
import type { RuntimeProcess, RuntimeSnapshot } from "./api/types";
import { mergeRuntimeTaskRuns, processFromMutationResult, reconcileSelectedPid, reconcileSelectedRunId, upsertRuntimeProcess, upsertRuntimeTaskRun } from "./selection";

describe("reconcileSelectedPid", () => {
  it("preserves an existing selected process", () => {
    expect(reconcileSelectedPid(snapshot(["pid_1", "pid_2"]), "pid_2")).toBe("pid_2");
  });

  it("falls back to the first process when selection is stale", () => {
    expect(reconcileSelectedPid(snapshot(["pid_1", "pid_2"]), "missing")).toBe("pid_1");
  });

  it("prefers actionable work over older terminal history", () => {
    const value = snapshot(["pid_done", "pid_ready"]);
    value.processes[0] = { ...value.processes[0], status: "exited", terminal: true };

    expect(reconcileSelectedPid(value, null)).toBe("pid_ready");
  });

  it("prefers a process waiting on a human decision when choosing an initial task", () => {
    const value = snapshot(["pid_ready", "pid_needs_input"]);
    value.human_requests = [{
      request_id: "hr_1",
      pid: "pid_needs_input",
      human: "owner",
      status: "pending",
      payload: { type: "approval" },
      decision: null,
      blocking: true,
      created_at: "2026-07-26T12:00:00Z",
      updated_at: "2026-07-26T12:00:00Z"
    }];

    expect(reconcileSelectedPid(value, null)).toBe("pid_needs_input");
  });

  it("resets selection when preserving is disabled", () => {
    expect(reconcileSelectedPid(snapshot(["pid_1", "pid_2"]), "pid_2", { preserveExisting: false })).toBe("pid_1");
  });

  it("returns null when no process exists", () => {
    expect(reconcileSelectedPid(snapshot([]), "pid_1")).toBeNull();
  });

  it("optimistically inserts or replaces the process returned by a mutation", () => {
    const original = snapshot(["pid_1", "pid_2"]);
    const updated = { ...original.processes[1], status: "waiting_message" } as RuntimeProcess;

    const next = upsertRuntimeProcess(original, updated);

    expect(next.processes.map((process) => process.pid)).toEqual(["pid_2", "pid_1"]);
    expect(next.processes[0].status).toBe("waiting_message");
    expect(original.processes[1].status).toBe("runnable");
  });

  it("does not roll a process back to an older state generation", () => {
    const current = snapshot(["pid_1"]);
    current.processes[0] = { ...current.processes[0], state_generation: 9, status: "waiting_message" };
    const stale = { ...current.processes[0], state_generation: 8, status: "running" };

    const next = upsertRuntimeProcess(current, stale);

    expect(next).toBe(current);
    expect(next.processes[0]).toMatchObject({ state_generation: 9, status: "waiting_message" });
  });

  it("extracts only process-shaped direct or wrapped mutation responses", () => {
    const process = snapshot(["pid_1"]).processes[0];
    expect(processFromMutationResult(process)).toBe(process);
    expect(processFromMutationResult({ process, scheduler: { running: true } })).toBe(process);
    expect(processFromMutationResult({ running: true })).toBeNull();
    expect(processFromMutationResult(null)).toBeNull();
  });
});

describe("durable run selection", () => {
  it("prefers attention, preserves selection, and ignores stale revisions", () => {
    const value = snapshot(["pid_1"]);
    const queued = taskRun("run_queued", 2, "queued");
    const attention = taskRun("run_attention", 4, "needs_attention");
    value.task_runs = [queued, attention];

    expect(reconcileSelectedRunId(value, null)).toBe("run_attention");
    expect(reconcileSelectedRunId(value, "run_queued")).toBe("run_queued");
    expect(upsertRuntimeTaskRun(value, { ...attention, revision: 3, status: "running" }).task_runs[1]).toBe(attention);
    expect(upsertRuntimeTaskRun(value, { ...attention, revision: 5, status: "cancelled" }).task_runs[0]).toMatchObject({ revision: 5, status: "cancelled" });
  });

  it("does not let a stale full snapshot roll back a locally observed run", () => {
    const current = snapshot(["pid_1"]);
    current.task_runs = [taskRun("run_1", 5, "needs_attention")];
    const stale = snapshot(["pid_1"]);
    stale.task_runs = [taskRun("run_1", 4, "running")];

    expect(mergeRuntimeTaskRuns(stale, current).task_runs[0]).toMatchObject({
      run_id: "run_1",
      revision: 5,
      status: "needs_attention"
    });
  });
});

function snapshot(pids: string[]): RuntimeSnapshot {
  return {
    schema_version: 2,
    db: "local",
    scheduler: {
      auto_run: true,
      running: false,
      paused: false,
      task_id: null,
      reason: null,
      last_result: [],
      last_error: null,
      started_at: null,
      finished_at: null,
      default_max_quanta: null
    },
    processes: pids.map((pid) => ({
      pid,
      parent_pid: null,
      image_id: "coding-agent:v0",
      llm_profile_id: "default",
      status: "runnable",
      goal_oid: null,
      checkpoint_head: null,
      working_directory: ".",
      status_message: null,
      wait_state: null,
      outcome: null,
      state_generation: 0,
      loaded_skills: {},
      tool_table: {},
      capabilities: [],
      terminal: false,
      unread_message_count: 0,
      interrupt_count: 0,
      messages: [],
      llm_call_count: 0,
      token_total: 0,
      rating: null
    })),
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

function taskRun(runId: string, revision: number, status: "queued" | "running" | "needs_attention") {
  return {
    schema_version: 1 as const,
    run_id: runId,
    revision,
    status,
    display_title: runId,
    root_pid: "pid_1",
    active_pid: "pid_1",
    allowed_actions: status === "needs_attention" ? ["recover" as const, "cancel" as const] : ["run" as const],
    blockers: status === "needs_attention" ? [{ kind: "unknown_effect" as const }] : [],
    retention: "purge_on_terminal" as const,
    payloads_purged: false
  };
}
