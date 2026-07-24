import { describe, expect, it } from "vitest";
import type { RuntimeProcess, RuntimeSnapshot } from "./api/types";
import { processFromMutationResult, reconcileSelectedPid, upsertRuntimeProcess } from "./selection";

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

  it("extracts only process-shaped direct or wrapped mutation responses", () => {
    const process = snapshot(["pid_1"]).processes[0];
    expect(processFromMutationResult(process)).toBe(process);
    expect(processFromMutationResult({ process, scheduler: { running: true } })).toBe(process);
    expect(processFromMutationResult({ running: true })).toBeNull();
    expect(processFromMutationResult(null)).toBeNull();
  });
});

function snapshot(pids: string[]): RuntimeSnapshot {
  return {
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
    tools: [],
    llm_profiles: [],
    images: [],
    skills: [],
    jsonrpc_endpoints: [],
    mcp_servers: [],
    modules: []
  };
}
