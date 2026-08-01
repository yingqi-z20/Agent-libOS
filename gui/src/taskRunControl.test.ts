import { describe, expect, it, vi } from "vitest";
import type { TaskRunDetail, TaskRunSpecV1, TaskRunSummary } from "./api/types";
import {
  bindTaskRunStartIntent,
  createAndRunTaskRun,
  clearTaskRunFollowUpDraft,
  rotateUnadmittedTaskRunStartCommand,
  submitTaskRunFollowUp,
  taskRunFollowUpIntent,
  taskRunMutationIntent,
  taskRunStartIntent
} from "./taskRunControl";

describe("standard-user durable run flow", () => {
  it("reuses the create id for the same unresolved intent", () => {
    const spec = taskRunSpec();
    const makeId = vi.fn((kind: "create" | "run") => `${kind}-${makeId.mock.calls.length}`);
    const first = taskRunStartIntent(null, spec, 7, makeId);
    const retry = taskRunStartIntent(first, { ...spec, launch_options: { working_directory: "." } }, 7, makeId);

    expect(retry).toBe(first);
    expect(makeId).toHaveBeenCalledTimes(2);
  });

  it("keeps create identity but rotates only the run command when quanta changes", () => {
    const spec = taskRunSpec();
    const makeId = vi.fn((kind: "create" | "run") => `${kind}-${makeId.mock.calls.length}`);
    const first = taskRunStartIntent(null, spec, 7, makeId);
    const changedRun = taskRunStartIntent(first, spec, 8, makeId);

    expect(changedRun.clientRequestId).toBe(first.clientRequestId);
    expect(changedRun.runCommandId).not.toBe(first.runCommandId);
    expect(changedRun.specFingerprint).toBe(first.specFingerprint);
    expect(changedRun.runFingerprint).not.toBe(first.runFingerprint);
    expect(makeId).toHaveBeenCalledTimes(3);
  });

  it("keeps the original run revision after a create replay returns a newer projection", () => {
    const makeId = vi.fn((kind: "create" | "run") => `${kind}-stable`);
    const unbound = taskRunStartIntent(null, taskRunSpec(), 7, makeId);
    const bound = bindTaskRunStartIntent(unbound, summary(3, "queued"));
    const replayed = bindTaskRunStartIntent(bound, summary(9, "running"));
    const rotated = rotateUnadmittedTaskRunStartCommand(replayed, () => "run-new");

    expect(replayed).toBe(bound);
    expect(replayed.runExpectedRevision).toBe(3);
    expect(rotated).toMatchObject({
      clientRequestId: bound.clientRequestId,
      runCommandId: "run-new",
      runId: null,
      runExpectedRevision: null
    });
  });

  it("keeps mutation revision, command, and request across an SSE race", () => {
    const makeCommandId = vi.fn(() => "run-stable");
    const first = taskRunMutationIntent(null, {
      runId: "run_1",
      action: "run",
      expectedRevision: 7,
      request: { max_quanta: 8 }
    }, makeCommandId);
    const retryAfterSse = taskRunMutationIntent(first, {
      runId: "run_1",
      action: "run",
      expectedRevision: 9,
      request: { max_quanta: 8 }
    }, makeCommandId);

    expect(retryAfterSse).toBe(first);
    expect(retryAfterSse).toMatchObject({
      expectedRevision: 7,
      commandId: "run-stable"
    });
    expect(makeCommandId).toHaveBeenCalledTimes(1);
  });

  it("allocates a new mutation command only when the canonical request changes", () => {
    const makeCommandId = vi.fn()
      .mockReturnValueOnce("recover-1")
      .mockReturnValueOnce("recover-2");
    const first = taskRunMutationIntent(null, {
      runId: "run_1",
      action: "recover",
      expectedRevision: 7,
      request: { option_id: "receipt", receipt: { b: 2, a: 1 } }
    }, makeCommandId);
    const reordered = taskRunMutationIntent(first, {
      runId: "run_1",
      action: "recover",
      expectedRevision: 8,
      request: { receipt: { a: 1, b: 2 }, option_id: "receipt" }
    }, makeCommandId);
    const changed = taskRunMutationIntent(reordered, {
      runId: "run_1",
      action: "recover",
      expectedRevision: 8,
      request: { receipt: { a: 1, b: 3 }, option_id: "receipt" }
    }, makeCommandId);

    expect(reordered).toBe(first);
    expect(changed.commandId).toBe("recover-2");
    expect(changed.expectedRevision).toBe(8);
    expect(makeCommandId).toHaveBeenCalledTimes(2);
  });

  it("creates queued state then runs it with the returned revision and quanta", async () => {
    const created = summary(3, "queued");
    const running = summary(4, "waiting_human");
    const client = {
      createTaskRun: vi.fn().mockResolvedValue(created),
      runTaskRun: vi.fn().mockResolvedValue(running),
      getTaskRun: vi.fn()
    };
    const observed: TaskRunSummary[] = [];

    await expect(createAndRunTaskRun(
      client,
      taskRunSpec(),
      {
        fingerprint: "f",
        specFingerprint: "spec",
        runFingerprint: "run",
        clientRequestId: "create-stable",
        runCommandId: "run-stable",
        runId: null,
        runExpectedRevision: null
      },
      7,
      { onCreated: vi.fn(), onIntent: vi.fn(), onSummary: (value) => observed.push(value) }
    )).resolves.toBe(running);

    expect(client.createTaskRun).toHaveBeenCalledWith(taskRunSpec(), "create-stable");
    expect(client.runTaskRun).toHaveBeenCalledWith("run_1", 3, "run-stable", 7);
    expect(observed.map((item) => item.revision)).toEqual([3, 4]);
  });

  it("reconciles an ambiguous run response without dispatching a second command", async () => {
    const created = summary(3, "queued");
    const reconciled = summary(4, "running");
    const error = new Error("connection lost");
    const client = {
      createTaskRun: vi.fn().mockResolvedValue(created),
      runTaskRun: vi.fn().mockRejectedValue(error),
      getTaskRun: vi.fn().mockResolvedValue(detail(reconciled))
    };
    const observed: TaskRunSummary[] = [];

    await expect(createAndRunTaskRun(
      client,
      taskRunSpec(),
      {
        fingerprint: "f",
        specFingerprint: "spec",
        runFingerprint: "run",
        clientRequestId: "create-stable",
        runCommandId: "run-stable",
        runId: null,
        runExpectedRevision: null
      },
      null,
      { onCreated: vi.fn(), onIntent: vi.fn(), onSummary: (value) => observed.push(value) }
    )).rejects.toBe(error);

    expect(client.runTaskRun).toHaveBeenCalledTimes(1);
    expect(client.getTaskRun).toHaveBeenCalledWith("run_1");
    expect(observed.map((item) => item.revision)).toEqual([3, 4]);
  });

  it("retries create then run with the originally bound run revision and command", async () => {
    const created = summary(3, "queued");
    const latestCreateReplay = summary(6, "running");
    const completed = summary(7, "waiting_human");
    const responseLost = new Error("response lost");
    const client = {
      createTaskRun: vi.fn()
        .mockResolvedValueOnce(created)
        .mockResolvedValueOnce(latestCreateReplay),
      runTaskRun: vi.fn()
        .mockRejectedValueOnce(responseLost)
        .mockResolvedValueOnce(completed),
      getTaskRun: vi.fn().mockResolvedValue(detail(latestCreateReplay))
    };
    let intent = taskRunStartIntent(
      null,
      taskRunSpec(),
      7,
      (kind) => `${kind}-stable`
    );
    const callbacks = {
      onCreated: vi.fn(),
      onIntent: (next: typeof intent) => { intent = next; },
      onSummary: vi.fn()
    };

    await expect(createAndRunTaskRun(
      client,
      taskRunSpec(),
      intent,
      7,
      callbacks
    )).rejects.toBe(responseLost);
    await expect(createAndRunTaskRun(
      client,
      taskRunSpec(),
      intent,
      7,
      callbacks
    )).resolves.toBe(completed);

    expect(client.createTaskRun).toHaveBeenNthCalledWith(1, taskRunSpec(), "create-stable");
    expect(client.createTaskRun).toHaveBeenNthCalledWith(2, taskRunSpec(), "create-stable");
    expect(client.runTaskRun.mock.calls[1]).toEqual(client.runTaskRun.mock.calls[0]);
    expect(client.runTaskRun.mock.calls[0]).toEqual(["run_1", 3, "run-stable", 7]);
  });

  it("keeps the original follow-up revision, body, and command across an SSE race", async () => {
    const makeCommandId = vi.fn(() => "follow-up-stable");
    const first = await taskRunFollowUpIntent(null, {
      runId: "run_1",
      expectedRevision: 7,
      body: "  追加检查  ",
      kind: "normal",
      required: true
    }, makeCommandId);
    const retryAfterSse = await taskRunFollowUpIntent(first, {
      runId: "run_1",
      expectedRevision: 8,
      body: "追加检查",
      kind: "normal",
      required: true
    }, makeCommandId);

    expect(retryAfterSse).toBe(first);
    expect(first).toMatchObject({
      expectedRevision: 7,
      body: "追加检查",
      requestHash: "e49eae85ce28f9924ef07e364f30a52426ab4d94ce21920493d224b04f2b7d33",
      commandId: "follow-up-stable"
    });
    expect(makeCommandId).toHaveBeenCalledTimes(1);

    const response = summary(8, "waiting_message");
    const client = { followUpTaskRun: vi.fn().mockResolvedValue(response) };
    await expect(submitTaskRunFollowUp(client, retryAfterSse)).resolves.toBe(response);
    expect(client.followUpTaskRun).toHaveBeenCalledWith(
      "run_1",
      "追加检查",
      7,
      "follow-up-stable",
      { kind: "normal", required: true }
    );
  });

  it("clears only the exact draft after this command receives an HTTP success", async () => {
    const intent = await taskRunFollowUpIntent(null, {
      runId: "run_1",
      expectedRevision: 7,
      body: "追加检查",
      kind: "normal",
      required: true
    }, () => "follow-up-stable");
    expect(clearTaskRunFollowUpDraft({
      "run:run_1": "  追加检查  ",
      pid_1: "keep process draft"
    }, intent)).toEqual({ pid_1: "keep process draft" });
    expect(clearTaskRunFollowUpDraft({
      "run:run_1": "new draft"
    }, intent)).toEqual({ "run:run_1": "new draft" });
  });
});

function taskRunSpec(): TaskRunSpecV1 {
  return {
    schema_version: 1,
    goal: "finish",
    display_title: "Finish",
    image_id: "coding-agent:v0",
    launch_options: { working_directory: "." },
    retention: "purge_on_terminal"
  };
}

function summary(revision: number, status: TaskRunSummary["status"]): TaskRunSummary {
  return {
    schema_version: 1,
    run_id: "run_1",
    revision,
    status,
    display_title: "Finish",
    root_pid: "pid_1",
    active_pid: "pid_1",
    allowed_actions: status === "queued" ? ["run", "cancel"] : ["pause", "cancel", "follow_up"],
    blockers: [],
    retention: "purge_on_terminal",
    payloads_purged: false
  };
}

function detail(value: TaskRunSummary): TaskRunDetail {
  return {
    summary: value,
    requirements: { items: [], next_cursor: null, has_more: false },
    recovery_options: []
  };
}
