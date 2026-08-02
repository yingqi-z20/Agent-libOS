// @vitest-environment jsdom

import { getByRole, waitFor } from "@testing-library/dom";
import userEvent from "@testing-library/user-event";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { LibOSClient } from "./api/client";
import type {
  RuntimeSnapshot,
  SseMessage,
  StreamConnectionStatus,
  TaskRunDetail,
  TaskRunSummary
} from "./api/types";
import { I18nProvider } from "./i18n";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type StreamSession = {
  onMessage(message: SseMessage): void;
  onStatus?: (status: StreamConnectionStatus) => void;
};

const mounted: Array<{ root: Root; container: HTMLDivElement }> = [];

afterEach(async () => {
  for (const { root, container } of mounted.splice(0)) {
    await act(() => root.unmount());
    container.remove();
  }
  localStorage.clear();
  sessionStorage.clear();
  delete window.libosApi;
  vi.restoreAllMocks();
});

describe("mounted App TaskRun stream reconciliation", () => {
  it("lets an invalidation HTTP snapshot land after a lower TaskRun SSE revision is ignored", async () => {
    const refresh = deferred<RuntimeSnapshot>();
    const snapshotSpy = vi.spyOn(LibOSClient.prototype, "snapshot")
      .mockResolvedValueOnce(runtimeSnapshot(5))
      .mockReturnValueOnce(refresh.promise);
    const streams = installPersistentStreamMock();
    installTaskRunReadMocks();
    const container = await renderApp();

    await waitFor(() => expect(runRevisionText(container)).toContain("r5"));
    expect(streams).toHaveLength(1);

    await act(async () => {
      streams[0].onMessage({ id: "10", event: "event.invalidated", data: { scope: "task_run" } });
    });
    await waitFor(() => expect(snapshotSpy).toHaveBeenCalledTimes(2));

    await act(async () => {
      streams[0].onMessage({ id: "11", event: "task_run.updated", data: taskRun(4) });
      refresh.resolve(runtimeSnapshot(6));
      await refresh.promise;
    });

    await waitFor(() => expect(runRevisionText(container)).toContain("r6"));
    expect(runRevisionText(container)).not.toContain("r4");
  });

  it("refreshes and resubscribes after terminal stream failure without rolling back a newer Run", async () => {
    const snapshotSpy = vi.spyOn(LibOSClient.prototype, "snapshot")
      .mockResolvedValueOnce(runtimeSnapshot(7))
      .mockResolvedValueOnce(runtimeSnapshot(6));
    const streams: StreamSession[] = [];
    const streamSpy = vi.spyOn(LibOSClient.prototype, "stream")
      .mockImplementation((onMessage, signal, _cursor, onStatus) => {
        streams.push({ onMessage, onStatus });
        if (streams.length === 1) {
          onStatus?.("connected");
          onStatus?.("failed");
          return Promise.reject(new Error("terminal SSE failure"));
        }
        onStatus?.("connected");
        return resolveWhenAborted(signal);
      });
    installTaskRunReadMocks();
    const container = await renderApp();

    await waitFor(() => expect(runRevisionText(container)).toContain("r7"));
    await waitFor(() => expect(container.textContent).toContain("terminal SSE failure"));
    const retry = getByRole(container, "button", { name: "Retry" });
    const user = userEvent.setup();

    retry.focus();
    await act(async () => user.keyboard("{Enter}"));

    await waitFor(() => expect(snapshotSpy).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(streamSpy).toHaveBeenCalledTimes(2));
    expect(streams).toHaveLength(2);
    expect(runRevisionText(container)).toContain("r7");
    expect(runRevisionText(container)).not.toContain("r6");
  });

  it("reconciles a lost lifecycle response and retries with the original revision and command", async () => {
    const snapshotSpy = vi.spyOn(LibOSClient.prototype, "snapshot")
      .mockResolvedValueOnce(runtimeSnapshot(7))
      .mockResolvedValueOnce(runtimeSnapshot(8))
      .mockResolvedValueOnce(runtimeSnapshot(9));
    const resumeSpy = vi.spyOn(LibOSClient.prototype, "resumeTaskRun")
      .mockRejectedValueOnce(new Error("response lost"))
      .mockResolvedValueOnce(taskRun(9));
    installPersistentStreamMock();
    installTaskRunReadMocks();
    const container = await renderApp();
    const user = userEvent.setup();

    await waitFor(() => expect(runRevisionText(container)).toContain("r7"));
    const resume = getByRole(container, "button", { name: "Resume" });
    await act(async () => user.click(resume));
    await waitFor(() => expect(snapshotSpy).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(runRevisionText(container)).toContain("r8"));
    await act(async () => user.click(getByRole(container, "button", { name: "Resume" })));
    await waitFor(() => expect(resumeSpy).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(snapshotSpy).toHaveBeenCalledTimes(3));

    expect(resumeSpy.mock.calls[1]).toEqual(resumeSpy.mock.calls[0]);
    expect(resumeSpy.mock.calls[0][0]).toBe("run_stream_test");
    expect(resumeSpy.mock.calls[0][1]).toBe(7);
    expect(resumeSpy.mock.calls[0][2]).toMatch(/^gui:resume:/);
  });
});

function installPersistentStreamMock(): StreamSession[] {
  const streams: StreamSession[] = [];
  vi.spyOn(LibOSClient.prototype, "stream").mockImplementation((onMessage, signal, _cursor, onStatus) => {
    streams.push({ onMessage, onStatus });
    onStatus?.("connected");
    return resolveWhenAborted(signal);
  });
  return streams;
}

function installTaskRunReadMocks() {
  vi.spyOn(LibOSClient.prototype, "getTaskRun").mockResolvedValue(taskRunDetail(taskRun(0)));
  vi.spyOn(LibOSClient.prototype, "listTaskRunHumanRequests").mockResolvedValue({
    items: [],
    next_cursor: null,
    has_more: false,
    presentation_truncated: false
  });
}

async function renderApp(): Promise<HTMLDivElement> {
  localStorage.setItem("agent-libos.gui.view", "user");
  window.libosApi = {
    getConnection: vi.fn().mockResolvedValue({
      url: "http://127.0.0.1:1",
      token: "test-token",
      db: "test.sqlite"
    }),
    chooseDatabase: vi.fn().mockResolvedValue(null),
    chooseImagePackage: vi.fn().mockResolvedValue(null),
    openExternal: vi.fn().mockResolvedValue(true)
  };
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  mounted.push({ root, container });
  await act(async () => {
    root.render(
      <I18nProvider initialLanguage="en">
        <App />
      </I18nProvider>
    );
  });
  return container;
}

function runRevisionText(container: HTMLElement): string {
  return container.querySelector(".conversationTitle p")?.textContent ?? "";
}

function runtimeSnapshot(revision: number): RuntimeSnapshot {
  return {
    schema_version: 3,
    db: "test.sqlite",
    scheduler: {
      auto_run: false,
      running: false,
      paused: true,
      task_id: null,
      reason: null,
      last_result: [],
      last_error: null,
      started_at: null,
      finished_at: null,
      default_max_quanta: null
    },
    processes: [],
    human_requests: [],
    events: [],
    audit: [],
    llm_calls: [],
    object_tasks: [],
    task_runs: [taskRun(revision)],
    tools: [],
    llm_profiles: [],
    images: [],
    skills: [],
    jsonrpc_endpoints: [],
    mcp_servers: [],
    modules: []
  };
}

function taskRun(revision: number): TaskRunSummary {
  return {
    schema_version: 1,
    run_id: "run_stream_test",
    revision,
    status: "paused",
    display_title: "Stream reconciliation",
    root_pid: null,
    active_pid: null,
    allowed_actions: ["resume", "cancel", "follow_up"],
    blockers: [],
    retention: "purge_on_terminal",
    payloads_purged: false,
    requirement_counts: { total: 1, satisfied: 0 }
  };
}

function taskRunDetail(summary: TaskRunSummary): TaskRunDetail {
  return {
    summary,
    requirements: { items: [], next_cursor: null, has_more: false },
    recovery_options: []
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

function resolveWhenAborted(signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.resolve();
  return new Promise((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
}
