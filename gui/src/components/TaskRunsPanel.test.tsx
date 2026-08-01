// @vitest-environment jsdom

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { ApiError, TaskRunMutationError } from "../api/client";
import type { LibOSClient } from "../api/client";
import type { TaskRunDetail, TaskRunLedgerItem, TaskRunSummary } from "../api/types";
import type { ConfirmationRequest, RunGuiAction } from "../adminTypes";
import { I18nProvider } from "../i18n";
import "../styles.css";
import { TaskRunsPanel } from "./TaskRunsPanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mounted: Array<{ root: Root; container: HTMLDivElement }> = [];

afterEach(async () => {
  for (const { root, container } of mounted.splice(0)) {
    await act(() => root.unmount());
    container.remove();
  }
});

describe("TaskRunsPanel", () => {
  it("localizes durable enums in a 360px panel and wraps narrow controls", async () => {
    const { container } = await renderPanel("zh-CN");
    const panel = required(container.querySelector<HTMLElement>(".taskRunsPanel"));
    const lifecycle = required(container.querySelector<HTMLElement>(".taskRunLifecycleActions"));
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

    expect(container.style.width).toBe("360px");
    expect(panel.scrollWidth).toBeLessThanOrEqual(panel.clientWidth);
    expect(panel.classList).toContain("taskRunsPanel");
    expect(lifecycle.classList).toContain("taskRunLifecycleActions");
    expect(styles).toMatch(/\.taskRunsPanel\s*\{[^}]*overflow-wrap:\s*anywhere;/s);
    expect(styles).toContain("grid-template-columns: repeat(auto-fit, minmax(min(100%, 8rem), 1fr));");
    expect(styles).toContain("@media (max-width: 420px)");
    expect(container.querySelector(".taskRunBlockers")?.textContent).toContain("未知外部效果");
    expect(container.querySelector(".taskRunRequirements")?.textContent).toContain("后续要求 · 处理中");
    expect(container.querySelector(".taskRunRequirements")?.textContent).toContain("已满足要求：1/3");
    expect(container.querySelector(".taskRunRetentionPolicy")?.textContent).toContain("永久保留（Host 显式清除除外）");
    expect(container.querySelector(".taskRunPayloadState")?.textContent).toContain("已保留且可取回");
    expect(container.querySelector(".taskRunRecoveryControls")?.textContent).toContain(
      "登记 Effect eff_1 的 Provider Receipt（状态：已派发，Epoch：12）"
    );
    expect(container.querySelector(".taskRunLedgerIdentity")?.textContent).toContain("外部 Effect");
    expect(container.querySelector(".taskRunLedgerIdentity")?.textContent).toContain("追加要求");
    expect(container.querySelector(".taskRunLedgerStatus")?.textContent).toBe("处理中");

    const visibleText = container.textContent ?? "";
    for (const raw of ["needs_attention", "unknown_effect", "follow_up", "in_progress", "permanent", "effect_receipt", "dispatched"]) {
      expect(visibleText).not.toContain(raw);
    }
  });

  it("keeps every visible control keyboard-focusable and hides ordinary Run/Resume for needs_attention", async () => {
    const { container } = await renderPanel("en");
    const panel = required(container.querySelector<HTMLElement>(".taskRunsPanel"));
    const select = required(panel.querySelector<HTMLSelectElement>("select"));

    expect(panel.getAttribute("aria-labelledby")).toBe("task-runs-panel-title");
    expect(select.labels?.[0]?.textContent).toContain("Selected run");
    expect(buttonWithExactText(panel, "Run")).toBeNull();
    expect(buttonWithExactText(panel, "Resume")).toBeNull();
    expect(buttonWithExactText(panel, "Create rerun")?.disabled).toBe(false);
    expect(panel.querySelector(".taskRunRerunGoal")).toBeNull();
    expect(buttonWithExactText(panel, "Cancel")).not.toBeNull();
    expect(panel.querySelector('[role="status"]')?.textContent).toContain("Host attention");

    const controls = [...panel.querySelectorAll<HTMLElement>("button:not(:disabled), select:not(:disabled), input:not(:disabled), textarea:not(:disabled)")];
    expect(controls.length).toBeGreaterThan(4);
    for (const control of controls) {
      control.focus();
      expect(document.activeElement).toBe(control);
      if (control instanceof HTMLButtonElement) {
        expect(control.textContent?.trim()).not.toBe("");
      } else {
        const field = control as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;
        const labelled = field.getAttribute("aria-label") || field.labels?.item(0)?.textContent;
        expect(labelled).toBeTruthy();
      }
    }
  });

  it("requires a replacement goal when rerunning a purged terminal run", async () => {
    const run = purgedTerminalSummary();
    const { container, client } = await renderPanel("en", run);
    const panel = required(container.querySelector<HTMLElement>(".taskRunsPanel"));
    const goal = required(panel.querySelector<HTMLTextAreaElement>(".taskRunRerunGoal textarea"));
    const rerunButton = required(buttonWithExactText(panel, "Create rerun"));

    expect(goal.labels?.[0]?.textContent).toContain("Replacement goal");
    expect(panel.querySelector(".taskRunRerunGoal")?.textContent).toContain(
      "cannot reuse the original goal"
    );
    expect(rerunButton.disabled).toBe(true);

    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(
        goal,
        "Rebuild the release evidence from the retained repository state."
      );
      goal.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(rerunButton.disabled).toBe(false);

    await act(async () => {
      rerunButton.click();
      await Promise.resolve();
    });
    expect(client.rerunTaskRun).toHaveBeenCalledWith(
      run.run_id,
      run.revision,
      expect.stringContaining("rerun"),
      {
        specOverrides: {
          goal: "Rebuild the release evidence from the retained repository state."
        }
      }
    );
  });

  it("also requires a replacement goal after an explicit purge of permanent retention", async () => {
    const run = {
      ...purgedTerminalSummary(),
      run_id: "run_explicitly_purged",
      retention: "permanent" as const,
      payloads_purged: true
    };
    const { container } = await renderPanel("en", run);
    const panel = required(container.querySelector<HTMLElement>(".taskRunsPanel"));

    expect(panel.querySelector<HTMLTextAreaElement>(".taskRunRerunGoal textarea")).not.toBeNull();
    expect(required(buttonWithExactText(panel, "Create rerun")).disabled).toBe(true);
    expect(panel.querySelector(".taskRunRetentionPolicy")?.textContent).toContain(
      "Permanent unless explicitly purged by Host"
    );
    expect(panel.querySelector(".taskRunPayloadState")?.textContent).toContain(
      "Purged; content is unavailable"
    );
  });

  it("retries an ambiguous Operator follow-up with the same immutable intent and clears only after success", async () => {
    const run = {
      ...summary(),
      status: "paused" as const,
      allowed_actions: ["resume", "cancel", "follow_up"] as TaskRunSummary["allowed_actions"],
      blockers: []
    };
    const accepted = { ...run, revision: run.revision + 1 };
    const followUpTaskRun = vi.fn()
      .mockRejectedValueOnce(new Error("response lost"))
      .mockResolvedValueOnce(accepted);
    const runAction: RunGuiAction = async (operation) => {
      try {
        await operation();
        return true;
      } catch {
        return false;
      }
    };
    const { container, rerender } = await renderPanel("en", run, { followUpTaskRun, runAction });
    const input = required(container.querySelector<HTMLInputElement>('input[aria-label="Add follow-up"]'));
    const send = required(buttonWithExactText(container, "Add follow-up"));

    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(input, "Keep the same durable requirement");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => {
      send.click();
      await vi.waitFor(() => expect(followUpTaskRun).toHaveBeenCalledTimes(1));
    });
    expect(input.value).toBe("Keep the same durable requirement");
    await rerender({ ...run, revision: run.revision + 1 });
    expect(input.value).toBe("Keep the same durable requirement");

    await act(async () => {
      send.click();
      await vi.waitFor(() => expect(followUpTaskRun).toHaveBeenCalledTimes(2));
    });
    expect(followUpTaskRun).toHaveBeenCalledTimes(2);
    expect(followUpTaskRun.mock.calls[1]).toEqual(followUpTaskRun.mock.calls[0]);
    expect(followUpTaskRun.mock.calls[0][2]).toBe(run.revision);
    expect(followUpTaskRun.mock.calls[0][3]).toMatch(/^gui:follow_up:/);
    expect(input.value).toBe("");
  });

  it("retries an ambiguous lifecycle mutation with the original revision and command", async () => {
    const run = {
      ...summary(),
      status: "paused" as const,
      allowed_actions: ["resume", "cancel", "follow_up"] as TaskRunSummary["allowed_actions"],
      blockers: []
    };
    const resumeTaskRun = vi.fn()
      .mockRejectedValueOnce(new Error("response lost"))
      .mockResolvedValueOnce({ ...run, revision: run.revision + 1 });
    const runAction: RunGuiAction = async (operation) => {
      try {
        await operation();
        return true;
      } catch {
        return false;
      }
    };
    const { container } = await renderPanel("en", run, {
      runAction,
      clientOverrides: { resumeTaskRun }
    });
    const resume = required(buttonWithExactText(container, "Resume"));

    await act(async () => {
      resume.click();
      await vi.waitFor(() => expect(resumeTaskRun).toHaveBeenCalledTimes(1));
    });
    await act(async () => {
      resume.click();
      await vi.waitFor(() => expect(resumeTaskRun).toHaveBeenCalledTimes(2));
    });

    expect(resumeTaskRun.mock.calls[1]).toEqual(resumeTaskRun.mock.calls[0]);
    expect(resumeTaskRun.mock.calls[0][1]).toBe(run.revision);
    expect(resumeTaskRun.mock.calls[0][2]).toMatch(/^gui:resume:/);
  });

  it("keeps a rerun intent and replacement goal across a source revision SSE race", async () => {
    const run = purgedTerminalSummary();
    const rerunResult = {
      ...run,
      run_id: "run_rerun",
      revision: 0,
      status: "queued" as const,
      allowed_actions: ["run"] as TaskRunSummary["allowed_actions"]
    };
    const rerunTaskRun = vi.fn()
      .mockRejectedValueOnce(new Error("response lost"))
      .mockResolvedValueOnce(rerunResult);
    const runAction: RunGuiAction = async (operation) => {
      try {
        await operation();
        return true;
      } catch {
        return false;
      }
    };
    const { container, rerender } = await renderPanel("en", run, {
      runAction,
      clientOverrides: { rerunTaskRun }
    });
    const goal = required(container.querySelector<HTMLTextAreaElement>(".taskRunRerunGoal textarea"));
    const rerun = required(buttonWithExactText(container, "Create rerun"));

    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(goal, "replacement");
      goal.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => {
      rerun.click();
      await vi.waitFor(() => expect(rerunTaskRun).toHaveBeenCalledTimes(1));
    });
    await rerender({ ...run, revision: run.revision + 1 });
    expect(goal.value).toBe("replacement");

    await act(async () => {
      rerun.click();
      await vi.waitFor(() => expect(rerunTaskRun).toHaveBeenCalledTimes(2));
    });
    expect(rerunTaskRun.mock.calls[1]).toEqual(rerunTaskRun.mock.calls[0]);
    expect(rerunTaskRun.mock.calls[0][1]).toBe(run.revision);
  });

  it("keeps explicit cancel confirmation and its command stable after response loss", async () => {
    const run = {
      ...summary(),
      status: "paused" as const,
      allowed_actions: ["cancel"] as TaskRunSummary["allowed_actions"],
      blockers: []
    };
    const cancelTaskRun = vi.fn()
      .mockRejectedValueOnce(new Error("response lost"))
      .mockResolvedValueOnce({ ...run, status: "cancelled", revision: run.revision + 1 });
    const { container, confirmAction } = await renderPanel("en", run, {
      clientOverrides: { cancelTaskRun }
    });

    await act(async () => required(buttonWithExactText(container, "Cancel")).click());
    const confirmation = required(confirmAction.mock.calls[0]?.[0] ?? null);
    await act(async () => {
      await expect(confirmation.action()).rejects.toThrow("response lost");
    });
    await act(async () => {
      await expect(confirmation.action()).resolves.toBeUndefined();
    });

    expect(cancelTaskRun.mock.calls[1]).toEqual(cancelTaskRun.mock.calls[0]);
    expect(cancelTaskRun.mock.calls[0][3]).toBe(true);
  });

  it("rotates only a proven-unadmitted revision conflict after a fresh explicit confirmation", async () => {
    const run = {
      ...summary(),
      status: "paused" as const,
      allowed_actions: ["cancel"] as TaskRunSummary["allowed_actions"],
      blockers: []
    };
    const latest = { ...run, revision: run.revision + 1 };
    const conflict = new TaskRunMutationError(new ApiError("stale", 409, {
      error: {
        code: "task_run_revision_conflict",
        command_admitted: false,
        current_summary: latest
      }
    }), latest);
    const cancelTaskRun = vi.fn()
      .mockRejectedValueOnce(conflict)
      .mockResolvedValueOnce({ ...latest, status: "cancelled", revision: latest.revision + 1 });
    const { container, confirmAction, rerender } = await renderPanel("en", run, {
      clientOverrides: { cancelTaskRun }
    });

    await act(async () => required(buttonWithExactText(container, "Cancel")).click());
    const first = required(confirmAction.mock.calls[0]?.[0] ?? null);
    await act(async () => {
      await expect(first.action()).rejects.toBe(conflict);
    });
    expect(first.onErrorReconciled?.(conflict)).toBe(true);
    await rerender(latest);
    await act(async () => required(buttonWithExactText(container, "Cancel")).click());
    const second = required(confirmAction.mock.calls[1]?.[0] ?? null);
    await act(async () => {
      await expect(second.action()).resolves.toBeUndefined();
    });

    expect(cancelTaskRun.mock.calls[0][1]).toBe(run.revision);
    expect(cancelTaskRun.mock.calls[1][1]).toBe(latest.revision);
    expect(cancelTaskRun.mock.calls[1][2]).not.toBe(cancelTaskRun.mock.calls[0][2]);
    expect(cancelTaskRun.mock.calls[0][3]).toBe(true);
    expect(cancelTaskRun.mock.calls[1][3]).toBe(true);
  });

  it("keeps explicit recovery confirmation, receipt, and command stable after response loss", async () => {
    const run = summary();
    const recoverTaskRun = vi.fn()
      .mockRejectedValueOnce(new Error("response lost"))
      .mockResolvedValueOnce({ ...run, revision: run.revision + 1 });
    const { container, confirmAction } = await renderPanel("en", run, {
      clientOverrides: { recoverTaskRun }
    });
    const receipt = required(container.querySelector<HTMLTextAreaElement>(".taskRunRecoveryControls textarea"));
    const recover = required(container.querySelector<HTMLButtonElement>(".taskRunRecoveryControls button.warning"));

    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(
        receipt,
        '{"receipt_id":"provider-1"}'
      );
      receipt.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => recover.click());
    const confirmation = required(confirmAction.mock.calls[0]?.[0] ?? null);
    await act(async () => {
      await expect(confirmation.action()).rejects.toThrow("response lost");
    });
    await act(async () => {
      await expect(confirmation.action()).resolves.toBeUndefined();
    });

    expect(recoverTaskRun.mock.calls[1]).toEqual(recoverTaskRun.mock.calls[0]);
    expect(recoverTaskRun.mock.calls[0][4]).toBe(true);
    expect(recoverTaskRun.mock.calls[0][5]).toEqual({ receipt_id: "provider-1" });
  });

  it("supports TaskRun lifecycle and follow-up controls through Tab and Enter", async () => {
    const run = {
      ...summary(),
      status: "paused" as const,
      allowed_actions: ["resume", "cancel", "follow_up"] as TaskRunSummary["allowed_actions"],
      blockers: []
    };
    const { container, client, confirmAction } = await renderPanel("en", run);
    const panel = required(container.querySelector<HTMLElement>(".taskRunsPanel"));
    const select = required(panel.querySelector<HTMLSelectElement>("select"));
    const user = userEvent.setup();

    select.focus();
    await act(async () => user.tab());
    expect(document.activeElement).toBe(buttonWithExactText(panel, "Resume"));
    await act(async () => user.keyboard("{Enter}"));
    await vi.waitFor(() => expect(client.resumeTaskRun).toHaveBeenCalledTimes(1));

    await act(async () => user.tab());
    expect(document.activeElement).toBe(buttonWithExactText(panel, "Cancel"));
    await act(async () => user.keyboard("{Enter}"));
    expect(confirmAction).toHaveBeenCalledTimes(1);

    const input = required(panel.querySelector<HTMLInputElement>('input[aria-label="Add follow-up"]'));
    input.focus();
    await act(async () => user.type(input, "Keyboard follow-up"));
    await act(async () => user.tab());
    expect(document.activeElement).toBe(buttonWithExactText(panel, "Add follow-up"));
    await act(async () => user.keyboard("{Enter}"));
    await vi.waitFor(() => expect(client.followUpTaskRun).toHaveBeenCalledTimes(1));
  });
});

async function renderPanel(
  language: "en" | "zh-CN",
  run: TaskRunSummary = summary(),
  options: {
    followUpTaskRun?: ReturnType<typeof vi.fn>;
    runAction?: RunGuiAction;
    clientOverrides?: Record<string, unknown>;
  } = {}
) {
  let currentRun = run;
  const detail: TaskRunDetail = {
    summary: run,
    requirements: {
      items: [{
        schema_version: 1,
        requirement_id: "req_1",
        run_id: run.run_id,
        ordinal: 1,
        kind: "follow_up",
        status: "in_progress",
        requirement_sha256: "a".repeat(64),
        label: "Verify provider receipt",
        created_by: "host",
        created_at: "2026-07-31T00:00:00Z",
        updated_at: "2026-07-31T00:00:01Z",
        started_at: "2026-07-31T00:00:01Z",
        completed_at: null,
        waived_by: null,
        content_available: true,
        content_retention: "plaintext",
        content_sha256: "b".repeat(64),
        content_text: "Reconcile the durable provider receipt.",
        content_truncated: false
      }],
      next_cursor: null,
      has_more: false
    },
    recovery_options: [{
      schema_version: 1,
      option_id: "register_receipt",
      kind: "effect_receipt",
      requires_receipt: true,
      effect_id: "eff_1",
      expected_transaction_state: "dispatched",
      runtime_epoch: 12
    }]
  };
  const ledger: TaskRunLedgerItem = {
    schema_version: 1,
    item_id: "ledger_1",
    run_id: run.run_id,
    seq: 1,
    kind: "effect",
    status: "in_progress",
    label: "follow_up",
    occurred_at: "2026-07-31T00:00:01Z",
    effect_id: "eff_1",
    metadata: {}
  };
  const client = {
    getTaskRun: vi.fn().mockImplementation(async () => ({ ...detail, summary: currentRun })),
    listTaskRunLedger: vi.fn().mockResolvedValue({ items: [ledger], next_cursor: null, has_more: false }),
    rerunTaskRun: vi.fn().mockResolvedValue({
      ...run,
      run_id: "run_rerun",
      revision: 0,
      status: "queued",
      allowed_actions: ["run"],
      blockers: []
    }),
    cancelTaskRun: vi.fn(),
    resumeTaskRun: vi.fn().mockResolvedValue({ ...run, revision: run.revision + 1 }),
    pauseTaskRun: vi.fn().mockResolvedValue({ ...run, revision: run.revision + 1 }),
    runTaskRun: vi.fn().mockResolvedValue({ ...run, revision: run.revision + 1 }),
    recoverTaskRun: vi.fn(),
    followUpTaskRun: options.followUpTaskRun ?? vi.fn().mockResolvedValue({ ...run, revision: run.revision + 1 }),
    explainOperation: vi.fn(),
    resolveOperation: vi.fn(),
    ...options.clientOverrides
  } as unknown as LibOSClient;
  const runAction: RunGuiAction = options.runAction ?? (async (operation) => {
    await operation();
    return true;
  });
  const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
  const container = document.createElement("div");
  container.style.width = "360px";
  document.body.append(container);
  const root = createRoot(container);
  mounted.push({ root, container });

  await act(async () => {
    root.render(
      <I18nProvider initialLanguage={language}>
        <TaskRunsPanel
          runs={[run]}
          client={client}
          runAction={runAction}
          confirmAction={confirmAction}
        />
      </I18nProvider>
    );
  });
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
  async function rerender(nextRun: TaskRunSummary) {
    currentRun = nextRun;
    await act(async () => {
      root.render(
        <I18nProvider initialLanguage={language}>
          <TaskRunsPanel
            runs={[nextRun]}
            client={client}
            runAction={runAction}
            confirmAction={confirmAction}
          />
        </I18nProvider>
      );
      await Promise.resolve();
    });
  }
  return { container, client, confirmAction, rerender };
}

function summary(): TaskRunSummary {
  return {
    schema_version: 1,
    run_id: "run_1",
    revision: 7,
    status: "needs_attention",
    display_title: "Provider reconciliation",
    root_pid: "pid_1",
    active_pid: null,
    allowed_actions: ["run", "resume", "recover", "rerun", "cancel", "follow_up"],
    blockers: [{
      kind: "unknown_effect",
      code: "provider_dispatch_unknown",
      evidence_ref: "evidence_1",
      effect_id: "eff_1"
    }],
    retention: "permanent",
    payloads_purged: false,
    requirement_counts: { total: 3, satisfied: 1, in_progress: 1, pending: 1 }
  };
}

function purgedTerminalSummary(): TaskRunSummary {
  return {
    ...summary(),
    run_id: "run_purged",
    revision: 11,
    status: "succeeded",
    active_pid: null,
    allowed_actions: ["rerun"],
    blockers: [],
    retention: "purge_on_terminal",
    payloads_purged: true
  };
}

function buttonWithExactText(container: ParentNode, text: string): HTMLButtonElement | null {
  return [...container.querySelectorAll<HTMLButtonElement>("button")]
    .find((button) => button.textContent?.trim() === text) ?? null;
}

function required<T>(value: T | null): T {
  if (value === null) throw new Error("Expected test element to exist.");
  return value;
}
