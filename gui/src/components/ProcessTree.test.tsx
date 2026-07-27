// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { RuntimeProcess } from "../api/types";
import { I18nProvider } from "../i18n";
import { filterProcesses, indexProcessTree, ProcessTree } from "./ProcessTree";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("indexProcessTree", () => {
  it("groups roots and siblings in one pass while preserving snapshot order", () => {
    const root = process("root", null);
    const firstChild = process("child-1", "root");
    const secondRoot = process("root-2", null);
    const secondChild = process("child-2", "root");

    const indexed = indexProcessTree([root, firstChild, secondRoot, secondChild]);

    expect(indexed.roots).toEqual([root, secondRoot]);
    expect(indexed.children.get("root")).toEqual([firstChild, secondChild]);
    expect(indexed.children.size).toBe(1);
  });

  it("keeps a source-window child visible when its parent is omitted", () => {
    const orphanedChild = process("active-child", "omitted-parent");
    const root = process("visible-root", null);

    const indexed = indexProcessTree([orphanedChild, root]);

    expect(indexed.roots).toEqual([orphanedChild, root]);
    expect(indexed.children.size).toBe(0);
  });

  it("filters by process metadata while retaining visible ancestors", () => {
    const root = process("root", null);
    const child = { ...process("worker", "root"), image_id: "research-agent:v2", working_directory: "reports" };
    const unrelated = process("other", null);

    expect(filterProcesses([root, child, unrelated], "research")).toEqual([root, child]);
    expect(filterProcesses([root, child, unrelated], "REPORTS")).toEqual([root, child]);
    expect(filterProcesses([root, child, unrelated], "missing")).toEqual([]);
    expect(filterProcesses([root, child, unrelated], "")).toEqual([root, child, unrelated]);
    expect(filterProcesses([root, child, unrelated], "quarterly report", { worker: "Quarterly report audit" })).toEqual([root, child]);
  });

  it("keeps the selected process and its ancestors visible when they do not match the search", () => {
    const root = process("root", null);
    const selectedChild = process("selected-child", "root");
    const matchingRoot = { ...process("matching-root", null), working_directory: "reports" };

    expect(filterProcesses(
      [root, selectedChild, matchingRoot],
      "reports",
      {},
      selectedChild.pid
    )).toEqual([root, selectedChild, matchingRoot]);
    expect(filterProcesses(
      [root, selectedChild, matchingRoot],
      "no-match",
      {},
      selectedChild.pid
    )).toEqual([root, selectedChild]);
  });

  it("keeps the selected tree item visibly selected while filtering for another process", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(() => {
      root.render(
        <I18nProvider initialLanguage="en">
          <ProcessTree
            processes={[
              process("root", null),
              process("selected-child", "root"),
              { ...process("matching-root", null), working_directory: "reports" }
            ]}
            selectedPid="selected-child"
            onSelect={() => undefined}
          />
        </I18nProvider>
      );
    });

    const search = container.querySelector<HTMLInputElement>('input[type="search"]');
    expect(search).not.toBeNull();
    await act(() => {
      if (!search) return;
      search.value = "reports";
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const items = Array.from(container.querySelectorAll<HTMLButtonElement>('[role="treeitem"]'));
    const selectedItem = items.find((item) => item.getAttribute("aria-selected") === "true");
    expect(items).toHaveLength(3);
    expect(selectedItem?.classList.contains("selected")).toBe(true);
    expect(selectedItem?.textContent).toContain("selected-child");
    expect(items.some((item) => item.textContent?.includes("matching-root"))).toBe(true);
    expect(items.find((item) => item.textContent?.includes("root"))?.getAttribute("aria-expanded")).toBe("true");
    expect(selectedItem?.hasAttribute("aria-expanded")).toBe(false);

    await act(() => root.unmount());
  });

  it("uses a single tab stop for tree keyboard navigation", () => {
    const html = renderToStaticMarkup(
      <I18nProvider initialLanguage="en">
        <ProcessTree
          processes={[process("root", null), process("child", "root")]}
          selectedPid="child"
          taskLabels={{ child: "Audit the GUI" }}
          onSelect={() => undefined}
        />
      </I18nProvider>
    );

    expect(html.match(/tabindex="0"/g)).toHaveLength(1);
    expect(html.match(/tabindex="-1"/g)).toHaveLength(1);
    expect(html).toContain("Audit the GUI");
    expect(html).toContain('aria-selected="true"');
    expect(html).toContain('aria-expanded="true"');
  });

  it("moves the roving tab stop with keyboard focus", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(() => {
      root.render(
        <I18nProvider initialLanguage="en">
          <ProcessTree
            processes={[process("root", null), process("child", "root")]}
            selectedPid="root"
            onSelect={() => undefined}
          />
        </I18nProvider>
      );
    });

    const before = Array.from(container.querySelectorAll<HTMLButtonElement>('[role="treeitem"]'));
    expect(before[0]?.tabIndex).toBe(0);
    await act(() => {
      before[0]?.focus();
      before[0]?.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }));
    });

    const after = Array.from(container.querySelectorAll<HTMLButtonElement>('[role="treeitem"]'));
    expect(document.activeElement).toBe(after[1]);
    expect(after[0]?.tabIndex).toBe(-1);
    expect(after[1]?.tabIndex).toBe(0);

    await act(() => root.unmount());
    container.remove();
  });
});

function process(pid: string, parentPid: string | null): RuntimeProcess {
  return {
    pid,
    parent_pid: parentPid,
    image_id: "base-agent:v0",
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
  };
}
