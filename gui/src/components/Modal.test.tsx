// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Modal } from "./Modal";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(() => {
  document.body.innerHTML = "";
  document.body.style.overflow = "";
});

describe("Modal", () => {
  it("locks background scrolling, traps focus, closes with Escape, and restores focus", async () => {
    const launcher = document.createElement("button");
    launcher.textContent = "Open";
    document.body.append(launcher);
    launcher.focus();

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    const onClose = vi.fn();

    await act(() => {
      root.render(
        <Modal title="Settings" onClose={onClose} actions={<button type="button">Save</button>}>
          <button type="button">First</button>
        </Modal>
      );
    });

    const dialog = container.querySelector<HTMLElement>("[role='dialog']");
    const first = container.querySelector<HTMLButtonElement>("button");
    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].at(-1);
    expect(document.body.style.overflow).toBe("hidden");
    expect(document.activeElement).toBe(dialog);

    await act(() => {
      dialog?.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true }));
    });
    expect(document.activeElement).toBe(save);

    save?.focus();
    await act(() => {
      dialog?.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true }));
    });
    expect(document.activeElement).toBe(first);

    await act(() => {
      dialog?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(onClose).toHaveBeenCalledTimes(1);

    await act(() => root.unmount());
    expect(document.body.style.overflow).toBe("");
    expect(document.activeElement).toBe(launcher);
  });

  it("only closes from the backdrop itself", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    const onClose = vi.fn();

    await act(() => {
      root.render(<Modal title="Settings" onClose={onClose}><span>Content</span></Modal>);
    });
    const backdrop = container.querySelector<HTMLElement>(".modalBackdrop");
    const dialog = container.querySelector<HTMLElement>("[role='dialog']");

    await act(() => dialog?.dispatchEvent(new MouseEvent("mousedown", { bubbles: true })));
    expect(onClose).not.toHaveBeenCalled();
    await act(() => backdrop?.dispatchEvent(new MouseEvent("mousedown", { bubbles: true })));
    expect(onClose).toHaveBeenCalledTimes(1);

    await act(() => root.unmount());
  });
});
