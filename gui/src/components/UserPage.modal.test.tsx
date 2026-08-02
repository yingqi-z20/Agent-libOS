// @vitest-environment jsdom

import { describe, expect, it, vi } from "vitest";
import { inertModalSiblings, installModalKeyboardBoundary } from "./UserPage";

describe("user reasoning drawer modal boundary", () => {
  it("inerts and hides every background sibling, then restores prior state", () => {
    const parent = document.createElement("main");
    const topBar = document.createElement("header");
    const workspace = document.createElement("section");
    const preHidden = document.createElement("section");
    preHidden.setAttribute("aria-hidden", "false");
    preHidden.setAttribute("inert", "");
    const drawer = document.createElement("aside");
    parent.append(topBar, workspace, preHidden, drawer);

    const restore = inertModalSiblings(drawer);
    for (const element of [topBar, workspace, preHidden]) {
      expect(element.hasAttribute("inert")).toBe(true);
      expect(element.getAttribute("aria-hidden")).toBe("true");
    }
    expect(drawer.hasAttribute("inert")).toBe(false);

    restore();
    expect(topBar.hasAttribute("inert")).toBe(false);
    expect(topBar.hasAttribute("aria-hidden")).toBe(false);
    expect(workspace.hasAttribute("inert")).toBe(false);
    expect(preHidden.hasAttribute("inert")).toBe(true);
    expect(preHidden.getAttribute("aria-hidden")).toBe("false");
  });

  it("cycles Tab through native summary controls and handles Escape", () => {
    const drawer = document.createElement("aside");
    const first = document.createElement("button");
    const middle = document.createElement("button");
    const disclosure = document.createElement("details");
    const last = document.createElement("summary");
    disclosure.append(last);
    drawer.append(first, middle, disclosure);
    document.body.append(drawer);
    const onEscape = vi.fn();
    const remove = installModalKeyboardBoundary(drawer, onEscape);

    last.focus();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true }));
    expect(document.activeElement).toBe(first);

    first.focus();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, cancelable: true }));
    expect(document.activeElement).toBe(last);

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }));
    expect(onEscape).toHaveBeenCalledOnce();

    remove();
    drawer.remove();
  });
});
