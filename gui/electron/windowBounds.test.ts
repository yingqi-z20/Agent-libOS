import { describe, expect, it } from "vitest";
import { mainWindowBounds, shouldCreateBrowserWindow } from "./windowBounds";

describe("mainWindowBounds", () => {
  it("allows the production Electron window to reach the narrow responsive layout", () => {
    expect(mainWindowBounds.minWidth).toBeGreaterThanOrEqual(320);
    expect(mainWindowBounds.minWidth).toBeLessThanOrEqual(720);
    expect(mainWindowBounds.minHeight).toBeGreaterThanOrEqual(480);
  });

  it("keeps the default Electron smoke path headless", () => {
    expect(shouldCreateBrowserWindow(false, false)).toBe(true);
    expect(shouldCreateBrowserWindow(true, false)).toBe(false);
    expect(shouldCreateBrowserWindow(true, true)).toBe(true);
  });
});
