import { describe, expect, it, vi } from "vitest";
import {
  assertTrustedIpcSender,
  enforceTrustedRendererNavigation,
  installDefaultDenyPermissions,
  installTrustedRendererNavigationGuard,
  sameRendererOrigin
} from "./security.js";

describe("Electron renderer security", () => {
  it("denies checked, requested, and device permissions by default", () => {
    const handlers: {
      check?: (...args: never[]) => boolean;
      request?: (webContents: unknown, permission: unknown, callback: (allowed: boolean) => void) => void;
      device?: (details: unknown) => boolean;
    } = {};
    const session = {
      setPermissionCheckHandler: vi.fn((handler) => { handlers.check = handler; }),
      setPermissionRequestHandler: vi.fn((handler) => { handlers.request = handler; }),
      setDevicePermissionHandler: vi.fn((handler) => { handlers.device = handler; })
    };

    installDefaultDenyPermissions(session as never);

    expect(handlers.check?.()).toBe(false);
    let allowed: boolean | null = null;
    handlers.request?.({}, "media", (value: boolean) => { allowed = value; });
    expect(allowed).toBe(false);
    expect(handlers.device?.({})).toBe(false);
  });

  it("accepts only the expected main frame and renderer origin", () => {
    const mainFrame = { url: "agent-libos://app/index.html" };
    const webContents = { mainFrame };
    const event = { sender: webContents, senderFrame: mainFrame };

    expect(() => assertTrustedIpcSender(event as never, webContents as never, "agent-libos://app")).not.toThrow();
    expect(() => assertTrustedIpcSender(
      { sender: webContents, senderFrame: { url: "agent-libos://app/frame.html" } } as never,
      webContents as never,
      "agent-libos://app"
    )).toThrow(/Untrusted IPC sender/);
    const attackerFrame = { url: "https://attacker.example/" };
    const attackerContents = { mainFrame: attackerFrame };
    expect(() => assertTrustedIpcSender(
      { sender: attackerContents, senderFrame: attackerFrame } as never,
      attackerContents as never,
      "agent-libos://app"
    )).toThrow(/Untrusted IPC sender/);
  });

  it("normalizes HTTP and custom-scheme renderer origins", () => {
    expect(sameRendererOrigin("http://127.0.0.1:5173/nested", "http://127.0.0.1:5173/")).toBe(true);
    expect(sameRendererOrigin("agent-libos://app/assets/main.js", "agent-libos://app/index.html")).toBe(true);
    expect(sameRendererOrigin("agent-libos://other/index.html", "agent-libos://app/index.html")).toBe(false);
  });

  it("installs one strict same-origin guard for navigations and server redirects", () => {
    const listeners = new Map<string, (event: { preventDefault(): void; url: string }) => void>();
    const webContents = {
      on: vi.fn((name: string, listener: (event: { preventDefault(): void; url: string }) => void) => {
        listeners.set(name, listener);
        return webContents;
      })
    };
    installTrustedRendererNavigationGuard(webContents as never, "http://127.0.0.1:5173/");

    expect([...listeners.keys()]).toEqual(["will-navigate", "will-redirect"]);
    expect(listeners.get("will-navigate")).toBe(listeners.get("will-redirect"));

    const allow = { url: "http://127.0.0.1:5173/next?view=user#latest", preventDefault: vi.fn() };
    listeners.get("will-navigate")?.(allow);
    expect(allow.preventDefault).not.toHaveBeenCalled();

    for (const candidate of [
      "https://127.0.0.1:5173/downgrade-confusion",
      "http://127.0.0.1:5174/other-port",
      "http://127.0.0.1.attacker.example:5173/",
      "http://127.0.0.1@attacker.example:5173/",
      "//attacker.example/protocol-relative",
      "agent-libos://app.attacker.example/index.html",
      "not a URL"
    ]) {
      const redirect = { url: candidate, preventDefault: vi.fn() };
      listeners.get("will-redirect")?.(redirect);
      expect(redirect.preventDefault, candidate).toHaveBeenCalledOnce();
    }
  });

  it("fails closed when a navigation URL cannot match the trusted origin", () => {
    const event = { preventDefault: vi.fn() };

    expect(enforceTrustedRendererNavigation(event, "https://attacker.example/", "agent-libos://app")).toBe(false);
    expect(event.preventDefault).toHaveBeenCalledOnce();
  });
});
