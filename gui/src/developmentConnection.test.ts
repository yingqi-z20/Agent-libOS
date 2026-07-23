import { describe, expect, it } from "vitest";
import { developmentConnection } from "./developmentConnection";

describe("developmentConnection", () => {
  it("is unavailable in production even when variables are present", () => {
    expect(developmentConnection({
      VITE_AGENT_LIBOS_GUI_URL: "http://127.0.0.1:8765",
      VITE_AGENT_LIBOS_GUI_TOKEN: "secret"
    }, false)).toBeNull();
  });

  it("accepts an explicitly configured loopback development server", () => {
    expect(developmentConnection({
      VITE_AGENT_LIBOS_GUI_URL: "http://127.0.0.1:8765/not-a-base-path?ignored=yes#ignored",
      VITE_AGENT_LIBOS_GUI_TOKEN: " local-token ",
      VITE_AGENT_LIBOS_GUI_DB: " local "
    }, true)).toEqual({
      url: "http://127.0.0.1:8765",
      token: "local-token",
      db: "local"
    });
  });

  it("rejects remote and partial development connections", () => {
    expect(() => developmentConnection({
      VITE_AGENT_LIBOS_GUI_URL: "https://example.com",
      VITE_AGENT_LIBOS_GUI_TOKEN: "secret"
    }, true)).toThrow(/loopback/);
    expect(() => developmentConnection({
      VITE_AGENT_LIBOS_GUI_URL: "http://127.0.0.1:8765"
    }, true)).toThrow(/both URL and token/);
    expect(() => developmentConnection({
      VITE_AGENT_LIBOS_GUI_URL: "http://user:password@127.0.0.1:8765",
      VITE_AGENT_LIBOS_GUI_TOKEN: "secret"
    }, true)).toThrow(/credentials/);
  });
});
