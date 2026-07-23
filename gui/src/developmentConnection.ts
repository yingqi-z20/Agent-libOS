import type { GuiConnection } from "./api/types";

type DevelopmentEnv = Record<string, string | boolean | undefined>;

/**
 * Opt-in Vite-only bridge for exercising the real local GUI server in a
 * browser. Production builds always return null and continue to require the
 * Electron preload bridge.
 */
export function developmentConnection(env: DevelopmentEnv, development: boolean): GuiConnection | null {
  if (!development) return null;
  const urlValue = typeof env.VITE_AGENT_LIBOS_GUI_URL === "string"
    ? env.VITE_AGENT_LIBOS_GUI_URL.trim()
    : "";
  const token = typeof env.VITE_AGENT_LIBOS_GUI_TOKEN === "string"
    ? env.VITE_AGENT_LIBOS_GUI_TOKEN.trim()
    : "";
  if (!urlValue && !token) return null;
  if (!urlValue || !token) throw new Error("Vite GUI development connection requires both URL and token.");
  const url = new URL(urlValue);
  if (url.protocol !== "http:" || !isLoopbackHost(url.hostname)) {
    throw new Error("Vite GUI development connection must use an HTTP loopback URL.");
  }
  if (url.username || url.password) {
    throw new Error("Vite GUI development connection must not include URL credentials.");
  }
  url.pathname = "/";
  url.search = "";
  url.hash = "";
  return {
    url: url.toString().replace(/\/$/, ""),
    token,
    db: typeof env.VITE_AGENT_LIBOS_GUI_DB === "string" && env.VITE_AGENT_LIBOS_GUI_DB.trim()
      ? env.VITE_AGENT_LIBOS_GUI_DB.trim()
      : "development"
  };
}

function isLoopbackHost(hostname: string): boolean {
  return hostname === "127.0.0.1" || hostname === "localhost" || hostname === "::1" || hostname === "[::1]";
}
