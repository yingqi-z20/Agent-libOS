// @vitest-environment jsdom

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import userEvent from "@testing-library/user-event";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { LibOSClient } from "../api/client";
import type { McpDiscoveryResult, McpServerSummary } from "../api/types";
import type { ConfirmationRequest } from "../adminTypes";
import { I18nProvider, type Language } from "../i18n";
import "../styles.css";
import { RemoteRegistryPanel, schemaArgumentFields } from "./RemoteRegistryPanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mounted: Array<{ root: Root; container: HTMLDivElement }> = [];

afterEach(async () => {
  for (const { root, container } of mounted.splice(0)) {
    await act(() => root.unmount());
    container.remove();
  }
});

describe("RemoteRegistryPanel MCP v2", () => {
  it("discovers modern protocol state with the keyboard and keeps it page-local", async () => {
    const discoverMcpServer = vi.fn().mockResolvedValue(discovery());
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await renderPanel({
      entries: [server("modern", "auto")],
      client: client({ discoverMcpServer }),
      confirmAction
    });
    const panel = required(container.querySelector<HTMLElement>(".remotePanel"));
    const discover = required(buttonWithExactText(panel, "Discover protocol"));

    expect(panel.getAttribute("aria-labelledby")).toBe("remote-mcp-panel-title");
    expect(panel.textContent).toContain("Not negotiated in this page session");
    expect(discover.disabled).toBe(false);

    const user = userEvent.setup();
    await act(async () => {
      discover.focus();
      await user.keyboard("{Enter}");
      await vi.waitFor(() => expect(discoverMcpServer).toHaveBeenCalledWith("modern"));
    });

    expect(confirmAction).not.toHaveBeenCalled();
    expect(panel.textContent).toContain("v2");
    expect(panel.textContent).toContain("Auto");
    expect(panel.textContent).toContain("Modern");
    expect(panel.textContent).toContain("2026-07-28");
    expect(panel.textContent).toContain("Sessionless");
    expect(panel.textContent).toContain("Not used");
    expect(panel.textContent).toContain("Example MCP 2.0.0");
    expect(panel.textContent).toContain("tools");
    expect(panel.textContent).toContain("resources");

    const controls = [...panel.querySelectorAll<HTMLElement>("button:not(:disabled), select:not(:disabled), input:not(:disabled), textarea:not(:disabled)")];
    expect(controls.length).toBeGreaterThan(4);
    for (const control of controls) {
      control.focus();
      expect(document.activeElement).toBe(control);
    }
  });

  it("disables discovery for legacy manifests and localizes the reason", async () => {
    const { container } = await renderPanel({
      language: "zh-CN",
      entries: [server("legacy", "legacy", 1)]
    });
    const panel = required(container.querySelector<HTMLElement>(".remotePanel"));
    const discover = required(buttonWithExactText(panel, "发现协议"));

    expect(discover.disabled).toBe(true);
    expect(discover.getAttribute("aria-describedby")).toBe("remote-mcp-discover-hint");
    expect(panel.textContent).toContain("协议发现仅适用于使用 auto 或 2026-07-28 模式的 Manifest v2 注册项");
    expect(panel.textContent).toContain("旧版");
    expect(panel.textContent).toContain("本次页面会话尚未协商");
  });

  it("shows discovery failures as recoverable alerts and clears stale connection state", async () => {
    const discoverMcpServer = vi.fn()
      .mockResolvedValueOnce(discovery())
      .mockRejectedValueOnce(new Error("server/discover rejected the revision"));
    const { container } = await renderPanel({
      entries: [server("modern", "auto")],
      client: client({ discoverMcpServer })
    });
    const discover = required(buttonWithExactText(container, "Discover protocol"));

    await act(async () => discover.click());
    await vi.waitFor(() => expect(container.textContent).toContain("2026-07-28"));
    await act(async () => discover.click());
    await vi.waitFor(() => expect(container.querySelector('[role="alert"]')?.textContent).toContain("server/discover rejected"));

    expect(container.textContent).toContain("Not negotiated in this page session");
    expect(discover.disabled).toBe(false);
    expect(container.querySelector(".remotePanel")?.getAttribute("aria-busy")).toBeNull();
  });

  it("ignores a discovery response that arrives after the selected registry entry changes", async () => {
    const pending = deferred<McpDiscoveryResult>();
    const discoverMcpServer = vi.fn().mockReturnValue(pending.promise);
    const entries = [server("first", "auto"), server("second", "auto")];
    const { container } = await renderPanel({ entries, client: client({ discoverMcpServer }) });
    const discover = required(buttonWithExactText(container, "Discover protocol"));
    const select = required(container.querySelector<HTMLSelectElement>("select"));

    await act(async () => discover.click());
    expect(discoverMcpServer).toHaveBeenCalledWith("first");
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set?.call(select, "second");
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => {
      pending.resolve(discovery("first", "Stale Server"));
      await pending.promise;
    });

    expect(select.value).toBe("second");
    expect(container.textContent).toContain("Not negotiated in this page session");
    expect(container.textContent).not.toContain("Stale Server");
  });

  it("uses wrapping protocol facts and actions at a 360px inspector width", async () => {
    const { container } = await renderPanel({ entries: [server("modern", "2026-07-28")] });
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

    expect(container.style.width).toBe("360px");
    expect(container.querySelector(".mcpProtocolFacts")).not.toBeNull();
    expect(styles).toMatch(/\.remotePanel\s*\{[^}]*overflow-wrap:\s*anywhere;/s);
    expect(styles).toContain("grid-template-columns: repeat(auto-fit, minmax(min(100%, 8rem), 1fr));");
    expect(styles).toContain("@container inspector (max-width: 420px)");
    expect(styles).toMatch(/\.remotePanel \.adminActions > button\s*\{[^}]*flex:\s*1 1 100%;/s);
  });

  it("projects bounded scalar JSON Schema fields without hiding the raw editor", async () => {
    const selected = server("modern", "2026-07-28");
    selected.tools = [{
      tool_id: "search",
      mcp_name: "search",
      right: "execute",
      resource: "mcp:modern",
      rollback_class: "none",
      rollback_status: "not_applicable",
      state_mutation: false,
      information_flow: true,
      input_schema: {
        type: "object",
        required: ["query"],
        properties: {
          query: { type: "string", title: "Search query" },
          limit: { type: "integer" },
          mode: { type: "string", enum: ["fast", "thorough"] },
          nested: { type: "object" }
        }
      },
      metadata: {}
    }];

    expect(schemaArgumentFields(selected, "search")).toEqual([
      { name: "query", label: "Search query", required: true, kind: "string", choices: [] },
      { name: "limit", label: "limit", required: false, kind: "integer", choices: [] },
      { name: "mode", label: "mode", required: false, kind: "string", choices: ["fast", "thorough"] }
    ]);
    const { container } = await renderPanel({ entries: [selected] });
    expect(container.querySelector('[aria-label="Schema-driven MCP arguments"]')).not.toBeNull();
    expect(container.querySelector('textarea[aria-label="MCP JSON arguments"]')).not.toBeNull();
  });
});

async function renderPanel({
  language = "en",
  entries,
  client: selectedClient = client(),
  confirmAction = () => undefined
}: {
  language?: Language;
  entries: McpServerSummary[];
  client?: LibOSClient;
  confirmAction?: (request: ConfirmationRequest) => void;
}) {
  const container = document.createElement("div");
  container.style.width = "360px";
  document.body.append(container);
  const root = createRoot(container);
  mounted.push({ root, container });
  await act(() => {
    root.render(
      <I18nProvider initialLanguage={language}>
        <RemoteRegistryPanel
          kind="mcp"
          process={null}
          entries={entries}
          client={selectedClient}
          confirmAction={confirmAction}
        />
      </I18nProvider>
    );
  });
  return { container, root };
}

function client(overrides: Partial<LibOSClient> = {}): LibOSClient {
  return {
    inspectMcpServer: vi.fn(),
    discoverMcpServer: vi.fn(),
    listMcpTools: vi.fn(),
    registerMcpServer: vi.fn(),
    callMcpTool: vi.fn(),
    ...overrides
  } as unknown as LibOSClient;
}

function server(serverId: string, protocolMode: McpServerSummary["protocol_mode"], schemaVersion: 1 | 2 = 2): McpServerSummary {
  return {
    schema_version: schemaVersion,
    server_id: serverId,
    protocol_mode: protocolMode,
    transport: { type: "streamable_http" },
    tools: [],
    timeout_s: 30,
    max_request_bytes: 65_536,
    max_response_bytes: 1_048_576,
    metadata: {},
    updated_at: `2026-08-02T00:00:00Z:${serverId}`
  };
}

function discovery(serverId = "modern", serverName = "Example MCP"): McpDiscoveryResult {
  return {
    server_id: serverId,
    connection: {
      protocol_mode: "auto",
      protocol_era: "modern",
      protocol_revision: "2026-07-28",
      sessionless: true,
      fallback_used: false,
      server_name: serverName,
      server_version: "2.0.0",
      capabilities: ["tools"],
      unsupported_capabilities: ["resources"]
    },
    request_bytes: 128,
    response_bytes: 256,
    duration_s: 0.01,
    receipts: [{
      phase: "server/discover",
      request_bytes: 128,
      response_bytes: 256,
      duration_s: 0.01,
      call_started: true
    }]
  };
}

function buttonWithExactText(container: ParentNode, text: string): HTMLButtonElement | null {
  return [...container.querySelectorAll<HTMLButtonElement>("button")]
    .find((button) => button.textContent?.trim() === text) ?? null;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function required<T>(value: T | null): T {
  if (value === null) throw new Error("Required test element is missing.");
  return value;
}
