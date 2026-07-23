import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { RuntimeProcess, RuntimeSnapshot } from "../api/types";
import { I18nProvider } from "../i18n";
import { DetailTabs, explainRefreshKey, tabIndexForKey } from "./DetailTabs";

describe("DetailTabs", () => {
  it("renders the MCP registry tab from snapshot data", () => {
    const html = renderToStaticMarkup(
      <I18nProvider>
        <DetailTabs
          process={null}
          snapshot={snapshot()}
          onImportImage={() => undefined}
          onCommitImage={() => undefined}
          onUseImageForSpawn={() => undefined}
          onUseImageForExec={() => undefined}
          onRate={async () => true}
          onInspectImage={async () => ({ image: {} as never, registry: {}, artifact: null })}
          onListOperations={async (pid) => ({
            schema_version: 1,
            pid,
            roots_only: true,
            operations: [],
            presentation_truncated: false,
            next_cursor: null
          })}
          onExplainOperation={async () => { throw new Error("not used"); }}
          onResolveOperation={async () => { throw new Error("not used"); }}
          explainLookup={null}
        />
      </I18nProvider>
    );

    expect(html).toContain("MCP");
    expect(html).toContain('role="tablist"');
    expect(html).toContain('role="tabpanel"');
    expect(html).toContain('aria-selected="true"');
  });

  it("supports wrapped arrow and boundary keyboard navigation", () => {
    expect(tabIndexForKey(0, "ArrowLeft", 4)).toBe(3);
    expect(tabIndexForKey(3, "ArrowRight", 4)).toBe(0);
    expect(tabIndexForKey(2, "Home", 4)).toBe(0);
    expect(tabIndexForKey(1, "End", 4)).toBe(3);
    expect(tabIndexForKey(1, "Enter", 4)).toBeNull();
  });

  it("changes the Explain refresh key when SSE-backed evidence advances", () => {
    const before = snapshot();
    const after = snapshot();
    after.events = [{
      event_id: "evt_new",
      type: "process.updated",
      source: "pid_1",
      target: "pid_1",
      payload: {},
      priority: "normal",
      created_at: "2026-07-10T00:00:00Z"
    }];

    expect(explainRefreshKey(process(), before)).not.toBe(explainRefreshKey(process(), after));
  });
});

function process(): RuntimeProcess {
  return {
    pid: "pid_1",
    parent_pid: null,
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
    resource_budget: {},
    resource_usage: {},
    resource_remaining: {},
    rating: null
  };
}

function snapshot(): RuntimeSnapshot {
  return {
    db: "local",
    scheduler: {
      auto_run: true,
      running: false,
      paused: false,
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
    tools: [],
    llm_profiles: [],
    images: [],
    skills: [],
    jsonrpc_endpoints: [],
    mcp_servers: [{ server_id: "demo-mcp" }],
    modules: []
  };
}
