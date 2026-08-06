import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { RuntimeProcess, RuntimeSnapshot } from "../api/types";
import { I18nProvider } from "../i18n";
import {
  adminPanelKey,
  adminRefreshKey,
  DetailTabs,
  explainPanelKey,
  explainRefreshKey,
  mergeAuditRecords,
  providerTracePanelKey,
  tabIndexForKey
} from "./DetailTabs";

describe("DetailTabs", () => {
  it("renders the MCP registry tab from snapshot data", () => {
    const html = renderTabs();

    expect(html).toContain("MCP");
    expect(html).toContain("Semantic");
    expect(html).toContain('role="tablist"');
    expect(html).toContain('role="tabpanel"');
    expect(html).toContain('aria-selected="true"');
  });

  it("connects the narrow tab selector to a panel named by its visible label", () => {
    const html = renderTabs();
    const selectTag = html.match(/<select[^>]*aria-controls="[^"]+"[^>]*>/)?.[0] ?? "";
    const panelTag = html.match(/<div class="tabPanel"[^>]*>/)?.[0] ?? "";
    const selectId = attribute(selectTag, "id");
    const selectLabelId = attribute(selectTag, "aria-labelledby");
    const controlledPanelId = attribute(selectTag, "aria-controls");
    const panelLabelledBy = attribute(panelTag, "aria-labelledby")?.split(" ") ?? [];

    expect(selectId).toBeTruthy();
    expect(selectLabelId).toBeTruthy();
    expect(controlledPanelId).toBeTruthy();
    expect(html).toContain(`<label class="detailTabSelect" for="${selectId}">`);
    expect(html).toContain(`<span id="${selectLabelId}">`);
    expect(attribute(panelTag, "id")).toBe(controlledPanelId);
    expect(panelLabelledBy).toContain(selectLabelId);
  });

  it("makes inspector actions inert while a global operator action is running", () => {
    const html = renderTabs(process(), true);

    expect(html).toContain('class="details"');
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain('inert=""');
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

  it("keeps the Explain component identity stable while evidence refreshes", () => {
    const selected = process();
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

    expect(explainRefreshKey(selected, before)).not.toBe(explainRefreshKey(selected, after));
    expect(explainPanelKey(selected)).toBe(explainPanelKey({ ...selected, state_generation: 99 }));
    expect(explainPanelKey(selected)).toBe("pid_1");
  });

  it("does not refresh Explain for unrelated-process evidence", () => {
    const before = snapshot();
    const after = snapshot();
    after.events = [{
      event_id: "evt_other",
      type: "process.updated",
      source: "pid_other",
      target: "pid_other",
      payload: {},
      priority: "normal",
      created_at: "2026-07-10T00:00:00Z"
    }];

    expect(explainRefreshKey(process(), before)).toBe(explainRefreshKey(process(), after));
  });

  it("includes the connection epoch in admin panel identity", () => {
    const selected = process();

    expect(adminPanelKey(selected, 1)).toBe("1:pid_1");
    expect(adminPanelKey(selected, 2)).not.toBe(adminPanelKey(selected, 1));
    expect(providerTracePanelKey("pid_1", 1)).toBe("1:pid_1");
    expect(providerTracePanelKey("pid_2", 1)).not.toBe(providerTracePanelKey("pid_1", 1));
    expect(providerTracePanelKey("pid_1", 2)).not.toBe(providerTracePanelKey("pid_1", 1));
  });

  it("refreshes admin data for current-process evidence", () => {
    const before = snapshot();
    const afterAudit = snapshot();
    afterAudit.audit = [{
      record_id: "audit_current",
      timestamp: "2026-07-10T00:00:00Z",
      actor: "pid_1",
      action: "capability.grant",
      target: "process:pid_1",
      decision: null,
      capability_refs: []
    }];
    const afterEvent = snapshot();
    afterEvent.events = [{
      event_id: "evt_current",
      type: "checkpoint.created",
      source: "pid_1",
      target: "process:pid_1",
      payload: {},
      priority: "normal",
      created_at: "2026-07-10T00:00:00Z"
    }];

    expect(adminRefreshKey(process(), afterAudit)).not.toBe(adminRefreshKey(process(), before));
    expect(adminRefreshKey(process(), afterEvent)).not.toBe(adminRefreshKey(process(), before));
  });

  it("does not refresh admin data for unrelated-process evidence", () => {
    const before = snapshot();
    const after = snapshot();
    after.audit = [{
      record_id: "audit_other",
      timestamp: "2026-07-10T00:00:00Z",
      actor: "pid_other",
      action: "capability.grant",
      target: "process:pid_other",
      decision: null,
      capability_refs: []
    }];
    after.events = [{
      event_id: "evt_other",
      type: "checkpoint.created",
      source: "pid_other",
      target: "process:pid_other",
      payload: {},
      priority: "normal",
      created_at: "2026-07-10T00:00:00Z"
    }];

    expect(adminRefreshKey(process(), before)).toBe(adminRefreshKey(process(), after));
  });

  it("merges older audit pages chronologically without duplicates", () => {
    const record = (record_id: string, timestamp: string) => ({
      record_id,
      timestamp,
      actor: "pid_1",
      action: "test",
      target: "process:pid_1",
      decision: null,
      capability_refs: []
    });

    expect(mergeAuditRecords(
      [record("audit_1", "2026-01-01T00:00:01Z"), record("audit_2", "2026-01-01T00:00:02Z")],
      [record("audit_2", "2026-01-01T00:00:02Z"), record("audit_3", "2026-01-01T00:00:03Z")]
    ).map((item) => item.record_id)).toEqual(["audit_1", "audit_2", "audit_3"]);
  });
});

function renderTabs(selectedProcess: RuntimeProcess | null = null, busy = false): string {
  return renderToStaticMarkup(
    <I18nProvider>
      <DetailTabs
        process={selectedProcess}
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
        busy={busy}
      />
    </I18nProvider>
  );
}

function attribute(markup: string, name: string): string | null {
  return markup.match(new RegExp(`${name}="([^"]+)"`))?.[1] ?? null;
}

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
    schema_version: 3,
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
    task_runs: [],
    tools: [],
    llm_profiles: [],
    images: [],
    skills: [],
    jsonrpc_endpoints: [],
    mcp_servers: [{
      schema_version: 1,
      server_id: "demo-mcp",
      protocol_mode: "legacy",
      transport: { type: "stdio" },
      tools: [],
      timeout_s: 30,
      max_request_bytes: 65_536,
      max_response_bytes: 1_048_576,
      metadata: {}
    }],
    modules: []
  };
}
