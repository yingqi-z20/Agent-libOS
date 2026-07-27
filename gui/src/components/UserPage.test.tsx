import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { RuntimeProcess, RuntimeSnapshot } from "../api/types";
import { I18nProvider } from "../i18n";
import { processStatusDetail, processStatusTone, UserPage } from "./UserPage";

describe("UserPage", () => {
  it("maps runtime states to consistent visual tones", () => {
    expect(processStatusTone("runnable")).toBe("running");
    expect(processStatusTone("waiting_human")).toBe("waiting");
    expect(processStatusTone("paused")).toBe("paused");
    expect(processStatusTone("exited")).toBe("completed");
    expect(processStatusTone("failed")).toBe("terminal");
    expect(processStatusTone("created")).toBe("idle");
    expect(processStatusTone("created", true)).toBe("running");
  });

  it("turns wait states into useful human-facing status details", () => {
    const t = (key: string, vars?: Record<string, string | number>) => `${key}${vars?.pid ? `:${vars.pid}` : ""}`;
    expect(processStatusDetail({ ...runtimeProcess(), wait_state: { schema_version: 1, kind: "human", request_ids: ["req_1"] } }, t)).toBe("user.statusWaitingHuman");
    expect(processStatusDetail({ ...runtimeProcess(), wait_state: { schema_version: 1, kind: "child", child_pid: "pid_1234567890abcdef" } }, t)).toBe("user.statusWaitingChild:pid_12345678…cdef");
    expect(processStatusDetail({ ...runtimeProcess(), status_message: "Provider retry scheduled" }, t)).toBe("Provider retry scheduled");
    expect(processStatusDetail({ ...runtimeProcess(), status: "exited", terminal: true, status_message: "result_oid:obj_1" }, t)).toBeNull();
  });

  it("renders a focused, accessible first-task workspace", () => {
    const html = renderToStaticMarkup(
      <I18nProvider initialLanguage="en">
        <UserPage
          notices={<section data-testid="embedded-notices">Runtime notice</section>}
          connection={{ url: "http://127.0.0.1:1", token: "test", db: "local" }}
          snapshot={emptySnapshot()}
          selectedPid={null}
          selectedProcess={null}
          taskLabels={{}}
          taskSettings={{
            image: "coding-agent:v0",
            llmProfile: "",
            maxQuantaInput: "4",
            workingDirectory: "",
            workspaceAccess: "edit",
            allowGitRequests: true,
            commandAccess: "none",
            contextMaintenance: true
          }}
          spawnGoal="Review this project"
          message=""
          images={[]}
          llmProfiles={[]}
          onSelectPid={() => undefined}
          onMaxQuantaChange={() => undefined}
          onSpawnGoalChange={() => undefined}
          onSpawnImageChange={() => undefined}
          onApplyTaskSettings={() => undefined}
          onMessageChange={() => undefined}
          onSpawn={() => undefined}
          onImportImage={() => undefined}
          onCommitImage={() => undefined}
          onSend={() => undefined}
          onRespond={async () => true}
          onRate={async () => true}
          onCreateLlmProfile={async () => true}
          onUpdateLlmProfile={async () => true}
          onDeleteLlmProfile={async () => true}
          onRun={() => undefined}
          onPause={() => undefined}
          onRefresh={() => undefined}
          onOpenDb={() => undefined}
          onShowOperator={() => undefined}
          onStop={() => undefined}
        />
      </I18nProvider>
    );

    expect(html).toContain('href="#primary-workspace"');
    expect(html).toContain('data-testid="embedded-notices"');
    expect(html.indexOf("userTopBar")).toBeLessThan(html.indexOf('data-testid="embedded-notices"'));
    expect(html.indexOf('data-testid="embedded-notices"')).toBeLessThan(html.indexOf("userWorkspace"));
    expect(html).toContain('aria-label="Task navigation and controls"');
    expect(html).toContain('aria-controls="task-sidebar"');
    expect(html).toContain("What should the Agent work on?");
    expect(html).toContain("Task settings");
    expect(html).toContain('aria-haspopup="dialog"');
    expect(html).toContain("Edit settings");
    expect(html).toContain("coding-agent:v0");
    expect(html).toContain("Image/runtime default");
    expect(html).toContain("Read and write (approval required)");
    expect(html).toContain("Git requests");
    expect(html).toContain("Context maintenance");
    expect(html).not.toContain('type="checkbox"');
    expect(html).not.toContain("Enable persistent context enrichment and bounded maintenance");
    expect(html).toContain("Ctrl/⌘+Enter");
    expect(html).not.toContain("userImageControls");
  });
});

function runtimeProcess(): RuntimeProcess {
  return {
    pid: "pid_test",
    parent_pid: null,
    image_id: "coding-agent:v0",
    llm_profile_id: "default",
    status: "runnable",
    goal_oid: null,
    checkpoint_head: null,
    working_directory: ".",
    status_message: null,
    wait_state: null,
    outcome: null,
    state_generation: 1,
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

function emptySnapshot(): RuntimeSnapshot {
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
      default_max_quanta: 4
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
    mcp_servers: [],
    modules: []
  };
}
