import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { HumanRequest, RuntimeProcess, RuntimeSnapshot, TaskRunDetail, TaskRunSummary } from "../api/types";
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
    expect(processStatusDetail({ ...runtimeProcess(), status: "paused", wait_state: {
      schema_version: 1,
      kind: "stale_execution",
      pid: "pid_stale",
      recovered_by_owner_sha256: "a".repeat(64),
      prior_owner_sha256: "b".repeat(64),
      prior_lease_sha256: "c".repeat(64),
      prior_execution_generation: 1,
      recovered_execution_generation: 2,
      recovered_state_generation: 3
    } }, t)).toBe("user.statusPaused");
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
            contextMaintenance: true,
            authorityManifestId: "authm_test"
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

  it("renders run-scoped Human pages and localized durable state at narrow widths", () => {
    const run = taskRun();
    const snapshot = emptySnapshot();
    snapshot.processes = [runtimeProcess()];
    snapshot.task_runs = [run];
    const html = renderToStaticMarkup(
      <I18nProvider initialLanguage="zh-CN">
        <UserPage
          connection={{ url: "http://127.0.0.1:1", token: "test", db: "local" }}
          snapshot={snapshot}
          selectedPid="pid_test"
          selectedProcess={snapshot.processes[0]}
          selectedRunId={run.run_id}
          selectedRun={run}
          selectedRunDetail={taskRunDetail(run)}
          taskRunHumanRequests={[runHumanRequest()]}
          taskRunHumanHasMore
          taskRunHumanPresentationTruncated
          taskRuns={[run]}
          taskLabels={{}}
          taskSettings={{
            image: "coding-agent:v0",
            llmProfile: "",
            maxQuantaInput: "4",
            workingDirectory: "",
            workspaceAccess: "none",
            allowGitRequests: false,
            commandAccess: "none",
            contextMaintenance: false,
            authorityManifestId: ""
          }}
          spawnGoal=""
          message=""
          images={[]}
          llmProfiles={[]}
          onSelectPid={() => undefined}
          onSelectRun={() => undefined}
          onLoadMoreTaskRunHumanRequests={() => undefined}
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

    expect(html).toContain("需要介入");
    expect(html).toContain("进入终态时清除 Payload");
    expect(html).toContain("当前 Payload 可取回状态");
    expect(html).toContain("已保留且可取回");
    expect(html).toContain("已满足要求：1/2");
    expect(html).not.toContain("已满足要求：1/4");
    expect(html).toContain("未知外部效果");
    expect(html).toContain("待处理");
    expect(html).toContain("Approve child effect");
    expect(html).toContain("仍有部分与 Run 关联的人工请求尚未显示");
    expect(html).toContain('<button type="button">加载更多</button>');
    expect(html).not.toContain("needs_attention");
    expect(html).not.toContain("purge_on_terminal");
    expect(html).not.toContain("unknown_effect");
  });
});

function taskRun(): TaskRunSummary {
  return {
    schema_version: 1,
    run_id: "run_test",
    revision: 7,
    status: "needs_attention",
    display_title: "耐久修复",
    root_pid: "pid_test",
    active_pid: "pid_test",
    allowed_actions: ["recover", "cancel"],
    blockers: [{ kind: "unknown_effect" }],
    retention: "purge_on_terminal",
    payloads_purged: false,
    requirement_counts: { total: 2, satisfied: 1, pending: 1 }
  };
}

function taskRunDetail(run: TaskRunSummary): TaskRunDetail {
  return {
    summary: run,
    requirements: {
      items: [{
        schema_version: 1,
        requirement_id: "requirement_1",
        run_id: run.run_id,
        ordinal: 0,
        kind: "initial",
        status: "pending",
        requirement_sha256: "a".repeat(64),
        label: "Initial requirement with a very long label that must wrap at 360 pixels",
        created_by: "host",
        created_at: "2030-01-01T00:00:00Z",
        updated_at: "2030-01-01T00:00:00Z",
        started_at: null,
        completed_at: null,
        waived_by: null,
        content_available: false,
        content_retention: "hash_only",
        content_sha256: "a".repeat(64)
      }],
      next_cursor: null,
      has_more: false
    },
    recovery_options: []
  };
}

function runHumanRequest(): HumanRequest {
  return {
    request_id: "human_child_1",
    pid: "pid_child_not_in_snapshot",
    human: "owner",
    payload: { type: "approval", question: "Approve child effect" },
    status: "pending",
    decision: null,
    blocking: true,
    created_at: "2030-01-01T00:00:00Z",
    updated_at: "2030-01-01T00:00:00Z"
  };
}

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
      default_max_quanta: 4
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
    mcp_servers: [],
    modules: []
  };
}
