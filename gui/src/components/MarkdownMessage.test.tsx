import { renderToReadableStream, renderToStaticMarkup } from "react-dom/server";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RuntimeSnapshot } from "../api/types";
import { I18nProvider } from "../i18n";
import { UserPage } from "./UserPage";
import { isSafeMarkdownHref, MarkdownMessage, openMarkdownHref } from "./MarkdownMessage";

describe("MarkdownMessage", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders common GFM markdown for assistant output", () => {
    const html = renderToStaticMarkup(
      <MarkdownMessage
        text={[
          "**bold** and `inline`",
          "",
          "- first",
          "- second",
          "",
          "```ts",
          "const answer = 42;",
          "```",
          "",
          "| name | value |",
          "| --- | --- |",
          "| ok | yes |"
        ].join("\n")}
        fallback=""
      />
    );

    expect(html).toContain("<strong>bold</strong>");
    expect(html).toContain("<code>inline</code>");
    expect(html).toContain("<ul>");
    expect(html).toContain("<pre>");
    expect(html).toContain("class=\"markdownTableWrap\"");
    expect(html).toContain("<table>");
  });

  it("does not inject raw HTML from markdown text", () => {
    const html = renderToStaticMarkup(<MarkdownMessage text="<script>alert(1)</script>" fallback="" />);

    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;alert(1)&lt;/script&gt;");
  });

  it("only treats explicitly safe external links as clickable", () => {
    expect(isSafeMarkdownHref("https://example.test/path")).toBe(true);
    expect(isSafeMarkdownHref("http://example.test/path")).toBe(true);
    expect(isSafeMarkdownHref("mailto:owner@example.test")).toBe(true);
    expect(isSafeMarkdownHref("javascript:alert(1)")).toBe(false);
    expect(isSafeMarkdownHref("file:///tmp/secret")).toBe(false);
    expect(isSafeMarkdownHref("/relative/path")).toBe(false);

    const html = renderToStaticMarkup(
      <MarkdownMessage text="[ok](https://example.test) [bad](javascript:alert(1))" fallback="" />
    );
    expect(html).toContain("href=\"https://example.test\"");
    expect(html).not.toContain("href=\"javascript:alert(1)\"");
  });

  it("opens safe links through the Electron preload bridge", () => {
    const openExternal = vi.fn();
    const preventDefault = vi.fn();
    vi.stubGlobal("window", { libosApi: { openExternal } });

    expect(openMarkdownHref("https://example.test/docs", { preventDefault })).toBe(true);

    expect(preventDefault).toHaveBeenCalledTimes(1);
    expect(openExternal).toHaveBeenCalledWith("https://example.test/docs");
  });

  it("keeps user messages as plain text while rendering assistant markdown", async () => {
    const snapshot = userPageSnapshot();
    const process = snapshot.processes[0];
    const html = await renderWithSuspense(
      <I18nProvider>
        <UserPage
          connection={{ url: "http://127.0.0.1:1", token: "token", db: "local" }}
          snapshot={snapshot}
          selectedPid="pid_1"
          selectedProcess={process}
          maxQuanta={null}
          spawnGoal="goal"
          spawnImage="coding-agent:v0"
          spawnLlmProfile=""
          spawnWorkingDirectory=""
          spawnWorkspaceAccess="edit"
          spawnAllowGitRequests={true}
          message=""
          images={[]}
          llmProfiles={[]}
          onSelectPid={() => undefined}
          onMaxQuantaChange={() => undefined}
          onSpawnGoalChange={() => undefined}
          onSpawnImageChange={() => undefined}
          onSpawnLlmProfileChange={() => undefined}
          onSpawnWorkingDirectoryChange={() => undefined}
          onSpawnWorkspaceAccessChange={() => undefined}
          onSpawnAllowGitRequestsChange={() => undefined}
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

    expect(html).toContain("**not bold**");
    expect(html).not.toContain("<strong>not bold</strong>");
    expect(html).toContain("<strong>bold</strong>");
  });
});

async function renderWithSuspense(node: ReactNode): Promise<string> {
  const stream = await renderToReadableStream(node);
  await stream.allReady;
  return new Response(stream).text();
}

function userPageSnapshot(): RuntimeSnapshot {
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
    processes: [
      {
        pid: "pid_1",
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
        state_generation: 0,
        loaded_skills: {},
        tool_table: {},
        capabilities: [],
        terminal: false,
        unread_message_count: 0,
        interrupt_count: 0,
        messages: [
          {
            message_id: "msg_1",
            sender: "human:owner",
            recipient_pid: "pid_1",
            kind: "normal",
            subject: "",
            body: "**not bold**",
            channel: "gui",
            status: "unread",
            created_at: "2026-06-19T01:00:00.000Z",
            payload: { source: "human_input" }
          }
        ],
        llm_call_count: 0,
        token_total: 0,
        rating: null
      }
    ],
    human_requests: [
      {
        request_id: "out_1",
        pid: "pid_1",
        human: "owner",
        payload: { type: "output", message: "**bold**" },
        status: "delivered",
        decision: { delivered: true },
        blocking: false,
        created_at: "2026-06-19T01:00:01.000Z",
        updated_at: "2026-06-19T01:00:01.000Z"
      }
    ],
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
