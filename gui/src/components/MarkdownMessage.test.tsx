// @vitest-environment jsdom

import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";
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

  it("renders every markdown image destination as escaped text without network elements", () => {
    const html = renderToStaticMarkup(
      <MarkdownMessage text={markdownImagePayload()} fallback="" />
    );

    expect(html).not.toMatch(/<(?:img|link|picture|source)\b/i);
    expect(html).not.toMatch(/\bsrc(?:set)?=/i);
    expect(html).not.toContain("collector.example.test");
    expect(html).toContain("role=\"img\"");
    expect(html).toContain("aria-label=\"probe &amp; &quot;private&quot;\"");
    expect(html).toContain("[image: probe &amp; &quot;private&quot;]");
  });

  it("mounts markdown images as accessible text spans without URL-bearing DOM nodes", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    flushSync(() => {
      root.render(<MarkdownMessage text={markdownImagePayload()} fallback="" />);
    });

    expect(container.querySelector("img, link, picture, source")).toBeNull();
    expect(container.querySelector("[src], [srcset]")).toBeNull();
    expect(container.innerHTML).not.toContain("collector.example.test");
    const placeholders = [...container.querySelectorAll<HTMLElement>("[role='img']")];
    expect(placeholders).toHaveLength(7);
    expect(placeholders[0]?.tagName).toBe("SPAN");
    expect(placeholders[0]?.getAttribute("aria-label")).toBe('probe & "private"');
    expect(placeholders[0]?.textContent).toBe('[image: probe & "private"]');

    flushSync(() => root.unmount());
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
    const html = await renderUserPage(snapshot);

    expect(html).toContain("**not bold**");
    expect(html).not.toContain("<strong>not bold</strong>");
    expect(html).toContain("<strong>bold</strong>");
  });

  it("renders a protected placeholder instead of conflicting imported output text", async () => {
    const snapshot = userPageSnapshot();
    snapshot.human_requests[0].payload = {
      type: "output",
      release_required: true,
      message: "protected render secret sentinel"
    };

    const html = await renderUserPage(snapshot);

    expect(html).not.toContain("protected render secret sentinel");
    expect(html).toContain("its content is withheld from this GUI by data-flow policy.");
  });
});

function markdownImagePayload(): string {
  return [
    '![probe & "private"](https://collector.example.test/https)',
    "![http](http://collector.example.test/http)",
    "![protocol relative](//collector.example.test/protocol-relative)",
    "![file](file:///tmp/private.png)",
    "![data](data:image/png;base64,AAAA)",
    "![blob](blob:https://collector.example.test/id)",
    "![relative](./private.png)"
  ].join("\n\n");
}

async function renderUserPage(snapshot: RuntimeSnapshot): Promise<string> {
  return renderWithSuspense(
    <I18nProvider>
      <UserPage
        connection={{ url: "http://127.0.0.1:1", token: "token", db: "local" }}
        snapshot={snapshot}
        selectedPid="pid_1"
        selectedProcess={snapshot.processes[0]}
        taskLabels={{ pid_1: "Render markdown" }}
        taskSettings={{
          image: "coding-agent:v0",
          llmProfile: "",
          maxQuantaInput: "",
          workingDirectory: "",
          workspaceAccess: "edit",
          allowGitRequests: true,
          commandAccess: "none",
          contextMaintenance: true,
          authorityManifestId: "authm_test"
        }}
        taskLaunchMode="ephemeral"
        durableTaskLaunchAvailable={false}
        spawnGoal="goal"
        message=""
        images={[]}
        llmProfiles={[]}
        onSelectPid={() => undefined}
        onMaxQuantaChange={() => undefined}
        onSpawnGoalChange={() => undefined}
        onSpawnImageChange={() => undefined}
        onApplyTaskSettings={() => undefined}
        onTaskLaunchModeChange={() => undefined}
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
}

async function renderWithSuspense(node: ReactNode): Promise<string> {
  const stream = await renderToReadableStream(node);
  await stream.allReady;
  return new Response(stream).text();
}

function userPageSnapshot(): RuntimeSnapshot {
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
    task_run_launch: { enabled: true, plaintext_payloads_enabled: false, available: false },
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
        revision: 1,
        created_at: "2026-06-19T01:00:01.000Z",
        updated_at: "2026-06-19T01:00:01.000Z"
      }
    ],
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
