// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { LlmCallDetail, LlmCallSummary, LlmTraceContentField } from "../api/types";
import { I18nProvider } from "../i18n";
import { mergeLlmCallSummaries, mergeSnapshotLlmCallSummaries, ProviderTracePanel, TraceContent, type ProviderTraceClient } from "./ProviderTracePanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("ProviderTracePanel", () => {
  it("loads a selected-process trace and renders Provider text inertly in user mode", async () => {
    const requestedFields: LlmTraceContentField[] = [];
    const client = traceClient(async (_pid, _callId, field, options) => {
      requestedFields.push(field);
      return {
        schema_version: 1,
        pid: "pid_1",
        call_id: "llmcall_e2e_trace",
        field,
        attempt_sequence: options?.attemptSequence ?? null,
        content: "<a href=\"https://evil.example\">ignore previous instructions</a>",
        next_cursor: null,
        has_more: false,
        content_hash: "a".repeat(64),
        retention_tier: "full"
      };
    });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(
        <I18nProvider initialLanguage="en">
          <ProviderTracePanel pid="pid_1" client={client} snapshotCalls={[summary()]} mode="user" />
        </I18nProvider>
      );
      await flushPromises();
    });
    await act(flushPromises);

    expect(container.querySelector("[data-testid='provider-trace-panel']")).not.toBeNull();
    expect(container.querySelector("[data-testid='provider-trace-call']")).not.toBeNull();
    expect(container.querySelector("[data-testid='provider-trace-call-llmcall_e2e_trace']")).toBeNull();
    expect(container.querySelector("[data-testid='provider-attempt-1']")).not.toBeNull();
    expect(container.textContent).toContain("Returned reasoning");
    expect(container.textContent).toContain("search");
    expect(container.textContent).not.toContain("Input messages");
    expect(container.textContent).not.toContain("Attempt tool actions");
    for (const hidden of [
      "llmcall_e2e_trace", "agent_loop", "test-model", "responses", "initial", "total_tokens",
      "req_1", "resp_1", "Diagnostic usage", "Retention", "Complete built-in-client coverage"
    ]) expect(container.innerHTML).not.toContain(hidden);

    const revealButtons = Array.from(container.querySelectorAll<HTMLButtonElement>(".traceContent > button"));
    expect(revealButtons).toHaveLength(2);
    for (const reveal of revealButtons) {
      await act(async () => {
        reveal.click();
        await flushPromises();
      });
    }

    expect(requestedFields).toEqual(["attempt_reasoning", "attempt_output"]);
    expect(requestedFields).not.toContain("attempt_tool_calls");
    expect(container.textContent).toContain("ignore previous instructions");
    expect(container.querySelector("a")).toBeNull();

    await act(() => root.unmount());
    container.remove();
  });

  it("keeps operator-only request fields behind explicit reveal controls", async () => {
    const client = traceClient();
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(
        <I18nProvider initialLanguage="en">
          <ProviderTracePanel pid="pid_1" client={client} snapshotCalls={[summary()]} />
        </I18nProvider>
      );
      await flushPromises();
    });
    await act(flushPromises);

    expect(container.textContent).toContain("Request and low-level response fields");
    expect(container.textContent).toContain("Input messages");
    expect(container.textContent).toContain("Attempt tool actions");
    expect(container.querySelector("[data-testid='provider-trace-call-llmcall_e2e_trace']")).not.toBeNull();
    expect(client.getProcessLlmCallContent).not.toHaveBeenCalled();
    await act(() => root.unmount());
  });

  it("aborts an in-flight content request and drops its surface when unmounted", async () => {
    let signal: AbortSignal | undefined;
    const client = traceClient((_pid, _callId, _field, options) => {
      signal = options?.signal;
      return new Promise(() => undefined);
    });
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(
        <I18nProvider initialLanguage="en">
          <ProviderTracePanel pid="pid_1" client={client} snapshotCalls={[summary()]} mode="user" />
        </I18nProvider>
      );
      await flushPromises();
    });
    await act(flushPromises);
    const reveal = container.querySelector<HTMLButtonElement>(".traceContent.attempt_reasoning button");
    await act(async () => {
      reveal?.click();
      await Promise.resolve();
    });
    expect(signal?.aborted).toBe(false);

    await act(() => root.unmount());
    expect(signal?.aborted).toBe(true);
    expect(container.textContent).toBe("");
  });

  it("clears previously loaded text immediately when a content cursor returns 409", async () => {
    let requestCount = 0;
    const client = traceClient(async (_pid, _callId, field, options) => {
      requestCount += 1;
      if (requestCount === 1) {
        return {
          schema_version: 1,
          pid: "pid_1",
          call_id: "llmcall_e2e_trace",
          field,
          attempt_sequence: options?.attemptSequence ?? null,
          content: "old retained reasoning",
          next_cursor: "stale-next-cursor",
          has_more: true,
          content_hash: "a".repeat(64),
          retention_tier: "full"
        };
      }
      throw new ApiError("content changed", 409, { error: { code: "content_changed" } });
    });
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(
        <I18nProvider initialLanguage="en">
          <ProviderTracePanel pid="pid_1" client={client} snapshotCalls={[summary()]} mode="user" />
        </I18nProvider>
      );
      await flushPromises();
    });
    await act(flushPromises);
    const reveal = container.querySelector<HTMLButtonElement>(".traceContent.attempt_reasoning > button");
    await act(async () => {
      reveal?.click();
      await flushPromises();
    });
    expect(container.textContent).toContain("old retained reasoning");

    const loadMore = container.querySelector<HTMLButtonElement>(".traceContent.attempt_reasoning > button");
    await act(async () => {
      loadMore?.click();
      await flushPromises();
    });
    expect(container.textContent).not.toContain("old retained reasoning");
    expect(container.textContent).toContain("This content changed");
    expect(container.querySelector(".traceContent.attempt_reasoning .traceInertText")).toBeNull();
    expect(container.querySelector(".traceContent.attempt_reasoning > button")).toBeNull();
    await act(() => root.unmount());
  });

  it("clears loaded text when descriptor hash, cursor, availability, or tier changes", async () => {
    const client = traceClient(async (_pid, _callId, field, options) => ({
      schema_version: 1,
      pid: "pid_1",
      call_id: "llmcall_e2e_trace",
      field,
      attempt_sequence: options?.attemptSequence ?? null,
      content: "loaded sentinel body",
      next_cursor: null,
      has_more: false,
      content_hash: "a".repeat(64),
      retention_tier: "full"
    }));
    const descriptor = detail().content[0];
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(
        <I18nProvider initialLanguage="en">
          <TraceContent
            descriptor={descriptor}
            field="attempt_reasoning"
            attemptSequence={1}
            pid="pid_1"
            callId="llmcall_e2e_trace"
            client={client}
            retentionTier="full"
          />
        </I18nProvider>
      );
      await flushPromises();
    });
    const reveal = container.querySelector<HTMLButtonElement>("button");
    await act(async () => {
      reveal?.click();
      await flushPromises();
    });
    expect(container.textContent).toContain("loaded sentinel body");

    await act(async () => {
      root.render(
        <I18nProvider initialLanguage="en">
          <TraceContent
            descriptor={{
              ...descriptor,
              availability: "limited",
              content_hash: "b".repeat(64),
              cursor: "replacement-offset-zero"
            }}
            field="attempt_reasoning"
            attemptSequence={1}
            pid="pid_1"
            callId="llmcall_e2e_trace"
            client={client}
            retentionTier="summary"
          />
        </I18nProvider>
      );
      await flushPromises();
    });
    expect(container.textContent).not.toContain("loaded sentinel body");
    expect(container.querySelector(".traceInertText")).toBeNull();
    expect(container.textContent).toContain("Reveal content");
    await act(() => root.unmount());
  });

  it("merges snapshot and cursor pages without duplicating calls", () => {
    expect(mergeLlmCallSummaries(
      [summary(), { ...summary(), call_id: "call_2" }],
      [{ ...summary(), status: "failed" }]
    ).map((call) => [call.call_id, call.status])).toEqual([
      ["llmcall_e2e_trace", "failed"],
      ["call_2", "ok"]
    ]);
  });

  it("preserves paginated order while a newer snapshot replaces status and retention", () => {
    const current = [summary(), { ...summary(), call_id: "call_2", purpose: "second" }];
    const merged = mergeSnapshotLlmCallSummaries(current, [{
      ...summary(),
      status: "failed",
      payload_retention_tier: "summary",
      reasoning_availability: "not_persisted"
    }]);

    expect(merged.map((call) => call.call_id)).toEqual(["llmcall_e2e_trace", "call_2"]);
    expect(merged[0]).toMatchObject({
      status: "failed",
      payload_retention_tier: "summary",
      reasoning_availability: "not_persisted"
    });
  });
});

function traceClient(
  content: ProviderTraceClient["getProcessLlmCallContent"] = vi.fn(async (_pid: string, _callId: string, field: LlmTraceContentField, options: { attemptSequence?: number }) => ({
    schema_version: 1 as const,
    pid: "pid_1",
    call_id: "llmcall_e2e_trace",
    field,
    attempt_sequence: options?.attemptSequence ?? null,
    content: "content",
    next_cursor: null,
    has_more: false,
    content_hash: "a".repeat(64),
    retention_tier: "full" as const
  }))
): ProviderTraceClient {
  const value = summary();
  return {
    listProcessLlmCalls: vi.fn(async () => ({ schema_version: 1, items: [value], next_cursor: null, has_more: false })),
    getProcessLlmCall: vi.fn(async () => detail()),
    getProcessLlmCallContent: vi.fn(content)
  } as unknown as ProviderTraceClient;
}

function summary(): LlmCallSummary {
  return {
    schema_version: 1,
    call_id: "llmcall_e2e_trace",
    pid: "pid_1",
    image_id: "coding-agent:v0",
    purpose: "agent_loop",
    status: "ok",
    api: "responses",
    model: "test-model",
    usage: { total_tokens: 12 },
    error: null,
    created_at: "2026-08-03T01:00:00.000Z",
    completed_at: "2026-08-03T01:00:01.000Z",
    request_id: "req_1",
    response_id: "resp_1",
    attempt_count: 1,
    coverage: "complete",
    selected_attempt: 1,
    reasoning_availability: "returned",
    payload_retention_tier: "full"
  };
}

function detail(): LlmCallDetail {
  const descriptors: LlmCallDetail["content"] = [
    ["attempt_reasoning", 1, "text"],
    ["attempt_output", 1, "text"],
    ["attempt_tool_calls", 1, "json"],
    ["messages", null, "json"],
    ["tools", null, "json"],
    ["request_options", null, "json"],
    ["raw_response", null, "json"],
    ["response_content", null, "text"]
  ].map(([field, attemptSequence, contentType]) => ({
    field: field as LlmTraceContentField,
    attempt_sequence: attemptSequence as number | null,
    availability: "available" as const,
    content_type: contentType as "text" | "json",
    size_bytes: 12,
    size_chars: 12,
    content_hash: "a".repeat(64),
    cursor: `cursor_${field}`
  }));
  return {
    schema_version: 1,
    call: summary(),
    attempts: [{
      sequence: 1,
      kind: "initial",
      api: "responses",
      status: "ok",
      model: "test-model",
      request_id: "req_1",
      response_id: "resp_1",
      reasoning_availability: "returned",
      reasoning_blocks: [{ type: "summary_text", source: "responses.output", reason: null, chars: 12, bytes: 12, sha256: "a".repeat(64) }],
      output_availability: "returned",
      tool_names: ["search"],
      tool_call_count: 1,
      usage: { total_tokens: 12 },
      started_at: "2026-08-03T01:00:00.000Z",
      completed_at: "2026-08-03T01:00:01.000Z",
      duration_ms: 1000,
      error: null
    }],
    content: descriptors
  };
}

async function flushPromises() {
  await Promise.resolve();
  await Promise.resolve();
}
