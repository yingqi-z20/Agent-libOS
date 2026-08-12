// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { AuditRecord, HumanRequest, LlmCall, ProcessMessage, RuntimeEvent } from "../api/types";
import { I18nProvider } from "../i18n";
import {
  buildTimelineItems,
  countTimelineItemsByKind,
  evidenceRef,
  filterTimelineItems,
  isTimelineNearLatest,
  Timeline
} from "./Timeline";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("Timeline", () => {
  it("builds selected-process timeline items in chronological order", () => {
    const items = buildTimelineItems({
      pid: "pid_1",
      messages: [message("msg_1", "2026-06-19T01:00:04.000Z")],
      humanRequests: [
        humanRequest("req_1", "pid_1", "2026-06-19T01:00:02.000Z"),
        humanRequest("req_other", "pid_2", "2026-06-19T01:00:01.000Z")
      ],
      llmCalls: [
        llmCall("llm_1", "pid_1", "2026-06-19T01:00:05.000Z"),
        llmCall("llm_other", "pid_2", "2026-06-19T01:00:00.000Z")
      ],
      events: [
        event("evt_source", "pid_1", null, "2026-06-19T01:00:01.000Z"),
        event("evt_target", "system", "pid_1", "2026-06-19T01:00:03.000Z"),
        event("evt_other", "system", "pid_2", "2026-06-19T01:00:00.000Z")
      ],
      audit: [
        auditRecord("audit_actor", "pid_1", null, "2026-06-19T01:00:06.000Z"),
        auditRecord("audit_target", "host", "process:pid_1", "2026-06-19T01:00:00.000Z"),
        auditRecord("audit_other", "host", "process:pid_2", "2026-06-19T01:00:00.500Z")
      ]
    });

    expect(items.map((item) => item.kind)).toEqual(["audit", "event", "human", "event", "message", "llm", "audit"]);
    expect(items.map((item) => item.time)).toEqual([
      "2026-06-19T01:00:00.000Z",
      "2026-06-19T01:00:01.000Z",
      "2026-06-19T01:00:02.000Z",
      "2026-06-19T01:00:03.000Z",
      "2026-06-19T01:00:04.000Z",
      "2026-06-19T01:00:05.000Z",
      "2026-06-19T01:00:06.000Z"
    ]);
  });

  it("counts and filters timeline items by kind", () => {
    const items = buildTimelineItems({
      pid: "pid_1",
      messages: [message("msg_1", "2026-06-19T01:00:04.000Z")],
      humanRequests: [humanRequest("req_1", "pid_1", "2026-06-19T01:00:02.000Z")],
      llmCalls: [llmCall("llm_1", "pid_1", "2026-06-19T01:00:05.000Z")],
      events: [
        event("evt_source", "pid_1", null, "2026-06-19T01:00:01.000Z"),
        event("evt_target", "system", "pid_1", "2026-06-19T01:00:03.000Z")
      ],
      audit: [auditRecord("audit_target", "host", "process:pid_1", "2026-06-19T01:00:00.000Z")]
    });

    expect(countTimelineItemsByKind(items)).toEqual({
      message: 1,
      human: 1,
      llm: 1,
      event: 2,
      audit: 1
    });
    expect(filterTimelineItems(items, "all")).toBe(items);
    expect(filterTimelineItems(items, "activity").map((item) => item.kind)).toEqual(["human", "message", "llm"]);
    expect(filterTimelineItems(items, "event").map((item) => item.kind)).toEqual(["event", "event"]);
    expect(filterTimelineItems(items, "audit").map((item) => item.kind)).toEqual(["audit"]);
  });

  it("maps explainable timeline records to explicit evidence ids", () => {
    const items = buildTimelineItems({
      pid: "pid_1",
      messages: [message("msg_1", "2026-06-19T01:00:04.000Z")],
      humanRequests: [humanRequest("req_1", "pid_1", "2026-06-19T01:00:02.000Z")],
      llmCalls: [llmCall("llm_1", "pid_1", "2026-06-19T01:00:05.000Z")],
      events: [event("evt_1", "pid_1", null, "2026-06-19T01:00:01.000Z")],
      audit: [auditRecord("audit_1", "pid_1", null, "2026-06-19T01:00:06.000Z")]
    });

    expect(items.map(evidenceRef)).toEqual([
      { kind: "event", id: "evt_1" },
      { kind: "request", id: "req_1" },
      null,
      { kind: "call", id: "llm_1" },
      { kind: "audit", id: "audit_1" }
    ]);
  });

  it("renders the default human-facing activity filter with type counts", () => {
    const html = renderToStaticMarkup(
      <I18nProvider>
        <Timeline
          pid="pid_1"
          messages={[message("msg_1", "2026-06-19T01:00:04.000Z")]}
          humanRequests={[humanRequest("req_1", "pid_1", "2026-06-19T01:00:02.000Z")]}
          llmCalls={[]}
          events={[]}
          audit={[]}
        />
      </I18nProvider>
    );

    expect(html).toMatch(/Filter timeline by type|按类型筛选时间线/);
    expect(html).toContain("aria-pressed=\"true\"");
    expect(html).toMatch(/Activity|活动/);
    expect(html).toMatch(/All|全部/);
    expect(html).toMatch(/Messages|消息/);
    expect(html).toMatch(/Human|人类/);
    expect(html).toContain("timelineFilterCount");
  });

  it("renders only canonical external-operation evidence, never its free-form payload", () => {
    const request = humanRequest("req_external", "pid_1", "2026-06-19T01:00:02.000Z");
    request.payload = {
      type: "external_operation_approval",
      question: "TIMELINE_QUESTION_SECRET_SENTINEL",
      reason: "TIMELINE_REASON_SECRET_SENTINEL",
      context: { argv: ["TIMELINE_CONTEXT_SECRET_SENTINEL"] }
    };
    const html = renderToStaticMarkup(
      <I18nProvider initialLanguage="en">
        <Timeline pid="pid_1" messages={[]} humanRequests={[request]} llmCalls={[]} events={[]} audit={[]} />
      </I18nProvider>
    );

    expect(html).toContain("External operation approval");
    expect(html).toContain("approval_preview_valid");
    expect(html).not.toContain("TIMELINE_QUESTION_SECRET_SENTINEL");
    expect(html).not.toContain("TIMELINE_REASON_SECRET_SENTINEL");
    expect(html).not.toContain("TIMELINE_CONTEXT_SECRET_SENTINEL");
  });

  it("keeps timeline controls outside live announcements while preserving a focusable scroll region", () => {
    const container = document.createElement("div");
    container.innerHTML = renderToStaticMarkup(
      <I18nProvider initialLanguage="en">
        <Timeline
          pid="pid_1"
          messages={[message("msg_1", "2026-06-19T01:00:04.000Z")]}
          humanRequests={[]}
          llmCalls={[]}
          events={[]}
          audit={[]}
        />
      </I18nProvider>
    );

    const scrollRegion = container.querySelector<HTMLElement>(".timeline");
    const liveLog = container.querySelector<HTMLElement>(".timelineEntries");
    const filter = container.querySelector<HTMLElement>(".timelineFilter");
    const jsonOperation = container.querySelector<HTMLElement>(".timelineJsonOperation");

    expect(scrollRegion?.getAttribute("role")).toBe("region");
    expect(scrollRegion?.getAttribute("aria-label")).toBe("Process timeline");
    expect(scrollRegion?.tabIndex).toBe(0);
    expect(scrollRegion?.hasAttribute("aria-live")).toBe(false);
    expect(liveLog?.getAttribute("role")).toBe("log");
    expect(liveLog?.getAttribute("aria-live")).toBe("polite");
    expect(liveLog?.getAttribute("aria-relevant")).toBe("additions text");
    expect(liveLog?.contains(filter ?? null)).toBe(false);
    expect(jsonOperation?.getAttribute("role")).toBe("group");
    expect(jsonOperation?.getAttribute("aria-live")).toBe("off");
    expect(jsonOperation?.querySelector("button.collapseToggle")).not.toBeNull();
  });

  it("uses a forgiving bottom threshold for following the latest activity", () => {
    expect(isTimelineNearLatest({ scrollHeight: 1000, scrollTop: 805, clientHeight: 100 })).toBe(true);
    expect(isTimelineNearLatest({ scrollHeight: 1000, scrollTop: 804, clientHeight: 100 })).toBe(false);
  });

  it("follows new activity until the user scrolls away, then offers a reduced-motion-safe jump", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    const firstMessage = message("msg_1", "2026-06-19T01:00:04.000Z");
    const renderTimeline = async (messages: ProcessMessage[]) => {
      await act(() => {
        root.render(
          <I18nProvider>
            <Timeline
              pid="pid_1"
              messages={messages}
              humanRequests={[]}
              llmCalls={[]}
              events={[]}
              audit={[]}
            />
          </I18nProvider>
        );
      });
    };

    await renderTimeline([firstMessage]);
    const timeline = container.querySelector<HTMLElement>(".timeline");
    expect(timeline).not.toBeNull();
    if (!timeline) throw new Error("Timeline did not render");
    Object.defineProperties(timeline, {
      clientHeight: { configurable: true, value: 200 },
      scrollHeight: { configurable: true, value: 1000 }
    });
    timeline.scrollTop = 800;

    await renderTimeline([
      firstMessage,
      message("msg_2", "2026-06-19T01:00:05.000Z")
    ]);
    expect(timeline.scrollTop).toBe(1000);

    timeline.scrollTop = 300;
    await act(() => {
      timeline.dispatchEvent(new Event("scroll", { bubbles: true }));
    });
    const jumpButton = container.querySelector<HTMLButtonElement>(".jumpToLatest");
    expect(jumpButton?.classList.contains("timelineJumpToLatest")).toBe(true);
    expect(jumpButton?.textContent).toMatch(/Jump to latest|回到最新消息/);

    Object.defineProperty(timeline, "scrollHeight", { configurable: true, value: 1200 });
    await renderTimeline([
      firstMessage,
      message("msg_2", "2026-06-19T01:00:05.000Z"),
      message("msg_3", "2026-06-19T01:00:06.000Z")
    ]);
    expect(timeline.scrollTop).toBe(300);

    const scrollTo = vi.fn();
    timeline.scrollTo = scrollTo;
    vi.stubGlobal("matchMedia", vi.fn(() => ({
      matches: true,
      media: "(prefers-reduced-motion: reduce)",
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    })));
    await act(() => {
      container.querySelector<HTMLButtonElement>(".jumpToLatest")?.click();
    });
    expect(scrollTo).toHaveBeenCalledWith({ top: 1200, behavior: "auto" });
    expect(container.querySelector(".jumpToLatest")).toBeNull();

    vi.unstubAllGlobals();
    await act(() => root.unmount());
  });

  it("continues following when an existing LLM summary changes without changing id", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    const call = llmCall("llm_1", "pid_1", "2026-06-19T01:00:05.000Z");
    const renderTimeline = async (llmCalls: LlmCall[]) => {
      await act(() => {
        root.render(
          <I18nProvider>
            <Timeline
              pid="pid_1"
              messages={[]}
              humanRequests={[]}
              llmCalls={llmCalls}
              events={[]}
              audit={[]}
            />
          </I18nProvider>
        );
      });
    };

    await renderTimeline([call]);
    const timeline = container.querySelector<HTMLElement>(".timeline");
    expect(timeline).not.toBeNull();
    if (!timeline) throw new Error("Timeline did not render");
    Object.defineProperties(timeline, {
      clientHeight: { configurable: true, value: 200 },
      scrollHeight: { configurable: true, value: 1000 }
    });
    timeline.scrollTop = 1000;

    Object.defineProperty(timeline, "scrollHeight", { configurable: true, value: 1400 });
    await renderTimeline([{
      ...call,
      payload_retention_tier: "summary"
    }]);

    expect(timeline.scrollTop).toBe(1400);
    expect(container.querySelector(".jumpToLatest")).toBeNull();

    await act(() => root.unmount());
  });
});

function message(messageId: string, createdAt: string): ProcessMessage {
  return {
    message_id: messageId,
    sender: "human:owner",
    recipient_pid: "pid_1",
    kind: "normal",
    subject: "subject",
    body: "body",
    channel: "gui",
    status: "unread",
    created_at: createdAt,
    payload: {}
  };
}

function humanRequest(requestId: string, pid: string, createdAt: string): HumanRequest {
  return {
    request_id: requestId,
    pid,
    human: "owner",
    payload: { question: "Continue?" },
    status: "pending",
    decision: null,
    blocking: true,
    revision: 0,
    created_at: createdAt,
    updated_at: createdAt
  };
}

function llmCall(callId: string, pid: string, createdAt: string): LlmCall {
  return {
    schema_version: 1,
    call_id: callId,
    pid,
    image_id: "coding-agent:v0",
    purpose: "quantum",
    status: "ok",
    api: "responses",
    model: "test-model",
    usage: {},
    error: null,
    created_at: createdAt,
    completed_at: createdAt,
    request_id: null,
    response_id: null,
    attempt_count: 1,
    coverage: "complete",
    selected_attempt: 1,
    reasoning_availability: "not_returned",
    payload_retention_tier: "full"
  };
}

function event(eventId: string, source: string, target: string | null, createdAt: string): RuntimeEvent {
  return {
    event_id: eventId,
    type: "process.updated",
    source,
    target,
    payload: {},
    priority: "normal",
    created_at: createdAt
  };
}

function auditRecord(recordId: string, actor: string, target: string | null, timestamp: string): AuditRecord {
  return {
    record_id: recordId,
    timestamp,
    actor,
    action: "scheduler.run_quantum",
    target,
    decision: null,
    capability_refs: []
  };
}
