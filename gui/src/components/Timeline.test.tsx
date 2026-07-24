import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { AuditRecord, HumanRequest, LlmCall, ProcessMessage, RuntimeEvent } from "../api/types";
import { I18nProvider } from "../i18n";
import {
  buildTimelineItems,
  countTimelineItemsByKind,
  evidenceRef,
  filterTimelineItems,
  Timeline
} from "./Timeline";

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
    created_at: createdAt,
    updated_at: createdAt
  };
}

function llmCall(callId: string, pid: string, createdAt: string): LlmCall {
  return {
    call_id: callId,
    pid,
    image_id: "coding-agent:v0",
    purpose: "quantum",
    status: "ok",
    api: "responses",
    model: "test-model",
    request_options: {},
    response_content: "response",
    tool_calls: [],
    usage: {},
    reasoning: null,
    error: null,
    created_at: createdAt,
    completed_at: createdAt
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
