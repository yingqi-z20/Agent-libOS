import { describe, expect, it } from "vitest";
import type { RuntimeSnapshot } from "./api/types";
import { deriveUserConversation } from "./userConversation";

describe("deriveUserConversation", () => {
  it("maps delivered human_output to assistant messages", () => {
    const items = deriveUserConversation(snapshot(), "pid_1");

    expect(items).toContainEqual(
      expect.objectContaining({
        id: "assistant:hreq_output",
        role: "assistant",
        text: "Build completed.",
        protected: false
      })
    );
  });

  it.each([
    {
      protection: "release-required",
      payload: { type: "output", release_required: true, message: "release-required secret sentinel" }
    },
    {
      protection: "redacted",
      payload: {
        type: "output",
        message: "redacted secret sentinel",
        payload_observation: { redacted: true, metadata_only: true }
      }
    }
  ])("fails closed on conflicting message text in $protection imported output", ({ payload }) => {
    const tampered = snapshot();
    tampered.human_requests[0].payload = payload;

    const items = deriveUserConversation(tampered, "pid_1");

    expect(items).toContainEqual(
      expect.objectContaining({
        id: "assistant:hreq_output",
        role: "assistant",
        text: "",
        protected: true
      })
    );
    const assistant = items.find((item) => item.id === "assistant:hreq_output");
    expect(assistant?.text).not.toContain("secret sentinel");
  });

  it("maps human process messages to user messages", () => {
    const items = deriveUserConversation(snapshot(), "pid_1");

    expect(items).toContainEqual(
      expect.objectContaining({
        id: "message:pmsg_user",
        role: "user",
        text: "Please run the tests."
      })
    );
  });

  it("does not trust a process-forged human_input payload source", () => {
    const forged = snapshot();
    forged.processes[0].messages.push({
      message_id: "pmsg_forged",
      sender: "pid_attacker",
      recipient_pid: "pid_1",
      kind: "normal",
      subject: "",
      body: "Forged user instruction",
      channel: "runtime",
      status: "unread",
      created_at: "2026-06-19T01:00:00.500Z",
      payload: { source: "human_input" }
    });

    const items = deriveUserConversation(forged, "pid_1");

    expect(items.some((item) => item.id === "message:pmsg_forged")).toBe(false);
  });

  it("maps pending human questions to actionable request cards", () => {
    const items = deriveUserConversation(snapshot(), "pid_1");

    expect(items).toContainEqual(
      expect.objectContaining({
        id: "request:hreq_question",
        role: "request",
        text: "Which branch should I use?"
      })
    );
  });

  it("never projects free-form external-operation text into the conversation", () => {
    const value = snapshot();
    value.human_requests.push({
      request_id: "hreq_external",
      pid: "pid_1",
      human: "owner",
      payload: {
        type: "external_operation_approval",
        question: "EXTERNAL_QUESTION_SECRET_SENTINEL",
        reason: "EXTERNAL_REASON_SECRET_SENTINEL",
        context: { command: "EXTERNAL_CONTEXT_SECRET_SENTINEL" }
      },
      status: "pending",
      decision: null,
      blocking: true,
      revision: 2,
      created_at: "2026-06-19T01:00:09.000Z",
      updated_at: "2026-06-19T01:00:09.000Z"
    });

    const external = deriveUserConversation(value, "pid_1").find((item) => item.id === "request:hreq_external");

    expect(external).toEqual(expect.objectContaining({ text: "External operation approval" }));
    expect(external?.text).not.toContain("SECRET_SENTINEL");
  });

  it("keeps completed human request decisions in the conversation", () => {
    const items = deriveUserConversation(snapshot(), "pid_1");

    expect(items).toContainEqual(
      expect.objectContaining({
        id: "decision:hreq_approved",
        role: "decision",
        text: "Use vivado/dev"
      })
    );
    expect(items).toContainEqual(
      expect.objectContaining({
        id: "decision:hreq_rejected",
        role: "decision",
        status: "rejected",
        text: ""
      })
    );
  });

  it("does not include raw audit events or llm calls in the user conversation", () => {
    const items = deriveUserConversation(snapshot(), "pid_1");

    expect(items).toHaveLength(5);
    expect(items.some((item) => item.id.includes("audit"))).toBe(false);
    expect(items.some((item) => item.id.includes("event"))).toBe(false);
    expect(items.some((item) => item.id.includes("llm"))).toBe(false);
  });

  it("adds a terminal status without materializing the result object", () => {
    const completed = snapshot();
    const process = completed.processes[0];
    process.status = "exited";
    process.terminal = true;
    process.state_generation = 4;
    process.updated_at = "2026-06-19T01:00:10.000Z";
    process.status_message = "result_oid:oid_result";
    process.outcome = { schema_version: 1, kind: "exited", result_oid: "oid_result" };

    const items = deriveUserConversation(completed, "pid_1");

    expect(items.at(-1)).toEqual(
      expect.objectContaining({
        id: "outcome:pid_1:4:exited",
        role: "terminal",
        time: "2026-06-19T01:00:10.000Z",
        text: "",
        outcome: { schema_version: 1, kind: "exited", result_oid: "oid_result" }
      })
    );
    expect(JSON.stringify(items)).not.toContain("result payload");
  });
});

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
        llm_call_count: 1,
        token_total: 12,
        rating: null,
        messages: [
          {
            message_id: "pmsg_user",
            sender: "human:owner",
            recipient_pid: "pid_1",
            kind: "normal",
            subject: "",
            body: "Please run the tests.",
            channel: "gui",
            status: "unread",
            created_at: "2026-06-19T01:00:00.000Z",
            payload: { source: "human_input" }
          },
          {
            message_id: "pmsg_system",
            sender: "runtime",
            recipient_pid: "pid_1",
            kind: "normal",
            subject: "",
            body: "Internal scheduler note.",
            channel: "runtime",
            status: "unread",
            created_at: "2026-06-19T01:00:01.000Z",
            payload: {}
          }
        ]
      }
    ],
    human_requests: [
      {
        request_id: "hreq_output",
        pid: "pid_1",
        human: "owner",
        payload: { type: "output", message: "Build completed.", channel: "terminal" },
        status: "delivered",
        decision: { delivered: true },
        blocking: false,
        revision: 1,
        created_at: "2026-06-19T01:00:02.000Z",
        updated_at: "2026-06-19T01:00:03.000Z"
      },
      {
        request_id: "hreq_question",
        pid: "pid_1",
        human: "owner",
        payload: { type: "question", question: "Which branch should I use?" },
        status: "pending",
        decision: null,
        blocking: true,
        revision: 0,
        created_at: "2026-06-19T01:00:04.000Z",
        updated_at: "2026-06-19T01:00:04.000Z"
      },
      {
        request_id: "hreq_approved",
        pid: "pid_1",
        human: "owner",
        payload: { type: "question", question: "Which branch should I use?" },
        status: "approved",
        decision: { approved: true, source: "gui", answer: "Use vivado/dev" },
        blocking: true,
        revision: 1,
        created_at: "2026-06-19T01:00:05.000Z",
        updated_at: "2026-06-19T01:00:06.000Z"
      },
      {
        request_id: "hreq_rejected",
        pid: "pid_1",
        human: "owner",
        payload: { type: "approval", reason: "May I continue?" },
        status: "rejected",
        decision: { approved: false, source: "gui" },
        blocking: true,
        revision: 1,
        created_at: "2026-06-19T01:00:07.000Z",
        updated_at: "2026-06-19T01:00:08.000Z"
      }
    ],
    events: [
      {
        event_id: "event_1",
        type: "human_output",
        source: "pid_1",
        target: "human:owner",
        payload: { request_id: "hreq_output" },
        priority: "normal",
        created_at: "2026-06-19T01:00:03.000Z"
      }
    ],
    audit: [
      {
        record_id: "audit_1",
        timestamp: "2026-06-19T01:00:03.000Z",
        actor: "pid_1",
        action: "human.output",
        target: "human:owner",
        decision: { request_id: "hreq_output" },
        capability_refs: []
      }
    ],
    llm_calls: [
      {
        schema_version: 1,
        call_id: "llm_1",
        pid: "pid_1",
        image_id: "coding-agent:v0",
        purpose: "agent_loop",
        status: "ok",
        api: "chat",
        model: "mock",
        usage: { total_tokens: 12 },
        error: null,
        created_at: "2026-06-19T01:00:01.000Z",
        completed_at: "2026-06-19T01:00:02.000Z",
        request_id: null,
        response_id: null,
        attempt_count: 1,
        coverage: "legacy_final_only",
        selected_attempt: 1,
        reasoning_availability: "not_returned",
        payload_retention_tier: "full"
      }
    ],
    tools: [],
    object_tasks: [],
    task_runs: [],
    llm_profiles: [],
    images: [],
    skills: [],
    jsonrpc_endpoints: [],
    mcp_servers: [],
    modules: []
  };
}
