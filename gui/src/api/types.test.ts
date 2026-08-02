import { describe, expect, it } from "vitest";
import { allowedTaskRunActions, assertLlmCallDetail, assertLlmCallPage, assertLlmTraceContentChunk, assertMcpDiscoveryResult, assertRuntimeSnapshot, assertTaskRunDetail, runtimeSnapshotFromSseData, taskRunSummaryFromSseData, upsertTaskRunSummary } from "./types";

describe("assertRuntimeSnapshot", () => {
  it("accepts the minimum same-build snapshot shape", () => {
    const value = snapshot();
    expect(() => assertRuntimeSnapshot(value)).not.toThrow();
  });

  it("rejects malformed scheduler and collection fields before rendering", () => {
    expect(() => assertRuntimeSnapshot({ ...snapshot(), scheduler: { auto_run: "yes" } })).toThrow(/scheduler/);
    expect(() => assertRuntimeSnapshot({ ...snapshot(), events: {} })).toThrow(/events/);
  });

  it("requires schema v3 and validates durable run controls", () => {
    expect(() => assertRuntimeSnapshot({ ...snapshot(), schema_version: 2 })).toThrow(/schema_version/);
    expect(() => assertRuntimeSnapshot({ ...snapshot(), task_runs: [{ ...run(), allowed_actions: ["retry"] }] })).toThrow(/allowed_actions/);
    expect(() => assertRuntimeSnapshot({ ...snapshot(), task_runs: [{ ...run(), allowed_actions: ["purge_payloads"] }] })).toThrow(/allowed_actions/);
  });

  it("rejects process rows without a valid pid", () => {
    expect(() => assertRuntimeSnapshot({ ...snapshot(), processes: [{ pid: "" }] })).toThrow(/pid/);
  });

  it("accepts only hash-redacted typed stale-execution wait receipts", () => {
    const receipt = {
      schema_version: 1,
      kind: "stale_execution",
      pid: "pid_stale",
      recovered_by_owner_sha256: "a".repeat(64),
      prior_owner_sha256: "b".repeat(64),
      prior_lease_sha256: "c".repeat(64),
      prior_execution_generation: 4,
      recovered_execution_generation: 5,
      recovered_state_generation: 6
    };
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      processes: [{ pid: "pid_stale", wait_state: receipt }]
    })).not.toThrow();
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      processes: [{
        pid: "pid_stale",
        wait_state: { ...receipt, prior_execution_lease_id: "raw-token" }
      }]
    })).toThrow(/wait state/);
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      processes: [{
        pid: "pid_stale",
        wait_state: { ...receipt, prior_lease_sha256: "C".repeat(64) }
      }]
    })).toThrow(/wait state/);
  });

  it("validates streamed snapshots before exposing them to React", () => {
    expect(runtimeSnapshotFromSseData({ snapshot: snapshot() })).toMatchObject({ db: "local" });
    expect(() => runtimeSnapshotFromSseData({ snapshot: { schema_version: 3, db: "local" } })).toThrow(/scheduler/);
    expect(() => runtimeSnapshotFromSseData({})).toThrow(/payload/);
  });

  it("accepts versioned MCP manifests but rejects persisted negotiation state", () => {
    const server = mcpServer();
    expect(() => assertRuntimeSnapshot({ ...snapshot(), mcp_servers: [server] })).not.toThrow();
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      mcp_servers: [{ ...server, connection: mcpConnection() }]
    })).toThrow(/operation-local/);
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      mcp_servers: [{ ...server, schema_version: 1, protocol_mode: "auto" }]
    })).toThrow(/summary/);
  });
});

describe("MCP v2 API projection", () => {
  it("accepts only the locked discovery connection and receipt fields", () => {
    const result = mcpDiscovery();
    expect(() => assertMcpDiscoveryResult(result)).not.toThrow();
    expect(() => assertMcpDiscoveryResult({
      ...result,
      connection: {
        configured_mode: "auto",
        era: "modern",
        protocol_revision: "2026-07-28",
        sessionless: true,
        fallback_used: false,
        capabilities: [],
        unsupported_capabilities: []
      }
    })).toThrow(/connection/);
    expect(() => assertMcpDiscoveryResult({
      ...result,
      connection: { ...result.connection, auth_challenge: "secret" }
    })).toThrow(/connection/);
    expect(() => assertMcpDiscoveryResult({
      ...result,
      receipts: [{ ...result.receipts[0], phase: "subscriptions/listen" }]
    })).toThrow(/receipt/);
  });
});

describe("LLM Provider trace API projection", () => {
  it("accepts the content-free list and bounded detail contract", () => {
    const detail = llmDetail();
    expect(() => assertLlmCallPage({ schema_version: 1, items: [detail.call], next_cursor: null, has_more: false })).not.toThrow();
    expect(() => assertLlmCallDetail(detail)).not.toThrow();
    expect(() => assertLlmTraceContentChunk({
      schema_version: 1,
      pid: "pid_1",
      call_id: "call_1",
      field: "attempt_reasoning",
      attempt_sequence: 1,
      content: "inert text",
      next_cursor: null,
      has_more: false,
      content_hash: "a".repeat(64),
      retention_tier: "full"
    })).not.toThrow();
  });

  it("rejects private summary fields, invalid availability, and unsigned readable content", () => {
    const detail = llmDetail();
    expect(() => assertLlmCallPage({
      schema_version: 1,
      items: [{ ...detail.call, response_content: "must stay on demand" }],
      next_cursor: null,
      has_more: false
    })).toThrow(/summary/);
    expect(() => assertLlmCallDetail({
      ...detail,
      call: { ...detail.call, reasoning_availability: "available" }
    })).toThrow(/summary/);
    expect(() => assertLlmCallDetail({
      ...detail,
      content: [{ ...detail.content[0], cursor: null }]
    })).toThrow(/cursor/);
    for (const usage of [
      { total_tokens: "10" },
      { total_tokens: -1 },
      { total_tokens: true },
      { provider_private_usage: 10 },
      { total_tokens: { nested: 10 } }
    ]) {
      expect(() => assertLlmCallPage({
        schema_version: 1,
        items: [{ ...detail.call, usage }],
        next_cursor: null,
        has_more: false
      })).toThrow(/summary/);
      expect(() => assertLlmCallDetail({
        ...detail,
        attempts: [{ ...detail.attempts[0], usage }]
      })).toThrow(/attempt/);
    }
  });
});

describe("task run SSE reconciliation", () => {
  it("ignores lower and equal revisions", () => {
    const current = run(3);
    expect(upsertTaskRunSummary([current], run(2))[0].revision).toBe(3);
    expect(upsertTaskRunSummary([current], { ...run(3), status: "running" })[0].status).toBe("paused");
    expect(upsertTaskRunSummary([current], run(4))[0].revision).toBe(4);
  });

  it("fails closed and never exposes ordinary resume for unknown effects", () => {
    const summary = {
      ...run(),
      status: "needs_attention" as const,
      blockers: [{ kind: "unknown_effect" }],
      allowed_actions: ["resume", "recover", "cancel"] as const
    };
    expect([...allowedTaskRunActions({ ...summary, allowed_actions: ["retry"] })]).toEqual([]);
    expect([...allowedTaskRunActions({ ...summary, allowed_actions: ["resume", "recover", "cancel"] })]).toEqual(["recover", "cancel"]);
    expect(taskRunSummaryFromSseData({ summary: { ...summary, allowed_actions: ["recover", "cancel"] } }).revision).toBe(1);
  });

  it("rejects private payload material in summaries", () => {
    expect(() => taskRunSummaryFromSseData({ ...run(), goal: "secret" })).toThrow(/private field/);
    expect(() => taskRunSummaryFromSseData({ ...run(), blockers: [{ kind: "unknown_effect", payload: { token: "x" } }] })).toThrow(/private field/);
    expect(() => taskRunSummaryFromSseData({ ...run(), payloads_purged: "false" })).toThrow(/purge state/);
  });
});

describe("task run recovery option projection", () => {
  it("accepts only the safe evidence binding used by localized recovery templates", () => {
    const detail = {
      summary: run(),
      requirements: { items: [], next_cursor: null, has_more: false },
      recovery_options: [{
        schema_version: 1,
        option_id: "effect_receipt:binding",
        kind: "effect_receipt",
        requires_receipt: true,
        effect_id: "effect_1",
        expected_transaction_state: "unknown",
        runtime_epoch: 12
      }]
    };

    expect(() => assertTaskRunDetail(detail)).not.toThrow();
    expect(() => assertTaskRunDetail({
      ...detail,
      recovery_options: [{ ...detail.recovery_options[0], provider_secret: "never-render" }]
    })).toThrow(/recovery options/);
    expect(() => assertTaskRunDetail({
      ...detail,
      recovery_options: [{ ...detail.recovery_options[0], runtime_epoch: -1 }]
    })).toThrow(/recovery options/);
  });
});

function snapshot(): Record<string, unknown> {
  return {
    schema_version: 3,
    db: "local",
    scheduler: { auto_run: true, running: false, paused: false },
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

function run(revision = 1) {
  return {
    schema_version: 1 as const,
    run_id: "run_1",
    revision,
    status: "paused" as const,
    display_title: "Durable task",
    root_pid: "pid_1",
    active_pid: "pid_1",
    allowed_actions: ["resume", "cancel"] as Array<"resume" | "cancel">,
    blockers: [],
    retention: "purge_on_terminal" as const,
    payloads_purged: false
  };
}

function mcpServer() {
  return {
    schema_version: 2,
    server_id: "modern",
    protocol_mode: "auto",
    transport: { type: "streamable_http" },
    tools: [],
    timeout_s: 30,
    max_request_bytes: 65_536,
    max_response_bytes: 1_048_576,
    metadata: {}
  };
}

function llmDetail() {
  const call = {
    schema_version: 1,
    call_id: "call_1",
    pid: "pid_1",
    image_id: "coding-agent:v0",
    purpose: "agent_loop",
    status: "ok",
    api: "responses",
    model: "test-model",
    usage: { total_tokens: 10 },
    error: null,
    created_at: "2030-01-01T00:00:00Z",
    completed_at: "2030-01-01T00:00:01Z",
    request_id: "req_1",
    response_id: "resp_1",
    attempt_count: 1,
    coverage: "complete",
    selected_attempt: 1,
    reasoning_availability: "returned",
    payload_retention_tier: "full"
  };
  return {
    schema_version: 1,
    call,
    attempts: [{
      sequence: 1,
      kind: "initial",
      api: "responses",
      status: "ok",
      model: "test-model",
      request_id: "req_1",
      response_id: "resp_1",
      reasoning_availability: "returned",
      reasoning_blocks: [{ type: "summary_text", source: "responses.output", reason: null, chars: 5, bytes: 5, sha256: "a".repeat(64) }],
      output_availability: "returned",
      tool_names: [],
      tool_call_count: 0,
      usage: { total_tokens: 10 },
      started_at: "2030-01-01T00:00:00Z",
      completed_at: "2030-01-01T00:00:01Z",
      duration_ms: 1000,
      error: null
    }],
    content: [{
      field: "attempt_reasoning",
      attempt_sequence: 1,
      availability: "available",
      content_type: "text",
      size_bytes: 5,
      size_chars: 5,
      content_hash: "a".repeat(64),
      cursor: "signed-offset-zero"
    }]
  };
}

function mcpConnection() {
  return {
    protocol_mode: "auto",
    protocol_era: "modern",
    protocol_revision: "2026-07-28",
    sessionless: true,
    fallback_used: false,
    capabilities: ["tools"],
    unsupported_capabilities: ["resources"]
  };
}

function mcpDiscovery() {
  return {
    server_id: "modern",
    connection: mcpConnection(),
    request_bytes: 64,
    response_bytes: 96,
    duration_s: 0.01,
    receipts: [{
      phase: "server/discover",
      request_bytes: 64,
      response_bytes: 96,
      duration_s: 0.01,
      call_started: true
    }]
  };
}
