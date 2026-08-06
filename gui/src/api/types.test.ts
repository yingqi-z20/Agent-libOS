import { describe, expect, it } from "vitest";
import { allowedTaskRunActions, assertLlmCallDetail, assertLlmCallPage, assertLlmTraceContentChunk, assertMcpDiscoveryResult, assertRuntimeSnapshot, assertSemanticAssessmentDetailResponse, assertSemanticAssessmentPage, assertSemanticStatus, assertTaskRunDetail, runtimeSnapshotFromSseData, taskRunSummaryFromSseData, upsertTaskRunSummary } from "./types";

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

describe("Semantic observability API projection", () => {
  it("accepts the bounded payload-free status, page, and detail contract", () => {
    expect(() => assertSemanticStatus(semanticStatus())).not.toThrow();
    expect(() => assertSemanticStatus({ ...semanticStatus(), adapter: "scripted" })).not.toThrow();
    expect(() => assertSemanticAssessmentPage({
      schema_version: 1,
      items: [{ ...semanticSummary(), human_outcome: "approved" }],
      next_cursor: "cursor_1"
    })).not.toThrow();
    expect(() => assertSemanticAssessmentDetailResponse(semanticDetailResponse())).not.toThrow();
  });

  it("accepts the safe-integer ceiling and rejects larger assessment metrics", () => {
    const fields = ["input_tokens", "output_tokens", "cost_microunits", "latency_ms"] as const;
    for (const field of fields) {
      expect(() => assertSemanticAssessmentPage({
        schema_version: 1,
        items: [{ ...semanticSummary(), [field]: Number.MAX_SAFE_INTEGER }],
        next_cursor: null
      })).not.toThrow();
      expect(() => assertSemanticAssessmentPage({
        schema_version: 1,
        items: [{ ...semanticSummary(), [field]: Number.MAX_SAFE_INTEGER + 1 }],
        next_cursor: null
      })).toThrow(/summary/);
    }
  });

  it("rejects private or unknown fields at every response boundary", () => {
    expect(() => assertSemanticStatus({ ...semanticStatus(), prompt: "secret" })).toThrow(/malformed/);
    expect(() => assertSemanticAssessmentPage({
      schema_version: 1,
      items: [{ ...semanticSummary(), raw_content: "secret", job_error: "provider secret", error_code: "private" }],
      next_cursor: null
    })).toThrow(/private field/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...semanticDetailResponse(),
      assessment: { ...semanticDetailResponse().assessment, reasoning: "hidden", raw_human_response: { body: "secret" } }
    })).toThrow(/private field/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...semanticDetailResponse(),
      assessment: {
        ...semanticDetailResponse().assessment,
        findings: [{ ...semanticDetailResponse().assessment.findings[0], explanation: "unbounded" }]
      }
    })).toThrow(/finding/);
  });

  it("rejects invalid digests, confidence, spans, and Shadow auto-approval rates", () => {
    expect(() => assertSemanticAssessmentPage({
      schema_version: 1,
      items: [{ ...semanticSummary(), confidence_bps: 10_001 }],
      next_cursor: null
    })).toThrow(/summary/);
    expect(() => assertSemanticAssessmentPage({
      schema_version: 1,
      items: [{ ...semanticSummary(), input_sha256: "A".repeat(64) }],
      next_cursor: null
    })).toThrow(/summary/);
    expect(() => assertSemanticAssessmentPage({
      schema_version: 1,
      items: [{ ...semanticSummary(), reason_codes: ["model_says_allow"] }],
      next_cursor: null
    })).toThrow(/summary/);
    expect(() => assertSemanticAssessmentPage({
      schema_version: 1,
      items: [{ ...semanticSummary(), human_outcome: "raw answer", cost_microunits: -1 }],
      next_cursor: null
    })).toThrow(/summary/);
    const detail = semanticDetailResponse();
    expect(() => assertSemanticAssessmentDetailResponse({
      ...detail,
      assessment: {
        ...detail.assessment,
        data_findings: [{ ...detail.assessment.data_findings[0], span_start: 9, span_end: 2 }]
      }
    })).toThrow(/data finding/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...detail,
      assessment: {
        ...detail.assessment,
        data_findings: [{
          ...detail.assessment.data_findings[0],
          field: "XcHJvamVjdGVkX2ludGVudF9zZW50aW5lbA",
          span_start: null,
          span_end: null
        }]
      }
    })).toThrow(/data finding/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...detail,
      assessment: {
        ...detail.assessment,
        data_findings: [{ ...detail.assessment.data_findings[0], field: "root_goal" }]
      }
    })).toThrow(/data finding/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...detail,
      assessment: {
        ...detail.assessment,
        data_findings: [{ ...detail.assessment.data_findings[0], span_start: 0, span_end: 1 }]
      }
    })).toThrow(/data finding/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...detail,
      assessment: {
        ...detail.assessment,
        data_findings: [{
          ...detail.assessment.data_findings[0],
          field: "redacted_intent",
          span_start: null,
          span_end: null
        }]
      }
    })).toThrow(/data finding/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...detail,
      assessment: {
        ...detail.assessment,
        data_findings: [{ ...detail.assessment.data_findings[0], category: "model_private_category" }]
      }
    })).toThrow(/data finding/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...detail,
      assessment: { ...detail.assessment, action_sha256: null }
    })).toThrow(/detail/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...detail,
      assessment: { ...detail.assessment, projection_sha256: "A".repeat(64) }
    })).toThrow(/detail/);
    expect(() => assertSemanticAssessmentDetailResponse({
      ...detail,
      assessment: {
        ...detail.assessment,
        manifest_sha256: null,
        resource_sha256: null,
        args_sha256: null,
        state_sha256: null
      }
    })).not.toThrow();
    expect(() => assertSemanticStatus({
      ...semanticStatus(),
      actual_auto_approval: { numerator: 0, denominator: 1, rate: 0 }
    })).toThrow(/malformed/);
  });

  it.each([
    ["mode", { mode: "enforce" }],
    ["adapter", { adapter: "unknown" }],
    ["profile traversal", { profile_id: "../classifier" }],
    ["profile boolean", { profile_id: true }],
    ["boolean counter", { queue: { ...semanticStatus().queue, queued: true } }],
    ["string counter", { queue: { ...semanticStatus().queue, leased: "0" } }],
    ["negative counter", { assessments: { ...semanticStatus().assessments, error: -1 } }]
  ])("rejects malformed Semantic status %s", (_label, override) => {
    expect(() => assertSemanticStatus({ ...semanticStatus(), ...override })).toThrow(/malformed/);
  });

  it("requires complete and consistent Semantic status aggregate mappings", () => {
    const value = semanticStatus();
    const { stale_input: _omitted, ...incompleteByStatus } = value.assessments.by_status;
    expect(() => assertSemanticStatus({
      ...value,
      assessments: { ...value.assessments, by_status: incompleteByStatus }
    })).toThrow(/malformed/);
    expect(() => assertSemanticStatus({
      ...value,
      assessments: {
        ...value.assessments,
        by_domain: { ...value.assessments.by_domain, filesystem: 1 }
      }
    })).toThrow(/malformed/);
  });

  it("rejects inconsistent Semantic status derived totals", () => {
    const value = semanticStatus();
    expect(() => assertSemanticStatus({
      ...value,
      assessments: { ...value.assessments, success: 1 }
    })).toThrow(/malformed/);
    expect(() => assertSemanticStatus({
      ...value,
      assessments: { ...value.assessments, would_issue_exact_once: 0 }
    })).toThrow(/malformed/);
    expect(() => assertSemanticStatus({
      ...value,
      assessments: { ...value.assessments, ood: 1 }
    })).toThrow(/malformed/);
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

function semanticStatus() {
  return {
    schema_version: 2,
    mode: "shadow",
    adapter: "deterministic",
    profile_id: null,
    queue: { queued: 1, leased: 0, succeeded: 2, failed: 0, cancelled: 0, capture_failures: 0 },
    assessments: {
      total: 2,
      success: 2,
      error: 0,
      ood: 0,
      would_issue_exact_once: 1,
      would_deny: 0,
      require_human: 1,
      by_status: {
        success: 2,
        skipped_policy: 0,
        egress_blocked: 0,
        timeout: 0,
        provider_error: 0,
        provider_outcome_unknown: 0,
        invalid_schema: 0,
        ood: 0,
        abstained: 0,
        stale_input: 0
      },
      by_domain: {
        filesystem: 2,
        shell: 0,
        git: 0,
        jsonrpc: 0,
        mcp: 0,
        runtime: 0,
        unknown: 0
      }
    },
    actual_auto_approval: { numerator: 0, denominator: 0, rate: null }
  };
}

function semanticSummary() {
  return {
    assessment_id: "semantic_1",
    job_id: "semantic_job_1",
    kind: "approval",
    status: "success",
    domain: "filesystem",
    action_id: "filesystem.read",
    pid: "pid_1",
    request_id: "human_1",
    operation_id: "operation_1",
    effect_id: "effect_1",
    shadow_outcome: "require_human",
    reason_codes: ["missing_authoritative_predicate"],
    ood: false,
    abstain: false,
    confidence_bps: 8750,
    calibration_bucket: "high",
    input_tokens: 120,
    output_tokens: 20,
    cost_microunits: 45,
    classifier_id: "scripted",
    classifier_version: "v1",
    artifact_sha256: "a".repeat(64),
    input_sha256: "b".repeat(64),
    feature_snapshot_sha256: "c".repeat(64),
    policy_sha256: "d".repeat(64),
    created_at: "2030-01-01T00:00:00Z",
    completed_at: "2030-01-01T00:00:01Z",
    latency_ms: 1000,
    human_outcome: null,
    tenant_bucket_sha256: "6".repeat(64)
  };
}

function semanticDetailResponse() {
  return {
    schema_version: 1,
    assessment: {
      ...semanticSummary(),
      findings: [{
        code: "missing_authoritative_predicate",
        severity: "medium",
        confidence_bps: 8750,
        evidence_sha256: "e".repeat(64),
        source: "model"
      }],
      data_findings: [{
        category: "source_code",
        field: "approval.request",
        span_start: null,
        span_end: null,
        sensitivity_floor: "confidential",
        integrity_ceiling: "unknown",
        trust_ceiling: "untrusted",
        confidence_bps: 9200,
        evidence_sha256: "f".repeat(64)
      }],
      matched_rule_ids: [],
      proven_predicates: ["binding_current"],
      missing_predicates: ["low_risk"],
      source_refs_sha256: "1".repeat(64),
      data_labels_sha256: "2".repeat(64),
      sink_identity_sha256: "3".repeat(64),
      tool_schema_sha256: "4".repeat(64),
      provider_spec_sha256: "5".repeat(64),
      manifest_sha256: "7".repeat(64),
      action_sha256: "8".repeat(64),
      resource_sha256: "9".repeat(64),
      args_sha256: "0".repeat(64),
      state_sha256: "a".repeat(64),
      projection_sha256: "b".repeat(64)
    }
  };
}
