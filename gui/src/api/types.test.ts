import { describe, expect, it } from "vitest";
import {
  allowedTaskRunActions,
  assertLlmCallDetail,
  assertLlmCallPage,
  assertLlmTraceContentChunk,
  assertMcpDiscoveryResult,
  assertMcpAuthorizationChallenge,
  assertMcpContinuationResult,
  assertMcpInputRequired,
  assertMcpOAuthProfileInput,
  assertMcpOAuthStatus,
  assertMcpOAuthStatuses,
  assertMcpPromptOperationResult,
  assertMcpRemoteTask,
  assertMcpResourceOperationResult,
  assertMcpResourcePage,
  assertMcpResourceTemplatePage,
  assertMcpSubscription,
  assertMcpSubscriptionEvents,
  assertRuntimeSnapshot,
  assertSemanticAssessmentDetailResponse,
  assertSemanticAssessmentPage,
  assertSemanticControlHistoryPage,
  assertSemanticControlState,
  assertSemanticFlowEdgePage,
  assertSemanticFlowEntityPage,
  assertSemanticFlowLineage,
  assertSemanticHealthEventPage,
  assertSemanticMachineSettlementPage,
  assertSemanticMetrics,
  assertSemanticPolicyEpochPage,
  assertSemanticStatus,
  assertTaskRunDetail,
  runtimeSnapshotFromSseData,
  taskRunSummaryFromSseData,
  upsertTaskRunSummary
} from "./types";

describe("assertRuntimeSnapshot", () => {
  it("accepts the minimum same-build snapshot shape", () => {
    const value = snapshot();
    expect(() => assertRuntimeSnapshot(value)).not.toThrow();
  });

  it("rejects malformed scheduler and collection fields before rendering", () => {
    expect(() => assertRuntimeSnapshot({ ...snapshot(), scheduler: { auto_run: "yes" } })).toThrow(/scheduler/);
    expect(() => assertRuntimeSnapshot({ ...snapshot(), events: {} })).toThrow(/events/);
  });

  it("requires consistent TaskRun launch availability", () => {
    const valid = snapshot();
    expect(() => assertRuntimeSnapshot({
      ...valid,
      task_run_launch: { enabled: true, plaintext_payloads_enabled: true, available: true }
    })).not.toThrow();
    expect(() => assertRuntimeSnapshot({ ...valid, task_run_launch: undefined })).toThrow(/TaskRun launch/);
    expect(() => assertRuntimeSnapshot({
      ...valid,
      task_run_launch: { enabled: true, plaintext_payloads_enabled: false, available: true }
    })).toThrow(/TaskRun launch/);
    expect(() => assertRuntimeSnapshot({
      ...valid,
      task_run_launch: { enabled: true, plaintext_payloads_enabled: false, available: false, secret: "no" }
    })).toThrow(/TaskRun launch/);
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

describe("MCP v3 GUI projection", () => {
  it("accepts exact v3 registry identity and modern pages", () => {
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      mcp_servers: [{ ...mcpServer(), schema_version: 3, protocol_mode: "2026-07-28" }]
    })).not.toThrow();
    expect(() => assertRuntimeSnapshot({
      ...snapshot(),
      mcp_servers: [{ ...mcpServer(), schema_version: 3, protocol_mode: "auto" }]
    })).toThrow(/summary/);
    expect(() => assertMcpResourcePage({
      items: [{ resource_id: "logical-doc", name: "Document" }],
      next_cursor: "opaque",
      cache_hint: { ttl_ms: 100, scope: "private" },
      has_more: true
    })).not.toThrow();
  });

  it("rejects MCP Apps and private provider identifiers", () => {
    expect(() => assertMcpResourcePage({
      items: [{ resource_id: "ui://app", name: "App" }],
      next_cursor: null,
      cache_hint: null
    })).toThrow(/unsupported MCP App/);
    expect(() => assertMcpPromptOperationResult({
      kind: "complete",
      preview_sha256: "a".repeat(64),
      value: {
        prompt_id: "review",
        messages: [],
        user_confirmation_required: true,
        access_token: "private"
      }
    })).toThrow(/private credential/);
    expect(() => assertMcpRemoteTask({
      kind: "remote_task",
      task_ref: "local-ref",
      remote_task_id: "private-id",
      status: "working"
    })).toThrow(/private credential/);
    expect(() => assertMcpPromptOperationResult({
      kind: "complete",
      value: {
        prompt_id: "review",
        messages: [],
        user_confirmation_required: true
      }
    })).toThrow(/preview binding/);
  });

  it("parses MIME parameters and rejects every equivalent MCP Apps profile", () => {
    const appMimeTypes = [
      "text/html;profile=mcp-app;charset=utf-8",
      'TEXT/HTML; charset="utf-8"; PROFILE = "MCP-APP"',
      'text/html; profile="mcp-app"; sandbox=yes; note="a;b"'
    ];
    for (const mime_type of appMimeTypes) {
      expect(() => assertMcpResourcePage({
        items: [{ resource_id: "app", name: "App", mime_type }],
        next_cursor: null,
        cache_hint: null
      })).toThrow(/unsupported MCP App/);
      expect(() => assertMcpResourceTemplatePage({
        items: [{ template_id: "app-template", name: "App", mime_type }],
        next_cursor: null,
        cache_hint: null
      })).toThrow(/unsupported MCP App/);
      expect(() => assertMcpResourceOperationResult({
        kind: "complete",
        value: {
          resource_id: "safe-resource",
          provenance: "untrusted_mcp_resource",
          contents: [{
            kind: "resource_link",
            resource_handle: "opaque-local-handle",
            name: "App",
            mime_type
          }]
        }
      })).toThrow(/unsupported MCP App/);
    }
    expect(() => assertMcpResourcePage({
      items: [{
        resource_id: "ordinary-html",
        name: "Document",
        mime_type: 'text/html; charset="utf-8"; profile=document'
      }],
      next_cursor: null,
      cache_hint: null
    })).not.toThrow();
  });

  it("rejects case-folded MCP Apps metadata namespaces at every nesting depth", () => {
    expect(() => assertMcpResourcePage({
      items: [{
        resource_id: "resource",
        name: "Resource",
        metadata: { "Ui/component": "ui://private-app" }
      }],
      next_cursor: null,
      cache_hint: null
    })).toThrow(/unsupported MCP Apps metadata/);
    expect(() => assertMcpResourceTemplatePage({
      items: [{
        template_id: "template",
        name: "Template",
        metadata: { nested: { "uI/widget": { enabled: true } } }
      }],
      next_cursor: null,
      cache_hint: null
    })).toThrow(/unsupported MCP Apps metadata/);
    expect(() => assertMcpResourceOperationResult({
      kind: "complete",
      value: {
        resource_id: "resource",
        provenance: "untrusted_mcp_resource",
        contents: [{
          kind: "text",
          text: "inert",
          metadata: { nested: { "UI/render": "remote-html" } }
        }]
      }
    })).toThrow(/unsupported MCP Apps metadata/);
  });

  it("accepts non-secret OAuth status metadata only", () => {
    expect(() => assertMcpOAuthStatus({
      profile_id: "profile-local",
      status: "authorized",
      scopes: ["resource.read"],
      principal_sha256: "a".repeat(64)
    })).not.toThrow();
    expect(() => assertMcpOAuthStatus({
      profile_id: "profile-local",
      status: "authorized",
      scopes: [],
      refresh_token: "private"
    })).toThrow(/private credential/);
    for (const extra of [
      { token: "private" },
      { state: "private" },
      { harmless_but_unknown: "still forbidden" }
    ]) {
      expect(() => assertMcpOAuthStatus({
        profile_id: "profile-local",
        status: "authorized",
        scopes: [],
        ...extra
      })).toThrow(/OAuth status/);
    }
    expect(() => assertMcpAuthorizationChallenge({
      challenge_id: "challenge-local",
      authorization_url: "https://authorization.invalid/authorize",
      expires_at: "2030-01-01T00:00:00Z"
    })).not.toThrow();
    for (const extra of [{ state: "private" }, { client_assertion: "private" }]) {
      expect(() => assertMcpAuthorizationChallenge({
        challenge_id: "challenge-local",
        authorization_url: "https://authorization.invalid/authorize",
        expires_at: "2030-01-01T00:00:00Z",
        ...extra
      })).toThrow(/OAuth challenge/);
    }
  });

  it("accepts exact non-secret OAuth Host profiles and status-only lists", () => {
    const profile = {
      profile_id: "profile-local",
      server_id: "server-local",
      resource_uri: "https://resource.example/mcp",
      expected_issuer: "https://issuer.example",
      redirect_uri: "http://127.0.0.1/callback",
      client_id: "gui-client",
      registration_mode: "preregistered",
      token_endpoint_auth_method: "client_secret_basic",
      allowed_scopes: ["resource.read"],
      default_scopes: ["resource.read"],
      allow_loopback_http: true,
      protocol_revision: "2026-07-28",
      transport: "streamable_http"
    };
    expect(() => assertMcpOAuthProfileInput(profile)).not.toThrow();
    expect(() => assertMcpOAuthProfileInput({
      ...profile,
      client_secret: "private"
    })).toThrow(/profile input/);
    expect(() => assertMcpOAuthProfileInput({
      ...profile,
      registration_mode: "dcr"
    })).toThrow(/profile input/);
    expect(() => assertMcpOAuthStatuses([{
      profile_id: "profile-local",
      status: "authorization_required",
      scopes: []
    }])).not.toThrow();
    expect(() => assertMcpOAuthStatuses([{
      profile_id: "profile-local",
      status: "authorized",
      scopes: [],
      access_token: "private"
    }])).toThrow(/private credential/);
  });

  it("rejects unknown keys on modern discriminated and subscription DTOs", () => {
    expect(() => assertMcpContinuationResult({
      kind: "complete",
      value: null,
      provider_debug: "forbidden"
    })).toThrow(/complete result/);
    expect(() => assertMcpContinuationResult({
      kind: "input_required",
      continuation_id: "continuation-local",
      revision: 1,
      respondable: true,
      input_requests: [{
        request_id: "input-local",
        kind: "elicitation",
        schema: { type: "object" },
        provider_debug: "forbidden"
      }],
      human_request_id: "human-local",
      human_revision: 1,
      human_preview_sha256: "a".repeat(64)
    })).toThrow(/input request/);
    expect(() => assertMcpRemoteTask({
      kind: "remote_task",
      task_ref: "task-local",
      revision: 1,
      status: "working",
      input_requests: [],
      provider_debug: "forbidden"
    })).toThrow(/remote task projection/);
    expect(() => assertMcpSubscription({
      subscription_id: "subscription-local",
      server_id: "server-local",
      status: "active",
      requested_filters: [],
      acknowledged_filters: [],
      provider_debug: "forbidden"
    })).toThrow(/subscription projection/);
    expect(() => assertMcpSubscriptionEvents([{
      sequence: 1,
      event_type: "resourcesListChanged",
      payload: {},
      received_at: "2030-01-01T00:00:00Z",
      provenance: "untrusted_mcp_notification",
      provider_debug: "forbidden"
    }])).toThrow(/subscription event/);
  });

  it("requires a real HumanRequest receipt for every input-required projection", () => {
    const inputRequired = {
      kind: "input_required",
      continuation_id: "continuation-local",
      revision: 3,
      respondable: true,
      input_requests: [{
        request_id: "field-local",
        kind: "elicitation",
        mode: "url",
        prompt: "Review the untrusted provider request",
        schema: { type: "object" },
        inert_url: "https://provider.invalid/review"
      }],
      human_request_id: "human-local",
      human_revision: 4,
      human_preview_sha256: "c".repeat(64)
    };
    expect(() => assertMcpContinuationResult(inputRequired)).not.toThrow();
    expect(() => assertMcpInputRequired(inputRequired)).not.toThrow();
    expect(() => assertMcpInputRequired({
      ...inputRequired,
      kind: "complete",
      value: null
    })).toThrow(/input-required/);
    expect(() => assertMcpContinuationResult({
      ...inputRequired,
      human_request_id: null
    })).toThrow(/input-required/);
    expect(() => assertMcpContinuationResult({
      ...inputRequired,
      human_preview_sha256: "C".repeat(64)
    })).toThrow(/input-required/);
    expect(() => assertMcpInputRequired({
      ...inputRequired,
      input_requests: []
    })).not.toThrow();

    const unsupported = {
      kind: "input_required",
      continuation_id: "",
      revision: 0,
      respondable: false,
      input_requests: [{
        request_id: "sampling-local",
        kind: "sampling_unsupported",
        schema: {}
      }],
      human_request_id: null,
      human_revision: null,
      human_preview_sha256: null
    };
    expect(() => assertMcpInputRequired(unsupported)).not.toThrow();
    expect(() => assertMcpInputRequired({
      ...unsupported,
      continuation_id: "forged-continuation"
    })).toThrow(/unsupported input-required/);
    expect(() => assertMcpInputRequired({
      ...unsupported,
      human_request_id: "forged-human"
    })).toThrow(/unsupported input-required/);
    expect(() => assertMcpInputRequired({
      ...unsupported,
      input_requests: [{
        request_id: "elicitation-local",
        kind: "elicitation",
        mode: "form",
        prompt: "not allowed on typed unsupported",
        schema: { type: "object", properties: {} }
      }]
    })).toThrow(/unsupported input-required/);

    const task = {
      kind: "remote_task",
      task_ref: "task-local",
      revision: 8,
      status: "input_required",
      input_requests: inputRequired.input_requests,
      human_request_id: "human-task-local",
      human_revision: 2,
      human_preview_sha256: "d".repeat(64)
    };
    expect(() => assertMcpRemoteTask(task)).not.toThrow();
    expect(() => assertMcpRemoteTask({
      ...task,
      human_preview_sha256: null
    })).toThrow(/Human request receipt/);
    expect(() => assertMcpRemoteTask({
      ...task,
      status: "working"
    })).toThrow(/Human request receipt/);
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

  it("rejects invalid digests, confidence, spans, and inconsistent auto-approval rates", () => {
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
      actual_auto_approval: { numerator: 0, denominator: 1, rate: 0.5 }
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
    expect(() => assertSemanticStatus({
      ...value,
      review_metrics: { ...value.review_metrics, issued_reviewed: 1, issued_review_rate: null }
    })).toThrow(/malformed/);
    expect(() => assertSemanticStatus({
      ...value,
      actual_auto_approval: { numerator: 0, denominator: 1, rate: 0 }
    })).toThrow(/malformed/);
  });

  it("does not echo rejected Semantic enum or private-field values", () => {
    const sentinel = "SEMANTIC_SECRET_SENTINEL_DO_NOT_ECHO";
    for (const candidate of [
      { ...semanticStatus(), [sentinel]: true },
      {
        schema_version: 1,
        items: [{ ...semanticSummary(), reason_codes: [sentinel] }],
        next_cursor: null
      }
    ]) {
      let message = "";
      try {
        if ("mode" in candidate) assertSemanticStatus(candidate);
        else assertSemanticAssessmentPage(candidate);
      } catch (error) {
        message = error instanceof Error ? error.message : String(error);
      }
      expect(message).not.toBe("");
      expect(message).not.toContain(sentinel);
    }
  });
});

describe("Semantic Phase 2-4 read-only evidence decoders", () => {
  it("accepts exact payload-free Flow, settlement, epoch, control, health, and metrics projections", () => {
    const value = semanticPhase24Fixtures();
    expect(() => assertSemanticFlowEntityPage({ schema_version: 1, items: [value.entity], next_cursor: null })).not.toThrow();
    expect(() => assertSemanticFlowLineage(value.lineage)).not.toThrow();
    expect(() => assertSemanticMachineSettlementPage({ schema_version: 1, items: [value.settlement], next_cursor: null })).not.toThrow();
    expect(() => assertSemanticPolicyEpochPage({ schema_version: 1, items: [value.epoch], next_cursor: null })).not.toThrow();
    expect(() => assertSemanticControlState(value.control)).not.toThrow();
    expect(() => assertSemanticControlHistoryPage({ schema_version: 1, items: [value.control], next_cursor: null })).not.toThrow();
    expect(() => assertSemanticHealthEventPage({ schema_version: 1, items: [value.health], next_cursor: null })).not.toThrow();
    expect(() => assertSemanticMetrics(value.metrics)).not.toThrow();
    expect(() => assertSemanticControlState({
      ...value.control,
      mode: "off",
      tripped: false,
      trip_code: null
    })).toThrow(/control/);
  });

  it("rejects private fields, invalid lineage identities, and inconsistent safety metrics", () => {
    const value = semanticPhase24Fixtures();
    expect(() => assertSemanticFlowEntityPage({
      schema_version: 1,
      items: [{ ...value.entity, raw_content: "SECRET_MUST_NOT_RENDER" }],
      next_cursor: null
    })).toThrow(/entity/);
    expect(() => assertSemanticFlowEntityPage({
      schema_version: 1,
      items: [{ ...value.entity, entity_id: "entity with spaces" }],
      next_cursor: null
    })).toThrow(/entity/);
    expect(() => assertSemanticFlowEntityPage({
      schema_version: 1,
      items: [{ ...value.entity, identity_present: false, identity_mixed: true }],
      next_cursor: null
    })).toThrow(/entity/);
    expect(() => assertSemanticFlowEdgePage({
      schema_version: 1,
      items: [{
        ...value.edge,
        source_node_id: "entity_1",
        source_node_type: "entity",
        target_node_id: "entity_1",
        target_node_type: "entity"
      }],
      next_cursor: null
    })).toThrow(/edge/);
    expect(() => assertSemanticFlowEntityPage({
      schema_version: 1,
      items: [{ ...value.entity, created_at: "2030-01-01T00:00:00" }],
      next_cursor: null
    })).toThrow(/entity/);
    expect(() => assertSemanticFlowLineage({
      ...value.lineage,
      items: [{ ...value.lineage.items[0], node_type: "entity", node: value.activity }]
    })).toThrow(/entity/);
    expect(() => assertSemanticFlowLineage({
      ...value.lineage,
      items: [{
        ...value.lineage.items[0],
        node: { ...value.activity, activity_id: "unrelated_activity" }
      }]
    })).toThrow(/edge endpoint/);
    expect(() => assertSemanticMachineSettlementPage({
      schema_version: 1,
      items: [{ ...value.settlement, reasoning: "hidden" }],
      next_cursor: null
    })).toThrow(/settlement/);
    expect(() => assertSemanticMachineSettlementPage({
      schema_version: 1,
      items: [{ ...value.settlement, capability_id: null }],
      next_cursor: null
    })).toThrow(/settlement/);
    expect(() => assertSemanticMachineSettlementPage({
      schema_version: 1,
      items: [{
        ...value.settlement,
        outcome: "denied",
        capability_id: null,
        matched_rule_id: null,
        reason_codes: ["risk_detected"]
      }],
      next_cursor: null
    })).toThrow(/settlement/);
    expect(() => assertSemanticHealthEventPage({
      schema_version: 1,
      items: [{ ...value.health, severity: "error" }],
      next_cursor: null
    })).toThrow(/health/);
    expect(() => assertSemanticMetrics({
      ...value.metrics,
      review_metrics: {
        ...value.metrics.review_metrics,
        reviewed: 2,
        safe: 1,
        unsafe: 1,
        unsafe_rate: 0.25
      }
    })).toThrow(/metrics/);
    expect(() => assertSemanticMetrics({ ...value.metrics, risk: "unknown" })).toThrow(/metrics/);
    expect(() => assertSemanticMetrics({
      ...value.metrics,
      review_metrics: {
        reviewed: 12,
        safe: 12,
        unsafe: 0,
        unsafe_rate: 0,
        issued_reviewed: 0,
        issued_review_rate: 0
      }
    })).not.toThrow();
    expect(() => assertSemanticMetrics({
      ...value.metrics,
      review_metrics: { ...value.metrics.review_metrics, issued_reviewed: 3, issued_review_rate: 1 }
    })).toThrow(/metrics/);
    expect(() => assertSemanticMetrics({
      ...value.metrics,
      actual_auto_approval: { numerator: 1, denominator: 2, rate: 0.5 }
    })).toThrow(/metrics/);
    expect(() => assertSemanticHealthEventPage({
      schema_version: 1,
      items: [{ ...value.health, event_kind: "SEMANTIC_SECRET_SENTINEL_DO_NOT_ECHO" }],
      next_cursor: null
    })).toThrow(/health/);
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
    task_run_launch: { enabled: true, plaintext_payloads_enabled: false, available: false },
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
    schema_version: 3,
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
    control: {
      catalog_version: null,
      active_epoch_id: null,
      active_epoch_sha256: null,
      generation: 0,
      state: "inactive",
      trip_reason_code: null
    },
    flow: {
      schema_version: 1,
      available: false,
      counts: { entities: 0, activities: 0, edges: 0, label_assertions: 0 },
      coverage: { complete: 0, partial: 0, unknown: 0, conflict: 0, stale: 0 },
      capture_failures: 0,
      legacy_history: {
        present: false,
        source_schema_version: null,
        assessment_count: 0,
        coverage: null,
        evidence_sha256: null,
        created_at: null
      }
    },
    machine: {
      eligible: 0,
      issued: 0,
      consumed: 0,
      succeeded: 0,
      failed: 0,
      unknown: 0,
      expired: 0,
      revoked: 0,
      race_lost: 0,
      denied: 0
    },
    actual_auto_approval: { numerator: 0, denominator: 0, rate: null },
    review_metrics: {
      reviewed: 0,
      safe: 0,
      unsafe: 0,
      unsafe_rate: null,
      issued_reviewed: 0,
      issued_review_rate: null
    }
  };
}

function semanticPhase24Fixtures() {
  const labels = {
    sensitivity: "normal",
    trust_level: "verified",
    integrity: "checked"
  };
  const entity = {
    schema_version: 1,
    entity_id: "entity_1",
    kind: "provider_result",
    pid: "pid_1",
    tenant_bucket_sha256: "1".repeat(64),
    content_sha256: "2".repeat(64),
    version_sha256: "3".repeat(64),
    provenance_sha256: "4".repeat(64),
    baseline_labels: labels,
    coverage: "complete",
    identity_present: true,
    identity_mixed: false,
    created_at: "2030-01-01T00:00:00Z"
  };
  const activity = {
    schema_version: 1,
    activity_id: "activity_1",
    kind: "provider_call",
    pid: "pid_1",
    action_id: "filesystem.read",
    effect_id: "effect_1",
    state_sha256: "5".repeat(64),
    provider_spec_sha256: "6".repeat(64),
    tool_schema_sha256: null,
    model_artifact_sha256: null,
    tenant_bucket_sha256: "1".repeat(64),
    created_at: "2030-01-01T00:00:01Z"
  };
  const edge = {
    schema_version: 1,
    edge_id: "edge_1",
    relation: "direct",
    source_node_id: "activity_1",
    source_node_type: "activity",
    target_node_id: "entity_1",
    target_node_type: "entity",
    pid: "pid_1",
    provenance_sha256: "7".repeat(64),
    created_at: "2030-01-01T00:00:02Z"
  };
  const lineage = {
    schema_version: 1,
    root_node_id: "entity_1",
    direction: "upstream",
    items: [{ depth: 1, edge, node_type: "activity", node: activity }],
    effective_labels: labels,
    coverage: "complete",
    next_cursor: null,
    truncated: false
  };
  const settlement = {
    schema_version: 1,
    settlement_id: "settlement_1",
    assessment_id: "assessment_1",
    job_id: "job_1",
    request_id: "request_1",
    request_revision: 2,
    pid: "pid_1",
    operation_id: "operation_1",
    effect_id: "effect_1",
    epoch_id: "epoch_1",
    policy_sha256: "8".repeat(64),
    tenant_bucket_sha256: "1".repeat(64),
    action_id: "filesystem.read",
    outcome: "issued",
    capability_id: "capability_1",
    binding_sha256: "9".repeat(64),
    decision_sha256: "a".repeat(64),
    matched_rule_id: "read_reports",
    reason_codes: ["policy_match"],
    created_at: "2030-01-01T00:00:03Z",
    human_outcome: null,
    human_outcome_source: null,
    human_outcome_request_revision: null,
    human_outcome_decision_sha256: null,
    human_outcome_created_at: null
  };
  const epoch = {
    schema_version: 1,
    epoch_id: "epoch_1",
    generation: 1,
    catalog_version: 1,
    policy_sha256: "8".repeat(64),
    expected_previous_sha256: null,
    created_at: "2030-01-01T00:00:04Z"
  };
  const control = {
    schema_version: 1,
    revision: 1,
    generation: 1,
    mode: "canary_auto",
    active_epoch_id: "epoch_1",
    active_policy_sha256: "8".repeat(64),
    tripped: false,
    trip_code: null,
    updated_at: "2030-01-01T00:00:05Z"
  };
  const health = {
    schema_version: 1,
    event_id: "health_1",
    event_kind: "semantic_policy_activated",
    severity: "info",
    epoch_id: "epoch_1",
    tenant_bucket_sha256: "1".repeat(64),
    evidence_sha256: "b".repeat(64),
    created_at: "2030-01-01T00:00:06Z"
  };
  const metrics = {
    schema_version: 1,
    window: "7d",
    action_id: "filesystem.read",
    tenant_bucket_sha256: "1".repeat(64),
    epoch_id: "epoch_1",
    risk: "low",
    machine: {
      eligible: 2,
      issued: 2,
      consumed: 1,
      succeeded: 1,
      failed: 0,
      unknown: 0,
      expired: 0,
      revoked: 0,
      race_lost: 0,
      denied: 0
    },
    actual_auto_approval: { numerator: 2, denominator: 2, rate: 1 },
    review_metrics: {
      reviewed: 2,
      safe: 2,
      unsafe: 0,
      unsafe_rate: 0,
      issued_reviewed: 2,
      issued_review_rate: 1
    }
  };
  return { entity, activity, edge, lineage, settlement, epoch, control, health, metrics };
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
