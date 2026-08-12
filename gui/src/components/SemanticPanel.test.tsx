// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import type {
  SemanticAssessmentDetail,
  SemanticAssessmentSummary,
  SemanticControlState,
  SemanticFlowEntity,
  SemanticFlowLineage,
  SemanticFlowStatus,
  SemanticHealthEvent,
  SemanticMachineSettlement,
  SemanticMetrics,
  SemanticPolicyEpochSummary,
  SemanticStatus
} from "../api/types";
import { I18nProvider } from "../i18n";
import { mergeAssessments, SemanticPanel, type SemanticPanelClient } from "./SemanticPanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("SemanticPanel", () => {
  it("renders queue health, Shadow outcomes, OOD, and payload-free provenance", async () => {
    const client = semanticClient();
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(
        <I18nProvider initialLanguage="en">
          <SemanticPanel client={client} pid="pid_1" connectionKey="1" />
        </I18nProvider>
      );
      await flushPromises();
    });
    await act(flushPromises);

    expect(container.querySelector("[data-testid='semantic-panel']")).not.toBeNull();
    expect(container.textContent).toContain("Read-only semantic evidence for process pid_1");
    expect(container.textContent).toContain("Semantic results never approve, reject, relabel, or release data");
    expect(container.textContent).toContain("Queued");
    expect(container.textContent).toContain("Would approve once");
    expect(container.textContent).toContain("Require human");
    expect(container.textContent).toContain("33.33%");
    expect(container.textContent).toContain("Data-flow lineage");
    expect(container.textContent).toContain("Legacy v5 flow history");
    expect(container.textContent).toContain("4 migrated assessment(s) have unknown lineage coverage");
    expect(container.textContent).toContain("provider_result");
    expect(container.textContent).toContain("Machine settlements");
    expect(container.textContent).toContain("Issued grants reviewed");
    expect(container.textContent).toContain("Issued-grant review coverage");
    expect(container.textContent).toContain("100.00%");
    expect(container.textContent).toContain("settlement_1");
    expect(container.textContent).toContain("approved · human · request revision 1");
    expect(container.textContent).toContain("c".repeat(64));
    expect(container.textContent).toContain("Policy epochs");
    expect(container.textContent).toContain("epoch_1");
    expect(container.textContent).toContain("Safety health events");
    expect(container.textContent).toContain("semantic_policy_activated");
    expect(container.textContent).toContain("OOD");
    const aggregateText = container.querySelector(".semanticAggregateGroups")?.textContent ?? "";
    expect(aggregateText).toContain("Success3");
    expect(aggregateText).toContain("Timed out1");
    expect(aggregateText).toContain("Stale input1");
    expect(aggregateText).toContain("Filesystem2");
    expect(aggregateText).toContain("Shell1");
    expect(aggregateText).toContain("Unknown1");
    expect(container.textContent).toContain("missing_authoritative_predicate");
    expect(container.textContent).toContain("filesystem.read");
    expect(container.textContent).toContain("Input tokens");
    expect(container.textContent).toContain("120");
    expect(container.textContent).toContain("Tenant bucket");
    expect(container.textContent).toContain("Feature snapshot");
    expect(container.textContent).toContain("Action binding");
    expect(container.textContent).toContain("Redacted projection");
    expect(container.textContent).toContain("b".repeat(64));
    expect(container.textContent).toContain("approval.request");
    expect(client.listSemanticAssessments).toHaveBeenCalledWith(
      { pid: "pid_1" },
      50,
      undefined,
      expect.objectContaining({ signal: expect.any(AbortSignal), timeoutMs: 15_000 })
    );
    expect(client.getSemanticAssessment).toHaveBeenCalledWith(
      "assessment_1",
      expect.objectContaining({ signal: expect.any(AbortSignal), timeoutMs: 15_000 })
    );
    const buttonText = [...container.querySelectorAll("button")].map((button) => button.textContent ?? "").join(" ");
    expect(buttonText).not.toMatch(/activate|rotate|revoke|trip|enable canary|disable canary/i);
    expect(container.querySelector("input[type='checkbox']")).toBeNull();

    await act(() => root.unmount());
    container.remove();
  });

  it("applies process, status, and domain filters and loads bounded older pages", async () => {
    const older = { ...summary(), assessment_id: "assessment_2", job_id: "job_2", domain: "git" as const };
    const client = semanticClient();
    vi.mocked(client.listSemanticAssessments)
      .mockResolvedValueOnce({ schema_version: 1, items: [summary()], next_cursor: "cursor_1" })
      .mockResolvedValueOnce({ schema_version: 1, items: [summary()], next_cursor: "cursor_2" })
      .mockResolvedValueOnce({ schema_version: 1, items: [summary()], next_cursor: "cursor_3" })
      .mockResolvedValueOnce({ schema_version: 1, items: [summary(), older], next_cursor: null });
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(
        <I18nProvider initialLanguage="en">
          <SemanticPanel client={client} pid="pid_1" />
        </I18nProvider>
      );
      await flushPromises();
    });
    await act(flushPromises);

    const selects = container.querySelectorAll<HTMLSelectElement>(".semanticFilters select");
    await act(async () => {
      selects[0].value = "success";
      selects[0].dispatchEvent(new Event("change", { bubbles: true }));
      await flushPromises();
    });
    await act(async () => {
      selects[1].value = "filesystem";
      selects[1].dispatchEvent(new Event("change", { bubbles: true }));
      await flushPromises();
    });

    const loadMore = container.querySelector<HTMLButtonElement>(".semanticLoadMore");
    await act(async () => {
      loadMore?.click();
      await flushPromises();
    });

    expect(client.listSemanticAssessments).toHaveBeenLastCalledWith(
      { pid: "pid_1", status: "success", domain: "filesystem" },
      50,
      "cursor_3",
      expect.objectContaining({ signal: expect.any(AbortSignal), timeoutMs: 15_000 })
    );
    expect(container.textContent).toContain("assessment_2");
    expect(container.querySelectorAll(".semanticList > button:not(.semanticLoadMore)")).toHaveLength(2);

    await act(() => root.unmount());
    container.remove();
  });

  it("merges cursor pages without duplicating assessments", () => {
    expect(mergeAssessments(
      [summary(), { ...summary(), assessment_id: "assessment_2", status: "timeout" }],
      [{ ...summary(), status: "ood" }]
    ).map((item) => [item.assessment_id, item.status])).toEqual([
      ["assessment_1", "ood"],
      ["assessment_2", "timeout"]
    ]);
  });

  it("fails closed when independently read control snapshots disagree", async () => {
    const client = semanticClient();
    vi.mocked(client.getSemanticControl).mockResolvedValue({ ...control(), generation: 2 });
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(<I18nProvider initialLanguage="en"><SemanticPanel client={client} /></I18nProvider>);
      await flushPromises();
    });
    await act(flushPromises);

    expect(container.textContent).toContain("evidence snapshots changed during the read");
    expect(container.querySelector(".semanticEvidenceDashboard")).toBeNull();

    await act(() => root.unmount());
    container.remove();
  });

  it("fails closed when independently read unfiltered canary metrics disagree", async () => {
    const client = semanticClient();
    vi.mocked(client.getSemanticMetrics).mockResolvedValue({
      ...metrics(),
      machine: { ...metrics().machine, denied: metrics().machine.denied + 1 }
    });
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(<I18nProvider initialLanguage="en"><SemanticPanel client={client} /></I18nProvider>);
      await flushPromises();
    });
    await act(flushPromises);

    expect(container.textContent).toContain("evidence snapshots changed during the read");
    expect(container.querySelector(".semanticEvidenceDashboard")).toBeNull();

    await act(() => root.unmount());
    container.remove();
  });

  it("fails closed when independently read Flow status snapshots disagree", async () => {
    const client = semanticClient();
    vi.mocked(client.getSemanticFlowStatus).mockResolvedValue({
      ...flowStatus(),
      counts: { ...flowStatus().counts, edges: flowStatus().counts.edges + 1 }
    });
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(<I18nProvider initialLanguage="en"><SemanticPanel client={client} /></I18nProvider>);
      await flushPromises();
    });
    await act(flushPromises);

    expect(container.textContent).toContain("evidence snapshots changed during the read");
    expect(container.querySelector(".semanticEvidenceDashboard")).toBeNull();

    await act(() => root.unmount());
    container.remove();
  });

  it("marks bounded evidence windows when a ledger has more records", async () => {
    const client = semanticClient();
    vi.mocked(client.listSemanticFlowEntities).mockResolvedValue({
      schema_version: 1,
      items: [flowEntity()],
      next_cursor: "entity_cursor"
    });
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => {
      root.render(<I18nProvider initialLanguage="en"><SemanticPanel client={client} /></I18nProvider>);
      await flushPromises();
    });
    await act(flushPromises);

    expect(container.textContent).toContain("Showing the first 50 records");

    await act(() => root.unmount());
    container.remove();
  });
});

function semanticClient(): SemanticPanelClient {
  return {
    getSemanticStatus: vi.fn(async () => status()),
    listSemanticAssessments: vi.fn(async () => ({ schema_version: 1 as const, items: [summary()], next_cursor: null })),
    getSemanticAssessment: vi.fn(async () => detail()),
    getSemanticFlowStatus: vi.fn(async () => flowStatus()),
    listSemanticFlowEntities: vi.fn(async () => ({ schema_version: 1 as const, items: [flowEntity()], next_cursor: null })),
    getSemanticFlowLineage: vi.fn(async () => lineage()),
    listSemanticSettlements: vi.fn(async () => ({ schema_version: 1 as const, items: [settlement()], next_cursor: null })),
    listSemanticPolicyEpochs: vi.fn(async () => ({ schema_version: 1 as const, items: [epoch()], next_cursor: null })),
    getSemanticControl: vi.fn(async () => control()),
    listSemanticControlHistory: vi.fn(async () => ({ schema_version: 1 as const, items: [control()], next_cursor: null })),
    listSemanticHealthEvents: vi.fn(async () => ({ schema_version: 1 as const, items: [healthEvent()], next_cursor: null })),
    getSemanticMetrics: vi.fn(async () => metrics())
  };
}

function status(): SemanticStatus {
  return {
    schema_version: 3,
    mode: "shadow",
    adapter: "external",
    profile_id: "semantic-classifier",
    queue: { queued: 1, leased: 2, succeeded: 3, failed: 4, cancelled: 5, capture_failures: 6 },
    assessments: {
      total: 6,
      success: 3,
      error: 3,
      ood: 1,
      would_issue_exact_once: 1,
      would_deny: 2,
      require_human: 3,
      by_status: {
        success: 3,
        skipped_policy: 0,
        egress_blocked: 0,
        timeout: 1,
        provider_error: 0,
        provider_outcome_unknown: 0,
        invalid_schema: 0,
        ood: 1,
        abstained: 0,
        stale_input: 1
      },
      by_domain: {
        filesystem: 2,
        shell: 1,
        git: 1,
        jsonrpc: 0,
        mcp: 0,
        runtime: 1,
        unknown: 1
      }
    },
    control: {
      catalog_version: null,
      active_epoch_id: null,
      active_epoch_sha256: null,
      generation: 1,
      state: "inactive",
      trip_reason_code: null
    },
    flow: flowStatus(),
    machine: {
      eligible: 3,
      issued: 1,
      consumed: 1,
      succeeded: 1,
      failed: 0,
      unknown: 0,
      expired: 0,
      revoked: 0,
      race_lost: 1,
      denied: 1
    },
    actual_auto_approval: { numerator: 1, denominator: 3, rate: 1 / 3 },
    review_metrics: {
      reviewed: 1,
      safe: 1,
      unsafe: 0,
      unsafe_rate: 0,
      issued_reviewed: 1,
      issued_review_rate: 1
    }
  };
}

function summary(): SemanticAssessmentSummary {
  return {
    assessment_id: "assessment_1",
    job_id: "job_1",
    kind: "approval",
    status: "success",
    domain: "filesystem",
    action_id: "filesystem.read",
    pid: "pid_1",
    request_id: "request_1",
    operation_id: "operation_1",
    effect_id: "effect_1",
    shadow_outcome: "require_human",
    reason_codes: ["missing_authoritative_predicate"],
    ood: true,
    abstain: false,
    confidence_bps: 8500,
    calibration_bucket: "high",
    input_tokens: 120,
    output_tokens: 20,
    cost_microunits: 45,
    classifier_id: "classifier",
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

function detail(): SemanticAssessmentDetail {
  return {
    ...summary(),
    findings: [{
      code: "missing_authoritative_predicate",
      severity: "medium",
      confidence_bps: 8500,
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
      confidence_bps: 9000,
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
  };
}

function flowStatus(): SemanticFlowStatus {
  return {
    schema_version: 1,
    available: true,
    counts: { entities: 1, activities: 1, edges: 1, label_assertions: 1 },
    coverage: { complete: 1, partial: 0, unknown: 0, conflict: 0, stale: 0 },
    capture_failures: 0,
    legacy_history: {
      present: true,
      source_schema_version: 5,
      assessment_count: 4,
      coverage: "unknown",
      evidence_sha256: "d".repeat(64),
      created_at: "2030-01-01T00:00:00Z"
    }
  };
}

function flowEntity(): SemanticFlowEntity {
  return {
    schema_version: 1,
    entity_id: "entity_1",
    kind: "provider_result",
    pid: "pid_1",
    tenant_bucket_sha256: "1".repeat(64),
    content_sha256: "2".repeat(64),
    version_sha256: "3".repeat(64),
    provenance_sha256: "4".repeat(64),
    baseline_labels: { sensitivity: "normal", trust_level: "trusted", integrity: "verified" },
    coverage: "complete",
    identity_present: true,
    identity_mixed: false,
    created_at: "2030-01-01T00:00:00Z"
  };
}

function lineage(): SemanticFlowLineage {
  return {
    schema_version: 1,
    root_node_id: "entity_1",
    direction: "upstream",
    items: [{
      depth: 1,
      edge: {
        schema_version: 1,
        edge_id: "edge_1",
        relation: "direct",
        source_node_id: "activity_1",
        source_node_type: "activity",
        target_node_id: "entity_1",
        target_node_type: "entity",
        pid: "pid_1",
        provenance_sha256: "5".repeat(64),
        created_at: "2030-01-01T00:00:00Z"
      },
      node_type: "activity",
      node: {
        schema_version: 1,
        activity_id: "activity_1",
        kind: "provider_call",
        pid: "pid_1",
        action_id: "filesystem.read",
        effect_id: "effect_1",
        state_sha256: "6".repeat(64),
        provider_spec_sha256: "7".repeat(64),
        tool_schema_sha256: "8".repeat(64),
        model_artifact_sha256: null,
        tenant_bucket_sha256: "1".repeat(64),
        created_at: "2030-01-01T00:00:00Z"
      }
    }],
    effective_labels: { sensitivity: "normal", trust_level: "trusted", integrity: "verified" },
    coverage: "complete",
    next_cursor: null,
    truncated: false
  };
}

function settlement(): SemanticMachineSettlement {
  return {
    schema_version: 1,
    settlement_id: "settlement_1",
    assessment_id: "assessment_1",
    job_id: "job_1",
    request_id: "request_1",
    request_revision: 0,
    pid: "pid_1",
    operation_id: "operation_1",
    effect_id: "effect_1",
    epoch_id: "epoch_1",
    policy_sha256: "9".repeat(64),
    tenant_bucket_sha256: "1".repeat(64),
    action_id: "filesystem.read",
    outcome: "issued",
    capability_id: "cap_1",
    binding_sha256: "a".repeat(64),
    decision_sha256: "b".repeat(64),
    matched_rule_id: "rule_1",
    reason_codes: ["policy_match"],
    created_at: "2030-01-01T00:00:01Z",
    human_outcome: "approved",
    human_outcome_source: "human",
    human_outcome_request_revision: 1,
    human_outcome_decision_sha256: "c".repeat(64),
    human_outcome_created_at: "2030-01-01T00:00:02Z"
  };
}

function epoch(): SemanticPolicyEpochSummary {
  return {
    schema_version: 1,
    epoch_id: "epoch_1",
    generation: 1,
    catalog_version: 1,
    policy_sha256: "9".repeat(64),
    expected_previous_sha256: null,
    created_at: "2030-01-01T00:00:00Z"
  };
}

function control(): SemanticControlState {
  return {
    schema_version: 1,
    revision: 1,
    generation: 1,
    mode: "shadow",
    active_epoch_id: null,
    active_policy_sha256: null,
    tripped: false,
    trip_code: null,
    updated_at: "2030-01-01T00:00:00Z"
  };
}

function healthEvent(): SemanticHealthEvent {
  return {
    schema_version: 1,
    event_id: "health_1",
    event_kind: "semantic_policy_activated",
    severity: "info",
    epoch_id: "epoch_1",
    tenant_bucket_sha256: "1".repeat(64),
    evidence_sha256: "c".repeat(64),
    created_at: "2030-01-01T00:00:00Z"
  };
}

function metrics(): SemanticMetrics {
  return {
    schema_version: 1,
    window: null,
    action_id: null,
    tenant_bucket_sha256: null,
    epoch_id: null,
    risk: null,
    machine: status().machine,
    actual_auto_approval: status().actual_auto_approval,
    review_metrics: {
      reviewed: 1,
      safe: 1,
      unsafe: 0,
      unsafe_rate: 0,
      issued_reviewed: 1,
      issued_review_rate: 1
    }
  };
}

async function flushPromises() {
  await Promise.resolve();
  await Promise.resolve();
}
