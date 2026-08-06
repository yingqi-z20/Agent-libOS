// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import type { SemanticAssessmentDetail, SemanticAssessmentSummary, SemanticStatus } from "../api/types";
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
    expect(container.textContent).toContain("Read-only Shadow evidence for process pid_1");
    expect(container.textContent).toContain("Semantic results never approve, reject, relabel, or release data");
    expect(container.textContent).toContain("Queued");
    expect(container.textContent).toContain("Would approve once");
    expect(container.textContent).toContain("Require human");
    expect(container.textContent).toContain("N/A (Shadow)");
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
});

function semanticClient(): SemanticPanelClient {
  return {
    getSemanticStatus: vi.fn(async () => status()),
    listSemanticAssessments: vi.fn(async () => ({ schema_version: 1 as const, items: [summary()], next_cursor: null })),
    getSemanticAssessment: vi.fn(async () => detail())
  };
}

function status(): SemanticStatus {
  return {
    schema_version: 2,
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
    actual_auto_approval: { numerator: 0, denominator: 0, rate: null }
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

async function flushPromises() {
  await Promise.resolve();
  await Promise.resolve();
}
