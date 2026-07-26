import { describe, expect, it } from "vitest";
import type { ExplainOperationResponse, OperationEvidence, OperationSummary } from "../api/types";
import { buildOperationChildren, filterOperationEvidence, mergeEvidencePage, operationIdForRefresh } from "./ExplainPanel";

describe("ExplainPanel", () => {
  it("builds explicit causal children without using timestamps", () => {
    const root = operation("op_root", null, "llm_request");
    const tool = operation("op_tool", root.operation_id, "tool_call");
    const primitive = operation("op_primitive", tool.operation_id, "primitive");

    const children = buildOperationChildren([primitive, root, tool]);

    expect(children.get(root.operation_id)?.map((item) => item.operation_id)).toEqual([tool.operation_id]);
    expect(children.get(tool.operation_id)?.map((item) => item.operation_id)).toEqual([primitive.operation_id]);
  });

  it("filters the evidence timeline by persisted evidence type", () => {
    const items = [evidence("audit", "audit_1"), evidence("event", "evt_1")];
    expect(filterOperationEvidence(items, "all")).toHaveLength(2);
    expect(filterOperationEvidence(items, "audit").map((item) => item.evidence_id)).toEqual(["audit_1"]);
  });

  it("merges paginated evidence while adopting the next server cursor", () => {
    const current = {
      evidence: [evidence("audit", "audit_1")],
      next_cursor: "cursor_1"
    } as ExplainOperationResponse;
    const next = {
      evidence: [evidence("event", "evt_1")],
      next_cursor: null,
      presentation_truncated: false
    } as ExplainOperationResponse;

    const merged = mergeEvidencePage(current, next);

    expect(merged.evidence.map((item) => item.evidence_id)).toEqual(["audit_1", "evt_1"]);
    expect(merged.next_cursor).toBeNull();
    expect(merged.presentation_truncated).toBe(false);
  });

  it("merges roles when adjacent pages repeat one evidence record", () => {
    const currentItem = { ...evidence("audit", "audit_1"), roles: ["audit"] };
    const nextItem = { ...evidence("audit", "audit_1"), roles: ["decision"] };
    const current = { evidence: [currentItem], next_cursor: "cursor_1" } as ExplainOperationResponse;
    const next = {
      evidence: [nextItem],
      next_cursor: null,
      presentation_truncated: false
    } as ExplainOperationResponse;

    const merged = mergeEvidencePage(current, next);

    expect(merged.evidence).toHaveLength(1);
    expect(merged.evidence[0].roles).toEqual(["audit", "decision"]);
  });

  it("preserves an exact selected operation even when it is not on the refreshed first page", () => {
    const firstPage = [operation("op_new", null, "runtime")];
    expect(operationIdForRefresh("op_selected", firstPage, false)).toBe("op_selected");
    expect(operationIdForRefresh(null, firstPage, true)).toBeNull();
    expect(operationIdForRefresh(null, firstPage, false)).toBe("op_new");
  });
});

function operation(operationId: string, parentOperationId: string | null, kind: OperationSummary["kind"]): OperationSummary {
  return {
    operation_id: operationId,
    root_operation_id: "op_root",
    parent_operation_id: parentOperationId,
    kind,
    name: operationId,
    actor: "pid_1",
    pid: "pid_1",
    state: "terminal",
    outcome: "succeeded",
    started_at: "2026-07-10T00:00:00Z",
    updated_at: "2026-07-10T00:00:00Z",
    completed_at: "2026-07-10T00:00:00Z"
  };
}

function evidence(evidenceType: string, evidenceId: string): OperationEvidence {
  return {
    evidence_type: evidenceType,
    evidence_id: evidenceId,
    roles: [evidenceType],
    occurred_at: null,
    metadata: {},
    data: {}
  };
}
