import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { HumanRequest } from "../api/types";
import { I18nProvider } from "../i18n";
import {
  buildHumanResponse,
  humanDecisionReducer,
  HumanRequestCard,
  parseDataReleaseApprovalContext,
  type HumanDecisionState
} from "./HumanRequestCard";

describe("HumanRequestCard", () => {
  it("builds directionally valid permission decisions", () => {
    const request = humanRequest("permission_request");

    expect(buildHumanResponse(request, true, { answer: "", policy: "always_allow" })).toEqual({
      response: { kind: "permission", approved: true, decision: { policy: "always_allow" } }
    });
    expect(buildHumanResponse(request, false, { answer: "", policy: "always_deny" })).toEqual({
      response: { kind: "permission", approved: false, decision: { policy: "always_deny" } }
    });
    expect(buildHumanResponse(request, true, { answer: "", policy: "ask_each_time" })).toEqual({
      response: { kind: "permission", approved: true, decision: { policy: "ask_each_time" } }
    });
    expect(buildHumanResponse(request, false, { answer: "", policy: "ask_each_time" })).toEqual({
      response: { kind: "permission", approved: false, decision: { policy: "ask_each_time" } }
    });
    expect(buildHumanResponse(request, false, { answer: "", policy: "always_allow" })).toEqual({
      error: "permission_reject_allow"
    });
    expect(buildHumanResponse(request, true, { answer: "", policy: "always_deny" })).toEqual({
      error: "permission_approve_deny"
    });
  });

  it("requires a non-empty string for approved questions and omits it on rejection", () => {
    const request = humanRequest("question");

    expect(buildHumanResponse(request, true, { answer: "   ", policy: "ask_each_time" })).toEqual({
      error: "question_answer_required"
    });
    expect(buildHumanResponse(request, true, { answer: " eu-west ", policy: "ask_each_time" })).toEqual({
      response: { kind: "question", approved: true, answer: "eu-west" }
    });
    expect(buildHumanResponse(request, false, { answer: "draft remains local", policy: "ask_each_time" })).toEqual({
      response: { kind: "question", approved: false }
    });
  });

  it("keeps ordinary approvals boolean-only", () => {
    const request = humanRequest("external_operation_approval");

    expect(buildHumanResponse(request, true, { answer: "ignored", policy: "always_allow" })).toEqual({
      response: { kind: "approval", approved: true }
    });
    expect(buildHumanResponse(request, false, { answer: "ignored", policy: "always_deny" })).toEqual({
      response: { kind: "approval", approved: false }
    });
  });

  it("blocks a withheld parent until its data release request is handled", () => {
    const request = humanRequest("question", {
      question: "ORIGINAL_PARENT_SECRET_MUST_NOT_RENDER",
      release_required: true,
      release_request_id: "request_release_1"
    });

    expect(buildHumanResponse(request, true, { answer: "must not submit", policy: "ask_each_time" })).toEqual({
      error: "release_required"
    });

    const html = render(request);
    expect(html).toContain("Data release required");
    expect(html).toContain("request_release_1");
    expect(html).not.toContain("ORIGINAL_PARENT_SECRET_MUST_NOT_RENDER");
    expect(html).not.toContain('name="human-answer"');
    expect(html).not.toContain("<button");
  });

  it("renders only structured metadata for a data release approval", () => {
    const payloadSha256 = "a".repeat(64);
    const request = humanRequest("data_release_approval", {
      question: "Release this labeled payload?",
      original_payload: "ORIGINAL_RELEASE_PAYLOAD_MUST_NOT_RENDER",
      context: {
        sink: "human:owner:gui",
        sensitivity: "secret",
        tenant: "tenant-alpha",
        principal: "principal-beta",
        payload_bytes: 87,
        payload_sha256: payloadSha256,
        source_count: 3,
        operation: "human.gui.present"
      }
    });

    expect(parseDataReleaseApprovalContext(request.payload)).toEqual({
      sink: "human:owner:gui",
      sensitivity: "secret",
      tenant: "tenant-alpha",
      principal: "principal-beta",
      payload_bytes: 87,
      payload_sha256: payloadSha256,
      source_count: 3,
      operation: "human.gui.present"
    });

    const html = render(request);
    expect(html).toContain("Data release approval");
    expect(html).toContain("Destination sink");
    expect(html).toContain("human:owner:gui");
    expect(html).toContain("Sensitivity");
    expect(html).toContain("secret");
    expect(html).toContain("tenant-alpha");
    expect(html).toContain("principal-beta");
    expect(html).toContain("Payload bytes");
    expect(html).toContain("87");
    expect(html).toContain(payloadSha256);
    expect(html).toContain("Source count");
    expect(html).toContain("3");
    expect(html).toContain("human.gui.present");
    expect(html).toContain('aria-label="Approve: human.gui.present · human:owner:gui · request_data_release_approval"');
    expect(html).not.toContain("ORIGINAL_RELEASE_PAYLOAD_MUST_NOT_RENDER");
    expect(html.match(/<button/g)).toHaveLength(2);
  });

  it("disables approval when release metadata fails strict validation", () => {
    const request = humanRequest("data_release_approval", {
      context: {
        sink: "human:owner:gui",
        sensitivity: "secret",
        payload_bytes: 87,
        payload_sha256: "not-a-sha256",
        source_count: 3,
        operation: "human.gui.present"
      }
    });

    expect(parseDataReleaseApprovalContext(request.payload)).toBeNull();
    const html = render(request);
    expect(html).toContain("Release metadata is incomplete");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Approve<\/button>/);
    expect(html).toMatch(/<button[^>]*class="danger"[^>]*>Reject<\/button>/);
  });

  it("keeps the draft when submission fails instead of optimistically clearing it", () => {
    const state: HumanDecisionState = {
      answer: "carefully chosen answer",
      policy: "always_allow",
      submitting: true,
      errorKey: null
    };

    expect(humanDecisionReducer(state, { type: "submission_finished", accepted: false })).toEqual({
      answer: "carefully chosen answer",
      policy: "always_allow",
      submitting: false,
      errorKey: "human.submitFailed"
    });
  });

  it("renders request-type-specific controls", () => {
    const permissionHtml = render(humanRequest("permission_request"));
    const questionHtml = render(humanRequest("question"));
    const approvalHtml = render(humanRequest("external_operation_approval"));

    expect(permissionHtml).toContain('value="always_allow"');
    expect(permissionHtml).toContain('value="ask_each_time"');
    expect(permissionHtml).toContain('value="always_deny"');
    expect(permissionHtml).not.toContain('name="human-answer"');
    expect(questionHtml).toContain('name="human-answer"');
    expect(questionHtml).toContain('required=""');
    expect(approvalHtml).not.toContain('name="human-answer"');
    expect(approvalHtml).not.toContain('name="permission-policy"');
  });

  it("renders the structured scope for permission decisions", () => {
    const request = humanRequest("permission_request", {
      requested_permission: {
        subject: "pid_1",
        resource: "filesystem:workspace/src",
        rights: ["read", "write"]
      },
      context: {
        risk: "high",
        resource_scope: "exact",
        request_origin: "model"
      }
    });

    const html = render(request);

    expect(html).toContain("Approval context");
    expect(html).toContain("filesystem:workspace/src");
    expect(html).toContain("high");
    expect(html).toContain('role="group"');
  });

  it("renders the exact bounded context for external-operation approvals", () => {
    const request = humanRequest("external_operation_approval");
    request.payload.context = {
      operation: "shell.run",
      argv: ["git", "status"],
      cwd: "workspace/sentinel-context"
    };

    const html = render(request);

    expect(html).toContain("Approval context");
    expect(html).toContain("workspace/sentinel-context");
    expect(html).toContain("shell.run");
  });
});

function render(request: HumanRequest): string {
  return renderToStaticMarkup(
    <I18nProvider initialLanguage="en">
      <HumanRequestCard request={request} onRespond={async () => true} />
    </I18nProvider>
  );
}

function humanRequest(type: string, payload: Record<string, unknown> = {}): HumanRequest {
  return {
    request_id: `request_${type}`,
    pid: "pid_1",
    human: "owner",
    payload: { type, question: `Handle ${type}?`, ...payload },
    status: "pending",
    decision: null,
    blocking: true,
    created_at: "2026-07-10T00:00:00Z",
    updated_at: "2026-07-10T00:00:00Z"
  };
}
