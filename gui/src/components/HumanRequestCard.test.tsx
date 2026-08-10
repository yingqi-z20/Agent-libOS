import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { HumanRequest } from "../api/types";
import { canonicalApprovalPreviewSha256 } from "../api/types";
import { I18nProvider } from "../i18n";
import {
  buildHumanResponse,
  humanDecisionReducer,
  HumanRequestCard,
  parseCanonicalApprovalPreview,
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

  it("binds external-operation decisions to the displayed request revision and preview", () => {
    const request = humanRequest("external_operation_approval");

    expect(buildHumanResponse(request, true, { answer: "ignored", policy: "always_allow" })).toEqual({
      response: {
        kind: "external_approval",
        approved: true,
        expected_revision: 4,
        preview_sha256: request.preview_sha256
      }
    });
    expect(buildHumanResponse(request, false, { answer: "ignored", policy: "always_deny" })).toEqual({
      response: {
        kind: "external_approval",
        approved: false,
        expected_revision: 4,
        preview_sha256: request.preview_sha256
      }
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

  it("renders only the Host canonical preview for external-operation approvals", () => {
    const request = humanRequest("external_operation_approval");
    request.payload.context = {
      operation: "shell.run",
      argv: ["LEAKED_FREE_FORM_ARGV"],
      cwd: "LEAKED_FREE_FORM_CONTEXT"
    };
    request.payload.question = "LEAKED_FREE_FORM_QUESTION";
    request.payload.reason = "LEAKED_FREE_FORM_REASON";

    const html = render(request);

    expect(html).toContain("Host-verified operation preview");
    expect(html).toContain("filesystem.read");
    expect(html).toContain("filesystem:workspace:reports/report.txt");
    expect(html).toContain("224edab2326c1b9269476e73975e79ad27ffce2e07aea3938a98d465504e2fee");
    expect(html).toContain("Arguments SHA-256");
    expect(html).toContain(request.preview_sha256 as string);
    expect(html).not.toContain("LEAKED_FREE_FORM_ARGV");
    expect(html).not.toContain("LEAKED_FREE_FORM_CONTEXT");
    expect(html).not.toContain("LEAKED_FREE_FORM_QUESTION");
    expect(html).not.toContain("LEAKED_FREE_FORM_REASON");
  });

  it("renders redacted digest-bound identities without echoing raw external authority payload", () => {
    const sentinel = "SEMANTIC_GUI_RESOURCE_SECRET_SENTINEL_54c2";
    const identitySha256 = "3a8447bd4e123a918c94fcbb63e2eea2c987483eb1f6a32d5e3a2f62b0f0e89a";
    const resourceSha256 = "5ecb1c298823f555c2e4df6f407d4657eaf27b38666a0901d037441a339c259f";
    const request = humanRequest("external_operation_approval");
    request.payload = {
      type: "external_operation_approval",
      question: `Approve ${sentinel}?`,
      context: {
        operation: "jsonrpc.call",
        resource: `jsonrpc:${sentinel}:${sentinel}`,
        endpoint_id: sentinel,
        method_id: sentinel
      },
      effect_binding: { effect_id: sentinel, canonical_args_hash: sentinel },
      requested_once_capability: {
        subject: sentinel,
        resource: `jsonrpc:${sentinel}:${sentinel}`,
        rights: ["execute"]
      }
    };
    request.approval_preview = {
      ...request.approval_preview!,
      action_id: "jsonrpc.call",
      resource_display: "<redacted>",
      resource_sha256: resourceSha256,
      rights: ["execute"],
      argument_projection: {
        ...request.approval_preview!.argument_projection,
        kind: "jsonrpc",
        operation: "call",
        endpoint_id: "<redacted>",
        endpoint_id_sha256: identitySha256,
        method_id: "<redacted>",
        method_id_sha256: identitySha256,
        registry_spec_sha256: "d".repeat(64),
        registry_generation: 7,
        payload_sha256: "c".repeat(64),
        path_sha256: null,
        read_max_bytes: null,
        text_encoding: null
      }
    };
    request.preview_sha256 = canonicalApprovalPreviewSha256(request.approval_preview);

    expect(parseCanonicalApprovalPreview(request)).not.toBeNull();
    const html = render(request);

    expect(html).toContain("&lt;redacted&gt;");
    expect(html).toContain(resourceSha256);
    expect(html).toContain(identitySha256);
    expect(html).not.toContain(sentinel);
    expect(html).not.toContain("effect_binding");
    expect(html).not.toContain("requested_once_capability");
  });

  it("renders role-labelled Git refs while withholding a secret ref from the DOM", () => {
    const sentinel = "refs/heads/ghp_SEMANTIC_GUI_REF_SECRET_abcdefghijkl";
    const request = gitPreview("status");
    request.payload = {
      type: "external_operation_approval",
      question: sentinel,
      context: { local_ref: "refs/heads/main", remote_ref: sentinel }
    };
    request.approval_preview = {
      ...request.approval_preview!,
      argument_projection: {
        ...request.approval_preview!.argument_projection,
        operation: "push",
        git_references: [
          {
            role: "local_ref",
            display: "refs/heads/main",
            sha256: "f921bd05e68b03740c450e565e0e6173e546193170b2dd404ddb6f153e9b5bf3"
          },
          { role: "remote_ref", display: "<redacted>", sha256: "f".repeat(64) }
        ]
      }
    };
    request.preview_sha256 = canonicalApprovalPreviewSha256(request.approval_preview);

    expect(parseCanonicalApprovalPreview(request)).not.toBeNull();
    const html = render(request);
    expect(html).toContain("local_ref: refs/heads/main");
    expect(html).toContain("remote_ref: &lt;redacted&gt;");
    expect(html).not.toContain(sentinel);
  });

  it("renders concrete filesystem and Git operations that would otherwise share one action", () => {
    const readText = humanRequest("external_operation_approval");
    readText.approval_preview = {
      ...readText.approval_preview!,
      argument_projection: {
        ...readText.approval_preview!.argument_projection,
        operation: "read_text",
        text_encoding: "utf-8"
      }
    };
    readText.preview_sha256 = canonicalApprovalPreviewSha256(readText.approval_preview);
    const readBytes = humanRequest("external_operation_approval");
    readBytes.approval_preview = {
      ...readBytes.approval_preview!,
      argument_projection: {
        ...readBytes.approval_preview!.argument_projection,
        operation: "read_bytes",
        text_encoding: null
      }
    };
    readBytes.preview_sha256 = canonicalApprovalPreviewSha256(readBytes.approval_preview);

    expect(render(readText)).toContain("read_text");
    expect(render(readText)).toContain("utf-8");
    expect(render(readBytes)).toContain("read_bytes");

    const gitStatus = gitPreview("status");
    const gitRemotes = gitPreview("list_remotes");
    expect(render(gitStatus)).toContain("status");
    expect(render(gitStatus)).toContain("Source argument SHA-256");
    expect(render(gitStatus)).toContain("e".repeat(64));
    expect(render(gitRemotes)).toContain("list_remotes");

    const gitPush = gitPreview("status");
    gitPush.approval_preview = {
      ...gitPush.approval_preview!,
      action_id: "git.write",
      argument_projection: {
        ...gitPush.approval_preview!.argument_projection,
        operation: "push",
        git_references: [
          {
            role: "local_ref",
            display: "refs/heads/main",
            sha256: "f921bd05e68b03740c450e565e0e6173e546193170b2dd404ddb6f153e9b5bf3"
          },
          {
            role: "remote_ref",
            display: "refs/heads/release",
            sha256: "57cbf0f45532ac4e52113610ebeca693963b75a7f1db6725dbf82d9fe8f2bb00"
          }
        ],
        git_fact_tokens: ["delete=false", "force_with_lease=false"]
      }
    };
    gitPush.preview_sha256 = canonicalApprovalPreviewSha256(gitPush.approval_preview);
    const pushHtml = render(gitPush);
    expect(pushHtml).toContain("local_ref: refs/heads/main");
    expect(pushHtml).toContain("remote_ref: refs/heads/release");
  });

  it("distinguishes fixed shell subcommands while withholding values and secrets", () => {
    const sentinel = "SEMANTIC_GUI_ARG_SECRET_SENTINEL_7f33";
    for (const subcommand of ["status", "push", "reset", "clean"]) {
      const request = shellPreview(subcommand, sentinel);
      const html = render(request);
      expect(html).toContain(`&quot;${subcommand}&quot;`);
      expect(html).toContain("&lt;redacted&gt;");
      expect(html).not.toContain(sentinel);
    }
  });

  it("fails closed on missing, stale, private, or malformed canonical previews", () => {
    const missing = humanRequest("external_operation_approval");
    delete missing.approval_preview;
    delete missing.preview_sha256;
    missing.payload.question = "INVALID_PREVIEW_QUESTION_MUST_NOT_RENDER";
    expect(parseCanonicalApprovalPreview(missing)).toBeNull();
    expect(buildHumanResponse(missing, true, { answer: "", policy: "ask_each_time" })).toEqual({
      response: { kind: "approval", approved: true }
    });
    expect(render(missing)).toContain("canonical operation preview is missing or malformed");
    expect(render(missing)).not.toContain("INVALID_PREVIEW_QUESTION_MUST_NOT_RENDER");
    expect(render(missing)).not.toMatch(/<button[^>]*disabled=""/);

    const stale = humanRequest("external_operation_approval");
    stale.approval_preview = { ...stale.approval_preview!, revision: stale.revision - 1 };
    expect(parseCanonicalApprovalPreview(stale)).toBeNull();

    const privateField = humanRequest("external_operation_approval");
    privateField.approval_preview = {
      ...privateField.approval_preview!,
      raw_command: "MUST_NOT_RENDER"
    } as typeof privateField.approval_preview;
    expect(parseCanonicalApprovalPreview(privateField)).toBeNull();
    expect(render(privateField)).not.toContain("MUST_NOT_RENDER");

    const upperDigest = humanRequest("external_operation_approval");
    upperDigest.preview_sha256 = "B".repeat(64);
    expect(parseCanonicalApprovalPreview(upperDigest)).toBeNull();

    const timezoneLess = humanRequest("external_operation_approval");
    timezoneLess.approval_preview = { ...timezoneLess.approval_preview!, expires_at: "2030-07-10T00:01:00" };
    timezoneLess.preview_sha256 = canonicalApprovalPreviewSha256(timezoneLess.approval_preview);
    expect(parseCanonicalApprovalPreview(timezoneLess)).toBeNull();

    const impossibleIdentity = humanRequest("external_operation_approval");
    impossibleIdentity.approval_preview = {
      ...impossibleIdentity.approval_preview!,
      source_labels: {
        ...impossibleIdentity.approval_preview!.source_labels,
        identity_present: false,
        identity_mixed: true
      }
    };
    impossibleIdentity.preview_sha256 = canonicalApprovalPreviewSha256(impossibleIdentity.approval_preview);
    expect(parseCanonicalApprovalPreview(impossibleIdentity)).toBeNull();

    for (const pollution of [
      { argv_truncated: true },
      { payload_sha256: "c".repeat(64) },
      { endpoint_id: "forged-endpoint" },
      { worktree_id: "forged-worktree" }
    ]) {
      const polluted = humanRequest("external_operation_approval");
      polluted.approval_preview = {
        ...polluted.approval_preview!,
        argument_projection: {
          ...polluted.approval_preview!.argument_projection,
          ...pollution
        }
      };
      polluted.preview_sha256 = canonicalApprovalPreviewSha256(polluted.approval_preview);
      expect(parseCanonicalApprovalPreview(polluted)).toBeNull();
    }

    const validReferences = [
      {
        role: "local_ref" as const,
        display: "refs/heads/main",
        sha256: "f921bd05e68b03740c450e565e0e6173e546193170b2dd404ddb6f153e9b5bf3"
      },
      {
        role: "remote_ref" as const,
        display: "refs/heads/release",
        sha256: "57cbf0f45532ac4e52113610ebeca693963b75a7f1db6725dbf82d9fe8f2bb00"
      }
    ];
    for (const references of [
      [{ ...validReferences[0], role: "unknown_ref" as never }],
      [{ ...validReferences[0], sha256: "0".repeat(64) }],
      [validReferences[0], { ...validReferences[1], role: "local_ref" as const }],
      [...validReferences].reverse()
    ]) {
      const malformedReference = gitPreview("status");
      malformedReference.approval_preview = {
        ...malformedReference.approval_preview!,
        argument_projection: {
          ...malformedReference.approval_preview!.argument_projection,
          git_references: references
        }
      };
      malformedReference.preview_sha256 = canonicalApprovalPreviewSha256(malformedReference.approval_preview);
      expect(parseCanonicalApprovalPreview(malformedReference)).toBeNull();
    }

    for (const injection of ["\n", "\u001b", "\u202e", "\u2066", "\u2028", "\u2029", "\ud800"]) {
      const injected = humanRequest("external_operation_approval");
      injected.approval_preview = {
        ...injected.approval_preview!,
        resource_display: `filesystem:workspace:report${injection}forged`
      };
      injected.preview_sha256 = canonicalApprovalPreviewSha256(injected.approval_preview);
      expect(parseCanonicalApprovalPreview(injected)).toBeNull();
    }

    const digestMismatch = humanRequest("external_operation_approval");
    digestMismatch.approval_preview = {
      ...digestMismatch.approval_preview!,
      resource_display: "filesystem:workspace:reports/attacker-substitution.txt"
    };
    expect(parseCanonicalApprovalPreview(digestMismatch)).toBeNull();
    expect(buildHumanResponse(digestMismatch, true, { answer: "", policy: "ask_each_time" })).toEqual({
      response: { kind: "approval", approved: true }
    });
    expect(render(digestMismatch)).not.toContain("attacker-substitution.txt");
  });

  it("matches Host canonical JSON digests for ASCII and non-BMP previews", () => {
    const request = humanRequest("external_operation_approval");
    expect(canonicalApprovalPreviewSha256(request.approval_preview!)).toBe(
      "c8d88c866a736f87c3204f1023dd459ed73ae84a5166fed3a775bceb1595b0ff"
    );
    expect(canonicalApprovalPreviewSha256({
      ...request.approval_preview!,
      resource_display: "filesystem:workspace:报告/😀.txt"
    })).toBe("a0d8d9a8141311ef9d5495beeff6401e5a34c84cd52dba2d604ad509d0f25563");
  });
});

function render(request: HumanRequest): string {
  return renderToStaticMarkup(
    <I18nProvider initialLanguage="en">
      <HumanRequestCard request={request} onRespond={async () => true} />
    </I18nProvider>
  );
}

function gitPreview(operation: "status" | "list_remotes"): HumanRequest {
  const request = humanRequest("external_operation_approval");
  request.approval_preview = {
    ...request.approval_preview!,
    action_id: "git.read",
    resource_display: "git:workspace",
    resource_sha256: "7c76e1729d96a138f6c56ba97a475e6e1f470fa05dd2bf1903289f2ec81dc25a",
    argument_projection: {
      ...request.approval_preview!.argument_projection,
      kind: "git",
      operation,
      path_sha256: null,
      read_max_bytes: null,
      text_encoding: null,
      worktree_id: "main",
      worktree_id_sha256: "0d6e4079e36703ebd37c00722f5891d28b0e2811dc114b129215123adcce3605",
      source_args_sha256: "e".repeat(64)
    }
  };
  request.preview_sha256 = canonicalApprovalPreviewSha256(request.approval_preview);
  return request;
}

function shellPreview(subcommand: string, sentinel: string): HumanRequest {
  const request = humanRequest("external_operation_approval");
  request.approval_preview = {
    ...request.approval_preview!,
    action_id: "shell.run",
    resource_display: "shell:git",
    resource_sha256: "e9272127e7e555606ca2941f53c3f6221893128229cbc415cbbaed2cfffe49b2",
    rights: ["execute"],
    risk: "high",
    argument_projection: {
      ...request.approval_preview!.argument_projection,
      kind: "shell",
      operation: "run",
      display_argv: ["git", subcommand, "<redacted>"],
      argv_count: 3,
      argv_sha256: "c".repeat(64),
      safe_cwd: null,
      cwd_sha256: "d".repeat(64),
      path_sha256: null,
      read_max_bytes: null,
      text_encoding: null,
      timeout_seconds: "60",
      continuous_session: false,
      network_access: true
    }
  };
  request.payload.context = { argv: ["git", subcommand, sentinel] };
  request.preview_sha256 = canonicalApprovalPreviewSha256(request.approval_preview);
  return request;
}

function humanRequest(type: string, payload: Record<string, unknown> = {}): HumanRequest {
  const request: HumanRequest = {
    request_id: `request_${type}`,
    pid: "pid_1",
    human: "owner",
    payload: { type, question: `Handle ${type}?`, ...payload },
    status: "pending",
    decision: null,
    blocking: true,
    revision: 4,
    created_at: "2026-07-10T00:00:00Z",
    updated_at: "2026-07-10T00:00:00Z"
  };
  if (type === "external_operation_approval") {
    request.approval_preview = {
      schema_version: 1,
      request_id: request.request_id,
      revision: request.revision,
      pid: request.pid,
      action_id: "filesystem.read",
      resource_display: "filesystem:workspace:reports/report.txt",
      resource_sha256: "224edab2326c1b9269476e73975e79ad27ffce2e07aea3938a98d465504e2fee",
      rights: ["read"],
      effect_id: "effect_1",
      canonical_args_sha256: "a".repeat(64),
      argument_projection: {
        kind: "filesystem",
        operation: "read",
        display_argv: [],
        argv_count: null,
        argv_truncated: false,
        argv_sha256: null,
        safe_cwd: null,
        cwd_sha256: null,
        endpoint_id: null,
        endpoint_id_sha256: null,
        method_id: null,
        method_id_sha256: null,
        server_id: null,
        server_id_sha256: null,
        tool_id: null,
        tool_id_sha256: null,
        registry_spec_sha256: null,
        registry_generation: null,
        payload_sha256: null,
        path_sha256: "b".repeat(64),
        content_sha256: null,
        content_bytes: null,
        read_max_bytes: 65536,
        entry_limit: null,
        text_encoding: "utf-8",
        expected_content_sha256: null,
        overwrite: null,
        parents: null,
        exist_ok: null,
        recursive: null,
        missing_ok: null,
        timeout_seconds: null,
        continuous_session: null,
        network_access: null,
        worktree_id: null,
        worktree_id_sha256: null,
        repository_state_sha256: null,
        source_args_sha256: null,
        git_references: [],
        git_fact_tokens: []
      },
      target_state_sha256: null,
      risk: "low",
      source_labels: {
        sensitivity: "normal",
        integrity: "verified",
        trust_level: "trusted",
        identity_present: true,
        identity_mixed: false
      },
      expires_at: "2030-07-10T00:01:00Z"
    };
    request.preview_sha256 = canonicalApprovalPreviewSha256(request.approval_preview);
  }
  return request;
}
