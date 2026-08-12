// @vitest-environment jsdom

import userEvent from "@testing-library/user-event";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { LibOSClient } from "../api/client";
import type { ConfirmationRequest } from "../adminTypes";
import { I18nProvider } from "../i18n";
import { McpModernPanel } from "./McpModernPanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mounted: Array<{ root: Root; container: HTMLDivElement }> = [];

afterEach(async () => {
  for (const { root, container } of mounted.splice(0)) {
    await act(() => root.unmount());
    container.remove();
  }
});

describe("McpModernPanel", () => {
  it("paginates resources only after explicit POST client calls", async () => {
    const listMcpResources = vi.fn()
      .mockResolvedValueOnce({ items: [{ resource_id: "first", name: "First" }], next_cursor: "next", cache_hint: null })
      .mockResolvedValueOnce({ items: [{ resource_id: "second", name: "Second" }], next_cursor: null, cache_hint: null });
    const { container } = await render({ listMcpResources });
    const user = userEvent.setup();

    await act(async () => user.click(button(container, "List resources")));
    await act(async () => user.click(button(container, "Next resources page")));

    expect(listMcpResources).toHaveBeenNthCalledWith(1, "server/1", null);
    expect(listMcpResources).toHaveBeenNthCalledWith(2, "server/1", "next");
    expect(container.textContent).toContain("resource catalog");
  });

  it("previews an untrusted prompt before a separately confirmed acceptance", async () => {
    const getMcpPrompt = vi.fn().mockResolvedValue({
      kind: "complete",
      preview_sha256: "a".repeat(64),
      value: {
        prompt_id: "review",
        messages: [{
          role: "user",
          content: { kind: "text", text: "provider text" },
          provenance: "untrusted_mcp_prompt"
        }],
        user_confirmation_required: true
      }
    });
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render({ getMcpPrompt }, confirmAction);
    const user = userEvent.setup();
    const promptId = inputAfterLabel(container, "Logical prompt id");

    await act(async () => user.type(promptId, "review"));
    await act(async () => user.click(button(container, "Preview untrusted prompt")));
    expect(getMcpPrompt).toHaveBeenCalledWith("server/1", "review", {}, false);
    expect(container.textContent).toContain("untrusted prompt preview");

    await act(async () => user.click(button(container, "Confirm as user context")));
    expect(confirmAction).toHaveBeenCalledTimes(1);
    expect(confirmAction.mock.calls[0]?.[0].details).toMatchObject({
      server_id: "server/1",
      prompt_id: "review",
      provenance: "untrusted_mcp_prompt",
      preview_sha256: "a".repeat(64)
    });
    await act(async () => confirmAction.mock.calls[0]?.[0].action());
    expect(getMcpPrompt).toHaveBeenLastCalledWith("server/1", "review", {}, true, "a".repeat(64));
  });

  it("sends only typed prompt and resource-template completion references", async () => {
    const completeMcpPrompt = vi.fn().mockResolvedValue({
      kind: "complete",
      value: { values: ["result"], has_more: false }
    });
    const { container } = await render({ completeMcpPrompt });
    const user = userEvent.setup();
    const referenceType = selectAfterLabel(container, "Completion reference type");

    expect(referenceType.value).toBe("prompt");
    expect([...referenceType.options].map((option) => option.value)).toEqual([
      "prompt",
      "resource_template"
    ]);
    await act(async () => user.type(inputAfterLabel(container, "Completion reference id"), "review"));
    await setTextarea(
      textareaAfterLabel(container, "Completion argument (string map)"),
      '{"name":"topic","value":"MCP"}'
    );
    await setTextarea(
      textareaAfterLabel(container, "Completion context (string map)"),
      '{"audience":"host"}'
    );
    await act(async () => user.click(button(container, "Complete")));
    expect(completeMcpPrompt).toHaveBeenLastCalledWith(
      "server/1",
      "prompt",
      "review",
      { name: "topic", value: "MCP" },
      { audience: "host" }
    );

    await act(async () => user.selectOptions(referenceType, "resource_template"));
    await act(async () => user.click(button(container, "Complete")));
    expect(completeMcpPrompt).toHaveBeenLastCalledWith(
      "server/1",
      "resource_template",
      "review",
      { name: "topic", value: "MCP" },
      { audience: "host" }
    );
  });

  it("shows an OAuth authorization URL as inert text and clears the callback", async () => {
    const beginMcpOAuth = vi.fn().mockResolvedValue({
      challenge_id: "challenge-local",
      authorization_url: "https://issuer.example/authorize?state=opaque",
      expires_at: "2030-01-01T00:00:00Z"
    });
    const completeMcpOAuth = vi.fn().mockResolvedValue({
      profile_id: "profile-local",
      status: "authorized",
      scopes: []
    });
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render({ beginMcpOAuth, completeMcpOAuth }, confirmAction);
    const user = userEvent.setup();

    await act(async () => user.type(inputAfterLabel(container, "Host auth profile id"), "profile-local"));
    await act(async () => user.click(button(container, "Login")));
    await act(async () => confirmAction.mock.calls[0]?.[0].action());

    const authorization = [...container.querySelectorAll<HTMLTextAreaElement>("textarea[readonly]")]
      .find((item) => item.value.includes("issuer.example"));
    expect(authorization?.value).toContain("state=opaque");
    expect(container.querySelector('a[href*="issuer.example"]')).toBeNull();

    const callback = inputAfterLabel(container, "Full callback URL");
    await act(async () => user.type(callback, "http://127.0.0.1/callback?code=one&state=two"));
    await act(async () => user.click(button(container, "Submit callback")));
    const completion = confirmAction.mock.calls[1]?.[0];
    expect(JSON.stringify(completion?.details)).not.toContain("code=one");
    expect(callback.value).toBe("");
    await act(async () => completion?.action());
    expect(completeMcpOAuth).toHaveBeenCalledWith(
      "challenge-local",
      "http://127.0.0.1/callback?code=one&state=two"
    );
    expect(callback.value).toBe("");
  });

  it("consumes an OAuth callback once and clears code and state after a failed completion", async () => {
    const callbackValue = "http://127.0.0.1/callback?code=PRIVATE-CODE&state=PRIVATE-STATE";
    const beginMcpOAuth = vi.fn().mockResolvedValue({
      challenge_id: "challenge-local",
      authorization_url: "https://issuer.example/authorize?state=CHALLENGE-STATE",
      expires_at: "2030-01-01T00:00:00Z"
    });
    const completeMcpOAuth = vi.fn().mockRejectedValue(
      new Error(`network rejected ${callbackValue}`)
    );
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render({ beginMcpOAuth, completeMcpOAuth }, confirmAction);
    const user = userEvent.setup();

    await act(async () => user.type(inputAfterLabel(container, "Host auth profile id"), "profile-local"));
    await act(async () => user.click(button(container, "Login")));
    await act(async () => confirmAction.mock.calls[0]?.[0].action());

    const callback = inputAfterLabel(container, "Full callback URL");
    await act(async () => user.type(callback, callbackValue));
    await act(async () => user.click(button(container, "Submit callback")));
    const completion = confirmAction.mock.calls[1]?.[0];

    expect(callback.value).toBe("");
    expect(JSON.stringify(completion?.details)).not.toContain("PRIVATE-CODE");
    expect(JSON.stringify(completion?.details)).not.toContain("PRIVATE-STATE");
    expect(mcpFormValues(container)).not.toContain("PRIVATE-CODE");
    expect(mcpFormValues(container)).not.toContain("PRIVATE-STATE");
    expect(mcpFormValues(container)).not.toContain("CHALLENGE-STATE");

    await act(async () => completion?.action());
    expect(completeMcpOAuth).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain("MCP OAuth callback was rejected; begin a new login.");
    expect(container.textContent).not.toContain("PRIVATE-CODE");
    expect(container.textContent).not.toContain("PRIVATE-STATE");
    expect(mcpFormValues(container)).not.toContain("PRIVATE-CODE");
    expect(mcpFormValues(container)).not.toContain("PRIVATE-STATE");

    await act(async () => completion?.action());
    expect(completeMcpOAuth).toHaveBeenCalledTimes(1);
    expect(container.textContent).not.toContain("PRIVATE-CODE");
    expect(container.textContent).not.toContain("PRIVATE-STATE");
  });

  it("clears a captured OAuth callback when confirmation is cancelled", async () => {
    const callbackValue = "http://127.0.0.1/callback?code=CANCEL-CODE&state=CANCEL-STATE";
    const beginMcpOAuth = vi.fn().mockResolvedValue({
      challenge_id: "challenge-local",
      authorization_url: "https://issuer.example/authorize?state=CHALLENGE-STATE",
      expires_at: "2030-01-01T00:00:00Z"
    });
    const completeMcpOAuth = vi.fn();
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render({ beginMcpOAuth, completeMcpOAuth }, confirmAction);
    const user = userEvent.setup();

    await act(async () => user.type(inputAfterLabel(container, "Host auth profile id"), "profile-local"));
    await act(async () => user.click(button(container, "Login")));
    await act(async () => confirmAction.mock.calls[0]?.[0].action());
    const callback = inputAfterLabel(container, "Full callback URL");
    await act(async () => user.type(callback, callbackValue));
    await act(async () => user.click(button(container, "Submit callback")));
    const completion = confirmAction.mock.calls[1]?.[0];

    await act(async () => completion?.onCancel?.());
    await act(async () => completion?.action());
    expect(callback.value).toBe("");
    expect(completeMcpOAuth).not.toHaveBeenCalled();
    expect(container.textContent).not.toContain("CANCEL-CODE");
    expect(container.textContent).not.toContain("CANCEL-STATE");
    expect(mcpFormValues(container)).not.toContain("CANCEL-CODE");
    expect(mcpFormValues(container)).not.toContain("CANCEL-STATE");
  });

  it("keeps OAuth Host profile secrets one-shot across validation, cancellation, and facade errors", async () => {
    const secret = "GUI-OAUTH-REACT-SECRET-SENTINEL";
    const configureMcpOAuthProfile = vi.fn()
      .mockRejectedValueOnce(new Error("profile change denied"))
      .mockResolvedValue({
        profile_id: "profile-local",
        status: "authorization_required",
        scopes: []
      });
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render({ configureMcpOAuthProfile }, confirmAction);
    const user = userEvent.setup();
    const profileEditor = textareaAfterLabel(container, "Strict non-secret OAuth Host profile");
    const secretInput = inputAfterLabel(container, "Optional client secret (transient broker input)");
    const profile = {
      profile_id: "profile-local",
      server_id: "server/1",
      resource_uri: "https://resource.example/mcp",
      expected_issuer: "https://issuer.example",
      redirect_uri: "http://127.0.0.1/callback",
      client_id: "gui-client",
      registration_mode: "preregistered",
      token_endpoint_auth_method: "client_secret_basic",
      allowed_scopes: ["resource.read"],
      default_scopes: ["resource.read"],
      allowed_endpoint_origins: ["https://issuer.example"],
      allow_loopback_http: true,
      protocol_revision: "2026-07-28",
      transport: "streamable_http"
    };

    await setTextarea(profileEditor, JSON.stringify(profile));
    await act(async () => user.type(secretInput, secret));
    await act(async () => user.click(button(container, "Replace Host profile")));

    expect(secretInput.value).toBe("");
    const confirmation = confirmAction.mock.calls[0]?.[0];
    expect(confirmation).toBeTruthy();
    expect(JSON.stringify(confirmation?.details)).not.toContain(secret);
    await act(async () => confirmation?.action());
    expect(configureMcpOAuthProfile).toHaveBeenNthCalledWith(
      1,
      profile,
      secret,
      true,
      true
    );
    expect(container.textContent).toContain("profile change denied");
    expect(container.textContent).not.toContain(secret);

    // The retry closure cannot replay the secret after an ambiguous facade
    // failure. A fresh input and confirmation are required to submit it again.
    await act(async () => confirmation?.action());
    expect(configureMcpOAuthProfile).toHaveBeenNthCalledWith(
      2,
      profile,
      null,
      true,
      true
    );

    await act(async () => user.type(secretInput, secret));
    await act(async () => user.click(button(container, "Add Host profile")));
    const cancelled = confirmAction.mock.calls[1]?.[0];
    expect(secretInput.value).toBe("");
    cancelled?.onCancel?.();
    await act(async () => cancelled?.action());
    expect(configureMcpOAuthProfile).toHaveBeenNthCalledWith(
      3,
      profile,
      null,
      false,
      true
    );

    await act(async () => user.type(secretInput, secret));
    await setTextarea(profileEditor, "{");
    await act(async () => user.click(button(container, "Add Host profile")));
    expect(secretInput.value).toBe("");
    expect(confirmAction).toHaveBeenCalledTimes(2);
    expect(container.textContent).not.toContain(secret);
  });

  it("binds continuation answers to the displayed durable HumanRequest receipt", async () => {
    const receipt = {
      human_request_id: "human-local",
      human_revision: 4,
      human_preview_sha256: "c".repeat(64)
    };
    const readMcpResource = vi.fn().mockResolvedValue({
      kind: "input_required",
      continuation_id: "continuation-local",
      revision: 3,
      respondable: true,
      input_requests: [{
        request_id: "field-local",
        kind: "elicitation",
        mode: "url",
        prompt: "Review the provider request",
        schema: {},
        inert_url: "https://provider.invalid/review"
      }],
      ...receipt
    });
    const respondMcpContinuation = vi.fn().mockResolvedValue({ kind: "complete", value: null });
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render({ readMcpResource, respondMcpContinuation }, confirmAction);
    const user = userEvent.setup();

    await act(async () => user.type(inputAfterLabel(container, "Logical resource id or opaque handle"), "document-local"));
    await act(async () => user.click(button(container, "Read resource")));

    expect(container.textContent).toContain("human-local");
    expect(container.textContent).toContain("Human revision4");
    expect(container.textContent).toContain("https://provider.invalid/review");
    expect(container.querySelector('a[href*="provider.invalid"]')).toBeNull();
    expect(button(container, "Respond").disabled).toBe(true);
    expect(button(container, "Bind reviewed Human request receipt").disabled).toBe(true);

    await act(async () => user.selectOptions(
      selectAfterLabel(container, "field-local response action"),
      "accept"
    ));
    expect(button(container, "Bind reviewed Human request receipt").disabled).toBe(true);
    await act(async () => user.click(checkboxAfterText(
      container,
      "I explicitly reviewed the inert URL for field-local"
    )));
    await act(async () => user.click(button(container, "Bind reviewed Human request receipt")));
    expect(container.textContent).toContain("Reviewed Human request receipt is bound.");
    expect(button(container, "Respond").disabled).toBe(false);

    await act(async () => user.selectOptions(
      selectAfterLabel(container, "field-local response action"),
      "decline"
    ));
    expect(button(container, "Respond").disabled).toBe(true);
    expect(container.textContent).toContain("any edit invalidates it");
    await act(async () => user.selectOptions(
      selectAfterLabel(container, "field-local response action"),
      "accept"
    ));
    await act(async () => user.click(button(container, "Bind reviewed Human request receipt")));
    await act(async () => user.click(button(container, "Respond")));

    expect(confirmAction.mock.calls[0]?.[0].details).toMatchObject({
      continuation_id: "continuation-local",
      expected_revision: 3,
      human_request_id: "human-local",
      human_revision: 4,
      human_preview_sha256: "c".repeat(64)
    });
    await act(async () => confirmAction.mock.calls[0]?.[0].action());
    expect(respondMcpContinuation).toHaveBeenCalledWith(
      "continuation-local",
      3,
      { "field-local": { action: "accept" } },
      receipt,
      true
    );
  });

  it("requires the Task Human receipt and invalidates it when answers change", async () => {
    const receipt = {
      human_request_id: "human-task-local",
      human_revision: 2,
      human_preview_sha256: "d".repeat(64)
    };
    const task = {
      kind: "remote_task",
      task_ref: "task-local",
      revision: 6,
      status: "input_required",
      input_requests: [{
        request_id: "task-field-local",
        kind: "elicitation",
        mode: "form",
        prompt: "Approve the remote Task?",
        schema: {
          type: "object",
          properties: {
            approved: { type: "boolean", title: "Approve task" }
          },
          required: ["approved"]
        }
      }],
      ...receipt
    };
    const getMcpRemoteTask = vi.fn().mockResolvedValue(task);
    const updateMcpRemoteTask = vi.fn().mockResolvedValue({
      kind: "remote_task",
      task_ref: "task-local",
      revision: 7,
      status: "working",
      input_requests: []
    });
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render({ getMcpRemoteTask, updateMcpRemoteTask }, confirmAction);
    const user = userEvent.setup();

    await act(async () => user.type(inputAfterLabel(container, "Local task reference"), "task-local"));
    await act(async () => user.click(button(container, "Load / reobserve once")));
    expect(getMcpRemoteTask).toHaveBeenCalledWith("task-local", undefined);
    await act(async () => user.selectOptions(
      selectAfterLabel(container, "task-field-local response action"),
      "accept"
    ));
    await act(async () => user.selectOptions(
      selectAfterLabel(container, "task-field-local Approve task *"),
      "true"
    ));
    await act(async () => user.click(button(container, "Bind reviewed Task Human receipt")));
    expect(button(container, "Update").disabled).toBe(false);
    await act(async () => user.selectOptions(
      selectAfterLabel(container, "task-field-local Approve task *"),
      "false"
    ));
    expect(button(container, "Update").disabled).toBe(true);
    await act(async () => user.click(button(container, "Bind reviewed Task Human receipt")));
    await act(async () => user.click(button(container, "Update")));
    await act(async () => confirmAction.mock.calls[0]?.[0].action());

    expect(updateMcpRemoteTask).toHaveBeenCalledWith(
      "task-local",
      6,
      {
        "task-field-local": {
          action: "accept",
          content: { approved: false }
        }
      },
      receipt,
      true
    );
  });

  it("restores a durable continuation after a fresh panel mount", async () => {
    const pending = {
      kind: "input_required",
      continuation_id: "continuation-reopened",
      revision: 14,
      respondable: true,
      input_requests: [{
        request_id: "approval-local",
        kind: "elicitation",
        mode: "form",
        prompt: "Approve after Runtime reopen?",
        schema: {
          type: "object",
          properties: { approved: { type: "boolean", title: "Approved" } },
          required: ["approved"]
        }
      }],
      human_request_id: "human-reopened",
      human_revision: 8,
      human_preview_sha256: "f".repeat(64)
    };
    const getMcpContinuation = vi.fn().mockResolvedValue(pending);
    const first = await render({ getMcpContinuation });
    const firstUser = userEvent.setup();

    await act(async () => firstUser.type(
      inputAfterLabel(first.container, "Local continuation id"),
      "continuation-reopened"
    ));
    await act(async () => firstUser.click(button(first.container, "Load / refresh")));
    expect(getMcpContinuation).toHaveBeenLastCalledWith("continuation-reopened");
    expect(first.container.textContent).toContain("human-reopened");
    expect(inputAfterLabel(first.container, "Expected revision").value).toBe("14");
    expect(selectAfterLabel(first.container, "approval-local response action")).toBeTruthy();

    // A new component has no renderer memory. The only recovery input is the
    // local continuation id; the reopened Runtime returns the durable schema
    // and HumanRequest receipt again.
    const second = await render({ getMcpContinuation });
    const secondUser = userEvent.setup();
    await act(async () => secondUser.type(
      inputAfterLabel(second.container, "Local continuation id"),
      "continuation-reopened"
    ));
    await act(async () => secondUser.click(button(second.container, "Load / refresh")));
    expect(getMcpContinuation).toHaveBeenCalledTimes(2);
    expect(second.container.textContent).toContain("Approve after Runtime reopen?");
    expect(inputAfterLabel(second.container, "Expected revision").value).toBe("14");
  });

  it("clears a loaded continuation when refresh reports expiry or absence", async () => {
    const getMcpContinuation = vi.fn()
      .mockResolvedValueOnce({
        kind: "input_required",
        continuation_id: "continuation-expiring",
        revision: 5,
        respondable: true,
        input_requests: [{
          request_id: "approval-local",
          kind: "elicitation",
          mode: "form",
          schema: {
            type: "object",
            properties: { approved: { type: "boolean" } }
          }
        }],
        human_request_id: "human-expiring",
        human_revision: 1,
        human_preview_sha256: "a".repeat(64)
      })
      .mockRejectedValueOnce(new Error("MCP continuation expired"))
      .mockRejectedValueOnce(new Error("MCP continuation was not found"));
    const { container } = await render({ getMcpContinuation });
    const user = userEvent.setup();
    await act(async () => user.type(
      inputAfterLabel(container, "Local continuation id"),
      "continuation-expiring"
    ));
    await act(async () => user.click(button(container, "Load / refresh")));
    expect(container.textContent).toContain("human-expiring");

    await act(async () => user.click(button(container, "Load / refresh")));
    expect(container.textContent).toContain("MCP continuation expired");
    expect(container.textContent).not.toContain("human-expiring");
    expect(button(container, "Bind reviewed Human request receipt").disabled).toBe(true);
    expect(button(container, "Respond").disabled).toBe(true);

    await act(async () => user.click(button(container, "Load / refresh")));
    expect(container.textContent).toContain("MCP continuation was not found");
    expect(button(container, "Cancel continuation").disabled).toBe(true);
  });

  it("recovers a nonzero Task revision before update or cancellation after reload", async () => {
    const task = {
      kind: "remote_task",
      task_ref: "task-reopened",
      revision: 23,
      status: "input_required",
      input_requests: [{
        request_id: "task-approval",
        kind: "elicitation",
        mode: "form",
        prompt: "Approve the reopened Task?",
        schema: {
          type: "object",
          properties: { approved: { type: "boolean", title: "Approved" } },
          required: ["approved"]
        }
      }],
      human_request_id: "human-task-reopened",
      human_revision: 7,
      human_preview_sha256: "b".repeat(64)
    };
    const getMcpRemoteTask = vi.fn().mockResolvedValue(task);
    const cancelMcpRemoteTask = vi.fn().mockResolvedValue({
      ...task,
      revision: 24,
      status: "cancel_requested",
      input_requests: [],
      human_request_id: null,
      human_revision: null,
      human_preview_sha256: null
    });
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render(
      { getMcpRemoteTask, cancelMcpRemoteTask },
      confirmAction
    );
    const user = userEvent.setup();
    await act(async () => user.type(
      inputAfterLabel(container, "Local task reference"),
      "task-reopened"
    ));
    expect(button(container, "Request cancellation").disabled).toBe(true);

    await act(async () => user.click(button(container, "Load / reobserve once")));
    expect(getMcpRemoteTask).toHaveBeenCalledWith("task-reopened", undefined);
    expect(inputAfterLabel(container, "Expected revision", 1).value).toBe("23");
    expect(button(container, "Request cancellation").disabled).toBe(false);
    expect(selectAfterLabel(container, "task-approval response action")).toBeTruthy();

    await act(async () => user.click(button(container, "Load / reobserve once")));
    expect(getMcpRemoteTask).toHaveBeenNthCalledWith(2, "task-reopened", 23);

    await act(async () => user.click(button(container, "Request cancellation")));
    expect(confirmAction.mock.calls[0]?.[0].details).toMatchObject({
      task_ref: "task-reopened",
      expected_revision: 23
    });
    await act(async () => confirmAction.mock.calls[0]?.[0].action());
    expect(cancelMcpRemoteTask).toHaveBeenCalledWith("task-reopened", 23, true);
  });

  it("fails closed on unsupported Elicitation schemas before confirmation or client dispatch", async () => {
    const readMcpResource = vi.fn().mockResolvedValue({
      kind: "input_required",
      continuation_id: "continuation-local",
      revision: 1,
      respondable: true,
      input_requests: [{
        request_id: "input-unsupported",
        kind: "elicitation",
        mode: "form",
        prompt: "Send an unsupported nested value",
        schema: {
          type: "object",
          properties: { nested: { type: "object" } },
          required: ["nested"]
        }
      }],
      human_request_id: "human-local",
      human_revision: 1,
      human_preview_sha256: "e".repeat(64)
    });
    const respondMcpContinuation = vi.fn();
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render({ readMcpResource, respondMcpContinuation }, confirmAction);
    const user = userEvent.setup();

    await act(async () => user.type(
      inputAfterLabel(container, "Logical resource id or opaque handle"),
      "document-local"
    ));
    await act(async () => user.click(button(container, "Read resource")));

    expect(container.textContent).toContain("Unsupported MCP Elicitation schema");
    expect(container.textContent).toContain("unsupported type");
    expect(button(container, "Bind reviewed Human request receipt").disabled).toBe(true);
    expect(button(container, "Respond").disabled).toBe(true);
    expect(confirmAction).not.toHaveBeenCalled();
    expect(respondMcpContinuation).not.toHaveBeenCalled();
    expect(() => textareaAfterLabel(container, "Human answers keyed by local request id")).toThrow();
  });

  it("requires an explicit empty response for requestState-only continuations", async () => {
    const receipt = {
      human_request_id: "human-load-shed",
      human_revision: 3,
      human_preview_sha256: "9".repeat(64)
    };
    const readMcpResource = vi.fn().mockResolvedValue({
      kind: "input_required",
      continuation_id: "continuation-load-shed",
      revision: 2,
      respondable: true,
      input_requests: [],
      ...receipt
    });
    const respondMcpContinuation = vi.fn().mockResolvedValue({
      kind: "complete",
      value: null
    });
    const confirmAction = vi.fn<(request: ConfirmationRequest) => void>();
    const { container } = await render(
      { readMcpResource, respondMcpContinuation },
      confirmAction
    );
    const user = userEvent.setup();
    await act(async () => user.type(
      inputAfterLabel(container, "Logical resource id or opaque handle"),
      "load-shed-resource"
    ));
    await act(async () => user.click(button(container, "Read resource")));

    expect(container.textContent).toContain("explicit response is the empty object {}");
    expect(button(container, "Bind reviewed Human request receipt").disabled).toBe(false);
    await act(async () => user.click(button(container, "Bind reviewed Human request receipt")));
    await act(async () => user.click(button(container, "Respond")));
    await act(async () => confirmAction.mock.calls[0]?.[0].action());
    expect(respondMcpContinuation).toHaveBeenCalledWith(
      "continuation-load-shed",
      2,
      {},
      receipt,
      true
    );
  });

  it("renders typed unsupported requests without inventing a continuation or Human receipt", async () => {
    const readMcpResource = vi.fn().mockResolvedValue({
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
    });
    const { container } = await render({ readMcpResource });
    const user = userEvent.setup();
    await act(async () => user.type(
      inputAfterLabel(container, "Logical resource id or opaque handle"),
      "unsupported-resource"
    ));
    await act(async () => user.click(button(container, "Read resource")));

    expect(container.textContent).toContain("No continuation or HumanRequest was captured");
    expect(container.textContent).not.toContain("Human request id");
    expect(button(container, "Bind reviewed Human request receipt").disabled).toBe(true);
    expect(button(container, "Respond").disabled).toBe(true);
    expect(button(container, "Cancel continuation").disabled).toBe(true);
  });

  it("does not poll Tasks or subscriptions merely because the panel is mounted", async () => {
    const getMcpRemoteTask = vi.fn();
    const getMcpSubscriptionStatus = vi.fn();
    const listMcpSubscriptionEvents = vi.fn();
    const { container } = await render({
      getMcpRemoteTask,
      getMcpSubscriptionStatus,
      listMcpSubscriptionEvents
    });

    await act(async () => new Promise((resolve) => setTimeout(resolve, 10)));

    expect(container.textContent).toContain("Lost subscriptions stay lost");
    expect(getMcpRemoteTask).not.toHaveBeenCalled();
    expect(getMcpSubscriptionStatus).not.toHaveBeenCalled();
    expect(listMcpSubscriptionEvents).not.toHaveBeenCalled();
  });

  it("advances the single subscription reader cursor after a delivered batch", async () => {
    const listMcpSubscriptionEvents = vi.fn()
      .mockResolvedValueOnce([{
        sequence: 1,
        event_type: "resourcesListChanged",
        payload: { changed: true },
        received_at: "2026-08-11T00:00:00Z",
        provenance: "untrusted_mcp_notification"
      }])
      .mockResolvedValueOnce([]);
    const { container } = await render({ listMcpSubscriptionEvents });
    const user = userEvent.setup();
    await act(async () => user.type(
      inputAfterLabel(container, "Local subscription id"),
      "subscription-local"
    ));

    await act(async () => user.click(button(container, "Read events once")));
    expect(listMcpSubscriptionEvents).toHaveBeenLastCalledWith(
      "subscription-local",
      0,
      100
    );
    expect(inputAfterLabel(container, "Events after sequence").value).toBe("1");

    await act(async () => user.click(button(container, "Read events once")));
    expect(listMcpSubscriptionEvents).toHaveBeenLastCalledWith(
      "subscription-local",
      1,
      100
    );
    expect(inputAfterLabel(container, "Events after sequence").value).toBe("1");
  });
});

async function render(
  overrides: Partial<LibOSClient>,
  confirmAction: (request: ConfirmationRequest) => void = () => undefined
) {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  mounted.push({ root, container });
  await act(() => root.render(
    <I18nProvider initialLanguage="en">
      <McpModernPanel
        serverId="server/1"
        client={overrides as LibOSClient}
        confirmAction={confirmAction}
      />
    </I18nProvider>
  ));
  return { container, root };
}

function button(container: HTMLElement, text: string): HTMLButtonElement {
  const selected = [...container.querySelectorAll("button")]
    .find((item) => item.textContent === text);
  if (!selected) throw new Error(`Missing button ${text}`);
  return selected;
}

function inputAfterLabel(container: HTMLElement, text: string, index = 0): HTMLInputElement {
  const label = [...container.querySelectorAll("label")]
    .filter((item) => item.querySelector("span")?.textContent === text)[index];
  const selected = label?.querySelector("input");
  if (!selected) throw new Error(`Missing input ${text}`);
  return selected;
}

function selectAfterLabel(container: HTMLElement, text: string): HTMLSelectElement {
  const label = [...container.querySelectorAll("label")]
    .find((item) => item.querySelector("span")?.textContent === text);
  const selected = label?.querySelector("select");
  if (!selected) throw new Error(`Missing select ${text}`);
  return selected;
}

function textareaAfterLabel(container: HTMLElement, text: string): HTMLTextAreaElement {
  const label = [...container.querySelectorAll("label")]
    .find((item) => item.querySelector("span")?.textContent === text);
  const selected = label?.querySelector("textarea");
  if (!selected) throw new Error(`Missing textarea ${text}`);
  return selected;
}

function mcpFormValues(container: HTMLElement): string {
  return [...container.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>("input, textarea")]
    .map((item) => item.value)
    .join("\n");
}

function checkboxAfterText(container: HTMLElement, text: string): HTMLInputElement {
  const label = [...container.querySelectorAll("label")]
    .find((item) => item.textContent?.trim() === text);
  const selected = label?.querySelector<HTMLInputElement>('input[type="checkbox"]');
  if (!selected) throw new Error(`Missing checkbox ${text}`);
  return selected;
}

async function setTextarea(element: HTMLTextAreaElement, value: string): Promise<void> {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    if (!setter) throw new Error("Missing textarea value setter");
    setter.call(element, value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
