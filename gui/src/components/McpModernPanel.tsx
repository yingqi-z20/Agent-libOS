import { useMemo, useRef, useState } from "react";
import type { LibOSClient } from "../api/client";
import type {
  McpAuthorizationChallenge,
  McpHumanReceipt,
  McpInputRequired,
  McpOAuthProfileInput,
  McpOAuthStatus,
  McpOperationResult,
  McpPrompt,
  McpPromptResult,
  McpRemoteTask,
  McpResource,
  McpResourceTemplate
} from "../api/types";
import { assertMcpOAuthProfileInput } from "../api/types";
import type { ConfirmationRequest } from "../adminTypes";
import { CollapsibleJson } from "./CollapsibleJson";
import { McpElicitationForm } from "./McpElicitationForm";
import {
  buildElicitationResponses,
  elicitationResponsesReady,
  initializeElicitationDrafts,
  type ElicitationDrafts
} from "./mcpElicitation";

/** Host-only MCP 2026-07-28 controls.
 *
 * Provider text is rendered only through text nodes/JSON blocks.  In
 * particular this component never injects HTML, opens OAuth/resource URLs,
 * follows ResourceLink blocks, polls remote Tasks, or reconnects a lost
 * subscription automatically.
 */
export function McpModernPanel({
  serverId,
  authProfileId,
  client,
  confirmAction
}: {
  serverId: string;
  authProfileId?: string | null;
  client: LibOSClient;
  confirmAction(request: ConfirmationRequest): void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<unknown>(null);

  const [resources, setResources] = useState<McpResource[]>([]);
  const [resourceCursor, setResourceCursor] = useState<string | null>(null);
  const [templates, setTemplates] = useState<McpResourceTemplate[]>([]);
  const [templateCursor, setTemplateCursor] = useState<string | null>(null);
  const [resourceId, setResourceId] = useState("");
  const [resourceVariables, setResourceVariables] = useState("{}");

  const [prompts, setPrompts] = useState<McpPrompt[]>([]);
  const [promptCursor, setPromptCursor] = useState<string | null>(null);
  const [promptId, setPromptId] = useState("");
  const [promptArguments, setPromptArguments] = useState("{}");
  const [promptPreview, setPromptPreview] = useState<McpOperationResult<McpPromptResult> | null>(null);
  const [promptPreviewBinding, setPromptPreviewBinding] = useState<{
    promptId: string;
    argumentsText: string;
    previewSha256: string;
  } | null>(null);
  const [completionReferenceType, setCompletionReferenceType] = useState<CompletionReferenceType>("prompt");
  const [completionReferenceId, setCompletionReferenceId] = useState("");
  const [completionArgument, setCompletionArgument] = useState('{"name":"","value":""}');
  const [completionContext, setCompletionContext] = useState("{}");

  const [oauthProfile, setOauthProfile] = useState(authProfileId ?? "");
  const [oauthProfiles, setOauthProfiles] = useState<McpOAuthStatus[]>([]);
  const [oauthProfileJson, setOauthProfileJson] = useState(() => oauthProfileTemplate(serverId, authProfileId));
  const [oauthScopes, setOauthScopes] = useState("");
  const [oauthChallenge, setOauthChallenge] = useState<McpAuthorizationChallenge | null>(null);
  const [oauthCallbackPresent, setOauthCallbackPresent] = useState(false);
  const oauthCallbackRef = useRef<HTMLInputElement>(null);
  const oauthClientSecretRef = useRef<HTMLInputElement>(null);

  const [continuationId, setContinuationId] = useState("");
  const [continuationRevision, setContinuationRevision] = useState(0);
  const [inputRequired, setInputRequired] = useState<McpInputRequired | null>(null);
  const [continuationDrafts, setContinuationDrafts] = useState<ElicitationDrafts>({});
  const [continuationReceipt, setContinuationReceipt] = useState<McpHumanReceipt | null>(null);

  const [taskRef, setTaskRef] = useState("");
  const [taskRevision, setTaskRevision] = useState(0);
  const [taskDrafts, setTaskDrafts] = useState<ElicitationDrafts>({});
  const [taskProjection, setTaskProjection] = useState<McpRemoteTask | null>(null);
  const [taskInputRequired, setTaskInputRequired] = useState<McpRemoteTask | null>(null);
  const [taskReceipt, setTaskReceipt] = useState<McpHumanReceipt | null>(null);

  const [subscriptionFilters, setSubscriptionFilters] = useState("");
  const [subscriptionId, setSubscriptionId] = useState("");
  const [subscriptionAfter, setSubscriptionAfter] = useState(0);

  const resourceIds = useMemo(
    () => [...resources.map((item) => item.resource_id), ...templates.map((item) => item.template_id)],
    [resources, templates]
  );
  const continuationAnswersReady = useMemo(
    () => Boolean(inputRequired?.respondable)
      && elicitationResponsesReady(inputRequired?.input_requests ?? [], continuationDrafts),
    [continuationDrafts, inputRequired]
  );
  const continuationLoaded = Boolean(
    inputRequired?.respondable
    && continuationId.trim()
    && inputRequired.continuation_id === continuationId.trim()
    && inputRequired.revision === continuationRevision
  );
  const taskLoaded = Boolean(
    taskProjection
    && taskProjection.task_ref === taskRef.trim()
    && taskProjection.revision === taskRevision
  );
  const taskAnswersReady = useMemo(
    () => taskLoaded
      && Boolean(taskInputRequired)
      && elicitationResponsesReady(taskInputRequired?.input_requests ?? [], taskDrafts),
    [taskDrafts, taskInputRequired, taskLoaded]
  );

  async function run<T>(operation: () => Promise<T>, apply?: (value: T) => void): Promise<T | null> {
    setBusy(true);
    setError(null);
    try {
      const value = await operation();
      apply?.(value);
      setResult(value);
      captureContinuationOrTask(value);
      return value;
    } catch (selected) {
      setError(describe(selected));
      return null;
    } finally {
      setBusy(false);
    }
  }

  function captureContinuationOrTask(value: unknown): void {
    if (!value || typeof value !== "object" || Array.isArray(value)) return;
    const selected = value as Record<string, unknown>;
    if (selected.kind === "input_required" && typeof selected.continuation_id === "string") {
      const projected = value as McpInputRequired;
      setInputRequired(projected);
      setContinuationId(projected.continuation_id);
      setContinuationRevision(projected.revision);
      setContinuationDrafts(initializeElicitationDrafts(projected.input_requests));
      setContinuationReceipt(null);
    }
    if (selected.kind === "remote_task" && typeof selected.task_ref === "string") {
      const projected = value as McpRemoteTask;
      setTaskRef(projected.task_ref);
      setTaskRevision(projected.revision);
      setTaskProjection(projected);
      setTaskInputRequired(projected.status === "input_required" ? projected : null);
      setTaskDrafts(
        projected.status === "input_required"
          ? initializeElicitationDrafts(projected.input_requests)
          : {}
      );
      setTaskReceipt(null);
    }
  }

  async function loadContinuation(): Promise<void> {
    const id = continuationId.trim();
    if (!id) return;
    setInputRequired(null);
    setContinuationDrafts({});
    setContinuationReceipt(null);
    setResult(null);
    await run(() => client.getMcpContinuation(id));
  }

  async function reobserveTask(): Promise<void> {
    const ref = taskRef.trim();
    if (!ref) return;
    const expectedRevision = taskLoaded ? taskRevision : undefined;
    setTaskProjection(null);
    setTaskInputRequired(null);
    setTaskDrafts({});
    setTaskReceipt(null);
    setResult(null);
    await run(() => client.getMcpRemoteTask(ref, expectedRevision));
  }

  function bindContinuationReceipt(): void {
    try {
      const source = inputRequired;
      const id = required(continuationId, "Continuation id");
      if (!source
          || !source.respondable
          || source.continuation_id !== id
          || source.revision !== continuationRevision) {
        throw new Error("Refresh the exact Human input request before binding its receipt.");
      }
      buildElicitationResponses(source.input_requests, continuationDrafts);
      setContinuationReceipt(humanReceipt(source));
      setError(null);
    } catch (selected) {
      setContinuationReceipt(null);
      setError(describe(selected));
    }
  }

  function bindTaskReceipt(): void {
    try {
      const source = taskInputRequired;
      const ref = required(taskRef, "Task reference");
      if (!source
          || source.status !== "input_required"
          || source.task_ref !== ref
          || source.revision !== taskRevision) {
        throw new Error("Refresh the exact Task Human input request before binding its receipt.");
      }
      buildElicitationResponses(source.input_requests, taskDrafts);
      setTaskReceipt(humanReceipt(source));
      setError(null);
    } catch (selected) {
      setTaskReceipt(null);
      setError(describe(selected));
    }
  }

  function confirm(
    title: string,
    details: Record<string, unknown>,
    action: () => Promise<unknown>,
    onCancel?: () => void
  ): void {
    confirmAction({
      title,
      message: "This Host action is explicit and is never performed by a model or notification.",
      details,
      onCancel,
      action: async () => {
        await run(action);
      }
    });
  }

  function prepareConfirmation(
    create: () => {
      title: string;
      details: Record<string, unknown>;
      action: () => Promise<unknown>;
      onCancel?: () => void;
    }
  ): void {
    try {
      const prepared = create();
      setError(null);
      confirm(prepared.title, prepared.details, prepared.action, prepared.onCancel);
    } catch (selected) {
      setError(describe(selected));
    }
  }

  async function listResources(next = false): Promise<void> {
    await run(
      () => client.listMcpResources(serverId, next ? resourceCursor : null),
      (page) => {
        setResources((current) => next ? [...current, ...page.items] : page.items);
        setResourceCursor(page.next_cursor);
        if (!resourceId && page.items[0]) setResourceId(page.items[0].resource_id);
      }
    );
  }

  async function listTemplates(next = false): Promise<void> {
    await run(
      () => client.listMcpResourceTemplates(serverId, next ? templateCursor : null),
      (page) => {
        setTemplates((current) => next ? [...current, ...page.items] : page.items);
        setTemplateCursor(page.next_cursor);
      }
    );
  }

  async function listPrompts(next = false): Promise<void> {
    await run(
      () => client.listMcpPrompts(serverId, next ? promptCursor : null),
      (page) => {
        setPrompts((current) => next ? [...current, ...page.items] : page.items);
        setPromptCursor(page.next_cursor);
        if (!promptId && page.items[0]) setPromptId(page.items[0].prompt_id);
      }
    );
  }

  async function previewPrompt(): Promise<void> {
    const selectedPromptId = promptId.trim();
    const selectedArgumentsText = promptArguments;
    await run(
      () => client.getMcpPrompt(
        serverId,
        required(promptId, "Prompt id"),
        parseStringMap(promptArguments, "Prompt arguments"),
        false
      ),
      (value) => {
        setPromptPreview(value);
        setPromptPreviewBinding(value.kind === "complete" ? {
          promptId: selectedPromptId,
          argumentsText: selectedArgumentsText,
          previewSha256: value.preview_sha256 ?? ""
        } : null);
      }
    );
  }

  function confirmPrompt(): void {
    prepareConfirmation(() => {
      const selectedPromptId = required(promptId, "Prompt id");
      const selectedArguments = parseStringMap(promptArguments, "Prompt arguments");
      if (!promptPreviewBinding
          || promptPreviewBinding.promptId !== selectedPromptId
          || promptPreviewBinding.argumentsText !== promptArguments
          || promptPreview?.kind !== "complete") {
        throw new Error("Preview this exact prompt and arguments before confirming it.");
      }
      return {
        title: "Accept MCP prompt as untrusted user context",
        details: {
          server_id: serverId,
          prompt_id: selectedPromptId,
          provenance: "untrusted_mcp_prompt",
          preview_sha256: promptPreviewBinding.previewSha256
        },
        action: async () => {
          const accepted = await client.getMcpPrompt(
            serverId,
            selectedPromptId,
            selectedArguments,
            true,
            promptPreviewBinding.previewSha256
          );
          setPromptPreview(accepted);
          return accepted;
        }
      };
    });
  }

  function prepareOAuthProfileChange(replace: boolean): void {
    prepareConfirmation(() => {
      const secretInput = oauthClientSecretRef.current;
      const secretSlot: { value: string | null } = {
        value: secretInput?.value || null
      };
      if (secretInput) secretInput.value = "";
      try {
        const profile = parseOAuthProfile(oauthProfileJson);
        if (profile.server_id !== serverId) {
          throw new Error("OAuth profile server_id must match this registered MCP server.");
        }
        const clientSecretPresent = secretSlot.value !== null;
        return {
          title: replace ? "Replace MCP OAuth Host profile" : "Add MCP OAuth Host profile",
          details: {
            profile_id: profile.profile_id,
            server_id: profile.server_id,
            replace,
            client_secret_present: clientSecretPresent,
            note: "secret is sent only to the Runtime credential broker"
          },
          onCancel: () => {
            secretSlot.value = null;
          },
          action: async () => {
            const selectedSecret = secretSlot.value;
            secretSlot.value = null;
            try {
              const status = await client.configureMcpOAuthProfile(
                profile,
                selectedSecret,
                replace,
                true
              );
              setOauthProfile(profile.profile_id);
              setOauthProfiles((current) => upsertOAuthStatus(current, status));
              return status;
            } finally {
              secretSlot.value = null;
              if (secretInput) secretInput.value = "";
            }
          }
        };
      } catch (selected) {
        secretSlot.value = null;
        throw selected;
      }
    });
  }

  function prepareOAuthProfileRemoval(): void {
    prepareConfirmation(() => {
      const profileId = required(oauthProfile, "Auth profile");
      return {
        title: "Remove MCP OAuth Host profile",
        details: {
          profile_id: profileId,
          note: "broker handles are removed and only revoked non-secret metadata remains"
        },
        action: async () => {
          const status = await client.removeMcpOAuthProfile(profileId, true);
          setOauthProfiles((current) => current.filter((item) => item.profile_id !== profileId));
          return status;
        }
      };
    });
  }

  function clearOAuthAttempt(): void {
    if (oauthCallbackRef.current) oauthCallbackRef.current.value = "";
    setOauthCallbackPresent(false);
    setOauthChallenge(null);
    setResult(null);
  }

  function prepareOAuthLogin(): void {
    clearOAuthAttempt();
    prepareConfirmation(() => {
      const profile = required(oauthProfile, "Auth profile");
      const scopes = words(oauthScopes);
      return {
        title: "Begin MCP OAuth login",
        details: { profile_id: profile, scopes },
        onCancel: clearOAuthAttempt,
        action: async () => {
          const challenge = await client.beginMcpOAuth(profile, scopes, true);
          setOauthChallenge(challenge);
          return challenge;
        }
      };
    });
  }

  function prepareOAuthCallbackCompletion(): void {
    const challengeId = oauthChallenge?.challenge_id ?? null;
    const callbackSlot: { value: string | null } = {
      value: oauthCallbackRef.current?.value ?? null
    };
    clearOAuthAttempt();
    prepareConfirmation(() => {
      if (!challengeId || !callbackSlot.value?.trim()) {
        callbackSlot.value = null;
        throw new Error("A fresh MCP OAuth callback is required.");
      }
      return {
        title: "Complete MCP OAuth login",
        details: {
          challenge_id: challengeId,
          note: "the one-time callback is never retained in confirmation details"
        },
        onCancel: () => {
          callbackSlot.value = null;
          clearOAuthAttempt();
        },
        action: async () => {
          let selected = callbackSlot.value;
          callbackSlot.value = null;
          if (selected === null) {
            throw new Error("MCP OAuth callback was already consumed; begin a new login.");
          }
          try {
            let pending: Promise<McpOAuthStatus>;
            try {
              pending = client.completeMcpOAuth(challengeId, selected);
            } finally {
              selected = null;
            }
            const status = await pending;
            setOauthProfiles((current) => upsertOAuthStatus(current, status));
            return status;
          } catch {
            throw new Error("MCP OAuth callback was rejected; begin a new login.");
          } finally {
            callbackSlot.value = null;
            clearOAuthAttempt();
          }
        }
      };
    });
  }

  return (
    <section className="mcpModernPanel" aria-label="MCP 2026-07-28 Host controls" aria-busy={busy || undefined}>
      <header>
        <h4>MCP 2026-07-28 Host controls</h4>
        <p>Remote data is untrusted. Nothing here is a model tool, automatic trigger, or HTML rendering surface.</p>
      </header>

      <details className="adminDisclosure">
        <summary>Resources and templates</summary>
        <div className="adminActions">
          <button disabled={busy} onClick={() => void listResources(false)}>List resources</button>
          <button disabled={busy || !resourceCursor} onClick={() => void listResources(true)}>Next resources page</button>
          <button disabled={busy} onClick={() => void listTemplates(false)}>List templates</button>
          <button disabled={busy || !templateCursor} onClick={() => void listTemplates(true)}>Next templates page</button>
        </div>
        <label className="fieldStack">
          <span>Logical resource id or opaque handle</span>
          <input list="mcp-modern-resource-options" value={resourceId} onChange={(event) => setResourceId(event.currentTarget.value)} />
          <datalist id="mcp-modern-resource-options">{resourceIds.map((id) => <option key={id} value={id} />)}</datalist>
        </label>
        <JsonField label="Template variables (string map)" value={resourceVariables} onChange={setResourceVariables} />
        <button disabled={busy || !resourceId.trim()} onClick={() => void run(
          () => client.readMcpResource(
            serverId,
            required(resourceId, "Resource id"),
            parseStringMap(resourceVariables, "Resource variables")
          )
        )}>Read resource</button>
        {resources.length || templates.length ? <CollapsibleJson value={{ resources, templates, next_resource_cursor: resourceCursor, next_template_cursor: templateCursor }} label="resource catalog" /> : null}
      </details>

      <details className="adminDisclosure">
        <summary>Prompts and completion</summary>
        <div className="adminActions">
          <button disabled={busy} onClick={() => void listPrompts(false)}>List prompts</button>
          <button disabled={busy || !promptCursor} onClick={() => void listPrompts(true)}>Next prompts page</button>
        </div>
        <label className="fieldStack">
          <span>Logical prompt id</span>
          <input list="mcp-modern-prompt-options" value={promptId} onChange={(event) => {
            setPromptId(event.currentTarget.value);
            setPromptPreview(null);
            setPromptPreviewBinding(null);
          }} />
          <datalist id="mcp-modern-prompt-options">{prompts.map((prompt) => <option key={prompt.prompt_id} value={prompt.prompt_id} />)}</datalist>
        </label>
        <JsonField label="Prompt arguments (string map)" value={promptArguments} onChange={(value) => {
          setPromptArguments(value);
          setPromptPreview(null);
          setPromptPreviewBinding(null);
        }} />
        <div className="adminActions">
          <button disabled={busy || !promptId.trim()} onClick={() => void previewPrompt()}>Preview untrusted prompt</button>
          <button disabled={busy || promptPreview?.kind !== "complete"} onClick={confirmPrompt}>Confirm as user context</button>
        </div>
        {promptPreview ? <CollapsibleJson value={promptPreview} label="untrusted prompt preview" defaultExpanded /> : null}
        <div className="adminFormGrid">
          <label className="fieldStack">
            <span>Completion reference type</span>
            <select
              value={completionReferenceType}
              onChange={(event) => setCompletionReferenceType(completionReferenceTypeValue(event.currentTarget.value))}
            >
              <option value="prompt">prompt</option>
              <option value="resource_template">resource_template</option>
            </select>
          </label>
          <TextField label="Completion reference id" value={completionReferenceId} onChange={setCompletionReferenceId} />
          <JsonField label="Completion argument (string map)" value={completionArgument} onChange={setCompletionArgument} />
          <JsonField label="Completion context (string map)" value={completionContext} onChange={setCompletionContext} />
        </div>
        <button disabled={busy || !completionReferenceId.trim()} onClick={() => void run(
          () => client.completeMcpPrompt(
            serverId,
            completionReferenceType,
            required(completionReferenceId, "Reference id"),
            parseStringMap(completionArgument, "Completion argument"),
            parseStringMap(completionContext, "Completion context")
          )
        )}>Complete</button>
      </details>

      <details className="adminDisclosure">
        <summary>OAuth</summary>
        <p>Profiles are Host-owned authority. Add or replace one in this long-lived Runtime before login. After a Runtime restart, resubmit the exact non-secret profile to rebind any broker-held credential; authorization challenges are never resumed.</p>
        <JsonField label="Strict non-secret OAuth Host profile" value={oauthProfileJson} onChange={setOauthProfileJson} />
        <label className="fieldStack">
          <span>Optional client secret (transient broker input)</span>
          <input ref={oauthClientSecretRef} type="password" autoComplete="off" />
        </label>
        <div className="adminActions">
          <button disabled={busy} onClick={() => void run(
            () => client.listMcpOAuthProfiles(),
            setOauthProfiles
          )}>List Host profiles</button>
          <button disabled={busy} onClick={() => prepareOAuthProfileChange(false)}>Add Host profile</button>
          <button disabled={busy} onClick={() => prepareOAuthProfileChange(true)}>Replace Host profile</button>
          <button disabled={busy || !oauthProfile.trim()} onClick={prepareOAuthProfileRemoval}>Remove Host profile</button>
        </div>
        {oauthProfiles.length ? <CollapsibleJson value={oauthProfiles} label="non-secret OAuth Host profile statuses" /> : null}
        <TextField label="Host auth profile id" value={oauthProfile} onChange={setOauthProfile} />
        <TextField label="Requested scopes (space separated)" value={oauthScopes} onChange={setOauthScopes} />
        <div className="adminActions">
          <button disabled={busy || !oauthProfile.trim()} onClick={() => void run(
            () => client.getMcpOAuthStatus(required(oauthProfile, "Auth profile")),
            (status) => setOauthProfiles((current) => upsertOAuthStatus(current, status))
          )}>Status</button>
          <button disabled={busy || !oauthProfile.trim()} onClick={prepareOAuthLogin}>Login</button>
          <button disabled={busy || !oauthProfile.trim()} onClick={() => prepareConfirmation(() => {
            const profile = required(oauthProfile, "Auth profile");
            return {
              title: "Log out MCP OAuth profile",
              details: { profile_id: profile },
              action: async () => {
                const status = await client.logoutMcpOAuth(profile, true);
                setOauthProfiles((current) => upsertOAuthStatus(current, status));
                return status;
              }
            };
          })}>Logout</button>
        </div>
        {oauthChallenge ? (
          <>
            <label className="fieldStack">
              <span>Authorization URL (copy manually; never opened automatically)</span>
              <textarea readOnly className="codeInput" value={oauthChallenge.authorization_url} />
            </label>
            <label className="fieldStack">
              <span>Full callback URL</span>
              <input
                ref={oauthCallbackRef}
                type="text"
                autoComplete="off"
                spellCheck={false}
                onChange={(event) => setOauthCallbackPresent(Boolean(event.currentTarget.value.trim()))}
              />
            </label>
            <div className="adminActions">
              <button disabled={busy || !oauthCallbackPresent} onClick={prepareOAuthCallbackCompletion}>Submit callback</button>
              <button disabled={busy} onClick={clearOAuthAttempt}>Discard OAuth attempt</button>
            </div>
          </>
        ) : null}
      </details>

      <details className="adminDisclosure">
        <summary>Elicitation continuation</summary>
        {inputRequired ? <div className="mcpUntrustedNotice">
          <strong>{inputRequired.respondable
            ? "Durable Human input required"
            : "Unsupported MCP client request"}</strong>
          {inputRequired.respondable
            ? <HumanReceiptView value={humanReceipt(inputRequired)} />
            : null}
          <CollapsibleJson value={inputRequired.input_requests} label="untrusted elicitation schemas and inert URLs" defaultExpanded />
          <p>{inputRequired.respondable
            ? "Respond first settles this exact HumanRequest. Only its canonical approved answer may continue; the editor object is never sent directly to the Provider."
            : "This is a typed unsupported request and cannot be answered. No continuation or HumanRequest was captured; Sampling and Roots are not implemented."}</p>
        </div> : null}
        <TextField label="Local continuation id" value={continuationId} onChange={(value) => {
          setContinuationId(value);
          setInputRequired(null);
          setContinuationDrafts({});
          setContinuationReceipt(null);
        }} />
        <NumberField label="Expected revision" value={continuationRevision} onChange={(value) => {
          setContinuationRevision(value);
          setInputRequired(null);
          setContinuationDrafts({});
          setContinuationReceipt(null);
        }} />
        {inputRequired ? <McpElicitationForm
          requests={inputRequired.input_requests}
          drafts={continuationDrafts}
          disabled={!inputRequired.respondable}
          onChange={(value) => {
            setContinuationDrafts(value);
            setContinuationReceipt(null);
          }}
        /> : null}
        <div className="adminActions">
          <button disabled={busy || !continuationId.trim()} onClick={() => void loadContinuation()}>Load / refresh</button>
          <button disabled={busy || !continuationAnswersReady} onClick={bindContinuationReceipt}>Bind reviewed Human request receipt</button>
          <button disabled={busy || !continuationId.trim() || !continuationReceipt} onClick={() => prepareConfirmation(() => {
            const id = required(continuationId, "Continuation id");
            const revision = continuationRevision;
            if (!inputRequired) throw new Error("Refresh the exact Human input request before responding.");
            const responses = buildElicitationResponses(inputRequired.input_requests, continuationDrafts);
            if (!continuationReceipt) throw new Error("Bind the reviewed Human request receipt before responding.");
            const receipt = continuationReceipt;
            return {
              title: "Respond to MCP elicitation",
              details: {
                continuation_id: id,
                expected_revision: revision,
                human_request_id: receipt.human_request_id,
                human_revision: receipt.human_revision,
                human_preview_sha256: receipt.human_preview_sha256,
                settlement: "durable HumanRequest before MCP continuation"
              },
              action: async () => {
                const next = await client.respondMcpContinuation(id, revision, responses, receipt, true);
                setContinuationReceipt(null);
                if (next.kind === "complete" || next.kind === "remote_task") setInputRequired(null);
                return next;
              }
            };
          })}>Respond</button>
          <button disabled={busy || !continuationLoaded} onClick={() => prepareConfirmation(() => {
            const id = required(continuationId, "Continuation id");
            const revision = continuationRevision;
            if (!inputRequired
                || inputRequired.continuation_id !== id
                || inputRequired.revision !== revision) {
              throw new Error("Load the exact continuation revision before cancellation.");
            }
            return {
              title: "Cancel MCP continuation",
              details: { continuation_id: id, expected_revision: revision, pending_human_request: "cancelled by Runtime" },
              action: async () => {
                const next = await client.cancelMcpContinuation(id, revision, true);
                setContinuationReceipt(null);
                setInputRequired(null);
                return next;
              }
            };
          })}>Cancel continuation</button>
        </div>
        <p>{continuationReceipt ? "Reviewed Human request receipt is bound." : "No Human request receipt is bound; any edit invalidates it."}</p>
      </details>

      <details className="adminDisclosure">
        <summary>Remote Task</summary>
        {taskInputRequired ? <div className="mcpUntrustedNotice">
          <strong>Task requires a durable Human answer</strong>
          <HumanReceiptView value={humanReceipt(taskInputRequired)} />
          <CollapsibleJson value={taskInputRequired.input_requests} label="untrusted Task elicitation schemas and inert URLs" defaultExpanded />
          <p>Update first settles this exact HumanRequest. Only its canonical approved answer may continue; the editor object is never sent directly to the Provider.</p>
        </div> : null}
        <TextField label="Local task reference" value={taskRef} onChange={(value) => {
          setTaskRef(value);
          setTaskProjection(null);
          setTaskInputRequired(null);
          setTaskDrafts({});
          setTaskReceipt(null);
        }} />
        <NumberField label="Expected revision" value={taskRevision} onChange={(value) => {
          setTaskRevision(value);
          setTaskProjection(null);
          setTaskInputRequired(null);
          setTaskDrafts({});
          setTaskReceipt(null);
        }} />
        {taskInputRequired ? <McpElicitationForm
          requests={taskInputRequired.input_requests}
          drafts={taskDrafts}
          onChange={(value) => {
            setTaskDrafts(value);
            setTaskReceipt(null);
          }}
        /> : null}
        <div className="adminActions">
          <button disabled={busy || !taskRef.trim()} onClick={() => void reobserveTask()}>Load / reobserve once</button>
          <button disabled={busy || !taskAnswersReady} onClick={bindTaskReceipt}>Bind reviewed Task Human receipt</button>
          <button disabled={busy || !taskRef.trim() || !taskReceipt} onClick={() => prepareConfirmation(() => {
            const ref = required(taskRef, "Task reference");
            const revision = taskRevision;
            if (!taskInputRequired) throw new Error("Refresh the exact Task Human input request before updating.");
            const responses = buildElicitationResponses(taskInputRequired.input_requests, taskDrafts);
            if (!taskReceipt) throw new Error("Bind the reviewed Task Human receipt before updating.");
            const receipt = taskReceipt;
            return {
              title: "Update MCP remote Task",
              details: {
                task_ref: ref,
                expected_revision: revision,
                human_request_id: receipt.human_request_id,
                human_revision: receipt.human_revision,
                human_preview_sha256: receipt.human_preview_sha256,
                settlement: "durable HumanRequest before MCP Task update"
              },
              action: async () => {
                const next = await client.updateMcpRemoteTask(ref, revision, responses, receipt, true);
                setTaskReceipt(null);
                if (next.status !== "input_required") setTaskInputRequired(null);
                return next;
              }
            };
          })}>Update</button>
          <button disabled={busy || !taskLoaded} onClick={() => prepareConfirmation(() => {
            const ref = required(taskRef, "Task reference");
            const revision = taskRevision;
            if (!taskProjection
                || taskProjection.task_ref !== ref
                || taskProjection.revision !== revision) {
              throw new Error("Load the exact remote Task revision before cancellation.");
            }
            return {
              title: "Request MCP remote Task cancellation",
              details: { task_ref: ref, expected_revision: revision, note: "acknowledgement does not prove stopped; pending HumanRequest is closed" },
              action: async () => {
                const next = await client.cancelMcpRemoteTask(ref, revision, true);
                setTaskReceipt(null);
                setTaskInputRequired(null);
                return next;
              }
            };
          })}>Request cancellation</button>
        </div>
        <p>{taskReceipt ? "Reviewed Task Human receipt is bound." : "No Task Human receipt is bound; any edit invalidates it."}</p>
      </details>

      <details className="adminDisclosure">
        <summary>Subscriptions</summary>
        <TextField label="Filters (space separated)" value={subscriptionFilters} onChange={setSubscriptionFilters} />
        <TextField label="Local subscription id" value={subscriptionId} onChange={(value) => {
          setSubscriptionId(value);
          setSubscriptionAfter(0);
        }} />
        <NumberField label="Events after sequence" value={subscriptionAfter} onChange={setSubscriptionAfter} />
        <div className="adminActions">
          <button disabled={busy || !words(subscriptionFilters).length} onClick={() => prepareConfirmation(() => {
            const filters = words(subscriptionFilters);
            if (!filters.length) throw new Error("At least one subscription filter is required.");
            return {
              title: "Start MCP subscription",
              details: { server_id: serverId, requested_filters: filters },
              action: async () => {
                const subscription = await client.startMcpSubscription(serverId, filters, true);
                setSubscriptionId(subscription.subscription_id);
                setSubscriptionAfter(0);
                return subscription;
              }
            };
          })}>Start</button>
          <button disabled={busy || !subscriptionId.trim()} onClick={() => void run(() => client.getMcpSubscriptionStatus(subscriptionId))}>Status</button>
          <button disabled={busy || !subscriptionId.trim()} onClick={() => void run(async () => {
            const events = await client.listMcpSubscriptionEvents(subscriptionId, subscriptionAfter, 100);
            let nextAfter = subscriptionAfter;
            for (const event of events) {
              if (event.sequence <= nextAfter) throw new Error("MCP subscription event sequence did not advance.");
              nextAfter = event.sequence;
            }
            if (nextAfter !== subscriptionAfter) setSubscriptionAfter(nextAfter);
            return events;
          })}>Read events once</button>
          <button disabled={busy || !subscriptionId.trim()} onClick={() => prepareConfirmation(() => {
            const id = required(subscriptionId, "Subscription id");
            return {
              title: "Stop MCP subscription",
              details: { subscription_id: id },
              action: () => client.stopMcpSubscription(id, true)
            };
          })}>Stop</button>
        </div>
        <p>Lost subscriptions stay lost. Reopen explicitly and perform a full refresh; notifications never trigger models or tools.</p>
      </details>

      {error ? <div className="inlineError" role="alert">{error}</div> : null}
      {result !== null ? <CollapsibleJson value={result} label="modern MCP result" defaultExpanded /> : null}
    </section>
  );
}

function TextField({ label, value, onChange }: { label: string; value: string; onChange(value: string): void }) {
  return <label className="fieldStack"><span>{label}</span><input value={value} onChange={(event) => onChange(event.currentTarget.value)} /></label>;
}

type CompletionReferenceType = "prompt" | "resource_template";

function completionReferenceTypeValue(value: string): CompletionReferenceType {
  if (value === "prompt" || value === "resource_template") return value;
  throw new Error("MCP completion reference type is unsupported.");
}

function NumberField({ label, value, onChange }: { label: string; value: number; onChange(value: number): void }) {
  return <label className="fieldStack"><span>{label}</span><input type="number" min={0} step={1} value={value} onChange={(event) => onChange(Math.max(0, Math.trunc(event.currentTarget.valueAsNumber || 0)))} /></label>;
}

function JsonField({ label, value, onChange }: { label: string; value: string; onChange(value: string): void }) {
  return <label className="fieldStack spanAll"><span>{label}</span><textarea className="codeInput" value={value} onChange={(event) => onChange(event.currentTarget.value)} /></label>;
}

function HumanReceiptView({ value }: { value: McpHumanReceipt }) {
  return <dl className="mcpHumanReceipt">
    <dt>Human request id</dt><dd><code>{value.human_request_id}</code></dd>
    <dt>Human revision</dt><dd>{value.human_revision}</dd>
    <dt>Human preview receipt</dt><dd><code>{value.human_preview_sha256}</code></dd>
  </dl>;
}

function humanReceipt(value: McpInputRequired | McpRemoteTask): McpHumanReceipt {
  if (!value.human_request_id
      || value.human_revision === null
      || value.human_revision === undefined
      || !value.human_preview_sha256) {
    throw new Error("MCP Human request receipt is missing.");
  }
  return {
    human_request_id: value.human_request_id,
    human_revision: value.human_revision,
    human_preview_sha256: value.human_preview_sha256
  };
}

function parseObject(value: string, label: string): Record<string, unknown> {
  const parsed = JSON.parse(value.trim() || "{}");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error(`${label} must be a JSON object.`);
  return parsed as Record<string, unknown>;
}

function parseStringMap(value: string, label: string): Record<string, string> {
  const selected = parseObject(value, label);
  if (Object.entries(selected).some(([key, item]) => !key.trim() || typeof item !== "string")) {
    throw new Error(`${label} must contain only non-empty string keys and string values.`);
  }
  return selected as Record<string, string>;
}

function parseOAuthProfile(value: string): McpOAuthProfileInput {
  if (new TextEncoder().encode(value).byteLength > 64 * 1_024) {
    throw new Error("OAuth Host profile exceeds the renderer input bound.");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error("OAuth Host profile must be valid JSON.");
  }
  assertMcpOAuthProfileInput(parsed);
  return parsed;
}

function oauthProfileTemplate(serverId: string, profileId?: string | null): string {
  return JSON.stringify({
    profile_id: profileId ?? "",
    server_id: serverId,
    resource_uri: "",
    expected_issuer: "",
    redirect_uri: "",
    client_id: "",
    registration_mode: "preregistered",
    token_endpoint_auth_method: "none",
    allowed_scopes: [],
    default_scopes: [],
    allowed_endpoint_origins: [],
    allow_loopback_http: false,
    protocol_revision: "2026-07-28",
    transport: "streamable_http"
  }, null, 2);
}

function upsertOAuthStatus(
  current: McpOAuthStatus[],
  status: McpOAuthStatus
): McpOAuthStatus[] {
  return [
    ...current.filter((item) => item.profile_id !== status.profile_id),
    status
  ].sort((left, right) => left.profile_id.localeCompare(right.profile_id));
}

function required(value: string, label: string): string {
  const selected = value.trim();
  if (!selected) throw new Error(`${label} is required.`);
  return selected;
}

function words(value: string): string[] {
  return [...new Set(value.split(/\s+/).map((item) => item.trim()).filter(Boolean))];
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
