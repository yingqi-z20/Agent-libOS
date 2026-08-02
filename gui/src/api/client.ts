import type { AgentRating, AuditRecord, CapabilityDelegationInput, CapabilityMutationInput, CapabilitySummary, CheckpointDiffResult, CheckpointInspectResult, CheckpointSummary, ExplainOperationResponse, GuiConnection, HumanRequest, HumanResponseInput, ImageInspectResult, ImageMutationResult, ImagePackageFile, ImageSummary, JsonRpcEndpointSummary, LLMProfileInput, LLMProfileSummary, McpCallResult, McpDiscoveryResult, McpServerSummary, McpToolListResult, ModuleSummary, ObjectTask, OperationListResponse, RuntimeSnapshot, SseMessage, StreamConnectionStatus, TaskRunDetail, TaskRunHumanRequestPage, TaskRunLedgerPage, TaskRunSpecV1, TaskRunSummary } from "./types";
import { assertMcpCallResult, assertMcpDiscoveryResult, assertMcpServerSummary, assertMcpToolListResult, assertRuntimeSnapshot, assertSchedulerStatus, assertTaskRunDetail, assertTaskRunSummary } from "./types";
import type { OptionalQuanta } from "../quanta";

type JsonBody = Record<string, unknown>;
export type RequestOptions = { signal?: AbortSignal; timeoutMs?: number | null };
const defaultReadRequestTimeoutMs = 30_000;
export const objectTaskWaitDeadlineMarginMs = 5_000;
export const capabilityInventoryMaxItems = 10_000;
export const capabilityInventoryMaxPages = capabilityInventoryMaxItems;

export type CapabilityPageResponse = {
  items: CapabilitySummary[];
  next_after: string | null;
  has_more: boolean;
};

export class LibOSClient {
  constructor(private connection: GuiConnection) {}

  get db() {
    return this.connection.db;
  }

  async snapshot(options: RequestOptions = {}): Promise<RuntimeSnapshot> {
    const snapshot = await this.request<unknown>("GET", "/api/snapshot", undefined, options);
    assertRuntimeSnapshot(snapshot);
    return snapshot;
  }

  async createTaskRun(
    spec: TaskRunSpecV1,
    clientRequestId: string
  ): Promise<TaskRunSummary> {
    const payload = await this.request<unknown>("POST", "/api/task-runs", {
      spec,
      client_request_id: requiredCommandId(clientRequestId, "clientRequestId")
    });
    return taskRunSummary(payload);
  }

  async getTaskRun(
    runId: string,
    options: { requirementsLimit?: number; requirementsCursor?: string } = {}
  ): Promise<TaskRunDetail> {
    const query = new URLSearchParams();
    if (options.requirementsLimit !== undefined) query.set("requirements_limit", String(options.requirementsLimit));
    if (options.requirementsCursor) query.set("requirements_cursor", options.requirementsCursor);
    const suffix = query.toString() ? `?${query.toString()}` : "";
    const value = await this.request<unknown>("GET", `/api/task-runs/${encodeURIComponent(runId)}${suffix}`);
    assertTaskRunDetail(value);
    if (value.summary.run_id !== runId) {
      throw new Error("GUI task run detail identity does not match the requested run id.");
    }
    return value;
  }

  async listTaskRunLedger(runId: string, limit = 100, cursor?: string): Promise<TaskRunLedgerPage> {
    const query = new URLSearchParams({ limit: String(limit) });
    if (cursor) query.set("cursor", cursor);
    return taskRunLedgerPage(await this.request<unknown>(
      "GET",
      `/api/task-runs/${encodeURIComponent(runId)}/ledger?${query.toString()}`
    ));
  }

  async listTaskRunHumanRequests(
    runId: string,
    limit = 100,
    cursor?: string,
    statuses?: readonly string[]
  ): Promise<TaskRunHumanRequestPage> {
    const query = new URLSearchParams({ limit: String(limit) });
    if (cursor) query.set("cursor", cursor);
    if (statuses?.length) query.set("status", statuses.join(","));
    return taskRunHumanRequestPage(await this.request<unknown>(
      "GET",
      `/api/task-runs/${encodeURIComponent(runId)}/human-requests?${query.toString()}`
    ));
  }

  async getHumanRequest(requestId: string): Promise<HumanRequest> {
    const value = await this.request<unknown>("GET", `/api/human-requests/${encodeURIComponent(requestId)}`);
    const request = humanRequest(value);
    if (request.request_id !== requestId) throw new Error("GUI human request identity does not match the requested id.");
    return request;
  }

  async runTaskRun(runId: string, expectedRevision: number, commandId: string, maxQuanta: OptionalQuanta): Promise<TaskRunSummary> {
    return this.taskRunMutation(runId, "run", withOptionalQuanta(
      taskRunCommand(expectedRevision, commandId),
      maxQuanta
    ));
  }

  async pauseTaskRun(runId: string, expectedRevision: number, commandId: string): Promise<TaskRunSummary> {
    return this.taskRunMutation(runId, "pause", taskRunCommand(expectedRevision, commandId));
  }

  async resumeTaskRun(runId: string, expectedRevision: number, commandId: string): Promise<TaskRunSummary> {
    return this.taskRunMutation(runId, "resume", taskRunCommand(expectedRevision, commandId));
  }

  async cancelTaskRun(runId: string, expectedRevision: number, commandId: string, confirmed: boolean, reason = "cancelled from GUI"): Promise<TaskRunSummary> {
    return this.taskRunMutation(runId, "cancel", {
      ...taskRunCommand(expectedRevision, commandId),
      confirmed,
      reason
    });
  }

  async followUpTaskRun(
    runId: string,
    body: string,
    expectedRevision: number,
    commandId: string,
    options: { kind?: "normal" | "interrupt"; required?: boolean } = {}
  ): Promise<TaskRunSummary> {
    return this.taskRunMutation(runId, "follow-ups", {
      ...taskRunCommand(expectedRevision, commandId),
      body,
      kind: options.kind ?? "normal",
      required: options.required ?? true
    });
  }

  async recoverTaskRun(
    runId: string,
    optionId: string,
    expectedRevision: number,
    commandId: string,
    confirmed: boolean,
    receipt?: Record<string, unknown>
  ): Promise<TaskRunSummary> {
    return this.taskRunMutation(runId, "recover", {
      ...taskRunCommand(expectedRevision, commandId),
      option_id: optionId,
      confirmed,
      ...(receipt ? { receipt } : {})
    });
  }

  async rerunTaskRun(
    runId: string,
    expectedRevision: number,
    commandId: string,
    options: { clientRequestId?: string; specOverrides?: Partial<TaskRunSpecV1> } = {}
  ): Promise<TaskRunSummary> {
    return this.taskRunMutation(runId, "rerun", {
      ...taskRunCommand(expectedRevision, commandId),
      // The linked create has its own idempotency namespace. Deriving its
      // default from the stable command id keeps transport retries identical.
      client_request_id: options.clientRequestId ?? `${commandId}:create`,
      ...(options.specOverrides ? { spec_overrides: options.specOverrides } : {})
    });
  }

  private async taskRunMutation(runId: string, action: string, body: JsonBody): Promise<TaskRunSummary> {
    try {
      const value = await this.request<unknown>(
        "POST",
        `/api/task-runs/${encodeURIComponent(runId)}/${action}`,
        body
      );
      return taskRunSummary(value);
    } catch (error) {
      if (!isTaskRunConflict(error)) throw error;
      let currentSummary = taskRunConflictSummary(error);
      try {
        const detail = await this.getTaskRun(runId);
        if (
          !currentSummary
          || (
            detail.summary.run_id === currentSummary.run_id
            && detail.summary.revision > currentSummary.revision
          )
        ) {
          currentSummary = detail.summary;
        }
      } catch {
        // Keep the original stable 409. The App-level snapshot refresh is the
        // final reconciliation fallback when this exact detail read fails.
      }
      throw new TaskRunMutationError(error, currentSummary);
    }
  }

  async listOperations(pid: string, limit = 100, cursor?: string, options: RequestOptions = {}): Promise<OperationListResponse> {
    const query = new URLSearchParams({ pid, limit: String(limit) });
    if (cursor) query.set("cursor", cursor);
    return this.request<OperationListResponse>("GET", `/api/operations?${query.toString()}`, undefined, options);
  }

  async explainOperation(operationId: string, evidenceLimit = 200, cursor?: string, options: RequestOptions = {}): Promise<ExplainOperationResponse> {
    const query = new URLSearchParams({ evidence_limit: String(evidenceLimit) });
    if (cursor) query.set("cursor", cursor);
    return this.request<ExplainOperationResponse>(
      "GET",
      `/api/operations/${encodeURIComponent(operationId)}?${query.toString()}`,
      undefined,
      options
    );
  }

  async resolveOperation(kind: string, evidenceId: string, options: RequestOptions = {}): Promise<ExplainOperationResponse> {
    const query = new URLSearchParams({ kind, id: evidenceId });
    return this.request<ExplainOperationResponse>("GET", `/api/operations/resolve?${query.toString()}`, undefined, options);
  }

  async listProcessAudit(pid: string, limit?: number, beforeRecordId?: string, options: RequestOptions = {}): Promise<AuditRecord[]> {
    const query = new URLSearchParams();
    if (limit !== undefined) query.set("limit", String(limit));
    if (beforeRecordId) query.set("before", beforeRecordId);
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return this.request<AuditRecord[]>(
      "GET",
      `/api/processes/${encodeURIComponent(pid)}/audit${suffix}`,
      undefined,
      options
    );
  }

  async images(): Promise<ImageSummary[]> {
    return this.request<ImageSummary[]>("GET", "/api/images");
  }

  async createLLMProfile(profile: LLMProfileInput): Promise<LLMProfileSummary> {
    return this.request<LLMProfileSummary>("POST", "/api/llm-profiles", profile as unknown as JsonBody);
  }

  async updateLLMProfile(profileId: string, profile: LLMProfileInput): Promise<LLMProfileSummary> {
    return this.request<LLMProfileSummary>("PUT", `/api/llm-profiles/${encodeURIComponent(profileId)}`, profile as unknown as JsonBody);
  }

  async deleteLLMProfile(profileId: string): Promise<{ ok: boolean; profile_id: string }> {
    return this.request("DELETE", `/api/llm-profiles/${encodeURIComponent(profileId)}`);
  }

  async inspectImage(imageId: string): Promise<ImageInspectResult> {
    return this.request<ImageInspectResult>("GET", `/api/images/${encodeURIComponent(imageId)}`);
  }

  async registerImagePackage(imagePackage: ImagePackageFile, confirmed: boolean, replace = false, actor?: string) {
    return this.request<ImageMutationResult>("POST", "/api/images/register", {
      files: imagePackage.files,
      source: imagePackage.name,
      confirmed,
      replace,
      ...(actor ? { actor } : {})
    });
  }

  async createCheckpoint(pid: string, reason: string) {
    return this.request<{ checkpoint_id: string }>("POST", "/api/checkpoints/create", { pid, reason });
  }

  async inspectCheckpoint(checkpointId: string): Promise<CheckpointInspectResult> {
    return this.request<CheckpointInspectResult>(
      "GET",
      `/api/checkpoints/${encodeURIComponent(checkpointId)}`
    );
  }

  async listCheckpoints(pid?: string): Promise<CheckpointSummary[]> {
    const query = pid ? `?pid=${encodeURIComponent(pid)}` : "";
    return this.request<CheckpointSummary[]>("GET", `/api/checkpoints${query}`);
  }

  async diffCheckpoint(checkpointId: string): Promise<CheckpointDiffResult> {
    return this.request<CheckpointDiffResult>(
      "GET",
      `/api/checkpoints/${encodeURIComponent(checkpointId)}/diff`
    );
  }

  async restoreCheckpoint(checkpointId: string, confirmed: boolean, actor?: string) {
    return this.request<Record<string, unknown>>(
      "POST",
      `/api/checkpoints/${encodeURIComponent(checkpointId)}/restore`,
      { confirmed, ...(actor ? { actor } : {}) }
    );
  }

  async forkCheckpoint(checkpointId: string, confirmed: boolean, parentPid?: string, actor?: string) {
    return this.request<Record<string, unknown>>(
      "POST",
      `/api/checkpoints/${encodeURIComponent(checkpointId)}/fork`,
      {
        confirmed,
        ...(parentPid ? { parent_pid: parentPid } : {}),
        ...(actor ? { actor } : {})
      }
    );
  }

  async inspectSkill(skillId: string): Promise<Record<string, unknown>> {
    return this.request("GET", `/api/skills/${encodeURIComponent(skillId)}`);
  }

  async registerSkill(path: string, actor: string, confirmed: boolean, replace = false) {
    return this.request<Record<string, unknown>>("POST", "/api/skills/register", {
      path,
      actor,
      confirmed,
      replace
    });
  }

  async activateSkill(
    skillId: string,
    pid: string,
    expectedPackageSha256: string,
    confirmed: boolean,
    actor?: string
  ) {
    return this.request<Record<string, unknown>>(
      "POST",
      `/api/skills/${encodeURIComponent(skillId)}/activate`,
      {
        pid,
        expected_package_sha256: expectedPackageSha256,
        confirmed,
        ...(actor ? { actor } : {})
      }
    );
  }

  async unloadSkill(skillId: string, pid: string, confirmed: boolean, actor?: string) {
    return this.request<Record<string, unknown>>(
      "POST",
      `/api/skills/${encodeURIComponent(skillId)}/unload`,
      { pid, confirmed, ...(actor ? { actor } : {}) }
    );
  }

  async listCapabilities(subject?: string): Promise<CapabilitySummary[]> {
    const capabilities = new Map<string, CapabilitySummary>();
    const seenCursors = new Set<string>();
    let after: string | undefined;
    let receivedItems = 0;
    for (let pageIndex = 0; pageIndex < capabilityInventoryMaxPages; pageIndex += 1) {
      const page = await this.listCapabilityPage(subject, after);
      receivedItems += page.items.length;
      if (receivedItems > capabilityInventoryMaxItems) {
        throw new Error(`GUI capability inventory exceeds ${capabilityInventoryMaxItems} items.`);
      }
      for (const capability of page.items) capabilities.set(capability.cap_id, capability);
      if (!page.has_more) return Array.from(capabilities.values());
      if (!page.next_after || page.next_after === after || seenCursors.has(page.next_after)) {
        throw new Error("GUI capability pagination returned a repeated or missing cursor.");
      }
      seenCursors.add(page.next_after);
      after = page.next_after;
    }
    throw new Error(`GUI capability inventory exceeds ${capabilityInventoryMaxPages} pages.`);
  }

  async listCapabilityPage(subject?: string, after?: string): Promise<CapabilityPageResponse> {
    const query = new URLSearchParams({ mode: "page" });
    if (subject) query.set("subject", subject);
    if (after) query.set("after", after);
    const payload = await this.request<unknown>("GET", `/api/capabilities?${query.toString()}`);
    return capabilityPageResponse(payload);
  }

  async inspectCapability(capabilityId: string): Promise<CapabilitySummary> {
    return this.request("GET", `/api/capabilities/${encodeURIComponent(capabilityId)}`);
  }

  async grantCapability(input: CapabilityMutationInput, confirmed: boolean) {
    return this.request<CapabilitySummary>("POST", "/api/capabilities/grant", {
      ...input,
      confirmed
    });
  }

  async delegateCapability(input: CapabilityDelegationInput, confirmed: boolean) {
    return this.request<CapabilitySummary>("POST", "/api/capabilities/delegate", {
      ...input,
      confirmed
    });
  }

  async revokeCapability(capabilityId: string, reason: string, confirmed: boolean, actor?: string) {
    return this.request<CapabilitySummary>(
      "POST",
      `/api/capabilities/${encodeURIComponent(capabilityId)}/revoke`,
      { reason, confirmed, ...(actor ? { actor } : {}) }
    );
  }

  async explainCapability(subject: string, resource: string, right: string) {
    return this.request<Record<string, unknown>>("POST", "/api/capabilities/explain", {
      subject,
      resource,
      right
    }, { timeoutMs: defaultReadRequestTimeoutMs });
  }

  async inspectJsonRpcEndpoint(endpointId: string): Promise<JsonRpcEndpointSummary> {
    return this.request("GET", `/api/jsonrpc/${encodeURIComponent(endpointId)}`);
  }

  async registerJsonRpcEndpoint(manifestText: string, confirmed: boolean, replace = false, actor?: string) {
    return this.request<JsonRpcEndpointSummary>("POST", "/api/jsonrpc/register", {
      manifest_text: manifestText,
      confirmed,
      replace,
      ...(actor ? { actor } : {})
    });
  }

  async callJsonRpc(endpointId: string, pid: string, methodId: string, params: unknown, confirmed: boolean) {
    return this.request<unknown>("POST", `/api/jsonrpc/${encodeURIComponent(endpointId)}/call`, {
      pid,
      method_id: methodId,
      params,
      confirmed
    });
  }

  async inspectMcpServer(serverId: string): Promise<McpServerSummary> {
    const value = await this.request<unknown>("GET", `/api/mcp/${encodeURIComponent(serverId)}`);
    assertMcpServerSummary(value);
    if (value.server_id !== serverId) throw new Error("GUI MCP server identity does not match the requested server id.");
    return value;
  }

  async listMcpTools(serverId: string, refresh = false): Promise<McpToolListResult> {
    const value = await this.request<unknown>(
      "GET",
      `/api/mcp/${encodeURIComponent(serverId)}/tools${refresh ? "?refresh=true" : ""}`
    );
    assertMcpToolListResult(value);
    if (value.server_id !== serverId) throw new Error("GUI MCP tool list identity does not match the requested server id.");
    return value;
  }

  async discoverMcpServer(serverId: string, actor?: string): Promise<McpDiscoveryResult> {
    const value = await this.request<unknown>("POST", `/api/mcp/${encodeURIComponent(serverId)}/discover`, {
      ...(actor ? { actor } : {})
    });
    assertMcpDiscoveryResult(value);
    if (value.server_id !== serverId) throw new Error("GUI MCP discovery identity does not match the requested server id.");
    return value;
  }

  async registerMcpServer(manifestText: string, confirmed: boolean, replace = false, actor?: string): Promise<McpServerSummary> {
    const value = await this.request<unknown>("POST", "/api/mcp/register", {
      manifest_text: manifestText,
      confirmed,
      replace,
      ...(actor ? { actor } : {})
    });
    assertMcpServerSummary(value);
    return value;
  }

  async callMcpTool(serverId: string, pid: string, toolId: string, args: Record<string, unknown>, confirmed: boolean): Promise<McpCallResult> {
    const value = await this.request<unknown>("POST", `/api/mcp/${encodeURIComponent(serverId)}/call`, {
      pid,
      tool_id: toolId,
      arguments: args,
      confirmed
    });
    assertMcpCallResult(value);
    if (value.server_id !== serverId || value.tool_id !== toolId) {
      throw new Error("GUI MCP call identity does not match the requested tool.");
    }
    return value;
  }

  async inspectModule(moduleId: string): Promise<ModuleSummary> {
    return this.request("GET", `/api/modules/${encodeURIComponent(moduleId)}`);
  }

  async commitCheckpointToImage({
    checkpointId,
    imageId,
    name,
    version,
    confirmed,
    replace = false,
    actor
  }: {
    checkpointId: string;
    imageId: string;
    name: string;
    version: string;
    confirmed: boolean;
    replace?: boolean;
    actor?: string;
  }) {
    return this.request<ImageMutationResult>("POST", "/api/images/commit", {
      checkpoint_id: checkpointId,
      image_id: imageId,
      name,
      version,
      confirmed,
      replace,
      ...(actor ? { actor } : {})
    });
  }

  async setAutoRun(enabled: boolean) {
    const status = await this.request<unknown>("POST", "/api/scheduler/auto", { enabled });
    assertSchedulerStatus(status);
    return status;
  }

  async pauseScheduler() {
    const status = await this.request<unknown>("POST", "/api/scheduler/pause", {});
    assertSchedulerStatus(status);
    return status;
  }

  async spawn(
    goal: string,
    image: string,
    maxQuanta: OptionalQuanta,
    autoRun: boolean,
    options: {
      authorityManifest?: Record<string, unknown>;
      llmProfile?: string;
      workingDirectory?: string;
    } = {}
  ) {
    const workingDirectory = options.workingDirectory?.trim();
    return this.request(
      "POST",
      "/api/processes",
      withOptionalQuanta({
        goal,
        image,
        auto_run: autoRun,
        ...(options.llmProfile ? { llm_profile: options.llmProfile } : {}),
        ...(workingDirectory ? { working_directory: workingDirectory } : {}),
        ...(options.authorityManifest ? { authority_manifest: options.authorityManifest } : {})
      }, maxQuanta)
    );
  }

  async run(pid: string, maxQuanta: OptionalQuanta) {
    return this.request("POST", `/api/processes/${encodeURIComponent(pid)}/run`, withOptionalQuanta({}, maxQuanta));
  }

  async startObjectTask({
    pid,
    ownerOid,
    ownerName,
    namespace,
    tool,
    args = {},
    notifyPid,
    notifyKind,
    notifyChannel,
    inheritCapabilities = [],
    grantResultToNotify = false,
    ownerWatch = false,
    watchEvents = [],
    watchChannel,
    watchKind
  }: {
    pid: string;
    ownerOid?: string;
    ownerName?: string;
    namespace?: string;
    tool: string;
    args?: Record<string, unknown>;
    notifyPid?: string;
    notifyKind?: "normal" | "interrupt";
    notifyChannel?: string;
    inheritCapabilities?: Record<string, unknown>[];
    grantResultToNotify?: boolean;
    ownerWatch?: boolean;
    watchEvents?: string[];
    watchChannel?: string;
    watchKind?: "normal" | "interrupt";
  }) {
    return this.request<ObjectTask>("POST", "/api/object-tasks/start", {
      pid,
      ...(ownerOid ? { owner_oid: ownerOid } : {}),
      ...(ownerName ? { owner_name: ownerName } : {}),
      ...(namespace ? { namespace } : {}),
      tool,
      args,
      ...(notifyPid ? { notify_pid: notifyPid } : {}),
      ...(notifyKind ? { notify_kind: notifyKind } : {}),
      ...(notifyChannel ? { notify_channel: notifyChannel } : {}),
      inherit_capabilities: inheritCapabilities,
      grant_result_to_notify: grantResultToNotify,
      owner_watch: ownerWatch,
      ...(watchEvents.length ? { watch_events: watchEvents } : {}),
      ...(watchChannel ? { watch_channel: watchChannel } : {}),
      ...(watchKind ? { watch_kind: watchKind } : {})
    });
  }

  async getObjectTask(taskId: string, pid?: string) {
    const query = pid ? `?pid=${encodeURIComponent(pid)}` : "";
    return this.request<ObjectTask>("GET", `/api/object-tasks/${encodeURIComponent(taskId)}${query}`);
  }

  async cancelObjectTask(taskId: string, pid: string, reason?: string) {
    return this.request<ObjectTask>("POST", `/api/object-tasks/${encodeURIComponent(taskId)}/cancel`, {
      pid,
      ...(reason ? { reason } : {})
    });
  }

  async waitObjectTask(taskId: string, pid?: string, timeoutS?: number) {
    return this.request<ObjectTask>("POST", `/api/object-tasks/${encodeURIComponent(taskId)}/wait`, {
      ...(pid ? { pid } : {}),
      ...(timeoutS !== undefined ? { timeout_s: timeoutS } : {})
    }, { timeoutMs: objectTaskWaitDeadlineMs(timeoutS) });
  }

  async watchObjectTaskOwner({
    taskId,
    pid,
    enabled = true,
    watchEvents,
    watchChannel,
    watchKind
  }: {
    taskId: string;
    pid: string;
    enabled?: boolean;
    watchEvents?: string[];
    watchChannel?: string;
    watchKind?: "normal" | "interrupt";
  }) {
    return this.request<ObjectTask>("POST", `/api/object-tasks/${encodeURIComponent(taskId)}/watch-owner`, {
      pid,
      enabled,
      ...(watchEvents ? { watch_events: watchEvents } : {}),
      ...(watchChannel ? { watch_channel: watchChannel } : {}),
      ...(watchKind ? { watch_kind: watchKind } : {})
    });
  }

  async step(pid: string) {
    return this.request("POST", `/api/processes/${encodeURIComponent(pid)}/step`, {});
  }

  async pauseProcess(pid: string) {
    return this.request("POST", `/api/processes/${encodeURIComponent(pid)}/pause`, { reason: "paused from GUI" });
  }

  async resumeProcess(pid: string, autoRun: boolean) {
    return this.request("POST", `/api/processes/${encodeURIComponent(pid)}/resume`, { auto_run: autoRun });
  }

  async sendMessage(pid: string, body: string, kind: "message" | "interrupt", autoRun: boolean, maxQuanta: OptionalQuanta) {
    return this.request("POST", `/api/processes/${encodeURIComponent(pid)}/${kind}`, withOptionalQuanta({
      body,
      auto_run: autoRun,
      channel: "gui"
    }, maxQuanta));
  }

  async changeDirectory(pid: string, path: string) {
    return this.request("POST", `/api/processes/${encodeURIComponent(pid)}/cd`, { path });
  }

  async submitAgentRating(pid: string, score: number, comment: string) {
    return this.request<AgentRating>("POST", `/api/processes/${encodeURIComponent(pid)}/rating`, { score, comment });
  }

  async execProcess(pid: string, image: string, goal: string, confirmed: boolean, autoRun: boolean, maxQuanta: OptionalQuanta, llmProfile?: string) {
    return this.request("POST", `/api/processes/${encodeURIComponent(pid)}/exec`, withOptionalQuanta({
      image,
      goal,
      confirmed,
      auto_run: autoRun,
      ...(llmProfile ? { llm_profile: llmProfile } : {})
    }, maxQuanta));
  }

  async exitProcess(pid: string, message: string, failed: boolean, confirmed: boolean) {
    return this.request("POST", `/api/processes/${encodeURIComponent(pid)}/exit`, { message, failed, confirmed });
  }

  async respondHumanRequest(requestId: string, response: HumanResponseInput, autoRun: boolean, maxQuanta: OptionalQuanta) {
    const { kind: _kind, ...decision } = response;
    return this.request("POST", `/api/human-requests/${encodeURIComponent(requestId)}/respond`, withOptionalQuanta({
      ...decision,
      auto_run: autoRun
    }, maxQuanta));
  }

  async request<T = unknown>(method: string, path: string, body?: JsonBody, options: RequestOptions = {}): Promise<T> {
    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.connection.token}`,
      Accept: "application/json"
    };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const timeoutMs = options.timeoutMs === undefined
      ? isReadOnlyHttpMethod(method) ? defaultReadRequestTimeoutMs : null
      : options.timeoutMs;
    const abort = requestAbortContext(options.signal, timeoutMs);
    try {
      const response = await fetch(`${this.connection.url}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: abort.signal
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const message = payload?.error?.message ?? `HTTP ${response.status}`;
        throw new ApiError(message, response.status, payload);
      }
      return payload as T;
    } finally {
      abort.cleanup();
    }
  }

  async stream(
    onMessage: (message: SseMessage) => void,
    signal: AbortSignal,
    cursor = "0",
    onStatus?: (status: StreamConnectionStatus) => void
  ) {
    let nextCursor = cursor;
    onStatus?.("connecting");
    while (!signal.aborted) {
      try {
        nextCursor = await this.readStreamUntilClosed(
          onMessage,
          signal,
          nextCursor,
          onStatus,
          (cursor) => { nextCursor = cursor; }
        );
      } catch (error) {
        if (signal.aborted) return;
        if (error instanceof SseHttpError) {
          onStatus?.("failed");
          throw error;
        }
      }
      if (signal.aborted) return;
      onStatus?.("reconnecting");
      await waitForReconnect(signal);
    }
  }

  private async readStreamUntilClosed(
    onMessage: (message: SseMessage) => void,
    signal: AbortSignal,
    cursor: string,
    onStatus?: (status: StreamConnectionStatus) => void,
    onCursor?: (cursor: string) => void
  ) {
    const response = await fetch(`${this.connection.url}/api/events/stream?cursor=${encodeURIComponent(cursor)}`, {
      headers: { Authorization: `Bearer ${this.connection.token}` },
      signal
    });
    if (!response.ok || !response.body) throw new SseHttpError(response.status);
    onStatus?.("connected");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let nextCursor = cursor;
    try {
      while (!signal.aborted) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary = findSseBoundary(buffer);
        while (boundary) {
          const frame = buffer.slice(0, boundary.index);
          buffer = buffer.slice(boundary.index + boundary.length);
          const parsed = parseSseFrame(frame);
          if (parsed) {
            if (parsed.id) {
              nextCursor = parsed.id;
              onCursor?.(nextCursor);
            }
            onMessage(parsed);
          }
          boundary = findSseBoundary(buffer);
        }
      }
      return nextCursor;
    } finally {
      reader.releaseLock();
    }
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly payload: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class TaskRunMutationError extends ApiError {
  constructor(
    readonly original: ApiError,
    readonly currentSummary: TaskRunSummary | null
  ) {
    super(original.message, original.status, original.payload);
    this.name = "TaskRunMutationError";
  }
}

export function isTaskRunConflict(error: unknown): error is ApiError {
  if (!(error instanceof ApiError) || error.status !== 409) return false;
  const envelope = taskRunErrorEnvelope(error);
  return typeof envelope?.code === "string" && envelope.code.startsWith("task_run_")
    && envelope.code.endsWith("conflict");
}

export function isUnadmittedTaskRunRevisionConflict(error: unknown): boolean {
  if (!isTaskRunConflict(error)) return false;
  const envelope = taskRunErrorEnvelope(error);
  return envelope?.code === "task_run_revision_conflict"
    && envelope.command_admitted === false
    && taskRunConflictSummary(error) !== null;
}

export function taskRunConflictSummary(error: unknown): TaskRunSummary | null {
  if (error instanceof TaskRunMutationError) return error.currentSummary;
  if (!isTaskRunConflict(error)) return null;
  const envelope = taskRunErrorEnvelope(error);
  try {
    return taskRunSummary(envelope?.current_summary);
  } catch {
    return null;
  }
}

function taskRunErrorEnvelope(error: ApiError): Record<string, unknown> | null {
  const payload = error.payload;
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null;
  const envelope = (payload as { error?: unknown }).error;
  return envelope && typeof envelope === "object" && !Array.isArray(envelope)
    ? envelope as Record<string, unknown>
    : null;
}

function withOptionalQuanta(body: JsonBody, maxQuanta: OptionalQuanta): JsonBody {
  return maxQuanta === null ? body : { ...body, max_quanta: maxQuanta };
}

export function parseSseFrame(frame: string): SseMessage | null {
  const lines = frame.split(/\r\n|\n|\r/).filter((line) => line.trim() && !line.startsWith(":"));
  if (lines.length === 0) return null;
  let id = "";
  let event = "message";
  const data: string[] = [];
  for (const line of lines) {
    if (line.startsWith("id:")) id = line.slice(3).trim();
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (data.length === 0) return { id, event, data: null };
  try {
    return { id, event, data: JSON.parse(data.join("\n")) };
  } catch {
    return null;
  }
}

function findSseBoundary(buffer: string): { index: number; length: number } | null {
  const match = /\r\n\r\n|\n\n|\r\r/.exec(buffer);
  return match ? { index: match.index, length: match[0].length } : null;
}

function waitForReconnect(signal: AbortSignal, delayMs = 500): Promise<void> {
  if (signal.aborted) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(done, delayMs);
    function done() {
      clearTimeout(timer);
      signal.removeEventListener("abort", done);
      resolve();
    }
    signal.addEventListener("abort", done, { once: true });
  });
}

class SseHttpError extends Error {
  constructor(readonly status: number) {
    super(`SSE connection failed: ${status}`);
  }
}

export function objectTaskWaitDeadlineMs(timeoutS?: number): number | null {
  if (timeoutS === undefined) return null;
  if (!Number.isFinite(timeoutS) || timeoutS < 0) {
    throw new Error("GUI object-task wait timeout must be a finite non-negative number.");
  }
  const deadlineMs = Math.ceil(timeoutS * 1_000) + objectTaskWaitDeadlineMarginMs;
  if (!Number.isSafeInteger(deadlineMs)) {
    throw new Error("GUI object-task wait timeout exceeds the supported deadline range.");
  }
  return deadlineMs;
}

function taskRunCommand(expectedRevision: number, commandId: string): JsonBody {
  if (!Number.isSafeInteger(expectedRevision) || expectedRevision < 0) {
    throw new Error("GUI task-run expected revision must be a non-negative safe integer.");
  }
  return {
    expected_revision: expectedRevision,
    command_id: requiredCommandId(commandId, "commandId")
  };
}

function requiredCommandId(value: string, name: string): string {
  const selected = value.trim();
  if (!selected) throw new Error(`GUI task-run ${name} must be non-empty.`);
  return selected;
}

function taskRunSummary(value: unknown): TaskRunSummary {
  assertTaskRunSummary(value);
  return value;
}

function taskRunLedgerPage(value: unknown): TaskRunLedgerPage {
  const page = pagedItems(value, "task run ledger");
  for (const item of page.items) {
    if (!isJsonObject(item)
        || item.schema_version !== 1
        || typeof item.item_id !== "string"
        || typeof item.run_id !== "string"
        || !Number.isSafeInteger(item.seq)
        || Number(item.seq) < 0
        || !["requirement", "process", "llm_turn", "tool_call", "human_wait", "message_wait", "checkpoint", "effect", "status_transition"].includes(String(item.kind))
        || typeof item.status !== "string"
        || typeof item.label !== "string"
        || typeof item.occurred_at !== "string"
        || !isJsonObject(item.metadata)) {
      throw new Error("GUI task run ledger page contains a malformed item.");
    }
  }
  return page as TaskRunLedgerPage;
}

function taskRunHumanRequestPage(value: unknown): TaskRunHumanRequestPage {
  const page = pagedItems(value, "task run human request");
  if (!isJsonObject(value) || typeof value.presentation_truncated !== "boolean") {
    throw new Error("GUI task run human request page truncation state is malformed.");
  }
  for (const item of page.items) humanRequest(item);
  return { ...page, presentation_truncated: value.presentation_truncated } as TaskRunHumanRequestPage;
}

function humanRequest(value: unknown): HumanRequest {
  if (!isJsonObject(value)
      || typeof value.request_id !== "string" || !value.request_id
      || typeof value.pid !== "string" || !value.pid
      || typeof value.human !== "string" || !value.human
      || !isJsonObject(value.payload)
      || typeof value.status !== "string" || !value.status
      || !(value.decision === null || isJsonObject(value.decision))
      || typeof value.blocking !== "boolean"
      || typeof value.created_at !== "string" || !value.created_at
      || typeof value.updated_at !== "string" || !value.updated_at
      || (value.release_request_id !== undefined && typeof value.release_request_id !== "string")
      || (value.release_for_request_id !== undefined && typeof value.release_for_request_id !== "string")) {
    throw new Error("GUI human request response is malformed.");
  }
  return value as HumanRequest;
}

function pagedItems(value: unknown, label: string): { items: unknown[]; next_cursor: string | null; has_more: boolean } {
  if (!isJsonObject(value) || !Array.isArray(value.items) || typeof value.has_more !== "boolean") {
    throw new Error(`GUI ${label} page response is malformed.`);
  }
  const cursor = value.next_cursor;
  if (!(cursor === null || typeof cursor === "string")) {
    throw new Error(`GUI ${label} page cursor is malformed.`);
  }
  return { items: value.items, next_cursor: cursor, has_more: value.has_more };
}

function capabilityPageResponse(payload: unknown): CapabilityPageResponse {
  if (!isJsonObject(payload) || !Array.isArray(payload.items) || typeof payload.has_more !== "boolean") {
    throw new Error("GUI capability page response is malformed.");
  }
  const nextAfter = payload.next_after;
  if (nextAfter !== null && typeof nextAfter !== "string") {
    throw new Error("GUI capability page cursor is malformed.");
  }
  const items = payload.items.map((item) => {
    if (
      !isJsonObject(item)
      || typeof item.cap_id !== "string"
      || !item.cap_id
      || typeof item.subject !== "string"
      || typeof item.resource !== "string"
      || !Array.isArray(item.rights)
      || item.rights.some((right) => typeof right !== "string")
    ) {
      throw new Error("GUI capability page contains a malformed capability.");
    }
    return item as CapabilitySummary;
  });
  return { items, next_after: nextAfter, has_more: payload.has_more };
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isReadOnlyHttpMethod(method: string): boolean {
  return ["GET", "HEAD", "OPTIONS"].includes(method.toUpperCase());
}

function requestAbortContext(external: AbortSignal | undefined, timeoutMs: number | null): {
  signal: AbortSignal;
  cleanup(): void;
} {
  if (timeoutMs !== null && (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0)) {
    throw new Error("GUI request timeout must be a positive safe integer.");
  }
  const controller = new AbortController();
  const abortFromExternal = () => controller.abort(external?.reason);
  if (external?.aborted) abortFromExternal();
  else external?.addEventListener("abort", abortFromExternal, { once: true });
  const timer = timeoutMs === null ? null : setTimeout(() => {
    controller.abort(new DOMException(`GUI request timed out after ${timeoutMs}ms`, "TimeoutError"));
  }, timeoutMs);
  return {
    signal: controller.signal,
    cleanup() {
      if (timer !== null) clearTimeout(timer);
      external?.removeEventListener("abort", abortFromExternal);
    }
  };
}
