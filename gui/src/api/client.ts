import type { AgentRating, CapabilityDelegationInput, CapabilityMutationInput, CapabilitySummary, CheckpointDiffResult, CheckpointInspectResult, CheckpointSummary, ExplainOperationResponse, GuiConnection, HumanResponseInput, ImageInspectResult, ImageMutationResult, ImagePackageFile, ImageSummary, JsonRpcEndpointSummary, LLMProfileInput, LLMProfileSummary, McpServerSummary, ModuleSummary, ObjectTask, OperationListResponse, RuntimeHealth, RuntimeSnapshot, SkillSummary, SseMessage, StreamConnectionStatus, WorkflowRunResult } from "./types";
import { assertRuntimeSnapshot } from "./types";
import type { OptionalQuanta } from "../quanta";

type JsonBody = Record<string, unknown>;

export class LibOSClient {
  constructor(private connection: GuiConnection) {}

  get db() {
    return this.connection.db;
  }

  updateConnection(connection: GuiConnection) {
    this.connection = connection;
  }

  async snapshot(): Promise<RuntimeSnapshot> {
    const snapshot = await this.request<unknown>("GET", "/api/snapshot");
    assertRuntimeSnapshot(snapshot);
    return snapshot;
  }

  async health(): Promise<RuntimeHealth> {
    return this.request<RuntimeHealth>("GET", "/api/health");
  }

  async listOperations(pid: string, limit = 100, cursor?: string): Promise<OperationListResponse> {
    const query = new URLSearchParams({ pid, limit: String(limit) });
    if (cursor) query.set("cursor", cursor);
    return this.request<OperationListResponse>("GET", `/api/operations?${query.toString()}`);
  }

  async explainOperation(operationId: string, evidenceLimit = 200, cursor?: string): Promise<ExplainOperationResponse> {
    const query = new URLSearchParams({ evidence_limit: String(evidenceLimit) });
    if (cursor) query.set("cursor", cursor);
    return this.request<ExplainOperationResponse>(
      "GET",
      `/api/operations/${encodeURIComponent(operationId)}?${query.toString()}`
    );
  }

  async resolveOperation(kind: string, evidenceId: string): Promise<ExplainOperationResponse> {
    const query = new URLSearchParams({ kind, id: evidenceId });
    return this.request<ExplainOperationResponse>("GET", `/api/operations/resolve?${query.toString()}`);
  }

  async images(): Promise<ImageSummary[]> {
    return this.request<ImageSummary[]>("GET", "/api/images");
  }

  async llmProfiles(): Promise<LLMProfileSummary[]> {
    return this.request<LLMProfileSummary[]>("GET", "/api/llm-profiles");
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

  async listSkills(text?: string): Promise<SkillSummary[]> {
    const query = text ? `?text=${encodeURIComponent(text)}` : "";
    return this.request<SkillSummary[]>("GET", `/api/skills${query}`);
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

  async activateSkill(skillId: string, pid: string, confirmed: boolean, actor?: string) {
    return this.request<Record<string, unknown>>(
      "POST",
      `/api/skills/${encodeURIComponent(skillId)}/activate`,
      { pid, confirmed, ...(actor ? { actor } : {}) }
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
    const query = subject ? `?subject=${encodeURIComponent(subject)}` : "";
    return this.request<CapabilitySummary[]>("GET", `/api/capabilities${query}`);
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
    });
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
    return this.request("GET", `/api/mcp/${encodeURIComponent(serverId)}`);
  }

  async listMcpTools(serverId: string, refresh = false): Promise<Record<string, unknown>> {
    return this.request(
      "GET",
      `/api/mcp/${encodeURIComponent(serverId)}/tools${refresh ? "?refresh=true" : ""}`
    );
  }

  async registerMcpServer(manifestText: string, confirmed: boolean, replace = false, actor?: string) {
    return this.request<McpServerSummary>("POST", "/api/mcp/register", {
      manifest_text: manifestText,
      confirmed,
      replace,
      ...(actor ? { actor } : {})
    });
  }

  async callMcpTool(serverId: string, pid: string, toolId: string, args: Record<string, unknown>, confirmed: boolean) {
    return this.request<unknown>("POST", `/api/mcp/${encodeURIComponent(serverId)}/call`, {
      pid,
      tool_id: toolId,
      arguments: args,
      confirmed
    });
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
    return this.request("POST", "/api/scheduler/auto", { enabled });
  }

  async pauseScheduler() {
    return this.request("POST", "/api/scheduler/pause", {});
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

  async runWorkflow({
    tool,
    args = {},
    image,
    goal,
    workingDirectory,
    confirmed
  }: {
    tool: string;
    args?: Record<string, unknown>;
    image?: string;
    goal?: string;
    workingDirectory?: string;
    confirmed?: boolean;
  }) {
    return this.request<WorkflowRunResult>("POST", "/api/workflows/run", {
      tool,
      args,
      ...(image ? { image } : {}),
      ...(goal ? { goal } : {}),
      ...(workingDirectory ? { working_directory: workingDirectory } : {}),
      ...(confirmed !== undefined ? { confirmed } : {})
    });
  }

  async listObjectTasks(params: { pid?: string; ownerOid?: string; active?: boolean; limit?: number } = {}) {
    const query = new URLSearchParams();
    if (params.pid) query.set("pid", params.pid);
    if (params.ownerOid) query.set("owner_oid", params.ownerOid);
    if (params.active) query.set("active", "true");
    if (params.limit !== undefined) query.set("limit", String(params.limit));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return this.request<ObjectTask[]>("GET", `/api/object-tasks${suffix}`);
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
    });
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

  async getAgentRating(pid: string) {
    return this.request<AgentRating | null>("GET", `/api/processes/${encodeURIComponent(pid)}/rating`);
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

  async request<T = unknown>(method: string, path: string, body?: JsonBody): Promise<T> {
    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.connection.token}`,
      Accept: "application/json"
    };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const response = await fetch(`${this.connection.url}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body)
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const message = payload?.error?.message ?? `HTTP ${response.status}`;
      throw new ApiError(message, response.status, payload);
    }
    return payload as T;
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
        nextCursor = await this.readStreamUntilClosed(onMessage, signal, nextCursor, onStatus);
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
    onStatus?: (status: StreamConnectionStatus) => void
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
            if (parsed.id) nextCursor = parsed.id;
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
