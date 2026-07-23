import type { AgentRating, CheckpointInspectResult, ExplainOperationResponse, GuiConnection, HumanResponseInput, ImageInspectResult, ImageMutationResult, ImagePackageFile, ImageSummary, LLMProfileInput, LLMProfileSummary, ObjectTask, OperationListResponse, RuntimeSnapshot, SseMessage, WorkflowRunResult } from "./types";
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
    return this.request<RuntimeSnapshot>("GET", "/api/snapshot");
  }

  async health() {
    return this.request("GET", "/api/health");
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
    options: { llmProfile?: string; workingDirectory?: string } = {}
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
        ...(workingDirectory ? { working_directory: workingDirectory } : {})
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
    const response = await fetch(`${this.connection.url}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${this.connection.token}`,
        "Content-Type": "application/json"
      },
      body: body === undefined ? undefined : JSON.stringify(body)
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const message = payload?.error?.message ?? `HTTP ${response.status}`;
      const error = new Error(message) as Error & { status?: number; payload?: unknown };
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload as T;
  }

  async stream(onMessage: (message: SseMessage) => void, signal: AbortSignal, cursor = "0") {
    let nextCursor = cursor;
    while (!signal.aborted) {
      try {
        nextCursor = await this.readStreamUntilClosed(onMessage, signal, nextCursor);
      } catch (error) {
        if (signal.aborted) return;
        if (error instanceof SseHttpError) throw error;
      }
      await waitForReconnect(signal);
    }
  }

  private async readStreamUntilClosed(onMessage: (message: SseMessage) => void, signal: AbortSignal, cursor: string) {
    const response = await fetch(`${this.connection.url}/api/events/stream?cursor=${encodeURIComponent(cursor)}`, {
      headers: { Authorization: `Bearer ${this.connection.token}` },
      signal
    });
    if (!response.ok || !response.body) throw new SseHttpError(response.status);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let nextCursor = cursor;
    try {
      while (!signal.aborted) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const parsed = parseSseFrame(frame);
          if (parsed) {
            if (parsed.id) nextCursor = parsed.id;
            onMessage(parsed);
          }
          boundary = buffer.indexOf("\n\n");
        }
      }
      return nextCursor;
    } finally {
      reader.releaseLock();
    }
  }
}

function withOptionalQuanta(body: JsonBody, maxQuanta: OptionalQuanta): JsonBody {
  return maxQuanta === null ? body : { ...body, max_quanta: maxQuanta };
}

export function parseSseFrame(frame: string): SseMessage | null {
  const lines = frame.split(/\r?\n/).filter((line) => line.trim() && !line.startsWith(":"));
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
