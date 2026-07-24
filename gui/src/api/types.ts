export type SchedulerStatus = {
  auto_run: boolean;
  running: boolean;
  paused: boolean;
  task_id: string | null;
  reason: string | null;
  last_result: unknown[];
  last_error: string | null;
  started_at: number | null;
  finished_at: number | null;
  default_max_quanta: number | null;
};

export type ProcessWaitState =
  | { schema_version: 1; kind: "child"; child_pid: string }
  | { schema_version: 1; kind: "message"; filters: Record<string, unknown> }
  | { schema_version: 1; kind: "human"; request_ids: string[] }
  | { schema_version: 1; kind: "tool"; operation_id: string }
  | { schema_version: 1; kind: "paused"; reason_oid: string | null }
  | { schema_version: 1; kind: "host_resume"; reason_oid: string };

export type ProcessOutcome =
  | { schema_version: 1; kind: "exited"; result_oid: string | null }
  | { schema_version: 1; kind: "failed"; result_oid: string | null; code: string | null }
  | { schema_version: 1; kind: "killed"; reason_oid: string | null; code: string | null };

export type RuntimeProcess = {
  pid: string;
  parent_pid: string | null;
  image_id: string;
  llm_profile_id: string;
  status: string;
  goal_oid: string | null;
  checkpoint_head: string | null;
  working_directory: string;
  status_message: string | null;
  wait_state: ProcessWaitState | null;
  outcome: ProcessOutcome | null;
  state_generation: number;
  created_at?: string;
  updated_at?: string;
  loaded_skills: Record<string, unknown>;
  tool_table: Record<string, string>;
  capabilities: string[];
  terminal: boolean;
  unread_message_count: number;
  interrupt_count: number;
  messages: ProcessMessage[];
  llm_call_count: number;
  token_total: number;
  resource_budget?: Record<string, unknown>;
  resource_usage?: Record<string, unknown>;
  resource_remaining?: Record<string, unknown>;
  rating: AgentRating | null;
};

export type CheckpointProcess = {
  pid: string;
  parent_pid: string | null;
  image_id: string;
  status: string;
  working_directory: string;
  goal_oid: string | null;
  wait_state: ProcessWaitState | null;
  outcome: ProcessOutcome | null;
  state_generation: number;
};

export type CheckpointInspectResult = {
  checkpoint: Record<string, unknown> & { checkpoint_id: string; pid: string };
  snapshot_version: number | null;
  subtree_pids: string[];
  modules: Record<string, unknown>[];
  counts: Record<string, number>;
  processes: CheckpointProcess[];
};

export type CheckpointSummary = Record<string, unknown> & {
  checkpoint_id: string;
  pid: string;
  parent_checkpoint_id?: string | null;
  created_at?: string;
  reason?: string;
};

export type CheckpointDiffResult = Record<string, unknown>;

export type CapabilitySummary = Record<string, unknown> & {
  cap_id: string;
  subject: string;
  resource: string;
  rights: string[];
  status?: string;
  effect?: string;
};

export type CapabilityMutationInput = {
  subject: string;
  resource: string;
  rights: string[];
  actor?: string;
};

export type CapabilityDelegationInput = {
  parent: string;
  child: string;
  resource: string;
  rights: string[];
  actor?: string;
};

export type SkillSummary = Record<string, unknown> & {
  skill_id: string;
  name?: string;
  description?: string;
  source?: string;
};

export type JsonRpcEndpointSummary = Record<string, unknown> & {
  endpoint_id: string;
  name?: string;
  description?: string;
};

export type McpServerSummary = Record<string, unknown> & {
  server_id: string;
  name?: string;
  description?: string;
};

export type ModuleSummary = Record<string, unknown> & {
  module_id: string;
  name?: string;
  version?: string;
};

export type AgentRating = {
  rating_id: string;
  pid: string;
  score: number;
  comment: string;
  rater: string;
  source: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
};

export type ProcessMessage = {
  message_id: string;
  sender: string;
  recipient_pid: string;
  kind: "normal" | "interrupt";
  subject: string;
  body: string;
  channel: string;
  status: string;
  created_at: string;
  payload: Record<string, unknown>;
};

export type HumanRequestPayload = Record<string, unknown> & {
  type?: string;
  question?: string;
  reason?: string;
  context?: Record<string, unknown>;
  release_required?: boolean;
  release_request_id?: string | null;
};

export type DataReleaseApprovalContext = Record<string, unknown> & {
  sink: string;
  sensitivity: string;
  tenant: string | null;
  principal: string | null;
  payload_bytes: number;
  payload_sha256: string;
  source_count: number;
  operation: string;
};

export type DataReleaseApprovalPayload = HumanRequestPayload & {
  type: "data_release_approval";
  context: DataReleaseApprovalContext;
};

export type HumanRequest = {
  request_id: string;
  pid: string;
  human: string;
  payload: HumanRequestPayload;
  status: string;
  decision: Record<string, unknown> | null;
  blocking: boolean;
  created_at: string;
  updated_at: string;
  release_request_id?: string;
  release_for_request_id?: string;
};

export type HumanPermissionPolicy = "always_allow" | "ask_each_time" | "always_deny";

export type HumanResponseInput =
  | {
      kind: "permission";
      approved: true;
      decision: { policy: Exclude<HumanPermissionPolicy, "always_deny"> };
    }
  | {
      kind: "permission";
      approved: false;
      decision: { policy: Exclude<HumanPermissionPolicy, "always_allow"> };
    }
  | { kind: "question"; approved: true; answer: string }
  | { kind: "question"; approved: false }
  | { kind: "approval"; approved: boolean };

export type AuditRecord = {
  record_id: string;
  timestamp: string;
  actor: string;
  action: string;
  target: string | null;
  decision: Record<string, unknown> | null;
  capability_refs: string[];
};

export type RuntimeEvent = {
  event_id: string;
  type: string;
  source: string;
  target: string | null;
  payload: Record<string, unknown>;
  priority: string;
  created_at: string;
};

export type LlmCall = {
  call_id: string;
  pid: string | null;
  image_id: string | null;
  purpose: string;
  status: string;
  api: string | null;
  model: string | null;
  request_options: Record<string, unknown>;
  response_content: string;
  tool_calls: unknown[];
  usage: Record<string, unknown>;
  reasoning: unknown;
  error: string | null;
  created_at: string;
  completed_at: string | null;
};

export type ToolSummary = {
  tool_id: string;
  name: string;
  scope: string;
  description: string;
  tags: string[];
  policy: Record<string, unknown>;
  ephemeral: boolean;
};

export type LLMProfileSummary = {
  profile_id: string;
  model: string | null;
  base_url: string | null;
  api_key_env: string;
  api_key_env_present: boolean;
  api_mode: "auto" | "responses" | "chat" | null;
  timeout_s: number | null;
  max_retries: number | null;
  store: boolean | null;
  reasoning_effort: string | null;
  verbosity: "low" | "medium" | "high" | null;
  safety_identifier_env: string | null;
  prompt_cache_retention: "in_memory" | "24h" | null;
  responses_previous_response_id: boolean | null;
  parallel_tool_calls: boolean | null;
  auto_wait_on_empty_tool_calls: boolean | null;
  fallback_json_actions: boolean | null;
  temperature: number | null;
  max_tokens: number | null;
  context_window_tokens: number | null;
  allow_custom_base_url: boolean;
  source: "config" | "user";
  editable: boolean;
  is_default: boolean;
};

export type LLMProfileInput = {
  profile_id?: string;
  model: string;
  base_url?: string | null;
  api_key_env: string;
  api_mode?: "auto" | "responses" | "chat" | null;
  timeout_s?: number | null;
  max_retries?: number | null;
  store?: boolean | null;
  reasoning_effort?: string | null;
  verbosity?: "low" | "medium" | "high" | null;
  safety_identifier_env?: string | null;
  prompt_cache_retention?: "in_memory" | "24h" | null;
  responses_previous_response_id?: boolean | null;
  parallel_tool_calls?: boolean | null;
  auto_wait_on_empty_tool_calls?: boolean | null;
  fallback_json_actions?: boolean | null;
  temperature?: number | null;
  max_tokens?: number | null;
  context_window_tokens?: number | null;
  allow_custom_base_url?: boolean | null;
};

export type WorkflowRunResult = {
  pid: string;
  image: string;
  tool: string;
  ok: boolean;
  status: string;
  call_id: string | null;
  tool_id: string | null;
  result_oid: string | null;
  payload: unknown;
  error: string | null;
  waiting_human: boolean;
  request_id: string | null;
  waiting_process: boolean;
  child_pid: string | null;
  waiting_message: boolean;
  filters: Record<string, unknown> | null;
};

export type ObjectTask = {
  task_id: string;
  owner_oid: string;
  creator_pid: string;
  runner_pid: string | null;
  tool: string;
  tool_id: string | null;
  status: string;
  notification: Record<string, unknown>;
  owner_watch: Record<string, unknown>;
  result_oid: string | null;
  error: string | null;
  wait: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
};

export type ImageSummary = {
  image_id: string;
  name: string;
  version: string;
  boot_kind: string;
  default_tools: string[];
  default_skills: string[];
  required_capabilities_count: number;
  required_modules_count: number;
  [key: string]: unknown;
};

export type ImageInspectResult = {
  image: {
    image_id: string;
    name: string;
    version: string;
    default_tools: string[];
    default_skills: string[];
    required_capabilities: Record<string, unknown>[];
    required_modules: Record<string, unknown>[];
    boot: Record<string, unknown>;
    metadata: Record<string, unknown>;
    [key: string]: unknown;
  };
  registry: Record<string, unknown>;
  artifact: Record<string, unknown> | null;
};

export type ImageMutationResult = {
  image_id: string;
  name: string;
  version: string;
  replaced: boolean;
  boot: Record<string, unknown>;
  default_tools?: string[];
  default_skills?: string[];
  package_sha256?: string;
  package_jit_tools?: string[];
  required_capabilities_count: number;
  required_modules_count: number;
  source?: string | null;
};

export type RuntimeSnapshot = {
  db: string;
  scheduler: SchedulerStatus;
  processes: RuntimeProcess[];
  human_requests: HumanRequest[];
  events: RuntimeEvent[];
  audit: AuditRecord[];
  llm_calls: LlmCall[];
  object_tasks: ObjectTask[];
  tools: ToolSummary[];
  llm_profiles: LLMProfileSummary[];
  images: ImageSummary[];
  skills: SkillSummary[];
  jsonrpc_endpoints: JsonRpcEndpointSummary[];
  mcp_servers: McpServerSummary[];
  modules: ModuleSummary[];
  _truncated?: Record<string, unknown>;
};

export type OperationSummary = {
  operation_id: string;
  root_operation_id: string;
  parent_operation_id: string | null;
  kind: "llm_request" | "tool_call" | "syscall" | "primitive" | "runtime";
  name: string;
  actor: string;
  pid: string | null;
  state: "running" | "waiting" | "terminal";
  outcome: "pending" | "succeeded" | "denied" | "failed" | "interrupted" | "unknown";
  started_at: string;
  updated_at: string;
  completed_at: string | null;
};

export type OperationRecord = OperationSummary & {
  expected_roles: string[];
  metadata: Record<string, unknown>;
};

export type OperationEvidence = {
  evidence_type: string;
  evidence_id: string;
  roles: string[];
  occurred_at: string | null;
  metadata: unknown;
  data: unknown;
};

export type OperationListResponse = {
  schema_version: number;
  pid: string;
  roots_only: boolean;
  operations: OperationSummary[];
  presentation_truncated: boolean;
  next_cursor: string | null;
};

export type ExplainOperationResponse = {
  schema_version: number;
  lookup: { kind: string; id: string };
  selected_operation_id: string;
  root: OperationSummary;
  summary: {
    headline: string;
    outcome: OperationSummary["outcome"];
    operation_count: number;
    authorization: unknown[];
    human: unknown[];
    external_effects: unknown[];
    resource_charge_evidence_count: number;
    resource_charge_count: number;
    resource_consumption: unknown[];
    context: unknown[];
  };
  operations: OperationRecord[];
  edges: Array<{ from: string; to: string; relation: string }>;
  evidence: OperationEvidence[];
  evidence_complete: boolean;
  missing_evidence: Array<{ operation_id: string; role: string }>;
  uncertainties: Array<{ operation_id?: string; evidence_id?: string; reason: string }>;
  presentation_truncated: boolean;
  next_cursor: string | null;
};

export type GuiConnection = {
  url: string;
  token: string;
  db: string;
};

export type ImagePackageFileValue = string | { base64: string };

export type ImagePackageFile = {
  name: string;
  manifest: string;
  files: Record<string, ImagePackageFileValue>;
};

export type SseMessage = {
  id: string;
  event: string;
  data: unknown;
};

export type StreamConnectionStatus = "connecting" | "connected" | "reconnecting" | "failed";

export type RuntimeHealth = {
  ok: boolean;
  db: string;
  scheduler: SchedulerStatus;
  process_count: number | null;
  runtime_busy: boolean;
};

const snapshotCollections = [
  "processes",
  "human_requests",
  "events",
  "audit",
  "llm_calls",
  "object_tasks",
  "tools",
  "llm_profiles",
  "images",
  "skills",
  "jsonrpc_endpoints",
  "mcp_servers",
  "modules"
] as const;

/** Fail closed on a malformed same-build response before React consumes it. */
export function assertRuntimeSnapshot(value: unknown): asserts value is RuntimeSnapshot {
  if (!isRecord(value)) throw new Error("GUI snapshot must be a JSON object.");
  if (typeof value.db !== "string") throw new Error("GUI snapshot is missing db.");
  if (!isRecord(value.scheduler)) throw new Error("GUI snapshot is missing scheduler state.");
  if (
    typeof value.scheduler.auto_run !== "boolean"
    || typeof value.scheduler.running !== "boolean"
    || typeof value.scheduler.paused !== "boolean"
  ) {
    throw new Error("GUI snapshot scheduler state is malformed.");
  }
  for (const key of snapshotCollections) {
    if (!Array.isArray(value[key])) throw new Error(`GUI snapshot collection is malformed: ${key}.`);
  }
  const processes = value.processes;
  if (!Array.isArray(processes)) throw new Error("GUI snapshot collection is malformed: processes.");
  for (const process of processes) {
    if (!isRecord(process) || typeof process.pid !== "string" || !process.pid) {
      throw new Error("GUI snapshot contains a process without a valid pid.");
    }
  }
}

/** Validate snapshots delivered inside an SSE event before replacing visible state. */
export function runtimeSnapshotFromSseData(value: unknown): RuntimeSnapshot {
  if (!isRecord(value) || !("snapshot" in value)) {
    throw new Error("GUI snapshot event is missing its snapshot payload.");
  }
  const snapshot = value.snapshot;
  assertRuntimeSnapshot(snapshot);
  return snapshot;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
