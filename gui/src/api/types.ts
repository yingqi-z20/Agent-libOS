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

export function assertSchedulerStatus(value: unknown): asserts value is SchedulerStatus {
  if (!isRecord(value)) throw new Error("GUI scheduler status must be an object.");
  if (
    typeof value.auto_run !== "boolean"
    || typeof value.running !== "boolean"
    || typeof value.paused !== "boolean"
    || ("task_id" in value && !(value.task_id === null || typeof value.task_id === "string"))
    || ("reason" in value && !(value.reason === null || typeof value.reason === "string"))
    || ("last_result" in value && !Array.isArray(value.last_result))
    || ("last_error" in value && !(value.last_error === null || typeof value.last_error === "string"))
    || ("started_at" in value && !(value.started_at === null || typeof value.started_at === "number"))
    || ("finished_at" in value && !(value.finished_at === null || typeof value.finished_at === "number"))
    || ("default_max_quanta" in value && !(value.default_max_quanta === null || (Number.isSafeInteger(value.default_max_quanta) && Number(value.default_max_quanta) > 0)))
  ) {
    throw new Error("GUI scheduler status is malformed.");
  }
}

export type ProcessWaitState =
  | { schema_version: 1; kind: "child"; child_pid: string }
  | { schema_version: 1; kind: "message"; filters: Record<string, unknown> }
  | { schema_version: 1; kind: "human"; request_ids: string[] }
  | { schema_version: 1; kind: "tool"; operation_id: string }
  | { schema_version: 1; kind: "paused"; reason_oid: string | null }
  | { schema_version: 1; kind: "host_resume"; reason_oid: string }
  | {
      schema_version: 1;
      kind: "stale_execution";
      pid: string;
      recovered_by_owner_sha256: string;
      prior_owner_sha256: string | null;
      prior_lease_sha256: string | null;
      prior_execution_generation: number;
      recovered_execution_generation: number;
      recovered_state_generation: number;
    };

const canonicalSha256 = /^[0-9a-f]{64}$/;

/** Reject malformed/private process wait projections before React sees them. */
export function assertProcessWaitState(value: unknown): asserts value is ProcessWaitState {
  if (!isRecord(value) || value.schema_version !== 1 || typeof value.kind !== "string") {
    throw new Error("GUI process wait state is malformed.");
  }
  const exactKeys = (...keys: string[]) => {
    const expected = new Set(["schema_version", "kind", ...keys]);
    return Object.keys(value).every((key) => expected.has(key)) && Object.keys(value).length === expected.size;
  };
  const nonEmpty = (item: unknown): item is string => typeof item === "string" && item.trim().length > 0;
  const sha256OrNull = (item: unknown) => item === null || (typeof item === "string" && canonicalSha256.test(item));
  const nonNegativeInteger = (item: unknown) => Number.isSafeInteger(item) && Number(item) >= 0;
  const positiveInteger = (item: unknown) => Number.isSafeInteger(item) && Number(item) > 0;
  let valid = false;
  switch (value.kind) {
    case "child":
      valid = exactKeys("child_pid") && nonEmpty(value.child_pid);
      break;
    case "message":
      valid = exactKeys("filters") && isRecord(value.filters);
      break;
    case "human":
      valid = exactKeys("request_ids")
        && Array.isArray(value.request_ids)
        && value.request_ids.length > 0
        && value.request_ids.every(nonEmpty)
        && new Set(value.request_ids).size === value.request_ids.length;
      break;
    case "tool":
      valid = exactKeys("operation_id") && nonEmpty(value.operation_id);
      break;
    case "paused":
      valid = exactKeys("reason_oid") && (value.reason_oid === null || nonEmpty(value.reason_oid));
      break;
    case "host_resume":
      valid = exactKeys("reason_oid") && nonEmpty(value.reason_oid);
      break;
    case "stale_execution":
      valid = exactKeys(
        "pid",
        "recovered_by_owner_sha256",
        "prior_owner_sha256",
        "prior_lease_sha256",
        "prior_execution_generation",
        "recovered_execution_generation",
        "recovered_state_generation"
      )
        && nonEmpty(value.pid)
        && typeof value.recovered_by_owner_sha256 === "string"
        && canonicalSha256.test(value.recovered_by_owner_sha256)
        && sha256OrNull(value.prior_owner_sha256)
        && sha256OrNull(value.prior_lease_sha256)
        && nonNegativeInteger(value.prior_execution_generation)
        && positiveInteger(value.recovered_execution_generation)
        && positiveInteger(value.recovered_state_generation);
      break;
  }
  if (!valid) throw new Error("GUI process wait state is malformed.");
}

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
  loaded_skills: Record<string, LoadedSkillSummary>;
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
  package_sha256?: string;
};

export type LoadedSkillSummary = Record<string, unknown> & {
  package_sha256?: string;
};

export type JsonRpcEndpointSummary = Record<string, unknown> & {
  endpoint_id: string;
  name?: string;
  description?: string;
};

export type McpProtocolMode = "legacy" | "auto" | "2026-07-28";

export type McpProtocolEra = "legacy" | "modern";

export type McpConnectionInfo = {
  protocol_mode: McpProtocolMode;
  protocol_era: McpProtocolEra;
  protocol_revision: string;
  sessionless: boolean;
  fallback_used: boolean;
  server_name?: string | null;
  server_version?: string | null;
  capabilities: string[];
  unsupported_capabilities: string[];
};

export type McpExchangeReceipt = {
  phase: "server/discover" | "initialize" | "tools/list" | "tools/call";
  request_bytes: number;
  response_bytes: number;
  duration_s: number;
  call_started: boolean;
};

export type McpToolSummary = Record<string, unknown> & {
  tool_id: string;
  mcp_name: string;
  right: string;
  resource: string;
  rollback_class: string;
  rollback_status: string;
  state_mutation: boolean;
  information_flow: boolean;
  input_schema: Record<string, unknown>;
  metadata: Record<string, unknown>;
  live?: {
    name: string;
    description?: string | null;
    input_schema: Record<string, unknown>;
    schema_matches_manifest: boolean;
  };
};

export type McpServerSummary = Record<string, unknown> & {
  schema_version: 1 | 2;
  server_id: string;
  protocol_mode: McpProtocolMode;
  transport: Record<string, unknown> & { type: string };
  tools: McpToolSummary[];
  timeout_s: number;
  max_request_bytes: number;
  max_response_bytes: number;
  metadata: Record<string, unknown>;
};

export type McpToolListResult = {
  server_id: string;
  schema_version: 1 | 2;
  transport: string;
  protocol_mode: McpProtocolMode;
  tools: McpToolSummary[];
  refreshed: boolean;
  response_bytes: number;
  connection?: McpConnectionInfo | null;
  receipts?: McpExchangeReceipt[];
};

export type McpDiscoveryResult = {
  server_id: string;
  connection: McpConnectionInfo;
  request_bytes: number;
  response_bytes: number;
  duration_s: number;
  receipts: McpExchangeReceipt[];
};

export type McpCallResult = {
  server_id: string;
  tool_id: string;
  mcp_name: string;
  status: "ok" | "mcp_error" | "transport_error" | "invalid_response" | "response_too_large" | "input_required_unsupported";
  ok: boolean;
  result?: unknown;
  error?: Record<string, unknown> | null;
  response_bytes: number;
  duration_s: number;
  connection?: McpConnectionInfo | null;
  receipts?: McpExchangeReceipt[];
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

export const taskRunStatuses = [
  "queued",
  "running",
  "waiting_human",
  "waiting_process",
  "waiting_message",
  "waiting_tool",
  "paused",
  "cancelling",
  "finalizing",
  "needs_attention",
  "succeeded",
  "failed",
  "cancelled"
] as const;

export type TaskRunStatus = (typeof taskRunStatuses)[number];

export const taskRunActions = [
  "run",
  "wait",
  "pause",
  "resume",
  "cancel",
  "follow_up",
  "recover",
  "rerun"
] as const;

export type TaskRunAction = (typeof taskRunActions)[number];
export type TaskRunRetention = "purge_on_terminal" | "permanent";

export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

export type TaskRunSpecV1 = {
  schema_version: 1;
  goal: JsonValue;
  display_title: string;
  image_id?: string;
  launch_options?: Record<string, unknown>;
  authority_manifest_id?: string | null;
  deadline_at?: string | null;
  retention?: TaskRunRetention;
};

export const taskRunBlockerKinds = [
  "unknown_effect",
  "effect_unknown",
  "payload_missing",
  "payload_corrupt",
  "binding_drift",
  "pending_action_unreplayable",
  "active_object_task",
  "requirements_unsatisfied",
  "cleanup_failed",
  "authority_revoked",
  "deadline_reached",
  "effect_unsettled",
  "reservation_unsettled",
  "publication_unsettled",
  "manual_recovery_required"
] as const;

export type TaskRunBlockerKind = (typeof taskRunBlockerKinds)[number];

export type TaskRunBlocker = {
  kind: TaskRunBlockerKind;
  code?: string;
  message?: string;
  evidence_ref?: string;
  process_id?: string;
  effect_id?: string;
};

export type TaskRunRequirement = {
  schema_version: 1;
  requirement_id: string;
  run_id: string;
  ordinal: number;
  kind: "initial" | "follow_up";
  status: "pending" | "in_progress" | "satisfied" | "blocked" | "waived";
  requirement_sha256: string;
  label: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  waived_by: string | null;
  content_available: boolean;
  content_retention: "plaintext" | "hash_only";
  content_sha256: string;
  content_text?: string;
  content_truncated?: boolean;
};

/**
 * Redacted current-state projection. Full goal/follow-up/transcript payloads are
 * deliberately absent so this shape is safe to carry over SSE.
 */
export type TaskRunSummary = Record<string, unknown> & {
  schema_version: 1;
  run_id: string;
  revision: number;
  status: TaskRunStatus;
  display_title: string;
  root_pid: string | null;
  active_pid: string | null;
  allowed_actions: TaskRunAction[];
  blockers: TaskRunBlocker[];
  retention: TaskRunRetention;
  payloads_purged: boolean;
  requirement_counts?: Record<string, number>;
  step_counts?: Record<string, number>;
  step_count?: number;
  completed_step_count?: number;
  result_ref?: string | null;
  created_at?: string;
  updated_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
};

export type TaskRunDetail = {
  summary: TaskRunSummary;
  requirements: {
    items: TaskRunRequirement[];
    next_cursor: string | null;
    has_more: boolean;
  };
  recovery_options: TaskRunRecoveryOption[];
};

export type TaskRunLedgerItem = {
  schema_version: 1;
  item_id: string;
  run_id: string;
  seq: number;
  kind: "requirement" | "process" | "llm_turn" | "tool_call" | "human_wait" | "message_wait" | "checkpoint" | "effect" | "status_transition";
  status: string;
  label: string;
  occurred_at: string;
  requirement_id?: string;
  pid?: string;
  operation_id?: string;
  effect_id?: string;
  human_request_id?: string;
  llm_call_id?: string;
  checkpoint_id?: string;
  object_task_id?: string;
  metadata: Record<string, string | number | boolean>;
};

export type TaskRunLedgerPage = {
  items: TaskRunLedgerItem[];
  next_cursor: string | null;
  has_more: boolean;
};

export type TaskRunHumanRequestPage = {
  items: HumanRequest[];
  next_cursor: string | null;
  has_more: boolean;
  presentation_truncated: boolean;
};

export const taskRunEffectTransactionStates = [
  "prepared",
  "authorized",
  "approved",
  "dispatched",
  "committed",
  "failed",
  "unknown",
  "compensated"
] as const;

export type TaskRunEffectTransactionState = (typeof taskRunEffectTransactionStates)[number];

export type TaskRunRecoveryOption = {
  schema_version?: 1;
  option_id: string;
  kind?: string;
  label?: string;
  description?: string;
  requires_confirmation?: boolean;
  requires_receipt?: boolean;
  receipt_fields?: string[];
  effect_id?: string;
  expected_transaction_state?: TaskRunEffectTransactionState;
  runtime_epoch?: number;
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

export type LlmTraceCoverage = "complete" | "custom_client_incomplete" | "legacy_final_only";
export type LlmReasoningAvailability = "returned" | "not_returned" | "not_persisted" | "purged" | "limited";
export type LlmPayloadRetentionTier = "full" | "summary" | "hash_only";
export type LlmUsageField =
  | "prompt_tokens"
  | "completion_tokens"
  | "total_tokens"
  | "input_tokens"
  | "output_tokens"
  | "cache_read_tokens"
  | "cache_write_tokens";
export type LlmUsage = Partial<Record<LlmUsageField, number>>;

/** Content-free projection used by snapshots, SSE, and paginated call lists. */
export type LlmCallSummary = {
  schema_version: 1;
  call_id: string;
  pid: string | null;
  image_id: string | null;
  purpose: string;
  status: string;
  api: string | null;
  model: string | null;
  usage: LlmUsage;
  error: string | null;
  created_at: string;
  completed_at: string | null;
  request_id: string | null;
  response_id: string | null;
  attempt_count: number;
  coverage: LlmTraceCoverage;
  selected_attempt: number | null;
  reasoning_availability: LlmReasoningAvailability;
  payload_retention_tier: LlmPayloadRetentionTier;
};

/** Kept as a source-compatible name for timeline consumers. */
export type LlmCall = LlmCallSummary;

export type LlmProviderAttempt = {
  sequence: number;
  kind: string;
  api: string | null;
  status: string;
  model: string | null;
  request_id: string | null;
  response_id: string | null;
  reasoning_availability: LlmReasoningAvailability;
  reasoning_blocks: LlmReasoningBlock[];
  output_availability: LlmReasoningAvailability;
  tool_names: string[];
  tool_call_count: number;
  usage: LlmUsage;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  error: LlmAttemptError | null;
};

export type LlmReasoningBlock = {
  type: "summary_text" | "reasoning_text" | "opaque" | "omitted";
  source: string | null;
  reason: "bounds" | "aggregate_limit" | "structure_limit" | "node_limit" | "non_finite_number" | null;
  chars: number | null;
  bytes: number | null;
  sha256: string | null;
};

export type LlmAttemptError = {
  error_type: string | null;
  status_code: number | null;
  message_bytes: number | null;
  message_sha256: string | null;
};

export type LlmTraceContentField =
  | "messages"
  | "tools"
  | "request_options"
  | "raw_response"
  | "response_content"
  | "attempt_reasoning"
  | "attempt_output"
  | "attempt_tool_calls";

export type LlmTraceContentAvailability = "available" | "not_returned" | "not_persisted" | "purged" | "limited";

export type LlmTraceContentDescriptor = {
  field: LlmTraceContentField;
  attempt_sequence: number | null;
  availability: LlmTraceContentAvailability;
  content_type: "text" | "json";
  size_bytes: number | null;
  size_chars: number | null;
  content_hash: string | null;
  cursor: string | null;
};

export type LlmCallPage = {
  schema_version: 1;
  items: LlmCallSummary[];
  next_cursor: string | null;
  has_more: boolean;
};

export type LlmCallDetail = {
  schema_version: 1;
  call: LlmCallSummary;
  attempts: LlmProviderAttempt[];
  content: LlmTraceContentDescriptor[];
};

export type LlmTraceContentChunk = {
  schema_version: 1;
  pid: string;
  call_id: string;
  field: LlmTraceContentField;
  attempt_sequence: number | null;
  content: string;
  next_cursor: string | null;
  has_more: boolean;
  content_hash: string | null;
  retention_tier: LlmPayloadRetentionTier;
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
  schema_version: 3;
  db: string;
  scheduler: SchedulerStatus;
  processes: RuntimeProcess[];
  human_requests: HumanRequest[];
  events: RuntimeEvent[];
  audit: AuditRecord[];
  llm_calls: LlmCallSummary[];
  object_tasks: ObjectTask[];
  task_runs: TaskRunSummary[];
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

export type SemanticMode = "off" | "shadow";
export type SemanticAdapter = "deterministic" | "external" | "scripted";
export type SemanticAssessmentKind = "approval" | "root_goal" | "provider_ingress";
export type SemanticAssessmentStatus =
  | "success"
  | "skipped_policy"
  | "egress_blocked"
  | "timeout"
  | "provider_error"
  | "provider_outcome_unknown"
  | "invalid_schema"
  | "ood"
  | "abstained"
  | "stale_input";
export type SemanticAssessmentDomain =
  | "filesystem"
  | "shell"
  | "git"
  | "jsonrpc"
  | "mcp"
  | "runtime"
  | "unknown";
export type SemanticShadowOutcome =
  | "would_issue_exact_once"
  | "would_deny"
  | "require_human";
export type SemanticFindingSeverity = "info" | "low" | "medium" | "high" | "critical";
export type SemanticSensitivity = "public" | "normal" | "confidential" | "restricted" | "secret";
export type SemanticIntegrity = "untrusted" | "unknown" | "checked" | "verified";
export type SemanticTrust = "untrusted" | "unknown" | "user_asserted" | "verified" | "trusted";
export type SemanticCalibrationBucket = "unknown" | "very_low" | "low" | "medium" | "high" | "very_high";
export type SemanticHumanOutcome = "pending" | "approved" | "rejected" | "edited" | "cancelled" | "delivered";
export type SemanticReasonCode =
  | "policy_match"
  | "hard_policy_violation"
  | "malformed_request"
  | "stale_binding"
  | "stale_manifest"
  | "stale_policy"
  | "unsupported_action"
  | "high_risk_action"
  | "control_right"
  | "data_release"
  | "ceiling_miss"
  | "missing_authoritative_predicate"
  | "schema_invalid"
  | "provider_error"
  | "provider_outcome_unknown"
  | "timeout"
  | "egress_blocked"
  | "out_of_distribution"
  | "abstained"
  | "risk_detected"
  | "sensitive_data"
  | "credential_material"
  | "prompt_injection"
  | "mixed_identity"
  | "low_integrity";
export type SemanticDataCategory =
  | "credential"
  | "personal"
  | "financial"
  | "health"
  | "legal"
  | "source_code"
  | "business_secret"
  | "instruction_attack"
  | "untrusted_content"
  | "other";
export type SemanticDataLocator =
  | "approval.request"
  | "root_goal"
  | "provider.result"
  | "redacted_intent";
export type SemanticPredicate =
  | "schema_valid"
  | "exact_external_operation"
  | "binding_current"
  | "manifest_current"
  | "policy_current"
  | "action_known"
  | "action_auto_eligible"
  | "low_risk"
  | "resource_exact"
  | "single_non_control_right"
  | "ceiling_matched"
  | "data_flow_allowed"
  | "profile_pinned";

export type SemanticStatus = {
  schema_version: 2;
  mode: SemanticMode;
  adapter: SemanticAdapter;
  profile_id: string | null;
  queue: {
    queued: number;
    leased: number;
    succeeded: number;
    failed: number;
    cancelled: number;
    capture_failures: number;
  };
  assessments: {
    total: number;
    success: number;
    error: number;
    ood: number;
    would_issue_exact_once: number;
    would_deny: number;
    require_human: number;
    by_status: Record<SemanticAssessmentStatus, number>;
    by_domain: Record<SemanticAssessmentDomain, number>;
  };
  actual_auto_approval: {
    numerator: 0;
    denominator: 0;
    rate: null;
  };
};

export type SemanticAssessmentSummary = {
  assessment_id: string;
  job_id: string;
  kind: SemanticAssessmentKind;
  status: SemanticAssessmentStatus;
  domain: SemanticAssessmentDomain;
  action_id: string;
  pid: string;
  request_id: string | null;
  operation_id: string | null;
  effect_id: string | null;
  shadow_outcome: SemanticShadowOutcome;
  reason_codes: SemanticReasonCode[];
  ood: boolean;
  abstain: boolean;
  confidence_bps: number | null;
  calibration_bucket: SemanticCalibrationBucket;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_microunits: number | null;
  classifier_id: string;
  classifier_version: string;
  artifact_sha256: string | null;
  input_sha256: string | null;
  feature_snapshot_sha256: string | null;
  policy_sha256: string | null;
  created_at: string;
  completed_at: string;
  latency_ms: number | null;
  human_outcome: SemanticHumanOutcome | null;
  tenant_bucket_sha256: string | null;
};

export type SemanticFinding = {
  code: SemanticReasonCode;
  severity: SemanticFindingSeverity;
  confidence_bps: number;
  evidence_sha256: string;
  source: "model" | "deterministic" | "host";
};

export type SemanticDataFinding = {
  category: SemanticDataCategory;
  field: SemanticDataLocator;
  span_start: number | null;
  span_end: number | null;
  sensitivity_floor: SemanticSensitivity;
  integrity_ceiling: SemanticIntegrity;
  trust_ceiling: SemanticTrust;
  confidence_bps: number;
  evidence_sha256: string;
};

export type SemanticAssessmentDetail = SemanticAssessmentSummary & {
  findings: SemanticFinding[];
  data_findings: SemanticDataFinding[];
  matched_rule_ids: string[];
  proven_predicates: SemanticPredicate[];
  missing_predicates: SemanticPredicate[];
  source_refs_sha256: string | null;
  data_labels_sha256: string | null;
  sink_identity_sha256: string | null;
  tool_schema_sha256: string | null;
  provider_spec_sha256: string | null;
  manifest_sha256: string | null;
  action_sha256: string;
  resource_sha256: string | null;
  args_sha256: string | null;
  state_sha256: string | null;
  projection_sha256: string;
};

export type SemanticAssessmentPage = {
  schema_version: 1;
  items: SemanticAssessmentSummary[];
  next_cursor: string | null;
};

export type SemanticAssessmentDetailResponse = {
  schema_version: 1;
  assessment: SemanticAssessmentDetail;
};

const semanticModeValues = new Set(["off", "shadow"]);
const semanticAdapterValues = new Set(["deterministic", "external", "scripted"]);
const semanticKindValues = new Set(["approval", "root_goal", "provider_ingress"]);
const semanticStatusValues = new Set([
  "success", "skipped_policy", "egress_blocked", "timeout", "provider_error",
  "provider_outcome_unknown", "invalid_schema", "ood", "abstained", "stale_input"
]);
const semanticDomainValues = new Set(["filesystem", "shell", "git", "jsonrpc", "mcp", "runtime", "unknown"]);
const semanticOutcomeValues = new Set(["would_issue_exact_once", "would_deny", "require_human"]);
const semanticSeverityValues = new Set(["info", "low", "medium", "high", "critical"]);
const semanticSensitivityValues = new Set(["public", "normal", "confidential", "restricted", "secret"]);
const semanticIntegrityValues = new Set(["untrusted", "unknown", "checked", "verified"]);
const semanticTrustValues = new Set(["untrusted", "unknown", "user_asserted", "verified", "trusted"]);
const semanticCalibrationValues = new Set(["unknown", "very_low", "low", "medium", "high", "very_high"]);
const semanticHumanOutcomeValues = new Set(["pending", "approved", "rejected", "edited", "cancelled", "delivered"]);
const semanticReasonCodeValues = new Set([
  "policy_match", "hard_policy_violation", "malformed_request", "stale_binding", "stale_manifest", "stale_policy",
  "unsupported_action", "high_risk_action", "control_right", "data_release", "ceiling_miss",
  "missing_authoritative_predicate", "schema_invalid", "provider_error", "provider_outcome_unknown", "timeout",
  "egress_blocked", "out_of_distribution", "abstained", "risk_detected", "sensitive_data", "credential_material",
  "prompt_injection", "mixed_identity", "low_integrity"
]);
const semanticDataCategoryValues = new Set([
  "credential", "personal", "financial", "health", "legal", "source_code", "business_secret",
  "instruction_attack", "untrusted_content", "other"
]);
const semanticDataLocatorValues = new Set<SemanticDataLocator>([
  "approval.request", "root_goal", "provider.result", "redacted_intent"
]);
const semanticCoarseDataLocatorByKind: Record<SemanticAssessmentKind, SemanticDataLocator> = {
  approval: "approval.request",
  root_goal: "root_goal",
  provider_ingress: "provider.result"
};
const semanticPredicateValues = new Set([
  "schema_valid", "exact_external_operation", "binding_current", "manifest_current", "policy_current", "action_known",
  "action_auto_eligible", "low_risk", "resource_exact", "single_non_control_right", "ceiling_matched",
  "data_flow_allowed", "profile_pinned"
]);
const semanticStatusKeys = new Set(["schema_version", "mode", "adapter", "profile_id", "queue", "assessments", "actual_auto_approval"]);
const semanticQueueKeys = new Set(["queued", "leased", "succeeded", "failed", "cancelled", "capture_failures"]);
const semanticCountKeys = new Set([
  "total", "success", "error", "ood", "would_issue_exact_once", "would_deny", "require_human", "by_status", "by_domain"
]);
const semanticScalarCountKeys = new Set(["total", "success", "error", "ood", "would_issue_exact_once", "would_deny", "require_human"]);
const semanticActualApprovalKeys = new Set(["numerator", "denominator", "rate"]);
const semanticSummaryKeys = new Set([
  "assessment_id", "job_id", "kind", "status", "domain", "action_id", "pid", "request_id", "operation_id", "effect_id",
  "shadow_outcome", "reason_codes", "ood", "abstain", "confidence_bps", "calibration_bucket", "input_tokens",
  "output_tokens", "cost_microunits", "classifier_id", "classifier_version",
  "artifact_sha256", "input_sha256", "feature_snapshot_sha256", "policy_sha256", "created_at", "completed_at",
  "latency_ms", "human_outcome", "tenant_bucket_sha256"
]);
const semanticDetailExtraKeys = new Set([
  "findings", "data_findings", "matched_rule_ids", "proven_predicates", "missing_predicates", "source_refs_sha256",
  "data_labels_sha256", "sink_identity_sha256", "tool_schema_sha256", "provider_spec_sha256", "manifest_sha256",
  "action_sha256", "resource_sha256", "args_sha256", "state_sha256", "projection_sha256"
]);
const semanticFindingKeys = new Set(["code", "severity", "confidence_bps", "evidence_sha256", "source"]);
const semanticDataFindingKeys = new Set([
  "category", "field", "span_start", "span_end", "sensitivity_floor", "integrity_ceiling", "trust_ceiling",
  "confidence_bps", "evidence_sha256"
]);
const semanticPageKeys = new Set(["schema_version", "items", "next_cursor"]);
const semanticDetailResponseKeys = new Set(["schema_version", "assessment"]);
const semanticIdentifierPattern = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;
const semanticRedactedIntentMaxChars = 2_000;
const semanticActionPattern = /^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$/;

/** Reject malformed, unbounded, or private Semantic status fields before rendering. */
export function assertSemanticStatus(value: unknown): asserts value is SemanticStatus {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticStatusKeys) || value.schema_version !== 2
      || typeof value.mode !== "string" || !semanticModeValues.has(value.mode)
      || typeof value.adapter !== "string" || !semanticAdapterValues.has(value.adapter)
      || !(value.profile_id === null || isSemanticIdentifier(value.profile_id))
      || !isExactNonNegativeCounterRecord(value.queue, semanticQueueKeys)
      || !isRecord(value.assessments)
      || !hasOnlyKeys(value.assessments, semanticCountKeys)
      || !hasNonNegativeCounterFields(value.assessments, semanticScalarCountKeys)
      || !isRecord(value.assessments.by_status)
      || !isExactNonNegativeCounterRecord(value.assessments.by_status, semanticStatusValues)
      || !isExactNonNegativeCounterRecord(value.assessments.by_domain, semanticDomainValues)
      || sumCounters(value.assessments.by_status) !== Number(value.assessments.total)
      || sumCounters(value.assessments.by_domain) !== Number(value.assessments.total)
      || Number(value.assessments.success) + Number(value.assessments.error) !== Number(value.assessments.total)
      || Number(value.assessments.would_issue_exact_once) + Number(value.assessments.would_deny)
        + Number(value.assessments.require_human) !== Number(value.assessments.total)
      || value.assessments.ood !== value.assessments.by_status.ood
      || !isRecord(value.actual_auto_approval)
      || !hasOnlyKeys(value.actual_auto_approval, semanticActualApprovalKeys)
      || value.actual_auto_approval.numerator !== 0
      || value.actual_auto_approval.denominator !== 0
      || value.actual_auto_approval.rate !== null) {
    throw new Error("GUI Semantic status is malformed.");
  }
}

/** Reject unknown fields so prompt, raw content, and reasoning can never reach React. */
export function assertSemanticAssessmentSummary(value: unknown): asserts value is SemanticAssessmentSummary {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticSummaryKeys)) {
    throw new Error("GUI Semantic assessment summary is malformed or contains a private field.");
  }
  if (!isBoundedString(value.assessment_id, 512) || !isBoundedString(value.job_id, 512)
      || typeof value.kind !== "string" || !semanticKindValues.has(value.kind)
      || typeof value.status !== "string" || !semanticStatusValues.has(value.status)
      || typeof value.domain !== "string" || !semanticDomainValues.has(value.domain)
      || !isSemanticAction(value.action_id)
      || !isBoundedString(value.pid, 1024)
      || !isNullableBoundedString(value.request_id, 1024)
      || !isNullableBoundedString(value.operation_id, 1024)
      || !isNullableBoundedString(value.effect_id, 1024)
      || typeof value.shadow_outcome !== "string" || !semanticOutcomeValues.has(value.shadow_outcome)
      || !isSemanticEnumList(value.reason_codes, 128, semanticReasonCodeValues)
      || typeof value.ood !== "boolean" || typeof value.abstain !== "boolean"
      || !isNullableConfidenceBps(value.confidence_bps)
      || typeof value.calibration_bucket !== "string" || !semanticCalibrationValues.has(value.calibration_bucket)
      || !isNullableNonNegativeInteger(value.input_tokens)
      || !isNullableNonNegativeInteger(value.output_tokens)
      || !isNullableNonNegativeInteger(value.cost_microunits)
      || !isBoundedString(value.classifier_id, 512) || !isBoundedString(value.classifier_version, 512)
      || !isNullableSha256(value.artifact_sha256) || !isNullableSha256(value.input_sha256)
      || !isNullableSha256(value.feature_snapshot_sha256) || !isNullableSha256(value.policy_sha256)
      || !isBoundedString(value.created_at, 128) || !isBoundedString(value.completed_at, 128)
      || !(value.latency_ms === null || isNonNegativeSafeInteger(value.latency_ms))
      || !(value.human_outcome === null || (typeof value.human_outcome === "string" && semanticHumanOutcomeValues.has(value.human_outcome)))
      || !isNullableSha256(value.tenant_bucket_sha256)) {
    throw new Error("GUI Semantic assessment summary is malformed or contains a private field.");
  }
}

export function assertSemanticAssessmentPage(value: unknown): asserts value is SemanticAssessmentPage {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticPageKeys) || value.schema_version !== 1
      || !Array.isArray(value.items) || value.items.length > 100
      || !isNullableBoundedString(value.next_cursor, 2048)) {
    throw new Error("GUI Semantic assessment page is malformed.");
  }
  for (const item of value.items) assertSemanticAssessmentSummary(item);
}

export function assertSemanticAssessmentDetailResponse(value: unknown): asserts value is SemanticAssessmentDetailResponse {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticDetailResponseKeys) || value.schema_version !== 1
      || !isRecord(value.assessment)) {
    throw new Error("GUI Semantic assessment detail response is malformed.");
  }
  const assessment = value.assessment;
  const detailKeys = new Set([...semanticSummaryKeys, ...semanticDetailExtraKeys]);
  if (!hasOnlyKeys(assessment, detailKeys)) {
    throw new Error("GUI Semantic assessment detail contains a private field.");
  }
  const summary = Object.fromEntries([...semanticSummaryKeys].map((key) => [key, assessment[key]]));
  assertSemanticAssessmentSummary(summary);
  if (!Array.isArray(assessment.findings) || assessment.findings.length > 128
      || !Array.isArray(assessment.data_findings) || assessment.data_findings.length > 128
      || !isSemanticIdentifierList(assessment.matched_rule_ids, 128)
      || !isSemanticEnumList(assessment.proven_predicates, 128, semanticPredicateValues)
      || !isSemanticEnumList(assessment.missing_predicates, 128, semanticPredicateValues)
      || !isNullableSha256(assessment.source_refs_sha256)
      || !isNullableSha256(assessment.data_labels_sha256)
      || !isNullableSha256(assessment.sink_identity_sha256)
      || !isNullableSha256(assessment.tool_schema_sha256)
      || !isNullableSha256(assessment.provider_spec_sha256)
      || !isNullableSha256(assessment.manifest_sha256)
      || !isSha256(assessment.action_sha256)
      || !isNullableSha256(assessment.resource_sha256)
      || !isNullableSha256(assessment.args_sha256)
      || !isNullableSha256(assessment.state_sha256)
      || !isSha256(assessment.projection_sha256)) {
    throw new Error("GUI Semantic assessment detail is malformed.");
  }
  for (const finding of assessment.findings) assertSemanticFinding(finding);
  for (const finding of assessment.data_findings) {
    assertSemanticDataFinding(finding, assessment.kind as SemanticAssessmentKind);
  }
}

function assertSemanticFinding(value: unknown): asserts value is SemanticFinding {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticFindingKeys)
      || typeof value.code !== "string" || !semanticReasonCodeValues.has(value.code)
      || typeof value.severity !== "string" || !semanticSeverityValues.has(value.severity)
      || !isConfidenceBps(value.confidence_bps)
      || !isSha256(value.evidence_sha256)
      || (value.source !== "model" && value.source !== "deterministic" && value.source !== "host")) {
    throw new Error("GUI Semantic finding is malformed.");
  }
}

function assertSemanticDataFinding(
  value: unknown,
  kind: SemanticAssessmentKind
): asserts value is SemanticDataFinding {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticDataFindingKeys)
      || typeof value.category !== "string" || !semanticDataCategoryValues.has(value.category)
      || typeof value.field !== "string" || !semanticDataLocatorValues.has(value.field as SemanticDataLocator)
      || !isNullableNonNegativeInteger(value.span_start) || !isNullableNonNegativeInteger(value.span_end)
      || ((value.span_start === null) !== (value.span_end === null))
      || (typeof value.span_start === "number" && typeof value.span_end === "number" && value.span_start >= value.span_end)
      || typeof value.sensitivity_floor !== "string" || !semanticSensitivityValues.has(value.sensitivity_floor)
      || typeof value.integrity_ceiling !== "string" || !semanticIntegrityValues.has(value.integrity_ceiling)
      || typeof value.trust_ceiling !== "string" || !semanticTrustValues.has(value.trust_ceiling)
      || !isConfidenceBps(value.confidence_bps) || !isSha256(value.evidence_sha256)) {
    throw new Error("GUI Semantic data finding is malformed.");
  }
  const locator = value.field as SemanticDataLocator;
  const validSpan = locator === "redacted_intent"
    ? typeof value.span_start === "number"
      && typeof value.span_end === "number"
      && value.span_end <= semanticRedactedIntentMaxChars
    : locator === semanticCoarseDataLocatorByKind[kind]
      && value.span_start === null
      && value.span_end === null;
  if (!validSpan) throw new Error("GUI Semantic data finding is malformed.");
}

function isExactNonNegativeCounterRecord(value: unknown, keys: ReadonlySet<string>): boolean {
  return isRecord(value) && hasOnlyKeys(value, keys)
    && Object.values(value).every((item) => isNonNegativeSafeInteger(item));
}

function hasNonNegativeCounterFields(
  value: Record<string, unknown>,
  keys: ReadonlySet<string>
): boolean {
  return [...keys].every((key) => isNonNegativeSafeInteger(value[key]));
}

function sumCounters(value: unknown): number {
  if (!isRecord(value)) return Number.NaN;
  return Object.values(value).reduce<number>(
    (total, item) => total + (typeof item === "number" ? item : Number.NaN),
    0
  );
}

function isSemanticIdentifier(value: unknown): value is string {
  return typeof value === "string" && semanticIdentifierPattern.test(value);
}

function isSemanticAction(value: unknown): value is string {
  return typeof value === "string" && value.length <= 128 && semanticActionPattern.test(value);
}

function isSemanticIdentifierList(value: unknown, maxItems: number): value is string[] {
  return Array.isArray(value) && value.length <= maxItems
    && value.every(isSemanticIdentifier) && new Set(value).size === value.length;
}

function isSemanticEnumList(value: unknown, maxItems: number, allowed: ReadonlySet<string>): value is string[] {
  return Array.isArray(value) && value.length <= maxItems
    && value.every((item) => typeof item === "string" && allowed.has(item))
    && new Set(value).size === value.length;
}

function isBoundedString(value: unknown, maxLength: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maxLength
    && value === value.trim() && !value.includes("\0");
}

function isNullableBoundedString(value: unknown, maxLength: number): value is string | null {
  return value === null || isBoundedString(value, maxLength);
}

function isConfidenceBps(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0 && Number(value) <= 10_000;
}

function isNullableConfidenceBps(value: unknown): value is number | null {
  return value === null || isConfidenceBps(value);
}

function isNullableNonNegativeInteger(value: unknown): value is number | null {
  return value === null || isNonNegativeSafeInteger(value);
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && canonicalSha256.test(value);
}

function isNullableSha256(value: unknown): value is string | null {
  return value === null || isSha256(value);
}

export type GuiConnection = {
  url: string;
  token: string;
  db: string;
};

export type ImagePackageFileValue = string | { base64: string };

export type ImagePackageFile = {
  name: string;
  manifest: string;
  manifest_sha256: string;
  files: Record<string, ImagePackageFileValue>;
};

export type SseMessage = {
  id: string;
  event: string;
  data: unknown;
};

export type StreamConnectionStatus = "connecting" | "connected" | "reconnecting" | "failed";

const snapshotCollections = [
  "processes",
  "human_requests",
  "events",
  "audit",
  "llm_calls",
  "object_tasks",
  "task_runs",
  "tools",
  "llm_profiles",
  "images",
  "skills",
  "jsonrpc_endpoints",
  "mcp_servers",
  "modules"
] as const;

const llmTraceCoverages = ["complete", "custom_client_incomplete", "legacy_final_only"] as const;
const llmReasoningAvailabilities = ["returned", "not_returned", "not_persisted", "purged", "limited"] as const;
const llmContentAvailabilities = ["available", "not_returned", "not_persisted", "purged", "limited"] as const;
const llmPayloadRetentionTiers = ["full", "summary", "hash_only"] as const;
const llmUsageFields = new Set<LlmUsageField>([
  "prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens",
  "cache_read_tokens", "cache_write_tokens"
]);
const llmTraceContentFields = [
  "messages",
  "tools",
  "request_options",
  "raw_response",
  "response_content",
  "attempt_reasoning",
  "attempt_output",
  "attempt_tool_calls"
] as const satisfies readonly LlmTraceContentField[];
const llmCallSummaryKeys = new Set([
  "schema_version", "call_id", "pid", "image_id", "purpose", "status", "api", "model", "usage", "error",
  "created_at", "completed_at", "request_id", "response_id", "attempt_count", "coverage", "selected_attempt",
  "reasoning_availability", "payload_retention_tier"
]);
const llmAttemptKeys = new Set([
  "sequence", "kind", "api", "status", "model", "request_id", "response_id", "reasoning_availability",
  "reasoning_blocks", "output_availability", "tool_names", "tool_call_count", "usage", "started_at",
  "completed_at", "duration_ms", "error"
]);
const llmReasoningBlockKeys = new Set(["type", "source", "reason", "chars", "bytes", "sha256"]);
const llmAttemptErrorKeys = new Set(["error_type", "status_code", "message_bytes", "message_sha256"]);
const llmContentDescriptorKeys = new Set([
  "field", "attempt_sequence", "availability", "content_type", "size_bytes", "size_chars", "content_hash", "cursor"
]);
const llmCallPageKeys = new Set(["schema_version", "items", "next_cursor", "has_more"]);
const llmCallDetailKeys = new Set(["schema_version", "call", "attempts", "content"]);
const llmContentChunkKeys = new Set([
  "schema_version", "pid", "call_id", "field", "attempt_sequence", "content", "next_cursor", "has_more",
  "content_hash", "retention_tier"
]);

export function assertLlmCallSummary(value: unknown): asserts value is LlmCallSummary {
  if (!isRecord(value) || !hasOnlyKeys(value, llmCallSummaryKeys)
      || value.schema_version !== 1
      || !isNonEmptyString(value.call_id)
      || !isOptionalNullableString(value.pid)
      || !isOptionalNullableString(value.image_id)
      || typeof value.purpose !== "string"
      || !isNonEmptyString(value.status)
      || !isOptionalNullableString(value.api)
      || !isOptionalNullableString(value.model)
      || !isCanonicalLlmUsage(value.usage)
      || !isOptionalNullableString(value.error)
      || !isNonEmptyString(value.created_at)
      || !isOptionalNullableString(value.completed_at)
      || !isOptionalNullableString(value.request_id)
      || !isOptionalNullableString(value.response_id)
      || !isNonNegativeSafeInteger(value.attempt_count)
      || !(llmTraceCoverages as readonly unknown[]).includes(value.coverage)
      || !(value.selected_attempt === null || (Number.isSafeInteger(value.selected_attempt) && Number(value.selected_attempt) > 0))
      || !(llmReasoningAvailabilities as readonly unknown[]).includes(value.reasoning_availability)
      || !(llmPayloadRetentionTiers as readonly unknown[]).includes(value.payload_retention_tier)) {
    throw new Error("GUI LLM call summary is malformed.");
  }
  if (value.selected_attempt !== null && Number(value.selected_attempt) > Number(value.attempt_count)) {
    throw new Error("GUI LLM call summary selects an unknown attempt.");
  }
}

export function assertLlmCallPage(value: unknown): asserts value is LlmCallPage {
  if (!isRecord(value) || !hasOnlyKeys(value, llmCallPageKeys) || value.schema_version !== 1
      || !Array.isArray(value.items) || !isOptionalNullableString(value.next_cursor) || typeof value.has_more !== "boolean") {
    throw new Error("GUI LLM call page is malformed.");
  }
  for (const item of value.items) assertLlmCallSummary(item);
  if (value.has_more !== Boolean(value.next_cursor)) throw new Error("GUI LLM call page cursor is inconsistent.");
}

export function assertLlmCallDetail(value: unknown): asserts value is LlmCallDetail {
  if (!isRecord(value) || !hasOnlyKeys(value, llmCallDetailKeys) || value.schema_version !== 1
      || !Array.isArray(value.attempts) || !Array.isArray(value.content)) {
    throw new Error("GUI LLM call detail is malformed.");
  }
  assertLlmCallSummary(value.call);
  let previousSequence = 0;
  for (const attempt of value.attempts) {
    assertLlmProviderAttempt(attempt);
    if (attempt.sequence <= previousSequence) throw new Error("GUI LLM attempts are not strictly ordered.");
    previousSequence = attempt.sequence;
  }
  for (const descriptor of value.content) assertLlmContentDescriptor(descriptor);
}

export function assertLlmTraceContentChunk(value: unknown): asserts value is LlmTraceContentChunk {
  if (!isRecord(value) || !hasOnlyKeys(value, llmContentChunkKeys) || value.schema_version !== 1
      || !isNonEmptyString(value.pid) || !isNonEmptyString(value.call_id)
      || !(llmTraceContentFields as readonly unknown[]).includes(value.field)
      || !(value.attempt_sequence === null || (Number.isSafeInteger(value.attempt_sequence) && Number(value.attempt_sequence) > 0))
      || typeof value.content !== "string" || !isOptionalNullableString(value.next_cursor)
      || typeof value.has_more !== "boolean" || !isOptionalContentHash(value.content_hash)
      || !(llmPayloadRetentionTiers as readonly unknown[]).includes(value.retention_tier)) {
    throw new Error("GUI LLM trace content chunk is malformed.");
  }
  if (value.has_more !== Boolean(value.next_cursor)) throw new Error("GUI LLM trace content cursor is inconsistent.");
}

function assertLlmProviderAttempt(value: unknown): asserts value is LlmProviderAttempt {
  if (!isRecord(value) || !hasOnlyKeys(value, llmAttemptKeys)
      || !Number.isSafeInteger(value.sequence) || Number(value.sequence) <= 0
      || !isNonEmptyString(value.kind) || !isOptionalNullableString(value.api) || !isNonEmptyString(value.status)
      || !isOptionalNullableString(value.model) || !isOptionalNullableString(value.request_id)
      || !isOptionalNullableString(value.response_id)
      || !(llmReasoningAvailabilities as readonly unknown[]).includes(value.reasoning_availability)
      || !Array.isArray(value.reasoning_blocks)
      || !(llmReasoningAvailabilities as readonly unknown[]).includes(value.output_availability)
      || !isUniqueStringArray(value.tool_names) || !isNonNegativeSafeInteger(value.tool_call_count)
      || !isCanonicalLlmUsage(value.usage)
      || !isOptionalNullableString(value.started_at) || !isOptionalNullableString(value.completed_at)
      || !(value.duration_ms === null || isNonNegativeFiniteNumber(value.duration_ms))
      || !(value.error === null || isLlmAttemptError(value.error))) {
    throw new Error("GUI LLM provider attempt is malformed.");
  }
  for (const block of value.reasoning_blocks) {
    if (!isRecord(block) || !hasOnlyKeys(block, llmReasoningBlockKeys)
        || !["summary_text", "reasoning_text", "opaque", "omitted"].includes(String(block.type))
        || !isOptionalNullableString(block.source)
        || !(block.reason === null || ["bounds", "aggregate_limit", "structure_limit", "node_limit", "non_finite_number"].includes(String(block.reason)))
        || !(block.chars === null || isNonNegativeSafeInteger(block.chars))
        || !(block.bytes === null || isNonNegativeSafeInteger(block.bytes))
        || !isOptionalContentHash(block.sha256)) {
      throw new Error("GUI LLM reasoning block metadata is malformed.");
    }
  }
}

function assertLlmContentDescriptor(value: unknown): asserts value is LlmTraceContentDescriptor {
  if (!isRecord(value) || !hasOnlyKeys(value, llmContentDescriptorKeys)
      || !(llmTraceContentFields as readonly unknown[]).includes(value.field)
      || !(value.attempt_sequence === null || (Number.isSafeInteger(value.attempt_sequence) && Number(value.attempt_sequence) > 0))
      || !(llmContentAvailabilities as readonly unknown[]).includes(value.availability)
      || !(value.content_type === "text" || value.content_type === "json")
      || !(value.size_bytes === null || isNonNegativeSafeInteger(value.size_bytes))
      || !(value.size_chars === null || isNonNegativeSafeInteger(value.size_chars))
      || !isOptionalContentHash(value.content_hash) || !isOptionalNullableString(value.cursor)) {
    throw new Error("GUI LLM trace content descriptor is malformed.");
  }
  const readable = value.availability === "available" || value.availability === "limited";
  if (readable !== (typeof value.cursor === "string" && Boolean(value.cursor))) {
    throw new Error("GUI LLM trace content descriptor cursor is inconsistent.");
  }
}

function isLlmAttemptError(value: unknown): value is LlmAttemptError {
  return isRecord(value) && hasOnlyKeys(value, llmAttemptErrorKeys)
    && isOptionalNullableString(value.error_type)
    && (value.status_code === null || (Number.isSafeInteger(value.status_code) && Number(value.status_code) >= 100 && Number(value.status_code) <= 599))
    && (value.message_bytes === null || isNonNegativeSafeInteger(value.message_bytes))
    && isOptionalContentHash(value.message_sha256);
}

function isCanonicalLlmUsage(value: unknown): value is LlmUsage {
  return isRecord(value)
    && Object.entries(value).every(([key, counter]) =>
      llmUsageFields.has(key as LlmUsageField)
      && Number.isSafeInteger(counter)
      && Number(counter) >= 0
    );
}

function hasOnlyKeys(value: Record<string, unknown>, allowed: ReadonlySet<string>): boolean {
  return Object.keys(value).every((key) => allowed.has(key)) && Object.keys(value).length === allowed.size;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && Boolean(value);
}

function isOptionalContentHash(value: unknown): value is string | null {
  return value === null || (typeof value === "string" && /^[a-f0-9]{64}$/.test(value));
}

/** Fail closed on a malformed same-build response before React consumes it. */
export function assertRuntimeSnapshot(value: unknown): asserts value is RuntimeSnapshot {
  if (!isRecord(value)) throw new Error("GUI snapshot must be a JSON object.");
  if (value.schema_version !== 3) throw new Error("GUI snapshot schema_version must be 3.");
  if (typeof value.db !== "string") throw new Error("GUI snapshot is missing db.");
  try {
    assertSchedulerStatus(value.scheduler);
  } catch {
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
    if ("wait_state" in process && process.wait_state !== null) {
      assertProcessWaitState(process.wait_state);
    }
  }
  for (const run of value.task_runs as unknown[]) assertTaskRunSummary(run);
  for (const call of value.llm_calls as unknown[]) assertLlmCallSummary(call);
  for (const server of value.mcp_servers as unknown[]) assertMcpServerSummary(server);
}

const mcpProtocolModes = ["legacy", "auto", "2026-07-28"] as const;
const mcpProtocolEras = ["legacy", "modern"] as const;
const mcpExchangePhases = ["server/discover", "initialize", "tools/list", "tools/call"] as const;
const mcpCallStatuses = ["ok", "mcp_error", "transport_error", "invalid_response", "response_too_large", "input_required_unsupported"] as const;
const mcpConnectionKeys = new Set([
  "protocol_mode",
  "protocol_era",
  "protocol_revision",
  "sessionless",
  "fallback_used",
  "server_name",
  "server_version",
  "capabilities",
  "unsupported_capabilities"
]);
const mcpReceiptKeys = new Set(["phase", "request_bytes", "response_bytes", "duration_s", "call_started"]);
const mcpDiscoveryKeys = new Set(["server_id", "connection", "request_bytes", "response_bytes", "duration_s", "receipts"]);

export function assertMcpServerSummary(value: unknown): asserts value is McpServerSummary {
  if (!isRecord(value)
      || (value.schema_version !== 1 && value.schema_version !== 2)
      || typeof value.server_id !== "string"
      || !value.server_id
      || !isMcpProtocolMode(value.protocol_mode)
      || (value.schema_version === 1 && value.protocol_mode !== "legacy")
      || !isRecord(value.transport)
      || typeof value.transport.type !== "string"
      || !value.transport.type
      || !Array.isArray(value.tools)
      || typeof value.timeout_s !== "number"
      || !Number.isFinite(value.timeout_s)
      || value.timeout_s <= 0
      || !isNonNegativeSafeInteger(value.max_request_bytes)
      || !isNonNegativeSafeInteger(value.max_response_bytes)
      || !isRecord(value.metadata)) {
    throw new Error("GUI MCP server summary is malformed.");
  }
  if (["connection", "receipts", "protocol_era", "protocol_revision", "fallback_used"].some((key) => key in value)) {
    throw new Error("GUI MCP server summary contains operation-local protocol state.");
  }
  for (const tool of value.tools) assertMcpToolSummary(tool);
}

export function assertMcpConnectionInfo(value: unknown): asserts value is McpConnectionInfo {
  if (!isRecord(value)
      || Object.keys(value).some((key) => !mcpConnectionKeys.has(key))
      || !isMcpProtocolMode(value.protocol_mode)
      || !isMcpProtocolEra(value.protocol_era)
      || typeof value.protocol_revision !== "string"
      || !/^\d{4}-\d{2}-\d{2}$/.test(value.protocol_revision)
      || typeof value.sessionless !== "boolean"
      || typeof value.fallback_used !== "boolean"
      || !isOptionalNullableString(value.server_name)
      || !isOptionalNullableString(value.server_version)
      || !isUniqueStringArray(value.capabilities)
      || !isUniqueStringArray(value.unsupported_capabilities)) {
    throw new Error("GUI MCP connection metadata is malformed.");
  }
}

export function assertMcpDiscoveryResult(value: unknown): asserts value is McpDiscoveryResult {
  if (!isRecord(value)
      || Object.keys(value).some((key) => !mcpDiscoveryKeys.has(key))
      || typeof value.server_id !== "string"
      || !value.server_id
      || !isNonNegativeSafeInteger(value.request_bytes)
      || !isNonNegativeSafeInteger(value.response_bytes)
      || !isNonNegativeFiniteNumber(value.duration_s)
      || !Array.isArray(value.receipts)) {
    throw new Error("GUI MCP discovery result is malformed.");
  }
  assertMcpConnectionInfo(value.connection);
  for (const receipt of value.receipts) assertMcpExchangeReceipt(receipt);
}

export function assertMcpToolListResult(value: unknown): asserts value is McpToolListResult {
  if (!isRecord(value)
      || typeof value.server_id !== "string"
      || !value.server_id
      || (value.schema_version !== 1 && value.schema_version !== 2)
      || typeof value.transport !== "string"
      || !value.transport
      || !isMcpProtocolMode(value.protocol_mode)
      || (value.schema_version === 1 && value.protocol_mode !== "legacy")
      || !Array.isArray(value.tools)
      || typeof value.refreshed !== "boolean"
      || !isNonNegativeSafeInteger(value.response_bytes)) {
    throw new Error("GUI MCP tool list result is malformed.");
  }
  for (const tool of value.tools) assertMcpToolSummary(tool);
  assertOptionalMcpOperationMetadata(value);
}

export function assertMcpCallResult(value: unknown): asserts value is McpCallResult {
  if (!isRecord(value)
      || typeof value.server_id !== "string"
      || !value.server_id
      || typeof value.tool_id !== "string"
      || !value.tool_id
      || typeof value.mcp_name !== "string"
      || !value.mcp_name
      || typeof value.status !== "string"
      || !(mcpCallStatuses as readonly string[]).includes(value.status)
      || typeof value.ok !== "boolean"
      || !(value.error === undefined || value.error === null || isRecord(value.error))
      || !isNonNegativeSafeInteger(value.response_bytes)
      || !isNonNegativeFiniteNumber(value.duration_s)) {
    throw new Error("GUI MCP call result is malformed.");
  }
  assertOptionalMcpOperationMetadata(value);
}

function assertMcpToolSummary(value: unknown): asserts value is McpToolSummary {
  if (!isRecord(value)
      || !["tool_id", "mcp_name", "right", "resource", "rollback_class", "rollback_status"].every(
        (key) => typeof value[key] === "string" && Boolean(value[key])
      )
      || typeof value.state_mutation !== "boolean"
      || typeof value.information_flow !== "boolean"
      || !isRecord(value.input_schema)
      || !isRecord(value.metadata)) {
    throw new Error("GUI MCP tool summary is malformed.");
  }
  if (value.live !== undefined && (
    !isRecord(value.live)
    || typeof value.live.name !== "string"
    || !value.live.name
    || !isOptionalNullableString(value.live.description)
    || !isRecord(value.live.input_schema)
    || typeof value.live.schema_matches_manifest !== "boolean"
  )) {
    throw new Error("GUI MCP live tool summary is malformed.");
  }
}

function assertMcpExchangeReceipt(value: unknown): asserts value is McpExchangeReceipt {
  if (!isRecord(value)
      || Object.keys(value).some((key) => !mcpReceiptKeys.has(key))
      || typeof value.phase !== "string"
      || !(mcpExchangePhases as readonly string[]).includes(value.phase)
      || !isNonNegativeSafeInteger(value.request_bytes)
      || !isNonNegativeSafeInteger(value.response_bytes)
      || !isNonNegativeFiniteNumber(value.duration_s)
      || typeof value.call_started !== "boolean") {
    throw new Error("GUI MCP exchange receipt is malformed.");
  }
}

function assertOptionalMcpOperationMetadata(value: Record<string, unknown>): void {
  const hasConnection = value.connection !== undefined && value.connection !== null;
  const hasReceipts = value.receipts !== undefined;
  if (hasConnection) assertMcpConnectionInfo(value.connection);
  if (hasReceipts) {
    if (!Array.isArray(value.receipts)) throw new Error("GUI MCP exchange receipts are malformed.");
    for (const receipt of value.receipts) assertMcpExchangeReceipt(receipt);
  }
  if ((hasConnection && !hasReceipts)
      || (!hasConnection && Array.isArray(value.receipts) && value.receipts.length > 0)) {
    throw new Error("GUI MCP operation metadata is incomplete.");
  }
}

function isMcpProtocolMode(value: unknown): value is McpProtocolMode {
  return typeof value === "string" && (mcpProtocolModes as readonly string[]).includes(value);
}

function isMcpProtocolEra(value: unknown): value is McpProtocolEra {
  return typeof value === "string" && (mcpProtocolEras as readonly string[]).includes(value);
}

function isOptionalNullableString(value: unknown): value is string | null | undefined {
  return value === undefined || value === null || typeof value === "string";
}

function isUniqueStringArray(value: unknown): value is string[] {
  return Array.isArray(value)
    && value.every((item) => typeof item === "string" && Boolean(item))
    && new Set(value).size === value.length;
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

function isNonNegativeFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
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

export function assertTaskRunSummary(value: unknown): asserts value is TaskRunSummary {
  if (!isRecord(value)) throw new Error("GUI task run summary must be an object.");
  if (value.schema_version !== 1) throw new Error("GUI task run summary schema_version must be 1.");
  const unknownKey = Object.keys(value).find((key) => !taskRunSummaryKeys.has(key));
  if (unknownKey) throw new Error(`GUI task run summary contains private field: $.${unknownKey}.`);
  const forbiddenPath = forbiddenTaskRunSummaryPath(value);
  if (forbiddenPath) throw new Error(`GUI task run summary contains private field: ${forbiddenPath}.`);
  if (typeof value.run_id !== "string" || !value.run_id) {
    throw new Error("GUI task run summary is missing run_id.");
  }
  if (!Number.isSafeInteger(value.revision) || Number(value.revision) < 0) {
    throw new Error("GUI task run revision must be a non-negative safe integer.");
  }
  if (typeof value.status !== "string" || !(taskRunStatuses as readonly string[]).includes(value.status)) {
    throw new Error("GUI task run status is malformed.");
  }
  if (typeof value.display_title !== "string" || !value.display_title.trim()) {
    throw new Error("GUI task run title is malformed.");
  }
  if (!(value.root_pid === null || typeof value.root_pid === "string")) {
    throw new Error("GUI task run root_pid is malformed.");
  }
  if (!(value.active_pid === null || typeof value.active_pid === "string")) {
    throw new Error("GUI task run active_pid is malformed.");
  }
  if (!Array.isArray(value.allowed_actions)
      || value.allowed_actions.some((action) => typeof action !== "string" || !(taskRunActions as readonly string[]).includes(action))) {
    throw new Error("GUI task run allowed_actions is malformed.");
  }
  if (!Array.isArray(value.blockers) || value.blockers.some((blocker) => (
    !isRecord(blocker)
    || typeof blocker.kind !== "string"
    || !(taskRunBlockerKinds as readonly string[]).includes(blocker.kind)
    || Object.keys(blocker).some((key) => !taskRunBlockerKeys.has(key))
    || ["code", "message", "evidence_ref", "process_id", "effect_id"].some(
      (key) => key in blocker && typeof blocker[key] !== "string"
    )
  ))) {
    throw new Error("GUI task run blockers are malformed.");
  }
  if (value.retention !== "purge_on_terminal" && value.retention !== "permanent") {
    throw new Error("GUI task run retention is malformed.");
  }
  if (typeof value.payloads_purged !== "boolean") {
    throw new Error("GUI task run payload purge state is malformed.");
  }
  for (const key of ["step_count", "completed_step_count"] as const) {
    if (value[key] !== undefined && (!Number.isSafeInteger(value[key]) || Number(value[key]) < 0)) {
      throw new Error(`GUI task run ${key} is malformed.`);
    }
  }
  for (const key of ["requirement_counts", "step_counts"] as const) {
    const counts = value[key];
    if (counts !== undefined && (
      !isRecord(counts)
      || Object.values(counts).some((count) => !Number.isSafeInteger(count) || Number(count) < 0)
    )) {
      throw new Error(`GUI task run ${key} is malformed.`);
    }
  }
  if (!(value.result_ref === undefined || value.result_ref === null || typeof value.result_ref === "string")) {
    throw new Error("GUI task run result_ref is malformed.");
  }
}

const taskRunSummaryKeys = new Set([
  "schema_version",
  "run_id",
  "revision",
  "status",
  "display_title",
  "root_pid",
  "active_pid",
  "allowed_actions",
  "blockers",
  "retention",
  "payloads_purged",
  "requirement_counts",
  "step_counts",
  "step_count",
  "completed_step_count",
  "result_ref",
  "created_at",
  "updated_at",
  "started_at",
  "completed_at"
]);

const taskRunBlockerKeys = new Set([
  "kind",
  "code",
  "message",
  "evidence_ref",
  "process_id",
  "effect_id"
]);

export function assertTaskRunDetail(value: unknown): asserts value is TaskRunDetail {
  if (!isRecord(value)) throw new Error("GUI task run detail must be an object.");
  assertTaskRunSummary(value.summary);
  if (!isRecord(value.requirements)
      || !Array.isArray(value.requirements.items)
      || typeof value.requirements.has_more !== "boolean"
      || !(value.requirements.next_cursor === null || typeof value.requirements.next_cursor === "string")) {
    throw new Error("GUI task run requirement page is malformed.");
  }
  for (const item of value.requirements.items) assertTaskRunRequirement(item);
  if (!Array.isArray(value.recovery_options) || value.recovery_options.some((item) => !isTaskRunRecoveryOption(item))) {
    throw new Error("GUI task run recovery options are malformed.");
  }
}

const taskRunRecoveryOptionKeys = new Set([
  "schema_version",
  "option_id",
  "kind",
  "label",
  "description",
  "requires_confirmation",
  "requires_receipt",
  "receipt_fields",
  "effect_id",
  "expected_transaction_state",
  "runtime_epoch"
]);

function isTaskRunRecoveryOption(value: unknown): value is TaskRunRecoveryOption {
  if (!isRecord(value)
      || Object.keys(value).some((key) => !taskRunRecoveryOptionKeys.has(key))
      || typeof value.option_id !== "string"
      || !value.option_id
      || (value.schema_version !== undefined && value.schema_version !== 1)
      || (value.kind !== undefined && (typeof value.kind !== "string" || !value.kind))
      || (value.label !== undefined && typeof value.label !== "string")
      || (value.description !== undefined && typeof value.description !== "string")
      || (value.requires_confirmation !== undefined && typeof value.requires_confirmation !== "boolean")
      || (value.requires_receipt !== undefined && typeof value.requires_receipt !== "boolean")
      || (value.effect_id !== undefined && (typeof value.effect_id !== "string" || !value.effect_id))
      || (value.expected_transaction_state !== undefined && (
        typeof value.expected_transaction_state !== "string"
        || !(taskRunEffectTransactionStates as readonly string[]).includes(value.expected_transaction_state)
      ))
      || (value.runtime_epoch !== undefined && (
        !Number.isSafeInteger(value.runtime_epoch) || Number(value.runtime_epoch) < 0
      ))
      || (value.receipt_fields !== undefined && (
        !Array.isArray(value.receipt_fields)
        || value.receipt_fields.some((field) => typeof field !== "string" || !field)
      ))) {
    return false;
  }
  if (value.kind === "effect_receipt") {
    return typeof value.effect_id === "string"
      && typeof value.expected_transaction_state === "string"
      && Number.isSafeInteger(value.runtime_epoch)
      && value.requires_receipt === true;
  }
  return true;
}

function assertTaskRunRequirement(value: unknown): asserts value is TaskRunRequirement {
  if (!isRecord(value)
      || value.schema_version !== 1
      || typeof value.requirement_id !== "string"
      || !value.requirement_id
      || typeof value.run_id !== "string"
      || !value.run_id
      || !Number.isSafeInteger(value.ordinal)
      || Number(value.ordinal) < 0
      || (value.kind !== "initial" && value.kind !== "follow_up")
      || !["pending", "in_progress", "satisfied", "blocked", "waived"].includes(String(value.status))
      || typeof value.requirement_sha256 !== "string"
      || !/^[0-9a-f]{64}$/.test(value.requirement_sha256)
      || typeof value.label !== "string"
      || typeof value.created_by !== "string"
      || typeof value.created_at !== "string"
      || typeof value.updated_at !== "string"
      || !(value.started_at === null || typeof value.started_at === "string")
      || !(value.completed_at === null || typeof value.completed_at === "string")
      || !(value.waived_by === null || typeof value.waived_by === "string")
      || typeof value.content_available !== "boolean"
      || (value.content_retention !== "plaintext" && value.content_retention !== "hash_only")
      || typeof value.content_sha256 !== "string"
      || !/^[0-9a-f]{64}$/.test(value.content_sha256)) {
    throw new Error("GUI task run requirement is malformed.");
  }
  if (value.content_text !== undefined && typeof value.content_text !== "string") {
    throw new Error("GUI task run requirement content text is malformed.");
  }
  if (value.content_truncated !== undefined && typeof value.content_truncated !== "boolean") {
    throw new Error("GUI task run requirement truncation state is malformed.");
  }
  if (value.content_retention === "hash_only" && (value.content_available || "content_text" in value)) {
    throw new Error("GUI hash-only task run requirement contains plaintext content.");
  }
  if (value.content_retention === "plaintext" && value.content_available !== (typeof value.content_text === "string")) {
    throw new Error("GUI task run requirement availability disagrees with plaintext content.");
  }
}

const forbiddenTaskRunSummaryKeys = new Set([
  "goal",
  "objective",
  "body",
  "payload",
  "transcript",
  "provider_payload",
  "provider_request",
  "provider_response",
  "tool_input",
  "tool_output",
  "llm_input",
  "llm_output",
  "internal_error",
  "exception",
  "traceback",
  "requirements",
  "recovery_options"
]);

function forbiddenTaskRunSummaryPath(value: unknown, path = "$", seen = new Set<object>()): string | null {
  if (!value || typeof value !== "object") return null;
  if (seen.has(value)) return null;
  seen.add(value);
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      const found = forbiddenTaskRunSummaryPath(value[index], `${path}[${index}]`, seen);
      if (found) return found;
    }
    return null;
  }
  for (const [key, item] of Object.entries(value)) {
    const next = `${path}.${key}`;
    if (forbiddenTaskRunSummaryKeys.has(key)) return next;
    const found = forbiddenTaskRunSummaryPath(item, next, seen);
    if (found) return found;
  }
  return null;
}

/** Return only server-authorized controls, with needs-attention fail-closed. */
export function allowedTaskRunActions(value: unknown): ReadonlySet<TaskRunAction> {
  try {
    assertTaskRunSummary(value);
  } catch {
    return new Set<TaskRunAction>();
  }
  const selected = new Set<TaskRunAction>(value.allowed_actions);
  if (value.status === "needs_attention") {
    selected.delete("run");
    selected.delete("resume");
  }
  return selected;
}

export function taskRunSummaryFromSseData(value: unknown): TaskRunSummary {
  const summary = isRecord(value) && "summary" in value ? value.summary : value;
  assertTaskRunSummary(summary);
  return summary;
}

/** Ignore stale/equal revisions so reconnect replay cannot roll visible state backward. */
export function upsertTaskRunSummary(
  summaries: readonly TaskRunSummary[],
  candidate: TaskRunSummary
): TaskRunSummary[] {
  const current = summaries.find((item) => item.run_id === candidate.run_id);
  if (current && current.revision >= candidate.revision) return [...summaries];
  return [candidate, ...summaries.filter((item) => item.run_id !== candidate.run_id)];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
