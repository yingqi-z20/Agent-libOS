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
  schema_version: 1 | 2 | 3;
  server_id: string;
  protocol_mode: McpProtocolMode;
  transport: Record<string, unknown> & { type: string };
  tools: McpToolSummary[];
  timeout_s: number;
  max_request_bytes: number;
  max_response_bytes: number;
  auth_profile_id?: string | null;
  metadata: Record<string, unknown>;
};

export type McpToolListResult = {
  server_id: string;
  schema_version: 1 | 2 | 3;
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

export type McpUnregisterResult = {
  server_id: string;
  deleted: true;
};

export type McpCacheHint = {
  ttl_ms: number;
  scope: "private" | "public";
};

export type McpPage<T> = {
  items: T[];
  next_cursor: string | null;
  cache_hint: McpCacheHint | null;
  has_more?: boolean;
};

export type McpResource = {
  resource_id: string;
  name: string;
  title?: string | null;
  description?: string | null;
  mime_type?: string | null;
  size?: number | null;
  metadata?: Record<string, unknown>;
};

export type McpResourceTemplate = {
  template_id: string;
  name: string;
  title?: string | null;
  description?: string | null;
  mime_type?: string | null;
  metadata?: Record<string, unknown>;
};

export type McpArtifactReceipt = {
  artifact_id: string;
  byte_length: number;
  sha256: string;
  mime_type?: string | null;
};

export type McpContentBlock =
  | { kind: "text"; text: string; metadata?: Record<string, unknown> }
  | { kind: "blob"; artifact: McpArtifactReceipt | null; metadata?: Record<string, unknown> }
  | {
      kind: "resource_link";
      resource_handle: string;
      name: string;
      title?: string | null;
      description?: string | null;
      mime_type?: string | null;
      metadata?: Record<string, unknown>;
    };

export type McpResourceContents = {
  resource_id: string;
  contents: McpContentBlock[];
  provenance: "untrusted_mcp_resource";
};

export type McpPrompt = {
  prompt_id: string;
  name: string;
  title?: string | null;
  description?: string | null;
  arguments: Array<{
    name: string;
    title?: string | null;
    description?: string | null;
    required: boolean;
  }>;
  metadata?: Record<string, unknown>;
};

export type McpPromptResult = {
  prompt_id: string;
  messages: Array<{
    role: "user" | "assistant";
    content: McpContentBlock;
    provenance: "untrusted_mcp_prompt";
  }>;
  description?: string | null;
  user_confirmation_required: true;
};

export type McpCompletionResult = {
  values: string[];
  total?: number | null;
  has_more: boolean;
};

export type McpHumanReceipt = {
  human_request_id: string;
  human_revision: number;
  human_preview_sha256: string;
};

export type McpInputRequest = {
  request_id: string;
  kind: "elicitation" | "sampling_unsupported" | "roots_unsupported";
  mode?: "form" | "url" | null;
  prompt?: string | null;
  schema: Record<string, unknown>;
  inert_url?: string | null;
};

type McpInputRequiredBase = {
  kind: "input_required";
  revision: number;
  input_requests: McpInputRequest[];
  expires_at?: string | null;
};

export type McpInputRequired = McpInputRequiredBase & (
  | (McpHumanReceipt & {
    continuation_id: string;
    respondable: true;
  })
  | {
    continuation_id: "";
    respondable: false;
    human_request_id?: null;
    human_revision?: null;
    human_preview_sha256?: null;
  }
);

export type McpRemoteTask = {
  kind: "remote_task";
  task_ref: string;
  revision: number;
  status: "working" | "input_required" | "completed" | "failed" | "cancelled" | "cancel_requested" | "needs_attention";
  status_message?: string | null;
  result?: unknown;
  input_requests: McpInputRequest[];
  human_request_id?: string | null;
  human_revision?: number | null;
  human_preview_sha256?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  ttl_ms?: number | null;
  poll_interval_ms?: number | null;
};

export type McpOperationResult<T> =
  | { kind: "complete"; value: T | null; preview_sha256?: string }
  | McpInputRequired
  | McpRemoteTask;

export type McpOAuthStatus = {
  profile_id: string;
  status: "unconfigured" | "authorization_required" | "authorized" | "expired" | "revoked" | "needs_attention";
  issuer?: string | null;
  resource?: string | null;
  scopes: string[];
  principal_sha256?: string | null;
  expires_at?: string | null;
};

export type McpOAuthProfileInput = {
  profile_id: string;
  server_id: string;
  resource_uri: string;
  expected_issuer: string;
  redirect_uri: string;
  client_id: string;
  registration_mode: "preregistered" | "cimd";
  token_endpoint_auth_method?: "none" | "client_secret_basic" | "client_secret_post";
  allowed_scopes?: string[];
  default_scopes?: string[];
  audience?: string | null;
  protected_resource_metadata_url?: string | null;
  authorization_server_metadata_url?: string | null;
  protected_resource_metadata_sha256?: string | null;
  authorization_server_metadata_sha256?: string | null;
  allowed_endpoint_origins?: string[];
  allow_loopback_http?: boolean;
  protocol_revision?: "2026-07-28";
  transport?: "streamable_http";
};

export type McpAuthorizationChallenge = {
  challenge_id: string;
  authorization_url: string;
  expires_at: string;
};

export type McpSubscription = {
  subscription_id: string;
  server_id: string;
  status: "opening" | "active" | "lost" | "closed";
  requested_filters: string[];
  acknowledged_filters: string[];
  opened_at?: string | null;
  closed_at?: string | null;
  lost_reason?: string | null;
};

export type McpSubscriptionEvent = {
  sequence: number;
  event_type: string;
  payload: unknown;
  received_at: string;
  provenance: "untrusted_mcp_notification";
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

export type SemanticRisk = "low" | "medium" | "high" | "critical";
export type SemanticApprovalArgumentKind = "filesystem" | "shell" | "git" | "jsonrpc" | "mcp" | "other";
export type SemanticApprovalGitReferenceRole =
  | "base" | "base_oid" | "base_ref" | "branch" | "expected_remote_oid" | "git_old_oid" | "git_remote"
  | "git_remote_ref" | "head" | "head_oid" | "head_ref" | "index_oid" | "local_ref" | "managed_worktree_id"
  | "new_branch" | "new_name" | "patch_oid" | "pr_id" | "ref" | "remote" | "remote_ref" | "source"
  | "start" | "tag" | "target";
export type SemanticApprovalGitReferenceV1 = {
  role: SemanticApprovalGitReferenceRole;
  display: string;
  sha256: string;
};
export type SemanticApprovalArgumentProjectionV1 = {
  kind: SemanticApprovalArgumentKind;
  operation: string;
  display_argv: string[];
  argv_count: number | null;
  argv_truncated: boolean;
  argv_sha256: string | null;
  safe_cwd: string | null;
  cwd_sha256: string | null;
  endpoint_id: string | null;
  endpoint_id_sha256: string | null;
  method_id: string | null;
  method_id_sha256: string | null;
  server_id: string | null;
  server_id_sha256: string | null;
  tool_id: string | null;
  tool_id_sha256: string | null;
  registry_spec_sha256: string | null;
  registry_generation: number | null;
  payload_sha256: string | null;
  path_sha256: string | null;
  content_sha256: string | null;
  content_bytes: number | null;
  read_max_bytes: number | null;
  entry_limit: number | null;
  text_encoding: string | null;
  expected_content_sha256: string | null;
  overwrite: boolean | null;
  parents: boolean | null;
  exist_ok: boolean | null;
  recursive: boolean | null;
  missing_ok: boolean | null;
  timeout_seconds: string | null;
  continuous_session: boolean | null;
  network_access: boolean | null;
  worktree_id: string | null;
  worktree_id_sha256: string | null;
  repository_state_sha256: string | null;
  source_args_sha256: string | null;
  git_references: SemanticApprovalGitReferenceV1[];
  git_fact_tokens: string[];
};
export type SemanticTripCode =
  | "unsafe_review"
  | "critical_high_grant"
  | "cross_tenant"
  | "secret_egress"
  | "replay_detected"
  | "binding_mismatch"
  | "unauthorized_effect"
  | "provider_outcome_unknown";

export type CanonicalApprovalPreviewV1 = {
  schema_version: 1;
  request_id: string;
  revision: number;
  pid: string;
  action_id: string;
  resource_display: string;
  resource_sha256: string;
  rights: string[];
  effect_id: string;
  canonical_args_sha256: string;
  argument_projection: SemanticApprovalArgumentProjectionV1;
  target_state_sha256: string | null;
  risk: SemanticRisk;
  source_labels: {
    sensitivity: "public" | "normal" | "confidential" | "restricted" | "secret";
    integrity: "untrusted" | "unknown" | "checked" | "verified";
    trust_level: "untrusted" | "unknown" | "user_asserted" | "verified" | "trusted";
    identity_present: boolean;
    identity_mixed: boolean;
  };
  expires_at: string | null;
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
  revision: number;
  created_at: string;
  updated_at: string;
  approval_preview?: CanonicalApprovalPreviewV1;
  preview_sha256?: string;
  release_request_id?: string;
  release_for_request_id?: string;
};

const canonicalApprovalPreviewKeys = new Set([
  "schema_version", "request_id", "revision", "pid", "action_id", "resource_display", "resource_sha256", "rights", "effect_id",
  "canonical_args_sha256", "argument_projection", "target_state_sha256", "risk", "source_labels", "expires_at"
]);
const canonicalArgumentProjectionKeys = new Set([
  "kind", "operation", "display_argv", "argv_count", "argv_truncated", "argv_sha256", "safe_cwd", "cwd_sha256",
  "endpoint_id", "endpoint_id_sha256", "method_id", "method_id_sha256", "server_id", "server_id_sha256",
  "tool_id", "tool_id_sha256", "registry_spec_sha256", "registry_generation", "payload_sha256", "path_sha256", "content_sha256",
  "content_bytes", "read_max_bytes", "entry_limit", "text_encoding", "expected_content_sha256", "overwrite", "parents",
  "exist_ok", "recursive", "missing_ok", "timeout_seconds", "continuous_session", "network_access", "worktree_id",
  "worktree_id_sha256", "repository_state_sha256", "source_args_sha256", "git_references", "git_fact_tokens"
]);
const canonicalSourceLabelKeys = new Set([
  "sensitivity", "integrity", "trust_level", "identity_present", "identity_mixed"
]);
const canonicalApprovalRisks = new Set(["low", "medium", "high", "critical"]);
const canonicalApprovalSensitivity = new Set(["public", "normal", "confidential", "restricted", "secret"]);
const canonicalApprovalIntegrity = new Set(["untrusted", "unknown", "checked", "verified"]);
const canonicalApprovalTrust = new Set(["untrusted", "unknown", "user_asserted", "verified", "trusted"]);
const canonicalApprovalRights = new Set([
  "read", "write", "execute", "link", "diff", "materialize", "delete", "grant", "revoke", "approve", "admin"
]);
const canonicalApprovalAction = /^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$/;
const canonicalApprovalOperation = /^[a-z][a-z0-9_]{0,127}$/;
const canonicalApprovalIdentity = /^[A-Za-z0-9][A-Za-z0-9_.@+-]{0,255}$/;
const canonicalApprovalResource = /^[A-Za-z0-9][A-Za-z0-9_.:@+/*=-]{0,511}$/;
const canonicalApprovalGitFact = /^[a-z][a-z0-9_]{0,31}=(?:true|false|[a-z][a-z0-9_]{0,31}|[0-9]{1,16}|[0-9a-f]{64})$/;

function canonicalApprovalArgumentKind(actionId: string): SemanticApprovalArgumentKind {
  if (actionId === "pty.spawn") return "shell";
  const prefix = actionId.split(".", 1)[0];
  return prefix === "filesystem" || prefix === "shell" || prefix === "git"
    || prefix === "jsonrpc" || prefix === "mcp"
    ? prefix
    : "other";
}

function isCanonicalIdentityPair(display: unknown, digest: unknown): boolean {
  if (display === null && digest === null) return true;
  if (typeof display !== "string" || !isSha256(digest)) return false;
  if (!isApprovalProjectionText(display, 256)) return false;
  if (display === "<redacted>") return true;
  return canonicalApprovalIdentity.test(display) && sha256Utf8(display) === digest;
}

function isCanonicalResourceDisplay(display: string, digest: string): boolean {
  if (display === "<redacted>") return true;
  return canonicalApprovalResource.test(display) && sha256Utf8(display) === digest;
}

/** Validate the exact Host projection used as the only external-operation decision basis. */
export function canonicalApprovalPreviewFromRequest(request: HumanRequest): CanonicalApprovalPreviewV1 | null {
  const preview = request.approval_preview;
  if (request.payload.type !== "external_operation_approval"
      || !isRecord(preview)
      || Object.keys(preview).length !== canonicalApprovalPreviewKeys.size
      || Object.keys(preview).some((key) => !canonicalApprovalPreviewKeys.has(key))
      || preview.schema_version !== 1
      || !isApprovalProjectionText(preview.request_id, 512)
      || preview.request_id !== request.request_id
      || !isApprovalProjectionText(preview.pid, 512)
      || preview.pid !== request.pid
      || !isNonNegativeSafeInteger(preview.revision)
      || preview.revision !== request.revision
      || !isSemanticPublicText(preview.action_id, 128) || !canonicalApprovalAction.test(preview.action_id)
      || !isApprovalProjectionText(preview.resource_display, 512)
      || !isSha256(preview.resource_sha256)
      || !isCanonicalResourceDisplay(preview.resource_display, preview.resource_sha256)
      || !Array.isArray(preview.rights) || preview.rights.length < 1 || preview.rights.length > canonicalApprovalRights.size
      || !preview.rights.every((right) => typeof right === "string" && canonicalApprovalRights.has(right))
      || new Set(preview.rights).size !== preview.rights.length
      || !isApprovalProjectionText(preview.effect_id, 512)
      || !isSha256(preview.canonical_args_sha256)
      || !isCanonicalApprovalArgumentProjection(preview.argument_projection)
      || !(preview.target_state_sha256 === null || isSha256(preview.target_state_sha256))
      || typeof preview.risk !== "string" || !canonicalApprovalRisks.has(preview.risk)
      || !isRecord(preview.source_labels)
      || Object.keys(preview.source_labels).length !== canonicalSourceLabelKeys.size
      || Object.keys(preview.source_labels).some((key) => !canonicalSourceLabelKeys.has(key))
      || typeof preview.source_labels.sensitivity !== "string" || !canonicalApprovalSensitivity.has(preview.source_labels.sensitivity)
      || typeof preview.source_labels.integrity !== "string" || !canonicalApprovalIntegrity.has(preview.source_labels.integrity)
      || typeof preview.source_labels.trust_level !== "string" || !canonicalApprovalTrust.has(preview.source_labels.trust_level)
      || typeof preview.source_labels.identity_present !== "boolean"
      || typeof preview.source_labels.identity_mixed !== "boolean"
      || (preview.source_labels.identity_mixed && !preview.source_labels.identity_present)
      || !(preview.expires_at === null || isSemanticTimestamp(preview.expires_at, 128))
      || !isSha256(request.preview_sha256)) {
    return null;
  }
  const selected = preview as CanonicalApprovalPreviewV1;
  if (selected.argument_projection.kind !== canonicalApprovalArgumentKind(selected.action_id)) return null;
  if (canonicalApprovalPreviewSha256(selected) !== request.preview_sha256) return null;
  return selected;
}

function isCanonicalApprovalArgumentProjection(value: unknown): value is SemanticApprovalArgumentProjectionV1 {
  if (!isRecord(value)
      || Object.keys(value).length !== canonicalArgumentProjectionKeys.size
      || Object.keys(value).some((key) => !canonicalArgumentProjectionKeys.has(key))
      || !new Set(["filesystem", "shell", "git", "jsonrpc", "mcp", "other"]).has(value.kind as string)
      || !isApprovalProjectionText(value.operation, 128) || !canonicalApprovalOperation.test(value.operation)
      || !Array.isArray(value.display_argv) || value.display_argv.length > 16
      || !value.display_argv.every((item) => isApprovalProjectionText(item, 128))
      || value.display_argv.reduce((total, item) => total + item.length, 0) > 1024
      || !(value.argv_count === null || isNonNegativeSafeInteger(value.argv_count))
      || typeof value.argv_truncated !== "boolean"
      || !(value.argv_sha256 === null || isSha256(value.argv_sha256))
      || !(value.safe_cwd === null || isApprovalProjectionText(value.safe_cwd, 512))
      || !(value.cwd_sha256 === null || isSha256(value.cwd_sha256))
      || !isCanonicalIdentityPair(value.endpoint_id, value.endpoint_id_sha256)
      || !isCanonicalIdentityPair(value.method_id, value.method_id_sha256)
      || !isCanonicalIdentityPair(value.server_id, value.server_id_sha256)
      || !isCanonicalIdentityPair(value.tool_id, value.tool_id_sha256)
      || !(value.registry_spec_sha256 === null || isSha256(value.registry_spec_sha256))
      || !(value.registry_generation === null || isNonNegativeSafeInteger(value.registry_generation))
      || ((value.registry_spec_sha256 === null) !== (value.registry_generation === null))
      || !(value.payload_sha256 === null || isSha256(value.payload_sha256))
      || !(value.path_sha256 === null || isSha256(value.path_sha256))
      || !(value.content_sha256 === null || isSha256(value.content_sha256))
      || !(value.content_bytes === null || isNonNegativeSafeInteger(value.content_bytes))
      || !(value.read_max_bytes === null || isNonNegativeSafeInteger(value.read_max_bytes))
      || !(value.entry_limit === null || isNonNegativeSafeInteger(value.entry_limit))
      || !(value.text_encoding === null || isApprovalProjectionText(value.text_encoding, 64))
      || !(value.expected_content_sha256 === null || value.expected_content_sha256 === "missing" || isSha256(value.expected_content_sha256))
      || !(value.overwrite === null || typeof value.overwrite === "boolean")
      || !(value.parents === null || typeof value.parents === "boolean")
      || !(value.exist_ok === null || typeof value.exist_ok === "boolean")
      || !(value.recursive === null || typeof value.recursive === "boolean")
      || !(value.missing_ok === null || typeof value.missing_ok === "boolean")
      || !(value.timeout_seconds === null || (typeof value.timeout_seconds === "string" && /^(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,9})?$/.test(value.timeout_seconds)))
      || !(value.continuous_session === null || typeof value.continuous_session === "boolean")
      || !(value.network_access === null || typeof value.network_access === "boolean")
      || !isCanonicalIdentityPair(value.worktree_id, value.worktree_id_sha256)
      || !(value.repository_state_sha256 === null || isSha256(value.repository_state_sha256))
      || !(value.source_args_sha256 === null || isSha256(value.source_args_sha256))
      || !Array.isArray(value.git_references) || value.git_references.length > 16
      || !value.git_references.every(isCanonicalGitReference)
      || new Set(value.git_references.map((item) => item.role)).size !== value.git_references.length
      || (value.git_references as SemanticApprovalGitReferenceV1[]).some(
        (item, index, values) => index > 0 && (values[index - 1] as SemanticApprovalGitReferenceV1).role >= item.role
      )
      || !Array.isArray(value.git_fact_tokens) || value.git_fact_tokens.length > 32
      || !value.git_fact_tokens.every((item) => typeof item === "string" && canonicalApprovalGitFact.test(item))
      || new Set(value.git_fact_tokens).size !== value.git_fact_tokens.length
      || (value.git_fact_tokens as string[]).some(
        (item, index, values) => index > 0 && (values[index - 1] as string) >= item
      )) {
    return false;
  }
  const projection = value as SemanticApprovalArgumentProjectionV1;
  const hasShell = projection.display_argv.length > 0 || projection.argv_count !== null || projection.argv_truncated
    || projection.argv_sha256 !== null || projection.safe_cwd !== null || projection.cwd_sha256 !== null
    || projection.timeout_seconds !== null || projection.continuous_session !== null || projection.network_access !== null;
  const hasRemote = projection.endpoint_id !== null || projection.endpoint_id_sha256 !== null
    || projection.method_id !== null || projection.method_id_sha256 !== null
    || projection.server_id !== null || projection.server_id_sha256 !== null
    || projection.tool_id !== null || projection.tool_id_sha256 !== null
    || projection.registry_spec_sha256 !== null || projection.registry_generation !== null;
  const hasFilesystemOnly = projection.content_sha256 !== null || projection.content_bytes !== null
    || projection.read_max_bytes !== null || projection.entry_limit !== null || projection.text_encoding !== null
    || projection.expected_content_sha256 !== null || projection.overwrite !== null || projection.parents !== null
    || projection.exist_ok !== null || projection.recursive !== null || projection.missing_ok !== null;
  const hasFile = projection.path_sha256 !== null || hasFilesystemOnly;
  const hasGit = projection.worktree_id !== null || projection.worktree_id_sha256 !== null
    || projection.repository_state_sha256 !== null || projection.source_args_sha256 !== null
    || projection.git_references.length > 0
    || projection.git_fact_tokens.length > 0;
  if (projection.kind === "filesystem") {
    return projection.path_sha256 !== null && projection.payload_sha256 === null && !hasShell && !hasRemote && !hasGit
      && ((projection.content_sha256 === null) === (projection.content_bytes === null));
  }
  if (projection.kind === "shell") {
    return projection.argv_count !== null && projection.argv_count > 0
      && projection.argv_sha256 !== null && projection.cwd_sha256 !== null
      && projection.argv_truncated === (projection.argv_count > projection.display_argv.length)
      && !hasRemote && !hasFile && !hasGit && projection.payload_sha256 === null;
  }
  if (projection.kind === "jsonrpc") {
    return projection.endpoint_id !== null && projection.endpoint_id_sha256 !== null
      && projection.method_id !== null && projection.method_id_sha256 !== null && projection.payload_sha256 !== null
      && projection.registry_spec_sha256 !== null && projection.registry_generation !== null
      && projection.server_id === null && projection.server_id_sha256 === null
      && projection.tool_id === null && projection.tool_id_sha256 === null && !hasShell && !hasFile && !hasGit;
  }
  if (projection.kind === "mcp") {
    return projection.server_id !== null && projection.server_id_sha256 !== null
      && projection.tool_id !== null && projection.tool_id_sha256 !== null && projection.payload_sha256 !== null
      && projection.registry_spec_sha256 !== null && projection.registry_generation !== null
      && projection.endpoint_id === null && projection.endpoint_id_sha256 === null
      && projection.method_id === null && projection.method_id_sha256 === null && !hasShell && !hasFile && !hasGit;
  }
  if (projection.kind === "git") {
    return projection.worktree_id !== null && projection.worktree_id_sha256 !== null
      && !hasShell && !hasRemote && !hasFilesystemOnly && projection.payload_sha256 === null;
  }
  return !hasShell && !hasRemote && !hasFile && !hasGit && projection.payload_sha256 === null;
}

const canonicalGitReferenceRoles = new Set<SemanticApprovalGitReferenceRole>([
  "base", "base_oid", "base_ref", "branch", "expected_remote_oid", "git_old_oid", "git_remote",
  "git_remote_ref", "head", "head_oid", "head_ref", "index_oid", "local_ref", "managed_worktree_id",
  "new_branch", "new_name", "patch_oid", "pr_id", "ref", "remote", "remote_ref", "source", "start",
  "tag", "target"
]);
const canonicalGitReferenceDisplay = /^[A-Za-z0-9][A-Za-z0-9._/@:+~^{}-]{0,255}$/;

function isCanonicalGitReference(value: unknown): value is SemanticApprovalGitReferenceV1 {
  if (!isRecord(value)
      || Object.keys(value).length !== 3
      || Object.keys(value).some((key) => !new Set(["role", "display", "sha256"]).has(key))
      || typeof value.role !== "string" || !canonicalGitReferenceRoles.has(value.role as SemanticApprovalGitReferenceRole)
      || !isApprovalProjectionText(value.display, 256)
      || !isSha256(value.sha256)) {
    return false;
  }
  return value.display === "<redacted>"
    || (canonicalGitReferenceDisplay.test(value.display) && sha256Utf8(value.display) === value.sha256);
}

/** Match Semantic Host compact sorted-key UTF-8 canonical JSON byte-for-byte. */
export function canonicalApprovalPreviewSha256(preview: CanonicalApprovalPreviewV1): string {
  return sha256Utf8(hostCanonicalJson(preview));
}

function hostCanonicalJson(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(hostCanonicalJson).join(",")}]`;
  if (isRecord(value)) {
    return `{${Object.keys(value)
      .sort((left, right) => left < right ? -1 : left > right ? 1 : 0)
      .map((key) => `${JSON.stringify(key)}:${hostCanonicalJson(value[key])}`)
      .join(",")}}`;
  }
  throw new Error("Canonical approval preview contains an unsupported JSON value.");
}

const sha256RoundConstants = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]);

function rotateRight(value: number, bits: number): number {
  return (value >>> bits) | (value << (32 - bits));
}

function sha256Utf8(value: string): string {
  const input = new TextEncoder().encode(value);
  const byteLength = Math.ceil((input.length + 9) / 64) * 64;
  const padded = new Uint8Array(byteLength);
  padded.set(input);
  padded[input.length] = 0x80;
  const bitLength = input.length * 8;
  const view = new DataView(padded.buffer);
  view.setUint32(byteLength - 8, Math.floor(bitLength / 0x100000000), false);
  view.setUint32(byteLength - 4, bitLength >>> 0, false);
  const hash = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
  ]);
  const words = new Uint32Array(64);
  for (let offset = 0; offset < byteLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) words[index] = view.getUint32(offset + index * 4, false);
    for (let index = 16; index < 64; index += 1) {
      const previous15 = words[index - 15] as number;
      const previous2 = words[index - 2] as number;
      const sigma0 = rotateRight(previous15, 7) ^ rotateRight(previous15, 18) ^ (previous15 >>> 3);
      const sigma1 = rotateRight(previous2, 17) ^ rotateRight(previous2, 19) ^ (previous2 >>> 10);
      words[index] = ((words[index - 16] as number) + sigma0 + (words[index - 7] as number) + sigma1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = hash;
    for (let index = 0; index < 64; index += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choose = (e & f) ^ (~e & g);
      const temporary1 = (h + sum1 + choose + (sha256RoundConstants[index] as number) + (words[index] as number)) >>> 0;
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temporary2 = (sum0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temporary1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temporary1 + temporary2) >>> 0;
    }
    hash[0] = ((hash[0] as number) + a) >>> 0;
    hash[1] = ((hash[1] as number) + b) >>> 0;
    hash[2] = ((hash[2] as number) + c) >>> 0;
    hash[3] = ((hash[3] as number) + d) >>> 0;
    hash[4] = ((hash[4] as number) + e) >>> 0;
    hash[5] = ((hash[5] as number) + f) >>> 0;
    hash[6] = ((hash[6] as number) + g) >>> 0;
    hash[7] = ((hash[7] as number) + h) >>> 0;
  }
  return [...hash].map((word) => word.toString(16).padStart(8, "0")).join("");
}

export function assertHumanRequest(value: unknown): asserts value is HumanRequest {
  if (!isRecord(value)
      || !isBoundedString(value.request_id, 1024)
      || !isBoundedString(value.pid, 1024)
      || !isBoundedString(value.human, 512)
      || !isRecord(value.payload)
      || !isBoundedString(value.status, 128)
      || !(value.decision === null || isRecord(value.decision))
      || typeof value.blocking !== "boolean"
      || !isNonNegativeSafeInteger(value.revision)
      || !isBoundedString(value.created_at, 128)
      || !isBoundedString(value.updated_at, 128)
      || !(value.release_request_id === undefined || isBoundedString(value.release_request_id, 1024))
      || !(value.release_for_request_id === undefined || isBoundedString(value.release_for_request_id, 1024))
      || !(value.approval_preview === undefined || isRecord(value.approval_preview))
      || !(value.preview_sha256 === undefined || isSha256(value.preview_sha256))) {
    throw new Error("GUI human request response is malformed.");
  }
  const request = value as HumanRequest;
  const hasPreview = request.approval_preview !== undefined || request.preview_sha256 !== undefined;
  if (hasPreview && canonicalApprovalPreviewFromRequest(request) === null) {
    throw new Error("GUI human request canonical approval preview is malformed.");
  }
}

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
  | {
      kind: "external_approval";
      approved: boolean;
      expected_revision: number;
      preview_sha256: string;
    }
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

export type SemanticMode = "off" | "shadow" | "enforce_deny" | "canary_auto";
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
  | "low_integrity"
  | "data_flow_denied"
  | "flow_coverage_incomplete"
  | "policy_hard_deny"
  | "tenant_not_allowed"
  | "budget_exhausted"
  | "control_disabled"
  | "control_tripped"
  | "confidence_too_low"
  | "calibration_too_low"
  | "digest_drift"
  | "revision_race_lost"
  | "capability_expired"
  | "capability_revoked";
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
  schema_version: 3;
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
  control: {
    catalog_version: 1 | null;
    active_epoch_id: string | null;
    active_epoch_sha256: string | null;
    generation: number;
    state: "inactive" | "active" | "tripped" | "revoked";
    trip_reason_code: SemanticTripCode | null;
  };
  flow: SemanticFlowStatus;
  machine: {
    eligible: number;
    issued: number;
    consumed: number;
    succeeded: number;
    failed: number;
    unknown: number;
    expired: number;
    revoked: number;
    race_lost: number;
    denied: number;
  };
  actual_auto_approval: {
    numerator: number;
    denominator: number;
    rate: number | null;
  };
  review_metrics: {
    reviewed: number;
    safe: number;
    unsafe: number;
    unsafe_rate: number | null;
    issued_reviewed: number;
    issued_review_rate: number | null;
  };
};

export type SemanticFlowCoverage = "complete" | "partial" | "unknown" | "conflict" | "stale";

export type SemanticLegacyFlowHistory = {
  present: boolean;
  source_schema_version: 5 | null;
  assessment_count: number;
  coverage: "unknown" | null;
  evidence_sha256: string | null;
  created_at: string | null;
};

export type SemanticFlowStatus = {
  schema_version: 1;
  available: boolean;
  counts: {
    entities: number;
    activities: number;
    edges: number;
    label_assertions: number;
  };
  coverage: Record<SemanticFlowCoverage, number>;
  capture_failures: number;
  legacy_history: SemanticLegacyFlowHistory;
};

export type SemanticFlowLabels = {
  sensitivity: SemanticSensitivity;
  trust_level: SemanticTrust;
  integrity: SemanticIntegrity;
};

export type SemanticFlowEntity = {
  schema_version: 1;
  entity_id: string;
  kind: "root_goal" | "object_version" | "file_binding_version" | "provider_result" | "tool_result" | "materialization" | "model_output";
  pid: string | null;
  tenant_bucket_sha256: string;
  content_sha256: string;
  version_sha256: string;
  provenance_sha256: string;
  baseline_labels: SemanticFlowLabels;
  coverage: SemanticFlowCoverage;
  identity_present: boolean;
  identity_mixed: boolean;
  created_at: string;
};

export type SemanticFlowActivity = {
  schema_version: 1;
  activity_id: string;
  kind: "process_spawn" | "provider_call" | "tool_call" | "llm_call" | "object_create" | "object_update" | "object_append" | "object_materialize" | "object_read" | "file_read" | "file_write" | "transformation" | "aggregation" | "conditional" | "tool_selection" | "memory_retrieval";
  pid: string;
  action_id: string | null;
  effect_id: string | null;
  state_sha256: string;
  provider_spec_sha256: string | null;
  tool_schema_sha256: string | null;
  model_artifact_sha256: string | null;
  tenant_bucket_sha256: string;
  created_at: string;
};

export type SemanticFlowNodeType = "entity" | "activity";
export type SemanticFlowDirection = "upstream" | "downstream";

export type SemanticFlowEdge = {
  schema_version: 1;
  edge_id: string;
  relation: "direct" | "indirect" | "control";
  source_node_id: string;
  source_node_type: SemanticFlowNodeType;
  target_node_id: string;
  target_node_type: SemanticFlowNodeType;
  pid: string;
  provenance_sha256: string;
  created_at: string;
};

export type SemanticFlowEntityPage = SemanticReadOnlyPage<SemanticFlowEntity>;
export type SemanticFlowEdgePage = SemanticReadOnlyPage<SemanticFlowEdge>;

export type SemanticFlowLineageItem = {
  depth: number;
  edge: SemanticFlowEdge;
  node_type: SemanticFlowNodeType;
  node: SemanticFlowEntity | SemanticFlowActivity;
};

export type SemanticFlowLineage = {
  schema_version: 1;
  root_node_id: string;
  direction: SemanticFlowDirection;
  items: SemanticFlowLineageItem[];
  effective_labels: SemanticFlowLabels | null;
  coverage: SemanticFlowCoverage;
  next_cursor: string | null;
  truncated: boolean;
};

export type SemanticSettlementOutcome =
  | "issued"
  | "denied"
  | "require_human"
  | "race_lost"
  | "stale"
  | "budget_exhausted"
  | "revoked"
  | "expired"
  | "failed";

export type SemanticMachineSettlement = {
  schema_version: 1;
  settlement_id: string;
  assessment_id: string | null;
  job_id: string | null;
  request_id: string;
  request_revision: number;
  pid: string;
  operation_id: string | null;
  effect_id: string;
  epoch_id: string;
  policy_sha256: string;
  tenant_bucket_sha256: string;
  action_id: string;
  outcome: SemanticSettlementOutcome;
  capability_id: string | null;
  binding_sha256: string;
  decision_sha256: string;
  matched_rule_id: string | null;
  reason_codes: SemanticReasonCode[];
  created_at: string;
  human_outcome: "approved" | "rejected" | "cancelled" | null;
  human_outcome_source: "human" | "machine_policy" | "cancel" | null;
  human_outcome_request_revision: number | null;
  human_outcome_decision_sha256: string | null;
  human_outcome_created_at: string | null;
};

export type SemanticPolicyEpochSummary = {
  schema_version: 1;
  epoch_id: string;
  generation: number;
  catalog_version: 1;
  policy_sha256: string;
  expected_previous_sha256: string | null;
  created_at: string;
};

export type SemanticControlState = {
  schema_version: 1;
  revision: number;
  generation: number;
  mode: SemanticMode;
  active_epoch_id: string | null;
  active_policy_sha256: string | null;
  tripped: boolean;
  trip_code: SemanticTripCode | null;
  updated_at: string;
};

export type SemanticHealthSeverity = "info" | "warning" | "critical";
export type SemanticHealthEventKind =
  | "semantic_control_disable_conflict"
  | "semantic_control_authority_cleared"
  | "semantic_policy_activated"
  | "semantic_policy_rotated"
  | "semantic_control_startup_conflict"
  | "capture_failed"
  | "semantic_unsafe_review_control_unsettled"
  | "semantic_unsafe_review_fallback_trip"
  | `semantic_safety_trip:${SemanticTripCode}`;

export type SemanticHealthEvent = {
  schema_version: 1;
  event_id: string;
  event_kind: SemanticHealthEventKind;
  severity: SemanticHealthSeverity;
  epoch_id: string | null;
  tenant_bucket_sha256: string | null;
  evidence_sha256: string;
  created_at: string;
};

export type SemanticMachineCounters = SemanticStatus["machine"];
export type SemanticActualApprovalMetrics = SemanticStatus["actual_auto_approval"];
export type SemanticReviewMetrics = SemanticStatus["review_metrics"];
export type SemanticCanaryReviewMetrics = SemanticReviewMetrics;

export type SemanticMetrics = {
  schema_version: 1;
  window: string | null;
  action_id: string | null;
  tenant_bucket_sha256: string | null;
  epoch_id: string | null;
  risk: SemanticRisk | null;
  machine: SemanticMachineCounters;
  actual_auto_approval: SemanticActualApprovalMetrics;
  review_metrics: SemanticCanaryReviewMetrics;
};

export type SemanticReadOnlyPage<T> = {
  schema_version: 1;
  items: T[];
  next_cursor: string | null;
};

export type SemanticMachineSettlementPage = SemanticReadOnlyPage<SemanticMachineSettlement>;
export type SemanticPolicyEpochPage = SemanticReadOnlyPage<SemanticPolicyEpochSummary>;
export type SemanticControlHistoryPage = SemanticReadOnlyPage<SemanticControlState>;
export type SemanticHealthEventPage = SemanticReadOnlyPage<SemanticHealthEvent>;

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

const semanticModeValues = new Set(["off", "shadow", "enforce_deny", "canary_auto"]);
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
  "prompt_injection", "mixed_identity", "low_integrity", "data_flow_denied", "flow_coverage_incomplete",
  "policy_hard_deny", "tenant_not_allowed", "budget_exhausted", "control_disabled", "control_tripped",
  "confidence_too_low", "calibration_too_low", "digest_drift", "revision_race_lost", "capability_expired",
  "capability_revoked"
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
const semanticStatusKeys = new Set([
  "schema_version", "mode", "adapter", "profile_id", "control", "queue", "assessments", "flow", "machine",
  "actual_auto_approval", "review_metrics"
]);
const semanticQueueKeys = new Set(["queued", "leased", "succeeded", "failed", "cancelled", "capture_failures"]);
const semanticCountKeys = new Set([
  "total", "success", "error", "ood", "would_issue_exact_once", "would_deny", "require_human", "by_status", "by_domain"
]);
const semanticScalarCountKeys = new Set(["total", "success", "error", "ood", "would_issue_exact_once", "would_deny", "require_human"]);
const semanticActualApprovalKeys = new Set(["numerator", "denominator", "rate"]);
const semanticStatusControlKeys = new Set([
  "catalog_version", "active_epoch_id", "active_epoch_sha256", "generation", "state", "trip_reason_code"
]);
const semanticStatusControlStates = new Set(["inactive", "active", "tripped", "revoked"]);
const semanticTripCodes = new Set([
  "unsafe_review", "critical_high_grant", "cross_tenant", "secret_egress", "replay_detected",
  "binding_mismatch", "unauthorized_effect", "provider_outcome_unknown"
]);
const semanticMachineKeys = new Set([
  "eligible", "issued", "consumed", "succeeded", "failed", "unknown", "expired", "revoked", "race_lost", "denied"
]);
const semanticReviewMetricKeys = new Set([
  "reviewed", "safe", "unsafe", "unsafe_rate", "issued_reviewed", "issued_review_rate"
]);
const semanticFlowStatusKeys = new Set([
  "schema_version", "available", "counts", "coverage", "capture_failures", "legacy_history"
]);
const semanticLegacyFlowHistoryKeys = new Set([
  "present", "source_schema_version", "assessment_count", "coverage", "evidence_sha256", "created_at"
]);
const semanticFlowCountKeys = new Set(["entities", "activities", "edges", "label_assertions"]);
const semanticFlowCoverages = new Set(["complete", "partial", "unknown", "conflict", "stale"]);
const semanticFlowLabelsKeys = new Set(["sensitivity", "trust_level", "integrity"]);
const semanticFlowEntityKeys = new Set([
  "schema_version", "entity_id", "kind", "pid", "tenant_bucket_sha256", "content_sha256", "version_sha256",
  "provenance_sha256", "baseline_labels", "coverage", "identity_present", "identity_mixed", "created_at"
]);
const semanticFlowActivityKeys = new Set([
  "schema_version", "activity_id", "kind", "pid", "action_id", "effect_id", "state_sha256",
  "provider_spec_sha256", "tool_schema_sha256", "model_artifact_sha256", "tenant_bucket_sha256", "created_at"
]);
const semanticFlowEdgeKeys = new Set([
  "schema_version", "edge_id", "relation", "source_node_id", "source_node_type", "target_node_id",
  "target_node_type", "pid", "provenance_sha256", "created_at"
]);
const semanticPageV1Keys = new Set(["schema_version", "items", "next_cursor"]);
const semanticFlowLineageKeys = new Set([
  "schema_version", "root_node_id", "direction", "items", "effective_labels", "coverage", "next_cursor", "truncated"
]);
const semanticFlowLineageItemKeys = new Set(["depth", "edge", "node_type", "node"]);
const semanticFlowNodeTypes = new Set(["entity", "activity"]);
const semanticFlowDirections = new Set(["upstream", "downstream"]);
const semanticFlowEntityKinds = new Set([
  "root_goal", "object_version", "file_binding_version", "provider_result", "tool_result", "materialization", "model_output"
]);
const semanticFlowActivityKinds = new Set([
  "process_spawn", "provider_call", "tool_call", "llm_call", "object_create", "object_update", "object_append",
  "object_materialize", "object_read", "file_read", "file_write", "transformation", "aggregation", "conditional",
  "tool_selection", "memory_retrieval"
]);
const semanticFlowRelations = new Set(["direct", "indirect", "control"]);
const semanticSettlementKeys = new Set([
  "schema_version", "settlement_id", "assessment_id", "job_id", "request_id", "request_revision", "pid",
  "operation_id", "effect_id", "epoch_id", "policy_sha256", "tenant_bucket_sha256", "action_id", "outcome",
  "capability_id", "binding_sha256", "decision_sha256", "matched_rule_id", "reason_codes", "created_at",
  "human_outcome", "human_outcome_source", "human_outcome_request_revision", "human_outcome_decision_sha256",
  "human_outcome_created_at"
]);
const semanticSettlementOutcomes = new Set([
  "issued", "denied", "require_human", "race_lost", "stale", "budget_exhausted", "revoked", "expired", "failed"
]);
const semanticExecutableDenyReasons = new Set([
  "hard_policy_violation", "malformed_request", "stale_binding", "stale_manifest", "stale_policy",
  "data_flow_denied", "policy_hard_deny", "digest_drift"
]);
const semanticPolicyEpochKeys = new Set([
  "schema_version", "epoch_id", "generation", "catalog_version", "policy_sha256", "expected_previous_sha256", "created_at"
]);
const semanticControlStateKeys = new Set([
  "schema_version", "revision", "generation", "mode", "active_epoch_id", "active_policy_sha256", "tripped",
  "trip_code", "updated_at"
]);
const semanticHealthEventKeys = new Set([
  "schema_version", "event_id", "event_kind", "severity", "epoch_id", "tenant_bucket_sha256", "evidence_sha256", "created_at"
]);
const semanticHealthSeverities = new Set(["info", "warning", "critical"]);
const semanticHealthEventKinds = new Set<SemanticHealthEventKind>([
  "semantic_control_disable_conflict",
  "semantic_control_authority_cleared",
  "semantic_policy_activated",
  "semantic_policy_rotated",
  "semantic_control_startup_conflict",
  "capture_failed",
  "semantic_unsafe_review_control_unsettled",
  "semantic_unsafe_review_fallback_trip",
  ...[...semanticTripCodes].map((code) => `semantic_safety_trip:${code}` as SemanticHealthEventKind)
]);
const semanticMetricsKeys = new Set([
  "schema_version", "window", "action_id", "tenant_bucket_sha256", "epoch_id", "risk", "machine", "actual_auto_approval",
  "review_metrics"
]);
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
const semanticPublicIdentifierPattern = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,511}$/;
const semanticRedactedIntentMaxChars = 2_000;
const semanticActionPattern = /^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$/;

/** Reject malformed, unbounded, or private Semantic status fields before rendering. */
export function assertSemanticStatus(value: unknown): asserts value is SemanticStatus {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticStatusKeys) || value.schema_version !== 3
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
      || !isSemanticStatusControl(value.control)
      || !isSemanticFlowStatus(value.flow)
      || !isExactNonNegativeCounterRecord(value.machine, semanticMachineKeys)
      || !isRecord(value.actual_auto_approval)
      || !hasOnlyKeys(value.actual_auto_approval, semanticActualApprovalKeys)
      || !isConsistentRate(value.actual_auto_approval, "numerator", "denominator", "rate")
      || value.actual_auto_approval.numerator !== (value.machine as Record<string, unknown>).issued
      || value.actual_auto_approval.denominator !== (value.machine as Record<string, unknown>).eligible
      || !isSemanticReviewMetrics(value.review_metrics, value.machine)) {
    throw new Error("GUI Semantic status is malformed.");
  }
}

function isSemanticStatusControl(value: unknown): boolean {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticStatusControlKeys)
      || !(value.catalog_version === null || value.catalog_version === 1)
      || !isNullableSemanticPublicText(value.active_epoch_id, 512)
      || !isNullableSha256(value.active_epoch_sha256)
      || ((value.active_epoch_id === null) !== (value.active_epoch_sha256 === null))
      || !isNonNegativeSafeInteger(value.generation)
      || typeof value.state !== "string" || !semanticStatusControlStates.has(value.state)
      || !(value.trip_reason_code === null
        || (typeof value.trip_reason_code === "string" && semanticTripCodes.has(value.trip_reason_code)))) return false;
  if (value.state === "inactive") {
    return value.catalog_version === null && value.active_epoch_id === null && value.trip_reason_code === null;
  }
  if (value.catalog_version !== 1 || value.active_epoch_id === null) return false;
  return (value.state === "tripped") === (value.trip_reason_code !== null);
}

export function assertSemanticFlowStatus(value: unknown): asserts value is SemanticFlowStatus {
  if (!isSemanticFlowStatus(value)) throw new Error("GUI Semantic flow status is malformed.");
}

function isSemanticFlowStatus(value: unknown): boolean {
  return isRecord(value)
    && hasOnlyKeys(value, semanticFlowStatusKeys)
    && value.schema_version === 1
    && typeof value.available === "boolean"
    && isExactNonNegativeCounterRecord(value.counts, semanticFlowCountKeys)
    && isNonNegativeSafeInteger(value.capture_failures)
    && isExactNonNegativeCounterRecord(value.coverage, semanticFlowCoverages)
    && isSemanticLegacyFlowHistory(value.legacy_history);
}

function isSemanticLegacyFlowHistory(value: unknown): boolean {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticLegacyFlowHistoryKeys)
      || typeof value.present !== "boolean"
      || !isNonNegativeSafeInteger(value.assessment_count)) return false;
  if (!value.present) {
    return value.source_schema_version === null
      && value.assessment_count === 0
      && value.coverage === null
      && value.evidence_sha256 === null
      && value.created_at === null;
  }
  return value.source_schema_version === 5
    && value.coverage === "unknown"
    && isSha256(value.evidence_sha256)
    && isSemanticTimestamp(value.created_at, 64);
}

function isConsistentRate(
  value: Record<string, unknown>,
  numeratorKey: string,
  denominatorKey: string,
  rateKey: string
): boolean {
  const numerator = value[numeratorKey];
  const denominator = value[denominatorKey];
  const rate = value[rateKey];
  if (!isNonNegativeSafeInteger(numerator) || !isNonNegativeSafeInteger(denominator)
      || Number(numerator) > Number(denominator)) return false;
  if (denominator === 0) return rate === null;
  return typeof rate === "number" && Number.isFinite(rate) && rate >= 0 && rate <= 1
    && Math.abs(rate - Number(numerator) / Number(denominator)) <= 1e-12;
}

export function assertSemanticFlowEntityPage(value: unknown): asserts value is SemanticFlowEntityPage {
  assertSemanticReadOnlyPage(value, assertSemanticFlowEntity, "flow entity");
}

export function assertSemanticFlowEdgePage(value: unknown): asserts value is SemanticFlowEdgePage {
  assertSemanticReadOnlyPage(value, assertSemanticFlowEdge, "flow edge");
}

export function assertSemanticFlowLineage(value: unknown): asserts value is SemanticFlowLineage {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticFlowLineageKeys) || value.schema_version !== 1
      || !isSemanticPublicIdentifier(value.root_node_id)
      || typeof value.direction !== "string" || !semanticFlowDirections.has(value.direction)
      || !Array.isArray(value.items) || value.items.length > 100
      || !(value.effective_labels === null || isSemanticFlowLabels(value.effective_labels))
      || typeof value.coverage !== "string" || !semanticFlowCoverages.has(value.coverage)
      || !isNullableBoundedString(value.next_cursor, 2048)
      || typeof value.truncated !== "boolean") {
    throw new Error("GUI Semantic flow lineage is malformed.");
  }
  for (const item of value.items) {
    if (!isRecord(item) || !hasOnlyKeys(item, semanticFlowLineageItemKeys)
        || !isNonNegativeSafeInteger(item.depth) || Number(item.depth) > 16
        || typeof item.node_type !== "string" || !semanticFlowNodeTypes.has(item.node_type)) {
      throw new Error("GUI Semantic flow lineage item is malformed.");
    }
    assertSemanticFlowEdge(item.edge);
    if (item.node_type === "entity") assertSemanticFlowEntity(item.node);
    else assertSemanticFlowActivity(item.node);
    const expectedNodeId = value.direction === "upstream" ? item.edge.source_node_id : item.edge.target_node_id;
    const expectedNodeType = value.direction === "upstream" ? item.edge.source_node_type : item.edge.target_node_type;
    const actualNodeId = "entity_id" in item.node ? item.node.entity_id : item.node.activity_id;
    if (item.node_type !== expectedNodeType || actualNodeId !== expectedNodeId) {
      throw new Error("GUI Semantic flow lineage node is not bound to its edge endpoint.");
    }
  }
}

export function assertSemanticMachineSettlementPage(value: unknown): asserts value is SemanticMachineSettlementPage {
  assertSemanticReadOnlyPage(value, assertSemanticMachineSettlement, "machine settlement");
}

export function assertSemanticPolicyEpochPage(value: unknown): asserts value is SemanticPolicyEpochPage {
  assertSemanticReadOnlyPage(value, assertSemanticPolicyEpochSummary, "policy epoch");
}

export function assertSemanticControlState(value: unknown): asserts value is SemanticControlState {
  if (!isSemanticControlStateValue(value)) {
    throw new Error("GUI Semantic control state is malformed.");
  }
}

function isSemanticControlStateValue(value: unknown): value is SemanticControlState {
  return isRecord(value) && hasOnlyKeys(value, semanticControlStateKeys) && value.schema_version === 1
    && isNonNegativeSafeInteger(value.revision) && isNonNegativeSafeInteger(value.generation)
    && typeof value.mode === "string" && semanticModeValues.has(value.mode)
    && isNullableSemanticPublicText(value.active_epoch_id, 512)
    && isNullableSha256(value.active_policy_sha256)
    && typeof value.tripped === "boolean"
    && (value.trip_code === null || (typeof value.trip_code === "string" && semanticTripCodes.has(value.trip_code)))
    && isSemanticTimestamp(value.updated_at, 64)
    && ((value.active_epoch_id === null) === (value.active_policy_sha256 === null))
    && (value.tripped === (value.trip_code !== null))
    && (["enforce_deny", "canary_auto"].includes(value.mode)
      ? value.active_epoch_id !== null
      : value.active_epoch_id === null && !value.tripped);
}

export function assertSemanticControlHistoryPage(value: unknown): asserts value is SemanticControlHistoryPage {
  assertSemanticReadOnlyPage(value, assertSemanticControlState, "control history");
}

export function assertSemanticHealthEventPage(value: unknown): asserts value is SemanticHealthEventPage {
  assertSemanticReadOnlyPage(value, assertSemanticHealthEvent, "health event");
}

export function assertSemanticMetrics(value: unknown): asserts value is SemanticMetrics {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticMetricsKeys) || value.schema_version !== 1
      || !isNullableSemanticPublicText(value.window, 512)
      || !(value.action_id === null || isSemanticAction(value.action_id))
      || !isNullableSha256(value.tenant_bucket_sha256)
      || !isNullableSemanticPublicText(value.epoch_id, 512)
      || !(value.risk === null || (typeof value.risk === "string" && canonicalApprovalRisks.has(value.risk)))
      || !isExactNonNegativeCounterRecord(value.machine, semanticMachineKeys)
      || !isRecord(value.actual_auto_approval)
      || !hasOnlyKeys(value.actual_auto_approval, semanticActualApprovalKeys)
      || !isConsistentRate(value.actual_auto_approval, "numerator", "denominator", "rate")
      || value.actual_auto_approval.numerator !== (value.machine as Record<string, unknown>).issued
      || value.actual_auto_approval.denominator !== (value.machine as Record<string, unknown>).eligible
      || !isSemanticReviewMetrics(value.review_metrics, value.machine)) {
    throw new Error("GUI Semantic metrics are malformed.");
  }
}

function assertSemanticFlowEntity(value: unknown): asserts value is SemanticFlowEntity {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticFlowEntityKeys) || value.schema_version !== 1
      || !isSemanticPublicIdentifier(value.entity_id)
      || typeof value.kind !== "string" || !semanticFlowEntityKinds.has(value.kind)
      || !isNullableSemanticPublicIdentifier(value.pid)
      || !isSha256(value.tenant_bucket_sha256) || !isSha256(value.content_sha256)
      || !isSha256(value.version_sha256) || !isSha256(value.provenance_sha256)
      || !isSemanticFlowLabels(value.baseline_labels)
      || typeof value.coverage !== "string" || !semanticFlowCoverages.has(value.coverage)
      || typeof value.identity_present !== "boolean" || typeof value.identity_mixed !== "boolean"
      || (value.identity_mixed && !value.identity_present)
      || !isSemanticTimestamp(value.created_at, 64)) {
    throw new Error("GUI Semantic flow entity is malformed.");
  }
}

function assertSemanticFlowActivity(value: unknown): asserts value is SemanticFlowActivity {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticFlowActivityKeys) || value.schema_version !== 1
      || !isSemanticPublicIdentifier(value.activity_id)
      || typeof value.kind !== "string" || !semanticFlowActivityKinds.has(value.kind)
      || !isSemanticPublicIdentifier(value.pid)
      || !(value.action_id === null || isSemanticAction(value.action_id))
      || !isNullableSemanticPublicIdentifier(value.effect_id)
      || !isSha256(value.state_sha256) || !isNullableSha256(value.provider_spec_sha256)
      || !isNullableSha256(value.tool_schema_sha256) || !isNullableSha256(value.model_artifact_sha256)
      || !isSha256(value.tenant_bucket_sha256) || !isSemanticTimestamp(value.created_at, 64)) {
    throw new Error("GUI Semantic flow activity is malformed.");
  }
}

function assertSemanticFlowEdge(value: unknown): asserts value is SemanticFlowEdge {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticFlowEdgeKeys) || value.schema_version !== 1
      || !isSemanticPublicIdentifier(value.edge_id)
      || typeof value.relation !== "string" || !semanticFlowRelations.has(value.relation)
      || !isSemanticPublicIdentifier(value.source_node_id)
      || typeof value.source_node_type !== "string" || !semanticFlowNodeTypes.has(value.source_node_type)
      || !isSemanticPublicIdentifier(value.target_node_id)
      || typeof value.target_node_type !== "string" || !semanticFlowNodeTypes.has(value.target_node_type)
      || (value.source_node_id === value.target_node_id && value.source_node_type === value.target_node_type)
      || !isSemanticPublicIdentifier(value.pid) || !isSha256(value.provenance_sha256)
      || !isSemanticTimestamp(value.created_at, 64)) {
    throw new Error("GUI Semantic flow edge is malformed.");
  }
}

function assertSemanticMachineSettlement(value: unknown): asserts value is SemanticMachineSettlement {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticSettlementKeys) || value.schema_version !== 1
      || !isSemanticPublicText(value.settlement_id, 512)
      || !isNullableSemanticPublicText(value.assessment_id, 512)
      || !isNullableSemanticPublicText(value.job_id, 512)
      || !isSemanticPublicText(value.request_id, 512) || !isNonNegativeSafeInteger(value.request_revision)
      || !isSemanticPublicText(value.pid, 512) || !isNullableSemanticPublicText(value.operation_id, 512)
      || !isSemanticPublicText(value.effect_id, 512) || !isSemanticPublicText(value.epoch_id, 512)
      || !isSha256(value.policy_sha256) || !isSha256(value.tenant_bucket_sha256)
      || !isSemanticAction(value.action_id)
      || typeof value.outcome !== "string" || !semanticSettlementOutcomes.has(value.outcome)
      || !isNullableSemanticPublicText(value.capability_id, 512)
      || !isSha256(value.binding_sha256) || !isSha256(value.decision_sha256)
      || !isNullableSemanticPublicText(value.matched_rule_id, 128)
      || !isSemanticEnumList(value.reason_codes, 128, semanticReasonCodeValues)
      || !isSemanticSettlementInvariant(value)
      || !isSemanticSettlementHumanOutcome(value)
      || !isSemanticTimestamp(value.created_at, 64)) {
    throw new Error("GUI Semantic machine settlement is malformed.");
  }
}

function isSemanticSettlementHumanOutcome(value: Record<string, unknown>): boolean {
  const fields = [
    value.human_outcome,
    value.human_outcome_source,
    value.human_outcome_request_revision,
    value.human_outcome_decision_sha256,
    value.human_outcome_created_at
  ];
  if (fields.every((item) => item === null)) return true;
  if (fields.some((item) => item === null)) return false;
  if (typeof value.human_outcome !== "string"
      || !new Set(["approved", "rejected", "cancelled"]).has(value.human_outcome)
      || typeof value.human_outcome_source !== "string"
      || !new Set(["human", "machine_policy", "cancel"]).has(value.human_outcome_source)
      || !isNonNegativeSafeInteger(value.human_outcome_request_revision)
      || !isSha256(value.human_outcome_decision_sha256)
      || !isSemanticTimestamp(value.human_outcome_created_at, 64)) return false;
  return (value.human_outcome === "cancelled") === (value.human_outcome_source === "cancel");
}

function isSemanticSettlementInvariant(value: Record<string, unknown>): boolean {
  const issued = value.outcome === "issued";
  if (issued !== (value.capability_id !== null)) return false;
  if (issued && value.matched_rule_id === null) return false;
  if (value.outcome !== "denied") return true;
  return Array.isArray(value.reason_codes) && value.reason_codes.length > 0
    && value.reason_codes.every((reason) => typeof reason === "string" && semanticExecutableDenyReasons.has(reason));
}

function assertSemanticPolicyEpochSummary(value: unknown): asserts value is SemanticPolicyEpochSummary {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticPolicyEpochKeys) || value.schema_version !== 1
      || !isSemanticPublicText(value.epoch_id, 512)
      || !isNonNegativeSafeInteger(value.generation) || Number(value.generation) < 1
      || value.catalog_version !== 1 || !isSha256(value.policy_sha256)
      || !isNullableSha256(value.expected_previous_sha256)
      || !isSemanticTimestamp(value.created_at, 64)) {
    throw new Error("GUI Semantic policy epoch is malformed.");
  }
}

function assertSemanticHealthEvent(value: unknown): asserts value is SemanticHealthEvent {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticHealthEventKeys) || value.schema_version !== 1
      || !isSemanticPublicText(value.event_id, 512)
      || typeof value.event_kind !== "string" || !semanticHealthEventKinds.has(value.event_kind as SemanticHealthEventKind)
      || typeof value.severity !== "string" || !semanticHealthSeverities.has(value.severity)
      || !isNullableSemanticPublicText(value.epoch_id, 512)
      || !isNullableSha256(value.tenant_bucket_sha256) || !isSha256(value.evidence_sha256)
      || !isSemanticTimestamp(value.created_at, 64)) {
    throw new Error("GUI Semantic health event is malformed.");
  }
}

function isSemanticFlowLabels(value: unknown): value is SemanticFlowLabels {
  return isRecord(value) && hasOnlyKeys(value, semanticFlowLabelsKeys)
    && typeof value.sensitivity === "string" && semanticSensitivityValues.has(value.sensitivity)
    && typeof value.trust_level === "string" && semanticTrustValues.has(value.trust_level)
    && typeof value.integrity === "string" && semanticIntegrityValues.has(value.integrity);
}

function isSemanticReviewMetrics(value: unknown, machine: unknown): value is SemanticReviewMetrics {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticReviewMetricKeys)
      || !isNonNegativeSafeInteger(value.reviewed) || !isNonNegativeSafeInteger(value.safe)
      || !isNonNegativeSafeInteger(value.unsafe)
      || Number(value.safe) + Number(value.unsafe) > Number(value.reviewed)
      || !isConsistentRate(value, "unsafe", "reviewed", "unsafe_rate")
      || !isNonNegativeSafeInteger(value.issued_reviewed)
      || Number(value.issued_reviewed) > Number(value.reviewed)
      || !isRecord(machine) || !isNonNegativeSafeInteger(machine.issued)
      || Number(value.issued_reviewed) > Number(machine.issued)) return false;
  return isConsistentRate({
    numerator: value.issued_reviewed,
    denominator: machine.issued,
    rate: value.issued_review_rate
  }, "numerator", "denominator", "rate");
}

function assertSemanticReadOnlyPage(
  value: unknown,
  assertItem: (item: unknown) => void,
  label: string
): void {
  if (!isRecord(value) || !hasOnlyKeys(value, semanticPageV1Keys) || value.schema_version !== 1
      || !Array.isArray(value.items) || value.items.length > 100
      || !isNullableBoundedString(value.next_cursor, 2048)) {
    throw new Error(`GUI Semantic ${label} page is malformed.`);
  }
  for (const item of value.items) assertItem(item);
}

function isSemanticPublicText(value: unknown, maximum: number): value is string {
  return isBoundedString(value, maximum) && !/[\u0000-\u001f\u007f-\u009f]/.test(value);
}

function isApprovalProjectionText(value: unknown, maximum: number): value is string {
  return isBoundedString(value, maximum)
    && value.trim() === value
    && !/[\p{Cc}\p{Cf}\p{Cs}\p{Zl}\p{Zp}]/u.test(value);
}

function isNullableSemanticPublicText(value: unknown, maximum: number): value is string | null {
  return value === null || isSemanticPublicText(value, maximum);
}

function isSemanticPublicIdentifier(value: unknown): value is string {
  return typeof value === "string" && semanticPublicIdentifierPattern.test(value);
}

function isNullableSemanticPublicIdentifier(value: unknown): value is string | null {
  return value === null || isSemanticPublicIdentifier(value);
}

function isSemanticTimestamp(value: unknown, maximum: number): value is string {
  return isSemanticPublicText(value, maximum)
    && /(?:Z|[+-]\d{2}:\d{2})$/.test(value)
    && Number.isFinite(Date.parse(value));
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
      || !isSemanticTimestamp(value.created_at, 128) || !isSemanticTimestamp(value.completed_at, 128)
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
  for (const request of value.human_requests as unknown[]) assertHumanRequest(request);
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
const mcpUnregisterKeys = new Set(["server_id", "deleted"]);

export function assertMcpServerSummary(value: unknown): asserts value is McpServerSummary {
  if (!isRecord(value)
      || (value.schema_version !== 1 && value.schema_version !== 2 && value.schema_version !== 3)
      || typeof value.server_id !== "string"
      || !value.server_id
      || !isMcpProtocolMode(value.protocol_mode)
      || (value.schema_version === 1 && value.protocol_mode !== "legacy")
      || (value.schema_version === 3 && value.protocol_mode !== "2026-07-28")
      || !isRecord(value.transport)
      || typeof value.transport.type !== "string"
      || !value.transport.type
      || !Array.isArray(value.tools)
      || typeof value.timeout_s !== "number"
      || !Number.isFinite(value.timeout_s)
      || value.timeout_s <= 0
      || !isNonNegativeSafeInteger(value.max_request_bytes)
      || !isNonNegativeSafeInteger(value.max_response_bytes)
      || !isOptionalNullableString(value.auth_profile_id)
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
      || (value.schema_version !== 1 && value.schema_version !== 2 && value.schema_version !== 3)
      || typeof value.transport !== "string"
      || !value.transport
      || !isMcpProtocolMode(value.protocol_mode)
      || (value.schema_version === 1 && value.protocol_mode !== "legacy")
      || (value.schema_version === 3 && value.protocol_mode !== "2026-07-28")
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

export function assertMcpUnregisterResult(value: unknown): asserts value is McpUnregisterResult {
  if (!isRecord(value)
      || Object.keys(value).some((key) => !mcpUnregisterKeys.has(key))
      || typeof value.server_id !== "string"
      || !value.server_id
      || value.deleted !== true) {
    throw new Error("GUI MCP unregister result is malformed.");
  }
}

const forbiddenMcpPublicKeys = new Set([
  "access_token",
  "refresh_token",
  "id_token",
  "client_secret",
  "authorization_code",
  "code_verifier",
  "pkce_verifier",
  "remote_task_id"
]);
const forbiddenMcpPublicKeysCompact = new Set([
  "accesstoken",
  "refreshtoken",
  "idtoken",
  "clientsecret",
  "authorizationcode",
  "codeverifier",
  "pkceverifier",
  "remotetaskid"
]);

const mcpPageKeys = new Set(["items", "next_cursor", "cache_hint", "has_more"]);
const mcpCacheHintKeys = new Set(["ttl_ms", "scope"]);
const mcpResourceKeys = new Set([
  "resource_id", "name", "title", "description", "mime_type", "size", "metadata"
]);
const mcpResourceTemplateKeys = new Set([
  "template_id", "name", "title", "description", "mime_type", "metadata"
]);
const mcpPromptKeys = new Set([
  "prompt_id", "name", "title", "description", "arguments", "metadata"
]);
const mcpPromptArgumentKeys = new Set(["name", "title", "description", "required"]);
const mcpResourceContentsKeys = new Set(["resource_id", "contents", "provenance"]);
const mcpPromptResultKeys = new Set([
  "prompt_id", "messages", "description", "user_confirmation_required"
]);
const mcpPromptMessageKeys = new Set(["role", "content", "provenance"]);
const mcpCompletionResultKeys = new Set(["values", "total", "has_more"]);
const mcpCompleteResultKeys = new Set(["kind", "value", "preview_sha256"]);
const mcpInputRequiredKeys = new Set([
  "kind", "continuation_id", "revision", "respondable", "input_requests",
  "expires_at", "human_request_id", "human_revision", "human_preview_sha256"
]);
const mcpRemoteTaskKeys = new Set([
  "kind", "task_ref", "revision", "status", "status_message", "result",
  "input_requests", "human_request_id", "human_revision", "human_preview_sha256",
  "created_at", "updated_at", "ttl_ms", "poll_interval_ms"
]);
const mcpInputRequestKeys = new Set([
  "request_id", "kind", "mode", "prompt", "schema", "inert_url"
]);
const mcpTextContentKeys = new Set(["kind", "text", "metadata"]);
const mcpBlobContentKeys = new Set(["kind", "artifact", "metadata"]);
const mcpResourceLinkKeys = new Set([
  "kind", "resource_handle", "name", "title", "description", "mime_type", "metadata"
]);
const mcpArtifactReceiptKeys = new Set([
  "artifact_id", "byte_length", "sha256", "mime_type"
]);
const mcpOAuthStatusKeys = new Set([
  "profile_id", "status", "issuer", "resource", "scopes", "principal_sha256", "expires_at"
]);
const mcpOAuthProfileInputKeys = new Set([
  "profile_id",
  "server_id",
  "resource_uri",
  "expected_issuer",
  "redirect_uri",
  "client_id",
  "registration_mode",
  "token_endpoint_auth_method",
  "allowed_scopes",
  "default_scopes",
  "audience",
  "protected_resource_metadata_url",
  "authorization_server_metadata_url",
  "protected_resource_metadata_sha256",
  "authorization_server_metadata_sha256",
  "allowed_endpoint_origins",
  "allow_loopback_http",
  "protocol_revision",
  "transport"
]);
const mcpAuthorizationChallengeKeys = new Set([
  "challenge_id", "authorization_url", "expires_at"
]);
const mcpSubscriptionKeys = new Set([
  "subscription_id", "server_id", "status", "requested_filters", "acknowledged_filters",
  "opened_at", "closed_at", "lost_reason"
]);
const mcpSubscriptionEventKeys = new Set([
  "sequence", "event_type", "payload", "received_at", "provenance"
]);

export function assertMcpResourcePage(value: unknown): asserts value is McpPage<McpResource> {
  assertMcpPage(value, assertMcpResource);
}

export function assertMcpResourceTemplatePage(value: unknown): asserts value is McpPage<McpResourceTemplate> {
  assertMcpPage(value, assertMcpResourceTemplate);
}

export function assertMcpPromptPage(value: unknown): asserts value is McpPage<McpPrompt> {
  assertMcpPage(value, assertMcpPrompt);
}

export function assertMcpResourceOperationResult(
  value: unknown
): asserts value is McpOperationResult<McpResourceContents> {
  assertMcpOperationResult(value, assertMcpResourceContents);
}

export function assertMcpPromptOperationResult(
  value: unknown
): asserts value is McpOperationResult<McpPromptResult> {
  assertMcpOperationResult(value, assertMcpPromptResult);
  if (value.kind === "complete" && (
    typeof value.preview_sha256 !== "string"
    || !canonicalSha256.test(value.preview_sha256)
  )) {
    throw new Error("GUI MCP prompt preview binding is missing or malformed.");
  }
}

export function assertMcpCompletionOperationResult(
  value: unknown
): asserts value is McpOperationResult<McpCompletionResult> {
  assertMcpOperationResult(value, assertMcpCompletionResult);
}

export function assertMcpContinuationResult(
  value: unknown
): asserts value is McpOperationResult<unknown> {
  assertMcpOperationResult(value, assertUnknownMcpCompleteValue);
}

export function assertMcpInputRequired(value: unknown): asserts value is McpInputRequired {
  assertNoForbiddenMcpFields(value);
  if (!isRecord(value)
      || value.kind !== "input_required"
      || !hasOnlyMcpKeys(value, mcpInputRequiredKeys)
      || !isNonNegativeSafeInteger(value.revision)
      || typeof value.respondable !== "boolean"
      || !Array.isArray(value.input_requests)
      || !isOptionalNullableString(value.expires_at)) {
    throw new Error("GUI MCP input-required result is malformed.");
  }
  for (const request of value.input_requests) assertMcpInputRequest(request);
  const hasHumanReceipt = isNonEmptyString(value.human_request_id)
    && isNonNegativeSafeInteger(value.human_revision)
    && typeof value.human_preview_sha256 === "string"
    && canonicalSha256.test(value.human_preview_sha256);
  const hasNoHumanReceipt = (value.human_request_id === undefined || value.human_request_id === null)
    && (value.human_revision === undefined || value.human_revision === null)
    && (value.human_preview_sha256 === undefined || value.human_preview_sha256 === null);
  if (value.respondable) {
    if (!isNonEmptyString(value.continuation_id) || !hasHumanReceipt) {
      throw new Error("GUI MCP respondable input-required result is malformed.");
    }
    return;
  }
  if (value.continuation_id !== ""
      || !hasNoHumanReceipt
      || value.input_requests.length === 0
      || value.input_requests.some((request) => (
        request.kind !== "sampling_unsupported" && request.kind !== "roots_unsupported"
      ))) {
    throw new Error("GUI MCP unsupported input-required result is malformed.");
  }
}

export function assertMcpRemoteTask(value: unknown): asserts value is McpRemoteTask {
  assertNoForbiddenMcpFields(value);
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpRemoteTaskKeys)
      || value.kind !== "remote_task"
      || !isNonEmptyString(value.task_ref)
      || !["working", "input_required", "completed", "failed", "cancelled", "cancel_requested", "needs_attention"].includes(String(value.status))
      || !isNonNegativeSafeInteger(value.revision)
      || !Array.isArray(value.input_requests)
      || !isOptionalNullableString(value.status_message)
      || !isOptionalNullableString(value.created_at)
      || !isOptionalNullableString(value.updated_at)
      || !(value.ttl_ms === undefined || value.ttl_ms === null || isNonNegativeSafeInteger(value.ttl_ms))
      || !(value.poll_interval_ms === undefined || value.poll_interval_ms === null || isNonNegativeSafeInteger(value.poll_interval_ms))) {
    throw new Error("GUI MCP remote task projection is malformed.");
  }
  for (const request of value.input_requests) assertMcpInputRequest(request);
  const hasHumanReceipt = isNonEmptyString(value.human_request_id)
    && isNonNegativeSafeInteger(value.human_revision)
    && typeof value.human_preview_sha256 === "string"
    && canonicalSha256.test(value.human_preview_sha256);
  const hasNoHumanReceipt = (value.human_request_id === undefined || value.human_request_id === null)
    && (value.human_revision === undefined || value.human_revision === null)
    && (value.human_preview_sha256 === undefined || value.human_preview_sha256 === null);
  if ((value.status === "input_required" && (!hasHumanReceipt || value.input_requests.length === 0))
      || (value.status !== "input_required" && !hasNoHumanReceipt)) {
    throw new Error("GUI MCP remote task Human request receipt is malformed.");
  }
}

export function assertMcpOAuthStatus(value: unknown): asserts value is McpOAuthStatus {
  assertNoForbiddenMcpFields(value);
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpOAuthStatusKeys)
      || !isNonEmptyString(value.profile_id)
      || !["unconfigured", "authorization_required", "authorized", "expired", "revoked", "needs_attention"].includes(String(value.status))
      || !isOptionalNullableString(value.issuer)
      || !isOptionalNullableString(value.resource)
      || !isUniqueStringArrayAllowEmpty(value.scopes)
      || !(value.principal_sha256 === undefined || value.principal_sha256 === null || (typeof value.principal_sha256 === "string" && canonicalSha256.test(value.principal_sha256)))
      || !isOptionalNullableString(value.expires_at)) {
    throw new Error("GUI MCP OAuth status is malformed.");
  }
}

export function assertMcpOAuthStatuses(
  value: unknown
): asserts value is McpOAuthStatus[] {
  assertNoForbiddenMcpFields(value);
  if (!Array.isArray(value) || value.length > 1_000) {
    throw new Error("GUI MCP OAuth profile list is malformed.");
  }
  for (const item of value) assertMcpOAuthStatus(item);
}

export function assertMcpOAuthProfileInput(
  value: unknown
): asserts value is McpOAuthProfileInput {
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpOAuthProfileInputKeys)
      || !isNonEmptyString(value.profile_id)
      || !isNonEmptyString(value.server_id)
      || !isNonEmptyString(value.resource_uri)
      || !isNonEmptyString(value.expected_issuer)
      || !isNonEmptyString(value.redirect_uri)
      || !isNonEmptyString(value.client_id)
      || (value.registration_mode !== "preregistered" && value.registration_mode !== "cimd")
      || !(value.token_endpoint_auth_method === undefined
        || value.token_endpoint_auth_method === "none"
        || value.token_endpoint_auth_method === "client_secret_basic"
        || value.token_endpoint_auth_method === "client_secret_post")
      || !(value.allowed_scopes === undefined || isUniqueStringArrayAllowEmpty(value.allowed_scopes))
      || !(value.default_scopes === undefined || isUniqueStringArrayAllowEmpty(value.default_scopes))
      || !isOptionalNullableNonEmptyString(value.audience)
      || !isOptionalNullableNonEmptyString(value.protected_resource_metadata_url)
      || !isOptionalNullableNonEmptyString(value.authorization_server_metadata_url)
      || !isOptionalNullableSha256(value.protected_resource_metadata_sha256)
      || !isOptionalNullableSha256(value.authorization_server_metadata_sha256)
      || !(value.allowed_endpoint_origins === undefined || isUniqueStringArrayAllowEmpty(value.allowed_endpoint_origins))
      || !(value.allow_loopback_http === undefined || typeof value.allow_loopback_http === "boolean")
      || !(value.protocol_revision === undefined || value.protocol_revision === "2026-07-28")
      || !(value.transport === undefined || value.transport === "streamable_http")) {
    throw new Error("GUI MCP OAuth profile input is malformed.");
  }
}

export function assertMcpAuthorizationChallenge(
  value: unknown
): asserts value is McpAuthorizationChallenge {
  assertNoForbiddenMcpFields(value);
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpAuthorizationChallengeKeys)
      || !isNonEmptyString(value.challenge_id)
      || !isNonEmptyString(value.authorization_url)
      || !isNonEmptyString(value.expires_at)) {
    throw new Error("GUI MCP OAuth challenge is malformed.");
  }
}

export function assertMcpSubscription(value: unknown): asserts value is McpSubscription {
  assertNoForbiddenMcpFields(value);
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpSubscriptionKeys)
      || !isNonEmptyString(value.subscription_id)
      || !isNonEmptyString(value.server_id)
      || !["opening", "active", "lost", "closed"].includes(String(value.status))
      || !isUniqueStringArrayAllowEmpty(value.requested_filters)
      || !isUniqueStringArrayAllowEmpty(value.acknowledged_filters)
      || !isOptionalNullableString(value.opened_at)
      || !isOptionalNullableString(value.closed_at)
      || !isOptionalNullableString(value.lost_reason)) {
    throw new Error("GUI MCP subscription projection is malformed.");
  }
}

export function assertMcpSubscriptionEvents(
  value: unknown
): asserts value is McpSubscriptionEvent[] {
  assertNoForbiddenMcpFields(value);
  if (!Array.isArray(value)) throw new Error("GUI MCP subscription events are malformed.");
  for (const event of value) {
    if (!isRecord(event)
        || !hasOnlyMcpKeys(event, mcpSubscriptionEventKeys)
        || !isNonNegativeSafeInteger(event.sequence)
        || !isNonEmptyString(event.event_type)
        || !isNonEmptyString(event.received_at)
        || event.provenance !== "untrusted_mcp_notification") {
      throw new Error("GUI MCP subscription event is malformed.");
    }
  }
}

function assertMcpPage<T>(
  value: unknown,
  itemValidator: (item: unknown) => asserts item is T
): asserts value is McpPage<T> {
  assertNoForbiddenMcpFields(value);
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpPageKeys)
      || !Array.isArray(value.items)
      || !(value.next_cursor === null || typeof value.next_cursor === "string")
      || !(value.has_more === undefined || typeof value.has_more === "boolean")
      || !(value.cache_hint === null || isMcpCacheHint(value.cache_hint))) {
    throw new Error("GUI MCP page is malformed.");
  }
  if (value.has_more !== undefined && value.has_more !== (value.next_cursor !== null)) {
    throw new Error("GUI MCP page truncation signal is inconsistent.");
  }
  for (const item of value.items) itemValidator(item);
}

function isMcpCacheHint(value: unknown): value is McpCacheHint {
  return isRecord(value)
    && hasOnlyMcpKeys(value, mcpCacheHintKeys)
    && isNonNegativeSafeInteger(value.ttl_ms)
    && (value.scope === "private" || value.scope === "public");
}

function assertMcpResource(value: unknown): asserts value is McpResource {
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpResourceKeys)
      || !isSafeMcpSelector(value.resource_id)
      || !isNonEmptyString(value.name)
      || !isOptionalNullableString(value.title)
      || !isOptionalNullableString(value.description)
      || !isSafeMcpMime(value.mime_type)
      || !(value.size === undefined || value.size === null || isNonNegativeSafeInteger(value.size))
      || !isOptionalMcpMetadata(value.metadata)) {
    throw new Error("GUI MCP resource is malformed or is an unsupported MCP App.");
  }
}

function assertMcpResourceTemplate(value: unknown): asserts value is McpResourceTemplate {
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpResourceTemplateKeys)
      || !isSafeMcpSelector(value.template_id)
      || !isNonEmptyString(value.name)
      || !isOptionalNullableString(value.title)
      || !isOptionalNullableString(value.description)
      || !isSafeMcpMime(value.mime_type)
      || !isOptionalMcpMetadata(value.metadata)) {
    throw new Error("GUI MCP resource template is malformed or is an unsupported MCP App.");
  }
}

function assertMcpPrompt(value: unknown): asserts value is McpPrompt {
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpPromptKeys)
      || !isNonEmptyString(value.prompt_id)
      || !isNonEmptyString(value.name)
      || !isOptionalNullableString(value.title)
      || !isOptionalNullableString(value.description)
      || !Array.isArray(value.arguments)
      || !isOptionalMcpMetadata(value.metadata)) {
    throw new Error("GUI MCP prompt is malformed.");
  }
  for (const argument of value.arguments) {
    if (!isRecord(argument)
        || !hasOnlyMcpKeys(argument, mcpPromptArgumentKeys)
        || !isNonEmptyString(argument.name)
        || !isOptionalNullableString(argument.title)
        || !isOptionalNullableString(argument.description)
        || typeof argument.required !== "boolean") {
      throw new Error("GUI MCP prompt argument is malformed.");
    }
  }
}

function assertMcpResourceContents(value: unknown): asserts value is McpResourceContents {
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpResourceContentsKeys)
      || !isNonEmptyString(value.resource_id)
      || value.provenance !== "untrusted_mcp_resource"
      || !Array.isArray(value.contents)) {
    throw new Error("GUI MCP resource contents are malformed.");
  }
  for (const content of value.contents) assertMcpContentBlock(content);
}

function assertMcpPromptResult(value: unknown): asserts value is McpPromptResult {
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpPromptResultKeys)
      || !isNonEmptyString(value.prompt_id)
      || value.user_confirmation_required !== true
      || !isOptionalNullableString(value.description)
      || !Array.isArray(value.messages)) {
    throw new Error("GUI MCP prompt result is malformed.");
  }
  for (const message of value.messages) {
    if (!isRecord(message)
        || !hasOnlyMcpKeys(message, mcpPromptMessageKeys)
        || (message.role !== "user" && message.role !== "assistant")
        || message.provenance !== "untrusted_mcp_prompt") {
      throw new Error("GUI MCP prompt message is malformed.");
    }
    assertMcpContentBlock(message.content);
  }
}

function assertMcpCompletionResult(value: unknown): asserts value is McpCompletionResult {
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpCompletionResultKeys)
      || !Array.isArray(value.values)
      || value.values.some((item) => typeof item !== "string")
      || !(value.total === undefined || value.total === null || isNonNegativeSafeInteger(value.total))
      || typeof value.has_more !== "boolean") {
    throw new Error("GUI MCP completion result is malformed.");
  }
}

function assertMcpOperationResult<T>(
  value: unknown,
  completeValidator: (item: unknown) => asserts item is T
): asserts value is McpOperationResult<T> {
  assertNoForbiddenMcpFields(value);
  if (!isRecord(value) || typeof value.kind !== "string") {
    throw new Error("GUI MCP operation result is malformed.");
  }
  if (value.kind === "complete") {
    if (!hasOnlyMcpKeys(value, mcpCompleteResultKeys) || !("value" in value)) {
      throw new Error("GUI MCP complete result is malformed or is missing value.");
    }
    if (!(value.preview_sha256 === undefined
      || (typeof value.preview_sha256 === "string" && canonicalSha256.test(value.preview_sha256)))) {
      throw new Error("GUI MCP preview binding is malformed.");
    }
    if (value.value !== null) completeValidator(value.value);
    return;
  }
  if (value.kind === "remote_task") {
    assertMcpRemoteTask(value);
    return;
  }
  assertMcpInputRequired(value);
}

function assertMcpInputRequest(value: unknown): asserts value is McpInputRequest {
  if (!isRecord(value)
      || !hasOnlyMcpKeys(value, mcpInputRequestKeys)
      || !isNonEmptyString(value.request_id)
      || !["elicitation", "sampling_unsupported", "roots_unsupported"].includes(String(value.kind))
      || !(value.mode === undefined || value.mode === null || value.mode === "form" || value.mode === "url")
      || !isOptionalNullableString(value.prompt)
      || !isRecord(value.schema)
      || !isOptionalNullableString(value.inert_url)) {
    throw new Error("GUI MCP input request is malformed.");
  }
}

function assertUnknownMcpCompleteValue(_value: unknown): asserts _value is unknown {
  // The continuation facade may complete any original protected operation.
}

function assertMcpContentBlock(value: unknown): asserts value is McpContentBlock {
  if (!isRecord(value) || typeof value.kind !== "string") {
    throw new Error("GUI MCP content block is malformed.");
  }
  if (value.kind === "text"
      && hasOnlyMcpKeys(value, mcpTextContentKeys)
      && isOptionalMcpMetadata(value.metadata)
      && typeof value.text === "string") return;
  if (value.kind === "blob") {
    if (!hasOnlyMcpKeys(value, mcpBlobContentKeys)
        || !isOptionalMcpMetadata(value.metadata)) {
      throw new Error("GUI MCP content block is malformed or is an unsupported MCP App.");
    }
    if (value.artifact === null) return;
    const artifact = value.artifact;
    if (isRecord(artifact)
        && hasOnlyMcpKeys(artifact, mcpArtifactReceiptKeys)
        && isNonEmptyString(artifact.artifact_id)
        && isNonNegativeSafeInteger(artifact.byte_length)
        && typeof artifact.sha256 === "string"
        && canonicalSha256.test(artifact.sha256)
        && isSafeMcpMime(artifact.mime_type)) return;
  }
  if (value.kind === "resource_link"
      && hasOnlyMcpKeys(value, mcpResourceLinkKeys)
      && isNonEmptyString(value.resource_handle)
      && !value.resource_handle.toLowerCase().startsWith("ui:")
      && isNonEmptyString(value.name)
      && isSafeMcpMime(value.mime_type)
      && isOptionalMcpMetadata(value.metadata)) return;
  throw new Error("GUI MCP content block is malformed or is an unsupported MCP App.");
}

function assertNoForbiddenMcpFields(value: unknown): void {
  const pending: unknown[] = [value];
  const seen = new Set<object>();
  while (pending.length) {
    const current = pending.pop();
    if (!current || typeof current !== "object") continue;
    if (seen.has(current)) throw new Error("GUI MCP result contains a cycle.");
    seen.add(current);
    if (Array.isArray(current)) {
      pending.push(...current);
      continue;
    }
    for (const [key, item] of Object.entries(current as Record<string, unknown>)) {
      const folded = key.toLowerCase();
      const compact = folded.replace(/[^a-z0-9]/g, "");
      if (folded.startsWith("ui/")
          || forbiddenMcpPublicKeys.has(folded)
          || forbiddenMcpPublicKeysCompact.has(compact)) {
        throw new Error(
          "GUI MCP result contains a private credential or remote task identifier, "
          + "or unsupported MCP Apps metadata."
        );
      }
      pending.push(item);
    }
  }
}

function hasOnlyMcpKeys(
  value: Record<string, unknown>,
  allowed: ReadonlySet<string>
): boolean {
  return Object.keys(value).every((key) => allowed.has(key));
}

function isOptionalMcpMetadata(value: unknown): value is Record<string, unknown> | undefined {
  return value === undefined || isRecord(value);
}

function isSafeMcpSelector(value: unknown): value is string {
  return isNonEmptyString(value) && !value.toLowerCase().startsWith("ui:");
}

function isSafeMcpMime(value: unknown): value is string | null | undefined {
  if (!isOptionalNullableString(value)) return false;
  if (!value) return true;
  const parsed = parseMcpMime(value);
  if (!parsed) return false;
  return !(parsed.mediaType === "text/html" && parsed.parameters.some(
    ([name, parameter]) => name === "profile" && parameter.toLowerCase() === "mcp-app"
  ));
}

const mcpMimeToken = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;

function parseMcpMime(
  value: string
): { mediaType: string; parameters: Array<[string, string]> } | null {
  const sections: string[] = [];
  let selected = "";
  let quoted = false;
  let escaped = false;
  for (const character of value) {
    if (escaped) {
      selected += character;
      escaped = false;
      continue;
    }
    if (quoted && character === "\\") {
      selected += character;
      escaped = true;
      continue;
    }
    if (character === '"') {
      selected += character;
      quoted = !quoted;
      continue;
    }
    if (character === ";" && !quoted) {
      sections.push(selected);
      selected = "";
      continue;
    }
    selected += character;
  }
  if (quoted || escaped) return null;
  sections.push(selected);

  const mediaParts = sections.shift()?.trim().split("/") ?? [];
  if (mediaParts.length !== 2
      || !mcpMimeToken.test(mediaParts[0] ?? "")
      || !mcpMimeToken.test(mediaParts[1] ?? "")) return null;
  const parameters: Array<[string, string]> = [];
  for (const section of sections) {
    const separator = section.indexOf("=");
    if (separator <= 0) return null;
    const name = section.slice(0, separator).trim();
    const rawParameter = section.slice(separator + 1).trim();
    if (!mcpMimeToken.test(name)) return null;
    const parameter = parseMcpMimeParameter(rawParameter);
    if (parameter === null) return null;
    parameters.push([name.toLowerCase(), parameter]);
  }
  return {
    mediaType: `${mediaParts[0]}/${mediaParts[1]}`.toLowerCase(),
    parameters
  };
}

function parseMcpMimeParameter(value: string): string | null {
  if (!value.startsWith('"')) return mcpMimeToken.test(value) ? value : null;
  if (value.length < 2 || !value.endsWith('"')) return null;
  let decoded = "";
  let escaped = false;
  for (const character of value.slice(1, -1)) {
    const code = character.charCodeAt(0);
    if ((code < 0x20 && code !== 0x09) || code === 0x7f) return null;
    if (escaped) {
      decoded += character;
      escaped = false;
    } else if (character === "\\") {
      escaped = true;
    } else if (character === '"') {
      return null;
    } else {
      decoded += character;
    }
  }
  return escaped ? null : decoded;
}

function isUniqueStringArrayAllowEmpty(value: unknown): value is string[] {
  return Array.isArray(value)
    && value.every((item) => typeof item === "string" && Boolean(item))
    && new Set(value).size === value.length;
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

function isOptionalNullableNonEmptyString(
  value: unknown
): value is string | null | undefined {
  return value === undefined || value === null || isNonEmptyString(value);
}

function isOptionalNullableSha256(
  value: unknown
): value is string | null | undefined {
  return value === undefined
    || value === null
    || (typeof value === "string" && canonicalSha256.test(value));
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
