import { useId, useReducer, useRef } from "react";
import type {
  CanonicalApprovalPreviewV1,
  DataReleaseApprovalContext,
  HumanPermissionPolicy,
  HumanRequest,
  HumanRequestPayload,
  HumanResponseInput
} from "../api/types";
import { canonicalApprovalPreviewFromRequest } from "../api/types";
import { useI18n, type TranslationKey } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";

export type HumanResponseOutcome = "accepted" | "settled" | "retryable" | "ambiguous";

export type HumanDecisionDraft = {
  answer: string;
  policy: HumanPermissionPolicy;
};

export type HumanResponseValidationError =
  | "question_answer_required"
  | "permission_approve_deny"
  | "permission_reject_allow"
  | "release_required";

type HumanResponseBuildResult =
  | { response: HumanResponseInput }
  | { error: HumanResponseValidationError };

type HumanRequestCardProps = {
  request: HumanRequest;
  className?: string;
  onRespond(request: HumanRequest, response: HumanResponseInput): Promise<HumanResponseOutcome | boolean>;
};

export type HumanDecisionState = HumanDecisionDraft & {
  submitting: boolean;
  settled?: boolean;
  ambiguousDecision?: boolean | null;
  errorKey: TranslationKey | null;
};

export type HumanDecisionAction =
  | { type: "answer_changed"; answer: string }
  | { type: "policy_changed"; policy: HumanPermissionPolicy }
  | { type: "validation_failed"; errorKey: TranslationKey }
  | { type: "submission_started" }
  | { type: "submission_finished"; outcome: HumanResponseOutcome; approved: boolean }
  | { type: "submission_finished"; accepted: boolean };

export function humanDecisionReducer(state: HumanDecisionState, action: HumanDecisionAction): HumanDecisionState {
  if (action.type === "answer_changed") return { ...state, answer: action.answer, errorKey: null };
  if (action.type === "policy_changed") return { ...state, policy: action.policy, errorKey: null };
  if (action.type === "validation_failed") return { ...state, errorKey: action.errorKey };
  if (action.type === "submission_started") return { ...state, submitting: true, errorKey: null };
  const outcome = "outcome" in action ? action.outcome : action.accepted ? "accepted" : "retryable";
  const approved = "approved" in action ? action.approved : false;
  if (outcome === "accepted" || outcome === "settled") {
    return { ...state, submitting: false, settled: true, ambiguousDecision: null, errorKey: null };
  }
  if (outcome === "ambiguous") {
    return {
      ...state,
      submitting: false,
      ambiguousDecision: approved,
      errorKey: "human.submitAmbiguous"
    };
  }
  return {
    ...state,
    submitting: false,
    ...("ambiguousDecision" in state ? { ambiguousDecision: null } : {}),
    errorKey: "human.submitFailed"
  };
}

export function buildHumanResponse(
  request: HumanRequest,
  approved: boolean,
  draft: HumanDecisionDraft
): HumanResponseBuildResult {
  if (request.payload.release_required === true) {
    return { error: "release_required" };
  }
  const requestType = request.payload?.type;
  if (requestType === "external_operation_approval") {
    const preview = parseCanonicalApprovalPreview(request);
    if (preview === null || request.preview_sha256 === undefined) {
      // Preserve the 1.4.0 off/shadow Human path for malformed historical
      // requests. Active enforcement modes reject this unfenced response at
      // the Host boundary; a valid preview always uses the fenced variant.
      return { response: { kind: "approval", approved } };
    }
    return {
      response: {
        kind: "external_approval",
        approved,
        expected_revision: preview.revision,
        preview_sha256: request.preview_sha256
      }
    };
  }
  if (requestType === "permission_request") {
    if (approved) {
      if (draft.policy === "always_deny") return { error: "permission_approve_deny" };
      return { response: { kind: "permission", approved: true, decision: { policy: draft.policy } } };
    }
    if (draft.policy === "always_allow") return { error: "permission_reject_allow" };
    return { response: { kind: "permission", approved: false, decision: { policy: draft.policy } } };
  }
  if (requestType === "question") {
    if (!approved) return { response: { kind: "question", approved: false } };
    const answer = draft.answer.trim();
    if (!answer) return { error: "question_answer_required" };
    return { response: { kind: "question", approved: true, answer } };
  }
  return { response: { kind: "approval", approved } };
}

export function HumanRequestCard({ request, className = "humanCard", onRespond }: HumanRequestCardProps) {
  const { t } = useI18n();
  const promptId = useId();
  const requestType = request.payload?.type;
  const isPermission = requestType === "permission_request";
  const isQuestion = requestType === "question";
  const isDataReleaseApproval = requestType === "data_release_approval";
  const isExternalOperationApproval = requestType === "external_operation_approval";
  const approvalPreview = isExternalOperationApproval ? parseCanonicalApprovalPreview(request) : null;
  const permissionContext = isPermission && (request.payload.requested_permission || request.payload.context)
    ? {
        requested_permission: request.payload.requested_permission ?? null,
        context: request.payload.context ?? null
      }
    : null;
  const releaseRequired = request.payload.release_required === true;
  const releaseContext = isDataReleaseApproval
    ? parseDataReleaseApprovalContext(request.payload)
    : null;
  const [state, dispatch] = useReducer(humanDecisionReducer, {
    answer: "",
    policy: "ask_each_time",
    submitting: false,
    settled: false,
    ambiguousDecision: null,
    errorKey: null
  });
  const submissionInFlight = useRef(false);

  async function submit(approved: boolean) {
    if (submissionInFlight.current) return;
    const built = buildHumanResponse(request, approved, state);
    if ("error" in built) {
      dispatch({ type: "validation_failed", errorKey: validationErrorKey(built.error) });
      return;
    }
    submissionInFlight.current = true;
    dispatch({ type: "submission_started" });
    try {
      const responseOutcome = await onRespond(request, built.response).catch(() => "ambiguous" as const);
      const outcome = typeof responseOutcome === "boolean"
        ? responseOutcome ? "accepted" : "retryable"
        : responseOutcome;
      dispatch({ type: "submission_finished", outcome, approved });
      // The authoritative snapshot removes a completed request. Keep the draft
      // intact until then so a failed request or refresh never destroys input.
    } finally {
      submissionInFlight.current = false;
    }
  }

  const approveDisabled = state.submitting || state.settled || state.ambiguousDecision === false
    || (isQuestion && !state.answer.trim())
    || (isPermission && state.policy === "always_deny")
    || (isDataReleaseApproval && releaseContext === null);
  const rejectDisabled = state.submitting || state.settled || state.ambiguousDecision === true
    || (isPermission && state.policy === "always_allow");
  const prompt = isExternalOperationApproval
    ? t("human.externalApprovalTitle")
    : String(request.payload?.question ?? request.payload?.reason ?? request.payload?.type ?? t("operator.humanRequestFallback"));
  const releaseRequestId = request.release_request_id
    ?? (typeof request.payload.release_request_id === "string" ? request.payload.release_request_id : null);
  const approveActionLabel = state.ambiguousDecision === true
    ? t("human.reconcile")
    : isQuestion
      ? t("human.submitAnswer")
      : t("human.approve");
  const rejectActionLabel = state.ambiguousDecision === false ? t("human.reconcile") : t("human.reject");
  const requestContextLabel = isDataReleaseApproval && releaseContext
    ? `${releaseContext.operation} · ${releaseContext.sink} · ${request.request_id}`
    : isExternalOperationApproval && approvalPreview
      ? `${approvalPreview.action_id} · ${approvalPreview.resource_display} · ${request.request_id}`
    : isExternalOperationApproval
      ? `${t("human.externalApprovalTitle")} · ${request.request_id}`
    : `${prompt} · ${request.request_id}`;

  if (releaseRequired) {
    return (
      <section className={`${className} typedHumanRequest withheldHumanRequest`} role="group" aria-labelledby={promptId}>
        <strong id={promptId} className="humanRequestPrompt">{t("human.releaseRequiredTitle")}</strong>
        <p className="humanReleaseNotice" role="status">
          {releaseRequestId
            ? t("human.releaseRequiredMessage", { requestId: releaseRequestId })
            : t("human.releaseRequiredMessageNoId")}
        </p>
      </section>
    );
  }

  return (
    <section
      className={`${className} typedHumanRequest${isDataReleaseApproval ? " dataReleaseApprovalCard" : ""}`}
      aria-busy={state.submitting || undefined}
      role="group"
      aria-labelledby={promptId}
    >
      <strong id={promptId} className="humanRequestPrompt">
        {isDataReleaseApproval
          ? t("human.releaseApprovalTitle")
          : isExternalOperationApproval
            ? t("human.externalApprovalTitle")
            : prompt}
      </strong>
      {isDataReleaseApproval ? (
        <p className="humanReleaseNotice">{t("human.releaseApprovalHint")}</p>
      ) : null}
      {permissionContext ? (
        <section className="humanApprovalContext permissionApprovalContext" aria-label={t("human.approvalContext")}>
          <strong>{t("human.approvalContext")}</strong>
          <p>{t("human.approvalContextHint")}</p>
          <CollapsibleJson value={permissionContext} label={t("human.approvalContext")} defaultExpanded />
        </section>
      ) : null}
      {isPermission ? (
        <label className="humanDecisionControl">
          <span>{t("human.permissionPolicy")}</span>
          <select
            name="permission-policy"
            aria-describedby={promptId}
            value={state.policy}
            disabled={state.submitting || state.settled || state.ambiguousDecision !== null}
            onChange={(event) => {
              dispatch({ type: "policy_changed", policy: event.currentTarget.value as HumanPermissionPolicy });
            }}
          >
            <option value="always_allow">{t("human.policyAlwaysAllow")}</option>
            <option value="ask_each_time">{t("human.policyAskEachTime")}</option>
            <option value="always_deny">{t("human.policyAlwaysDeny")}</option>
          </select>
        </label>
      ) : null}
      {isQuestion ? (
        <label className="humanDecisionControl">
          <span>{t("human.answer")}</span>
          <input
            name="human-answer"
            aria-describedby={promptId}
            required
            placeholder={t("human.answerPlaceholder")}
            value={state.answer}
            disabled={state.submitting || state.settled || state.ambiguousDecision !== null}
            onChange={(event) => {
              dispatch({ type: "answer_changed", answer: event.currentTarget.value });
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.nativeEvent.isComposing && state.answer.trim()) void submit(true);
            }}
          />
        </label>
      ) : null}
      {isDataReleaseApproval && releaseContext ? (
        <DataReleaseMetadata context={releaseContext} />
      ) : null}
      {isDataReleaseApproval && !releaseContext ? (
        <span className="humanDecisionError" role="alert">
          {t("human.releaseMetadataInvalid")}
        </span>
      ) : null}
      {isExternalOperationApproval && approvalPreview ? (
        <ExternalApprovalPreview preview={approvalPreview} previewSha256={request.preview_sha256 as string} />
      ) : null}
      {isExternalOperationApproval && !approvalPreview ? (
        <span className="humanDecisionError" role="alert">{t("human.externalPreviewInvalid")}</span>
      ) : null}
      <div className="humanDecisionActions">
        <button aria-label={`${approveActionLabel}: ${requestContextLabel}`} aria-describedby={promptId} disabled={approveDisabled} onClick={() => void submit(true)}>
          {approveActionLabel}
        </button>
        <button aria-label={`${rejectActionLabel}: ${requestContextLabel}`} aria-describedby={promptId} disabled={rejectDisabled} className="danger" onClick={() => void submit(false)}>
          {rejectActionLabel}
        </button>
      </div>
      {state.errorKey ? <span className="humanDecisionError" role="alert">{t(state.errorKey)}</span> : null}
    </section>
  );
}

function validationErrorKey(error: HumanResponseValidationError): TranslationKey {
  if (error === "question_answer_required") return "human.answerRequired";
  if (error === "permission_approve_deny") return "human.approveDenyInvalid";
  if (error === "release_required") return "human.releaseRequiredMessageNoId";
  return "human.rejectAllowInvalid";
}

export function parseCanonicalApprovalPreview(request: HumanRequest): CanonicalApprovalPreviewV1 | null {
  return canonicalApprovalPreviewFromRequest(request);
}

function ExternalApprovalPreview({
  preview,
  previewSha256
}: {
  preview: CanonicalApprovalPreviewV1;
  previewSha256: string;
}) {
  const { formatTime, t } = useI18n();
  const argument = preview.argument_projection;
  const argumentRows: Array<{ key: TranslationKey; value: string; code?: boolean }> = [
    { key: "human.externalArgumentOperation", value: argument.operation, code: true }
  ];
  if (argument.kind === "filesystem") {
    argumentRows.push({ key: "human.externalPathDigest", value: argument.path_sha256 as string, code: true });
    if (argument.content_sha256 !== null) {
      argumentRows.push({ key: "human.externalContentDigest", value: argument.content_sha256, code: true });
      argumentRows.push({ key: "human.externalContentBytes", value: String(argument.content_bytes) });
    }
    if (argument.read_max_bytes !== null) {
      argumentRows.push({ key: "human.externalReadMaxBytes", value: String(argument.read_max_bytes) });
    }
    if (argument.entry_limit !== null) argumentRows.push({ key: "human.externalEntryLimit", value: String(argument.entry_limit) });
    if (argument.text_encoding !== null) argumentRows.push({ key: "human.externalEncoding", value: argument.text_encoding, code: true });
    if (argument.expected_content_sha256 !== null) {
      argumentRows.push({ key: "human.externalExpectedContent", value: argument.expected_content_sha256, code: true });
    }
    for (const [key, value] of [
      ["human.externalOverwrite", argument.overwrite],
      ["human.externalParents", argument.parents],
      ["human.externalExistOk", argument.exist_ok],
      ["human.externalRecursive", argument.recursive],
      ["human.externalMissingOk", argument.missing_ok]
    ] as const) {
      if (value !== null) argumentRows.push({ key, value: String(value) });
    }
  } else if (argument.kind === "shell") {
    const argv = `[${argument.display_argv.map((item) => JSON.stringify(item)).join(", ")}]${argument.argv_truncated ? " …" : ""}`;
    argumentRows.push({ key: "human.externalArgv", value: argv, code: true });
    argumentRows.push({ key: "human.externalArgvCount", value: String(argument.argv_count) });
    argumentRows.push({ key: "human.externalArgvDigest", value: argument.argv_sha256 as string, code: true });
    argumentRows.push({ key: "human.externalCwd", value: argument.safe_cwd ?? t("human.externalRedacted"), code: argument.safe_cwd !== null });
    argumentRows.push({ key: "human.externalCwdDigest", value: argument.cwd_sha256 as string, code: true });
    if (argument.timeout_seconds !== null) argumentRows.push({ key: "human.externalTimeout", value: argument.timeout_seconds });
    if (argument.continuous_session !== null) {
      argumentRows.push({ key: "human.externalContinuousSession", value: String(argument.continuous_session) });
    }
    if (argument.network_access !== null) argumentRows.push({ key: "human.externalNetworkAccess", value: String(argument.network_access) });
  } else if (argument.kind === "jsonrpc") {
    argumentRows.push({ key: "human.externalEndpoint", value: argument.endpoint_id as string, code: true });
    argumentRows.push({ key: "human.externalEndpointDigest", value: argument.endpoint_id_sha256 as string, code: true });
    argumentRows.push({ key: "human.externalMethod", value: argument.method_id as string, code: true });
    argumentRows.push({ key: "human.externalMethodDigest", value: argument.method_id_sha256 as string, code: true });
    argumentRows.push({ key: "human.externalPayloadDigest", value: argument.payload_sha256 as string, code: true });
    if (argument.registry_spec_sha256 !== null) {
      argumentRows.push({ key: "human.externalRegistrySpecDigest", value: argument.registry_spec_sha256, code: true });
      argumentRows.push({ key: "human.externalRegistryGeneration", value: String(argument.registry_generation) });
    }
  } else if (argument.kind === "mcp") {
    argumentRows.push({ key: "human.externalServer", value: argument.server_id as string, code: true });
    argumentRows.push({ key: "human.externalServerDigest", value: argument.server_id_sha256 as string, code: true });
    argumentRows.push({ key: "human.externalTool", value: argument.tool_id as string, code: true });
    argumentRows.push({ key: "human.externalToolDigest", value: argument.tool_id_sha256 as string, code: true });
    argumentRows.push({ key: "human.externalPayloadDigest", value: argument.payload_sha256 as string, code: true });
    if (argument.registry_spec_sha256 !== null) {
      argumentRows.push({ key: "human.externalRegistrySpecDigest", value: argument.registry_spec_sha256, code: true });
      argumentRows.push({ key: "human.externalRegistryGeneration", value: String(argument.registry_generation) });
    }
  } else if (argument.kind === "git") {
    if (argument.worktree_id !== null) argumentRows.push({ key: "human.externalWorktree", value: argument.worktree_id, code: true });
    if (argument.worktree_id_sha256 !== null) {
      argumentRows.push({ key: "human.externalWorktreeDigest", value: argument.worktree_id_sha256, code: true });
    }
    if (argument.path_sha256 !== null) argumentRows.push({ key: "human.externalPathDigest", value: argument.path_sha256, code: true });
    if (argument.repository_state_sha256 !== null) {
      argumentRows.push({ key: "human.externalRepositoryStateDigest", value: argument.repository_state_sha256, code: true });
    }
    if (argument.source_args_sha256 !== null) {
      argumentRows.push({ key: "human.externalSourceArgsDigest", value: argument.source_args_sha256, code: true });
    }
    for (const reference of argument.git_references) {
      argumentRows.push({
        key: "human.externalGitReference",
        value: `${reference.role}: ${reference.display} (sha256=${reference.sha256})`,
        code: true
      });
    }
    if (argument.git_fact_tokens.length > 0) {
      argumentRows.push({ key: "human.externalGitFacts", value: argument.git_fact_tokens.join(", "), code: true });
    }
  }
  const rows: Array<{ key: TranslationKey; value: string; code?: boolean }> = [
    { key: "human.externalAction", value: preview.action_id, code: true },
    { key: "human.externalResource", value: preview.resource_display, code: true },
    { key: "human.externalResourceDigest", value: preview.resource_sha256, code: true },
    { key: "human.externalRight", value: preview.rights.join(", "), code: true },
    { key: "human.externalRisk", value: preview.risk },
    { key: "human.externalEffect", value: preview.effect_id, code: true },
    ...argumentRows,
    { key: "human.externalArgsDigest", value: preview.canonical_args_sha256, code: true },
    ...(preview.target_state_sha256
      ? [{ key: "human.externalStateDigest" as const, value: preview.target_state_sha256, code: true }]
      : []),
    { key: "human.externalSensitivity", value: preview.source_labels.sensitivity },
    { key: "human.externalIntegrity", value: preview.source_labels.integrity },
    { key: "human.externalTrust", value: preview.source_labels.trust_level },
    { key: "human.externalIdentity", value: preview.source_labels.identity_mixed
      ? t("human.externalIdentityMixed")
      : preview.source_labels.identity_present
        ? t("human.externalIdentityPresent")
        : t("human.externalIdentityAbsent") },
    { key: "human.externalExpires", value: preview.expires_at === null ? t("semantic.none") : formatTime(preview.expires_at) },
    { key: "human.externalPreviewDigest", value: previewSha256, code: true }
  ];
  return (
    <section className="humanApprovalContext canonicalApprovalPreview" aria-label={t("human.externalPreviewLabel")}>
      <strong>{t("human.externalPreviewLabel")}</strong>
      <p>{t("human.externalPreviewHint")}</p>
      <dl className="humanReleaseMetadata">
        {rows.map((row) => (
          <div className="humanReleaseMetadataRow" key={row.key}>
            <dt>{t(row.key)}</dt>
            <dd>{row.code ? <code><bdi>{row.value}</bdi></code> : <bdi>{row.value}</bdi>}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export function parseDataReleaseApprovalContext(
  payload: HumanRequestPayload
): DataReleaseApprovalContext | null {
  if (payload.type !== "data_release_approval" || !isRecord(payload.context)) return null;
  const context = payload.context;
  if (
    !isNonEmptyString(context.sink)
    || !isNonEmptyString(context.sensitivity)
    || !isNonNegativeInteger(context.payload_bytes)
    || !isSha256(context.payload_sha256)
    || !isNonNegativeInteger(context.source_count)
    || !isNonEmptyString(context.operation)
    || !isNullableString(context.tenant)
    || !isNullableString(context.principal)
  ) {
    return null;
  }
  return {
    sink: context.sink,
    sensitivity: context.sensitivity,
    tenant: normalizeOptionalString(context.tenant),
    principal: normalizeOptionalString(context.principal),
    payload_bytes: context.payload_bytes,
    payload_sha256: context.payload_sha256,
    source_count: context.source_count,
    operation: context.operation
  };
}

function DataReleaseMetadata({ context }: { context: DataReleaseApprovalContext }) {
  const { t } = useI18n();
  const rows: Array<{ key: TranslationKey; value: string; code?: boolean }> = [
    { key: "human.releaseSink", value: context.sink, code: true },
    { key: "human.releaseSensitivity", value: context.sensitivity },
    ...(context.tenant ? [{ key: "human.releaseTenant" as const, value: context.tenant, code: true }] : []),
    ...(context.principal ? [{ key: "human.releasePrincipal" as const, value: context.principal, code: true }] : []),
    { key: "human.releasePayloadBytes", value: String(context.payload_bytes) },
    { key: "human.releasePayloadSha256", value: context.payload_sha256, code: true },
    { key: "human.releaseSourceCount", value: String(context.source_count) },
    { key: "human.releaseOperation", value: context.operation, code: true }
  ];

  return (
    <dl className="humanReleaseMetadata" aria-label={t("human.releaseMetadataLabel")}>
      {rows.map((row) => (
        <div className="humanReleaseMetadataRow" key={row.key}>
          <dt>{t(row.key)}</dt>
          <dd>{row.code ? <code>{row.value}</code> : row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function isNullableString(value: unknown): value is string | null | undefined {
  return value === undefined || value === null || typeof value === "string";
}

function normalizeOptionalString(value: string | null | undefined): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}
