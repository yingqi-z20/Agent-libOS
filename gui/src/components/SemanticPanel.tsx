import { LoaderCircle, RefreshCw, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { LibOSClient, SemanticAssessmentFilters } from "../api/client";
import type {
  SemanticAssessmentDetail,
  SemanticAssessmentDomain,
  SemanticAssessmentStatus,
  SemanticAssessmentSummary,
  SemanticShadowOutcome,
  SemanticStatus
} from "../api/types";
import { useI18n, type TranslationKey } from "../i18n";

export type SemanticPanelClient = Pick<
  LibOSClient,
  "getSemanticStatus" | "listSemanticAssessments" | "getSemanticAssessment"
>;

const statuses: SemanticAssessmentStatus[] = [
  "success", "skipped_policy", "egress_blocked", "timeout", "provider_error",
  "provider_outcome_unknown", "invalid_schema", "ood", "abstained", "stale_input"
];
const domains: SemanticAssessmentDomain[] = ["filesystem", "shell", "git", "jsonrpc", "mcp", "runtime", "unknown"];
const pageSize = 50;

export function SemanticPanel({
  client,
  pid = null,
  connectionKey = ""
}: {
  client: SemanticPanelClient;
  pid?: string | null;
  connectionKey?: string;
}) {
  const { formatTime, t } = useI18n();
  const [semanticStatus, setSemanticStatus] = useState<SemanticStatus | null>(null);
  const [items, setItems] = useState<SemanticAssessmentSummary[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SemanticAssessmentDetail | null>(null);
  const [selectedStatus, setSelectedStatus] = useState<SemanticAssessmentStatus | "">("");
  const [selectedDomain, setSelectedDomain] = useState<SemanticAssessmentDomain | "">("");
  const [listBusy, setListBusy] = useState(false);
  const [detailBusy, setDetailBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshSequence, setRefreshSequence] = useState(0);
  const [detailRefreshSequence, setDetailRefreshSequence] = useState(0);
  const listAbort = useRef<AbortController | null>(null);
  const detailAbort = useRef<AbortController | null>(null);
  const listRequest = useRef(0);
  const detailRequest = useRef(0);
  const filters = useMemo<SemanticAssessmentFilters>(() => ({
    ...(pid ? { pid } : {}),
    ...(selectedStatus ? { status: selectedStatus } : {}),
    ...(selectedDomain ? { domain: selectedDomain } : {})
  }), [pid, selectedDomain, selectedStatus]);

  useEffect(() => () => {
    listAbort.current?.abort();
    detailAbort.current?.abort();
  }, []);

  useEffect(() => {
    listAbort.current?.abort();
    detailAbort.current?.abort();
    detailRequest.current += 1;
    setSemanticStatus(null);
    setItems([]);
    setCursor(null);
    setSelectedId(null);
    setDetail(null);
    setError(null);
    const controller = new AbortController();
    listAbort.current = controller;
    const request = ++listRequest.current;
    setListBusy(true);
    void Promise.all([
      client.getSemanticStatus({ signal: controller.signal, timeoutMs: 15_000 }),
      client.listSemanticAssessments(filters, pageSize, undefined, { signal: controller.signal, timeoutMs: 15_000 })
    ]).then(([nextStatus, page]) => {
      if (request !== listRequest.current) return;
      setSemanticStatus(nextStatus);
      setItems(page.items);
      setCursor(page.next_cursor);
      setSelectedId((current) => current && page.items.some((item) => item.assessment_id === current)
        ? current
        : page.items[0]?.assessment_id ?? null);
      setDetailRefreshSequence((value) => value + 1);
    }).catch((reason) => {
      if (request === listRequest.current && !isAbort(reason)) setError(errorText(reason));
    }).finally(() => {
      if (request === listRequest.current) setListBusy(false);
    });
    return () => controller.abort();
  }, [client, connectionKey, filters, refreshSequence]);

  useEffect(() => {
    detailAbort.current?.abort();
    setDetail(null);
    if (!selectedId) {
      setDetailBusy(false);
      return;
    }
    const controller = new AbortController();
    detailAbort.current = controller;
    const request = ++detailRequest.current;
    setDetailBusy(true);
    setError(null);
    void client.getSemanticAssessment(selectedId, { signal: controller.signal, timeoutMs: 15_000 }).then((next) => {
      if (request !== detailRequest.current) return;
      setDetail(next);
      setItems((current) => current.map((item) => item.assessment_id === next.assessment_id ? summaryFromDetail(next) : item));
    }).catch((reason) => {
      if (request === detailRequest.current && !isAbort(reason)) setError(errorText(reason));
    }).finally(() => {
      if (request === detailRequest.current) setDetailBusy(false);
    });
    return () => controller.abort();
  }, [client, connectionKey, detailRefreshSequence, selectedId]);

  async function loadMore() {
    if (!cursor || listBusy) return;
    listAbort.current?.abort();
    const controller = new AbortController();
    listAbort.current = controller;
    const request = ++listRequest.current;
    setListBusy(true);
    setError(null);
    try {
      const page = await client.listSemanticAssessments(filters, pageSize, cursor, {
        signal: controller.signal,
        timeoutMs: 15_000
      });
      if (request !== listRequest.current) return;
      setItems((current) => mergeAssessments(current, page.items));
      setCursor(page.next_cursor);
    } catch (reason) {
      if (request === listRequest.current && !isAbort(reason)) setError(errorText(reason));
    } finally {
      if (request === listRequest.current) setListBusy(false);
    }
  }

  return (
    <section className="providerTracePanel semanticPanel" data-testid="semantic-panel" aria-busy={listBusy || detailBusy || undefined}>
      <header className="providerTraceHeader semanticHeader">
        <span className="traceTitleIcon"><ShieldCheck size={17} aria-hidden="true" /></span>
        <span>
          <strong>{t("semantic.title")}</strong>
          <small>{pid ? t("semantic.processSubtitle", { pid }) : t("semantic.hostSubtitle")}</small>
        </span>
        <button
          type="button"
          className="semanticRefresh"
          aria-label={t("semantic.refresh")}
          disabled={listBusy}
          onClick={() => setRefreshSequence((value) => value + 1)}
        >
          <RefreshCw className={listBusy ? "spin" : undefined} size={15} aria-hidden="true" />
          {t("semantic.refresh")}
        </button>
      </header>

      <p className="semanticShadowNotice">{t("semantic.shadowNotice")}</p>

      {error ? <div className="inlineError" role="alert">{error}</div> : null}
      {semanticStatus ? <StatusOverview value={semanticStatus} /> : null}

      <div className="semanticFilters" aria-label={t("semantic.filters")}>
        <label>
          <span>{t("semantic.statusFilter")}</span>
          <select value={selectedStatus} onChange={(event) => setSelectedStatus(event.currentTarget.value as SemanticAssessmentStatus | "")}>
            <option value="">{t("semantic.all")}</option>
            {statuses.map((value) => <option value={value} key={value}>{statusLabel(value, t)}</option>)}
          </select>
        </label>
        <label>
          <span>{t("semantic.domainFilter")}</span>
          <select value={selectedDomain} onChange={(event) => setSelectedDomain(event.currentTarget.value as SemanticAssessmentDomain | "")}>
            <option value="">{t("semantic.all")}</option>
            {domains.map((value) => <option value={value} key={value}>{domainLabel(value, t)}</option>)}
          </select>
        </label>
      </div>

      <div className="providerTraceLayout semanticLayout">
        <section className="providerTraceCalls semanticList" aria-label={t("semantic.history")}>
          <h3>{t("semantic.history")}</h3>
          {items.map((item) => (
            <button
              type="button"
              key={item.assessment_id}
              className={selectedId === item.assessment_id ? "active" : ""}
              aria-pressed={selectedId === item.assessment_id}
              onClick={() => setSelectedId(item.assessment_id)}
            >
              <span><strong><bdi>{domainLabel(item.domain, t)}</bdi></strong><time>{formatTime(item.created_at)}</time></span>
              <small><bdi>{item.action_id} · {item.assessment_id}</bdi></small>
              <span className="semanticBadges">
                <OutcomeBadge outcome={item.shadow_outcome} />
                <span className={`semanticBadge status ${statusTone(item.status)}`}>{statusLabel(item.status, t)}</span>
                {item.ood ? <span className="semanticBadge warning">{t("semantic.ood")}</span> : null}
              </span>
            </button>
          ))}
          {listBusy && items.length === 0 ? <div className="empty"><LoaderCircle className="spin" size={16} />{t("semantic.loading")}</div> : null}
          {!listBusy && items.length === 0 ? <div className="empty">{t("semantic.empty")}</div> : null}
          {cursor ? <button type="button" className="traceLoadMore semanticLoadMore" disabled={listBusy} onClick={() => void loadMore()}>{t("semantic.loadMore")}</button> : null}
        </section>

        <section className="providerTraceDetail semanticDetail" aria-label={t("semantic.details")}>
          {detailBusy ? <div className="empty"><LoaderCircle className="spin" size={16} />{t("semantic.loadingDetail")}</div> : null}
          {!detailBusy && !detail ? <div className="empty">{items.length ? t("semantic.select") : t("semantic.empty")}</div> : null}
          {detail ? <AssessmentDetail detail={detail} /> : null}
        </section>
      </div>
    </section>
  );
}

function StatusOverview({ value }: { value: SemanticStatus }) {
  const { t } = useI18n();
  const queue = [
    ["semantic.queued", value.queue.queued],
    ["semantic.leased", value.queue.leased],
    ["semantic.succeeded", value.queue.succeeded],
    ["semantic.failed", value.queue.failed],
    ["semantic.cancelled", value.queue.cancelled],
    ["semantic.captureFailures", value.queue.capture_failures]
  ] as const satisfies ReadonlyArray<readonly [TranslationKey, number]>;
  return (
    <section className="semanticOverview" aria-label={t("semantic.queueHealth")}>
      <div className="semanticRuntimeFacts">
        <span><small>{t("semantic.mode")}</small><strong>{value.mode}</strong></span>
        <span><small>{t("semantic.adapter")}</small><strong>{value.adapter}</strong></span>
        <span><small>{t("semantic.profile")}</small><strong><bdi>{value.profile_id ?? t("semantic.none")}</bdi></strong></span>
        <span><small>{t("semantic.actualAutoApproval")}</small><strong>{t("semantic.notApplicable")}</strong></span>
      </div>
      <div className="semanticCounters">
        {queue.map(([label, count]) => <span key={label}><small>{t(label)}</small><strong>{count}</strong></span>)}
      </div>
      <div className="semanticCounters">
        <span><small>{t("semantic.totalAssessments")}</small><strong>{value.assessments.total}</strong></span>
        <span><small>{t("semantic.assessmentSuccess")}</small><strong>{value.assessments.success}</strong></span>
        <span><small>{t("semantic.assessmentErrors")}</small><strong>{value.assessments.error}</strong></span>
      </div>
      <div className="semanticOutcomeCounts">
        <span><OutcomeBadge outcome="would_issue_exact_once" /><strong>{value.assessments.would_issue_exact_once}</strong></span>
        <span><OutcomeBadge outcome="would_deny" /><strong>{value.assessments.would_deny}</strong></span>
        <span><OutcomeBadge outcome="require_human" /><strong>{value.assessments.require_human}</strong></span>
        <span className="semanticBadge warning">{t("semantic.ood")}: {value.assessments.ood}</span>
      </div>
      <div className="semanticAggregateGroups">
        <section>
          <h4>{t("semantic.byStatus")}</h4>
          <div className="semanticCounters compact">
            {statuses.map((status) => (
              <span key={status}>
                <small>{statusLabel(status, t)}</small>
                <strong>{value.assessments.by_status[status]}</strong>
              </span>
            ))}
          </div>
        </section>
        <section>
          <h4>{t("semantic.byDomain")}</h4>
          <div className="semanticCounters compact">
            {domains.map((domain) => (
              <span key={domain}>
                <small>{domainLabel(domain, t)}</small>
                <strong>{value.assessments.by_domain[domain]}</strong>
              </span>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}

function AssessmentDetail({ detail }: { detail: SemanticAssessmentDetail }) {
  const { formatTime, t } = useI18n();
  const digests = [
    ["semantic.artifactDigest", detail.artifact_sha256],
    ["semantic.inputDigest", detail.input_sha256],
    ["semantic.featureDigest", detail.feature_snapshot_sha256],
    ["semantic.policyDigest", detail.policy_sha256],
    ["semantic.sourceDigest", detail.source_refs_sha256],
    ["semantic.labelsDigest", detail.data_labels_sha256],
    ["semantic.sinkDigest", detail.sink_identity_sha256],
    ["semantic.toolDigest", detail.tool_schema_sha256],
    ["semantic.providerDigest", detail.provider_spec_sha256],
    ["semantic.manifestDigest", detail.manifest_sha256],
    ["semantic.actionDigest", detail.action_sha256],
    ["semantic.resourceDigest", detail.resource_sha256],
    ["semantic.argsDigest", detail.args_sha256],
    ["semantic.stateDigest", detail.state_sha256],
    ["semantic.projectionDigest", detail.projection_sha256],
    ["semantic.tenantDigest", detail.tenant_bucket_sha256]
  ] as const satisfies ReadonlyArray<readonly [TranslationKey, string | null]>;
  return (
    <div className="traceDetailBody semanticDetailBody">
      <header className="traceDetailHeader">
        <div><strong><bdi>{detail.assessment_id}</bdi></strong><small><bdi>{detail.kind} · {detail.domain}</bdi></small></div>
        <span className="semanticBadges"><OutcomeBadge outcome={detail.shadow_outcome} />{detail.ood ? <span className="semanticBadge warning">{t("semantic.ood")}</span> : null}{detail.abstain ? <span className="semanticBadge warning">{t("semantic.abstain")}</span> : null}</span>
      </header>
      <dl className="traceFacts semanticFacts">
        <Fact label={t("semantic.pid")} value={detail.pid} />
        <Fact label={t("semantic.jobId")} value={detail.job_id} />
        <Fact label={t("semantic.requestId")} value={detail.request_id ?? t("semantic.none")} />
        <Fact label={t("semantic.operationId")} value={detail.operation_id ?? t("semantic.none")} />
        <Fact label={t("semantic.effectId")} value={detail.effect_id ?? t("semantic.none")} />
        <Fact label={t("semantic.action")} value={detail.action_id} />
        <Fact label={t("semantic.classifier")} value={`${detail.classifier_id} @ ${detail.classifier_version}`} />
        <Fact label={t("semantic.confidence")} value={detail.confidence_bps === null ? t("semantic.none") : `${(detail.confidence_bps / 100).toFixed(2)}%`} />
        <Fact label={t("semantic.calibration")} value={detail.calibration_bucket} />
        <Fact label={t("semantic.inputTokens")} value={detail.input_tokens === null ? t("semantic.none") : String(detail.input_tokens)} />
        <Fact label={t("semantic.outputTokens")} value={detail.output_tokens === null ? t("semantic.none") : String(detail.output_tokens)} />
        <Fact label={t("semantic.cost")} value={detail.cost_microunits === null ? t("semantic.none") : `${detail.cost_microunits} µunits`} />
        <Fact label={t("semantic.latency")} value={detail.latency_ms === null ? t("semantic.none") : `${detail.latency_ms} ms`} />
        <Fact label={t("semantic.humanOutcome")} value={detail.human_outcome ?? t("semantic.none")} />
        <Fact label={t("semantic.created")} value={formatTime(detail.created_at)} />
        <Fact label={t("semantic.completed")} value={formatTime(detail.completed_at)} />
      </dl>
      <CodeGroup title={t("semantic.reasons")} values={detail.reason_codes} />
      <CodeGroup title={t("semantic.matchedRules")} values={detail.matched_rule_ids} />
      <CodeGroup title={t("semantic.proven")} values={detail.proven_predicates} />
      <CodeGroup title={t("semantic.missing")} values={detail.missing_predicates} />
      <section className="semanticDigestList">
        <h4>{t("semantic.provenance")}</h4>
        <dl>{digests.map(([label, digest]) => <div key={label}><dt>{t(label)}</dt><dd><code><bdi>{digest ?? t("semantic.none")}</bdi></code></dd></div>)}</dl>
      </section>
      <section className="semanticFindings">
        <h4>{t("semantic.findings")}</h4>
        {detail.findings.length === 0 ? <p className="empty">{t("semantic.noFindings")}</p> : detail.findings.map((finding, index) => (
          <article key={`${finding.evidence_sha256}:${index}`}>
            <span><strong><bdi>{finding.code}</bdi></strong><span className={`semanticBadge ${finding.severity === "high" || finding.severity === "critical" ? "danger" : "neutral"}`}>{finding.severity}</span></span>
            <small>{finding.source} · {(finding.confidence_bps / 100).toFixed(2)}%</small>
            <code><bdi>{finding.evidence_sha256}</bdi></code>
          </article>
        ))}
      </section>
      <section className="semanticFindings">
        <h4>{t("semantic.dataFindings")}</h4>
        {detail.data_findings.length === 0 ? <p className="empty">{t("semantic.noFindings")}</p> : detail.data_findings.map((finding, index) => (
          <article key={`${finding.evidence_sha256}:${index}`}>
            <span><strong><bdi>{finding.category}</bdi></strong><span className="semanticBadge neutral">{finding.sensitivity_floor}</span></span>
            <small><bdi>{finding.field}</bdi>{finding.span_start === null ? "" : ` [${finding.span_start}, ${finding.span_end}]`} · {finding.integrity_ceiling} · {finding.trust_ceiling} · {(finding.confidence_bps / 100).toFixed(2)}%</small>
            <code><bdi>{finding.evidence_sha256}</bdi></code>
          </article>
        ))}
      </section>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return <div><dt>{label}</dt><dd><bdi>{value}</bdi></dd></div>;
}

function CodeGroup({ title, values }: { title: string; values: string[] }) {
  const { t } = useI18n();
  return (
    <section className="semanticCodeGroup">
      <h4>{title}</h4>
      {values.length ? <div>{values.map((value) => <code key={value}><bdi>{value}</bdi></code>)}</div> : <small>{t("semantic.none")}</small>}
    </section>
  );
}

function OutcomeBadge({ outcome }: { outcome: SemanticShadowOutcome }) {
  const { t } = useI18n();
  const tone = outcome === "would_issue_exact_once" ? "success" : outcome === "would_deny" ? "danger" : "warning";
  return <span className={`semanticBadge ${tone}`}>{outcomeLabel(outcome, t)}</span>;
}

function outcomeLabel(outcome: SemanticShadowOutcome, t: (key: TranslationKey) => string): string {
  const keys: Record<SemanticShadowOutcome, TranslationKey> = {
    would_issue_exact_once: "semantic.outcome.wouldIssueExactOnce",
    would_deny: "semantic.outcome.wouldDeny",
    require_human: "semantic.outcome.requireHuman"
  };
  return t(keys[outcome]);
}

function statusLabel(status: SemanticAssessmentStatus, t: (key: TranslationKey) => string): string {
  return t(`semantic.status.${status}` as TranslationKey);
}

function domainLabel(domain: SemanticAssessmentDomain, t: (key: TranslationKey) => string): string {
  return t(`semantic.domain.${domain}` as TranslationKey);
}

function statusTone(status: SemanticAssessmentStatus): "success" | "warning" | "danger" | "neutral" {
  if (status === "success") return "success";
  if (["provider_error", "provider_outcome_unknown", "invalid_schema"].includes(status)) return "danger";
  if (["egress_blocked", "timeout", "ood", "abstained", "stale_input"].includes(status)) return "warning";
  return "neutral";
}

export function mergeAssessments(
  current: readonly SemanticAssessmentSummary[],
  incoming: readonly SemanticAssessmentSummary[]
): SemanticAssessmentSummary[] {
  const merged = new Map(current.map((item) => [item.assessment_id, item]));
  for (const item of incoming) merged.set(item.assessment_id, item);
  return [...merged.values()];
}

function summaryFromDetail(detail: SemanticAssessmentDetail): SemanticAssessmentSummary {
  const {
    findings: _findings,
    data_findings: _dataFindings,
    matched_rule_ids: _matchedRuleIds,
    proven_predicates: _provenPredicates,
    missing_predicates: _missingPredicates,
    source_refs_sha256: _sourceRefsSha256,
    data_labels_sha256: _dataLabelsSha256,
    sink_identity_sha256: _sinkIdentitySha256,
    tool_schema_sha256: _toolSchemaSha256,
    provider_spec_sha256: _providerSpecSha256,
    manifest_sha256: _manifestSha256,
    action_sha256: _actionSha256,
    resource_sha256: _resourceSha256,
    args_sha256: _argsSha256,
    state_sha256: _stateSha256,
    projection_sha256: _projectionSha256,
    ...summary
  } = detail;
  return summary;
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
