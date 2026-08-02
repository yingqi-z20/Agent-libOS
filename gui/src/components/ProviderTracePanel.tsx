import { AlertTriangle, Bot, ChevronRight, LoaderCircle, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, type LibOSClient } from "../api/client";
import type {
  LlmCallDetail,
  LlmCallSummary,
  LlmPayloadRetentionTier,
  LlmProviderAttempt,
  LlmTraceContentDescriptor,
  LlmTraceContentField
} from "../api/types";
import { useI18n, type TranslationKey } from "../i18n";

export type ProviderTraceClient = Pick<
  LibOSClient,
  "listProcessLlmCalls" | "getProcessLlmCall" | "getProcessLlmCallContent"
>;

export type LlmTraceFocus = { callId: string; nonce: number };

const userContentFields = new Set<LlmTraceContentField>([
  "attempt_reasoning",
  "attempt_output"
]);
const operatorCallFields = ["messages", "tools", "request_options", "raw_response", "response_content"] as const;
const attemptFields = ["attempt_reasoning", "attempt_output", "attempt_tool_calls"] as const;
const maxClientContentChars = 4 * 1024 * 1024;

export function ProviderTracePanel({
  pid,
  client,
  snapshotCalls = [],
  focus = null,
  mode = "operator",
  connectionKey = ""
}: {
  pid: string;
  client: ProviderTraceClient;
  snapshotCalls?: LlmCallSummary[];
  focus?: LlmTraceFocus | null;
  mode?: "operator" | "user";
  connectionKey?: string;
}) {
  const { formatTime, t } = useI18n();
  const seedCalls = useMemo(
    () => snapshotCalls.filter((call) => call.pid === pid),
    [pid, snapshotCalls]
  );
  const [calls, setCalls] = useState<LlmCallSummary[]>(seedCalls);
  const [cursor, setCursor] = useState<string | null>(null);
  const [selectedCallId, setSelectedCallId] = useState<string | null>(focus?.callId ?? seedCalls[0]?.call_id ?? null);
  const [detail, setDetail] = useState<LlmCallDetail | null>(null);
  const [listBusy, setListBusy] = useState(false);
  const [detailBusy, setDetailBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listAbort = useRef<AbortController | null>(null);
  const detailAbort = useRef<AbortController | null>(null);
  const panelRef = useRef<HTMLElement>(null);
  const listRequest = useRef(0);
  const detailRequest = useRef(0);
  const selectedSnapshotRevision = useMemo(() => {
    const selected = seedCalls.find((call) => call.call_id === selectedCallId);
    return selected
      ? `${selected.status}:${selected.completed_at ?? ""}:${selected.attempt_count}:${selected.reasoning_availability}:${selected.payload_retention_tier}`
      : "";
  }, [seedCalls, selectedCallId]);

  useEffect(() => () => {
    listAbort.current?.abort();
    detailAbort.current?.abort();
  }, []);

  useEffect(() => {
    listAbort.current?.abort();
    detailAbort.current?.abort();
    listRequest.current += 1;
    detailRequest.current += 1;
    setCalls(seedCalls);
    setCursor(null);
    setDetail(null);
    setError(null);
    setSelectedCallId(focus?.callId ?? seedCalls[0]?.call_id ?? null);
    const controller = new AbortController();
    listAbort.current = controller;
    const request = ++listRequest.current;
    setListBusy(true);
    void client.listProcessLlmCalls(pid, 50, undefined, { signal: controller.signal, timeoutMs: 15_000 }).then((page) => {
      if (request !== listRequest.current) return;
      setCalls(mergeLlmCallSummaries(page.items, seedCalls));
      setCursor(page.next_cursor);
      setSelectedCallId((current) => current ?? page.items[0]?.call_id ?? null);
    }).catch((reason) => {
      if (request === listRequest.current && !isAbort(reason)) setError(errorText(reason));
    }).finally(() => {
      if (request === listRequest.current) setListBusy(false);
    });
    return () => controller.abort();
  }, [client, connectionKey, pid]);

  useEffect(() => {
    setCalls((current) => mergeSnapshotLlmCallSummaries(current, seedCalls));
  }, [seedCalls]);

  useEffect(() => {
    if (!focus) return;
    setSelectedCallId(focus.callId);
    globalThis.requestAnimationFrame?.(() => panelRef.current?.focus({ preventScroll: true }));
  }, [focus?.nonce]);

  useEffect(() => {
    detailAbort.current?.abort();
    detailRequest.current += 1;
    setDetail(null);
    if (!selectedCallId) {
      setDetailBusy(false);
      return;
    }
    const controller = new AbortController();
    detailAbort.current = controller;
    const request = ++detailRequest.current;
    setDetailBusy(true);
    setError(null);
    void client.getProcessLlmCall(pid, selectedCallId, { signal: controller.signal, timeoutMs: 15_000 }).then((next) => {
      if (request !== detailRequest.current) return;
      setDetail(next);
      setCalls((current) => mergeLlmCallSummaries(current, [next.call]));
    }).catch((reason) => {
      if (request === detailRequest.current && !isAbort(reason)) setError(errorText(reason));
    }).finally(() => {
      if (request === detailRequest.current) setDetailBusy(false);
    });
    return () => controller.abort();
  }, [client, connectionKey, pid, selectedCallId, selectedSnapshotRevision]);

  async function loadMoreCalls() {
    if (!cursor || listBusy) return;
    listAbort.current?.abort();
    const controller = new AbortController();
    listAbort.current = controller;
    const request = ++listRequest.current;
    setListBusy(true);
    setError(null);
    try {
      const page = await client.listProcessLlmCalls(pid, 50, cursor, { signal: controller.signal, timeoutMs: 15_000 });
      if (request !== listRequest.current) return;
      setCalls((current) => mergeLlmCallSummaries(current, page.items));
      setCursor(page.next_cursor);
    } catch (reason) {
      if (request === listRequest.current && !isAbort(reason)) setError(errorText(reason));
    } finally {
      if (request === listRequest.current) setListBusy(false);
    }
  }

  return (
    <section
      ref={panelRef}
      className={`providerTracePanel ${mode}`}
      data-testid="provider-trace-panel"
      aria-label={t("trace.title")}
      aria-busy={listBusy || detailBusy || undefined}
      tabIndex={-1}
    >
      <header className="providerTraceHeader">
        <span className="traceTitleIcon"><Bot size={17} aria-hidden="true" /></span>
        <span><strong>{t("trace.title")}</strong><small>{t("trace.subtitle")}</small></span>
      </header>
      <p className="providerTraceWarning"><AlertTriangle size={14} aria-hidden="true" />{t("trace.untrusted")}</p>
      {error ? <div className="inlineError" role="alert">{error}</div> : null}
      <div className="providerTraceLayout">
        <section className="providerTraceCalls" aria-label={t("trace.calls")}>
          {calls.map((call, index) => (
            <button
              type="button"
              key={call.call_id}
              data-testid={mode === "operator" ? `provider-trace-call-${call.call_id}` : "provider-trace-call"}
              className={selectedCallId === call.call_id ? "active" : ""}
              aria-pressed={selectedCallId === call.call_id}
              onClick={() => setSelectedCallId(call.call_id)}
            >
              {mode === "operator" ? (
                <>
                  <span><strong><bdi>{call.model ?? call.api ?? t("trace.unknownModel")}</bdi></strong><time>{formatTime(call.created_at)}</time></span>
                  <small><bdi>{call.purpose || call.call_id}</bdi></small>
                  <span className="traceCallMeta"><span>{call.status}</span><span>{t("trace.attemptCount", { count: call.attempt_count })}</span><span>{availabilityLabel(call.reasoning_availability, t)}</span></span>
                </>
              ) : (
                <>
                  <span><strong>{t("trace.call", { sequence: index + 1 })}</strong><span>{call.status}</span></span>
                  <small>{availabilityLabel(call.reasoning_availability, t)}</small>
                </>
              )}
            </button>
          ))}
          {!listBusy && calls.length === 0 ? <div className="empty">{t("trace.empty")}</div> : null}
          {listBusy && calls.length === 0 ? <div className="empty"><LoaderCircle className="spin" size={16} />{t("trace.loading")}</div> : null}
          {cursor ? <button type="button" className="traceLoadMore" disabled={listBusy} onClick={() => void loadMoreCalls()}>{t("trace.loadMore")}</button> : null}
        </section>
        <section className="providerTraceDetail" aria-label={t("trace.detail")}>
          {detailBusy ? <div className="empty"><LoaderCircle className="spin" size={16} />{t("trace.loading")}</div> : null}
          {!detailBusy && !detail ? <div className="empty">{calls.length ? t("trace.selectCall") : t("trace.empty")}</div> : null}
          {detail ? (
            <TraceDetail
              key={`${detail.call.call_id}:${detail.call.payload_retention_tier}:${detail.call.reasoning_availability}`}
              detail={detail}
              pid={pid}
              client={client}
              mode={mode}
            />
          ) : null}
        </section>
      </div>
    </section>
  );
}

function TraceDetail({
  detail,
  pid,
  client,
  mode
}: {
  detail: LlmCallDetail;
  pid: string;
  client: ProviderTraceClient;
  mode: "operator" | "user";
}) {
  const { formatTime, t } = useI18n();
  const callDescriptors = detail.content.filter((item) => item.attempt_sequence === null);
  return (
    <div className="traceDetailBody">
      <header className="traceDetailHeader">
        {mode === "operator" ? (
          <div><strong><bdi>{detail.call.model ?? detail.call.api ?? t("trace.unknownModel")}</bdi></strong><small><bdi>{detail.call.call_id}</bdi></small></div>
        ) : <div><strong>{t("trace.response")}</strong></div>}
        <span className="statusPill"><span className="statusDot" />{detail.call.status}</span>
      </header>
      {mode === "operator" ? (
        <>
          <dl className="traceFacts">
            <div><dt>{t("trace.coverage")}</dt><dd>{coverageLabel(detail.call.coverage, t)}</dd></div>
            <div><dt>{t("trace.retention")}</dt><dd>{detail.call.payload_retention_tier}</dd></div>
            <div><dt>{t("trace.started")}</dt><dd>{formatTime(detail.call.created_at)}</dd></div>
            <div><dt>{t("trace.usage")}</dt><dd><InertJson value={detail.call.usage} /></dd></div>
          </dl>
          {detail.call.coverage !== "complete" ? <p className="inlineWarning">{coverageNotice(detail.call.coverage, t)}</p> : null}
        </>
      ) : null}
      <section className="traceAttempts" aria-label={t("trace.attempts")}>
        <h3>{t("trace.attempts")}</h3>
        {detail.attempts.length === 0 ? <div className="empty">{t("trace.noAttempts")}</div> : detail.attempts.map((attempt) => (
          <TraceAttempt
            key={attempt.sequence}
            attempt={attempt}
            descriptors={detail.content.filter((item) => item.attempt_sequence === attempt.sequence)}
            pid={pid}
            callId={detail.call.call_id}
            client={client}
            retentionTier={detail.call.payload_retention_tier}
            mode={mode}
            defaultOpen={detail.call.selected_attempt === attempt.sequence}
          />
        ))}
      </section>
      {mode === "operator" ? (
        <details className="traceLowLevel">
          <summary><ChevronRight size={14} aria-hidden="true" />{t("trace.lowLevel")}</summary>
          <p>{t("trace.lowLevelHint")}</p>
          <div className="traceContentGrid">
            {operatorCallFields.map((field) => (
              <TraceContent
                key={field}
                descriptor={descriptorFor(callDescriptors, field)}
                field={field}
                pid={pid}
                callId={detail.call.call_id}
                client={client}
                retentionTier={detail.call.payload_retention_tier}
              />
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}

function TraceAttempt({
  attempt,
  descriptors,
  pid,
  callId,
  client,
  retentionTier,
  mode,
  defaultOpen
}: {
  attempt: LlmProviderAttempt;
  descriptors: LlmTraceContentDescriptor[];
  pid: string;
  callId: string;
  client: ProviderTraceClient;
  retentionTier: LlmPayloadRetentionTier;
  mode: "operator" | "user";
  defaultOpen: boolean;
}) {
  const { formatTime, t } = useI18n();
  return (
    <details className="traceAttempt" data-testid={`provider-attempt-${attempt.sequence}`} open={defaultOpen || undefined}>
      <summary>
        <ChevronRight size={15} aria-hidden="true" />
        <span>
          <strong>{t("trace.attempt", { sequence: attempt.sequence })}</strong>
          <small>{mode === "operator" ? <bdi>{attempt.kind} · {attempt.api ?? t("trace.unknownApi")}</bdi> : availabilityLabel(attempt.reasoning_availability, t)}</small>
        </span>
        <span className={`traceAttemptStatus ${attempt.status}`}>{attempt.status}</span>
      </summary>
      <div className="traceAttemptBody">
        {mode === "operator" ? (
          <>
            <dl className="traceFacts compact">
              <div><dt>{t("trace.model")}</dt><dd><bdi>{attempt.model ?? t("trace.unknownModel")}</bdi></dd></div>
              <div><dt>{t("trace.duration")}</dt><dd>{attempt.duration_ms === null ? t("trace.notAvailable") : `${attempt.duration_ms.toLocaleString()} ms`}</dd></div>
              <div><dt>{t("trace.started")}</dt><dd>{attempt.started_at ? formatTime(attempt.started_at) : t("trace.notAvailable")}</dd></div>
              <div><dt>{t("trace.usage")}</dt><dd><InertJson value={attempt.usage} /></dd></div>
            </dl>
        {attempt.error ? <p className="inlineError"><bdi>{attempt.error.error_type ?? t("trace.failedAttempt")}</bdi>{attempt.error.status_code !== null ? ` · HTTP ${attempt.error.status_code}` : ""}{attempt.error.message_sha256 ? ` · sha256:${attempt.error.message_sha256}` : ""}</p> : null}
          </>
        ) : null}
        {attempt.tool_names.length ? <p className="traceToolNames"><strong>{t("trace.tools")}</strong> {attempt.tool_names.map((name) => <bdi key={name}>{name}</bdi>)}</p> : null}
        {mode === "operator" && attempt.reasoning_blocks.length ? (
          <details className="traceBlockMetadata"><summary>{t("trace.blockMetadata")}</summary><InertJson value={attempt.reasoning_blocks} /></details>
        ) : null}
        <div className="traceContentGrid">
          {attemptFields.filter((field) => mode === "operator" || userContentFields.has(field)).map((field) => (
            <TraceContent
              key={field}
              descriptor={descriptorFor(descriptors, field, attempt.sequence)}
              field={field}
              attemptSequence={attempt.sequence}
              pid={pid}
              callId={callId}
              client={client}
              retentionTier={retentionTier}
              showMetadata={mode === "operator"}
            />
          ))}
        </div>
      </div>
    </details>
  );
}

export function TraceContent({
  descriptor,
  field,
  attemptSequence,
  pid,
  callId,
  client,
  retentionTier,
  showMetadata = true
}: {
  descriptor: LlmTraceContentDescriptor | null;
  field: LlmTraceContentField;
  attemptSequence?: number;
  pid: string;
  callId: string;
  client: ProviderTraceClient;
  retentionTier: LlmPayloadRetentionTier;
  showMetadata?: boolean;
}) {
  const { t } = useI18n();
  const [content, setContent] = useState("");
  const [cursor, setCursor] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [invalidated, setInvalidated] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const request = useRef(0);
  const descriptorIdentity = JSON.stringify([
    retentionTier,
    descriptor?.content_hash ?? "",
    descriptor?.cursor ?? "",
    descriptor?.availability ?? "not_returned"
  ]);

  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => {
    abort.current?.abort();
    request.current += 1;
    setContent("");
    setCursor(null);
    setLoaded(false);
    setBusy(false);
    setError(null);
    setInvalidated(false);
  }, [descriptorIdentity]);

  async function load(nextCursor: string, append: boolean) {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    const currentRequest = ++request.current;
    setBusy(true);
    setError(null);
    try {
      const chunk = await client.getProcessLlmCallContent(pid, callId, field, {
        attemptSequence,
        cursor: nextCursor,
        limit: 32_768,
        signal: controller.signal,
        timeoutMs: 15_000
      });
      if (request.current !== currentRequest) return;
      const combined = append ? content + chunk.content : chunk.content;
      const clientLimited = combined.length > maxClientContentChars;
      setContent(clientLimited ? combined.slice(0, maxClientContentChars) : combined);
      setLoaded(true);
      setInvalidated(false);
      if (clientLimited) {
        setError(t("trace.clientLimit"));
        setCursor(null);
      } else {
        setCursor(chunk.next_cursor);
      }
    } catch (reason) {
      if (request.current !== currentRequest || isAbort(reason)) return;
      if (reason instanceof ApiError && reason.status === 409) {
        controller.abort();
        request.current += 1;
        setContent("");
        setCursor(null);
        setLoaded(false);
        setBusy(false);
        setInvalidated(true);
        setError(t("trace.contentChanged"));
        return;
      }
      setError(errorText(reason));
    } finally {
      if (request.current === currentRequest) setBusy(false);
    }
  }

  const availability = descriptor?.availability ?? "not_returned";
  return (
    <section className={`traceContent ${field}`} data-testid="provider-trace-content" aria-label={contentFieldLabel(field, t)}>
      <header><strong>{contentFieldLabel(field, t)}</strong>{showMetadata && descriptor?.size_chars !== null && descriptor?.size_chars !== undefined ? <small>{t("trace.chars", { count: descriptor.size_chars })}</small> : null}</header>
      {availability !== "available" && availability !== "limited" ? <p className="traceUnavailable">{contentAvailabilityLabel(availability, t)}</p> : null}
      {!loaded && !invalidated && (availability === "available" || availability === "limited") ? (
        <button type="button" disabled={busy || !descriptor?.cursor} onClick={() => descriptor?.cursor && void load(descriptor.cursor, false)}>{busy ? <LoaderCircle className="spin" size={14} /> : null}{t("trace.reveal")}</button>
      ) : null}
      {loaded ? <pre className="traceInertText" dir="auto"><bdi>{content || t("trace.emptyContent")}</bdi></pre> : null}
      {loaded && availability === "limited" ? <p className="inlineWarning">{t("trace.limited")}</p> : null}
      {error ? <p className="inlineError" role="alert">{error}</p> : null}
      {cursor ? <button type="button" disabled={busy} onClick={() => void load(cursor, true)}>{busy ? <LoaderCircle className="spin" size={14} /> : <RefreshCw size={14} />}{t("trace.loadMoreContent")}</button> : null}
    </section>
  );
}

function descriptorFor(
  descriptors: LlmTraceContentDescriptor[],
  field: LlmTraceContentField,
  attemptSequence?: number
): LlmTraceContentDescriptor | null {
  return descriptors.find((item) => item.field === field && item.attempt_sequence === (attemptSequence ?? null)) ?? null;
}

function InertJson({ value }: { value: unknown }) {
  return <pre className="traceInertJson" dir="auto"><bdi>{JSON.stringify(value, null, 2)}</bdi></pre>;
}

export function mergeLlmCallSummaries(...pages: LlmCallSummary[][]): LlmCallSummary[] {
  const merged = new Map<string, LlmCallSummary>();
  const order: string[] = [];
  for (const page of pages) {
    for (const item of page) {
      if (!merged.has(item.call_id)) order.push(item.call_id);
      merged.set(item.call_id, item);
    }
  }
  return order.map((id) => merged.get(id)!).filter(Boolean);
}

export function mergeSnapshotLlmCallSummaries(
  current: LlmCallSummary[],
  snapshot: LlmCallSummary[]
): LlmCallSummary[] {
  return mergeLlmCallSummaries(current, snapshot);
}

function contentFieldLabel(field: LlmTraceContentField, t: Translate): string {
  return t(`trace.field.${field}` as TranslationKey);
}

function availabilityLabel(value: LlmCallSummary["reasoning_availability"], t: Translate): string {
  return t(`trace.reasoning.${value}` as TranslationKey);
}

function contentAvailabilityLabel(value: LlmTraceContentDescriptor["availability"], t: Translate): string {
  return t(`trace.content.${value}` as TranslationKey);
}

function coverageLabel(value: LlmCallSummary["coverage"], t: Translate): string {
  return t(`trace.coverage.${value}` as TranslationKey);
}

function coverageNotice(value: LlmCallSummary["coverage"], t: Translate): string {
  return t(`trace.coverageNotice.${value}` as TranslationKey);
}

function isAbort(reason: unknown): boolean {
  return reason instanceof DOMException && reason.name === "AbortError";
}

function errorText(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;
