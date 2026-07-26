import { useEffect, useMemo, useRef, useState } from "react";
import type { ExplainOperationResponse, OperationEvidence, OperationListResponse, OperationSummary } from "../api/types";
import { useI18n } from "../i18n";
import { RequestEpoch } from "../requestEpoch";
import { CollapsibleJson } from "./CollapsibleJson";

export function ExplainPanel({
  pid,
  listOperations,
  explainOperation,
  resolveOperation,
  lookup,
  refreshKey,
  connectionKey
}: {
  pid: string;
  listOperations(pid: string, cursor?: string, signal?: AbortSignal): Promise<OperationListResponse>;
  explainOperation(operationId: string, cursor?: string, signal?: AbortSignal): Promise<ExplainOperationResponse>;
  resolveOperation(kind: string, id: string, signal?: AbortSignal): Promise<ExplainOperationResponse>;
  lookup: { kind: string; id: string; nonce: number } | null;
  refreshKey: string;
  connectionKey: string;
}) {
  const { formatTime, t } = useI18n();
  const [operations, setOperations] = useState<OperationSummary[]>([]);
  const [operationCursor, setOperationCursor] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<ExplainOperationResponse | null>(null);
  const [evidenceType, setEvidenceType] = useState("all");
  const [listBusy, setListBusy] = useState(false);
  const [detailBusy, setDetailBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listRequests = useRef(new RequestEpoch());
  const detailRequests = useRef(new RequestEpoch());
  const selectedIdRef = useRef(selectedId);
  const lookupRef = useRef(lookup);
  const handledLookupNonce = useRef<number | null>(null);
  const listAbort = useRef<AbortController | null>(null);
  const detailAbort = useRef<AbortController | null>(null);
  const listOperationsRef = useRef(listOperations);
  const explainOperationRef = useRef(explainOperation);
  const resolveOperationRef = useRef(resolveOperation);
  selectedIdRef.current = selectedId;
  lookupRef.current = lookup;
  listOperationsRef.current = listOperations;
  explainOperationRef.current = explainOperation;
  resolveOperationRef.current = resolveOperation;
  const busy = listBusy || detailBusy;

  useEffect(() => () => {
    listAbort.current?.abort();
    detailAbort.current?.abort();
    listRequests.current.invalidate();
    detailRequests.current.invalidate();
  }, []);

  useEffect(() => {
    listAbort.current?.abort();
    detailAbort.current?.abort();
    const listController = new AbortController();
    const detailController = new AbortController();
    listAbort.current = listController;
    detailAbort.current = detailController;
    const listRequest = listRequests.current.begin();
    const detailRequest = detailRequests.current.begin();
    setError(null);
    setListBusy(true);
    setDetailBusy(false);
    void listOperationsRef.current(pid, undefined, listController.signal).then((response) => {
      if (!listRequests.current.isCurrent(listRequest)) return;
      setOperations(response.operations);
      setOperationCursor(response.next_cursor);
      const pendingLookup = lookupRef.current && handledLookupNonce.current !== lookupRef.current.nonce;
      const targetId = operationIdForRefresh(selectedIdRef.current, response.operations, Boolean(pendingLookup));
      const target = targetId ? { operation_id: targetId } : null;
      if (!target) {
        if (!selectedIdRef.current) setExplanation(null);
        return;
      }
      if (pendingLookup || !detailRequests.current.isCurrent(detailRequest)) return;
      setDetailBusy(true);
      void explainOperationRef.current(target.operation_id, undefined, detailController.signal).then((detail) => {
        if (!detailRequests.current.isCurrent(detailRequest)) return;
        setSelectedId(detail.selected_operation_id);
        setExplanation(detail);
        setEvidenceType("all");
      }).catch((reason) => {
        if (detailRequests.current.isCurrent(detailRequest)) setError(String(reason));
      }).finally(() => {
        if (detailRequests.current.isCurrent(detailRequest)) setDetailBusy(false);
      });
    }).catch((reason) => {
      if (listRequests.current.isCurrent(listRequest)) setError(String(reason));
    }).finally(() => {
      if (listRequests.current.isCurrent(listRequest)) setListBusy(false);
    });
    return () => {
      listController.abort();
      detailController.abort();
      if (listRequests.current.isCurrent(listRequest)) listRequests.current.invalidate();
      if (detailRequests.current.isCurrent(detailRequest)) detailRequests.current.invalidate();
    };
  }, [connectionKey, pid, refreshKey]);

  useEffect(() => {
    if (!lookup || handledLookupNonce.current === lookup.nonce) return;
    handledLookupNonce.current = lookup.nonce;
    detailAbort.current?.abort();
    const controller = new AbortController();
    detailAbort.current = controller;
    const request = detailRequests.current.begin();
    setDetailBusy(true);
    setError(null);
    void resolveOperationRef.current(lookup.kind, lookup.id, controller.signal).then((detail) => {
      if (!detailRequests.current.isCurrent(request)) return;
      setOperations((current) => mergeOperations(current, detail.operations));
      setSelectedId(detail.selected_operation_id);
      setExplanation(detail);
      setEvidenceType("all");
    }).catch((reason) => {
      if (detailRequests.current.isCurrent(request)) setError(String(reason));
    }).finally(() => {
      if (detailRequests.current.isCurrent(request)) setDetailBusy(false);
    });
    return () => {
      controller.abort();
      if (detailRequests.current.isCurrent(request)) detailRequests.current.invalidate();
    };
  }, [connectionKey, lookup?.nonce, pid]);

  const evidenceTypes = useMemo(
    () => ["all", ...Array.from(new Set((explanation?.evidence ?? []).map((item) => item.evidence_type))).sort()],
    [explanation]
  );
  const visibleEvidence = useMemo(
    () => filterOperationEvidence(explanation?.evidence ?? [], evidenceType),
    [evidenceType, explanation]
  );

  async function select(operationId: string) {
    handledLookupNonce.current = lookupRef.current?.nonce ?? handledLookupNonce.current;
    detailAbort.current?.abort();
    const controller = new AbortController();
    detailAbort.current = controller;
    const request = detailRequests.current.begin();
    setDetailBusy(true);
    setError(null);
    setSelectedId(operationId);
    try {
      const detail = await explainOperationRef.current(operationId, undefined, controller.signal);
      if (!detailRequests.current.isCurrent(request)) return;
      setSelectedId(detail.selected_operation_id);
      setExplanation(detail);
      setEvidenceType("all");
    } catch (reason) {
      if (detailRequests.current.isCurrent(request)) setError(String(reason));
    } finally {
      if (detailRequests.current.isCurrent(request)) setDetailBusy(false);
    }
  }

  async function loadMoreOperations() {
    if (!operationCursor) return;
    const cursor = operationCursor;
    listAbort.current?.abort();
    const controller = new AbortController();
    listAbort.current = controller;
    const request = listRequests.current.begin();
    setListBusy(true);
    setError(null);
    try {
      const response = await listOperationsRef.current(pid, cursor, controller.signal);
      if (!listRequests.current.isCurrent(request)) return;
      setOperations((current) => mergeOperations(current, response.operations));
      setOperationCursor(response.next_cursor);
    } catch (reason) {
      if (listRequests.current.isCurrent(request)) setError(String(reason));
    } finally {
      if (listRequests.current.isCurrent(request)) setListBusy(false);
    }
  }

  async function loadMoreEvidence() {
    if (!explanation?.next_cursor) return;
    detailAbort.current?.abort();
    const controller = new AbortController();
    detailAbort.current = controller;
    const current = explanation;
    const request = detailRequests.current.begin();
    setDetailBusy(true);
    setError(null);
    try {
      const next = await explainOperationRef.current(current.selected_operation_id, current.next_cursor ?? undefined, controller.signal);
      if (!detailRequests.current.isCurrent(request)) return;
      setExplanation((latest) => latest?.selected_operation_id === current.selected_operation_id
        ? mergeEvidencePage(latest, next)
        : latest);
    } catch (reason) {
      if (detailRequests.current.isCurrent(request)) setError(String(reason));
    } finally {
      if (detailRequests.current.isCurrent(request)) setDetailBusy(false);
    }
  }

  if (busy && operations.length === 0 && !explanation) return <div className="empty">{t("explain.loading")}</div>;
  if (error) return <div className="empty explainError">{error}</div>;
  if (operations.length === 0 && !explanation) return <div className="empty">{t("explain.empty")}</div>;

  return (
    <div className="explainPanel">
      <section className="explainOperationList">
        <h3>{t("explain.operations")}</h3>
        {operations.map((operation) => (
          <button
            type="button"
            className={selectedId === operation.operation_id ? "active" : ""}
            key={operation.operation_id}
            onClick={() => void select(operation.operation_id)}
          >
            <strong>{operation.name}</strong>
            <span>{operation.outcome} · {formatTime(operation.started_at)}</span>
          </button>
        ))}
        {operationCursor ? <button type="button" disabled={listBusy} onClick={() => void loadMoreOperations()}>{t("explain.loadMore")}</button> : null}
      </section>

      {explanation ? (
        <>
          <section className={`explainSummary ${explanation.summary.outcome}`}>
            <h3>{explanation.summary.headline}</h3>
            <div className="explainBadges">
              <span>{t("explain.outcome")}: {explanation.summary.outcome}</span>
              <span>{t("explain.complete")}: {explanation.evidence_complete ? t("common.yes") : t("common.no")}</span>
              <span>{t("explain.effects")}: {explanation.summary.external_effects.length}</span>
              <span>{t("explain.resourceCharges")}: {explanation.summary.resource_charge_count}</span>
            </div>
            {explanation.missing_evidence.length ? <CollapsibleJson value={{ missing_evidence: explanation.missing_evidence }} /> : null}
            {explanation.uncertainties.length ? <CollapsibleJson value={{ uncertainties: explanation.uncertainties }} /> : null}
            <CollapsibleJson value={{
              authorization: explanation.summary.authorization,
              human: explanation.summary.human,
              external_effects: explanation.summary.external_effects,
              resource_consumption: explanation.summary.resource_consumption,
              context: explanation.summary.context
            }} />
          </section>

          <section className="explainTree">
            <h3>{t("explain.causalTree")}</h3>
            <OperationTree operations={explanation.operations} rootId={explanation.root.operation_id} selectedId={selectedId} onSelect={select} />
          </section>

          <section className="explainEvidence">
            <div className="explainEvidenceHeader">
              <h3>{t("explain.timeline")}</h3>
              <select aria-label={t("explain.timeline")} value={evidenceType} onChange={(event) => setEvidenceType(event.currentTarget.value)}>
                {evidenceTypes.map((value) => <option value={value} key={value}>{value}</option>)}
              </select>
            </div>
            {visibleEvidence.map((item) => <EvidenceRow key={`${item.evidence_type}:${item.evidence_id}`} item={item} />)}
            {explanation.next_cursor ? <button type="button" disabled={detailBusy} onClick={() => void loadMoreEvidence()}>{t("explain.loadMore")}</button> : null}
          </section>
        </>
      ) : null}
    </div>
  );
}

function OperationTree({
  operations,
  rootId,
  selectedId,
  onSelect
}: {
  operations: OperationSummary[];
  rootId: string;
  selectedId: string | null;
  onSelect(operationId: string): Promise<void>;
}) {
  const root = operations.find((item) => item.operation_id === rootId);
  if (!root) return null;
  const children = buildOperationChildren(operations);
  const render = (item: OperationSummary) => (
    <li key={item.operation_id}>
      <button type="button" className={selectedId === item.operation_id ? "active" : ""} onClick={() => void onSelect(item.operation_id)}>
        {item.kind} · {item.name} · {item.outcome}
      </button>
      {(children.get(item.operation_id) ?? []).length ? <ul>{(children.get(item.operation_id) ?? []).map(render)}</ul> : null}
    </li>
  );
  return <ul className="operationTree">{render(root)}</ul>;
}

export function buildOperationChildren(operations: OperationSummary[]): Map<string, OperationSummary[]> {
  const children = new Map<string, OperationSummary[]>();
  for (const item of operations) {
    if (!item.parent_operation_id) continue;
    children.set(item.parent_operation_id, [...(children.get(item.parent_operation_id) ?? []), item]);
  }
  return children;
}

export function operationIdForRefresh(
  selectedId: string | null,
  operations: OperationSummary[],
  lookupPending: boolean
): string | null {
  if (selectedId) return selectedId;
  return lookupPending ? null : operations[0]?.operation_id ?? null;
}

export function filterOperationEvidence(items: OperationEvidence[], evidenceType: string): OperationEvidence[] {
  return items.filter((item) => evidenceType === "all" || item.evidence_type === evidenceType);
}

export function mergeEvidencePage(
  current: ExplainOperationResponse,
  next: ExplainOperationResponse
): ExplainOperationResponse {
  const evidence = new Map<string, OperationEvidence>();
  for (const item of [...current.evidence, ...next.evidence]) {
    const key = `${item.evidence_type}:${item.evidence_id}`;
    const existing = evidence.get(key);
    evidence.set(key, existing ? {
      ...existing,
      ...item,
      roles: Array.from(new Set([...existing.roles, ...item.roles])).sort()
    } : item);
  }
  return {
    ...next,
    evidence: Array.from(evidence.values())
  };
}

function mergeOperations(current: OperationSummary[], next: OperationSummary[]): OperationSummary[] {
  const merged = new Map(current.map((item) => [item.operation_id, item]));
  for (const item of next) merged.set(item.operation_id, item);
  return Array.from(merged.values());
}

function EvidenceRow({ item }: { item: OperationEvidence }) {
  const { formatTime } = useI18n();
  return (
    <article className="explainEvidenceRow">
      <header>
        <strong>{item.evidence_type}</strong>
        <span>{item.roles.join(", ")}</span>
        <time>{item.occurred_at ? formatTime(item.occurred_at) : "—"}</time>
      </header>
      <code>{item.evidence_id}</code>
      <CollapsibleJson value={item.data} />
    </article>
  );
}
