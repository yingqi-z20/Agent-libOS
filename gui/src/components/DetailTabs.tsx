import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { LibOSClient } from "../api/client";
import type { AuditRecord, ExplainOperationResponse, ImageInspectResult, OperationListResponse, RuntimeProcess, RuntimeSnapshot } from "../api/types";
import type { ConfirmationRequest, RunGuiAction } from "../adminTypes";
import { useI18n, type TranslationKey } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";
import { ImagePanel } from "./ImagePanel";
import { ExplainPanel } from "./ExplainPanel";
import { RatingPanel } from "./RatingPanel";
import { CapabilityPanel } from "./CapabilityPanel";
import { CheckpointPanel } from "./CheckpointPanel";
import { ModulesPanel } from "./ModulesPanel";
import { ObjectTasksPanel } from "./ObjectTasksPanel";
import { TaskRunsPanel } from "./TaskRunsPanel";
import { ProcessOverview } from "./ProcessOverview";
import { RemoteRegistryPanel } from "./RemoteRegistryPanel";
import { SkillsPanel } from "./SkillsPanel";
import { RequestEpoch } from "../requestEpoch";
import { ProviderTracePanel, type LlmTraceFocus } from "./ProviderTracePanel";
import { SemanticPanel } from "./SemanticPanel";

const tabs = [
  { key: "overview", label: "details.overview" },
  { key: "rating", label: "details.rating" },
  { key: "capabilities", label: "details.capabilities" },
  { key: "toolsSkills", label: "details.toolsSkills" },
  { key: "checkpoints", label: "details.checkpoints" },
  { key: "taskRuns", label: "details.taskRuns" },
  { key: "tasks", label: "details.tasks" },
  { key: "audit", label: "details.audit" },
  { key: "explain", label: "details.explain" },
  { key: "llmCalls", label: "details.llmCalls" },
  { key: "semantic", label: "details.semantic" },
  { key: "jsonRpc", label: "details.jsonRpc" },
  { key: "mcp", label: "details.mcp" },
  { key: "modules", label: "details.modules" },
  { key: "images", label: "details.images" },
  { key: "objectMemory", label: "details.objectMemory" }
] as const satisfies ReadonlyArray<{ key: string; label: TranslationKey }>;
const hostTabs = new Set<TabKey>(["taskRuns", "toolsSkills", "semantic", "jsonRpc", "mcp", "modules", "images"]);

type TabKey = (typeof tabs)[number]["key"];

export function DetailTabs({
  process,
  snapshot,
  onImportImage,
  onCommitImage,
  onUseImageForSpawn,
  onUseImageForExec,
  onRate,
  onInspectImage,
  onListOperations,
  onExplainOperation,
  onResolveOperation,
  explainLookup,
  llmTraceFocus = null,
  connectionEpoch = 0,
  client,
  runAction,
  confirmAction,
  busy = false
}: {
  process: RuntimeProcess | null;
  snapshot: RuntimeSnapshot | null;
  onImportImage(replace: boolean): void;
  onCommitImage(request: { imageId: string; name: string; version: string; replace: boolean; checkpointId?: string }): void;
  onUseImageForSpawn(imageId: string): void;
  onUseImageForExec(imageId: string): void;
  onRate(pid: string, score: number, comment: string): Promise<boolean>;
  onInspectImage(imageId: string): Promise<ImageInspectResult>;
  onListOperations(pid: string, cursor?: string, signal?: AbortSignal): Promise<OperationListResponse>;
  onExplainOperation(operationId: string, cursor?: string, signal?: AbortSignal): Promise<ExplainOperationResponse>;
  onResolveOperation(kind: string, id: string, signal?: AbortSignal): Promise<ExplainOperationResponse>;
  explainLookup: { kind: string; id: string; nonce: number } | null;
  llmTraceFocus?: LlmTraceFocus | null;
  connectionEpoch?: number;
  client?: LibOSClient | null;
  runAction?: RunGuiAction;
  confirmAction?(request: ConfirmationRequest): void;
  busy?: boolean;
}) {
  const { t } = useI18n();
  const [tab, setTab] = useState<TabKey>(() => process ? "overview" : "toolsSkills");
  const tabsId = useId();
  const tabSelectId = `${tabsId}-select`;
  const tabSelectLabelId = `${tabsId}-select-label`;
  const tabPanelId = `${tabsId}-panel`;
  const tabListRef = useRef<HTMLDivElement>(null);
  const previousProcessPidRef = useRef(process?.pid ?? null);
  const visibleTabs = useMemo(
    () => process ? tabs : tabs.filter(({ key }) => hostTabs.has(key)),
    [process]
  );
  useEffect(() => {
    if (explainLookup) setTab("explain");
  }, [explainLookup?.nonce]);
  useEffect(() => {
    if (llmTraceFocus) setTab("llmCalls");
  }, [llmTraceFocus?.nonce]);
  useEffect(() => {
    const previousPid = previousProcessPidRef.current;
    const nextPid = process?.pid ?? null;
    previousProcessPidRef.current = nextPid;
    if (!nextPid && !visibleTabs.some(({ key }) => key === tab)) {
      setTab(visibleTabs[0]?.key ?? "toolsSkills");
    } else if (!previousPid && nextPid && tab === "toolsSkills") {
      setTab("overview");
    }
  }, [process?.pid, tab, visibleTabs]);
  useEffect(() => {
    const active = tabListRef.current?.querySelector<HTMLElement>("[role='tab'][aria-selected='true']");
    if (active && typeof active.scrollIntoView === "function") {
      const reducedMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
      active.scrollIntoView({ block: "nearest", inline: "nearest", behavior: reducedMotion ? "auto" : "smooth" });
    }
  }, [tab]);
  if (!snapshot) return <div className="empty">{t("details.snapshotMissing")}</div>;
  return (
    <aside className="details" aria-busy={busy || undefined} inert={busy ? true : undefined}>
      <label className="detailTabSelect" htmlFor={tabSelectId}>
        <span id={tabSelectLabelId}>{t("details.tabsLabel")}</span>
        <select
          id={tabSelectId}
          aria-labelledby={tabSelectLabelId}
          aria-controls={tabPanelId}
          value={tab}
          onChange={(event) => setTab(event.currentTarget.value as TabKey)}
        >
          {visibleTabs.map(({ key, label }) => <option value={key} key={key}>{t(label)}</option>)}
        </select>
      </label>
      <div ref={tabListRef} className="tabs" role="tablist" aria-label={t("details.tabsLabel")}>
        {visibleTabs.map(({ key, label }, index) => (
          <button
            type="button"
            role="tab"
            id={`${tabsId}-tab-${key}`}
            aria-controls={tabPanelId}
            aria-selected={tab === key}
            tabIndex={tab === key ? 0 : -1}
            key={key}
            className={tab === key ? "active" : ""}
            onClick={() => setTab(key)}
            onKeyDown={(event) => {
              const nextIndex = tabIndexForKey(index, event.key, visibleTabs.length);
              if (nextIndex === null) return;
              event.preventDefault();
              setTab(visibleTabs[nextIndex].key);
              const buttons = event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>("[role='tab']");
              buttons?.[nextIndex]?.focus();
            }}
          >
            {t(label)}
          </button>
        ))}
      </div>
      <div
        className="tabPanel"
        id={tabPanelId}
        role="tabpanel"
        aria-labelledby={`${tabSelectLabelId} ${tabsId}-tab-${tab}`}
        tabIndex={0}
      >
        {renderTab(tab, process, snapshot, t, {
          onImportImage,
          onCommitImage,
          onUseImageForSpawn,
          onUseImageForExec,
          onRate,
          onInspectImage,
          onListOperations,
          onExplainOperation,
          onResolveOperation,
          explainLookup,
          llmTraceFocus,
          connectionEpoch,
          client: client ?? null,
          runAction,
          confirmAction
        })}
      </div>
    </aside>
  );
}

export function tabIndexForKey(index: number, key: string, count: number): number | null {
  if (count < 1) return null;
  if (key === "ArrowRight" || key === "ArrowDown") return (index + 1) % count;
  if (key === "ArrowLeft" || key === "ArrowUp") return (index - 1 + count) % count;
  if (key === "Home") return 0;
  if (key === "End") return count - 1;
  return null;
}

function renderTab(
  tab: TabKey,
  process: RuntimeProcess | null,
  snapshot: RuntimeSnapshot,
  t: (key: TranslationKey) => string,
  imageActions: {
    onImportImage(replace: boolean): void;
    onCommitImage(request: { imageId: string; name: string; version: string; replace: boolean; checkpointId?: string }): void;
    onUseImageForSpawn(imageId: string): void;
    onUseImageForExec(imageId: string): void;
    onRate(pid: string, score: number, comment: string): Promise<boolean>;
    onInspectImage(imageId: string): Promise<ImageInspectResult>;
    onListOperations(pid: string, cursor?: string, signal?: AbortSignal): Promise<OperationListResponse>;
    onExplainOperation(operationId: string, cursor?: string, signal?: AbortSignal): Promise<ExplainOperationResponse>;
    onResolveOperation(kind: string, id: string, signal?: AbortSignal): Promise<ExplainOperationResponse>;
    explainLookup: { kind: string; id: string; nonce: number } | null;
    llmTraceFocus: LlmTraceFocus | null;
    connectionEpoch: number;
    client: LibOSClient | null;
    runAction?: RunGuiAction;
    confirmAction?(request: ConfirmationRequest): void;
  }
) {
  if (!process && !["taskRuns", "jsonRpc", "mcp", "toolsSkills", "semantic", "images", "modules"].includes(tab)) return <div className="empty">{t("details.selectProcess")}</div>;
  const adminReady = imageActions.client && imageActions.runAction && imageActions.confirmAction;
  if (tab === "overview" && process) return <ProcessOverview process={process} />;
  if (tab === "rating") return <RatingPanel process={process} onSave={imageActions.onRate} />;
  if (tab === "capabilities" && process && adminReady) return <CapabilityPanel key={adminPanelKey(process, imageActions.connectionEpoch)} process={process} client={imageActions.client!} confirmAction={imageActions.confirmAction!} reloadKey={adminRefreshKey(process, snapshot)} />;
  if (tab === "toolsSkills" && process && adminReady) return <SkillsPanel key={`${imageActions.connectionEpoch}:${process.pid}`} process={process} skills={snapshot.skills} tools={snapshot.tools} client={imageActions.client!} confirmAction={imageActions.confirmAction!} />;
  if (tab === "toolsSkills") return <JsonBlock value={{ process_tools: process?.tool_table, loaded_skills: process?.loaded_skills, registry: snapshot.skills, tools: snapshot.tools }} />;
  if (tab === "checkpoints" && process && adminReady) return <CheckpointPanel key={adminPanelKey(process, imageActions.connectionEpoch)} process={process} client={imageActions.client!} runAction={imageActions.runAction!} confirmAction={imageActions.confirmAction!} reloadKey={adminRefreshKey(process, snapshot)} />;
  if (tab === "taskRuns" && adminReady) return <TaskRunsPanel key={`${imageActions.connectionEpoch}:task-runs`} runs={snapshot.task_runs} client={imageActions.client!} runAction={imageActions.runAction!} confirmAction={imageActions.confirmAction!} />;
  if (tab === "tasks" && process && adminReady) return <ObjectTasksPanel key={`${imageActions.connectionEpoch}:${process.pid}`} process={process} tasks={snapshot.object_tasks} tools={snapshot.tools} client={imageActions.client!} runAction={imageActions.runAction!} confirmAction={imageActions.confirmAction!} />;
  if (tab === "audit" && process && imageActions.client) return <AuditPanel key={`${imageActions.connectionEpoch}:${process.pid}`} process={process} snapshot={snapshot} client={imageActions.client} />;
  if (tab === "audit") return <JsonBlock value={snapshot.audit.filter((item) => item.actor === process?.pid || item.target === `process:${process?.pid}`)} />;
  if (tab === "explain" && process) {
    return (
      <ExplainPanel
        key={explainPanelKey(process, imageActions.connectionEpoch)}
        pid={process.pid}
        listOperations={imageActions.onListOperations}
        explainOperation={imageActions.onExplainOperation}
        resolveOperation={imageActions.onResolveOperation}
        lookup={imageActions.explainLookup}
        refreshKey={explainRefreshKey(process, snapshot)}
        connectionKey={String(imageActions.connectionEpoch)}
      />
    );
  }
  if (tab === "llmCalls" && process && imageActions.client) {
    return (
      <ProviderTracePanel
        key={providerTracePanelKey(process.pid, imageActions.connectionEpoch)}
        pid={process.pid}
        client={imageActions.client}
        snapshotCalls={snapshot.llm_calls}
        focus={imageActions.llmTraceFocus}
        connectionKey={String(imageActions.connectionEpoch)}
      />
    );
  }
  if (tab === "llmCalls") return <div className="empty">{t("app.clientUnavailable")}</div>;
  if (tab === "semantic" && imageActions.client) {
    return (
      <SemanticPanel
        key={`${imageActions.connectionEpoch}:${process?.pid ?? "host"}`}
        pid={process?.pid ?? null}
        client={imageActions.client}
        connectionKey={String(imageActions.connectionEpoch)}
      />
    );
  }
  if (tab === "semantic") return <div className="empty">{t("app.clientUnavailable")}</div>;
  if (tab === "jsonRpc" && imageActions.client && imageActions.confirmAction) return <RemoteRegistryPanel key={`${imageActions.connectionEpoch}:jsonrpc:${process?.pid ?? "host"}`} kind="jsonrpc" process={process} entries={snapshot.jsonrpc_endpoints} client={imageActions.client} confirmAction={imageActions.confirmAction} />;
  if (tab === "jsonRpc") return <JsonBlock value={snapshot.jsonrpc_endpoints} />;
  if (tab === "mcp" && imageActions.client && imageActions.confirmAction) return <RemoteRegistryPanel key={`${imageActions.connectionEpoch}:mcp:${process?.pid ?? "host"}`} kind="mcp" process={process} entries={snapshot.mcp_servers} client={imageActions.client} confirmAction={imageActions.confirmAction} />;
  if (tab === "mcp") return <JsonBlock value={snapshot.mcp_servers} />;
  if (tab === "modules" && imageActions.client) return <ModulesPanel key={imageActions.connectionEpoch} modules={snapshot.modules} client={imageActions.client} />;
  if (tab === "modules") return <JsonBlock value={snapshot.modules} />;
  if (tab === "images") {
    return (
      <ImagePanel
        key={`${imageActions.connectionEpoch}:${process?.pid ?? "no-process"}`}
        images={snapshot.images}
        selectedProcess={process}
        allowReplace
        onImportImage={imageActions.onImportImage}
        onCommitImage={imageActions.onCommitImage}
        onUseForSpawn={imageActions.onUseImageForSpawn}
        onUseForExec={imageActions.onUseImageForExec}
        onInspectImage={imageActions.onInspectImage}
      />
    );
  }
  return <JsonBlock value={{ goal_oid: process?.goal_oid, note: t("details.objectMemoryNote") }} />;
}

export function adminPanelKey(process: RuntimeProcess, connectionEpoch = 0): string {
  return `${connectionEpoch}:${process.pid}`;
}

export function providerTracePanelKey(pid: string, connectionEpoch = 0): string {
  return `${connectionEpoch}:${pid}`;
}

export function adminRefreshKey(process: RuntimeProcess, snapshot: RuntimeSnapshot): string {
  const processAudit = snapshot.audit.filter((item) =>
    item.actor === process.pid || item.target === `process:${process.pid}`
  );
  const processEvents = snapshot.events.filter((item) =>
    item.source === process.pid || item.target === process.pid || item.target === `process:${process.pid}`
  );
  return [
    process.pid,
    process.state_generation,
    process.checkpoint_head,
    process.capabilities.join(","),
    processAudit.at(-1)?.record_id,
    processEvents.at(-1)?.event_id
  ].join(":");
}

export function explainRefreshKey(process: RuntimeProcess, snapshot: RuntimeSnapshot): string {
  const processAudit = snapshot.audit.filter((item) => item.actor === process.pid || item.target === `process:${process.pid}`);
  const processEvents = snapshot.events.filter((item) => item.source === process.pid || item.target === process.pid || item.target === `process:${process.pid}`);
  const processLlmCalls = snapshot.llm_calls.filter((item) => item.pid === process.pid);
  const processHumanRequests = snapshot.human_requests.filter((item) => item.pid === process.pid);
  return [
    process.pid,
    process.state_generation,
    processAudit.at(-1)?.record_id,
    processEvents.at(-1)?.event_id,
    processLlmCalls.at(-1)?.call_id,
    processHumanRequests.at(-1)?.updated_at
  ].join(":");
}

export function explainPanelKey(process: RuntimeProcess, connectionEpoch = 0): string {
  return connectionEpoch === 0 ? process.pid : `${connectionEpoch}:${process.pid}`;
}

function AuditPanel({ process, snapshot, client }: { process: RuntimeProcess; snapshot: RuntimeSnapshot; client: LibOSClient }) {
  const { t } = useI18n();
  const snapshotRecords = useMemo(
    () => snapshot.audit.filter((item) => item.actor === process.pid || item.target === `process:${process.pid}`),
    [process.pid, snapshot.audit]
  );
  const [loadedRecords, setLoadedRecords] = useState<AuditRecord[]>([]);
  const [canLoadMore, setCanLoadMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requests = useRef(new RequestEpoch());
  const abort = useRef<AbortController | null>(null);
  const records = useMemo(() => mergeAuditRecords(loadedRecords, snapshotRecords), [loadedRecords, snapshotRecords]);

  useEffect(() => {
    abort.current?.abort();
    requests.current.invalidate();
    setLoadedRecords([]);
    setCanLoadMore(true);
    setLoading(false);
    setError(null);
    return () => {
      abort.current?.abort();
      requests.current.invalidate();
    };
  }, [client, process.pid]);

  async function loadOlder() {
    const before = records[0]?.record_id;
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    const request = requests.current.begin();
    setLoading(true);
    setError(null);
    try {
      const page = await client.listProcessAudit(process.pid, undefined, before, { signal: controller.signal, timeoutMs: 15_000 });
      if (!requests.current.isCurrent(request)) return;
      setLoadedRecords((current) => mergeAuditRecords(page, current));
      setCanLoadMore(page.length > 0);
    } catch (reason) {
      if (requests.current.isCurrent(request)) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (requests.current.isCurrent(request)) setLoading(false);
    }
  }

  return (
    <section className="auditPanel" aria-busy={loading || undefined}>
      <JsonBlock value={records} />
      {error ? <div className="inlineError" role="alert">{error}</div> : null}
      {canLoadMore ? <button type="button" disabled={loading} onClick={() => void loadOlder()}>{t("explain.loadMore")}</button> : null}
    </section>
  );
}

export function mergeAuditRecords(...pages: AuditRecord[][]): AuditRecord[] {
  const records = new Map<string, AuditRecord>();
  for (const page of pages) for (const record of page) records.set(record.record_id, record);
  return Array.from(records.values()).sort((left, right) =>
    left.timestamp.localeCompare(right.timestamp) || left.record_id.localeCompare(right.record_id)
  );
}

function JsonBlock({ value }: { value: unknown }) {
  const { t } = useI18n();
  return <CollapsibleJson value={value} label={t("details.rawData")} />;
}
