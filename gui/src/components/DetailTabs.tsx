import { useEffect, useId, useState } from "react";
import type { LibOSClient } from "../api/client";
import type { ExplainOperationResponse, ImageInspectResult, OperationListResponse, RuntimeProcess, RuntimeSnapshot } from "../api/types";
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
import { ProcessOverview } from "./ProcessOverview";
import { RemoteRegistryPanel } from "./RemoteRegistryPanel";
import { SkillsPanel } from "./SkillsPanel";

const tabs = [
  { key: "overview", label: "details.overview" },
  { key: "rating", label: "details.rating" },
  { key: "capabilities", label: "details.capabilities" },
  { key: "toolsSkills", label: "details.toolsSkills" },
  { key: "checkpoints", label: "details.checkpoints" },
  { key: "tasks", label: "details.tasks" },
  { key: "audit", label: "details.audit" },
  { key: "explain", label: "details.explain" },
  { key: "llmCalls", label: "details.llmCalls" },
  { key: "jsonRpc", label: "details.jsonRpc" },
  { key: "mcp", label: "details.mcp" },
  { key: "modules", label: "details.modules" },
  { key: "images", label: "details.images" },
  { key: "objectMemory", label: "details.objectMemory" }
] as const satisfies ReadonlyArray<{ key: string; label: TranslationKey }>;

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
  client,
  runAction,
  confirmAction
}: {
  process: RuntimeProcess | null;
  snapshot: RuntimeSnapshot | null;
  onImportImage(replace: boolean): void;
  onCommitImage(request: { imageId: string; name: string; version: string; replace: boolean; checkpointId?: string }): void;
  onUseImageForSpawn(imageId: string): void;
  onUseImageForExec(imageId: string): void;
  onRate(pid: string, score: number, comment: string): Promise<boolean>;
  onInspectImage(imageId: string): Promise<ImageInspectResult>;
  onListOperations(pid: string, cursor?: string): Promise<OperationListResponse>;
  onExplainOperation(operationId: string, cursor?: string): Promise<ExplainOperationResponse>;
  onResolveOperation(kind: string, id: string): Promise<ExplainOperationResponse>;
  explainLookup: { kind: string; id: string; nonce: number } | null;
  client?: LibOSClient | null;
  runAction?: RunGuiAction;
  confirmAction?(request: ConfirmationRequest): void;
}) {
  const { t } = useI18n();
  const [tab, setTab] = useState<TabKey>("overview");
  const tabsId = useId();
  useEffect(() => {
    if (explainLookup) setTab("explain");
  }, [explainLookup?.nonce]);
  if (!snapshot) return <div className="empty">{t("details.snapshotMissing")}</div>;
  return (
    <aside className="details">
      <div className="tabs" role="tablist" aria-label={t("details.tabsLabel")}>
        {tabs.map(({ key, label }, index) => (
          <button
            type="button"
            role="tab"
            id={`${tabsId}-tab-${key}`}
            aria-controls={`${tabsId}-panel`}
            aria-selected={tab === key}
            tabIndex={tab === key ? 0 : -1}
            key={key}
            className={tab === key ? "active" : ""}
            onClick={() => setTab(key)}
            onKeyDown={(event) => {
              const nextIndex = tabIndexForKey(index, event.key, tabs.length);
              if (nextIndex === null) return;
              event.preventDefault();
              setTab(tabs[nextIndex].key);
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
        id={`${tabsId}-panel`}
        role="tabpanel"
        aria-labelledby={`${tabsId}-tab-${tab}`}
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
    onListOperations(pid: string, cursor?: string): Promise<OperationListResponse>;
    onExplainOperation(operationId: string, cursor?: string): Promise<ExplainOperationResponse>;
    onResolveOperation(kind: string, id: string): Promise<ExplainOperationResponse>;
    explainLookup: { kind: string; id: string; nonce: number } | null;
    client: LibOSClient | null;
    runAction?: RunGuiAction;
    confirmAction?(request: ConfirmationRequest): void;
  }
) {
  if (!process && !["jsonRpc", "mcp", "toolsSkills", "images", "modules"].includes(tab)) return <div className="empty">{t("details.selectProcess")}</div>;
  const adminReady = imageActions.client && imageActions.runAction && imageActions.confirmAction;
  if (tab === "overview" && process) return <ProcessOverview process={process} />;
  if (tab === "rating") return <RatingPanel process={process} onSave={imageActions.onRate} />;
  if (tab === "capabilities" && process && adminReady) return <CapabilityPanel key={process.pid} process={process} client={imageActions.client!} confirmAction={imageActions.confirmAction!} reloadKey={adminRefreshKey(process, snapshot)} />;
  if (tab === "toolsSkills" && process && adminReady) return <SkillsPanel key={process.pid} process={process} skills={snapshot.skills} tools={snapshot.tools} client={imageActions.client!} confirmAction={imageActions.confirmAction!} />;
  if (tab === "toolsSkills") return <JsonBlock value={{ process_tools: process?.tool_table, loaded_skills: process?.loaded_skills, registry: snapshot.skills, tools: snapshot.tools }} />;
  if (tab === "checkpoints" && process && adminReady) return <CheckpointPanel key={process.pid} process={process} client={imageActions.client!} runAction={imageActions.runAction!} confirmAction={imageActions.confirmAction!} reloadKey={adminRefreshKey(process, snapshot)} />;
  if (tab === "tasks" && process && adminReady) return <ObjectTasksPanel key={process.pid} process={process} tasks={snapshot.object_tasks} tools={snapshot.tools} client={imageActions.client!} runAction={imageActions.runAction!} confirmAction={imageActions.confirmAction!} />;
  if (tab === "audit") return <JsonBlock value={snapshot.audit.filter((item) => item.actor === process?.pid || item.target === `process:${process?.pid}`)} />;
  if (tab === "explain" && process) {
    return (
      <ExplainPanel
        key={explainRefreshKey(process, snapshot)}
        pid={process.pid}
        listOperations={imageActions.onListOperations}
        explainOperation={imageActions.onExplainOperation}
        resolveOperation={imageActions.onResolveOperation}
        lookup={imageActions.explainLookup}
      />
    );
  }
  if (tab === "llmCalls") return <JsonBlock value={snapshot.llm_calls.filter((item) => item.pid === process?.pid)} />;
  if (tab === "jsonRpc" && imageActions.client && imageActions.confirmAction) return <RemoteRegistryPanel key={`jsonrpc:${process?.pid ?? "host"}`} kind="jsonrpc" process={process} entries={snapshot.jsonrpc_endpoints} client={imageActions.client} confirmAction={imageActions.confirmAction} />;
  if (tab === "jsonRpc") return <JsonBlock value={snapshot.jsonrpc_endpoints} />;
  if (tab === "mcp" && imageActions.client && imageActions.confirmAction) return <RemoteRegistryPanel key={`mcp:${process?.pid ?? "host"}`} kind="mcp" process={process} entries={snapshot.mcp_servers} client={imageActions.client} confirmAction={imageActions.confirmAction} />;
  if (tab === "mcp") return <JsonBlock value={snapshot.mcp_servers} />;
  if (tab === "modules" && imageActions.client) return <ModulesPanel modules={snapshot.modules} client={imageActions.client} />;
  if (tab === "modules") return <JsonBlock value={snapshot.modules} />;
  if (tab === "images") {
    return (
      <ImagePanel
        key={process?.pid ?? "no-process"}
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

function adminRefreshKey(process: RuntimeProcess, snapshot: RuntimeSnapshot): string {
  return [
    process.pid,
    process.state_generation,
    process.checkpoint_head,
    process.capabilities.join(","),
    snapshot.audit.at(-1)?.record_id,
    snapshot.events.at(-1)?.event_id
  ].join(":");
}

export function explainRefreshKey(process: RuntimeProcess, snapshot: RuntimeSnapshot): string {
  return [
    process.pid,
    snapshot.audit.at(-1)?.record_id,
    snapshot.events.at(-1)?.event_id,
    snapshot.llm_calls.at(-1)?.call_id,
    snapshot.human_requests.at(-1)?.updated_at
  ].join(":");
}

function JsonBlock({ value }: { value: unknown }) {
  const { t } = useI18n();
  return <CollapsibleJson value={value} label={t("details.rawData")} />;
}
