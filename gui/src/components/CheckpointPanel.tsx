import { GitBranch, History, RefreshCw, RotateCcw, Save, Search } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { LibOSClient } from "../api/client";
import type { CheckpointSummary, RuntimeProcess } from "../api/types";
import type { ConfirmationRequest, RunGuiAction } from "../adminTypes";
import { useI18n } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";

export function CheckpointPanel({
  process,
  client,
  runAction,
  confirmAction,
  reloadKey
}: {
  process: RuntimeProcess;
  client: LibOSClient;
  runAction: RunGuiAction;
  confirmAction(request: ConfirmationRequest): void;
  reloadKey: string;
}) {
  const { t } = useI18n();
  const [items, setItems] = useState<CheckpointSummary[]>([]);
  const [selectedId, setSelectedId] = useState(process.checkpoint_head ?? "");
  const [reason, setReason] = useState("");
  const [details, setDetails] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const requestSequence = useRef(0);
  const activePid = useRef(process.pid);
  activePid.current = process.pid;

  useEffect(() => {
    requestSequence.current += 1;
    setItems([]);
    setDetails(null);
    setSelectedId(process.checkpoint_head ?? "");
    void load();
    return () => { requestSequence.current += 1; };
    // reloadKey is the explicit evidence/snapshot invalidation signal.
  }, [process.pid, reloadKey]);

  async function load(): Promise<CheckpointSummary[]> {
    const pid = process.pid;
    const sequence = ++requestSequence.current;
    setLoading(true);
    setLocalError(null);
    try {
      const loaded = await client.listCheckpoints(pid);
      if (sequence !== requestSequence.current || activePid.current !== pid) return [];
      setItems(loaded);
      setSelectedId((current) => loaded.some((item) => item.checkpoint_id === current)
        ? current
        : loaded[0]?.checkpoint_id ?? "");
      return loaded;
    } catch (error) {
      if (sequence === requestSequence.current && activePid.current === pid) {
        setLocalError(describe(error));
      }
      return [];
    } finally {
      if (sequence === requestSequence.current && activePid.current === pid) setLoading(false);
    }
  }

  async function create() {
    const pid = process.pid;
    const sequence = ++requestSequence.current;
    const ok = await runAction(async () => {
      const result = await client.createCheckpoint(pid, reason.trim() || "GUI checkpoint");
      if (sequence === requestSequence.current && activePid.current === pid) {
        setSelectedId(result.checkpoint_id);
        setReason("");
      }
    }, "checkpoint.create");
    if (ok && sequence === requestSequence.current && activePid.current === pid) await load();
  }

  async function inspect(mode: "inspect" | "diff") {
    if (!selectedId) return;
    const pid = process.pid;
    const sequence = ++requestSequence.current;
    setLoading(true);
    setLocalError(null);
    try {
      const nextDetails = mode === "inspect"
        ? await client.inspectCheckpoint(selectedId)
        : await client.diffCheckpoint(selectedId);
      if (sequence === requestSequence.current && activePid.current === pid) setDetails(nextDetails);
    } catch (error) {
      if (sequence === requestSequence.current && activePid.current === pid) {
        setLocalError(describe(error));
      }
    } finally {
      if (sequence === requestSequence.current && activePid.current === pid) setLoading(false);
    }
  }

  function confirm(kind: "restore" | "fork") {
    if (!selectedId) return;
    const checkpointId = selectedId;
    confirmAction({
      title: t(kind === "restore" ? "checkpoint.restoreTitle" : "checkpoint.forkTitle"),
      message: t(kind === "restore" ? "checkpoint.restoreMessage" : "checkpoint.forkMessage"),
      details: {
        checkpoint_id: checkpointId,
        source_pid: process.pid,
        mode: "host-admin"
      },
      action: async () => {
        if (kind === "restore") await client.restoreCheckpoint(checkpointId, true);
        else await client.forkCheckpoint(checkpointId, true, process.pid);
      }
    });
  }

  return (
    <section className="adminPanel checkpointPanel" aria-busy={loading || undefined}>
      <header className="adminPanelHeader">
        <div>
          <h3><History size={16} />{t("checkpoint.title")}</h3>
          <p>{t("checkpoint.description")}</p>
        </div>
        <button className="iconOnly" aria-label={t("checkpoint.refresh")} title={t("checkpoint.refresh")} disabled={loading} onClick={() => void load()}>
          <RefreshCw className={loading ? "spin" : ""} size={14} />
        </button>
      </header>

      <div className="inlineForm">
        <label className="growField">
          <span>{t("checkpoint.reason")}</span>
          <input value={reason} placeholder={t("checkpoint.reasonPlaceholder")} onChange={(event) => setReason(event.currentTarget.value)} />
        </label>
        <button className="primary" disabled={loading} onClick={() => void create()}><Save size={14} />{t("checkpoint.create")}</button>
      </div>

      <label className="fieldStack">
        <span>{t("checkpoint.selected")}</span>
        <select value={selectedId} disabled={loading || items.length === 0} onChange={(event) => setSelectedId(event.currentTarget.value)}>
          {items.length === 0 ? <option value="">{t("checkpoint.empty")}</option> : null}
          {items.map((checkpoint) => (
            <option key={checkpoint.checkpoint_id} value={checkpoint.checkpoint_id}>
              {checkpointLabel(checkpoint)}
            </option>
          ))}
        </select>
      </label>

      <div className="adminActions">
        <button disabled={!selectedId || loading} onClick={() => void inspect("inspect")}><Search size={14} />{t("checkpoint.inspect")}</button>
        <button disabled={!selectedId || loading} onClick={() => void inspect("diff")}><GitBranch size={14} />{t("checkpoint.diff")}</button>
        <button className="warning" disabled={!selectedId || loading} onClick={() => confirm("fork")}><GitBranch size={14} />{t("checkpoint.fork")}</button>
        <button className="danger" disabled={!selectedId || loading} onClick={() => confirm("restore")}><RotateCcw size={14} />{t("checkpoint.restore")}</button>
      </div>

      {localError ? <div className="inlineError" role="alert">{localError}</div> : null}
      {details !== null ? <CollapsibleJson value={details} label={t("checkpoint.result")} defaultExpanded /> : null}
    </section>
  );
}

export function checkpointLabel(checkpoint: CheckpointSummary): string {
  const parts = [checkpoint.checkpoint_id];
  if (checkpoint.created_at) parts.push(formatCompactDate(checkpoint.created_at));
  if (checkpoint.reason) parts.push(checkpoint.reason);
  return parts.join(" · ");
}

function formatCompactDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
