import { Ban, Eye, Play, RefreshCw, Timer, Workflow } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { LibOSClient } from "../api/client";
import type { ObjectTask, RuntimeProcess, ToolSummary } from "../api/types";
import type { ConfirmationRequest, RunGuiAction } from "../adminTypes";
import { useI18n } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";

const terminalTaskStatuses = new Set([
  "succeeded",
  "failed",
  "cancelled",
  "abandoned",
  "superseded_by_restore",
  "result_unavailable_after_reopen"
]);

export function ObjectTasksPanel({
  process,
  tasks,
  tools,
  client,
  runAction,
  confirmAction
}: {
  process: RuntimeProcess;
  tasks: ObjectTask[];
  tools: ToolSummary[];
  client: LibOSClient;
  runAction: RunGuiAction;
  confirmAction(request: ConfirmationRequest): void;
}) {
  const { t } = useI18n();
  const relevantTasks = useMemo(
    () => tasks.filter((task) => task.creator_pid === process.pid || task.runner_pid === process.pid),
    [process.pid, tasks]
  );
  const [selectedId, setSelectedId] = useState(relevantTasks[0]?.task_id ?? "");
  const [tool, setTool] = useState(tools[0]?.name ?? tools[0]?.tool_id ?? "");
  const [argsText, setArgsText] = useState("{}");
  const [ownerOid, setOwnerOid] = useState(relevantTasks[0]?.owner_oid ?? "");
  const [ownerWatch, setOwnerWatch] = useState(false);
  const [watchEvents, setWatchEvents] = useState("");
  const [result, setResult] = useState<unknown>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const selected = relevantTasks.find((task) => task.task_id === selectedId) ?? null;

  useEffect(() => {
    if (!relevantTasks.some((task) => task.task_id === selectedId)) {
      setSelectedId(relevantTasks[0]?.task_id ?? "");
    }
  }, [relevantTasks, selectedId]);

  function start() {
    const selectedOwnerOid = ownerOid.trim();
    if (!selectedOwnerOid) {
      setLocalError(t("tasks.ownerRequired"));
      return;
    }
    let args: Record<string, unknown>;
    try {
      args = parseObjectInput(argsText);
      setLocalError(null);
    } catch (error) {
      setLocalError(error instanceof SyntaxError ? t("json.invalidInput") : t("json.objectInput"));
      return;
    }
    const selectedTool = tool.trim();
    const selectedEvents = splitCsv(watchEvents);
    const request = buildObjectTaskStartRequest({
      pid: process.pid,
      ownerOid: selectedOwnerOid,
      tool: selectedTool,
      args,
      ownerWatch,
      watchEvents: selectedEvents
    });
    confirmAction({
      title: t("tasks.startTitle"),
      message: t("tasks.startMessage"),
      details: { pid: process.pid, owner_oid: selectedOwnerOid, tool: selectedTool, arguments: args, owner_watch: ownerWatch, watch_events: selectedEvents },
      action: async () => {
        const task = await client.startObjectTask(request);
        setSelectedId(task.task_id);
        setResult(task);
      }
    });
  }

  async function inspect() {
    if (!selectedId) return;
    try {
      setLocalError(null);
      setResult(await client.getObjectTask(selectedId, process.pid));
    } catch (error) {
      setLocalError(describe(error));
    }
  }

  function cancel() {
    if (!selectedId) return;
    const taskId = selectedId;
    confirmAction({
      title: t("tasks.cancelTitle"),
      message: t("tasks.cancelMessage"),
      details: { pid: process.pid, task_id: taskId },
      action: async () => {
        setResult(await client.cancelObjectTask(taskId, process.pid, "Cancelled from GUI"));
      }
    });
  }

  async function wait() {
    if (!selectedId) return;
    await runAction(async () => {
      setResult(await client.waitObjectTask(selectedId, process.pid));
    }, "object_task.wait");
  }

  async function updateWatch() {
    if (!selectedId) return;
    await runAction(async () => {
      setResult(await client.watchObjectTaskOwner({
        taskId: selectedId,
        pid: process.pid,
        enabled: ownerWatch,
        watchEvents: splitCsv(watchEvents)
      }));
    }, "object_task.watch_owner");
  }

  const terminal = selected ? isObjectTaskTerminal(selected.status) : false;
  return (
    <section className="adminPanel objectTasksPanel">
      <header className="adminPanelHeader">
        <div>
          <h3><Workflow size={16} />{t("tasks.title")}</h3>
          <p>{t("tasks.description")}</p>
        </div>
      </header>

      <details className="adminDisclosure" open={relevantTasks.length === 0}>
        <summary>{t("tasks.start")}</summary>
        <div className="adminFormGrid">
          <label className="fieldStack spanAll">
            <span>{t("tasks.tool")}</span>
            <input list="runtime-tool-options" value={tool} onChange={(event) => setTool(event.currentTarget.value)} />
            <datalist id="runtime-tool-options">
              {tools.map((item) => <option key={item.tool_id} value={item.name || item.tool_id} />)}
            </datalist>
          </label>
          <label className="fieldStack spanAll">
            <span>{t("tasks.ownerOid")}</span>
            <input value={ownerOid} placeholder={t("tasks.ownerOidPlaceholder")} onChange={(event) => setOwnerOid(event.currentTarget.value)} />
          </label>
          <label className="fieldStack spanAll">
            <span>{t("tasks.arguments")}</span>
            <textarea className="codeInput" value={argsText} onChange={(event) => setArgsText(event.currentTarget.value)} />
          </label>
          <label className="toggle">
            <input type="checkbox" checked={ownerWatch} onChange={(event) => setOwnerWatch(event.currentTarget.checked)} />
            {t("tasks.ownerWatch")}
          </label>
          <label className="fieldStack">
            <span>{t("tasks.watchEvents")}</span>
            <input value={watchEvents} placeholder={t("tasks.watchEventsPlaceholder")} onChange={(event) => setWatchEvents(event.currentTarget.value)} />
          </label>
        </div>
        <button className="primary" disabled={!tool.trim() || !ownerOid.trim()} onClick={start}><Play size={14} />{t("tasks.startAction")}</button>
      </details>

      <label className="fieldStack">
        <span>{t("tasks.selected")}</span>
        <select value={selectedId} disabled={!relevantTasks.length} onChange={(event) => setSelectedId(event.currentTarget.value)}>
          {!relevantTasks.length ? <option value="">{t("tasks.empty")}</option> : null}
          {relevantTasks.map((task) => <option key={task.task_id} value={task.task_id}>{task.task_id} · {task.status} · {task.tool}</option>)}
        </select>
      </label>
      <div className="adminActions">
        <button disabled={!selectedId} onClick={() => void inspect()}><Eye size={14} />{t("tasks.inspect")}</button>
        <button disabled={!selectedId || terminal} onClick={() => void wait()}><Timer size={14} />{t("tasks.wait")}</button>
        <button disabled={!selectedId || terminal} onClick={() => void updateWatch()}><RefreshCw size={14} />{t("tasks.updateWatch")}</button>
        <button className="danger" disabled={!selectedId || terminal} onClick={cancel}><Ban size={14} />{t("tasks.cancel")}</button>
      </div>
      {selected ? <CollapsibleJson value={selected} label={t("tasks.summary")} /> : null}
      {localError ? <div className="inlineError" role="alert">{localError}</div> : null}
      {result !== null ? <CollapsibleJson value={result} label={t("tasks.result")} defaultExpanded /> : null}
    </section>
  );
}

export function parseObjectInput(value: string): Record<string, unknown> {
  const parsed = JSON.parse(value.trim() || "{}");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Arguments must be a JSON object.");
  }
  return parsed as Record<string, unknown>;
}

export function isObjectTaskTerminal(status: string): boolean {
  return terminalTaskStatuses.has(status);
}

export function buildObjectTaskStartRequest(input: {
  pid: string;
  ownerOid: string;
  tool: string;
  args: Record<string, unknown>;
  ownerWatch: boolean;
  watchEvents: string[];
}): Parameters<LibOSClient["startObjectTask"]>[0] {
  return {
    pid: input.pid,
    ownerOid: input.ownerOid.trim(),
    tool: input.tool.trim(),
    args: input.args,
    ownerWatch: input.ownerWatch,
    watchEvents: input.watchEvents
  };
}

function splitCsv(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
