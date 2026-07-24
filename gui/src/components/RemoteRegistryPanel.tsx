import { Eye, ListTree, Plug, RadioTower, Send } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { LibOSClient } from "../api/client";
import type { JsonRpcEndpointSummary, McpServerSummary, RuntimeProcess } from "../api/types";
import type { ConfirmationRequest } from "../adminTypes";
import { useI18n } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";

type RemoteKind = "jsonrpc" | "mcp";
type RemoteSummary = JsonRpcEndpointSummary | McpServerSummary;

export function RemoteRegistryPanel({
  kind,
  process,
  entries,
  client,
  confirmAction
}: {
  kind: RemoteKind;
  process: RuntimeProcess | null;
  entries: RemoteSummary[];
  client: LibOSClient;
  confirmAction(request: ConfirmationRequest): void;
}) {
  const { t } = useI18n();
  const [selectedId, setSelectedId] = useState(() => entryId(kind, entries[0]) ?? "");
  const [manifest, setManifest] = useState("");
  const [replace, setReplace] = useState(false);
  const [processAuthority, setProcessAuthority] = useState(true);
  const [operationId, setOperationId] = useState("");
  const [argumentsText, setArgumentsText] = useState("{}");
  const [refreshTools, setRefreshTools] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const ids = useMemo(() => entries.map((entry) => entryId(kind, entry)).filter(isString), [entries, kind]);
  const selectedEntry = entries.find((entry) => entryId(kind, entry) === selectedId);
  const operationIds = useMemo(() => remoteOperationIds(kind, selectedEntry), [kind, selectedEntry]);

  useEffect(() => {
    if (!ids.includes(selectedId)) setSelectedId(ids[0] ?? "");
  }, [ids.join("\n"), selectedId]);

  useEffect(() => {
    setOperationId((current) => reconcileRemoteOperationId(current, operationIds));
  }, [selectedId, operationIds.join("\n")]);

  async function inspect() {
    if (!selectedId) return;
    setLoading(true);
    setLocalError(null);
    try {
      setResult(kind === "jsonrpc"
        ? await client.inspectJsonRpcEndpoint(selectedId)
        : await client.inspectMcpServer(selectedId));
    } catch (error) {
      setLocalError(describe(error));
    } finally {
      setLoading(false);
    }
  }

  async function listTools() {
    if (kind !== "mcp" || !selectedId) return;
    setLoading(true);
    setLocalError(null);
    try {
      setResult(await client.listMcpTools(selectedId, refreshTools));
    } catch (error) {
      setLocalError(describe(error));
    } finally {
      setLoading(false);
    }
  }

  function confirmRegister() {
    const manifestText = manifest.trim();
    if (!manifestText) return;
    const actor = processAuthority ? process?.pid : undefined;
    confirmAction({
      title: t(kind === "jsonrpc" ? "remote.registerJsonRpcTitle" : "remote.registerMcpTitle"),
      message: t("remote.registerMessage"),
      details: {
        kind,
        manifest_bytes: new Blob([manifestText]).size,
        replace,
        authority: actor ? "process" : "host-admin",
        actor: actor ?? null
      },
      action: async () => {
        if (kind === "jsonrpc") await client.registerJsonRpcEndpoint(manifestText, true, replace, actor);
        else await client.registerMcpServer(manifestText, true, replace, actor);
      }
    });
  }

  function confirmCall() {
    const pid = process?.pid;
    const selectedOperation = operationId.trim();
    if (!pid || !selectedId || !selectedOperation) return;
    let args: unknown;
    try {
      args = parseJsonInput(argumentsText, kind === "mcp");
      setLocalError(null);
    } catch (error) {
      setLocalError(error instanceof SyntaxError ? t("json.invalidInput") : t("json.objectInput"));
      return;
    }
    confirmAction({
      title: t(kind === "jsonrpc" ? "remote.callJsonRpcTitle" : "remote.callMcpTitle"),
      message: t("remote.callMessage"),
      details: {
        pid,
        registry_id: selectedId,
        operation_id: selectedOperation,
        arguments: args
      },
      action: async () => {
        const response = kind === "jsonrpc"
          ? await client.callJsonRpc(selectedId, pid, selectedOperation, args, true)
          : await client.callMcpTool(selectedId, pid, selectedOperation, args as Record<string, unknown>, true);
        setResult(response);
      }
    });
  }

  const label = kind === "jsonrpc" ? "JSON-RPC" : "MCP";
  return (
    <section className="adminPanel remotePanel" aria-busy={loading || undefined}>
      <header className="adminPanelHeader">
        <div>
          <h3>{kind === "jsonrpc" ? <RadioTower size={16} /> : <Plug size={16} />}{label}</h3>
          <p>{t("remote.description", { kind: label })}</p>
        </div>
      </header>

      <label className="fieldStack">
        <span>{t("remote.selected", { kind: label })}</span>
        <select value={selectedId} disabled={!ids.length} onChange={(event) => setSelectedId(event.currentTarget.value)}>
          {!ids.length ? <option value="">{t("remote.empty", { kind: label })}</option> : null}
          {ids.map((id) => <option value={id} key={id}>{id}</option>)}
        </select>
      </label>
      <div className="adminActions">
        <button disabled={!selectedId || loading} onClick={() => void inspect()}><Eye size={14} />{t("remote.inspect")}</button>
        {kind === "mcp" ? (
          <>
            <button disabled={!selectedId || loading} onClick={() => void listTools()}><ListTree size={14} />{t("remote.listTools")}</button>
            <label className="toggle"><input type="checkbox" checked={refreshTools} onChange={(event) => setRefreshTools(event.currentTarget.checked)} />{t("remote.refreshTools")}</label>
          </>
        ) : null}
      </div>

      <details className="adminDisclosure">
        <summary>{t("remote.register", { kind: label })}</summary>
        <label className="fieldStack">
          <span>{t("remote.manifest")}</span>
          <textarea className="codeInput" value={manifest} placeholder={t("remote.manifestPlaceholder")} onChange={(event) => setManifest(event.currentTarget.value)} />
        </label>
        <div className="adminActions">
          <label className="toggle"><input type="checkbox" checked={replace} onChange={(event) => setReplace(event.currentTarget.checked)} />{t("remote.replace")}</label>
          <label className="toggle">
            <input type="checkbox" disabled={!process} checked={processAuthority && Boolean(process)} onChange={(event) => setProcessAuthority(event.currentTarget.checked)} />
            {t("remote.processAuthority")}
          </label>
          <button className="warning" disabled={!manifest.trim()} onClick={confirmRegister}>{t("remote.registerAction")}</button>
        </div>
      </details>

      <details className="adminDisclosure">
        <summary>{t("remote.call", { kind: label })}</summary>
        {!process ? <p className="inlineWarning">{t("remote.selectProcess")}</p> : null}
        <div className="adminFormGrid">
          <label className="fieldStack spanAll">
            <span>{kind === "jsonrpc" ? t("remote.methodId") : t("remote.toolId")}</span>
            <input list={`${kind}-operation-options`} value={operationId} onChange={(event) => setOperationId(event.currentTarget.value)} />
            <datalist id={`${kind}-operation-options`}>
              {operationIds.map((id) => <option value={id} key={id} />)}
            </datalist>
          </label>
          <label className="fieldStack spanAll">
            <span>{t("remote.arguments")}</span>
            <textarea className="codeInput" value={argumentsText} onChange={(event) => setArgumentsText(event.currentTarget.value)} />
          </label>
        </div>
        <button className="danger" disabled={!process || !selectedId || !operationId.trim()} onClick={confirmCall}><Send size={14} />{t("remote.callAction")}</button>
      </details>

      {localError ? <div className="inlineError" role="alert">{localError}</div> : null}
      {result !== null ? <CollapsibleJson value={result} label={t("remote.result")} defaultExpanded /> : null}
    </section>
  );
}

export function parseJsonInput(value: string, requireObject: boolean): unknown {
  const selected = value.trim();
  const parsed = selected ? JSON.parse(selected) : {};
  if (requireObject && (!parsed || typeof parsed !== "object" || Array.isArray(parsed))) {
    throw new Error("MCP arguments must be a JSON object.");
  }
  return parsed;
}

export function remoteOperationIds(kind: RemoteKind, entry: RemoteSummary | undefined): string[] {
  if (!entry) return [];
  const value = kind === "jsonrpc" ? entry.methods : entry.tools;
  if (!Array.isArray(value)) return [];
  const key = kind === "jsonrpc" ? "method_id" : "tool_id";
  return value.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return [];
    const id = (item as Record<string, unknown>)[key];
    return typeof id === "string" && id ? [id] : [];
  });
}

export function reconcileRemoteOperationId(current: string, operationIds: string[]): string {
  return operationIds.includes(current) ? current : (operationIds[0] ?? "");
}

function entryId(kind: RemoteKind, entry: RemoteSummary | undefined): string | null {
  if (!entry) return null;
  const id = kind === "jsonrpc"
    ? (entry as JsonRpcEndpointSummary).endpoint_id
    : (entry as McpServerSummary).server_id;
  return typeof id === "string" && id ? id : null;
}

function isString(value: string | null): value is string {
  return value !== null;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
