import { Compass, Eye, ListTree, Plug, RadioTower, Send } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { LibOSClient } from "../api/client";
import type { JsonRpcEndpointSummary, McpConnectionInfo, McpProtocolEra, McpProtocolMode, McpServerSummary, RuntimeProcess } from "../api/types";
import type { ConfirmationRequest } from "../adminTypes";
import { useI18n } from "../i18n";
import { RequestEpoch } from "../requestEpoch";
import { CollapsibleJson } from "./CollapsibleJson";
import { McpModernPanel } from "./McpModernPanel";

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
  const [mcpConnection, setMcpConnection] = useState<McpConnectionInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const inspectionRequests = useRef(new RequestEpoch());
  const selectedIdRef = useRef(selectedId);
  const clientRef = useRef(client);
  const kindRef = useRef(kind);
  selectedIdRef.current = selectedId;
  clientRef.current = client;
  kindRef.current = kind;
  const ids = useMemo(() => entries.map((entry) => entryId(kind, entry)).filter(isString), [entries, kind]);
  const selectedEntry = entries.find((entry) => entryId(kind, entry) === selectedId);
  const operationIds = useMemo(() => remoteOperationIds(kind, selectedEntry), [kind, selectedEntry]);
  const mcpArgumentFields = useMemo(
    () => kind === "mcp" ? schemaArgumentFields(selectedEntry as McpServerSummary | undefined, operationId) : [],
    [kind, selectedEntry, operationId]
  );
  const selectedEntryVersion = registryEntryVersion(kind, selectedEntry);
  const selectedMcpEntry = kind === "mcp" ? selectedEntry as McpServerSummary | undefined : undefined;
  const canDiscover = (selectedMcpEntry?.schema_version === 2
    && (selectedMcpEntry.protocol_mode === "auto" || selectedMcpEntry.protocol_mode === "2026-07-28"))
    || (selectedMcpEntry?.schema_version === 3
      && selectedMcpEntry.protocol_mode === "2026-07-28");
  const panelTitleId = `remote-${kind}-panel-title`;
  const discoverHintId = `remote-${kind}-discover-hint`;

  useEffect(() => {
    if (!ids.includes(selectedId)) setSelectedId(ids[0] ?? "");
  }, [ids.join("\n"), selectedId]);

  useEffect(() => {
    setOperationId((current) => reconcileRemoteOperationId(current, operationIds));
  }, [selectedId, operationIds.join("\n")]);

  useEffect(() => {
    inspectionRequests.current.invalidate();
    setLoading(false);
    setResult(null);
    setMcpConnection(null);
    setLocalError(null);
    return () => inspectionRequests.current.invalidate();
  }, [client, kind, selectedEntryVersion, selectedId]);

  async function inspect() {
    if (!selectedId) return;
    const inspectedId = selectedId;
    const inspectedClient = client;
    const inspectedKind = kind;
    const request = inspectionRequests.current.begin();
    setLoading(true);
    setLocalError(null);
    try {
      const response = kind === "jsonrpc"
        ? await client.inspectJsonRpcEndpoint(selectedId)
        : await client.inspectMcpServer(selectedId);
      if (inspectionRequests.current.isCurrent(request) && selectedIdRef.current === inspectedId && clientRef.current === inspectedClient && kindRef.current === inspectedKind) {
        setResult(response);
        updateMcpConnection(response, setMcpConnection);
      }
    } catch (error) {
      if (inspectionRequests.current.isCurrent(request) && selectedIdRef.current === inspectedId && clientRef.current === inspectedClient && kindRef.current === inspectedKind) setLocalError(describe(error));
    } finally {
      if (inspectionRequests.current.isCurrent(request) && selectedIdRef.current === inspectedId && clientRef.current === inspectedClient && kindRef.current === inspectedKind) setLoading(false);
    }
  }

  async function listTools() {
    if (kind !== "mcp" || !selectedId) return;
    const inspectedId = selectedId;
    const inspectedClient = client;
    const request = inspectionRequests.current.begin();
    setLoading(true);
    setLocalError(null);
    if (refreshTools) {
      setResult(null);
      setMcpConnection(null);
    }
    try {
      const response = await client.listMcpTools(selectedId, refreshTools);
      if (inspectionRequests.current.isCurrent(request) && selectedIdRef.current === inspectedId && clientRef.current === inspectedClient && kindRef.current === "mcp") {
        setResult(response);
        updateMcpConnection(response, setMcpConnection);
      }
    } catch (error) {
      if (inspectionRequests.current.isCurrent(request) && selectedIdRef.current === inspectedId && clientRef.current === inspectedClient && kindRef.current === "mcp") setLocalError(describe(error));
    } finally {
      if (inspectionRequests.current.isCurrent(request) && selectedIdRef.current === inspectedId && clientRef.current === inspectedClient && kindRef.current === "mcp") setLoading(false);
    }
  }

  async function discover() {
    if (kind !== "mcp" || !selectedId || !canDiscover) return;
    const inspectedId = selectedId;
    const inspectedClient = client;
    const request = inspectionRequests.current.begin();
    setLoading(true);
    setLocalError(null);
    setResult(null);
    setMcpConnection(null);
    try {
      const response = await client.discoverMcpServer(selectedId);
      if (inspectionRequests.current.isCurrent(request) && selectedIdRef.current === inspectedId && clientRef.current === inspectedClient && kindRef.current === "mcp") {
        const connection = connectionFromResult(response);
        if (!connection) throw new Error(t("remote.discoveryInvalid"));
        setMcpConnection(connection);
        setResult(response);
      }
    } catch (error) {
      if (inspectionRequests.current.isCurrent(request) && selectedIdRef.current === inspectedId && clientRef.current === inspectedClient && kindRef.current === "mcp") {
        setMcpConnection(null);
        setLocalError(describe(error));
      }
    } finally {
      if (inspectionRequests.current.isCurrent(request) && selectedIdRef.current === inspectedId && clientRef.current === inspectedClient && kindRef.current === "mcp") setLoading(false);
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
        const calledId = selectedId;
        const calledClient = client;
        const calledKind = kind;
        const response = kind === "jsonrpc"
          ? await client.callJsonRpc(selectedId, pid, selectedOperation, args, true)
          : await client.callMcpTool(selectedId, pid, selectedOperation, args as Record<string, unknown>, true);
        if (selectedIdRef.current === calledId && clientRef.current === calledClient && kindRef.current === calledKind) {
          setResult(response);
          updateMcpConnection(response, setMcpConnection);
        }
      }
    });
  }

  function confirmUnregister() {
    if (kind !== "mcp" || !selectedId) return;
    const selectedServerId = selectedId;
    const actor = processAuthority ? process?.pid : undefined;
    confirmAction({
      title: "Unregister MCP server",
      message: "This closes Runtime-owned connections and revokes the selected registry authority.",
      details: { server_id: selectedServerId, authority: actor ? "process" : "host-admin", actor: actor ?? null },
      action: async () => {
        await client.unregisterMcpServer(selectedServerId, true, actor);
      }
    });
  }

  const label = kind === "jsonrpc" ? "JSON-RPC" : "MCP";
  return (
    <section className="adminPanel remotePanel" aria-busy={loading || undefined} aria-labelledby={panelTitleId}>
      <header className="adminPanelHeader">
        <div>
          <h3 id={panelTitleId}>{kind === "jsonrpc" ? <RadioTower size={16} /> : <Plug size={16} />}{label}</h3>
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
            <button
              disabled={!selectedId || loading || !canDiscover}
              aria-describedby={discoverHintId}
              onClick={() => void discover()}
            >
              <Compass size={14} />{t("remote.discover")}
            </button>
            <button disabled={!selectedId || loading} onClick={() => void listTools()}><ListTree size={14} />{t("remote.listTools")}</button>
            <label className="toggle"><input type="checkbox" checked={refreshTools} onChange={(event) => setRefreshTools(event.currentTarget.checked)} />{t("remote.refreshTools")}</label>
            <button className="warning" disabled={!selectedId || loading} onClick={confirmUnregister}>Unregister</button>
          </>
        ) : null}
      </div>
      {kind === "mcp" ? (
        <p id={discoverHintId} className="remoteDiscoverHint">
          {canDiscover ? t("remote.discoverHint") : t("remote.discoverUnavailable")}
        </p>
      ) : null}

      {kind === "mcp" ? <McpProtocolSummary server={selectedMcpEntry} connection={mcpConnection} /> : null}
      {kind === "mcp" && selectedMcpEntry?.schema_version === 3 ? (
        <McpModernPanel
          key={`${selectedId}:${selectedEntryVersion}`}
          serverId={selectedId}
          authProfileId={selectedMcpEntry.auth_profile_id}
          client={client}
          confirmAction={confirmAction}
        />
      ) : null}

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
          <div className="fieldStack spanAll">
            <span>{t("remote.arguments")}</span>
            {kind === "mcp" && mcpArgumentFields.length ? (
              <div className="mcpSchemaArguments" aria-label="Schema-driven MCP arguments">
                {mcpArgumentFields.map((field) => (
                  <McpSchemaArgumentField
                    key={field.name}
                    field={field}
                    argumentsText={argumentsText}
                    onChange={setArgumentsText}
                  />
                ))}
              </div>
            ) : null}
            <textarea aria-label={`${label} JSON arguments`} className="codeInput" value={argumentsText} onChange={(event) => setArgumentsText(event.currentTarget.value)} />
          </div>
        </div>
        <button className="danger" disabled={!process || !selectedId || !operationId.trim()} onClick={confirmCall}><Send size={14} />{t("remote.callAction")}</button>
      </details>

      {localError ? <div className="inlineError" role="alert">{localError}</div> : null}
      {result !== null ? <CollapsibleJson value={result} label={t("remote.result")} defaultExpanded /> : null}
    </section>
  );
}

function McpProtocolSummary({
  server,
  connection
}: {
  server: McpServerSummary | undefined;
  connection: McpConnectionInfo | null;
}) {
  const { t } = useI18n();
  const serverIdentity = connection
    ? [connection.server_name, connection.server_version].filter(isPresentString).join(" ")
    : "";
  return (
    <section className="mcpProtocolSummary" aria-label={t("remote.protocolSummary")} aria-live="polite">
      <dl className="mcpProtocolFacts">
        <ProtocolFact label={t("remote.manifestVersion")} value={server ? `v${server.schema_version}` : t("remote.none")} />
        <ProtocolFact label={t("remote.protocolMode")} value={server ? protocolModeLabel(server.protocol_mode, t) : t("remote.none")} />
        {connection ? (
          <>
            <ProtocolFact label={t("remote.protocolEra")} value={protocolEraLabel(connection.protocol_era, t)} />
            <ProtocolFact label={t("remote.protocolRevision")} value={connection.protocol_revision} code />
            <ProtocolFact label={t("remote.sessionMode")} value={connection.sessionless ? t("remote.sessionless") : t("remote.sessionful")} />
            <ProtocolFact label={t("remote.fallback")} value={connection.fallback_used ? t("remote.fallbackUsed") : t("remote.fallbackUnused")} />
            <ProtocolFact label={t("remote.serverIdentity")} value={serverIdentity || t("remote.notReported")} />
          </>
        ) : null}
      </dl>
      {!connection ? <p className="mcpNegotiationState" role="status">{t("remote.notNegotiated")}</p> : (
        <div className="mcpCapabilityGroups">
          <CapabilityList label={t("remote.capabilities")} values={connection.capabilities} empty={t("remote.noneReported")} />
          <CapabilityList label={t("remote.unsupportedCapabilities")} values={connection.unsupported_capabilities} empty={t("remote.none")} warning />
        </div>
      )}
    </section>
  );
}

function ProtocolFact({ label, value, code = false }: { label: string; value: string; code?: boolean }) {
  return <div><dt>{label}</dt><dd>{code ? <code>{value}</code> : value}</dd></div>;
}

function CapabilityList({
  label,
  values,
  empty,
  warning = false
}: {
  label: string;
  values: string[];
  empty: string;
  warning?: boolean;
}) {
  return (
    <div className={`mcpCapabilityGroup${warning && values.length ? " warning" : ""}`}>
      <strong>{label}</strong>
      {values.length ? <ul>{values.map((value) => <li key={value}><code>{value}</code></li>)}</ul> : <span>{empty}</span>}
    </div>
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

type McpSchemaArgument = {
  name: string;
  label: string;
  required: boolean;
  kind: "string" | "number" | "integer" | "boolean";
  choices: string[];
};

export function schemaArgumentFields(
  entry: McpServerSummary | undefined,
  toolId: string
): McpSchemaArgument[] {
  const tool = entry?.tools.find((item) => item.tool_id === toolId);
  const schema = tool?.live?.input_schema ?? tool?.input_schema;
  if (!schema || schema.type !== "object" || !isRecord(schema.properties)) return [];
  const required = new Set(
    Array.isArray(schema.required)
      ? schema.required.filter((item): item is string => typeof item === "string")
      : []
  );
  return Object.entries(schema.properties).slice(0, 64).flatMap(([name, raw]) => {
    if (!name || !isRecord(raw)) return [];
    const kind = raw.type;
    if (kind !== "string" && kind !== "number" && kind !== "integer" && kind !== "boolean") return [];
    const choices = kind === "string" && Array.isArray(raw.enum)
      ? raw.enum.filter((item): item is string => typeof item === "string").slice(0, 256)
      : [];
    return [{
      name,
      label: typeof raw.title === "string" && raw.title ? raw.title : name,
      required: required.has(name),
      kind,
      choices
    }];
  });
}

function McpSchemaArgumentField({
  field,
  argumentsText,
  onChange
}: {
  field: McpSchemaArgument;
  argumentsText: string;
  onChange(value: string): void;
}) {
  const current = schemaArgumentValue(argumentsText, field.name);
  const label = `${field.label}${field.required ? " *" : ""}`;
  if (field.kind === "boolean") {
    return (
      <label className="toggle">
        <input
          type="checkbox"
          checked={current === true}
          onChange={(event) => onChange(updateSchemaArgument(argumentsText, field.name, event.currentTarget.checked))}
        />
        {label}
      </label>
    );
  }
  if (field.choices.length) {
    return (
      <label className="fieldStack">
        <span>{label}</span>
        <select
          value={typeof current === "string" ? current : ""}
          onChange={(event) => onChange(updateSchemaArgument(argumentsText, field.name, event.currentTarget.value, !event.currentTarget.value && !field.required))}
        >
          {!field.required ? <option value="">(omit)</option> : null}
          {field.required && !field.choices.includes(String(current ?? "")) ? <option value="">Select…</option> : null}
          {field.choices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
        </select>
      </label>
    );
  }
  return (
    <label className="fieldStack">
      <span>{label}</span>
      <input
        type={field.kind === "string" ? "text" : "number"}
        step={field.kind === "integer" ? 1 : "any"}
        value={typeof current === "string" || typeof current === "number" ? String(current) : ""}
        onChange={(event) => {
          const raw = event.currentTarget.value;
          if (!raw) {
            onChange(updateSchemaArgument(argumentsText, field.name, undefined, true));
            return;
          }
          const value = field.kind === "string" ? raw : Number(raw);
          if (field.kind !== "string" && !Number.isFinite(value)) return;
          onChange(updateSchemaArgument(argumentsText, field.name, value));
        }}
      />
    </label>
  );
}

function schemaArgumentValue(argumentsText: string, name: string): unknown {
  try {
    const parsed = JSON.parse(argumentsText || "{}");
    return isRecord(parsed) ? parsed[name] : undefined;
  } catch {
    return undefined;
  }
}

function updateSchemaArgument(
  argumentsText: string,
  name: string,
  value: unknown,
  omit = false
): string {
  let parsed: Record<string, unknown> = {};
  try {
    const current = JSON.parse(argumentsText || "{}");
    if (isRecord(current)) parsed = { ...current };
  } catch {
    // A schema field edit deliberately starts a fresh valid object after the
    // raw editor became invalid; it never submits until the user confirms.
  }
  if (omit) delete parsed[name];
  else parsed[name] = value;
  return JSON.stringify(parsed, null, 2);
}

function entryId(kind: RemoteKind, entry: RemoteSummary | undefined): string | null {
  if (!entry) return null;
  const id = kind === "jsonrpc"
    ? (entry as JsonRpcEndpointSummary).endpoint_id
    : (entry as McpServerSummary).server_id;
  return typeof id === "string" && id ? id : null;
}

function registryEntryVersion(kind: RemoteKind, entry: RemoteSummary | undefined): string {
  if (!entry) return `${kind}:missing`;
  const value = entry as Record<string, unknown>;
  return JSON.stringify([
    kind,
    entryId(kind, entry),
    value.schema_version ?? null,
    value.protocol_mode ?? null,
    value.transport ?? null,
    value.tools ?? value.methods ?? null,
    value.timeout_s ?? null,
    value.max_request_bytes ?? null,
    value.max_response_bytes ?? null,
    value.updated_at ?? null
  ]);
}

export function connectionFromResult(value: unknown): McpConnectionInfo | null {
  if (!isRecord(value) || !isRecord(value.connection)) return null;
  const connection = value.connection;
  if (!isProtocolMode(connection.protocol_mode)
    || !isProtocolEra(connection.protocol_era)
    || typeof connection.protocol_revision !== "string"
    || !connection.protocol_revision
    || typeof connection.sessionless !== "boolean"
    || typeof connection.fallback_used !== "boolean"
    || !isOptionalString(connection.server_name)
    || !isOptionalString(connection.server_version)
    || !isStringArray(connection.capabilities)
    || !isStringArray(connection.unsupported_capabilities)) {
    return null;
  }
  return connection as unknown as McpConnectionInfo;
}

function updateMcpConnection(value: unknown, update: (connection: McpConnectionInfo) => void): void {
  const connection = connectionFromResult(value);
  if (connection) update(connection);
}

function protocolModeLabel(mode: McpProtocolMode, t: ReturnType<typeof useI18n>["t"]): string {
  if (mode === "auto") return t("remote.protocolModeAuto");
  if (mode === "legacy") return t("remote.protocolModeLegacy");
  return mode;
}

function protocolEraLabel(era: McpProtocolEra, t: ReturnType<typeof useI18n>["t"]): string {
  return era === "modern" ? t("remote.protocolEraModern") : t("remote.protocolEraLegacy");
}

function isProtocolMode(value: unknown): value is McpProtocolMode {
  return value === "legacy" || value === "auto" || value === "2026-07-28";
}

function isProtocolEra(value: unknown): value is McpProtocolEra {
  return value === "legacy" || value === "modern";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isOptionalString(value: unknown): value is string | null | undefined {
  return value === undefined || value === null || typeof value === "string";
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isPresentString(value: string | null | undefined): value is string {
  return typeof value === "string" && value.length > 0;
}

function isString(value: string | null): value is string {
  return value !== null;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
