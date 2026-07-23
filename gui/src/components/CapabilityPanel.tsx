import { RefreshCw, Search, Shield, ShieldMinus, ShieldPlus, Share2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { LibOSClient } from "../api/client";
import type { CapabilitySummary, RuntimeProcess } from "../api/types";
import type { ConfirmationRequest } from "../adminTypes";
import { useI18n } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";

export const capabilityRights = [
  "read",
  "write",
  "execute",
  "link",
  "diff",
  "materialize",
  "delete",
  "grant",
  "revoke",
  "approve",
  "admin"
] as const;

export function CapabilityPanel({
  process,
  client,
  confirmAction,
  reloadKey
}: {
  process: RuntimeProcess;
  client: LibOSClient;
  confirmAction(request: ConfirmationRequest): void;
  reloadKey: string;
}) {
  const { t } = useI18n();
  const [capabilities, setCapabilities] = useState<CapabilitySummary[]>([]);
  const [selectedId, setSelectedId] = useState(process.capabilities[0] ?? "");
  const [resource, setResource] = useState("");
  const [rights, setRights] = useState<string[]>(["read"]);
  const [child, setChild] = useState("");
  const [explainRight, setExplainRight] = useState("read");
  const [result, setResult] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const requestSequence = useRef(0);
  const activePid = useRef(process.pid);
  activePid.current = process.pid;

  const selected = useMemo(
    () => capabilities.find((capability) => capability.cap_id === selectedId) ?? null,
    [capabilities, selectedId]
  );

  useEffect(() => {
    requestSequence.current += 1;
    setCapabilities([]);
    setResult(null);
    void load();
    return () => { requestSequence.current += 1; };
  }, [process.pid, reloadKey]);

  async function load() {
    const pid = process.pid;
    const sequence = ++requestSequence.current;
    setLoading(true);
    setLocalError(null);
    try {
      const loaded = await client.listCapabilities(pid);
      if (sequence !== requestSequence.current || activePid.current !== pid) return;
      setCapabilities(loaded);
      setSelectedId((current) => loaded.some((item) => item.cap_id === current)
        ? current
        : loaded[0]?.cap_id ?? "");
    } catch (error) {
      if (sequence === requestSequence.current && activePid.current === pid) {
        setLocalError(describe(error));
      }
    } finally {
      if (sequence === requestSequence.current && activePid.current === pid) setLoading(false);
    }
  }

  async function inspect() {
    if (!selectedId) return;
    const pid = process.pid;
    const sequence = ++requestSequence.current;
    setLoading(true);
    setLocalError(null);
    try {
      const nextResult = await client.inspectCapability(selectedId);
      if (sequence === requestSequence.current && activePid.current === pid) setResult(nextResult);
    } catch (error) {
      if (sequence === requestSequence.current && activePid.current === pid) {
        setLocalError(describe(error));
      }
    } finally {
      if (sequence === requestSequence.current && activePid.current === pid) setLoading(false);
    }
  }

  async function explain() {
    if (!resource.trim()) return;
    const pid = process.pid;
    const sequence = ++requestSequence.current;
    setLoading(true);
    setLocalError(null);
    try {
      const nextResult = await client.explainCapability(pid, resource.trim(), explainRight);
      if (sequence === requestSequence.current && activePid.current === pid) setResult(nextResult);
    } catch (error) {
      if (sequence === requestSequence.current && activePid.current === pid) {
        setLocalError(describe(error));
      }
    } finally {
      if (sequence === requestSequence.current && activePid.current === pid) setLoading(false);
    }
  }

  function confirmGrant() {
    const selectedResource = resource.trim();
    if (!selectedResource || rights.length === 0) return;
    confirmAction({
      title: t("capability.grantTitle"),
      message: t("capability.grantMessage"),
      details: { subject: process.pid, resource: selectedResource, rights, mode: "host-admin" },
      action: async () => { await client.grantCapability({ subject: process.pid, resource: selectedResource, rights }, true); }
    });
  }

  function confirmDelegate() {
    const selectedResource = resource.trim();
    const selectedChild = child.trim();
    if (!selectedResource || !selectedChild || rights.length === 0) return;
    confirmAction({
      title: t("capability.delegateTitle"),
      message: t("capability.delegateMessage"),
      details: { parent: process.pid, child: selectedChild, resource: selectedResource, rights, mode: "host-admin" },
      action: async () => {
        await client.delegateCapability({ parent: process.pid, child: selectedChild, resource: selectedResource, rights }, true);
      }
    });
  }

  function confirmRevoke() {
    if (!selectedId) return;
    const capabilityId = selectedId;
    confirmAction({
      title: t("capability.revokeTitle"),
      message: t("capability.revokeMessage"),
      details: {
        capability_id: capabilityId,
        subject: selected?.subject ?? process.pid,
        resource: selected?.resource ?? null,
        rights: selected?.rights ?? [],
        mode: "host-admin"
      },
      action: async () => { await client.revokeCapability(capabilityId, "Revoked from GUI", true); }
    });
  }

  return (
    <section className="adminPanel capabilityPanel" aria-busy={loading || undefined}>
      <header className="adminPanelHeader">
        <div>
          <h3><Shield size={16} />{t("capability.title")}</h3>
          <p>{t("capability.adminNotice")}</p>
        </div>
        <button className="iconOnly" disabled={loading} aria-label={t("capability.refresh")} onClick={() => void load()}>
          <RefreshCw className={loading ? "spin" : ""} size={14} />
        </button>
      </header>

      <label className="fieldStack">
        <span>{t("capability.selected")}</span>
        <select value={selectedId} disabled={loading || capabilities.length === 0} onChange={(event) => setSelectedId(event.currentTarget.value)}>
          {capabilities.length === 0 ? <option value="">{t("capability.empty")}</option> : null}
          {capabilities.map((capability) => (
            <option key={capabilityIdentity(capability)} value={capabilityIdentity(capability)}>
              {capability.resource} · {capability.rights.join(", ")} · {capability.status ?? "active"}
            </option>
          ))}
        </select>
      </label>
      <div className="adminActions">
        <button disabled={!selectedId || loading} onClick={() => void inspect()}><Search size={14} />{t("capability.inspect")}</button>
        <button className="danger" disabled={!selectedId || loading} onClick={confirmRevoke}><ShieldMinus size={14} />{t("capability.revoke")}</button>
      </div>

      <div className="adminFormGrid">
        <label className="fieldStack spanAll">
          <span>{t("capability.resource")}</span>
          <input value={resource} placeholder={t("capability.resourcePlaceholder")} onChange={(event) => setResource(event.currentTarget.value)} />
        </label>
        <RightsPicker value={rights} onChange={setRights} />
        <label className="fieldStack">
          <span>{t("capability.child")}</span>
          <input value={child} placeholder={t("capability.childPlaceholder")} onChange={(event) => setChild(event.currentTarget.value)} />
        </label>
        <label className="fieldStack">
          <span>{t("capability.explainRight")}</span>
          <select value={explainRight} onChange={(event) => setExplainRight(event.currentTarget.value)}>
            {capabilityRights.map((right) => <option value={right} key={right}>{right}</option>)}
          </select>
        </label>
      </div>
      <div className="adminActions">
        <button className="warning" disabled={!resource.trim() || !rights.length} onClick={confirmGrant}><ShieldPlus size={14} />{t("capability.grant")}</button>
        <button className="warning" disabled={!resource.trim() || !child.trim() || !rights.length} onClick={confirmDelegate}><Share2 size={14} />{t("capability.delegate")}</button>
        <button disabled={!resource.trim() || loading} onClick={() => void explain()}><Search size={14} />{t("capability.explain")}</button>
      </div>

      {localError ? <div className="inlineError" role="alert">{localError}</div> : null}
      {result !== null ? <CollapsibleJson value={result} label={t("capability.result")} defaultExpanded /> : null}
    </section>
  );
}

export function capabilityIdentity(capability: CapabilitySummary): string {
  return capability.cap_id;
}

function RightsPicker({ value, onChange }: { value: string[]; onChange(value: string[]): void }) {
  const { t } = useI18n();
  return (
    <fieldset className="rightsPicker spanAll">
      <legend>{t("capability.rights")}</legend>
      {capabilityRights.map((right) => (
        <label key={right}>
          <input
            type="checkbox"
            checked={value.includes(right)}
            onChange={(event) => onChange(event.currentTarget.checked
              ? [...value, right]
              : value.filter((item) => item !== right))}
          />
          {right}
        </label>
      ))}
    </fieldset>
  );
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
