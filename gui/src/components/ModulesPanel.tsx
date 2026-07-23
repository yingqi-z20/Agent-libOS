import { Box, Eye } from "lucide-react";
import { useEffect, useState } from "react";
import type { LibOSClient } from "../api/client";
import type { ModuleSummary } from "../api/types";
import { useI18n } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";

export function ModulesPanel({ modules, client }: { modules: ModuleSummary[]; client: LibOSClient }) {
  const { t } = useI18n();
  const [selectedId, setSelectedId] = useState(modules[0]?.module_id ?? "");
  const [details, setDetails] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    if (!modules.some((module) => module.module_id === selectedId)) setSelectedId(modules[0]?.module_id ?? "");
  }, [modules, selectedId]);

  async function inspect() {
    if (!selectedId) return;
    setLoading(true);
    setLocalError(null);
    try {
      setDetails(await client.inspectModule(selectedId));
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="adminPanel modulesPanel" aria-busy={loading || undefined}>
      <header className="adminPanelHeader">
        <div>
          <h3><Box size={16} />{t("modules.title")}</h3>
          <p>{t("modules.description")}</p>
        </div>
      </header>
      <label className="fieldStack">
        <span>{t("modules.selected")}</span>
        <select value={selectedId} disabled={!modules.length} onChange={(event) => setSelectedId(event.currentTarget.value)}>
          {!modules.length ? <option value="">{t("modules.empty")}</option> : null}
          {modules.map((module) => (
            <option value={module.module_id} key={module.module_id}>
              {module.module_id}{module.version ? ` · ${module.version}` : ""}
            </option>
          ))}
        </select>
      </label>
      <button disabled={!selectedId || loading} onClick={() => void inspect()}><Eye size={14} />{t("modules.inspect")}</button>
      {localError ? <div className="inlineError" role="alert">{localError}</div> : null}
      {details !== null ? <CollapsibleJson value={details} label={t("modules.details")} defaultExpanded /> : null}
    </section>
  );
}
