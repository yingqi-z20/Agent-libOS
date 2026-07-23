import { Download, Eye, Play, RefreshCw, Search, Unplug } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { LibOSClient } from "../api/client";
import type { RuntimeProcess, SkillSummary } from "../api/types";
import type { ConfirmationRequest } from "../adminTypes";
import { useI18n } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";

export function SkillsPanel({
  process,
  skills,
  tools,
  client,
  confirmAction
}: {
  process: RuntimeProcess;
  skills: SkillSummary[];
  tools: unknown[];
  client: LibOSClient;
  confirmAction(request: ConfirmationRequest): void;
}) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(skills[0]?.skill_id ?? "");
  const [workspacePath, setWorkspacePath] = useState("");
  const [replace, setReplace] = useState(false);
  const [processAuthority, setProcessAuthority] = useState(true);
  const [details, setDetails] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const filtered = useMemo(() => filterSkills(skills, query), [query, skills]);
  const loadedIds = new Set(Object.keys(process.loaded_skills));

  useEffect(() => {
    if (!filtered.some((skill) => skill.skill_id === selectedId)) setSelectedId(filtered[0]?.skill_id ?? "");
  }, [filtered, selectedId]);

  async function inspect() {
    if (!selectedId) return;
    setLoading(true);
    setLocalError(null);
    try {
      setDetails(await client.inspectSkill(selectedId));
    } catch (error) {
      setLocalError(describe(error));
    } finally {
      setLoading(false);
    }
  }

  function confirmRegister() {
    const path = workspacePath.trim();
    if (!path) return;
    confirmAction({
      title: t("skills.registerTitle"),
      message: t("skills.registerMessage"),
      details: { actor: process.pid, workspace_path: path, replace, authority: "process" },
      action: async () => { await client.registerSkill(path, process.pid, true, replace); }
    });
  }

  function confirmLifecycle(action: "activate" | "unload") {
    if (!selectedId) return;
    const skillId = selectedId;
    const actor = processAuthority ? process.pid : undefined;
    confirmAction({
      title: t(action === "activate" ? "skills.activateTitle" : "skills.unloadTitle"),
      message: t(action === "activate" ? "skills.activateMessage" : "skills.unloadMessage"),
      details: {
        pid: process.pid,
        skill_id: skillId,
        authority: actor ? "process" : "host-admin"
      },
      action: async () => {
        if (action === "activate") await client.activateSkill(skillId, process.pid, true, actor);
        else await client.unloadSkill(skillId, process.pid, true, actor);
      }
    });
  }

  return (
    <section className="adminPanel skillsPanel" aria-busy={loading || undefined}>
      <header className="adminPanelHeader">
        <div>
          <h3><Download size={16} />{t("skills.title")}</h3>
          <p>{t("skills.description", { skills: skills.length, tools: tools.length })}</p>
        </div>
      </header>

      <label className="searchField">
        <Search size={14} aria-hidden="true" />
        <span className="srOnly">{t("skills.search")}</span>
        <input type="search" value={query} placeholder={t("skills.searchPlaceholder")} onChange={(event) => setQuery(event.currentTarget.value)} />
      </label>
      <label className="fieldStack">
        <span>{t("skills.selected")}</span>
        <select value={selectedId} disabled={filtered.length === 0} onChange={(event) => setSelectedId(event.currentTarget.value)}>
          {filtered.length === 0 ? <option value="">{t("skills.empty")}</option> : null}
          {filtered.map((skill) => (
            <option key={skill.skill_id} value={skill.skill_id}>
              {skill.skill_id}{loadedIds.has(skill.skill_id) ? ` · ${t("skills.loaded")}` : ""}
            </option>
          ))}
        </select>
      </label>
      <label className="toggle">
        <input type="checkbox" checked={processAuthority} onChange={(event) => setProcessAuthority(event.currentTarget.checked)} />
        {t("skills.processAuthority")}
      </label>
      <div className="adminActions">
        <button disabled={!selectedId || loading} onClick={() => void inspect()}><Eye size={14} />{t("skills.inspect")}</button>
        <button className="warning" disabled={!selectedId || loadedIds.has(selectedId)} onClick={() => confirmLifecycle("activate")}><Play size={14} />{t("skills.activate")}</button>
        <button className="danger" disabled={!selectedId || !loadedIds.has(selectedId)} onClick={() => confirmLifecycle("unload")}><Unplug size={14} />{t("skills.unload")}</button>
      </div>

      <div className="inlineForm skillRegisterForm">
        <label className="growField">
          <span>{t("skills.workspacePath")}</span>
          <input value={workspacePath} placeholder={t("skills.workspacePathPlaceholder")} onChange={(event) => setWorkspacePath(event.currentTarget.value)} />
        </label>
        <label className="toggle">
          <input type="checkbox" checked={replace} onChange={(event) => setReplace(event.currentTarget.checked)} />
          {t("skills.replace")}
        </label>
        <button className="warning" disabled={!workspacePath.trim()} onClick={confirmRegister}><RefreshCw size={14} />{t("skills.register")}</button>
      </div>

      {localError ? <div className="inlineError" role="alert">{localError}</div> : null}
      {details !== null ? <CollapsibleJson value={details} label={t("skills.details")} defaultExpanded /> : null}
      <CollapsibleJson value={{ loaded_skills: process.loaded_skills, process_tools: process.tool_table }} label={t("skills.processState")} />
    </section>
  );
}

export function filterSkills(skills: SkillSummary[], query: string): SkillSummary[] {
  const selected = query.trim().toLocaleLowerCase();
  if (!selected) return skills;
  return skills.filter((skill) => [skill.skill_id, skill.name, skill.description, skill.source]
    .filter((value): value is string => typeof value === "string")
    .some((value) => value.toLocaleLowerCase().includes(selected)));
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
