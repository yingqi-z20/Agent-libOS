import { Plus, Save, Settings, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { LLMProfileInput, LLMProfileSummary } from "../api/types";
import { useI18n } from "../i18n";

type LLMProfileSelectProps = {
  profiles: LLMProfileSummary[];
  value: string;
  label?: string;
  disabled?: boolean;
  initialManageOpen?: boolean;
  onChange(value: string): void;
  onCreate(profile: LLMProfileInput): Promise<boolean>;
  onUpdate(profileId: string, profile: LLMProfileInput): Promise<boolean>;
  onDelete(profileId: string): Promise<boolean>;
};

type ProfileFormState = {
  profile_id: string;
  model: string;
  base_url: string;
  api_key_env: string;
  api_mode: "" | "auto" | "responses" | "chat";
  temperature: string;
  max_tokens: string;
  context_window_tokens: string;
  timeout_s: string;
  max_retries: string;
  store: "" | "true" | "false";
  parallel_tool_calls: "" | "true" | "false";
  auto_wait_on_empty_tool_calls: "" | "true" | "false";
  allow_custom_base_url: boolean;
};

const emptyForm: ProfileFormState = {
  profile_id: "",
  model: "",
  base_url: "",
  api_key_env: "OPENAI_API_KEY",
  api_mode: "",
  temperature: "",
  max_tokens: "",
  context_window_tokens: "",
  timeout_s: "",
  max_retries: "",
  store: "",
  parallel_tool_calls: "",
  auto_wait_on_empty_tool_calls: "",
  allow_custom_base_url: false
};

export function LLMProfileSelect({
  profiles,
  value,
  label,
  disabled = false,
  initialManageOpen = false,
  onChange,
  onCreate,
  onUpdate,
  onDelete
}: LLMProfileSelectProps) {
  const { t } = useI18n();
  const [manageOpen, setManageOpen] = useState(initialManageOpen);
  const selected = profiles.find((profile) => profile.profile_id === value) ?? null;
  return (
    <div className="llmProfileSelect">
      <label>
        <span>{label ?? t("llmProfile.label")}</span>
        <div className="llmProfileSelectRow">
          <select value={selected ? value : ""} disabled={disabled} onChange={(event) => onChange(event.currentTarget.value)}>
            <option value="">{t("llmProfile.defaultOption")}</option>
            {profiles.map((profile) => (
              <option key={profile.profile_id} value={profile.profile_id}>
                {profile.profile_id}{profile.model ? ` · ${profile.model}` : ""}{profile.api_key_env_present ? "" : ` · ${t("llmProfile.envMissingShort")}`}
              </option>
            ))}
          </select>
          <button type="button" className="iconTextButton" onClick={() => setManageOpen(true)} title={t("llmProfile.manage")}>
            <Settings size={14} />{t("llmProfile.manage")}
          </button>
        </div>
      </label>
      {selected && !selected.api_key_env_present ? (
        <div className="llmProfileWarning">{t("llmProfile.envMissing", { env: selected.api_key_env })}</div>
      ) : null}
      {manageOpen ? (
        <LLMProfileManagerDialog
          profiles={profiles}
          selectedProfileId={selected?.profile_id ?? ""}
          onCreate={onCreate}
          onUpdate={onUpdate}
          onDelete={onDelete}
          onClose={() => setManageOpen(false)}
        />
      ) : null}
    </div>
  );
}

function LLMProfileManagerDialog({
  profiles,
  selectedProfileId,
  onCreate,
  onUpdate,
  onDelete,
  onClose
}: {
  profiles: LLMProfileSummary[];
  selectedProfileId: string;
  onCreate(profile: LLMProfileInput): Promise<boolean>;
  onUpdate(profileId: string, profile: LLMProfileInput): Promise<boolean>;
  onDelete(profileId: string): Promise<boolean>;
  onClose(): void;
}) {
  const { t } = useI18n();
  const initialProfile = profiles.find((profile) => profile.profile_id === selectedProfileId && profile.editable) ?? null;
  const [editingId, setEditingId] = useState(initialProfile?.profile_id ?? "");
  const [form, setForm] = useState<ProfileFormState>(() => initialProfile ? formFromProfile(initialProfile) : emptyForm);
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const editing = useMemo(() => profiles.find((profile) => profile.profile_id === editingId) ?? null, [editingId, profiles]);
  const canSave = Boolean(form.profile_id.trim() && form.model.trim() && form.api_key_env.trim() && !busy && (!editing || editing.editable));

  function edit(profile: LLMProfileSummary) {
    setEditingId(profile.profile_id);
    setForm(formFromProfile(profile));
    setLocalError(null);
  }

  function startNew() {
    setEditingId("");
    setForm(emptyForm);
    setLocalError(null);
  }

  async function save() {
    if (!canSave) return;
    setBusy(true);
    setLocalError(null);
    try {
      const input = formToInput(form);
      const ok = editingId ? await onUpdate(editingId, input) : await onCreate(input);
      if (ok) startNew();
      else setLocalError(t("llmProfile.saveFailed"));
    } finally {
      setBusy(false);
    }
  }

  async function remove(profileId: string) {
    if (busy) return;
    setBusy(true);
    setLocalError(null);
    try {
      const ok = await onDelete(profileId);
      if (ok && editingId === profileId) startNew();
      if (!ok) setLocalError(t("llmProfile.deleteFailed"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modalBackdrop" role="presentation">
      <div className="modal llmProfileModal" role="dialog" aria-modal="true" aria-labelledby="llm-profile-title">
        <h2 id="llm-profile-title">{t("llmProfile.manageTitle")}</h2>
        <div className="llmProfileManager">
          <section className="llmProfileList" aria-label={t("llmProfile.list")}>
            <button type="button" className={!editingId ? "active" : ""} onClick={startNew}><Plus size={14} />{t("llmProfile.add")}</button>
            {profiles.map((profile) => (
              <div className="llmProfileListItem" key={profile.profile_id}>
                <button type="button" className={editingId === profile.profile_id ? "active" : ""} onClick={() => edit(profile)}>
                  <span>{profile.profile_id}</span>
                  <small>{profile.source}{profile.is_default ? ` · ${t("llmProfile.defaultBadge")}` : ""}</small>
                </button>
                <button
                  type="button"
                  className="iconOnly danger"
                  disabled={!profile.editable || busy}
                  title={profile.editable ? t("llmProfile.delete") : t("llmProfile.readOnly")}
                  onClick={() => void remove(profile.profile_id)}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </section>
          <section className="llmProfileForm" aria-label={t("llmProfile.form")}>
            {editing && !editing.editable ? <div className="llmProfileWarning">{t("llmProfile.readOnly")}</div> : null}
            <label>
              {t("llmProfile.profileId")}
              <input value={form.profile_id} disabled={Boolean(editingId)} onChange={(event) => setForm({ ...form, profile_id: event.currentTarget.value })} />
            </label>
            <label>
              {t("llmProfile.model")}
              <input value={form.model} onChange={(event) => setForm({ ...form, model: event.currentTarget.value })} />
            </label>
            <label>
              {t("llmProfile.baseUrl")}
              <input value={form.base_url} placeholder="https://provider.example/v1" onChange={(event) => setForm({ ...form, base_url: event.currentTarget.value })} />
            </label>
            <label>
              {t("llmProfile.apiKeyEnv")}
              <input value={form.api_key_env} onChange={(event) => setForm({ ...form, api_key_env: event.currentTarget.value })} />
            </label>
            <label>
              {t("llmProfile.apiMode")}
              <select value={form.api_mode} onChange={(event) => setForm({ ...form, api_mode: event.currentTarget.value as ProfileFormState["api_mode"] })}>
                <option value="">{t("llmProfile.inherit")}</option>
                <option value="auto">auto</option>
                <option value="responses">responses</option>
                <option value="chat">chat</option>
              </select>
            </label>
            <div className="llmProfileFormGrid">
              <label>
                {t("llmProfile.temperature")}
                <input type="number" step="0.1" value={form.temperature} onChange={(event) => setForm({ ...form, temperature: event.currentTarget.value })} />
              </label>
              <label>
                {t("llmProfile.maxTokens")}
                <input type="number" min={1} step={1} value={form.max_tokens} onChange={(event) => setForm({ ...form, max_tokens: event.currentTarget.value })} />
              </label>
              <label>
                {t("llmProfile.contextWindowTokens")}
                <input type="number" min={1} step={1} value={form.context_window_tokens} onChange={(event) => setForm({ ...form, context_window_tokens: event.currentTarget.value })} />
              </label>
              <label>
                {t("llmProfile.timeout")}
                <input type="number" min={0.1} step="0.1" value={form.timeout_s} onChange={(event) => setForm({ ...form, timeout_s: event.currentTarget.value })} />
              </label>
              <label>
                {t("llmProfile.maxRetries")}
                <input type="number" min={0} step={1} value={form.max_retries} onChange={(event) => setForm({ ...form, max_retries: event.currentTarget.value })} />
              </label>
            </div>
            <div className="llmProfileFormGrid">
              <BooleanSelect label={t("llmProfile.store")} value={form.store} onChange={(store) => setForm({ ...form, store })} />
              <BooleanSelect label={t("llmProfile.parallelTools")} value={form.parallel_tool_calls} onChange={(parallel_tool_calls) => setForm({ ...form, parallel_tool_calls })} />
              <BooleanSelect label={t("llmProfile.autoWait")} value={form.auto_wait_on_empty_tool_calls} onChange={(auto_wait_on_empty_tool_calls) => setForm({ ...form, auto_wait_on_empty_tool_calls })} />
              <label className="toggle">
                <input type="checkbox" checked={form.allow_custom_base_url} onChange={(event) => setForm({ ...form, allow_custom_base_url: event.currentTarget.checked })} />
                {t("llmProfile.allowCustomBaseUrl")}
              </label>
            </div>
            {localError ? <div className="llmProfileWarning">{localError}</div> : null}
          </section>
        </div>
        <div className="modalActions">
          <button className="secondary" disabled={busy} onClick={onClose}>{t("confirm.cancel")}</button>
          <button className="primary" disabled={!canSave} onClick={() => void save()}><Save size={14} />{t("llmProfile.save")}</button>
        </div>
      </div>
    </div>
  );
}

function BooleanSelect({
  label,
  value,
  onChange
}: {
  label: string;
  value: "" | "true" | "false";
  onChange(value: "" | "true" | "false"): void;
}) {
  const { t } = useI18n();
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.currentTarget.value as "" | "true" | "false")}>
        <option value="">{t("llmProfile.inherit")}</option>
        <option value="true">{t("llmProfile.enabled")}</option>
        <option value="false">{t("llmProfile.disabled")}</option>
      </select>
    </label>
  );
}

function formFromProfile(profile: LLMProfileSummary): ProfileFormState {
  return {
    profile_id: profile.profile_id,
    model: profile.model ?? "",
    base_url: profile.base_url ?? "",
    api_key_env: profile.api_key_env,
    api_mode: profile.api_mode ?? "",
    temperature: stringifyNumber(profile.temperature),
    max_tokens: stringifyNumber(profile.max_tokens),
    context_window_tokens: stringifyNumber(profile.context_window_tokens),
    timeout_s: stringifyNumber(profile.timeout_s),
    max_retries: stringifyNumber(profile.max_retries),
    store: boolToForm(profile.store),
    parallel_tool_calls: boolToForm(profile.parallel_tool_calls),
    auto_wait_on_empty_tool_calls: boolToForm(profile.auto_wait_on_empty_tool_calls),
    allow_custom_base_url: profile.allow_custom_base_url
  };
}

function formToInput(form: ProfileFormState): LLMProfileInput {
  return {
    profile_id: form.profile_id.trim(),
    model: form.model.trim(),
    base_url: trimOrNull(form.base_url),
    api_key_env: form.api_key_env.trim(),
    api_mode: form.api_mode || null,
    temperature: numberOrNull(form.temperature),
    max_tokens: integerOrNull(form.max_tokens),
    context_window_tokens: integerOrNull(form.context_window_tokens),
    timeout_s: numberOrNull(form.timeout_s),
    max_retries: integerOrNull(form.max_retries),
    store: formBoolToValue(form.store),
    parallel_tool_calls: formBoolToValue(form.parallel_tool_calls),
    auto_wait_on_empty_tool_calls: formBoolToValue(form.auto_wait_on_empty_tool_calls),
    allow_custom_base_url: form.allow_custom_base_url
  };
}

function stringifyNumber(value: number | null): string {
  return value === null ? "" : String(value);
}

function boolToForm(value: boolean | null): "" | "true" | "false" {
  if (value === true) return "true";
  if (value === false) return "false";
  return "";
}

function formBoolToValue(value: "" | "true" | "false"): boolean | null {
  if (value === "true") return true;
  if (value === "false") return false;
  return null;
}

function trimOrNull(value: string): string | null {
  const selected = value.trim();
  return selected || null;
}

function numberOrNull(value: string): number | null {
  const selected = value.trim();
  return selected ? Number(selected) : null;
}

function integerOrNull(value: string): number | null {
  const selected = value.trim();
  return selected ? Number.parseInt(selected, 10) : null;
}
