import { Plus, Save, Settings, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { LLMProfileInput, LLMProfileSummary } from "../api/types";
import { useI18n } from "../i18n";
import { Modal } from "./Modal";

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
  reasoning_effort: string;
  verbosity: "" | "low" | "medium" | "high";
  safety_identifier_env: string;
  prompt_cache_retention: "" | "in-memory" | "24h";
  responses_previous_response_id: "" | "true" | "false";
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
  reasoning_effort: "",
  verbosity: "",
  safety_identifier_env: "",
  prompt_cache_retention: "",
  responses_previous_response_id: "",
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
          <button type="button" className="iconTextButton" disabled={disabled} onClick={() => setManageOpen(true)} title={t("llmProfile.manage")}>
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
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const editing = useMemo(() => profiles.find((profile) => profile.profile_id === editingId) ?? null, [editingId, profiles]);
  const canSave = Boolean(form.profile_id.trim() && form.model.trim() && form.api_key_env.trim() && !busy && (!editing || editing.editable));

  function edit(profile: LLMProfileSummary) {
    setEditingId(profile.profile_id);
    setForm(formFromProfile(profile));
    setLocalError(null);
    setPendingDeleteId(null);
  }

  function startNew() {
    setEditingId("");
    setForm(emptyForm);
    setLocalError(null);
    setPendingDeleteId(null);
  }

  async function save() {
    if (!canSave) return;
    let input: LLMProfileInput;
    try {
      input = formToInput(form);
    } catch {
      setLocalError(t("llmProfile.invalidNumericInput"));
      return;
    }
    setBusy(true);
    setLocalError(null);
    try {
      const ok = editingId ? await onUpdate(editingId, input) : await onCreate(input);
      if (ok) startNew();
      else setLocalError(t("llmProfile.saveFailed"));
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : t("llmProfile.saveFailed"));
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
      if (ok) setPendingDeleteId(null);
      if (ok && editingId === profileId) startNew();
      if (!ok) setLocalError(t("llmProfile.deleteFailed"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      className="llmProfileModal"
      title={t("llmProfile.manageTitle")}
      busy={busy}
      onClose={onClose}
      actions={
        <>
          <button className="secondary" disabled={busy} onClick={onClose}>{t("confirm.cancel")}</button>
          <button className="primary" disabled={!canSave} onClick={() => void save()}><Save size={14} />{t("llmProfile.save")}</button>
        </>
      }
    >
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
                  aria-label={profile.editable ? t("llmProfile.delete") : t("llmProfile.readOnly")}
                  title={profile.editable ? t("llmProfile.delete") : t("llmProfile.readOnly")}
                  onClick={() => setPendingDeleteId(profile.profile_id)}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </section>
          <section className="llmProfileForm" aria-label={t("llmProfile.form")}>
            {pendingDeleteId ? (
              <div className="inlineConfirm" role="group" aria-label={t("llmProfile.deleteConfirmTitle")}>
                <strong>{t("llmProfile.deleteConfirmTitle")}</strong>
                <span>{t("llmProfile.deleteConfirmMessage", { profile: pendingDeleteId })}</span>
                <div className="adminActions">
                  <button className="secondary" disabled={busy} onClick={() => setPendingDeleteId(null)}>{t("confirm.cancel")}</button>
                  <button className="danger" disabled={busy} onClick={() => void remove(pendingDeleteId)}>{t("llmProfile.delete")}</button>
                </div>
              </div>
            ) : null}
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
                {t("llmProfile.reasoningEffort")}
                <input value={form.reasoning_effort} onChange={(event) => setForm({ ...form, reasoning_effort: event.currentTarget.value })} />
              </label>
              <label>
                {t("llmProfile.verbosity")}
                <select value={form.verbosity} onChange={(event) => setForm({ ...form, verbosity: event.currentTarget.value as ProfileFormState["verbosity"] })}>
                  <option value="">{t("llmProfile.inherit")}</option>
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                </select>
              </label>
              <label>
                {t("llmProfile.safetyIdentifierEnv")}
                <input value={form.safety_identifier_env} placeholder="OPENAI_SAFETY_IDENTIFIER" onChange={(event) => setForm({ ...form, safety_identifier_env: event.currentTarget.value })} />
              </label>
              <label>
                {t("llmProfile.promptCacheRetention")}
                <select value={form.prompt_cache_retention} onChange={(event) => setForm({ ...form, prompt_cache_retention: event.currentTarget.value as ProfileFormState["prompt_cache_retention"] })}>
                  <option value="">{t("llmProfile.inherit")}</option>
                  <option value="in-memory">in-memory</option>
                  <option value="24h">24h</option>
                </select>
              </label>
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
              <BooleanSelect label={t("llmProfile.previousResponseId")} value={form.responses_previous_response_id} onChange={(responses_previous_response_id) => setForm({ ...form, responses_previous_response_id })} />
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
    </Modal>
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
    reasoning_effort: profile.reasoning_effort ?? "",
    verbosity: profile.verbosity ?? "",
    safety_identifier_env: profile.safety_identifier_env ?? "",
    prompt_cache_retention: profile.prompt_cache_retention ?? "",
    responses_previous_response_id: boolToForm(profile.responses_previous_response_id),
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
    temperature: parseProfileNumber(form.temperature, { minimum: 0 }),
    max_tokens: parseProfileNumber(form.max_tokens, { integer: true, minimum: 0, exclusiveMinimum: true }),
    context_window_tokens: parseProfileNumber(form.context_window_tokens, { integer: true, minimum: 0, exclusiveMinimum: true }),
    timeout_s: parseProfileNumber(form.timeout_s, { minimum: 0, exclusiveMinimum: true }),
    max_retries: parseProfileNumber(form.max_retries, { integer: true, minimum: 0 }),
    reasoning_effort: trimOrNull(form.reasoning_effort),
    verbosity: form.verbosity || null,
    safety_identifier_env: trimOrNull(form.safety_identifier_env),
    prompt_cache_retention: form.prompt_cache_retention || null,
    responses_previous_response_id: formBoolToValue(form.responses_previous_response_id),
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

export function parseProfileNumber(
  value: string,
  options: { integer?: boolean; minimum?: number; exclusiveMinimum?: boolean } = {}
): number | null {
  const selected = value.trim();
  if (!selected) return null;
  const parsed = Number(selected);
  if (!Number.isFinite(parsed)) throw new Error("Profile numeric value must be finite.");
  if (options.integer && !Number.isInteger(parsed)) throw new Error("Profile numeric value must be an integer.");
  if (options.minimum !== undefined) {
    const outsideRange = options.exclusiveMinimum
      ? parsed <= options.minimum
      : parsed < options.minimum;
    if (outsideRange) throw new Error("Profile numeric value is outside the accepted range.");
  }
  return parsed;
}
