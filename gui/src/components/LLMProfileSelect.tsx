import { Plus, Save, Settings, Trash2 } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { LLMProfileInput, LLMProfileSummary, LLMProviderTools } from "../api/types";
import { useI18n, type TranslationKey } from "../i18n";
import { Modal } from "./Modal";

export type LLMProfileSelectProps = {
  profiles: LLMProfileSummary[];
  value: string;
  label?: string;
  disabled?: boolean;
  initialManageOpen?: boolean;
  onManage?: () => void;
  onChange(value: string): void;
  onCreate(profile: LLMProfileInput): Promise<boolean>;
  onUpdate(profileId: string, profile: LLMProfileInput): Promise<boolean>;
  onDelete(profileId: string): Promise<boolean>;
};

export type LLMProfileManagerDialogProps = {
  profiles: LLMProfileSummary[];
  selectedProfileId: string;
  onCreate(profile: LLMProfileInput): Promise<boolean>;
  onUpdate(profileId: string, profile: LLMProfileInput): Promise<boolean>;
  onDelete(profileId: string): Promise<boolean>;
  onClose(): void;
};

type ProfileFormState = {
  profile_id: string;
  model: string;
  base_url: string;
  api_key_env: string;
  api_mode: "" | "auto" | "responses" | "chat";
  provider_tools_provider: "" | "openai" | "aliyun";
  provider_tools_web_search: boolean;
  provider_tools_web_extractor: boolean;
  provider_tools_code_interpreter: boolean;
  provider_tools_file_ids: string;
  temperature: string;
  max_tokens: string;
  context_window_tokens: string;
  timeout_s: string;
  max_retries: string;
  reasoning_effort: string;
  verbosity: "" | "low" | "medium" | "high";
  safety_identifier_env: string;
  prompt_cache_retention: "" | "in_memory" | "24h";
  responses_previous_response_id: "" | "true" | "false";
  store: "" | "true" | "false";
  parallel_tool_calls: "" | "true" | "false";
  auto_wait_on_empty_tool_calls: "" | "true" | "false";
  fallback_json_actions: "" | "true" | "false";
  allow_custom_base_url: boolean;
};

const emptyForm: ProfileFormState = {
  profile_id: "",
  model: "",
  base_url: "",
  api_key_env: "OPENAI_API_KEY",
  api_mode: "",
  provider_tools_provider: "",
  provider_tools_web_search: false,
  provider_tools_web_extractor: false,
  provider_tools_code_interpreter: false,
  provider_tools_file_ids: "",
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
  fallback_json_actions: "",
  allow_custom_base_url: false
};

export function LLMProfileSelect({
  profiles,
  value,
  label,
  disabled = false,
  initialManageOpen = false,
  onManage,
  onChange,
  onCreate,
  onUpdate,
  onDelete
}: LLMProfileSelectProps) {
  const { t } = useI18n();
  const labelId = useId();
  const [manageOpen, setManageOpen] = useState(initialManageOpen);
  const selected = profiles.find((profile) => profile.profile_id === value) ?? null;
  return (
    <div className="llmProfileSelect">
      <div className="llmProfileSelectField">
        <span id={labelId}>{label ?? t("llmProfile.label")}</span>
        <div className="llmProfileSelectRow">
          <select aria-labelledby={labelId} value={selected ? value : ""} disabled={disabled} onChange={(event) => onChange(event.currentTarget.value)}>
            <option value="">{t("llmProfile.defaultOption")}</option>
            {profiles.map((profile) => (
              <option key={profile.profile_id} value={profile.profile_id}>
                {profile.profile_id}{profile.model ? ` · ${profile.model}` : ""}{profile.api_key_env_present ? "" : ` · ${t("llmProfile.envMissingShort")}`}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="iconTextButton"
            disabled={disabled}
            onClick={() => {
              if (onManage) onManage();
              else setManageOpen(true);
            }}
            title={t("llmProfile.manage")}
          >
            <Settings size={14} />{t("llmProfile.manage")}
          </button>
        </div>
      </div>
      {selected && !selected.api_key_env_present ? (
        <div className="llmProfileWarning">{t("llmProfile.envMissing", { env: selected.api_key_env })}</div>
      ) : null}
      {!onManage && manageOpen ? (
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

export function LLMProfileManagerDialog({
  profiles,
  selectedProfileId,
  onCreate,
  onUpdate,
  onDelete,
  onClose
}: LLMProfileManagerDialogProps) {
  const { t } = useI18n();
  const initialProfile = profiles.find((profile) => profile.profile_id === selectedProfileId && profile.editable) ?? null;
  const [editingId, setEditingId] = useState(initialProfile?.profile_id ?? "");
  const [form, setForm] = useState<ProfileFormState>(() => initialProfile ? formFromProfile(initialProfile) : emptyForm);
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const deleteConfirmTitleId = useId();
  const deleteConfirmDescriptionId = useId();
  const deleteCancelRef = useRef<HTMLButtonElement>(null);
  const deleteButtonRefs = useRef(new Map<string, HTMLButtonElement>());
  const editing = useMemo(() => profiles.find((profile) => profile.profile_id === editingId) ?? null, [editingId, profiles]);
  const providerToolsError = validateProviderTools(form);
  const canSave = Boolean(form.profile_id.trim() && form.model.trim() && form.api_key_env.trim() && !providerToolsError && !busy && (!editing || editing.editable));

  useEffect(() => {
    if (pendingDeleteId) deleteCancelRef.current?.focus();
  }, [pendingDeleteId]);

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

  function cancelDelete() {
    const profileId = pendingDeleteId;
    setPendingDeleteId(null);
    if (!profileId) return;
    globalThis.setTimeout(() => {
      const trigger = deleteButtonRefs.current.get(profileId);
      if (trigger?.isConnected && !trigger.disabled) trigger.focus();
    }, 0);
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
            <button type="button" className={!editingId ? "active" : ""} aria-pressed={!editingId} disabled={busy} onClick={startNew}><Plus size={14} />{t("llmProfile.add")}</button>
            {profiles.map((profile) => (
              <div className="llmProfileListItem" key={profile.profile_id}>
                <button type="button" className={editingId === profile.profile_id ? "active" : ""} aria-pressed={editingId === profile.profile_id} disabled={busy} onClick={() => edit(profile)}>
                  <span>{profile.profile_id}</span>
                  <small>{profile.source}{profile.is_default ? ` · ${t("llmProfile.defaultBadge")}` : ""}</small>
                </button>
                <button
                  ref={(node) => {
                    if (node) deleteButtonRefs.current.set(profile.profile_id, node);
                    else deleteButtonRefs.current.delete(profile.profile_id);
                  }}
                  type="button"
                  className="iconOnly danger"
                  disabled={!profile.editable || busy}
                  aria-label={`${profile.editable ? t("llmProfile.delete") : t("llmProfile.readOnly")}: ${profile.profile_id}`}
                  title={`${profile.editable ? t("llmProfile.delete") : t("llmProfile.readOnly")}: ${profile.profile_id}`}
                  onClick={() => setPendingDeleteId(profile.profile_id)}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </section>
          <section className="llmProfileForm" aria-label={t("llmProfile.form")}>
            {pendingDeleteId ? (
              <div
                className="inlineConfirm"
                role="alertdialog"
                aria-labelledby={deleteConfirmTitleId}
                aria-describedby={deleteConfirmDescriptionId}
                onKeyDown={(event) => {
                  if (event.key !== "Escape" || busy) return;
                  event.preventDefault();
                  event.stopPropagation();
                  cancelDelete();
                }}
              >
                <strong id={deleteConfirmTitleId}>{t("llmProfile.deleteConfirmTitle")}</strong>
                <span id={deleteConfirmDescriptionId}>{t("llmProfile.deleteConfirmMessage", { profile: pendingDeleteId })}</span>
                <div className="adminActions">
                  <button ref={deleteCancelRef} className="secondary" disabled={busy} onClick={cancelDelete}>{t("confirm.cancel")}</button>
                  <button className="danger" disabled={busy} onClick={() => void remove(pendingDeleteId)}>{t("llmProfile.delete")}</button>
                </div>
              </div>
            ) : null}
            {editing && !editing.editable ? <div className="llmProfileWarning">{t("llmProfile.readOnly")}</div> : null}
            <fieldset className="llmProfileFormFields" disabled={busy || Boolean(editing && !editing.editable)}>
              <legend className="srOnly">{t("llmProfile.form")}</legend>
              <label>
                {t("llmProfile.profileId")}
                <input required value={form.profile_id} disabled={Boolean(editingId)} onChange={(event) => setForm({ ...form, profile_id: event.currentTarget.value })} />
              </label>
              <label>
                {t("llmProfile.model")}
                <input required value={form.model} onChange={(event) => setForm({ ...form, model: event.currentTarget.value })} />
              </label>
              <label>
                {t("llmProfile.baseUrl")}
                <input value={form.base_url} placeholder="https://provider.example/v1" onChange={(event) => setForm({ ...form, base_url: event.currentTarget.value })} />
              </label>
              <label>
                {t("llmProfile.apiKeyEnv")}
                <input required value={form.api_key_env} onChange={(event) => setForm({ ...form, api_key_env: event.currentTarget.value })} />
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
              <fieldset className="llmProviderTools">
                <legend>{t("llmProfile.providerTools")}</legend>
                <label>
                  {t("llmProfile.providerToolsProvider")}
                  <select value={form.provider_tools_provider} onChange={(event) => {
                    const provider = event.currentTarget.value as ProfileFormState["provider_tools_provider"];
                    setForm({ ...form, provider_tools_provider: provider,
                      api_mode: provider && !form.api_mode ? "auto" : form.api_mode,
                      provider_tools_web_extractor: provider === "aliyun" && form.provider_tools_web_extractor,
                      provider_tools_file_ids: provider === "openai" ? form.provider_tools_file_ids : ""
                    });
                  }}>
                    <option value="">{t("llmProfile.disabled")}</option>
                    <option value="openai">OpenAI</option>
                    <option value="aliyun">{t("llmProfile.aliyun")}</option>
                  </select>
                </label>
                {form.provider_tools_provider ? <>
                  <label className="toggle">
                    <input type="checkbox" checked={form.provider_tools_web_search} onChange={(event) => setForm({ ...form, provider_tools_web_search: event.currentTarget.checked })} />
                    {t("llmProfile.webSearch")}
                  </label>
                  {form.provider_tools_provider === "aliyun" ? <label className="toggle">
                    <input type="checkbox" checked={form.provider_tools_web_extractor} onChange={(event) => setForm({ ...form, provider_tools_web_extractor: event.currentTarget.checked })} />
                    {t("llmProfile.webExtractor")}
                  </label> : null}
                  <label className="toggle">
                    <input type="checkbox" checked={form.provider_tools_code_interpreter} onChange={(event) => setForm({ ...form, provider_tools_code_interpreter: event.currentTarget.checked,
                      provider_tools_file_ids: event.currentTarget.checked ? form.provider_tools_file_ids : "" })} />
                    {t("llmProfile.codeInterpreter")}
                  </label>
                  {form.provider_tools_provider === "openai" && form.provider_tools_code_interpreter ? <label>
                    {t("llmProfile.fileIds")}
                    <textarea rows={3} value={form.provider_tools_file_ids} placeholder="file-..." onChange={(event) => setForm({ ...form, provider_tools_file_ids: event.currentTarget.value })} />
                    <small>{t("llmProfile.fileIdsHint")}</small>
                  </label> : null}
                  <p className="llmProviderToolsHint">{t(form.provider_tools_code_interpreter ? "llmProfile.codeIsolationHint" : "llmProfile.providerToolsHint")}</p>
                  {providerToolsError ? <p className="llmProfileWarning" role="alert">{t(providerToolsError)}</p> : null}
                </> : null}
              </fieldset>
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
                    <option value="in_memory">in_memory</option>
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
                <BooleanSelect label={t("llmProfile.fallbackJsonActions")} value={form.fallback_json_actions} onChange={(fallback_json_actions) => setForm({ ...form, fallback_json_actions })} />
                <label className="toggle">
                  <input type="checkbox" checked={form.allow_custom_base_url} onChange={(event) => setForm({ ...form, allow_custom_base_url: event.currentTarget.checked })} />
                  {t("llmProfile.allowCustomBaseUrl")}
                </label>
              </div>
            </fieldset>
            {localError ? <div className="llmProfileWarning" role="alert">{localError}</div> : null}
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
    provider_tools_provider: profile.provider_tools?.provider ?? "",
    provider_tools_web_search: profile.provider_tools?.web_search ?? false,
    provider_tools_web_extractor: profile.provider_tools?.web_extractor ?? false,
    provider_tools_code_interpreter: profile.provider_tools?.code_interpreter ?? false,
    provider_tools_file_ids: profile.provider_tools?.file_ids.join("\n") ?? "",
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
    fallback_json_actions: boolToForm(profile.fallback_json_actions),
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
    provider_tools: providerToolsFromForm(form),
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
    fallback_json_actions: formBoolToValue(form.fallback_json_actions),
    allow_custom_base_url: form.allow_custom_base_url
  };
}

function providerToolsFromForm(form: ProfileFormState): LLMProviderTools | null {
  if (!form.provider_tools_provider) return null;
  return {
    provider: form.provider_tools_provider,
    web_search: form.provider_tools_web_search,
    web_extractor: form.provider_tools_provider === "aliyun" && form.provider_tools_web_extractor,
    code_interpreter: form.provider_tools_code_interpreter,
    file_ids: form.provider_tools_provider === "openai" && form.provider_tools_code_interpreter
      ? Array.from(new Set(form.provider_tools_file_ids.split(/[\n,]/).map((id) => id.trim()).filter(Boolean))) : []
  };
}

function validateProviderTools(form: ProfileFormState): TranslationKey | null {
  const tools = providerToolsFromForm(form);
  if (!tools) return null;
  if (tools.web_extractor && !tools.web_search) return "llmProfile.extractorRequiresSearch";
  if (form.api_mode === "chat" && (tools.web_search || tools.web_extractor || tools.code_interpreter)
      && (tools.provider !== "aliyun" || tools.web_extractor || tools.code_interpreter)) return "llmProfile.toolsRequireResponses";
  if (tools.provider === "aliyun" && (tools.web_extractor || tools.code_interpreter)
      && form.reasoning_effort.trim().toLowerCase() === "none") return "llmProfile.toolsRequireThinking";
  if (tools.file_ids.length > 100 || tools.file_ids.some((id) => !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(id))) return "llmProfile.invalidFileIds";
  return null;
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
