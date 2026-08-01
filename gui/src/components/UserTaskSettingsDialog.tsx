import { useEffect, useId, useMemo, useState } from "react";
import type { ImageSummary, LLMProfileInput, LLMProfileSummary } from "../api/types";
import { useI18n } from "../i18n";
import { parseQuantaDraft } from "../quanta";
import {
  normalizeWorkspaceDirectory,
  type CommandAccess,
  type WorkspaceAccess
} from "../taskAuthority";
import { ImageSelect } from "./ImageSelect";
import {
  LLMProfileManagerDialog,
  LLMProfileSelect
} from "./LLMProfileSelect";
import { Modal } from "./Modal";

export type TaskLaunchSettings = {
  image: string;
  llmProfile: string;
  maxQuantaInput: string;
  workingDirectory: string;
  workspaceAccess: WorkspaceAccess;
  allowGitRequests: boolean;
  commandAccess: CommandAccess;
  contextMaintenance: boolean;
  authorityManifestId: string;
};

export type UserTaskSettingsDialogProps = {
  value: TaskLaunchSettings;
  images: ImageSummary[];
  llmProfiles: LLMProfileSummary[];
  busy?: boolean;
  onApply(next: TaskLaunchSettings): void;
  onClose(): void;
  onCreateLlmProfile(profile: LLMProfileInput): Promise<boolean>;
  onUpdateLlmProfile(profileId: string, profile: LLMProfileInput): Promise<boolean>;
  onDeleteLlmProfile(profileId: string): Promise<boolean>;
};

type DialogView = "settings" | "profiles";

/** Edits the launch-only task configuration as one local, atomic draft. */
export function UserTaskSettingsDialog({
  value,
  images,
  llmProfiles,
  busy = false,
  onApply,
  onClose,
  onCreateLlmProfile,
  onUpdateLlmProfile,
  onDeleteLlmProfile
}: UserTaskSettingsDialogProps) {
  const { t } = useI18n();
  const [draft, setDraft] = useState<TaskLaunchSettings>(() => ({ ...value }));
  const [view, setView] = useState<DialogView>("settings");
  const descriptionId = useId();
  const formId = useId();
  const quantaHintId = useId();
  const workingDirectoryErrorId = useId();
  const authorityManifestErrorId = useId();
  const quanta = useMemo(() => parseQuantaDraft(draft.maxQuantaInput), [draft.maxQuantaInput]);
  const workingDirectoryValid = useMemo(() => {
    try {
      normalizeWorkspaceDirectory(draft.workingDirectory);
      return true;
    } catch {
      return false;
    }
  }, [draft.workingDirectory]);
  const requiresAuthorityManifest = draft.workspaceAccess !== "none" || draft.allowGitRequests;
  const authorityManifestValid = !requiresAuthorityManifest || Boolean(draft.authorityManifestId.trim());
  const valid = quanta.valid && workingDirectoryValid && authorityManifestValid;

  useEffect(() => {
    if (!draft.llmProfile) return;
    if (llmProfiles.some((profile) => profile.profile_id === draft.llmProfile)) return;
    setDraft((current) => current.llmProfile ? { ...current, llmProfile: "" } : current);
  }, [draft.llmProfile, llmProfiles]);

  async function deleteProfile(profileId: string): Promise<boolean> {
    const deleted = await onDeleteLlmProfile(profileId);
    if (deleted) {
      setDraft((current) => current.llmProfile === profileId
        ? { ...current, llmProfile: "" }
        : current);
    }
    return deleted;
  }

  function apply() {
    if (busy || !valid) return;
    onApply({
      ...draft,
      authorityManifestId: draft.authorityManifestId.trim()
    });
    onClose();
  }

  if (view === "profiles") {
    return (
      <LLMProfileManagerDialog
        profiles={llmProfiles}
        selectedProfileId={draft.llmProfile}
        onCreate={onCreateLlmProfile}
        onUpdate={onUpdateLlmProfile}
        onDelete={deleteProfile}
        onClose={() => setView("settings")}
      />
    );
  }

  return (
    <Modal
      className="taskSettingsModal"
      title={t("user.taskSettingsDialogTitle")}
      descriptionId={descriptionId}
      busy={busy}
      onClose={onClose}
      actions={
        <div className="taskSettingsActions">
          <button type="button" className="secondary" disabled={busy} onClick={onClose}>
            {t("confirm.cancel")}
          </button>
          <button type="submit" form={formId} className="primary" disabled={busy || !valid}>
            {t("user.taskSettingsSave")}
          </button>
        </div>
      }
    >
      <p id={descriptionId} className="taskSettingsDescription">
        {t("user.taskSettingsDialogDescription")}
      </p>
      <form
        id={formId}
        className="taskSettingsForm"
        onSubmit={(event) => {
          event.preventDefault();
          apply();
        }}
      >
        <div className="taskSettingsSections">
          <section className="taskSettingsSection" aria-labelledby={`${descriptionId}-runtime`}>
            <h3 id={`${descriptionId}-runtime`}>{t("user.taskSettingsRuntime")}</h3>
            <div className="taskSettingsGrid">
              <div className="taskSettingsField">
                <ImageSelect
                  images={images}
                  value={draft.image}
                  disabled={busy}
                  onChange={(image) => setDraft((current) => ({ ...current, image }))}
                />
              </div>
              <div className="taskSettingsField">
                <LLMProfileSelect
                  profiles={llmProfiles}
                  value={draft.llmProfile}
                  label={t("llmProfile.spawnLabel")}
                  disabled={busy}
                  onChange={(llmProfile) => setDraft((current) => ({ ...current, llmProfile }))}
                  onManage={() => setView("profiles")}
                  onCreate={onCreateLlmProfile}
                  onUpdate={onUpdateLlmProfile}
                  onDelete={deleteProfile}
                />
              </div>
              <label className="fieldStack taskSettingsField">
                <span>{t("user.quanta")}</span>
                <input
                  type="text"
                  inputMode="numeric"
                  value={draft.maxQuantaInput}
                  disabled={busy}
                  aria-invalid={!quanta.valid || undefined}
                  aria-describedby={quantaHintId}
                  placeholder={t("scheduler.unlimitedPlaceholder")}
                  onChange={(event) => {
                    const maxQuantaInput = event.currentTarget.value;
                    setDraft((current) => ({ ...current, maxQuantaInput }));
                  }}
                />
                <small
                  id={quantaHintId}
                  className={quanta.valid ? "fieldHint" : "taskSettingsInlineError"}
                  role={quanta.valid ? undefined : "alert"}
                >
                  {quanta.valid ? t("scheduler.unlimitedHint") : t("scheduler.invalidQuanta")}
                </small>
              </label>
              <label className="fieldStack taskSettingsField">
                <span>{t("user.initialCwd")}</span>
                <input
                  value={draft.workingDirectory}
                  disabled={busy}
                  aria-invalid={!workingDirectoryValid || undefined}
                  aria-describedby={!workingDirectoryValid ? workingDirectoryErrorId : undefined}
                  placeholder={t("user.initialCwdPlaceholder")}
                  onChange={(event) => {
                    const workingDirectory = event.currentTarget.value;
                    setDraft((current) => ({ ...current, workingDirectory }));
                  }}
                />
                {!workingDirectoryValid ? (
                  <small id={workingDirectoryErrorId} className="taskSettingsInlineError" role="alert">
                    {t("user.invalidWorkingDirectory")}
                  </small>
                ) : null}
              </label>
            </div>
          </section>

          <section className="taskSettingsSection" aria-labelledby={`${descriptionId}-permissions`}>
            <h3 id={`${descriptionId}-permissions`}>{t("user.taskSettingsPermissions")}</h3>
            <div className="taskSettingsGrid">
              <label className="taskAuthorityField taskSettingsField">
                <span>{t("taskAuthority.workspaceAccess")}</span>
                <select
                  value={draft.workspaceAccess}
                  disabled={busy}
                  onChange={(event) => {
                    const workspaceAccess = event.currentTarget.value as WorkspaceAccess;
                    setDraft((current) => ({ ...current, workspaceAccess }));
                  }}
                >
                  <option value="none">{t("taskAuthority.none")}</option>
                  <option value="read">{t("taskAuthority.read")}</option>
                  <option value="edit">{t("taskAuthority.edit")}</option>
                  <option value="manage">{t("taskAuthority.manage")}</option>
                </select>
              </label>
              <label className="fieldStack taskSettingsField">
                <span>{t("taskAuthority.manifestId")}</span>
                <input
                  value={draft.authorityManifestId}
                  disabled={busy}
                  aria-invalid={!authorityManifestValid || undefined}
                  aria-describedby={!authorityManifestValid ? authorityManifestErrorId : undefined}
                  placeholder={t("taskAuthority.manifestIdPlaceholder")}
                  onChange={(event) => {
                    const authorityManifestId = event.currentTarget.value;
                    setDraft((current) => ({ ...current, authorityManifestId }));
                  }}
                />
                <small
                  id={authorityManifestErrorId}
                  className={authorityManifestValid ? "fieldHint" : "taskSettingsInlineError"}
                  role={authorityManifestValid ? undefined : "alert"}
                >
                  {authorityManifestValid
                    ? t("taskAuthority.manifestIdHint")
                    : t("taskRuns.authorityManifestRequired")}
                </small>
              </label>
              <label className="taskAuthorityToggle taskSettingsField">
                <input
                  type="checkbox"
                  checked={draft.allowGitRequests}
                  disabled={busy}
                  onChange={(event) => {
                    const allowGitRequests = event.currentTarget.checked;
                    setDraft((current) => ({ ...current, allowGitRequests }));
                  }}
                />
                <span>{t("taskAuthority.git")}</span>
              </label>
              <label className="taskAuthorityField taskSettingsField">
                <span>{t("taskAuthority.commandAccess")}</span>
                <select
                  value={draft.commandAccess}
                  disabled={busy}
                  onChange={(event) => {
                    const commandAccess = event.currentTarget.value as CommandAccess;
                    setDraft((current) => ({ ...current, commandAccess }));
                  }}
                >
                  <option value="none">{t("taskAuthority.commandNone")}</option>
                  <option value="reviewed">{t("taskAuthority.commandReviewed")}</option>
                </select>
              </label>
              <label className="taskAuthorityToggle taskSettingsField">
                <input
                  type="checkbox"
                  checked={draft.contextMaintenance}
                  disabled={busy}
                  onChange={(event) => {
                    const contextMaintenance = event.currentTarget.checked;
                    setDraft((current) => ({ ...current, contextMaintenance }));
                  }}
                />
                <span>{t("taskAuthority.contextMaintenance")}</span>
              </label>
            </div>
            <p className="taskAuthorityHint">{t("taskAuthority.hint")}</p>
          </section>
        </div>
      </form>
    </Modal>
  );
}
