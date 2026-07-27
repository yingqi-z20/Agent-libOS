import {
  Activity,
  AlertTriangle,
  ArrowDown,
  Bot,
  CheckCircle2,
  ChevronDown,
  CirclePlus,
  Cpu,
  Database,
  Gauge,
  Image as ImageIcon,
  LoaderCircle,
  ListTree,
  MessageSquare,
  Pause,
  Play,
  RefreshCw,
  Send,
  Settings,
  ShieldCheck,
  Sparkles,
  Square,
  Star,
  X
} from "lucide-react";
import { lazy, Suspense, useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import type {
  GuiConnection,
  HumanRequest,
  HumanResponseInput,
  ImageSummary,
  LLMProfileInput,
  LLMProfileSummary,
  RuntimeProcess,
  RuntimeSnapshot,
  StreamConnectionStatus
} from "../api/types";
import { useI18n, type TranslationKey } from "../i18n";
import { shortProcessId, taskDisplayLabel } from "../taskPresentation";
import { deriveUserConversation, humanRequestPrompt, type UserConversationItem } from "../userConversation";
import { HumanRequestCard, type HumanResponseOutcome } from "./HumanRequestCard";
import { ImageSelect } from "./ImageSelect";
import { LanguageSwitch } from "./LanguageSwitch";
import { RatingPanel } from "./RatingPanel";
import { UserTaskSettingsDialog, type TaskLaunchSettings } from "./UserTaskSettingsDialog";

const MarkdownMessage = lazy(async () => {
  const module = await import("./MarkdownMessage");
  return { default: module.MarkdownMessage };
});

const starterPromptKeys = [
  "user.starterPromptReview",
  "user.starterPromptTests",
  "user.starterPromptExplain"
] as const satisfies readonly TranslationKey[];

const processStatusKeys: Partial<Record<string, TranslationKey>> = {
  created: "process.status.created",
  runnable: "process.status.runnable",
  running: "process.status.running",
  paused: "process.status.paused",
  exited: "process.status.exited",
  failed: "process.status.failed",
  killed: "process.status.killed"
};

const workspaceAccessTranslationKeys: Record<TaskLaunchSettings["workspaceAccess"], TranslationKey> = {
  none: "taskAuthority.none",
  read: "taskAuthority.read",
  edit: "taskAuthority.edit",
  manage: "taskAuthority.manage"
};

type UserPageProps = {
  notices?: ReactNode;
  connection: GuiConnection | null;
  snapshot: RuntimeSnapshot | null;
  selectedPid: string | null;
  selectedProcess: RuntimeProcess | null;
  taskLabels: Readonly<Record<string, string>>;
  taskSettings: TaskLaunchSettings;
  quantaValid?: boolean;
  spawnGoal: string;
  message: string;
  images: ImageSummary[];
  llmProfiles: LLMProfileSummary[];
  onSelectPid(pid: string): void;
  onMaxQuantaChange(value: string): void;
  onSpawnGoalChange(value: string): void;
  onSpawnImageChange(value: string): void;
  onApplyTaskSettings(next: TaskLaunchSettings): void;
  onMessageChange(value: string): void;
  onSpawn(): void;
  onImportImage(): void;
  onCommitImage(request: { imageId: string; name: string; version: string; replace: boolean; checkpointId?: string }): void;
  onSend(kind: "message" | "interrupt"): void;
  onRespond(request: HumanRequest, response: HumanResponseInput): Promise<HumanResponseOutcome | boolean>;
  onRate(pid: string, score: number, comment: string): Promise<boolean>;
  onCreateLlmProfile(profile: LLMProfileInput): Promise<boolean>;
  onUpdateLlmProfile(profileId: string, profile: LLMProfileInput): Promise<boolean>;
  onDeleteLlmProfile(profileId: string): Promise<boolean>;
  onRun(): void;
  onPause(): void;
  onRefresh(): void;
  onOpenDb(): void;
  onShowOperator(): void;
  onStop(): void;
  busy?: boolean;
  streamStatus?: StreamConnectionStatus;
  lastUpdatedAt?: Date | null;
};

export function UserPage({
  notices,
  connection,
  snapshot,
  selectedPid,
  selectedProcess,
  taskLabels,
  taskSettings,
  quantaValid = true,
  spawnGoal,
  message,
  images,
  llmProfiles,
  onSelectPid,
  onMaxQuantaChange,
  onSpawnGoalChange,
  onSpawnImageChange,
  onApplyTaskSettings,
  onMessageChange,
  onSpawn,
  onImportImage,
  onCommitImage,
  onSend,
  onRespond,
  onRate,
  onCreateLlmProfile,
  onUpdateLlmProfile,
  onDeleteLlmProfile,
  onRun,
  onPause,
  onRefresh,
  onOpenDb,
  onShowOperator,
  onStop,
  busy = false,
  streamStatus = "connected",
  lastUpdatedAt = null
}: UserPageProps) {
  const { formatTime, t } = useI18n();
  const [commitImageId, setCommitImageId] = useState("");
  const [commitName, setCommitName] = useState("");
  const [commitVersion, setCommitVersion] = useState("v0");
  const [showNewTask, setShowNewTask] = useState(!selectedProcess);
  const [taskSettingsOpen, setTaskSettingsOpen] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const sidebarQuantaErrorId = useId();
  const newTaskStatusId = useId();
  const conversation = useMemo(() => deriveUserConversation(snapshot, selectedPid), [snapshot, selectedPid]);
  const pendingRequests = conversation.filter((item): item is Extract<UserConversationItem, { role: "request" }> => item.role === "request");
  const allPendingRequests = useMemo(
    () => (snapshot?.human_requests ?? []).filter((request) => request.status === "pending"),
    [snapshot?.human_requests]
  );
  const pendingRequestsByPid = useMemo(() => {
    const counts = new Map<string, number>();
    for (const request of allPendingRequests) counts.set(request.pid, (counts.get(request.pid) ?? 0) + 1);
    return counts;
  }, [allPendingRequests]);
  const schedulerRunning = Boolean(snapshot?.scheduler.running);
  const hasProcess = Boolean(selectedProcess);
  const processTerminal = Boolean(
    selectedProcess?.terminal || (selectedProcess && ["exited", "failed", "killed"].includes(selectedProcess.status))
  );
  const showTaskComposer = !hasProcess || showNewTask;
  const commitReady = Boolean(hasProcess && commitImageId.trim() && commitName.trim() && commitVersion.trim());
  const conversationRef = useRef<HTMLElement>(null);
  const goalInputRef = useRef<HTMLTextAreaElement>(null);
  const taskSettingsButtonRef = useRef<HTMLButtonElement>(null);
  const workspaceRef = useRef<HTMLElement>(null);
  const followConversationRef = useRef(true);
  const processRunning = selectedProcess?.status === "running";
  const statusTone = selectedProcess ? processStatusTone(selectedProcess.status) : "idle";
  const selectedStatusLabel = selectedProcess
    ? processStatusLabel(selectedProcess.status, t)
    : t("process.status.created");
  const selectedTaskLabel = selectedProcess
    ? taskLabels[selectedProcess.pid]?.trim() || t("user.untitledTask")
    : t("user.workspaceLabel");
  const selectedStatusDetail = selectedProcess ? processStatusDetail(selectedProcess, t) : null;
  const taskSettingsWorkspaceLabel = t(workspaceAccessTranslationKeys[taskSettings.workspaceAccess]);

  function closeTaskSettings() {
    setTaskSettingsOpen(false);
    globalThis.queueMicrotask(() => taskSettingsButtonRef.current?.focus());
  }

  useEffect(() => {
    setShowNewTask(!selectedProcess);
  }, [selectedProcess?.pid]);

  useEffect(() => {
    if (!showTaskComposer) setTaskSettingsOpen(false);
  }, [showTaskComposer]);

  useEffect(() => {
    const container = conversationRef.current;
    if (!container) return;
    if (followConversationRef.current) {
      container.scrollTop = container.scrollHeight;
      setShowJumpToLatest(false);
    } else {
      setShowJumpToLatest(true);
    }
  }, [conversation.at(-1)?.id, selectedPid]);

  function openNewTask() {
    setShowNewTask(true);
    setMobileSidebarOpen(false);
    globalThis.requestAnimationFrame?.(() => goalInputRef.current?.focus());
  }

  function openNextPendingRequest() {
    const next = allPendingRequests[0];
    if (!next) return;
    onSelectPid(next.pid);
    setShowNewTask(false);
    setMobileSidebarOpen(false);
    globalThis.requestAnimationFrame?.(() => document.getElementById("user-pending-requests")?.focus());
  }

  function scrollConversationToLatest() {
    const container = conversationRef.current;
    if (!container) return;
    const reducedMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    container.scrollTo({ top: container.scrollHeight, behavior: reducedMotion ? "auto" : "smooth" });
    followConversationRef.current = true;
    setShowJumpToLatest(false);
  }

  return (
    <main className="userPage">
      <a className="skipLink" href="#primary-workspace">{t("user.skipToWorkspace")}</a>

      <header className="userTopBar">
        <div className="userBrand">
          <span className="brandMark" aria-hidden="true"><Bot size={18} /></span>
          <div>
            <strong>Agent libOS</strong>
            <span>{t("user.workspaceLabel")} · {connection?.db ?? t("app.defaultDb")}</span>
          </div>
        </div>
        <div className="userTopActions">
          {busy ? (
            <span className="globalBusy" role="status">
              <LoaderCircle className="spin" size={14} aria-hidden="true" />{t("user.working")}
            </span>
          ) : null}
          <span className={`connectionBadge ${streamStatus}`} role="status">
            <span className="statusDot" />
            {t(`connection.${streamStatus}`)}
          </span>
          {lastUpdatedAt ? (
            <time className="lastUpdated" dateTime={lastUpdatedAt.toISOString()}>
              {t("connection.updated", { time: formatTime(lastUpdatedAt.toISOString()) })}
            </time>
          ) : null}
          {allPendingRequests.length > 0 ? (
            <button
              type="button"
              className="pendingInboxButton"
              aria-label={`${t("user.pendingRequests")}: ${allPendingRequests.length}`}
              title={t("user.pendingRequests")}
              onClick={openNextPendingRequest}
            >
              <AlertTriangle size={14} aria-hidden="true" />
              <span>{t("user.pendingRequests")}</span>
              <strong>{allPendingRequests.length}</strong>
            </button>
          ) : null}
          <LanguageSwitch />
          <button type="button" className="toolbarButton" disabled={busy} aria-label={t("user.openDbTitle")} title={t("user.openDbTitle")} onClick={onOpenDb}>
            <Database size={15} aria-hidden="true" /><span>{t("user.openDb")}</span>
          </button>
          <button type="button" className="iconOnly softButton" disabled={busy} aria-label={t("user.refreshTitle")} title={t("user.refreshTitle")} onClick={onRefresh}>
            <RefreshCw size={15} />
          </button>
          <button type="button" className="secondary toolbarButton" aria-label={t("user.operatorConsole")} title={t("user.operatorConsole")} onClick={onShowOperator}>
            <Settings size={15} aria-hidden="true" /><span>{t("user.operatorConsole")}</span>
          </button>
        </div>
      </header>

      {notices}

      <div className="userWorkspace">
        <button
          type="button"
          className="mobileSidebarToggle"
          aria-controls="task-sidebar"
          aria-expanded={mobileSidebarOpen}
          onClick={() => setMobileSidebarOpen((current) => !current)}
        >
          <ListTree size={16} aria-hidden="true" />
          <span><strong>{selectedTaskLabel}</strong><small>{selectedStatusLabel}</small></span>
          <ChevronDown size={16} className="disclosureChevron" aria-hidden="true" />
        </button>

        <aside id="task-sidebar" className={`userSidebar ${mobileSidebarOpen ? "mobileOpen" : ""}`} aria-label={t("user.sidebarLabel")}>
          <div className="sidebarHeading">
            <div>
              <span className="eyebrow">{t("user.tasksLabel")}</span>
              <strong>{t("user.workspaceLabel")}</strong>
            </div>
            <button
              type="button"
              className="iconOnly primary"
              aria-label={t("user.newTask")}
              title={t("user.newTask")}
              disabled={busy || showTaskComposer}
              onClick={openNewTask}
            >
              <CirclePlus size={16} />
            </button>
          </div>

          <label className="taskPicker">
            <span>{t("user.process")}</span>
            <select value={selectedPid ?? ""} disabled={busy} onChange={(event) => {
              onSelectPid(event.currentTarget.value);
              setMobileSidebarOpen(false);
              if (globalThis.matchMedia?.("(max-width: 820px)").matches) {
                globalThis.setTimeout(() => workspaceRef.current?.focus(), 0);
              }
            }}>
              {(snapshot?.processes.length ?? 0) === 0 ? <option value="">{t("user.noProcess")}</option> : null}
              {(snapshot?.processes ?? []).map((process) => (
                <option key={process.pid} value={process.pid}>
                  {taskDisplayLabel(process, taskLabels)} · {processStatusLabel(process.status, t)}
                  {pendingRequestsByPid.has(process.pid) ? ` · ⚠ ${pendingRequestsByPid.get(process.pid)}` : ""}
                </option>
              ))}
            </select>
          </label>

          {selectedProcess ? (
            <>
              <section className="taskSummaryCard" aria-label={t("user.processDetails")}>
                <div className="taskSummaryHeader">
                  <span className={`statusPill ${statusTone}`}>
                    <span className="statusDot" />{selectedStatusLabel}
                  </span>
                  <span className="taskPid" title={selectedProcess.pid}>{shortProcessId(selectedProcess.pid)}</span>
                </div>
                <strong className="taskDisplayLabel" title={selectedProcess.pid}>{selectedTaskLabel}</strong>
                <div className="taskRuntime">
                  <Cpu size={15} aria-hidden="true" />
                  <span>{selectedProcess.image_id}</span>
                  <small>{selectedProcess.llm_profile_id}</small>
                </div>
                {selectedStatusDetail ? <p className="taskStatusDetail">{selectedStatusDetail}</p> : null}
                <div className="metricGrid">
                  <div><strong>{selectedProcess.llm_call_count}</strong><span>{t("user.metricCalls")}</span></div>
                  <div><strong>{selectedProcess.token_total.toLocaleString()}</strong><span>{t("user.metricTokens")}</span></div>
                </div>
              </section>

              <section className="taskControlCard" aria-label={t("user.taskControls")}>
                <div className="sidebarSectionTitle">
                  <span><Gauge size={14} />{t("user.taskControls")}</span>
                  <span className={`statusDot ${processRunning ? "running" : ""}`} />
                </div>
                <label className="quanta sidebarQuanta">
                  <span>{t("user.quanta")}</span>
                  <input
                    type="text"
                    inputMode="numeric"
                    disabled={busy}
                    value={taskSettings.maxQuantaInput}
                    aria-invalid={!quantaValid || undefined}
                    aria-errormessage={!quantaValid ? sidebarQuantaErrorId : undefined}
                    placeholder={t("scheduler.unlimitedPlaceholder")}
                    title={quantaValid ? t("scheduler.unlimitedHint") : t("scheduler.invalidQuanta")}
                    onChange={(event) => onMaxQuantaChange(event.currentTarget.value)}
                  />
                  {!quantaValid ? <small id={sidebarQuantaErrorId} className="inlineError" role="alert">{t("scheduler.invalidQuanta")}</small> : null}
                </label>
                <div className="sidebarRunControls">
                  <button type="button" className="primary" disabled={busy || !quantaValid || schedulerRunning || processRunning || processTerminal} onClick={onRun}><Play size={15} />{t("user.run")}</button>
                  <button type="button" disabled={busy || !processRunning} onClick={onPause}><Pause size={15} />{t("user.pause")}</button>
                  <button type="button" className="dangerGhost" disabled={busy || processTerminal} onClick={onStop}><Square size={13} />{t("user.stop")}</button>
                </div>
              </section>

              <details className="sidebarDisclosure">
                <summary>
                  <span><ImageIcon size={15} />{t("user.imageTools")}</span>
                  <ChevronDown size={15} className="disclosureChevron" />
                </summary>
                <div className="sidebarDisclosureBody">
                  <p>{t("user.imageToolsHint")}</p>
                  <ImageSelect images={images} value={taskSettings.image} disabled={busy} onChange={onSpawnImageChange} />
                  <button type="button" className="fullWidthButton" disabled={busy} onClick={onImportImage}>{t("image.import")}</button>
                  <div className="sidebarImageForm">
                    <input disabled={busy} aria-label={t("image.commitIdPlaceholder")} value={commitImageId} onChange={(event) => setCommitImageId(event.currentTarget.value)} placeholder={t("image.commitIdPlaceholder")} />
                    <input disabled={busy} aria-label={t("image.commitNamePlaceholder")} value={commitName} onChange={(event) => setCommitName(event.currentTarget.value)} placeholder={t("image.commitNamePlaceholder")} />
                    <input disabled={busy} aria-label={t("image.version")} value={commitVersion} onChange={(event) => setCommitVersion(event.currentTarget.value)} placeholder={t("image.version")} />
                    <button
                      type="button"
                      className="warning fullWidthButton"
                      disabled={busy || !commitReady}
                      onClick={() => onCommitImage({
                        imageId: commitImageId.trim(),
                        name: commitName.trim(),
                        version: commitVersion.trim(),
                        replace: false
                      })}
                    >
                      {t("image.save")}
                    </button>
                  </div>
                </div>
              </details>

              <details className="sidebarDisclosure">
                <summary>
                  <span><Star size={15} />{t("rating.title")}</span>
                  <ChevronDown size={15} className="disclosureChevron" />
                </summary>
                <div className="sidebarDisclosureBody ratingDisclosureBody">
                  <RatingPanel process={selectedProcess} onSave={onRate} />
                </div>
              </details>
            </>
          ) : (
            <div className="sidebarEmpty">
              <Sparkles size={20} aria-hidden="true" />
              <strong>{t("user.noProcessYet")}</strong>
              <p>{t("user.noProcessHint")}</p>
            </div>
          )}

          <div className="sidebarFooter">
            <Database size={13} />
            <span title={connection?.db ?? t("app.defaultDb")}>{connection?.db ?? t("app.defaultDb")}</span>
          </div>
        </aside>

        <section ref={workspaceRef} className={`userMainPanel ${showTaskComposer ? "taskComposerMode" : ""}`} id="primary-workspace" tabIndex={-1} aria-label={showTaskComposer ? t("user.startTask") : t("user.conversation")}>
          {showTaskComposer ? (
            <div className="newTaskCanvas">
              <header className="newTaskHero">
                <span className="heroIcon" aria-hidden="true"><Sparkles size={22} /></span>
                <div>
                  <span className="eyebrow">{t("user.newTask")}</span>
                  <h1>{t("user.startTaskTitle")}</h1>
                  <p>{t("user.startTaskSubtitle")}</p>
                </div>
                {hasProcess ? (
                  <button type="button" className="iconOnly softButton" aria-label={t("user.cancelNewTask")} title={t("user.cancelNewTask")} onClick={() => setShowNewTask(false)}>
                    <X size={16} />
                  </button>
                ) : null}
              </header>

              <section className="newTaskCard" aria-label={t("user.startTask")}>
                <label className="newTaskGoalField">
                  <span>{t("user.goalLabel")}</span>
                  <textarea
                    ref={goalInputRef}
                    disabled={busy}
                    value={spawnGoal}
                    rows={5}
                    placeholder={t("user.goalPlaceholder")}
                    onChange={(event) => onSpawnGoalChange(event.currentTarget.value)}
                    onKeyDown={(event) => {
                      if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && spawnGoal.trim() && !busy) {
                        event.preventDefault();
                        onSpawn();
                      }
                    }}
                  />
                </label>

                <div className="starterPrompts" aria-label={t("user.starterPrompts")}>
                  <span>{t("user.starterPrompts")}</span>
                  <div>
                    {starterPromptKeys.map((key) => (
                      <button type="button" className="promptChip" key={key} disabled={busy} onClick={() => onSpawnGoalChange(t(key))}>
                        {t(key)}
                      </button>
                    ))}
                  </div>
                </div>

                <section className="taskSettingsLauncher" aria-label={t("user.taskSettings")}>
                  <div className="taskSettingsLauncherCopy">
                    <span className="settingsSummaryIcon" aria-hidden="true"><ShieldCheck size={17} /></span>
                    <span><strong>{t("user.taskSettings")}</strong><small>{t("user.taskSettingsHint")}</small></span>
                  </div>
                  <div className="taskSettingsSummary">
                    <div className="taskSettingsPrimarySummary">
                      <span className="taskSettingsSummaryItem">
                        <small>{t("user.taskSettingsImageSummary")}</small>
                        <strong title={taskSettings.image}>{taskSettings.image}</strong>
                      </span>
                      <span className="taskSettingsSummaryItem">
                        <small>{t("user.taskSettingsProfileSummary")}</small>
                        <strong title={taskSettings.llmProfile || t("llmProfile.defaultOption")}>
                          {taskSettings.llmProfile || t("llmProfile.defaultOption")}
                        </strong>
                      </span>
                      <span className="taskSettingsSummaryItem">
                        <small>{t("user.taskSettingsWorkspaceSummary")}</small>
                        <strong title={taskSettingsWorkspaceLabel}>{taskSettingsWorkspaceLabel}</strong>
                      </span>
                    </div>
                    <div className="taskSettingsBadges" aria-label={t("user.taskSettingsPermissions")}>
                      {taskSettings.allowGitRequests ? <span className="taskSettingsBadge">{t("user.taskSettingsGitBadge")}</span> : null}
                      {taskSettings.commandAccess === "reviewed" ? <span className="taskSettingsBadge">{t("user.taskSettingsCommandBadge")}</span> : null}
                      {taskSettings.contextMaintenance ? <span className="taskSettingsBadge">{t("user.taskSettingsContextBadge")}</span> : null}
                      {!taskSettings.allowGitRequests && taskSettings.commandAccess === "none" && !taskSettings.contextMaintenance
                        ? <span className="taskSettingsBadge muted">{t("user.taskSettingsNoExtraPermissions")}</span>
                        : null}
                    </div>
                  </div>
                  <button
                    ref={taskSettingsButtonRef}
                    type="button"
                    className="softButton"
                    aria-haspopup="dialog"
                    aria-expanded={taskSettingsOpen}
                    disabled={busy}
                    onClick={() => setTaskSettingsOpen(true)}
                  >
                    <Settings size={15} aria-hidden="true" />{t("user.taskSettingsEdit")}
                  </button>
                </section>

                {taskSettingsOpen ? (
                  <UserTaskSettingsDialog
                    value={taskSettings}
                    images={images}
                    llmProfiles={llmProfiles}
                    busy={busy}
                    onApply={onApplyTaskSettings}
                    onClose={closeTaskSettings}
                    onCreateLlmProfile={onCreateLlmProfile}
                    onUpdateLlmProfile={onUpdateLlmProfile}
                    onDeleteLlmProfile={onDeleteLlmProfile}
                  />
                ) : null}

                <footer className="newTaskActions">
                  <span id={newTaskStatusId} className={!quantaValid ? "actionHintError" : undefined}>
                    {quantaValid
                      ? `${t("user.startTaskHint")} ${t("user.startShortcut")}`
                      : t("scheduler.invalidQuanta")}
                  </span>
                  <button type="button" className="primary startTaskButton" aria-describedby={newTaskStatusId} disabled={busy || !quantaValid || !spawnGoal.trim()} onClick={onSpawn}>
                    {busy ? <LoaderCircle className="spin" size={16} aria-hidden="true" /> : <Play size={16} />}
                    {busy ? t("user.working") : t("user.start")}
                  </button>
                </footer>
              </section>
            </div>
          ) : (
            <>
              <header className="conversationHeader">
                <div className="conversationTitle">
                  <span className="eyebrow"><Activity size={12} />{t("user.activity")}</span>
                  <h1 title={selectedProcess?.pid}>{selectedTaskLabel}</h1>
                  <p>{selectedProcess ? `${shortProcessId(selectedProcess.pid)} · ${selectedProcess.image_id} · ${selectedProcess.llm_profile_id}` : ""}</p>
                  {selectedStatusDetail ? <span className="conversationStatusDetail">{selectedStatusDetail}</span> : null}
                </div>
                <div className="conversationHeaderMeta">
                  {selectedProcess?.interrupt_count ? <span className="interruptBanner"><AlertTriangle size={15} />{t("operator.interruptPending")}</span> : null}
                  <span className={`statusPill ${statusTone}`}><span className="statusDot" />{selectedStatusLabel}</span>
                </div>
              </header>

              {pendingRequests.length > 0 ? (
                <section id="user-pending-requests" className="userPendingRequests" aria-label={t("user.pendingRequests")} tabIndex={-1}>
                  <div className="pendingRequestsHeading"><AlertTriangle size={15} /><strong>{t("user.pendingRequests")}</strong><span>{pendingRequests.length}</span></div>
                  {pendingRequests.map(({ request }) => (
                    <HumanRequestCard className="userRequestCard" key={request.request_id} request={request} onRespond={onRespond} />
                  ))}
                </section>
              ) : null}

              <section
                ref={conversationRef}
                className="userConversation"
                aria-label={t("user.conversation")}
                role="log"
                tabIndex={0}
                aria-live="polite"
                aria-relevant="additions"
                onScroll={(event) => {
                  const element = event.currentTarget;
                  const nearLatest = element.scrollHeight - element.scrollTop - element.clientHeight < 96;
                  followConversationRef.current = nearLatest;
                  if (nearLatest) setShowJumpToLatest(false);
                }}
              >
                {conversation.length === 0 ? (
                  <div className="userEmpty">
                    <span className="emptyIllustration"><MessageSquare size={22} /></span>
                    <strong>{t("user.readyTitle")}</strong>
                    <span>{t("user.emptyConversation")}</span>
                  </div>
                ) : conversation.map((item) => <ConversationBubble key={item.id} item={item} />)}
              </section>

              {showJumpToLatest ? (
                <button type="button" className="jumpToLatest" onClick={scrollConversationToLatest}>
                  <ArrowDown size={14} aria-hidden="true" />{t("user.jumpToLatest")}
                </button>
              ) : null}

              {processTerminal ? (
                <footer className="terminalNextStep" role="status">
                  <span><CheckCircle2 size={17} aria-hidden="true" /><span><strong>{t("user.taskClosed")}</strong><small>{t("user.taskClosedHint")}</small></span></span>
                  <button type="button" className="primary" disabled={busy} onClick={openNewTask}><CirclePlus size={15} />{t("user.startAnotherTask")}</button>
                </footer>
              ) : <footer className="userComposer">
                <div className="composerField">
                  <textarea
                    rows={1}
                    value={message}
                    disabled={busy || !hasProcess}
                    aria-label={t("user.messageAgent")}
                    aria-describedby="composer-hint"
                    onChange={(event) => onMessageChange(event.currentTarget.value)}
                    placeholder={t("user.messageAgent")}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                        event.preventDefault();
                        if (quantaValid && message.trim()) onSend("message");
                      }
                    }}
                  />
                  <span id="composer-hint">{t("user.sendHint")}</span>
                </div>
                <button type="button" className="primary sendButton" disabled={busy || !quantaValid || !hasProcess || !message.trim()} onClick={() => onSend("message")}>
                  <Send size={15} />{t("user.send")}
                </button>
                <button
                  type="button"
                  disabled={busy || !quantaValid || !hasProcess || !message.trim()}
                  className="iconOnly warning"
                  aria-label={t("user.interrupt")}
                  title={t("user.interrupt")}
                  onClick={() => onSend("interrupt")}
                >
                  <AlertTriangle size={15} />
                </button>
              </footer>}
            </>
          )}
        </section>
      </div>
    </main>
  );
}

export type ProcessStatusTone = "running" | "waiting" | "completed" | "terminal" | "paused" | "idle";

export function processStatusLabel(
  status: string,
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string
): string {
  if (status.startsWith("waiting")) {
    return t(status.includes("human") ? "process.status.waitingHuman" : "process.status.waiting");
  }
  const key = processStatusKeys[status];
  return key ? t(key) : status;
}

export function processStatusTone(status: string, schedulerRunning = false): ProcessStatusTone {
  if (status === "exited") return "completed";
  if (["failed", "killed"].includes(status)) return "terminal";
  if (status === "paused") return "paused";
  if (status.startsWith("waiting")) return "waiting";
  if (schedulerRunning || status === "running" || status === "runnable") return "running";
  return "idle";
}

export function processStatusDetail(
  process: RuntimeProcess,
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string
): string | null {
  const wait = process.wait_state;
  if (wait?.kind === "human") return t("user.statusWaitingHuman");
  if (wait?.kind === "message") return t("user.statusWaitingMessage");
  if (wait?.kind === "child") return t("user.statusWaitingChild", { pid: shortProcessId(wait.child_pid) });
  if (wait?.kind === "tool") return t("user.statusWaitingTool");
  if (wait?.kind === "paused") return t("user.statusPaused");
  if (wait?.kind === "host_resume") return t("user.statusWaitingResume");
  if (process.status.startsWith("waiting_human")) return t("user.statusWaitingHuman");
  if (process.status.startsWith("waiting_message")) return t("user.statusWaitingMessage");
  if (process.status === "paused") return t("user.statusPaused");
  if (process.terminal || ["exited", "failed", "killed"].includes(process.status)) return null;
  return process.status_message?.trim() || null;
}

function ConversationBubble({ item }: { item: UserConversationItem }) {
  const { formatTime, t } = useI18n();
  if (item.role === "request") {
    return (
      <article className="conversationBubble request">
        <span className="bubbleRole">{t("user.needsInput")}</span>
        <p>{humanRequestPrompt(item.request)}</p>
        <time>{formatTime(item.time)}</time>
      </article>
    );
  }
  if (item.role === "decision") {
    const fallback = item.status === "rejected" ? t("user.requestRejected") : t("user.requestApproved");
    return (
      <article className="conversationBubble user">
        <span className="bubbleRole">{t("user.you")}</span>
        <p>{item.text || fallback}</p>
        <time>{formatTime(item.time)}</time>
      </article>
    );
  }
  if (item.role === "terminal") {
    const title = item.outcome.kind === "exited"
      ? t("user.taskCompleted")
      : item.outcome.kind === "failed"
        ? t("user.taskFailed")
        : t("user.taskStopped");
    const reference = item.outcome.kind === "killed" ? item.outcome.reason_oid : item.outcome.result_oid;
    const code = item.outcome.kind === "exited" ? null : item.outcome.code;
    return (
      <article className={`conversationBubble terminal ${item.outcome.kind}`}>
        <span className="bubbleRole">{t("user.taskStatus")}</span>
        <p>{code ? `${title} (${code})` : title}</p>
        {item.text ? <p>{item.text}</p> : null}
        {reference ? <p className="terminalReference">{t("user.resultReference", { oid: reference })}</p> : null}
        <time>{formatTime(item.time)}</time>
      </article>
    );
  }
  return (
    <article className={`conversationBubble ${item.role}`}>
      <span className="bubbleRole">{item.role === "assistant" ? t("user.agent") : t("user.you")}</span>
      {item.role === "assistant" ? (
        <Suspense fallback={<p>{item.protected ? t("user.protectedOutput") : item.text || t("user.empty")}</p>}>
          <MarkdownMessage text={item.protected ? "" : item.text} fallback={item.protected ? t("user.protectedOutput") : t("user.empty")} />
        </Suspense>
      ) : (
        <p>{item.text || t("user.empty")}</p>
      )}
      <time>{formatTime(item.time)}</time>
    </article>
  );
}
