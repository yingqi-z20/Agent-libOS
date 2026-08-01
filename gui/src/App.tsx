import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, AlertTriangle, ChevronDown, CirclePlus, PanelRight, Send, Settings } from "lucide-react";
import {
  isUnadmittedTaskRunRevisionConflict,
  LibOSClient,
  taskRunConflictSummary
} from "./api/client";
import { allowedTaskRunActions, assertSchedulerStatus, runtimeSnapshotFromSseData, taskRunSummaryFromSseData } from "./api/types";
import type { GuiConnection, HumanRequest, HumanResponseInput, RuntimeProcess, RuntimeSnapshot, SchedulerStatus, StreamConnectionStatus, TaskRunDetail, TaskRunHumanRequestPage, TaskRunSpecV1, TaskRunSummary } from "./api/types";
import { AppNotices, LoadingScreen } from "./components/AppNotices";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { DetailTabs } from "./components/DetailTabs";
import { ImageSelect } from "./components/ImageSelect";
import { HumanRequestCard, type HumanResponseOutcome } from "./components/HumanRequestCard";
import { LLMProfileSelect } from "./components/LLMProfileSelect";
import { ProcessTree } from "./components/ProcessTree";
import { Timeline } from "./components/Timeline";
import { TopBar } from "./components/TopBar";
import { processStatusLabel, processStatusTone, UserPage } from "./components/UserPage";
import type { TaskLaunchSettings } from "./components/UserTaskSettingsDialog";
import { useI18n } from "./i18n";
import { parseQuantaDraft } from "./quanta";
import { mergeRuntimeTaskRuns, processFromMutationResult, reconcileSelectedPid, reconcileSelectedRunId, upsertRuntimeProcess, upsertRuntimeTaskRun } from "./selection";
import { runOrResumeProcess } from "./runControl";
import type { LLMProfileInput } from "./api/types";
import type { ConfirmationRequest } from "./adminTypes";
import { developmentConnection } from "./developmentConnection";
import { buildGuiDurableTaskAuthority, buildGuiTaskAuthorityManifest, DEFAULT_DURABLE_TASK_LAUNCH, type CommandAccess, type WorkspaceAccess } from "./taskAuthority";
import {
  shortProcessId,
  taskDisplayLabel,
  taskLabelFromGoal,
  taskLabelsForStorage,
  taskLabelsFromStorage
} from "./taskPresentation";
import { SnapshotEpoch } from "./snapshotEpoch";
import {
  createAndRunTaskRun,
  clearTaskRunFollowUpDraft,
  rotateUnadmittedTaskRunStartCommand,
  submitTaskRunFollowUp,
  taskRunFollowUpIntent,
  taskRunMutationIntent,
  taskRunStartIntent,
  type TaskRunFollowUpIntent,
  type TaskRunMutationIntent,
  type TaskRunMutationKind,
  type TaskRunStartIntent
} from "./taskRunControl";

type PendingConfirm = ConfirmationRequest;
const TASK_LABELS_STORAGE_KEY = "agent-libos.gui.task-labels";
const SELECTED_PID_STORAGE_KEY = "agent-libos.gui.selected-pid";
const SELECTED_RUN_STORAGE_KEY = "agent-libos.gui.selected-run-id";

export function App() {
  const { t } = useI18n();
  const [view, setViewState] = useState<"user" | "operator">(() => readStoredView());
  const [connection, setConnection] = useState<GuiConnection | null>(null);
  const [client, setClient] = useState<LibOSClient | null>(null);
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [selectedPid, setSelectedPid] = useState<string | null>(readStoredSelectedPid);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(readStoredSelectedRunId);
  const [selectedRunDetail, setSelectedRunDetail] = useState<TaskRunDetail | null>(null);
  const [selectedRunHumanPage, setSelectedRunHumanPage] = useState<(
    TaskRunHumanRequestPage & { run_id: string; revision: number }
  ) | null>(null);
  const [selectedRunHumanLoading, setSelectedRunHumanLoading] = useState(false);
  const [maxQuantaInput, setMaxQuantaInput] = useState("");
  const [spawnGoal, setSpawnGoal] = useState("");
  const [spawnImage, setSpawnImage] = useState<string>(DEFAULT_DURABLE_TASK_LAUNCH.imageId);
  const [spawnLlmProfile, setSpawnLlmProfile] = useState<string>(DEFAULT_DURABLE_TASK_LAUNCH.llmProfileId);
  const [spawnWorkingDirectory, setSpawnWorkingDirectory] = useState<string>(DEFAULT_DURABLE_TASK_LAUNCH.workingDirectory);
  const [spawnWorkspaceAccess, setSpawnWorkspaceAccess] = useState<WorkspaceAccess>(DEFAULT_DURABLE_TASK_LAUNCH.workspaceAccess);
  const [spawnAllowGitRequests, setSpawnAllowGitRequests] = useState<boolean>(DEFAULT_DURABLE_TASK_LAUNCH.allowGitRequests);
  const [spawnCommandAccess, setSpawnCommandAccess] = useState<CommandAccess>(DEFAULT_DURABLE_TASK_LAUNCH.commandAccess);
  const [spawnContextMaintenance, setSpawnContextMaintenance] = useState<boolean>(DEFAULT_DURABLE_TASK_LAUNCH.contextMaintenance);
  const [spawnAuthorityManifestId, setSpawnAuthorityManifestId] = useState<string>(DEFAULT_DURABLE_TASK_LAUNCH.authorityManifestId);
  const [spawnPanelOpen, setSpawnPanelOpen] = useState(false);
  const [taskLabels, setTaskLabels] = useState<Record<string, string>>(readStoredTaskLabels);
  const [messageDrafts, setMessageDrafts] = useState<Record<string, string>>({});
  const [cwdDrafts, setCwdDrafts] = useState<Record<string, string>>({});
  const [execImage, setExecImage] = useState("base-agent:v0");
  const [execLlmProfile, setExecLlmProfile] = useState("");
  const [execGoalDrafts, setExecGoalDrafts] = useState<Record<string, string>>({});
  const [pendingConfirm, setPendingConfirm] = useState<PendingConfirm | null>(null);
  const [confirmBusy, setConfirmBusy] = useState(false);
  const [confirmError, setConfirmError] = useState<string | null>(null);
  const [initializing, setInitializing] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeAction, setActiveAction] = useState<string | null>(null);
  const [streamStatus, setStreamStatus] = useState<StreamConnectionStatus>("connecting");
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explainLookup, setExplainLookup] = useState<{ pid: string; kind: string; id: string; nonce: number } | null>(null);
  const [connectionEpoch, setConnectionEpoch] = useState(0);
  const [streamEpoch, setStreamEpoch] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const requestAbortRef = useRef<AbortController | null>(null);
  const activeClientRef = useRef<LibOSClient | null>(null);
  const snapshotStateRef = useRef<RuntimeSnapshot | null>(null);
  const initializationInFlightRef = useRef<Promise<void> | null>(null);
  const refreshInFlightRef = useRef<Promise<boolean> | null>(null);
  const actionGuardRef = useRef(false);
  const confirmGuardRef = useRef(false);
  const snapshotEpochRef = useRef(new SnapshotEpoch());
  const ambiguousHumanRequestsRef = useRef(new Set<string>());
  const taskRunDetailEpochRef = useRef(0);
  const taskRunHumanEpochRef = useRef(0);
  const taskRunStartIntentRef = useRef<TaskRunStartIntent | null>(null);
  const taskRunFollowUpIntentsRef = useRef(new Map<string, TaskRunFollowUpIntent>());
  const taskRunMutationIntentsRef = useRef(new Map<string, TaskRunMutationIntent>());
  const quantaDraft = useMemo(() => parseQuantaDraft(maxQuantaInput), [maxQuantaInput]);
  const maxQuanta = quantaDraft.value;
  const taskLaunchSettings = useMemo<TaskLaunchSettings>(() => ({
    image: spawnImage,
    llmProfile: spawnLlmProfile,
    maxQuantaInput,
    workingDirectory: spawnWorkingDirectory,
    workspaceAccess: spawnWorkspaceAccess,
    allowGitRequests: spawnAllowGitRequests,
    commandAccess: spawnCommandAccess,
    contextMaintenance: spawnContextMaintenance,
    authorityManifestId: spawnAuthorityManifestId
  }), [
    maxQuantaInput,
    spawnAllowGitRequests,
    spawnCommandAccess,
    spawnContextMaintenance,
    spawnAuthorityManifestId,
    spawnImage,
    spawnLlmProfile,
    spawnWorkingDirectory,
    spawnWorkspaceAccess
  ]);

  function replaceSnapshotState(next: RuntimeSnapshot | null): RuntimeSnapshot | null {
    snapshotStateRef.current = next;
    setSnapshot(next);
    return next;
  }

  function updateSnapshotState(
    updater: (current: RuntimeSnapshot | null) => RuntimeSnapshot | null
  ): RuntimeSnapshot | null {
    return replaceSnapshotState(updater(snapshotStateRef.current));
  }

  function applyTaskLaunchSettings(next: TaskLaunchSettings) {
    setSpawnImage(next.image);
    setSpawnLlmProfile(next.llmProfile);
    setMaxQuantaInput(next.maxQuantaInput);
    setSpawnWorkingDirectory(next.workingDirectory);
    setSpawnWorkspaceAccess(next.workspaceAccess);
    setSpawnAllowGitRequests(next.allowGitRequests);
    setSpawnCommandAccess(next.commandAccess);
    setSpawnContextMaintenance(next.contextMaintenance);
    setSpawnAuthorityManifestId(next.authorityManifestId);
  }

  useEffect(() => {
    void initialize();
    return () => {
      abortRef.current?.abort();
      requestAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    try {
      globalThis.sessionStorage?.setItem(TASK_LABELS_STORAGE_KEY, taskLabelsForStorage(taskLabels));
    } catch {
      // Task labels are an optional session-only convenience in restricted renderers.
    }
  }, [taskLabels]);

  useEffect(() => {
    try {
      if (selectedPid) globalThis.sessionStorage?.setItem(SELECTED_PID_STORAGE_KEY, selectedPid);
      else globalThis.sessionStorage?.removeItem(SELECTED_PID_STORAGE_KEY);
    } catch {
      // Selection persistence is optional in restricted renderer environments.
    }
  }, [selectedPid]);

  useEffect(() => {
    try {
      if (selectedRunId) globalThis.sessionStorage?.setItem(SELECTED_RUN_STORAGE_KEY, selectedRunId);
      else globalThis.sessionStorage?.removeItem(SELECTED_RUN_STORAGE_KEY);
    } catch {
      // Selection persistence is optional in restricted renderer environments.
    }
  }, [selectedRunId]);

  useEffect(() => {
    if (pendingConfirm) setConfirmError(null);
  }, [pendingConfirm]);

  useEffect(() => {
    if (initializing) return;
    const shell = document.querySelector<HTMLElement>(view === "user" ? ".userAppShell" : ".appShell");
    const header = shell?.querySelector<HTMLElement>(view === "user" ? ".userTopBar" : ".topBar");
    if (!shell || !header) return;
    const updateHeaderInset = () => {
      shell.style.setProperty("--app-header-height", `${Math.ceil(header.getBoundingClientRect().height)}px`);
    };
    updateHeaderInset();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(updateHeaderInset);
    observer?.observe(header);
    globalThis.addEventListener("resize", updateHeaderInset);
    return () => {
      observer?.disconnect();
      globalThis.removeEventListener("resize", updateHeaderInset);
    };
  }, [initializing, view]);

  useEffect(() => {
    if (!client) return;
    const streamClient = client;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    void client.stream((message) => {
      if (activeClientRef.current !== streamClient) return;
      if (message.event === "snapshot") {
        try {
          const next = runtimeSnapshotFromSseData(message.data);
          snapshotEpochRef.current.acceptStreamSnapshot();
          const merged = updateSnapshotState((current) => current ? mergeRuntimeTaskRuns(next, current) : next);
          if (!merged) return;
          setSelectedPid((current) => reconcileSelectedPid(merged, current));
          setSelectedRunId((current) => reconcileSelectedRunId(merged, current));
          setLastUpdatedAt(new Date());
        } catch (reason) {
          setError(describeError(reason, t("app.confirmationRequiredSuffix")));
        }
      }
      if (message.event === "task_run.updated") {
        try {
          const run = taskRunSummaryFromSseData(message.data);
          const current = snapshotStateRef.current;
          const previous = current?.task_runs.find((item) => item.run_id === run.run_id);
          if (!current || (previous && previous.revision >= run.revision)) return;
          snapshotEpochRef.current.acceptStreamSnapshot();
          updateSnapshotState((value) => value ? upsertRuntimeTaskRun(value, run) : value);
          setSelectedRunId((current) => current ?? run.run_id);
          setLastUpdatedAt(new Date());
        } catch (reason) {
          setError(describeError(reason, t("app.confirmationRequiredSuffix")));
          void refresh();
        }
      }
      if (message.event === "snapshot_truncated" || message.event === "event.invalidated") {
        void refresh();
      }
      if (message.event === "scheduler.status") {
        try {
          assertSchedulerStatus(message.data);
          mergeSchedulerStatus(message.data);
          setLastUpdatedAt(new Date());
        } catch (reason) {
          setError(describeError(reason, t("app.confirmationRequiredSuffix")));
        }
      }
    }, controller.signal, "0", (status) => {
      if (activeClientRef.current === streamClient) setStreamStatus(status);
    }).catch((reason) => {
      if (!controller.signal.aborted && activeClientRef.current === streamClient) {
        setError(describeError(reason, t("app.confirmationRequiredSuffix")));
      }
    });
    return () => controller.abort();
  }, [client, streamEpoch]);

  const selectedProcess = useMemo(
    () => snapshot?.processes.find((process) => process.pid === selectedPid) ?? null,
    [snapshot, selectedPid]
  );
  const selectedRun = useMemo(
    () => snapshot?.task_runs.find((run) => run.run_id === selectedRunId) ?? null,
    [snapshot, selectedRunId]
  );
  const visibleSelectedRunHumanPage = selectedRunHumanPage
    && selectedRun
    && selectedRunHumanPage.run_id === selectedRun.run_id
    && selectedRunHumanPage.revision === selectedRun.revision
    ? selectedRunHumanPage
    : null;
  useEffect(() => {
    if (view !== "user" || !selectedRun) return;
    const pid = selectedRun.active_pid ?? selectedRun.root_pid;
    if (pid && pid !== selectedPid) setSelectedPid(pid);
  }, [selectedRun?.run_id, selectedRun?.revision, selectedPid, view]);
  useEffect(() => {
    const requestClient = client;
    const run = selectedRun;
    const epoch = ++taskRunDetailEpochRef.current;
    if (!requestClient || !run) {
      setSelectedRunDetail(null);
      return;
    }
    void requestClient.getTaskRun(run.run_id, { requirementsLimit: 100 }).then((detail) => {
      if (taskRunDetailEpochRef.current !== epoch || activeClientRef.current !== requestClient) return;
      if (detail.summary.run_id !== run.run_id || detail.summary.revision < run.revision) return;
      if (detail.summary.revision > run.revision) {
        snapshotEpochRef.current.acceptAuthoritativeSnapshot();
        updateSnapshotState((current) => current
          ? upsertRuntimeTaskRun(current, detail.summary)
          : current);
      }
      setSelectedRunDetail(detail);
    }).catch((reason) => {
      if (taskRunDetailEpochRef.current === epoch && activeClientRef.current === requestClient) {
        setSelectedRunDetail(null);
        setError(describeError(reason, t("app.confirmationRequiredSuffix")));
      }
    });
  }, [client, selectedRun?.run_id, selectedRun?.revision]);
  useEffect(() => {
    taskRunHumanEpochRef.current += 1;
    setSelectedRunHumanPage(null);
    setSelectedRunHumanLoading(false);
    if (client && selectedRun) void loadSelectedRunHumanRequests(false);
    return () => {
      taskRunHumanEpochRef.current += 1;
    };
    // The loader snapshots both identities and is fenced by taskRunHumanEpochRef.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, selectedRun?.run_id, selectedRun?.revision]);
  const messageDraftKey = selectedRun
    ? `run:${selectedRun.run_id}`
    : selectedPid;
  const message = messageDraftKey ? messageDrafts[messageDraftKey] ?? "" : "";
  const cwd = selectedPid ? cwdDrafts[selectedPid] ?? "" : "";
  const execGoal = selectedPid ? execGoalDrafts[selectedPid] ?? "" : "";
  const selectedExplainLookup = explainLookup?.pid === selectedPid ? explainLookup : null;

  function setMessage(value: string) {
    if (!messageDraftKey) return;
    setMessageDrafts((current) => ({ ...current, [messageDraftKey]: value }));
  }

  function setCwd(value: string) {
    if (!selectedPid) return;
    setCwdDrafts((current) => ({ ...current, [selectedPid]: value }));
  }

  function setExecGoal(value: string) {
    if (!selectedPid) return;
    setExecGoalDrafts((current) => ({ ...current, [selectedPid]: value }));
  }

  async function loadSelectedRunHumanRequests(append: boolean): Promise<void> {
    const requestClient = client;
    const run = selectedRun;
    const currentPage = selectedRunHumanPage;
    if (!requestClient || !run || (append && (
      currentPage?.run_id !== run.run_id
      || currentPage.revision !== run.revision
      || !currentPage.has_more
      || !currentPage.next_cursor
    ))) return;
    const epoch = ++taskRunHumanEpochRef.current;
    setSelectedRunHumanLoading(true);
    try {
      const page = await requestClient.listTaskRunHumanRequests(
        run.run_id,
        100,
        append ? currentPage?.next_cursor ?? undefined : undefined,
        ["pending"]
      );
      if (taskRunHumanEpochRef.current !== epoch || activeClientRef.current !== requestClient) return;
      setSelectedRunHumanPage({
        ...page,
        run_id: run.run_id,
        revision: run.revision,
        items: append && currentPage
          ? dedupeHumanRequests([...currentPage.items, ...page.items])
          : page.items
      });
    } catch (reason) {
      if (taskRunHumanEpochRef.current === epoch && activeClientRef.current === requestClient) {
        setError(describeError(reason, t("app.confirmationRequiredSuffix")));
      }
    } finally {
      if (taskRunHumanEpochRef.current === epoch) setSelectedRunHumanLoading(false);
    }
  }

  function initialize(): Promise<void> {
    if (initializationInFlightRef.current) return initializationInFlightRef.current;
    const request = initializeOnce().finally(() => {
      if (initializationInFlightRef.current === request) initializationInFlightRef.current = null;
    });
    initializationInFlightRef.current = request;
    return request;
  }

  async function initializeOnce() {
    setInitializing(true);
    setError(null);
    try {
      const conn = await window.libosApi?.getConnection()
        ?? (import.meta.env.DEV ? developmentConnection(import.meta.env, true) : null);
      if (!conn) throw new Error(t("app.preloadMissing"));
      const nextClient = new LibOSClient(conn);
      requestAbortRef.current?.abort();
      const requestController = new AbortController();
      requestAbortRef.current = requestController;
      const nextSnapshot = await nextClient.snapshot({ signal: requestController.signal, timeoutMs: 15_000 });
      activeClientRef.current = nextClient;
      snapshotEpochRef.current.acceptAuthoritativeSnapshot();
      setConnection(conn);
      setClient(nextClient);
      setConnectionEpoch((current) => current + 1);
      replaceSnapshotState(nextSnapshot);
      setSelectedPid((current) => reconcileSelectedPid(nextSnapshot, current));
      setSelectedRunId((current) => reconcileSelectedRunId(nextSnapshot, current));
      setMaxQuantaInput(nextSnapshot.scheduler.default_max_quanta?.toString() ?? "");
      setLastUpdatedAt(new Date());
    } catch (reason) {
      setError(describeError(reason, t("app.confirmationRequiredSuffix")));
    } finally {
      setInitializing(false);
    }
  }

  function refresh(): Promise<boolean> {
    if (!client) return Promise.resolve(false);
    if (refreshInFlightRef.current) return refreshInFlightRef.current;
    const requestClient = client;
    const requestEpoch = snapshotEpochRef.current.beginHttpRequest();
    setRefreshing(true);
    let request: Promise<boolean>;
    request = requestClient.snapshot({ signal: requestAbortRef.current?.signal, timeoutMs: 15_000 }).then((next) => {
      if (activeClientRef.current !== requestClient) return false;
      if (!snapshotEpochRef.current.acceptHttpResponse(requestEpoch)) return true;
      const merged = snapshotStateRef.current
        ? mergeRuntimeTaskRuns(next, snapshotStateRef.current)
        : next;
      replaceSnapshotState(merged);
      setSelectedPid((current) => reconcileSelectedPid(merged, current));
      setSelectedRunId((current) => reconcileSelectedRunId(merged, current));
      setLastUpdatedAt(new Date());
      return true;
    }).catch((reason) => {
      if (activeClientRef.current === requestClient && snapshotEpochRef.current.isCurrent(requestEpoch)) {
        setError(describeError(reason, t("app.confirmationRequiredSuffix")));
      }
      return !snapshotEpochRef.current.isCurrent(requestEpoch);
    }).finally(() => {
      if (refreshInFlightRef.current === request) {
        setRefreshing(false);
        refreshInFlightRef.current = null;
      }
    });
    refreshInFlightRef.current = request;
    return request;
  }

  async function reconnect(next: GuiConnection | null) {
    if (!next) return;
    if (connection && sameConnection(connection, next)) return;
    const nextClient = new LibOSClient(next);
    requestAbortRef.current?.abort();
    const requestController = new AbortController();
    requestAbortRef.current = requestController;
    const nextSnapshot = await nextClient.snapshot({ signal: requestController.signal, timeoutMs: 15_000 });
    abortRef.current?.abort();
    activeClientRef.current = nextClient;
    snapshotEpochRef.current.acceptAuthoritativeSnapshot();
    refreshInFlightRef.current = null;
    setRefreshing(false);
    setConnection(next);
    setClient(nextClient);
    taskRunStartIntentRef.current = null;
    taskRunFollowUpIntentsRef.current.clear();
    taskRunMutationIntentsRef.current.clear();
    setConnectionEpoch((current) => current + 1);
    replaceSnapshotState(nextSnapshot);
    setSelectedPid(reconcileSelectedPid(nextSnapshot, null, { preserveExisting: false }));
    setSelectedRunId(reconcileSelectedRunId(nextSnapshot, null, { preserveExisting: false }));
    setMaxQuantaInput(nextSnapshot.scheduler.default_max_quanta?.toString() ?? "");
    setLastUpdatedAt(new Date());
    setError(null);
  }

  async function openDatabase() {
    if (actionGuardRef.current) return;
    actionGuardRef.current = true;
    setActiveAction("database.open");
    try {
      setError(null);
      if (refreshInFlightRef.current) await refreshInFlightRef.current;
      const next = await window.libosApi?.chooseDatabase();
      await reconnect(next ?? null);
    } catch (reason) {
      setError(describeError(reason, t("app.confirmationRequiredSuffix")));
    } finally {
      actionGuardRef.current = false;
      setActiveAction(null);
    }
  }

  async function safe(
    action: () => Promise<void>,
    label = "action",
    refreshAfter = true
  ): Promise<boolean> {
    if (actionGuardRef.current) return false;
    actionGuardRef.current = true;
    setActiveAction(label);
    try {
      setError(null);
      if (refreshAfter && refreshInFlightRef.current) await refreshInFlightRef.current;
      await action();
      return refreshAfter ? await refresh() : true;
    } catch (reason) {
      const message = describeError(reason, t("app.confirmationRequiredSuffix"));
      const reconciled = taskRunConflictSummary(reason);
      if (reconciled) mergeTaskRunResult(reconciled);
      if (refreshAfter) await refresh();
      setError(message);
      return false;
    } finally {
      actionGuardRef.current = false;
      setActiveAction(null);
    }
  }

  async function spawnProcess() {
    if (!client || !requireValidQuanta()) return;
    let spawnedPid: string | null = null;
    let submittedLabel = "";
    let spawnedProcess: RuntimeProcess | null = null;
    await safe(async () => {
      const submittedGoal = spawnGoal.trim();
      submittedLabel = taskLabelFromGoal(submittedGoal);
      const authorityManifest = buildGuiTaskAuthorityManifest({
        workingDirectory: spawnWorkingDirectory,
        workspaceAccess: spawnWorkspaceAccess,
        allowGitRequests: spawnAllowGitRequests,
        commandAccess: spawnCommandAccess,
        contextMaintenance: spawnContextMaintenance
      });
      const result = await client.spawn(submittedGoal, spawnImage, maxQuanta, Boolean(snapshot?.scheduler.auto_run), {
        authorityManifest,
        workingDirectory: spawnWorkingDirectory,
        llmProfile: spawnLlmProfile || undefined
      });
      const pid = (result as { pid?: string }).pid;
      if (pid) spawnedPid = pid;
      spawnedProcess = processFromMutationResult(result);
    }, "process.spawn", false);
    const pid = spawnedPid as string | null;
    if (!pid) return;
    if (spawnedProcess) {
      snapshotEpochRef.current.acceptAuthoritativeSnapshot();
      updateSnapshotState((current) => current ? upsertRuntimeProcess(current, spawnedProcess!) : current);
    }
    setTaskLabels((current) => ({ ...current, [pid]: submittedLabel }));
    setSelectedPid(pid);
    setSpawnGoal("");
    setSpawnPanelOpen(false);
  }

  async function startTaskRun() {
    if (!client || !requireValidQuanta()) return;
    const submittedGoal = spawnGoal.trim();
    if (!submittedGoal) return;
    const durableAuthority = buildGuiDurableTaskAuthority({
      workingDirectory: spawnWorkingDirectory,
      workspaceAccess: spawnWorkspaceAccess,
      allowGitRequests: spawnAllowGitRequests,
      commandAccess: spawnCommandAccess,
      contextMaintenance: spawnContextMaintenance
    });
    const authorityManifestId = spawnAuthorityManifestId.trim();
    if (durableAuthority.requiresAuthorityManifest && !authorityManifestId) {
      setError(t("taskRuns.authorityManifestRequired"));
      return;
    }
    const spec: TaskRunSpecV1 = {
      schema_version: 1,
      goal: submittedGoal,
      display_title: taskLabelFromGoal(submittedGoal),
      image_id: spawnImage,
      launch_options: {
        ...(spawnLlmProfile ? { llm_profile_id: spawnLlmProfile } : {}),
        ...(spawnWorkingDirectory.trim() ? { working_directory: spawnWorkingDirectory.trim() } : {}),
        capabilities: durableAuthority.capabilities
      },
      ...(authorityManifestId ? { authority_manifest_id: authorityManifestId } : {}),
      retention: "purge_on_terminal"
    };
    const intent = taskRunStartIntent(
      taskRunStartIntentRef.current,
      spec,
      maxQuanta,
      (kind) => newTaskRunCommandId(kind, "new")
    );
    taskRunStartIntentRef.current = intent;
    let created: TaskRunSummary | null = null;
    let unadmittedConflict = false;
    const succeeded = await safe(async () => {
      try {
        await createAndRunTaskRun(client, spec, intent, maxQuanta, {
          onCreated: (summary) => {
            created = summary;
          },
          onIntent: (bound) => {
            if (taskRunStartIntentRef.current === intent) {
              taskRunStartIntentRef.current = bound;
            }
          },
          onSummary: mergeTaskRunResult
        });
      } catch (error) {
        unadmittedConflict = isUnadmittedTaskRunRevisionConflict(error);
        throw error;
      }
    }, "task_run.create", false);
    const run = created as TaskRunSummary | null;
    if (!succeeded || !run) {
      const current = taskRunStartIntentRef.current;
      if (unadmittedConflict && current?.fingerprint === intent.fingerprint) {
        taskRunStartIntentRef.current = rotateUnadmittedTaskRunStartCommand(
          current,
          () => newTaskRunCommandId("run", "new")
        );
      }
      return;
    }
    if (taskRunStartIntentRef.current?.fingerprint === intent.fingerprint) {
      taskRunStartIntentRef.current = null;
    }
    setSpawnGoal("");
    setSpawnPanelOpen(false);
  }

  function selectTaskRun(runId: string) {
    setSelectedRunId(runId || null);
    const run = snapshot?.task_runs.find((item) => item.run_id === runId);
    const pid = run?.active_pid ?? run?.root_pid;
    setSelectedPid(pid ?? null);
  }

  function mergeTaskRunResult(run: TaskRunSummary) {
    snapshotEpochRef.current.acceptAuthoritativeSnapshot();
    updateSnapshotState((current) => current ? upsertRuntimeTaskRun(current, run) : current);
    setSelectedRunId(run.run_id);
    const pid = run.active_pid ?? run.root_pid;
    setSelectedPid(pid ?? null);
  }

  function durableMutationIntent(
    run: TaskRunSummary,
    action: TaskRunMutationKind,
    request: unknown = {}
  ): TaskRunMutationIntent {
    const key = taskRunMutationIntentKey(run.run_id, action);
    const intent = taskRunMutationIntent(
      taskRunMutationIntentsRef.current.get(key) ?? null,
      {
        runId: run.run_id,
        action,
        expectedRevision: run.revision,
        request
      },
      () => newTaskRunCommandId(action, run.run_id)
    );
    taskRunMutationIntentsRef.current.set(key, intent);
    return intent;
  }

  function clearDurableMutationIntent(intent: TaskRunMutationIntent) {
    const key = taskRunMutationIntentKey(intent.runId, intent.action);
    if (taskRunMutationIntentsRef.current.get(key) === intent) {
      taskRunMutationIntentsRef.current.delete(key);
    }
  }

  function selectLegacyProcess(pid: string) {
    setSelectedRunId(null);
    setSelectedPid(pid || null);
  }

  async function send(kind: "message" | "interrupt"): Promise<boolean> {
    if (view === "user" && client && selectedRun && message.trim()) {
      const run = selectedRun;
      const actions = allowedTaskRunActions(run);
      if (!actions.has("follow_up")) return false;
      let submitted: TaskRunFollowUpIntent | null = null;
      let unadmittedConflict = false;
      const succeeded = await safe(async () => {
        const intent = await taskRunFollowUpIntent(
          taskRunFollowUpIntentsRef.current.get(run.run_id) ?? null,
          {
            runId: run.run_id,
            expectedRevision: run.revision,
            body: message,
            kind: kind === "interrupt" ? "interrupt" : "normal",
            required: true
          },
          () => newTaskRunCommandId("follow-up", run.run_id)
        );
        submitted = intent;
        taskRunFollowUpIntentsRef.current.set(run.run_id, intent);
        try {
          const response = await submitTaskRunFollowUp(client, intent);
          mergeTaskRunResult(response);
        } catch (error) {
          unadmittedConflict = isUnadmittedTaskRunRevisionConflict(error);
          throw error;
        }
        if (taskRunFollowUpIntentsRef.current.get(run.run_id) === intent) {
          taskRunFollowUpIntentsRef.current.delete(run.run_id);
        }
        setMessageDrafts((current) => clearTaskRunFollowUpDraft(current, intent));
      }, "task_run.follow_up");
      const intent = submitted as TaskRunFollowUpIntent | null;
      if (
        !succeeded
        && unadmittedConflict
        && intent
        && taskRunFollowUpIntentsRef.current.get(intent.runId) === intent
      ) {
        taskRunFollowUpIntentsRef.current.delete(intent.runId);
      }
      return succeeded;
    }
    if (!client || !selectedProcess || !message.trim() || !requireValidQuanta()) return false;
    const pid = selectedProcess.pid;
    return safe(async () => {
      const result = await client.sendMessage(pid, message.trim(), kind, Boolean(snapshot?.scheduler.auto_run), maxQuanta);
      mergeProcessResult(result);
      setMessageDrafts((current) => ({ ...current, [pid]: "" }));
    }, `process.${kind}`, false);
  }

  async function runSelectedProcess(): Promise<boolean> {
    if (view === "user" && client && selectedRun && requireValidQuanta()) {
      const run = selectedRun;
      const actions = allowedTaskRunActions(run);
      if (!actions.has("run") && !actions.has("resume")) return false;
      const action = actions.has("resume") ? "resume" : "run";
      const intent = durableMutationIntent(
        run,
        action,
        action === "run" ? { max_quanta: maxQuanta } : {}
      );
      let unadmittedConflict = false;
      const succeeded = await safe(async () => {
        try {
          const response = action === "resume"
            ? await client.resumeTaskRun(intent.runId, intent.expectedRevision, intent.commandId)
            : await client.runTaskRun(intent.runId, intent.expectedRevision, intent.commandId, maxQuanta);
          mergeTaskRunResult(response);
        } catch (error) {
          unadmittedConflict = isUnadmittedTaskRunRevisionConflict(error);
          throw error;
        }
      }, `task_run.${action}`);
      if (succeeded || unadmittedConflict) clearDurableMutationIntent(intent);
      return succeeded;
    }
    if (!client || !selectedProcess || !requireValidQuanta()) return false;
    const pid = selectedProcess.pid;
    return safe(async () => {
      mergeProcessResult(await runOrResumeProcess(client, selectedProcess, maxQuanta));
    }, "process.run", false);
  }

  async function pauseSelectedProcess(): Promise<boolean> {
    if (view === "user" && client && selectedRun && allowedTaskRunActions(selectedRun).has("pause")) {
      const run = selectedRun;
      const intent = durableMutationIntent(run, "pause");
      let unadmittedConflict = false;
      const succeeded = await safe(async () => {
        try {
          mergeTaskRunResult(await client.pauseTaskRun(
            intent.runId,
            intent.expectedRevision,
            intent.commandId
          ));
        } catch (error) {
          unadmittedConflict = isUnadmittedTaskRunRevisionConflict(error);
          throw error;
        }
      }, "task_run.pause");
      if (succeeded || unadmittedConflict) clearDurableMutationIntent(intent);
      return succeeded;
    }
    if (!client || !selectedProcess) return false;
    return safe(async () => {
      mergeProcessResult(await client.pauseProcess(selectedProcess.pid));
    }, "process.pause", false);
  }

  function confirmCancelSelectedRun() {
    if (!client || !selectedRun || !allowedTaskRunActions(selectedRun).has("cancel")) return;
    const run = selectedRun;
    const reason = "cancelled from GUI";
    const intent = durableMutationIntent(run, "cancel", { reason });
    setPendingConfirm({
      title: t("taskRuns.cancelTitle"),
      message: t("taskRuns.cancelMessage"),
      details: { run_id: intent.runId, expected_revision: intent.expectedRevision },
      action: async () => {
        mergeTaskRunResult(await client.cancelTaskRun(
          intent.runId,
          intent.expectedRevision,
          intent.commandId,
          true,
          reason
        ));
        clearDurableMutationIntent(intent);
        setPendingConfirm(null);
      },
      onErrorReconciled: (error) => {
        if (!isUnadmittedTaskRunRevisionConflict(error)) return false;
        clearDurableMutationIntent(intent);
        return true;
      }
    });
  }

  async function resumeSelectedProcess(): Promise<boolean> {
    if (!client || !selectedProcess) return false;
    return safe(async () => {
      mergeProcessResult(await client.resumeProcess(selectedProcess.pid, Boolean(snapshot?.scheduler.auto_run)));
    }, "process.resume", false);
  }

  function mergeProcessResult(result: unknown): RuntimeProcess | null {
    mergeSchedulerResult(result);
    const process = processFromMutationResult(result);
    if (process) {
      snapshotEpochRef.current.acceptAuthoritativeSnapshot();
      updateSnapshotState((current) => current ? upsertRuntimeProcess(current, process) : current);
    }
    return process;
  }

  function mergeSchedulerResult(result: unknown): SchedulerStatus | null {
    const candidate = result && typeof result === "object" && !Array.isArray(result) && "scheduler" in result
      ? (result as { scheduler?: unknown }).scheduler
      : result;
    try {
      assertSchedulerStatus(candidate);
      mergeSchedulerStatus(candidate);
      return candidate;
    } catch {
      return null;
    }
  }

  function mergeSchedulerStatus(status: SchedulerStatus) {
    updateSnapshotState((current) => current ? { ...current, scheduler: status } : current);
  }

  function requireValidQuanta(): boolean {
    if (quantaDraft.valid) return true;
    setError(t("scheduler.invalidQuanta"));
    return false;
  }

  async function respond(request: HumanRequest, response: HumanResponseInput): Promise<HumanResponseOutcome> {
    if (!client || !requireValidQuanta()) return "retryable";
    // A message/interrupt request with auto-run may remain in flight while the
    // scheduler is waiting for this exact Human decision. Do not route Human
    // responses through the global action guard or the GUI deadlocks: the
    // pending run waits for approval while approval is discarded as "busy".
    // HumanRequestCard owns per-request duplicate-submission protection.
    if (ambiguousHumanRequestsRef.current.has(request.request_id)) {
      const pending = await readbackHumanRequest(request.request_id);
      if (pending === null) return "ambiguous";
      ambiguousHumanRequestsRef.current.delete(request.request_id);
      if (!pending) return "settled";
    }
    try {
      setError(null);
      const result = await client.respondHumanRequest(request.request_id, response, Boolean(snapshot?.scheduler.auto_run), maxQuanta);
      mergeProcessResult(result);
      void loadSelectedRunHumanRequests(false);
      return "accepted";
    } catch (reason) {
      setError(describeError(reason, t("app.confirmationRequiredSuffix")));
      const pending = await readbackHumanRequest(request.request_id);
      if (pending === false) return "settled";
      if (pending === true) return "retryable";
      ambiguousHumanRequestsRef.current.add(request.request_id);
      return "ambiguous";
    }
  }

  async function readbackHumanRequest(requestId: string): Promise<boolean | null> {
    const requestClient = client;
    if (!requestClient) return null;
    try {
      const request = await requestClient.getHumanRequest(requestId);
      if (activeClientRef.current !== requestClient) return null;
      updateSnapshotState((current) => current ? {
        ...current,
        human_requests: [request, ...current.human_requests.filter((item) => item.request_id !== request.request_id)]
      } : current);
      setSelectedRunHumanPage((current) => current ? {
        ...current,
        items: request.status === "pending"
          ? dedupeHumanRequests([request, ...current.items])
          : current.items.filter((item) => item.request_id !== request.request_id)
      } : current);
      setLastUpdatedAt(new Date());
      return request.status === "pending";
    } catch {
      return null;
    }
  }

  async function rateProcess(pid: string, score: number, comment: string): Promise<boolean> {
    if (!client) return false;
    return safe(async () => {
      await client.submitAgentRating(pid, score, comment);
    });
  }

  async function createLlmProfile(profile: LLMProfileInput): Promise<boolean> {
    if (!client) return false;
    return safe(async () => {
      await client.createLLMProfile(profile);
    });
  }

  async function updateLlmProfile(profileId: string, profile: LLMProfileInput): Promise<boolean> {
    if (!client) return false;
    return safe(async () => {
      await client.updateLLMProfile(profileId, profile);
    });
  }

  async function deleteLlmProfile(profileId: string): Promise<boolean> {
    if (!client) return false;
    return safe(async () => {
      await client.deleteLLMProfile(profileId);
      if (spawnLlmProfile === profileId) setSpawnLlmProfile("");
      if (execLlmProfile === profileId) setExecLlmProfile("");
    });
  }

  function confirmExec() {
    if (!client || !selectedProcess || !requireValidQuanta()) return;
    const pid = selectedProcess.pid;
    setPendingConfirm({
      title: t("app.exec.title"),
      message: t("app.exec.message"),
      details: { pid, image: execImage, goal: execGoal, llm_profile: execLlmProfile || null, auto_run: snapshot?.scheduler.auto_run, max_quanta: maxQuanta },
      action: async () => {
        await client.execProcess(pid, execImage, execGoal, true, Boolean(snapshot?.scheduler.auto_run), maxQuanta, execLlmProfile || undefined);
        setExecGoalDrafts((current) => ({ ...current, [pid]: "" }));
        setPendingConfirm(null);
        await refresh();
      }
    });
  }

  function confirmExit() {
    if (!client || !selectedProcess) return;
    const pid = selectedProcess.pid;
    setPendingConfirm({
      title: t("app.exit.title"),
      message: t("app.exit.message"),
      details: { pid },
      action: async () => {
        await client.exitProcess(pid, "Exited from GUI", false, true);
        setPendingConfirm(null);
        await refresh();
      }
    });
  }

  async function chooseAndConfirmImageImport(replace = false) {
    if (!client) return;
    try {
      const imagePackage = await window.libosApi?.chooseImagePackage();
      if (!imagePackage) return;
      setPendingConfirm({
        title: t("image.register.title"),
        message: t("image.register.message"),
        details: {
          source: imagePackage.name,
          manifest_sha256: imagePackage.manifest_sha256,
          manifest_bytes: new Blob([imagePackage.manifest]).size,
          files: Object.keys(imagePackage.files).length,
          bytes: new Blob([JSON.stringify(imagePackage.files)]).size,
          replace
        },
        action: async () => {
          const result = await client.registerImagePackage(imagePackage, true, replace);
          setSpawnImage(result.image_id);
          setExecImage(result.image_id);
          setPendingConfirm(null);
          await refresh();
        }
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function confirmCommitImage(request: { imageId: string; name: string; version: string; replace: boolean; checkpointId?: string }) {
    if (!client || !selectedProcess) return;
    const pid = selectedProcess.pid;
    setPendingConfirm({
      title: t("image.commit.title"),
      message: t("image.commit.message"),
      details: {
        pid,
        checkpoint: request.checkpointId ?? t("image.autoCheckpoint"),
        image_id: request.imageId,
        name: request.name,
        version: request.version,
        replace: request.replace
      },
      action: async () => {
        const checkpointId = request.checkpointId
          ?? (await client.createCheckpoint(pid, "GUI image commit")).checkpoint_id;
        const result = await client.commitCheckpointToImage({
          checkpointId,
          imageId: request.imageId,
          name: request.name,
          version: request.version,
          confirmed: true,
          replace: request.replace
        });
        setSpawnImage(result.image_id);
        setExecImage(result.image_id);
        setPendingConfirm(null);
        await refresh();
      }
    });
  }

  async function confirmPendingAction() {
    if (!pendingConfirm || confirmGuardRef.current) return;
    const confirmation = pendingConfirm;
    confirmGuardRef.current = true;
    setConfirmBusy(true);
    setError(null);
    setConfirmError(null);
    try {
      if (refreshInFlightRef.current) await refreshInFlightRef.current;
      await confirmation.action();
    } catch (reason) {
      const message = describeError(reason, t("app.confirmationRequiredSuffix"));
      const reconciled = taskRunConflictSummary(reason);
      if (reconciled) mergeTaskRunResult(reconciled);
      await refresh();
      if (confirmation.onErrorReconciled?.(reason)) {
        setPendingConfirm(null);
        setError(message);
      } else {
        setConfirmError(message);
      }
    } finally {
      confirmGuardRef.current = false;
      setConfirmBusy(false);
    }
  }

  function queueConfirmation(request: ConfirmationRequest) {
    setPendingConfirm({
      ...request,
      action: async () => {
        await request.action();
        setPendingConfirm(null);
        await refresh();
      }
    });
  }

  function setView(next: "user" | "operator") {
    setViewState(next);
    try {
      globalThis.localStorage?.setItem("agent-libos.gui.view", next);
    } catch {
      // Persistence is optional in restricted renderer environments.
    }
    globalThis.setTimeout(() => document.getElementById("primary-workspace")?.focus(), 0);
  }

  async function refreshAndClearError() {
    if (await refresh()) {
      setError(null);
      // An HTTP-level SSE failure ends the stream loop. A snapshot refresh can
      // prove the backend is healthy, but it must also restart the subscription
      // or the renderer remains silently stale after reporting success.
      if (streamStatus === "failed") setStreamEpoch((current) => current + 1);
    }
  }

  if (initializing && !snapshot) {
    return <LoadingScreen error={error} onRetry={() => void initialize()} />;
  }

  // A bounded snapshot refresh may wait behind an in-flight LLM quantum. Keep
  // that passive synchronization visible without disabling message, approval,
  // pause, or interrupt controls that are needed to steer the running process.
  const interactionBusy = Boolean(activeAction);
  const selectedStatusTone = selectedProcess
    ? processStatusTone(selectedProcess.status)
    : "idle";
  const selectedTaskLabel = selectedProcess
    ? taskDisplayLabel(selectedProcess, taskLabels)
    : null;
  const selectedProcessMetadata = selectedProcess
    ? `${shortProcessId(selectedProcess.pid)} · ${selectedProcess.image_id} · ${selectedProcess.llm_profile_id} · ${t("operator.cwd")} ${selectedProcess.working_directory}`
    : null;
  const selectedProcessTerminal = Boolean(
    selectedProcess?.terminal || (selectedProcess && ["exited", "failed", "killed"].includes(selectedProcess.status))
  );
  const selectedPendingRequests = (snapshot?.human_requests ?? []).filter(
    (request) => request.status === "pending" && request.pid === selectedProcess?.pid
  );
  const allPendingRequests = (snapshot?.human_requests ?? []).filter((request) => request.status === "pending");

  function openOperatorPendingRequest() {
    const next = allPendingRequests[0];
    if (!next) return;
    setSelectedPid(next.pid);
    globalThis.setTimeout(() => {
      const region = document.getElementById("operator-pending-requests");
      region?.focus({ preventScroll: true });
      region?.scrollIntoView({ block: "nearest", behavior: globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }, 0);
  }

  function openOperatorSpawn() {
    setSpawnPanelOpen(true);
    globalThis.setTimeout(() => {
      const disclosure = document.querySelector<HTMLElement>(".spawnBox");
      disclosure?.querySelector<HTMLElement>("summary")?.focus({ preventScroll: true });
      disclosure?.scrollIntoView({
        block: "start",
        behavior: globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"
      });
    }, 0);
  }

  function explainOperatorEvidence(kind: string, id: string) {
    if (!selectedProcess) return;
    setExplainLookup({ pid: selectedProcess.pid, kind, id, nonce: Date.now() });
    globalThis.setTimeout(() => {
      const panel = document.querySelector<HTMLElement>(".rightPane .tabPanel");
      panel?.focus({ preventScroll: true });
      panel?.scrollIntoView({
        block: "start",
        behavior: globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"
      });
    }, 0);
  }

  const notices = (
    <AppNotices
      key="runtime-notices"
      error={error}
      snapshot={snapshot}
      streamStatus={streamStatus}
      refreshing={refreshing}
      showSnapshotDiagnostics={view === "operator"}
      onDismissError={() => setError(null)}
      onRetry={() => client ? void refreshAndClearError() : void initialize()}
    />
  );

  return (
    <div
      className={view === "user" ? "userAppShell" : "appShell"}
      data-refreshing={refreshing || undefined}
    >
      {view === "user" ? (
        <UserPage
          notices={notices}
          connection={connection}
          snapshot={snapshot}
          selectedPid={selectedPid}
          selectedProcess={selectedProcess}
          selectedRunId={selectedRunId}
          selectedRun={selectedRun}
          selectedRunDetail={selectedRunDetail}
          taskRunHumanRequests={visibleSelectedRunHumanPage?.items ?? null}
          taskRunHumanHasMore={Boolean(visibleSelectedRunHumanPage?.has_more)}
          taskRunHumanPresentationTruncated={Boolean(visibleSelectedRunHumanPage?.presentation_truncated)}
          taskRunHumanLoading={selectedRunHumanLoading}
          taskRuns={snapshot?.task_runs ?? []}
          taskLabels={taskLabels}
          taskSettings={taskLaunchSettings}
          quantaValid={quantaDraft.valid}
          spawnGoal={spawnGoal}
          message={message}
          images={snapshot?.images ?? []}
          llmProfiles={snapshot?.llm_profiles ?? []}
          onSelectPid={selectLegacyProcess}
          onSelectRun={selectTaskRun}
          onLoadMoreTaskRunHumanRequests={() => void loadSelectedRunHumanRequests(true)}
          onMaxQuantaChange={setMaxQuantaInput}
          onSpawnGoalChange={setSpawnGoal}
          onSpawnImageChange={setSpawnImage}
          onApplyTaskSettings={applyTaskLaunchSettings}
          onMessageChange={setMessage}
          onSpawn={() => void startTaskRun()}
          onImportImage={() => void chooseAndConfirmImageImport(false)}
          onCommitImage={confirmCommitImage}
          onSend={(kind) => void send(kind)}
          onRespond={respond}
          onRate={rateProcess}
          onCreateLlmProfile={createLlmProfile}
          onUpdateLlmProfile={updateLlmProfile}
          onDeleteLlmProfile={deleteLlmProfile}
          onRun={() => void runSelectedProcess()}
          onPause={() => void pauseSelectedProcess()}
          onRefresh={() => void refreshAndClearError()}
          onOpenDb={() => void openDatabase()}
          onShowOperator={() => setView("operator")}
          onStop={selectedRun ? confirmCancelSelectedRun : confirmExit}
          busy={interactionBusy}
          streamStatus={streamStatus}
          lastUpdatedAt={lastUpdatedAt}
        />
      ) : (
        <>
          <a className="skipLink" href="#primary-workspace">{t("user.skipToWorkspace")}</a>
          <TopBar
            db={connection?.db ?? t("app.defaultDb")}
            scheduler={snapshot?.scheduler ?? null}
            maxQuantaInput={maxQuantaInput}
            quantaValid={quantaDraft.valid}
            selectedPid={selectedProcess && !selectedProcessTerminal ? selectedProcess.pid : null}
            onMaxQuantaChange={setMaxQuantaInput}
            onOpenDb={() => void openDatabase()}
            onSpawn={openOperatorSpawn}
            onRun={() => void runSelectedProcess()}
            onStep={() => selectedProcess && client && quantaDraft.valid && void safe(async () => { mergeProcessResult(await client.step(selectedProcess.pid)); }, "process.step", false)}
            onPause={() => client && void safe(async () => { mergeSchedulerStatus(await client.pauseScheduler()); }, "scheduler.pause", false)}
            onAutoRunChange={(value) => client && void safe(async () => { mergeSchedulerStatus(await client.setAutoRun(value)); }, "scheduler.auto", false)}
            onRefresh={() => void refreshAndClearError()}
            pendingHumanCount={allPendingRequests.length}
            onOpenPending={openOperatorPendingRequest}
            onShowUser={() => setView("user")}
            busy={interactionBusy}
            streamStatus={streamStatus}
            lastUpdatedAt={lastUpdatedAt}
          />
          {notices}

          <main className="workspace" id="primary-workspace" tabIndex={-1}>
            <section className="leftPane operatorPane" aria-label={t("operator.processes.title")}>
              <header className="paneHeader operatorPaneHeader">
                <div className="paneTitle">
                  <span className="eyebrow">{t("operator.processes.title")}</span>
                  <h1>{t("operator.processes.title")}</h1>
                  <p>{t("operator.processes.subtitle")}</p>
                </div>
                <span className="countPill">{snapshot?.processes.length ?? 0}</span>
              </header>
              <details
                className="spawnBox operatorDisclosure"
                open={spawnPanelOpen}
                onToggle={(event) => setSpawnPanelOpen(event.currentTarget.open)}
              >
                <summary>
                  <span className="operatorDisclosureIcon"><CirclePlus size={15} /></span>
                  <span><strong>{t("operator.launchConfiguration")}</strong><small>{t("operator.launchConfigurationHint")}</small></span>
                  <ChevronDown size={15} className="disclosureChevron" />
                </summary>
                <div className="operatorDisclosureBody">
                  <ImageSelect images={snapshot?.images ?? []} value={spawnImage} label={t("operator.spawnImage")} disabled={interactionBusy} onChange={setSpawnImage} />
                  <LLMProfileSelect
                    profiles={snapshot?.llm_profiles ?? []}
                    value={spawnLlmProfile}
                    label={t("llmProfile.spawnLabel")}
                    disabled={interactionBusy}
                    onChange={setSpawnLlmProfile}
                    onCreate={createLlmProfile}
                    onUpdate={updateLlmProfile}
                    onDelete={deleteLlmProfile}
                  />
                  <label className="fieldStack">
                    <span>{t("operator.initialCwd")}</span>
                    <input
                      value={spawnWorkingDirectory}
                      disabled={interactionBusy}
                      onChange={(event) => setSpawnWorkingDirectory(event.currentTarget.value)}
                      placeholder={t("operator.initialCwdPlaceholder")}
                    />
                  </label>
                  <label className="taskAuthorityField">
                    <span>{t("taskAuthority.workspaceAccess")}</span>
                    <select
                      value={spawnWorkspaceAccess}
                      disabled={interactionBusy}
                      onChange={(event) => setSpawnWorkspaceAccess(event.currentTarget.value as WorkspaceAccess)}
                    >
                      <option value="none">{t("taskAuthority.none")}</option>
                      <option value="read">{t("taskAuthority.read")}</option>
                      <option value="edit">{t("taskAuthority.edit")}</option>
                      <option value="manage">{t("taskAuthority.manage")}</option>
                    </select>
                  </label>
                  <label className="fieldStack">
                    <span>{t("taskAuthority.manifestId")}</span>
                    <input
                      value={spawnAuthorityManifestId}
                      disabled={interactionBusy}
                      onChange={(event) => setSpawnAuthorityManifestId(event.currentTarget.value)}
                      placeholder={t("taskAuthority.manifestIdPlaceholder")}
                    />
                    <small className="fieldHint">{t("taskAuthority.manifestIdHint")}</small>
                  </label>
                  <label className="taskAuthorityToggle operatorAuthorityToggle">
                    <input
                      type="checkbox"
                      checked={spawnContextMaintenance}
                      disabled={interactionBusy}
                      onChange={(event) => setSpawnContextMaintenance(event.currentTarget.checked)}
                    />
                    <span>{t("taskAuthority.contextMaintenance")}</span>
                  </label>
                  <label className="taskAuthorityToggle operatorAuthorityToggle">
                    <input
                      type="checkbox"
                      checked={spawnAllowGitRequests}
                      disabled={interactionBusy}
                      onChange={(event) => setSpawnAllowGitRequests(event.currentTarget.checked)}
                    />
                    <span>{t("taskAuthority.git")}</span>
                  </label>
                  <label className="taskAuthorityField">
                    <span>{t("taskAuthority.commandAccess")}</span>
                    <select
                      value={spawnCommandAccess}
                      disabled={interactionBusy}
                      onChange={(event) => setSpawnCommandAccess(event.currentTarget.value as CommandAccess)}
                    >
                      <option value="none">{t("taskAuthority.commandNone")}</option>
                      <option value="reviewed">{t("taskAuthority.commandReviewed")}</option>
                    </select>
                  </label>
                  <label className="fieldStack">
                    <span>{t("operator.spawnGoal")}</span>
                    <textarea value={spawnGoal} disabled={interactionBusy} onChange={(event) => setSpawnGoal(event.currentTarget.value)} />
                  </label>
                  <p className="taskAuthorityHint">{t("taskAuthority.hint")}</p>
                  <button type="button" className="primary launchProcessButton" disabled={interactionBusy || !quantaDraft.valid || !spawnGoal.trim()} onClick={() => void spawnProcess()}>
                    <CirclePlus size={15} />{t("operator.launchProcess")}
                  </button>
                </div>
              </details>
              <ProcessTree
                processes={snapshot?.processes ?? []}
                selectedPid={selectedPid}
                taskLabels={taskLabels}
                humanRequests={snapshot?.human_requests ?? []}
                disabled={interactionBusy}
                onSelect={setSelectedPid}
              />
            </section>

            <section className="centerPane operatorPane" aria-label={t("operator.activity")}>
              <header className="paneHeader activityPaneHeader">
                <div className="paneTitle">
                  <span className="eyebrow"><Activity size={12} />{t("operator.activity")}</span>
                  <h1 title={selectedTaskLabel ?? selectedProcess?.pid}>{selectedTaskLabel ?? t("operator.noProcessSelected")}</h1>
                  {selectedProcessMetadata ? <p title={selectedProcessMetadata}>{selectedProcessMetadata}</p> : <p>{t("operator.activityHint")}</p>}
                </div>
                <div className="paneHeaderMeta">
                  {selectedProcess?.interrupt_count ? <span className="interruptBanner"><AlertTriangle size={15} />{t("operator.interruptPending")}</span> : null}
                  {selectedProcess ? <span className={`statusPill ${selectedStatusTone}`}><span className="statusDot" />{processStatusLabel(selectedProcess.status, t)}</span> : null}
                </div>
              </header>

              {selectedPendingRequests.length ? <section id="operator-pending-requests" className="humanRequests" aria-label={t("user.pendingRequests")} tabIndex={-1}>
                <div className="pendingRequestsHeading"><AlertTriangle size={15} /><strong>{t("user.pendingRequests")}</strong><span>{selectedPendingRequests.length}</span></div>
                {selectedPendingRequests.map((request) => (
                  <HumanRequestCard key={request.request_id} request={request} onRespond={respond} />
                ))}
              </section> : null}

              <Timeline
                pid={selectedProcess?.pid ?? null}
                messages={selectedProcess?.messages ?? []}
                humanRequests={snapshot?.human_requests ?? []}
                llmCalls={snapshot?.llm_calls ?? []}
                events={snapshot?.events ?? []}
                audit={snapshot?.audit ?? []}
                onExplainEvidence={explainOperatorEvidence}
              />

              <div className="composer">
                <div className="composerField operatorComposerField">
                  <textarea
                    rows={1}
                    value={message}
                    disabled={!selectedProcess || selectedProcessTerminal || interactionBusy}
                    aria-label={t("operator.messagePlaceholder")}
                    onChange={(event) => setMessage(event.currentTarget.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                        event.preventDefault();
                        if (quantaDraft.valid && message.trim()) void send("message");
                      }
                    }}
                    placeholder={t("operator.messagePlaceholder")}
                  />
                  <span>{t("user.sendHint")}</span>
                </div>
                <button type="button" className="primary" disabled={interactionBusy || !quantaDraft.valid || !selectedProcess || selectedProcessTerminal || !message.trim()} onClick={() => void send("message")}><Send size={15} />{t("operator.message")}</button>
                <button type="button" disabled={interactionBusy || !quantaDraft.valid || !selectedProcess || selectedProcessTerminal || !message.trim()} className="iconOnly warning" aria-label={t("operator.interrupt")} title={t("operator.interrupt")} onClick={() => void send("interrupt")}><AlertTriangle size={15} /></button>
              </div>
            </section>

            <section className="rightPane operatorPane" aria-label={t("operator.inspector")}>
              <header className="inspectorHeader">
                <div className="paneTitle">
                  <span className="eyebrow"><PanelRight size={12} />{t("operator.inspector")}</span>
                  <h2>{t("operator.inspector")}</h2>
                  <p title={selectedTaskLabel ?? selectedProcess?.pid}>{selectedTaskLabel ?? t("operator.noSelectionHint")}</p>
                </div>
              </header>
              <details className="quickActions operatorDisclosure">
                <summary>
                  <span className="operatorDisclosureIcon"><Settings size={15} /></span>
                  <span><strong>{t("operator.processControls")}</strong><small>{t("operator.processControlsHint")}</small></span>
                  <ChevronDown size={15} className="disclosureChevron" />
                </summary>
                <div className="operatorDisclosureBody processActionBody">
                  <section className="processActionGroup">
                    <span className="actionGroupLabel">{t("operator.workingDirectory")}</span>
                    <div className="inlineAction">
                      <input
                        value={cwd}
                        disabled={interactionBusy || !selectedProcess}
                        aria-label={t("operator.newCwdPlaceholder")}
                        placeholder={t("operator.newCwdPlaceholder")}
                        onChange={(event) => setCwd(event.currentTarget.value)}
                      />
                      <button type="button" disabled={interactionBusy || !client || !selectedProcess || !cwd.trim()} onClick={() => selectedProcess && void safe(async () => {
                        await client!.changeDirectory(selectedProcess.pid, cwd.trim());
                        setCwd("");
                      }, "process.cd")}>cd</button>
                    </div>
                  </section>
                  <section className="processActionGroup">
                    <span className="actionGroupLabel">{t("operator.exec")}</span>
                    <ImageSelect images={snapshot?.images ?? []} value={execImage} label={t("operator.exec")} disabled={interactionBusy || !selectedProcess} onChange={setExecImage} />
                    <LLMProfileSelect
                      profiles={snapshot?.llm_profiles ?? []}
                      value={execLlmProfile}
                      label={t("llmProfile.execLabel")}
                      disabled={interactionBusy || !selectedProcess}
                      onChange={setExecLlmProfile}
                      onCreate={createLlmProfile}
                      onUpdate={updateLlmProfile}
                      onDelete={deleteLlmProfile}
                    />
                    <input value={execGoal} disabled={interactionBusy || !selectedProcess} onChange={(event) => setExecGoal(event.currentTarget.value)} aria-label={t("operator.spawnGoal")} placeholder={t("operator.execGoalPlaceholder")} />
                    <button type="button" disabled={interactionBusy || !quantaDraft.valid || !selectedProcess} className="warning fullWidthButton" onClick={confirmExec}>{t("operator.exec")}</button>
                  </section>
                  <section className="processActionGroup">
                    <span className="actionGroupLabel">{t("operator.lifecycle")}</span>
                    <div className="lifecycleActions">
                      <button
                        type="button"
                        disabled={interactionBusy || !selectedProcess || selectedProcess.terminal || selectedProcess.status === "paused"}
                        onClick={() => void pauseSelectedProcess()}
                      >{t("operator.pauseProcess")}</button>
                      <button
                        type="button"
                        disabled={interactionBusy || !selectedProcess || selectedProcess.status !== "paused"}
                        onClick={() => void resumeSelectedProcess()}
                      >{t("operator.resumeProcess")}</button>
                    </div>
                  </section>
                  <section className="dangerZone">
                    <div><strong>{t("operator.dangerZone")}</strong><span>{t("operator.dangerZoneHint")}</span></div>
                    <button type="button" disabled={interactionBusy || !selectedProcess} className="danger" onClick={confirmExit}>{t("operator.exit")}</button>
                  </section>
                </div>
              </details>
              <DetailTabs
                process={selectedProcess}
                snapshot={snapshot}
                onImportImage={(replace) => void chooseAndConfirmImageImport(replace)}
                onCommitImage={confirmCommitImage}
                onUseImageForSpawn={setSpawnImage}
                onUseImageForExec={setExecImage}
                onRate={rateProcess}
                onInspectImage={(imageId) => {
                  if (!client) throw new Error(t("app.clientUnavailable"));
                  return client.inspectImage(imageId);
                }}
                onListOperations={(pid, cursor, signal) => {
                  if (!client) throw new Error(t("app.clientUnavailable"));
                  return client.listOperations(pid, 100, cursor, { signal, timeoutMs: 15_000 });
                }}
                onExplainOperation={(operationId, cursor, signal) => {
                  if (!client) throw new Error(t("app.clientUnavailable"));
                  return client.explainOperation(operationId, 200, cursor, { signal, timeoutMs: 15_000 });
                }}
                onResolveOperation={(kind, id, signal) => {
                  if (!client) throw new Error(t("app.clientUnavailable"));
                  return client.resolveOperation(kind, id, { signal, timeoutMs: 15_000 });
                }}
                explainLookup={selectedExplainLookup}
                connectionEpoch={connectionEpoch}
                client={client}
                runAction={safe}
                confirmAction={queueConfirmation}
                busy={interactionBusy}
              />
            </section>
          </main>
        </>
      )}

      {pendingConfirm ? (
        <ConfirmDialog
          title={pendingConfirm.title}
          message={pendingConfirm.message}
          details={pendingConfirm.details}
          busy={confirmBusy}
          error={confirmError}
          onCancel={() => {
            setPendingConfirm(null);
            setConfirmError(null);
          }}
          onConfirm={() => void confirmPendingAction()}
        />
      ) : null}
    </div>
  );
}

function readStoredView(): "user" | "operator" {
  try {
    return globalThis.localStorage?.getItem("agent-libos.gui.view") === "operator" ? "operator" : "user";
  } catch {
    return "user";
  }
}

function readStoredTaskLabels(): Record<string, string> {
  try {
    return taskLabelsFromStorage(globalThis.sessionStorage?.getItem(TASK_LABELS_STORAGE_KEY) ?? null);
  } catch {
    return {};
  }
}

function readStoredSelectedPid(): string | null {
  try {
    const pid = globalThis.sessionStorage?.getItem(SELECTED_PID_STORAGE_KEY) ?? "";
    return /^pid_[A-Za-z0-9_-]{1,156}$/.test(pid) ? pid : null;
  } catch {
    return null;
  }
}

function readStoredSelectedRunId(): string | null {
  try {
    const runId = globalThis.sessionStorage?.getItem(SELECTED_RUN_STORAGE_KEY) ?? "";
    return /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$/.test(runId) ? runId : null;
  } catch {
    return null;
  }
}

function newTaskRunCommandId(action: string, runId: string): string {
  const random = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `gui:${action}:${runId}:${random}`;
}

function taskRunMutationIntentKey(runId: string, action: TaskRunMutationKind): string {
  return `${runId}\u0000${action}`;
}

function dedupeHumanRequests(requests: readonly HumanRequest[]): HumanRequest[] {
  const selected = new Map<string, HumanRequest>();
  for (const request of requests) selected.set(request.request_id, request);
  return [...selected.values()];
}

function sameConnection(left: GuiConnection, right: GuiConnection): boolean {
  return left.url === right.url && left.token === right.token && left.db === right.db;
}

function describeError(reason: unknown, confirmationSuffix: string): string {
  const err = reason && typeof reason === "object"
    ? reason as Error & { payload?: { error?: { confirmation_required?: boolean } } }
    : null;
  const message = reason instanceof Error
    ? reason.message
    : typeof reason === "string"
      ? reason
      : String(reason ?? "Unknown GUI error");
  if (err?.payload?.error?.confirmation_required) {
    return `${message}. ${confirmationSuffix}`;
  }
  return message;
}
