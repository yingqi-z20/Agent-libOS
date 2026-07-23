import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Send } from "lucide-react";
import { LibOSClient } from "./api/client";
import { runtimeSnapshotFromSseData } from "./api/types";
import type { GuiConnection, HumanRequest, HumanResponseInput, RuntimeSnapshot, StreamConnectionStatus } from "./api/types";
import { AppNotices, LoadingScreen } from "./components/AppNotices";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { DetailTabs } from "./components/DetailTabs";
import { ImageSelect } from "./components/ImageSelect";
import { HumanRequestCard } from "./components/HumanRequestCard";
import { LLMProfileSelect } from "./components/LLMProfileSelect";
import { ProcessTree } from "./components/ProcessTree";
import { Timeline } from "./components/Timeline";
import { TopBar } from "./components/TopBar";
import { UserPage } from "./components/UserPage";
import { previewImageManifest } from "./imagePreview";
import { useI18n } from "./i18n";
import type { OptionalQuanta } from "./quanta";
import { reconcileSelectedPid } from "./selection";
import type { LLMProfileInput } from "./api/types";
import type { ConfirmationRequest } from "./adminTypes";
import { developmentConnection } from "./developmentConnection";
import { buildGuiTaskAuthorityManifest, type WorkspaceAccess } from "./taskAuthority";

type PendingConfirm = ConfirmationRequest;

export function App() {
  const { t } = useI18n();
  const [view, setViewState] = useState<"user" | "operator">(() => readStoredView());
  const [connection, setConnection] = useState<GuiConnection | null>(null);
  const [client, setClient] = useState<LibOSClient | null>(null);
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [selectedPid, setSelectedPid] = useState<string | null>(null);
  const [maxQuanta, setMaxQuanta] = useState<OptionalQuanta>(null);
  const [spawnGoal, setSpawnGoal] = useState("Summarize the current project state.");
  const [spawnImage, setSpawnImage] = useState("coding-agent:v0");
  const [spawnLlmProfile, setSpawnLlmProfile] = useState("");
  const [spawnWorkingDirectory, setSpawnWorkingDirectory] = useState("");
  const [spawnWorkspaceAccess, setSpawnWorkspaceAccess] = useState<WorkspaceAccess>("edit");
  const [spawnAllowGitRequests, setSpawnAllowGitRequests] = useState(true);
  const [messageDrafts, setMessageDrafts] = useState<Record<string, string>>({});
  const [cwdDrafts, setCwdDrafts] = useState<Record<string, string>>({});
  const [execImage, setExecImage] = useState("base-agent:v0");
  const [execLlmProfile, setExecLlmProfile] = useState("");
  const [execGoalDrafts, setExecGoalDrafts] = useState<Record<string, string>>({});
  const [pendingConfirm, setPendingConfirm] = useState<PendingConfirm | null>(null);
  const [confirmBusy, setConfirmBusy] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeAction, setActiveAction] = useState<string | null>(null);
  const [streamStatus, setStreamStatus] = useState<StreamConnectionStatus>("connecting");
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explainLookup, setExplainLookup] = useState<{ pid: string; kind: string; id: string; nonce: number } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const activeClientRef = useRef<LibOSClient | null>(null);
  const initializationInFlightRef = useRef<Promise<void> | null>(null);
  const refreshInFlightRef = useRef<Promise<boolean> | null>(null);
  const actionGuardRef = useRef(false);
  const confirmGuardRef = useRef(false);

  useEffect(() => {
    void initialize();
    return () => abortRef.current?.abort();
  }, []);

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
          setSnapshot(next);
          setSelectedPid((current) => reconcileSelectedPid(next, current));
          setLastUpdatedAt(new Date());
        } catch (reason) {
          setError(describeError(reason, t("app.confirmationRequiredSuffix")));
        }
      }
      if (message.event === "snapshot_truncated" || message.event === "event.invalidated") {
        void refresh();
      }
    }, controller.signal, "0", (status) => {
      if (activeClientRef.current === streamClient) setStreamStatus(status);
    }).catch((reason) => {
      if (!controller.signal.aborted && activeClientRef.current === streamClient) {
        setError(describeError(reason, t("app.confirmationRequiredSuffix")));
      }
    });
    return () => controller.abort();
  }, [client]);

  const selectedProcess = useMemo(
    () => snapshot?.processes.find((process) => process.pid === selectedPid) ?? null,
    [snapshot, selectedPid]
  );
  const message = selectedPid ? messageDrafts[selectedPid] ?? "" : "";
  const cwd = selectedPid ? cwdDrafts[selectedPid] ?? "" : "";
  const execGoal = selectedPid ? execGoalDrafts[selectedPid] ?? "" : "";
  const selectedExplainLookup = explainLookup?.pid === selectedPid ? explainLookup : null;

  function setMessage(value: string) {
    if (!selectedPid) return;
    setMessageDrafts((current) => ({ ...current, [selectedPid]: value }));
  }

  function setCwd(value: string) {
    if (!selectedPid) return;
    setCwdDrafts((current) => ({ ...current, [selectedPid]: value }));
  }

  function setExecGoal(value: string) {
    if (!selectedPid) return;
    setExecGoalDrafts((current) => ({ ...current, [selectedPid]: value }));
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
      const nextSnapshot = await nextClient.snapshot();
      activeClientRef.current = nextClient;
      setConnection(conn);
      setClient(nextClient);
      setSnapshot(nextSnapshot);
      setSelectedPid(reconcileSelectedPid(nextSnapshot, null));
      setMaxQuanta(nextSnapshot.scheduler.default_max_quanta ?? null);
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
    setRefreshing(true);
    let request: Promise<boolean>;
    request = requestClient.snapshot().then((next) => {
      if (activeClientRef.current !== requestClient) return false;
      setSnapshot(next);
      setSelectedPid((current) => reconcileSelectedPid(next, current));
      setLastUpdatedAt(new Date());
      return true;
    }).catch((reason) => {
      if (activeClientRef.current === requestClient) {
        setError(describeError(reason, t("app.confirmationRequiredSuffix")));
      }
      return false;
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
    const nextSnapshot = await nextClient.snapshot();
    abortRef.current?.abort();
    activeClientRef.current = nextClient;
    refreshInFlightRef.current = null;
    setRefreshing(false);
    setConnection(next);
    setClient(nextClient);
    setSnapshot(nextSnapshot);
    setSelectedPid(reconcileSelectedPid(nextSnapshot, null, { preserveExisting: false }));
    setMaxQuanta(nextSnapshot.scheduler.default_max_quanta ?? null);
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

  async function safe(action: () => Promise<void>, label = "action"): Promise<boolean> {
    if (actionGuardRef.current) return false;
    actionGuardRef.current = true;
    setActiveAction(label);
    try {
      setError(null);
      if (refreshInFlightRef.current) await refreshInFlightRef.current;
      await action();
      return await refresh();
    } catch (reason) {
      setError(describeError(reason, t("app.confirmationRequiredSuffix")));
      return false;
    } finally {
      actionGuardRef.current = false;
      setActiveAction(null);
    }
  }

  async function spawnProcess() {
    if (!client) return;
    await safe(async () => {
      const authorityManifest = buildGuiTaskAuthorityManifest({
        workingDirectory: spawnWorkingDirectory,
        workspaceAccess: spawnWorkspaceAccess,
        allowGitRequests: spawnAllowGitRequests
      });
      const result = await client.spawn(spawnGoal, spawnImage, maxQuanta, Boolean(snapshot?.scheduler.auto_run), {
        authorityManifest,
        workingDirectory: spawnWorkingDirectory,
        llmProfile: spawnLlmProfile || undefined
      });
      const pid = (result as { pid?: string }).pid;
      if (pid) setSelectedPid(pid);
    });
  }

  async function send(kind: "message" | "interrupt"): Promise<boolean> {
    if (!client || !selectedProcess || !message.trim()) return false;
    const pid = selectedProcess.pid;
    return safe(async () => {
      await client.sendMessage(pid, message.trim(), kind, Boolean(snapshot?.scheduler.auto_run), maxQuanta);
      setMessageDrafts((current) => ({ ...current, [pid]: "" }));
    });
  }

  async function respond(request: HumanRequest, response: HumanResponseInput): Promise<boolean> {
    if (!client) return false;
    return safe(async () => {
      await client.respondHumanRequest(request.request_id, response, Boolean(snapshot?.scheduler.auto_run), maxQuanta);
    });
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
    if (!client || !selectedProcess) return;
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
      const preview = previewImageManifest(imagePackage.manifest);
      setPendingConfirm({
        title: t("image.register.title"),
        message: t("image.register.message"),
        details: {
          source: imagePackage.name,
          image_id: preview.image_id,
          name: preview.name,
          version: preview.version,
          default_tools_count: preview.default_tools_count,
          required_capabilities_count: preview.required_capabilities_count,
          required_modules_count: preview.required_modules_count,
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
    confirmGuardRef.current = true;
    setConfirmBusy(true);
    setError(null);
    try {
      if (refreshInFlightRef.current) await refreshInFlightRef.current;
      await pendingConfirm.action();
    } catch (reason) {
      setError(describeError(reason, t("app.confirmationRequiredSuffix")));
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
  }

  async function refreshAndClearError() {
    if (await refresh()) setError(null);
  }

  if (initializing && !snapshot) {
    return <LoadingScreen error={error} onRetry={() => void initialize()} />;
  }

  const interactionBusy = Boolean(activeAction) || refreshing;

  return (
    <div
      className={view === "user" ? "userAppShell" : "appShell"}
      aria-busy={interactionBusy || undefined}
    >
      <AppNotices
        error={error}
        snapshot={snapshot}
        streamStatus={streamStatus}
        refreshing={refreshing}
        onDismissError={() => setError(null)}
        onRetry={() => client ? void refreshAndClearError() : void initialize()}
      />
      {view === "user" ? (
        <UserPage
          key={selectedPid ?? "no-process"}
          connection={connection}
          snapshot={snapshot}
          selectedPid={selectedPid}
          selectedProcess={selectedProcess}
          maxQuanta={maxQuanta}
          spawnGoal={spawnGoal}
          spawnImage={spawnImage}
          spawnLlmProfile={spawnLlmProfile}
          spawnWorkingDirectory={spawnWorkingDirectory}
          spawnWorkspaceAccess={spawnWorkspaceAccess}
          spawnAllowGitRequests={spawnAllowGitRequests}
          message={message}
          images={snapshot?.images ?? []}
          llmProfiles={snapshot?.llm_profiles ?? []}
          onSelectPid={setSelectedPid}
          onMaxQuantaChange={setMaxQuanta}
          onSpawnGoalChange={setSpawnGoal}
          onSpawnImageChange={setSpawnImage}
          onSpawnLlmProfileChange={setSpawnLlmProfile}
          onSpawnWorkingDirectoryChange={setSpawnWorkingDirectory}
          onSpawnWorkspaceAccessChange={setSpawnWorkspaceAccess}
          onSpawnAllowGitRequestsChange={setSpawnAllowGitRequests}
          onMessageChange={setMessage}
          onSpawn={() => void spawnProcess()}
          onImportImage={() => void chooseAndConfirmImageImport(false)}
          onCommitImage={confirmCommitImage}
          onSend={(kind) => void send(kind)}
          onRespond={respond}
          onRate={rateProcess}
          onCreateLlmProfile={createLlmProfile}
          onUpdateLlmProfile={updateLlmProfile}
          onDeleteLlmProfile={deleteLlmProfile}
          onRun={() => selectedProcess && client && void safe(() => client.run(selectedProcess.pid, maxQuanta).then(() => undefined))}
          onPause={() => client && void safe(() => client.pauseScheduler().then(() => undefined))}
          onRefresh={() => void refreshAndClearError()}
          onOpenDb={() => void openDatabase()}
          onShowOperator={() => setView("operator")}
          onStop={confirmExit}
          busy={interactionBusy}
          streamStatus={streamStatus}
          lastUpdatedAt={lastUpdatedAt}
        />
      ) : (
        <>
          <TopBar
            db={connection?.db ?? t("app.defaultDb")}
            scheduler={snapshot?.scheduler ?? null}
            maxQuanta={maxQuanta}
            selectedPid={selectedProcess?.pid ?? null}
            onMaxQuantaChange={setMaxQuanta}
            onOpenDb={() => void openDatabase()}
            onSpawn={() => void spawnProcess()}
            onRun={() => selectedProcess && client && void safe(() => client.run(selectedProcess.pid, maxQuanta).then(() => undefined))}
            onStep={() => selectedProcess && client && void safe(() => client.step(selectedProcess.pid).then(() => undefined))}
            onPause={() => client && void safe(() => client.pauseScheduler().then(() => undefined))}
            onAutoRunChange={(value) => client && void safe(() => client.setAutoRun(value).then(() => undefined))}
            onRefresh={() => void refreshAndClearError()}
            onShowUser={() => setView("user")}
            busy={interactionBusy}
            streamStatus={streamStatus}
            lastUpdatedAt={lastUpdatedAt}
          />

          <main className="workspace">
            <section className="leftPane">
              <div className="paneHeader">
                <h1>{t("operator.processes.title")}</h1>
                <span>{snapshot?.processes.length ?? 0}</span>
              </div>
              <div className="spawnBox">
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
                <input
                  value={spawnWorkingDirectory}
                  disabled={interactionBusy}
                  onChange={(event) => setSpawnWorkingDirectory(event.currentTarget.value)}
                  placeholder={t("operator.initialCwdPlaceholder")}
                  aria-label={t("operator.initialCwd")}
                />
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
                <label className="taskAuthorityToggle">
                  <input
                    type="checkbox"
                    checked={spawnAllowGitRequests}
                    disabled={interactionBusy}
                    onChange={(event) => setSpawnAllowGitRequests(event.currentTarget.checked)}
                  />
                  <span>{t("taskAuthority.git")}</span>
                </label>
                <p className="taskAuthorityHint">{t("taskAuthority.hint")}</p>
                <textarea value={spawnGoal} disabled={interactionBusy} onChange={(event) => setSpawnGoal(event.currentTarget.value)} aria-label={t("operator.spawnGoal")} />
              </div>
              <ProcessTree processes={snapshot?.processes ?? []} selectedPid={selectedPid} disabled={interactionBusy} onSelect={setSelectedPid} />
            </section>

            <section className="centerPane">
              <div className="paneHeader">
                <div>
                  <h1>{selectedProcess?.pid ?? t("operator.noProcessSelected")}</h1>
                  {selectedProcess ? <span>{selectedProcess.image_id} · {selectedProcess.status} · {selectedProcess.llm_profile_id} · {t("operator.cwd")} {selectedProcess.working_directory}</span> : null}
                </div>
                {selectedProcess?.interrupt_count ? <span className="interruptBanner"><AlertTriangle size={16} /> {t("operator.interruptPending")}</span> : null}
              </div>

              <div className="humanRequests">
                {(snapshot?.human_requests ?? []).filter((request) => request.status === "pending" && request.pid === selectedProcess?.pid).map((request) => (
                  <HumanRequestCard key={request.request_id} request={request} onRespond={respond} />
                ))}
              </div>

              <Timeline
                pid={selectedProcess?.pid ?? null}
                messages={selectedProcess?.messages ?? []}
                humanRequests={snapshot?.human_requests ?? []}
                llmCalls={snapshot?.llm_calls ?? []}
                events={snapshot?.events ?? []}
                audit={snapshot?.audit ?? []}
                onExplainEvidence={(kind, id) => selectedProcess && setExplainLookup({ pid: selectedProcess.pid, kind, id, nonce: Date.now() })}
              />

              <div className="composer">
                <input
                  value={message}
                  disabled={!selectedProcess || interactionBusy}
                  aria-label={t("operator.messagePlaceholder")}
                  onChange={(event) => setMessage(event.currentTarget.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.nativeEvent.isComposing && message.trim()) void send("message");
                  }}
                  placeholder={t("operator.messagePlaceholder")}
                />
                <button disabled={interactionBusy || !selectedProcess || !message.trim()} onClick={() => void send("message")}><Send size={16} />{t("operator.message")}</button>
                <button disabled={interactionBusy || !selectedProcess || !message.trim()} className="warning" onClick={() => void send("interrupt")}>{t("operator.interrupt")}</button>
              </div>
            </section>

            <section className="rightPane">
              <div className="quickActions">
                <input
                  value={cwd}
                  disabled={interactionBusy}
                  aria-label={t("operator.newCwdPlaceholder")}
                  placeholder={t("operator.newCwdPlaceholder")}
                  onChange={(event) => setCwd(event.currentTarget.value)}
                />
                <button disabled={interactionBusy || !client || !selectedProcess || !cwd.trim()} onClick={() => selectedProcess && void safe(async () => {
                  await client!.changeDirectory(selectedProcess.pid, cwd.trim());
                  setCwd("");
                }, "process.cd")}>cd</button>
                <ImageSelect images={snapshot?.images ?? []} value={execImage} label={t("operator.exec")} disabled={interactionBusy} onChange={setExecImage} />
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
                <input value={execGoal} disabled={interactionBusy} onChange={(event) => setExecGoal(event.currentTarget.value)} aria-label={t("operator.spawnGoal")} />
                <button disabled={interactionBusy || !selectedProcess} className="warning" onClick={confirmExec}>{t("operator.exec")}</button>
                <button
                  disabled={interactionBusy || !selectedProcess || selectedProcess.terminal || selectedProcess.status === "paused"}
                  onClick={() => selectedProcess && client && void safe(() => client.pauseProcess(selectedProcess.pid).then(() => undefined), "process.pause")}
                >{t("operator.pauseProcess")}</button>
                <button
                  disabled={interactionBusy || !selectedProcess || selectedProcess.status !== "paused"}
                  onClick={() => selectedProcess && client && void safe(() => client.resumeProcess(selectedProcess.pid, Boolean(snapshot?.scheduler.auto_run)).then(() => undefined), "process.resume")}
                >{t("operator.resumeProcess")}</button>
                <button disabled={interactionBusy || !selectedProcess} className="danger" onClick={confirmExit}>{t("operator.exit")}</button>
              </div>
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
                onListOperations={(pid, cursor) => {
                  if (!client) throw new Error(t("app.clientUnavailable"));
                  return client.listOperations(pid, 100, cursor);
                }}
                onExplainOperation={(operationId, cursor) => {
                  if (!client) throw new Error(t("app.clientUnavailable"));
                  return client.explainOperation(operationId, 200, cursor);
                }}
                onResolveOperation={(kind, id) => {
                  if (!client) throw new Error(t("app.clientUnavailable"));
                  return client.resolveOperation(kind, id);
                }}
                explainLookup={selectedExplainLookup}
                client={client}
                runAction={safe}
                confirmAction={queueConfirmation}
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
          onCancel={() => setPendingConfirm(null)}
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
