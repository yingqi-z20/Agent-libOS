import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Send } from "lucide-react";
import { LibOSClient, type OptionalQuanta } from "./api/client";
import type { GuiConnection, HumanRequest, RuntimeSnapshot } from "./api/types";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { DetailTabs } from "./components/DetailTabs";
import { ImageSelect } from "./components/ImageSelect";
import { ProcessTree } from "./components/ProcessTree";
import { Timeline } from "./components/Timeline";
import { TopBar } from "./components/TopBar";
import { UserPage } from "./components/UserPage";
import { previewImageManifest } from "./imagePreview";
import { useI18n } from "./i18n";

type PendingConfirm = {
  title: string;
  message: string;
  details: Record<string, unknown>;
  action(): Promise<void>;
};

export function App() {
  const { t } = useI18n();
  const [view, setView] = useState<"user" | "operator">("user");
  const [connection, setConnection] = useState<GuiConnection | null>(null);
  const [client, setClient] = useState<LibOSClient | null>(null);
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [selectedPid, setSelectedPid] = useState<string | null>(null);
  const [maxQuanta, setMaxQuanta] = useState<OptionalQuanta>(null);
  const [spawnGoal, setSpawnGoal] = useState("Summarize the current project state.");
  const [spawnImage, setSpawnImage] = useState("coding-agent:v0");
  const [message, setMessage] = useState("");
  const [cwd, setCwd] = useState("");
  const [execImage, setExecImage] = useState("base-agent:v0");
  const [execGoal, setExecGoal] = useState("");
  const [pendingConfirm, setPendingConfirm] = useState<PendingConfirm | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    void initialize();
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (!client) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    void client.stream((message) => {
      if (message.event === "snapshot") {
        const next = (message.data as { snapshot?: RuntimeSnapshot }).snapshot;
        if (next) {
          setSnapshot(next);
          setSelectedPid((current) => current ?? next.processes[0]?.pid ?? null);
        }
      }
    }, controller.signal).catch((reason) => {
      if (!controller.signal.aborted) setError(String(reason));
    });
    return () => controller.abort();
  }, [client]);

  const selectedProcess = useMemo(
    () => snapshot?.processes.find((process) => process.pid === selectedPid) ?? null,
    [snapshot, selectedPid]
  );

  async function initialize() {
    try {
      const conn = await window.libosApi?.getConnection();
      if (!conn) throw new Error(t("app.preloadMissing"));
      const nextClient = new LibOSClient(conn);
      setConnection(conn);
      setClient(nextClient);
      const nextSnapshot = await nextClient.snapshot();
      setSnapshot(nextSnapshot);
      setSelectedPid(nextSnapshot.processes[0]?.pid ?? null);
      setMaxQuanta(nextSnapshot.scheduler.default_max_quanta ?? null);
    } catch (reason) {
      setError(String(reason));
    }
  }

  async function refresh() {
    if (!client) return;
    const next = await client.snapshot();
    setSnapshot(next);
    setSelectedPid((current) => current ?? next.processes[0]?.pid ?? null);
  }

  async function reconnect(next: GuiConnection | null) {
    if (!next) return;
    const nextClient = new LibOSClient(next);
    setConnection(next);
    setClient(nextClient);
    const nextSnapshot = await nextClient.snapshot();
    setSnapshot(nextSnapshot);
    setMaxQuanta(nextSnapshot.scheduler.default_max_quanta ?? null);
  }

  async function safe(action: () => Promise<void>) {
    try {
      setError(null);
      await action();
      await refresh();
    } catch (reason) {
      const err = reason as Error & { payload?: { error?: { confirmation_required?: boolean; preview?: Record<string, unknown>; action?: string } } };
      if (err.payload?.error?.confirmation_required) {
        setError(`${err.message}. ${t("app.confirmationRequiredSuffix")}`);
      } else {
        setError(err.message ?? String(reason));
      }
    }
  }

  async function spawnProcess() {
    if (!client) return;
    await safe(async () => {
      const result = await client.spawn(spawnGoal, spawnImage, maxQuanta, Boolean(snapshot?.scheduler.auto_run));
      const pid = (result as { pid?: string }).pid;
      if (pid) setSelectedPid(pid);
    });
  }

  async function send(kind: "message" | "interrupt") {
    if (!client || !selectedPid || !message.trim()) return;
    await safe(async () => {
      await client.sendMessage(selectedPid, message.trim(), kind, Boolean(snapshot?.scheduler.auto_run), maxQuanta);
      setMessage("");
    });
  }

  async function respond(request: HumanRequest, approved: boolean, answer = "") {
    if (!client) return;
    await safe(async () => {
      await client.respondHumanRequest(request.request_id, approved, answer);
    });
  }

  function confirmExec() {
    if (!client || !selectedPid) return;
    setPendingConfirm({
      title: t("app.exec.title"),
      message: t("app.exec.message"),
      details: { pid: selectedPid, image: execImage, goal: execGoal, auto_run: snapshot?.scheduler.auto_run },
      action: async () => {
        await client.execProcess(selectedPid, execImage, execGoal, true, Boolean(snapshot?.scheduler.auto_run));
        setPendingConfirm(null);
        await refresh();
      }
    });
  }

  function confirmExit() {
    if (!client || !selectedPid) return;
    setPendingConfirm({
      title: t("app.exit.title"),
      message: t("app.exit.message"),
      details: { pid: selectedPid },
      action: async () => {
        await client.exitProcess(selectedPid, "Exited from GUI", false, true);
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
          files: Object.keys(imagePackage.files).length,
          bytes: JSON.stringify(imagePackage.files).length,
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
    if (!client || !selectedPid) return;
    const pid = selectedPid;
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

  return (
    <div className={view === "user" ? "userAppShell" : "appShell"}>
      {view === "user" ? (
        <UserPage
          connection={connection}
          snapshot={snapshot}
          selectedPid={selectedPid}
          selectedProcess={selectedProcess}
          maxQuanta={maxQuanta}
          spawnGoal={spawnGoal}
          spawnImage={spawnImage}
          message={message}
          images={snapshot?.images ?? []}
          onSelectPid={setSelectedPid}
          onMaxQuantaChange={setMaxQuanta}
          onSpawnGoalChange={setSpawnGoal}
          onSpawnImageChange={setSpawnImage}
          onMessageChange={setMessage}
          onSpawn={() => void spawnProcess()}
          onImportImage={() => void chooseAndConfirmImageImport(false)}
          onCommitImage={confirmCommitImage}
          onSend={(kind) => void send(kind)}
          onRespond={(request, approved, answer = "") => void respond(request, approved, answer)}
          onRun={() => selectedPid && client && void safe(() => client.run(selectedPid, maxQuanta).then(() => undefined))}
          onPause={() => client && void safe(() => client.pauseScheduler().then(() => undefined))}
          onRefresh={() => void refresh()}
          onOpenDb={() => void window.libosApi?.chooseDatabase().then(reconnect)}
          onShowOperator={() => setView("operator")}
          onStop={confirmExit}
        />
      ) : (
        <>
          <TopBar
            db={connection?.db ?? t("app.defaultDb")}
            scheduler={snapshot?.scheduler ?? null}
            maxQuanta={maxQuanta}
            selectedPid={selectedPid}
            onMaxQuantaChange={setMaxQuanta}
            onOpenDb={() => void window.libosApi?.chooseDatabase().then(reconnect)}
            onSpawn={() => void spawnProcess()}
            onRun={() => selectedPid && client && void safe(() => client.run(selectedPid, maxQuanta).then(() => undefined))}
            onStep={() => selectedPid && client && void safe(() => client.step(selectedPid).then(() => undefined))}
            onPause={() => client && void safe(() => client.pauseScheduler().then(() => undefined))}
            onAutoRunChange={(value) => client && void safe(() => client.setAutoRun(value).then(() => undefined))}
            onRefresh={() => void refresh()}
            onShowUser={() => setView("user")}
          />

          <main className="workspace">
            <section className="leftPane">
              <div className="paneHeader">
                <h1>{t("operator.processes.title")}</h1>
                <span>{snapshot?.processes.length ?? 0}</span>
              </div>
              <div className="spawnBox">
                <ImageSelect images={snapshot?.images ?? []} value={spawnImage} label={t("operator.spawnImage")} onChange={setSpawnImage} />
                <textarea value={spawnGoal} onChange={(event) => setSpawnGoal(event.currentTarget.value)} aria-label={t("operator.spawnGoal")} />
              </div>
              <ProcessTree processes={snapshot?.processes ?? []} selectedPid={selectedPid} onSelect={setSelectedPid} />
            </section>

            <section className="centerPane">
              <div className="paneHeader">
                <div>
                  <h1>{selectedProcess?.pid ?? t("operator.noProcessSelected")}</h1>
                  {selectedProcess ? <span>{selectedProcess.image_id} · {selectedProcess.status} · {t("operator.cwd")} {selectedProcess.working_directory}</span> : null}
                </div>
                {selectedProcess?.interrupt_count ? <span className="interruptBanner"><AlertTriangle size={16} /> {t("operator.interruptPending")}</span> : null}
              </div>

              <div className="humanRequests">
                {(snapshot?.human_requests ?? []).filter((request) => request.status === "pending").map((request) => (
                  <div className="humanCard" key={request.request_id}>
                    <strong>{String(request.payload?.question ?? request.payload?.type ?? t("operator.humanRequestFallback"))}</strong>
                    <input placeholder={t("operator.answerPlaceholder")} onKeyDown={(event) => {
                      if (event.key === "Enter") void respond(request, true, event.currentTarget.value);
                    }} />
                    <button onClick={() => void respond(request, true)}>{t("operator.approve")}</button>
                    <button className="danger" onClick={() => void respond(request, false)}>{t("operator.reject")}</button>
                  </div>
                ))}
              </div>

              <Timeline
                pid={selectedPid}
                messages={selectedProcess?.messages ?? []}
                humanRequests={snapshot?.human_requests ?? []}
                llmCalls={snapshot?.llm_calls ?? []}
                events={snapshot?.events ?? []}
                audit={snapshot?.audit ?? []}
              />

              <div className="composer">
                <input value={message} onChange={(event) => setMessage(event.currentTarget.value)} placeholder={t("operator.messagePlaceholder")} />
                <button disabled={!selectedPid || !message.trim()} onClick={() => void send("message")}><Send size={16} />{t("operator.message")}</button>
                <button disabled={!selectedPid || !message.trim()} className="warning" onClick={() => void send("interrupt")}>{t("operator.interrupt")}</button>
              </div>
            </section>

            <section className="rightPane">
              <div className="quickActions">
                <input value={cwd} placeholder={t("operator.newCwdPlaceholder")} onChange={(event) => setCwd(event.currentTarget.value)} />
                <button disabled={!client || !selectedPid || !cwd.trim()} onClick={() => selectedPid && void safe(() => client!.changeDirectory(selectedPid, cwd).then(() => undefined))}>cd</button>
                <ImageSelect images={snapshot?.images ?? []} value={execImage} label={t("operator.exec")} onChange={setExecImage} />
                <input value={execGoal} onChange={(event) => setExecGoal(event.currentTarget.value)} aria-label={t("operator.spawnGoal")} />
                <button disabled={!selectedPid} className="warning" onClick={confirmExec}>{t("operator.exec")}</button>
                <button disabled={!selectedPid} className="danger" onClick={confirmExit}>{t("operator.exit")}</button>
              </div>
              <DetailTabs
                process={selectedProcess}
                snapshot={snapshot}
                onImportImage={(replace) => void chooseAndConfirmImageImport(replace)}
                onCommitImage={confirmCommitImage}
                onUseImageForSpawn={setSpawnImage}
                onUseImageForExec={setExecImage}
                onInspectImage={(imageId) => {
                  if (!client) throw new Error(t("app.clientUnavailable"));
                  return client.inspectImage(imageId);
                }}
              />
            </section>
          </main>
        </>
      )}

      {error ? <div className="toast" role="alert">{error}</div> : null}
      {pendingConfirm ? (
        <ConfirmDialog
          title={pendingConfirm.title}
          message={pendingConfirm.message}
          details={pendingConfirm.details}
          onCancel={() => setPendingConfirm(null)}
          onConfirm={() => void pendingConfirm.action()}
        />
      ) : null}
    </div>
  );
}
