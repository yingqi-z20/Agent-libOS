import { AlertTriangle, Bot, Database, MessageSquare, Pause, Play, RefreshCw, Send, Settings, Square } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import type { GuiConnection, HumanRequest, HumanResponseInput, ImageSummary, LLMProfileInput, LLMProfileSummary, RuntimeProcess, RuntimeSnapshot, StreamConnectionStatus } from "../api/types";
import { useI18n } from "../i18n";
import { parseOptionalQuanta } from "../quanta";
import type { WorkspaceAccess } from "../taskAuthority";
import { deriveUserConversation, humanRequestPrompt, type UserConversationItem } from "../userConversation";
import { ImageSelect } from "./ImageSelect";
import { HumanRequestCard } from "./HumanRequestCard";
import { LanguageSwitch } from "./LanguageSwitch";
import { LLMProfileSelect } from "./LLMProfileSelect";
import { RatingPanel } from "./RatingPanel";

const MarkdownMessage = lazy(async () => {
  const module = await import("./MarkdownMessage");
  return { default: module.MarkdownMessage };
});

type UserPageProps = {
  connection: GuiConnection | null;
  snapshot: RuntimeSnapshot | null;
  selectedPid: string | null;
  selectedProcess: RuntimeProcess | null;
  maxQuanta: number | null;
  spawnGoal: string;
  spawnImage: string;
  spawnLlmProfile: string;
  spawnWorkingDirectory: string;
  spawnWorkspaceAccess: WorkspaceAccess;
  spawnAllowGitRequests: boolean;
  message: string;
  images: ImageSummary[];
  llmProfiles: LLMProfileSummary[];
  onSelectPid(pid: string): void;
  onMaxQuantaChange(value: number | null): void;
  onSpawnGoalChange(value: string): void;
  onSpawnImageChange(value: string): void;
  onSpawnLlmProfileChange(value: string): void;
  onSpawnWorkingDirectoryChange(value: string): void;
  onSpawnWorkspaceAccessChange(value: WorkspaceAccess): void;
  onSpawnAllowGitRequestsChange(value: boolean): void;
  onMessageChange(value: string): void;
  onSpawn(): void;
  onImportImage(): void;
  onCommitImage(request: { imageId: string; name: string; version: string; replace: boolean; checkpointId?: string }): void;
  onSend(kind: "message" | "interrupt"): void;
  onRespond(request: HumanRequest, response: HumanResponseInput): Promise<boolean>;
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
  connection,
  snapshot,
  selectedPid,
  selectedProcess,
  maxQuanta,
  spawnGoal,
  spawnImage,
  spawnLlmProfile,
  spawnWorkingDirectory,
  spawnWorkspaceAccess,
  spawnAllowGitRequests,
  message,
  images,
  llmProfiles,
  onSelectPid,
  onMaxQuantaChange,
  onSpawnGoalChange,
  onSpawnImageChange,
  onSpawnLlmProfileChange,
  onSpawnWorkingDirectoryChange,
  onSpawnWorkspaceAccessChange,
  onSpawnAllowGitRequestsChange,
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
  const conversation = useMemo(() => deriveUserConversation(snapshot, selectedPid), [snapshot, selectedPid]);
  const pendingRequests = conversation.filter((item): item is Extract<UserConversationItem, { role: "request" }> => item.role === "request");
  const isRunning = Boolean(snapshot?.scheduler.running);
  const hasProcess = Boolean(selectedProcess);
  const commitReady = Boolean(hasProcess && commitImageId.trim() && commitName.trim() && commitVersion.trim());
  const conversationRef = useRef<HTMLElement>(null);
  const followConversationRef = useRef(true);

  useEffect(() => {
    const container = conversationRef.current;
    if (container && followConversationRef.current) container.scrollTop = container.scrollHeight;
  }, [conversation.at(-1)?.id, selectedPid]);

  return (
    <main className="userPage">
      <header className="userTopBar">
        <div className="userBrand">
          <Bot size={18} />
          <div>
            <strong>Agent libOS</strong>
            <span>{connection?.db ?? t("app.defaultDb")}</span>
          </div>
        </div>
        <div className="userTopActions">
          <span className={`connectionBadge ${streamStatus}`} role="status">
            <span className="statusDot" />
            {t(`connection.${streamStatus}`)}
          </span>
          {lastUpdatedAt ? (
            <time className="lastUpdated" dateTime={lastUpdatedAt.toISOString()}>
              {t("connection.updated", { time: formatTime(lastUpdatedAt.toISOString()) })}
            </time>
          ) : null}
          <LanguageSwitch />
          <button disabled={busy} title={t("user.openDbTitle")} onClick={onOpenDb}><Database size={15} />{t("user.openDb")}</button>
          <button disabled={busy} aria-label={t("user.refreshTitle")} title={t("user.refreshTitle")} onClick={onRefresh}><RefreshCw size={15} /></button>
          <button className="secondary" onClick={onShowOperator}><Settings size={15} />{t("user.operatorConsole")}</button>
        </div>
      </header>

      <section className="userTaskBar">
        <div className="userTaskMain">
          <label>
            {t("user.process")}
            <select value={selectedPid ?? ""} disabled={busy} onChange={(event) => onSelectPid(event.currentTarget.value)}>
              {(snapshot?.processes.length ?? 0) === 0 ? <option value="">{t("user.noProcess")}</option> : null}
              {(snapshot?.processes ?? []).map((process) => (
                <option key={process.pid} value={process.pid}>{process.pid} · {process.status}</option>
              ))}
            </select>
          </label>
          <div className="userStatus">
            <span className={`statusDot ${isRunning ? "running" : ""}`} />
            {isRunning ? t("user.running") : snapshot?.scheduler.paused ? t("user.paused") : t("user.idle")}
          </div>
          {selectedProcess ? (
            <div className="userProcessMeta">
              <span>{selectedProcess.image_id}</span>
              <span>{selectedProcess.llm_profile_id}</span>
              <span>{selectedProcess.status}</span>
              <span>{t("user.llmCalls", { count: selectedProcess.llm_call_count })}</span>
              <span>{t("user.tokens", { count: selectedProcess.token_total })}</span>
            </div>
          ) : <span className="subtle">{t("user.noProcessYet")}</span>}
        </div>
        <div className="userRunControls">
          <label className="quanta">
            {t("user.quanta")}
            <input
              type="number"
              min={1}
              step={1}
              disabled={busy}
              value={maxQuanta ?? ""}
              placeholder={t("scheduler.unlimitedPlaceholder")}
              title={t("scheduler.unlimitedHint")}
              onChange={(event) => onMaxQuantaChange(parseOptionalQuanta(event.currentTarget.value))}
            />
          </label>
          <button disabled={busy || !hasProcess || isRunning} onClick={onRun}><Play size={15} />{t("user.run")}</button>
          <button disabled={busy} onClick={onPause}><Pause size={15} />{t("user.pause")}</button>
          <button className="danger" disabled={busy || !hasProcess} onClick={onStop}><Square size={13} />{t("user.stop")}</button>
        </div>
      </section>

      <div className="userNotices">
        <section className="userImageControls">
          <ImageSelect images={images} value={spawnImage} disabled={busy} onChange={onSpawnImageChange} />
          <button disabled={busy} onClick={() => onImportImage()}>{t("image.import")}</button>
          <input disabled={busy} aria-label={t("image.commitIdPlaceholder")} value={commitImageId} onChange={(event) => setCommitImageId(event.currentTarget.value)} placeholder={t("image.commitIdPlaceholder")} />
          <input disabled={busy} aria-label={t("image.commitNamePlaceholder")} value={commitName} onChange={(event) => setCommitName(event.currentTarget.value)} placeholder={t("image.commitNamePlaceholder")} />
          <input disabled={busy} aria-label={t("image.version")} value={commitVersion} onChange={(event) => setCommitVersion(event.currentTarget.value)} placeholder={t("image.version")} />
          <button
            className="warning"
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
        </section>

        {hasProcess ? <RatingPanel process={selectedProcess} onSave={onRate} /> : null}

        {!hasProcess ? (
          <section className="userStart">
            <h1>{t("user.startTask")}</h1>
            <input
              value={spawnWorkingDirectory}
              disabled={busy}
              onChange={(event) => onSpawnWorkingDirectoryChange(event.currentTarget.value)}
              placeholder={t("user.initialCwdPlaceholder")}
              aria-label={t("user.initialCwd")}
            />
            <LLMProfileSelect
              profiles={llmProfiles}
              value={spawnLlmProfile}
              label={t("llmProfile.spawnLabel")}
              disabled={busy}
              onChange={onSpawnLlmProfileChange}
              onCreate={onCreateLlmProfile}
              onUpdate={onUpdateLlmProfile}
              onDelete={onDeleteLlmProfile}
            />
            <label className="taskAuthorityField">
              <span>{t("taskAuthority.workspaceAccess")}</span>
              <select
                value={spawnWorkspaceAccess}
                disabled={busy}
                onChange={(event) => onSpawnWorkspaceAccessChange(event.currentTarget.value as WorkspaceAccess)}
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
                disabled={busy}
                onChange={(event) => onSpawnAllowGitRequestsChange(event.currentTarget.checked)}
              />
              <span>{t("taskAuthority.git")}</span>
            </label>
            <p className="taskAuthorityHint">{t("taskAuthority.hint")}</p>
            <textarea disabled={busy} aria-label={t("user.startTask")} value={spawnGoal} onChange={(event) => onSpawnGoalChange(event.currentTarget.value)} />
            <button className="primary" disabled={busy || !spawnGoal.trim()} onClick={onSpawn}>{t("user.start")}</button>
          </section>
        ) : null}

        {pendingRequests.length > 0 ? (
          <section className="userPendingRequests" aria-label={t("user.pendingRequests")}>
            {pendingRequests.map(({ request }) => (
              <HumanRequestCard
                className="userRequestCard"
                key={request.request_id}
                request={request}
                onRespond={onRespond}
              />
            ))}
          </section>
        ) : null}
      </div>

      <section
        ref={conversationRef}
        className="userConversation"
        aria-label={t("user.conversation")}
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        onScroll={(event) => {
          const element = event.currentTarget;
          followConversationRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 96;
        }}
      >
        {conversation.length === 0 ? (
          <div className="userEmpty">
            <MessageSquare size={20} />
            <span>{t("user.emptyConversation")}</span>
          </div>
        ) : conversation.map((item) => <ConversationBubble key={item.id} item={item} />)}
      </section>

      <footer className="userComposer">
        <div className="userComposerStatus">
          {selectedProcess?.interrupt_count ? <span className="interruptBanner"><AlertTriangle size={15} /> {t("operator.interruptPending")}</span> : null}
        </div>
        <input
          value={message}
          disabled={busy || !hasProcess}
          aria-label={t("user.messageAgent")}
          onChange={(event) => onMessageChange(event.currentTarget.value)}
          placeholder={t("user.messageAgent")}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.nativeEvent.isComposing && message.trim()) onSend("message");
          }}
        />
        <button disabled={busy || !hasProcess || !message.trim()} onClick={() => onSend("message")}><Send size={15} />{t("user.send")}</button>
        <button disabled={busy || !hasProcess || !message.trim()} className="warning" onClick={() => onSend("interrupt")}>{t("user.interrupt")}</button>
      </footer>
    </main>
  );
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
        <Suspense fallback={<p>{item.text || (item.protected ? t("user.protectedOutput") : t("user.empty"))}</p>}>
          <MarkdownMessage
            text={item.text}
            fallback={item.protected ? t("user.protectedOutput") : t("user.empty")}
          />
        </Suspense>
      ) : (
        <p>{item.text || t("user.empty")}</p>
      )}
      <time>{formatTime(item.time)}</time>
    </article>
  );
}
