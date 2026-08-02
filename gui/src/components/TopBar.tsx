import { AlertTriangle, Bot, CirclePlus, Database, FolderOpen, Gauge, LoaderCircle, Pause, Play, RefreshCw, StepForward, UserRound } from "lucide-react";
import { useId } from "react";
import type { SchedulerStatus, StreamConnectionStatus } from "../api/types";
import { useI18n } from "../i18n";
import { LanguageSwitch } from "./LanguageSwitch";

export function TopBar({
  db,
  scheduler,
  maxQuantaInput,
  quantaValid = true,
  selectedPid,
  onMaxQuantaChange,
  onOpenDb,
  onSpawn,
  onRun,
  onStep,
  onPause,
  onAutoRunChange,
  onRefresh,
  pendingHumanCount = 0,
  onOpenPending,
  onShowUser,
  busy,
  streamStatus,
  lastUpdatedAt
}: {
  db: string;
  scheduler: SchedulerStatus | null;
  maxQuantaInput: string;
  quantaValid?: boolean;
  selectedPid: string | null;
  onMaxQuantaChange(value: string): void;
  onOpenDb(): void;
  onSpawn(): void;
  onRun(): void;
  onStep(): void;
  onPause(): void;
  onAutoRunChange(value: boolean): void;
  onRefresh(): void;
  pendingHumanCount?: number;
  onOpenPending?: () => void;
  onShowUser?: () => void;
  busy: boolean;
  streamStatus: StreamConnectionStatus;
  lastUpdatedAt: Date | null;
}) {
  const { formatTime, t } = useI18n();
  const quantaErrorId = useId();
  const schedulerLabel = scheduler?.running ? t("top.running") : scheduler?.paused ? t("top.paused") : t("top.idle");

  return (
    <header className="topBar">
      <div className="operatorBrand">
        <span className="brandMark" aria-hidden="true"><Bot size={17} /></span>
        <div>
          <strong>Agent libOS</strong>
          <span>{t("top.operatorWorkspace")}</span>
        </div>
      </div>

      <div className="operatorDbChip" title={db}>
        <Database size={16} aria-hidden="true" />
        <div>
          <span>{t("top.runtimeDatabase")}</span>
          <strong>{db}</strong>
        </div>
        <button type="button" className="iconOnly softButton" aria-label={t("top.openSqlite")} title={t("top.openSqlite")} disabled={busy} onClick={onOpenDb}>
          <FolderOpen size={14} />
        </button>
      </div>

      <div className="operatorCommandBar" role="group" aria-label={t("top.runtimeControls")}>
        <button type="button" className="primary spawnProcessButton" disabled={busy || !quantaValid} onClick={onSpawn}>
          <CirclePlus size={15} />{t("top.spawn")}
        </button>
        <label className="switchControl">
          <input type="checkbox" disabled={busy} checked={Boolean(scheduler?.auto_run)} onChange={(event) => onAutoRunChange(event.currentTarget.checked)} />
          <span className="switchVisual" aria-hidden="true"><span /></span>
          <span>{t("top.autoRun")}</span>
        </label>
        <label className="operatorQuanta">
          <Gauge size={14} aria-hidden="true" />
          <span>{t("top.quanta")}</span>
          <input
            type="text"
            inputMode="numeric"
            disabled={busy}
            value={maxQuantaInput}
            aria-invalid={!quantaValid || undefined}
            aria-errormessage={!quantaValid ? quantaErrorId : undefined}
            placeholder={t("scheduler.unlimitedPlaceholder")}
            title={quantaValid ? t("scheduler.unlimitedHint") : t("scheduler.invalidQuanta")}
            onChange={(event) => onMaxQuantaChange(event.currentTarget.value)}
          />
          {!quantaValid ? <small id={quantaErrorId} className="inlineError" role="alert">{t("scheduler.invalidQuanta")}</small> : null}
        </label>
        <div className="operatorRunGroup">
          <button type="button" title={t("top.runSelected")} disabled={busy || !quantaValid || !selectedPid || scheduler?.running} onClick={onRun}><Play size={15} />{t("user.run")}</button>
          <button type="button" title={t("top.stepSelected")} disabled={busy || !quantaValid || !selectedPid || scheduler?.running} onClick={onStep}><StepForward size={15} />{t("top.step")}</button>
          <button type="button" title={t("top.pauseScheduler")} disabled={busy || !scheduler?.running} onClick={onPause}><Pause size={15} />{t("user.pause")}</button>
          <button type="button" className="iconOnly softButton" aria-label={t("top.refreshSnapshot")} title={t("top.refreshSnapshot")} disabled={busy} onClick={onRefresh}><RefreshCw size={15} /></button>
        </div>
      </div>

      <div className="operatorTopMeta">
        {busy ? (
          <span className="operatorBusyStatus" role="status">
            <LoaderCircle className="spin" size={14} aria-hidden="true" />
            <span>{t("user.working")}</span>
          </span>
        ) : (
          <span className={`schedulerPill ${scheduler?.running ? "running" : scheduler?.paused ? "paused" : "idle"}`}>
            <span className="statusDot" />{schedulerLabel}
          </span>
        )}
        <span
          className={`connectionBadge ${streamStatus}`}
          role="status"
          title={lastUpdatedAt ? t("connection.updated", { time: formatTime(lastUpdatedAt.toISOString()) }) : undefined}
        >
          <span className="statusDot" />{t(`connection.${streamStatus}`)}
        </span>
        {pendingHumanCount > 0 && onOpenPending ? (
          <button
            type="button"
            className="pendingInboxButton compact"
            aria-label={`${t("user.pendingRequests")}: ${pendingHumanCount}`}
            title={t("user.pendingRequests")}
            onClick={onOpenPending}
          >
            <AlertTriangle size={14} aria-hidden="true" />
            <span>{t("user.pendingRequests")}</span>
            <strong>{pendingHumanCount}</strong>
          </button>
        ) : null}
        <LanguageSwitch />
        {onShowUser ? (
          <button type="button" className="secondary userViewButton" aria-label={t("top.userPage")} title={t("top.userPage")} onClick={onShowUser}>
            <UserRound size={15} /><span>{t("top.userPage")}</span>
          </button>
        ) : null}
      </div>
    </header>
  );
}
