import { Database, Pause, Play, RefreshCw, Square, StepForward } from "lucide-react";
import type { SchedulerStatus, StreamConnectionStatus } from "../api/types";
import { useI18n } from "../i18n";
import { parseOptionalQuanta } from "../quanta";
import { LanguageSwitch } from "./LanguageSwitch";

export function TopBar({
  db,
  scheduler,
  maxQuanta,
  selectedPid,
  onMaxQuantaChange,
  onOpenDb,
  onSpawn,
  onRun,
  onStep,
  onPause,
  onAutoRunChange,
  onRefresh,
  onShowUser,
  busy,
  streamStatus,
  lastUpdatedAt
}: {
  db: string;
  scheduler: SchedulerStatus | null;
  maxQuanta: number | null;
  selectedPid: string | null;
  onMaxQuantaChange(value: number | null): void;
  onOpenDb(): void;
  onSpawn(): void;
  onRun(): void;
  onStep(): void;
  onPause(): void;
  onAutoRunChange(value: boolean): void;
  onRefresh(): void;
  onShowUser?: () => void;
  busy: boolean;
  streamStatus: StreamConnectionStatus;
  lastUpdatedAt: Date | null;
}) {
  const { formatTime, t } = useI18n();

  return (
    <header className="topBar">
      <div className="dbGroup">
        <Database size={17} />
        <input
          aria-label={t("top.runtimeDatabase")}
          value={db}
          readOnly
        />
        <button title={t("top.openSqlite")} disabled={busy} onClick={onOpenDb}>{t("top.open")}</button>
      </div>
      <button className="primary" disabled={busy} onClick={onSpawn}>{t("top.spawn")}</button>
      <label className="toggle">
        <input type="checkbox" disabled={busy} checked={Boolean(scheduler?.auto_run)} onChange={(event) => onAutoRunChange(event.currentTarget.checked)} />
        {t("top.autoRun")}
      </label>
      <label className="quanta">
        {t("top.quanta")}
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
      <button title={t("top.runSelected")} disabled={busy || !selectedPid || scheduler?.running} onClick={onRun}><Play size={16} />{t("user.run")}</button>
      <button title={t("top.stepSelected")} disabled={busy || !selectedPid || scheduler?.running} onClick={onStep}><StepForward size={16} />{t("top.step")}</button>
      <button title={t("top.pauseScheduler")} disabled={busy} onClick={onPause}><Pause size={16} />{t("user.pause")}</button>
      <button aria-label={t("top.refreshSnapshot")} title={t("top.refreshSnapshot")} disabled={busy} onClick={onRefresh}><RefreshCw size={16} /></button>
      <LanguageSwitch />
      {onShowUser ? <button className="secondary" onClick={onShowUser}>{t("top.userPage")}</button> : null}
      <span
        className={`connectionBadge ${streamStatus}`}
        role="status"
        title={lastUpdatedAt ? t("connection.updated", { time: formatTime(lastUpdatedAt.toISOString()) }) : undefined}
      >
        <span className="statusDot" />{t(`connection.${streamStatus}`)}
      </span>
      <span className={`scheduler ${scheduler?.running ? "running" : ""}`}><Square size={11} />{scheduler?.running ? t("top.running") : scheduler?.paused ? t("top.paused") : t("top.idle")}</span>
    </header>
  );
}
