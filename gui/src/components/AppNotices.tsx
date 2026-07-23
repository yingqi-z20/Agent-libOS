import { AlertCircle, AlertTriangle, LoaderCircle, RefreshCw, WifiOff, X } from "lucide-react";
import type { RuntimeSnapshot, StreamConnectionStatus } from "../api/types";
import { useI18n } from "../i18n";

export function LoadingScreen({ error, onRetry }: { error: string | null; onRetry(): void }) {
  const { t } = useI18n();
  return (
    <main className="loadingScreen" aria-busy={!error}>
      <div className="loadingCard">
        {error ? <AlertCircle size={28} aria-hidden="true" /> : <LoaderCircle className="spin" size={28} aria-hidden="true" />}
        <h1>{error ? t("app.connectionFailed") : t("app.connecting")}</h1>
        <p>{error ?? t("app.connectingHint")}</p>
        {error ? <button className="primary" onClick={onRetry}><RefreshCw size={15} />{t("app.retry")}</button> : null}
      </div>
    </main>
  );
}

export function AppNotices({
  error,
  snapshot,
  streamStatus,
  refreshing,
  onDismissError,
  onRetry
}: {
  error: string | null;
  snapshot: RuntimeSnapshot | null;
  streamStatus: StreamConnectionStatus;
  refreshing: boolean;
  onDismissError(): void;
  onRetry(): void;
}) {
  const { t } = useI18n();
  const truncatedSections = snapshot?._truncated ? Object.keys(snapshot._truncated) : [];
  const truncatedCount = truncatedSections.length;
  const schedulerError = snapshot?.scheduler.last_error;
  const streamUnavailable = streamStatus === "reconnecting" || streamStatus === "failed";

  if (!error && !schedulerError && !truncatedCount && !streamUnavailable && !refreshing) return null;
  return (
    <section className="appNotices" aria-label={t("app.notices")} aria-live="polite">
      {error ? (
        <div className="appNotice error" role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          <span>{error}</span>
          <button className="noticeAction" onClick={onRetry}><RefreshCw size={14} />{t("app.retry")}</button>
          <button className="iconOnly noticeDismiss" aria-label={t("app.dismiss")} onClick={onDismissError}><X size={14} /></button>
        </div>
      ) : null}
      {schedulerError ? (
        <div className="appNotice error" role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          <span>{t("app.schedulerError", { message: schedulerError })}</span>
        </div>
      ) : null}
      {streamUnavailable ? (
        <div className="appNotice warning" role="status">
          <WifiOff size={16} aria-hidden="true" />
          <span>{streamStatus === "failed" ? t("app.streamFailed") : t("app.streamReconnecting")}</span>
          <button className="noticeAction" onClick={onRetry}><RefreshCw size={14} />{t("app.refreshNow")}</button>
        </div>
      ) : null}
      {truncatedCount ? (
        <div className="appNotice warning" role="status">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>{t("app.snapshotTruncated", { count: truncatedCount, sections: truncatedSections.join(", ") })}</span>
        </div>
      ) : null}
      {refreshing && !error ? (
        <div className="appNotice progress" role="status">
          <LoaderCircle className="spin" size={15} aria-hidden="true" />
          <span>{t("app.refreshing")}</span>
        </div>
      ) : null}
    </section>
  );
}
