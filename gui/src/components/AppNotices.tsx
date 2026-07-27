import { AlertCircle, AlertTriangle, LoaderCircle, RefreshCw, WifiOff, X } from "lucide-react";
import { useState } from "react";
import type { RuntimeSnapshot, StreamConnectionStatus } from "../api/types";
import { useI18n } from "../i18n";

const MAX_VISIBLE_TRUNCATED_SECTIONS = 3;

export function summarizeTruncatedSections(sections: string[]): string {
  const visible = sections.slice(0, MAX_VISIBLE_TRUNCATED_SECTIONS);
  const remaining = sections.length - visible.length;
  return remaining > 0
    ? `${visible.join(", ")} … (+${remaining})`
    : visible.join(", ");
}

export function LoadingScreen({ error, onRetry }: { error: string | null; onRetry(): void }) {
  const { t } = useI18n();
  return (
    <main className="loadingScreen" aria-busy={!error}>
      <div className="loadingCard">
        {error ? <AlertCircle size={28} aria-hidden="true" /> : <LoaderCircle className="spin" size={28} aria-hidden="true" />}
        <h1>{error ? t("app.connectionFailed") : t("app.connecting")}</h1>
        <p role={error ? "alert" : "status"}>{error ?? t("app.connectingHint")}</p>
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
  showSnapshotDiagnostics = false,
  onDismissError,
  onRetry
}: {
  error: string | null;
  snapshot: RuntimeSnapshot | null;
  streamStatus: StreamConnectionStatus;
  refreshing: boolean;
  showSnapshotDiagnostics?: boolean;
  onDismissError(): void;
  onRetry(): void;
}) {
  const { t } = useI18n();
  const [dismissedTruncationKey, setDismissedTruncationKey] = useState<string | null>(null);
  const truncatedSections = snapshot?._truncated ? Object.keys(snapshot._truncated).sort() : [];
  const truncatedCount = truncatedSections.length;
  const truncatedSummary = summarizeTruncatedSections(truncatedSections);
  const truncationKey = truncatedSections.join("\n");
  const showTruncation = showSnapshotDiagnostics
    && truncatedCount > 0
    && dismissedTruncationKey !== truncationKey;
  const schedulerError = snapshot?.scheduler.last_error;
  const streamUnavailable = streamStatus === "reconnecting" || streamStatus === "failed";
  const pendingHumanCount = (snapshot?.human_requests ?? []).filter((request) => request.status === "pending").length;
  const hasVisibleNotice = Boolean(
    error || schedulerError || streamUnavailable || showTruncation || (refreshing && !error)
  );

  return (
    <section className="appNotices" aria-label={t("app.notices")} data-has-visible={hasVisibleNotice || undefined}>
      <span className="srOnly" role="status" aria-live="polite" aria-atomic="true">
        {pendingHumanCount > 0 ? `${t("user.pendingRequests")}: ${pendingHumanCount}` : ""}
      </span>
      {error ? (
        <div className="appNotice error" role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          <span>{error}</span>
          <button className="noticeAction" onClick={onRetry}><RefreshCw size={14} />{t("app.retry")}</button>
          <button className="iconOnly noticeDismiss" aria-label={t("app.dismiss")} onClick={onDismissError}><X size={14} /></button>
        </div>
      ) : null}
      {schedulerError ? (
        <div className="appNotice error passive" role="alert">
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
      {showTruncation ? (
        <div className="appNotice warning snapshotWarning" role="status">
          <AlertTriangle size={16} aria-hidden="true" />
          <span title={t("app.snapshotTruncated", { count: truncatedCount, sections: truncatedSummary })}>
            {t("app.snapshotTruncated", { count: truncatedCount, sections: truncatedSummary })}
          </span>
          <button
            type="button"
            className="iconOnly noticeDismiss"
            aria-label={t("app.dismiss")}
            onClick={() => setDismissedTruncationKey(truncationKey)}
          >
            <X size={14} />
          </button>
        </div>
      ) : null}
      {refreshing && !error ? (
        <div className="appNotice progress passive" role="status">
          <LoaderCircle className="spin" size={15} aria-hidden="true" />
          <span>{t("app.refreshing")}</span>
        </div>
      ) : null}
    </section>
  );
}
