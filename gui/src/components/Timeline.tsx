import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, ArrowDown, Bot, CheckCircle2, MessageSquare, Radio, UserRound } from "lucide-react";
import type { AuditRecord, HumanRequest, LlmCall, ProcessMessage, RuntimeEvent } from "../api/types";
import { useI18n, type TranslationKey } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";
import { isHumanOutput } from "../userConversation";

export type TimelineItemKind = "message" | "human" | "llm" | "event" | "audit";
export type TimelineFilter = "activity" | "all" | TimelineItemKind;

export type TimelineItem =
  | { kind: "message"; time: string; item: ProcessMessage }
  | { kind: "human"; time: string; item: HumanRequest }
  | { kind: "llm"; time: string; item: LlmCall }
  | { kind: "event"; time: string; item: RuntimeEvent }
  | { kind: "audit"; time: string; item: AuditRecord };

const timelineItemKinds = ["message", "human", "llm", "event", "audit"] as const satisfies readonly TimelineItemKind[];
const timelineFilters = ["activity", "all", ...timelineItemKinds] as const satisfies readonly TimelineFilter[];
const timelineFilterLabels: Record<TimelineFilter, TranslationKey> = {
  activity: "timeline.filter.activity",
  all: "timeline.filter.all",
  message: "timeline.filter.message",
  human: "timeline.filter.human",
  llm: "timeline.filter.llm",
  event: "timeline.filter.event",
  audit: "timeline.filter.audit"
};
export const TIMELINE_LATEST_THRESHOLD_PX = 96;

export function Timeline({
  pid,
  messages,
  humanRequests,
  llmCalls,
  events,
  audit,
  onExplainEvidence
}: {
  pid: string | null;
  messages: ProcessMessage[];
  humanRequests: HumanRequest[];
  llmCalls: LlmCall[];
  events: RuntimeEvent[];
  audit: AuditRecord[];
  onExplainEvidence?(kind: string, id: string): void;
}) {
  const { formatTime, t } = useI18n();
  const [filter, setFilter] = useState<TimelineFilter>("activity");
  const items = useMemo(
    () => pid ? buildTimelineItems({ pid, messages, humanRequests, llmCalls, events, audit }) : [],
    [audit, events, humanRequests, llmCalls, messages, pid]
  );
  const counts = useMemo(() => countTimelineItemsByKind(items), [items]);
  const filteredItems = useMemo(() => filterTimelineItems(items, filter), [filter, items]);
  const timelineRef = useRef<HTMLElement>(null);
  const followedPidRef = useRef(pid);
  const followLatestRef = useRef(true);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const latestItemKey = filteredItems.length > 0
    ? timelineItemKey(filteredItems[filteredItems.length - 1])
    : null;
  const itemsRevision = useMemo(
    () => filteredItems.map(timelineItemRevision).join("\n"),
    [filteredItems]
  );

  useEffect(() => {
    const processChanged = followedPidRef.current !== pid;
    followedPidRef.current = pid;
    if (processChanged) followLatestRef.current = true;

    const container = timelineRef.current;
    if (!container) return;
    if (followLatestRef.current) {
      container.scrollTop = container.scrollHeight;
      setShowJumpToLatest(false);
      return;
    }

    const nearLatest = isTimelineNearLatest(container);
    followLatestRef.current = nearLatest;
    setShowJumpToLatest(!nearLatest);
  }, [filter, filteredItems.length, itemsRevision, latestItemKey, pid]);

  function scrollToLatest() {
    const container = timelineRef.current;
    if (!container) return;
    const reducedMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    container.scrollTo({
      top: container.scrollHeight,
      behavior: reducedMotion ? "auto" : "smooth"
    });
    followLatestRef.current = true;
    setShowJumpToLatest(false);
  }

  if (!pid) return <div className="empty">{t("timeline.selectProcess")}</div>;
  if (items.length === 0) return <div className="empty">{t("timeline.empty")}</div>;

  return (
    <div className="timelineShell" style={{ minHeight: 0, overflow: "hidden", position: "relative" }}>
      <section
        ref={timelineRef}
        className="timeline"
        aria-label={t("timeline.label")}
        role="region"
        tabIndex={0}
        style={{ height: "100%" }}
        onScroll={(event) => {
          const nearLatest = isTimelineNearLatest(event.currentTarget);
          followLatestRef.current = nearLatest;
          setShowJumpToLatest(!nearLatest);
        }}
      >
        <div className="timelineFilter" role="group" aria-label={t("timeline.filterLabel")}>
          {timelineFilters.map((option) => {
            const count = option === "all"
              ? items.length
              : option === "activity"
                ? counts.message + counts.human + counts.llm
                : counts[option];
            const active = filter === option;
            return (
              <button
                type="button"
                key={option}
                className={active ? "active" : ""}
                aria-pressed={active}
                onClick={() => setFilter(option)}
              >
                {t(timelineFilterLabel(option))}
                <span className="timelineFilterCount">{count}</span>
              </button>
            );
          })}
        </div>
        <div className="timelineEntries" role="log" aria-live="polite" aria-relevant="additions text">
          {filteredItems.length === 0 ? (
            <div className="empty timelineEmpty">{t("timeline.filterEmpty")}</div>
          ) : filteredItems.map((entry) => (
            <article className={`timelineItem ${entry.kind}`} key={timelineItemKey(entry)}>
              <div className="timelineIcon">{icon(entry)}</div>
              <div className="timelineBody">
                <div className="timelineHeader">
                  <strong>{title(entry, t)}</strong>
                  <time>{formatTime(entry.time)}</time>
                </div>
                <p className="timelineSummary" title={summary(entry, t)}>{summary(entry, t)}</p>
                {evidenceRef(entry) && onExplainEvidence ? (
                  <button
                    type="button"
                    className="timelineExplain"
                    onClick={() => {
                      const ref = evidenceRef(entry);
                      if (ref) onExplainEvidence(ref.kind, ref.id);
                    }}
                  >
                    {t("timeline.explain")}
                  </button>
                ) : null}
                <div className="timelineJsonOperation" role="group" aria-live="off">
                  <CollapsibleJson value={entry.item} />
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>
      {showJumpToLatest ? (
        <button
          type="button"
          className="jumpToLatest timelineJumpToLatest"
          style={{ bottom: 12 }}
          onClick={scrollToLatest}
        >
          <ArrowDown size={14} aria-hidden="true" />{t("user.jumpToLatest")}
        </button>
      ) : null}
    </div>
  );
}

export function isTimelineNearLatest(
  metrics: Pick<HTMLElement, "clientHeight" | "scrollHeight" | "scrollTop">,
  threshold = TIMELINE_LATEST_THRESHOLD_PX
): boolean {
  return metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight < threshold;
}

export function timelineItemKey(entry: TimelineItem): string {
  if (entry.kind === "message") return `message-${entry.item.message_id}`;
  if (entry.kind === "human") return `human-${entry.item.request_id}`;
  if (entry.kind === "llm") return `llm-${entry.item.call_id}`;
  if (entry.kind === "event") return `event-${entry.item.event_id}`;
  return `audit-${entry.item.record_id}`;
}

export function timelineItemRevision(entry: TimelineItem): string {
  const key = timelineItemKey(entry);
  if (entry.kind === "message") return `${key}:${entry.item.status}:${contentRevision(entry.item.subject)}:${contentRevision(entry.item.body)}`;
  if (entry.kind === "human") return `${key}:${entry.item.status}:${entry.item.updated_at}`;
  if (entry.kind === "llm") {
    return `${key}:${entry.item.status}:${entry.item.completed_at ?? ""}:${contentRevision(entry.item.response_content)}:${entry.item.error ?? ""}`;
  }
  return key;
}

function contentRevision(value: string): string {
  return `${value.length}:${value.slice(0, 64)}:${value.slice(-64)}`;
}

export function evidenceRef(entry: TimelineItem): { kind: string; id: string } | null {
  if (entry.kind === "human") return { kind: "request", id: entry.item.request_id };
  if (entry.kind === "llm") return { kind: "call", id: entry.item.call_id };
  if (entry.kind === "event") return { kind: "event", id: entry.item.event_id };
  if (entry.kind === "audit") return { kind: "audit", id: entry.item.record_id };
  return null;
}

export function buildTimelineItems({
  pid,
  messages,
  humanRequests,
  llmCalls,
  events,
  audit
}: {
  pid: string;
  messages: ProcessMessage[];
  humanRequests: HumanRequest[];
  llmCalls: LlmCall[];
  events: RuntimeEvent[];
  audit: AuditRecord[];
}): TimelineItem[] {
  return [
    ...messages.map((item) => ({ kind: "message" as const, time: item.created_at, item })),
    ...humanRequests.filter((item) => item.pid === pid).map((item) => ({ kind: "human" as const, time: item.created_at, item })),
    ...llmCalls.filter((item) => item.pid === pid).map((item) => ({ kind: "llm" as const, time: item.created_at, item })),
    ...events.filter((item) => item.target === pid || item.source === pid).map((item) => ({ kind: "event" as const, time: item.created_at, item })),
    ...audit.filter((item) => item.actor === pid || item.target === `process:${pid}`).map((item) => ({ kind: "audit" as const, time: item.timestamp, item }))
  ].sort((a, b) => a.time.localeCompare(b.time));
}

export function countTimelineItemsByKind(items: TimelineItem[]): Record<TimelineItemKind, number> {
  const counts: Record<TimelineItemKind, number> = {
    message: 0,
    human: 0,
    llm: 0,
    event: 0,
    audit: 0
  };
  for (const item of items) counts[item.kind] += 1;
  return counts;
}

export function filterTimelineItems(items: TimelineItem[], filter: TimelineFilter): TimelineItem[] {
  if (filter === "all") return items;
  if (filter === "activity") return items.filter((item) => item.kind === "message" || item.kind === "human" || item.kind === "llm");
  return items.filter((item) => item.kind === filter);
}

function timelineFilterLabel(filter: TimelineFilter): TranslationKey {
  return timelineFilterLabels[filter];
}

function icon(entry: TimelineItem) {
  if (entry.kind === "message") return entry.item.kind === "interrupt" ? <AlertTriangle size={16} /> : <MessageSquare size={16} />;
  if (entry.kind === "human") return <UserRound size={16} />;
  if (entry.kind === "llm") return <Bot size={16} />;
  if (entry.kind === "audit") return <CheckCircle2 size={16} />;
  return <Radio size={16} />;
}

function title(entry: TimelineItem, t: (key: TranslationKey, vars?: Record<string, string | number>) => string) {
  if (entry.kind === "message") return t("timeline.messageTitle", { kind: entry.item.kind });
  if (entry.kind === "human" && isHumanOutput(entry.item)) return t("timeline.agentOutput");
  if (entry.kind === "human") return t("timeline.humanRequest", { status: entry.item.status });
  if (entry.kind === "llm") return t("timeline.llmStatus", { status: entry.item.status });
  if (entry.kind === "audit") return entry.item.action;
  return entry.item.type;
}

function summary(entry: TimelineItem, t: (key: TranslationKey) => string) {
  if (entry.kind === "message") return entry.item.subject || entry.item.body || t("timeline.emptyMessage");
  if (entry.kind === "human" && isHumanOutput(entry.item)) return String(entry.item.payload.message ?? t("timeline.emptyOutput"));
  if (entry.kind === "human") return String(entry.item.payload?.question ?? entry.item.payload?.type ?? t("timeline.humanInteraction"));
  if (entry.kind === "llm") return entry.item.error ?? entry.item.response_content ?? entry.item.purpose;
  if (entry.kind === "audit") return entry.item.target ?? t("timeline.auditRecord");
  return entry.item.target ?? entry.item.source;
}
