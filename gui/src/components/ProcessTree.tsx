import { AlertTriangle, Circle, Pause, Play, Search, Square, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { HumanRequest, RuntimeProcess } from "../api/types";
import { useI18n } from "../i18n";
import { shortProcessId, taskDisplayLabel } from "../taskPresentation";

type ProcessTreeProps = {
  processes: RuntimeProcess[];
  selectedPid: string | null;
  taskLabels?: Readonly<Record<string, string>>;
  humanRequests?: HumanRequest[];
  disabled?: boolean;
  onSelect(pid: string): void;
};

export function ProcessTree({ processes, selectedPid, taskLabels = {}, humanRequests = [], disabled = false, onSelect }: ProcessTreeProps) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [rovingPid, setRovingPid] = useState<string | null>(selectedPid);
  const visibleProcesses = useMemo(
    () => filterProcesses(processes, query, taskLabels, selectedPid),
    [processes, query, selectedPid, taskLabels]
  );
  const pendingHumanCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const request of humanRequests) {
      if (request.status === "pending") counts.set(request.pid, (counts.get(request.pid) ?? 0) + 1);
    }
    return counts;
  }, [humanRequests]);
  const { roots, children } = indexProcessTree(visibleProcesses);
  const focusPid = visibleProcesses.some((process) => process.pid === rovingPid)
    ? rovingPid
    : visibleProcesses.some((process) => process.pid === selectedPid)
      ? selectedPid
    : visibleProcesses[0]?.pid ?? null;

  useEffect(() => {
    if (selectedPid) setRovingPid(selectedPid);
  }, [selectedPid]);

  return (
    <div className="processTreeShell">
      <label className="processSearch">
        <Search size={14} aria-hidden="true" />
        <span className="srOnly">{t("processTree.search")}</span>
        <input
          type="search"
          value={query}
          disabled={disabled}
          placeholder={t("processTree.searchPlaceholder")}
          onChange={(event) => setQuery(event.currentTarget.value)}
        />
        {query ? (
          <button type="button" className="iconOnly" disabled={disabled} aria-label={t("processTree.clearSearch")} onClick={() => setQuery("")}>
            <X size={13} />
          </button>
        ) : null}
      </label>
      <nav
        className="processTree"
        aria-label={t("processTree.label")}
        role="tree"
        onKeyDown={(event) => moveTreeFocus(event)}
      >
        {roots.map((process) => (
          <ProcessNode
            key={process.pid}
            process={process}
            selectedPid={selectedPid}
            focusPid={focusPid}
            taskLabels={taskLabels}
            pendingHumanCounts={pendingHumanCounts}
            pendingHumanLabel={t("user.pendingRequests")}
            disabled={disabled}
            childrenByPid={children}
            onSelect={onSelect}
            onFocusPid={setRovingPid}
            depth={0}
          />
        ))}
        {processes.length === 0 ? <div className="empty">{t("processTree.empty")}</div> : null}
        {processes.length > 0 && visibleProcesses.length === 0 ? <div className="empty">{t("processTree.noMatches")}</div> : null}
      </nav>
    </div>
  );
}

export function filterProcesses(
  processes: RuntimeProcess[],
  query: string,
  taskLabels: Readonly<Record<string, string>> = {},
  selectedPid: string | null = null
): RuntimeProcess[] {
  const selected = query.trim().toLocaleLowerCase();
  if (!selected) return processes;
  const byPid = new Map(processes.map((process) => [process.pid, process]));
  const included = new Set<string>();
  const includeWithAncestors = (pid: string) => {
    let currentPid: string | null = pid;
    while (currentPid && !included.has(currentPid)) {
      const current = byPid.get(currentPid);
      if (!current) break;
      included.add(currentPid);
      currentPid = current.parent_pid;
    }
  };
  for (const process of processes) {
    const haystack = [
      process.pid,
      process.image_id,
      process.status,
      process.working_directory,
      process.llm_profile_id,
      taskLabels[process.pid] ?? ""
    ].join("\n").toLocaleLowerCase();
    if (!haystack.includes(selected)) continue;
    includeWithAncestors(process.pid);
  }
  if (selectedPid) includeWithAncestors(selectedPid);
  return processes.filter((process) => included.has(process.pid));
}

export function indexProcessTree(processes: RuntimeProcess[]) {
  const roots: RuntimeProcess[] = [];
  const children = new Map<string, RuntimeProcess[]>();
  const visiblePids = new Set(processes.map((process) => process.pid));
  for (const process of processes) {
    // A source-bounded snapshot can contain a high-priority child while its
    // lower-priority ancestor falls outside the visible window. Keep that
    // child reachable instead of attaching it to an absent node.
    if (!process.parent_pid || !visiblePids.has(process.parent_pid)) {
      roots.push(process);
      continue;
    }
    const siblings = children.get(process.parent_pid);
    if (siblings) {
      siblings.push(process);
    } else {
      children.set(process.parent_pid, [process]);
    }
  }
  return { roots, children };
}

function ProcessNode({
  process,
  selectedPid,
  focusPid,
  taskLabels,
  pendingHumanCounts,
  pendingHumanLabel,
  disabled,
  childrenByPid,
  onSelect,
  onFocusPid,
  depth
}: {
  process: RuntimeProcess;
  selectedPid: string | null;
  focusPid: string | null;
  taskLabels: Readonly<Record<string, string>>;
  pendingHumanCounts: ReadonlyMap<string, number>;
  pendingHumanLabel: string;
  disabled: boolean;
  childrenByPid: Map<string, RuntimeProcess[]>;
  onSelect(pid: string): void;
  onFocusPid(pid: string): void;
  depth: number;
}) {
  const icon = iconForStatus(process.status);
  const childProcesses = childrenByPid.get(process.pid) ?? [];
  const displayLabel = taskDisplayLabel(process, taskLabels);
  const pendingHumanCount = pendingHumanCounts.get(process.pid) ?? 0;
  return (
    <div role="none">
      <button
        type="button"
        role="treeitem"
        aria-level={depth + 1}
        aria-selected={selectedPid === process.pid}
        aria-expanded={childProcesses.length ? true : undefined}
        aria-label={`${displayLabel}, ${process.pid}, ${process.status}, ${process.image_id}${pendingHumanCount ? `, ${pendingHumanLabel}: ${pendingHumanCount}` : ""}`}
        tabIndex={focusPid === process.pid ? 0 : -1}
        disabled={disabled}
        className={`processNode ${selectedPid === process.pid ? "selected" : ""}`}
        style={{ paddingLeft: 12 + depth * 14 }}
        onClick={() => onSelect(process.pid)}
        onFocus={() => onFocusPid(process.pid)}
      >
        {icon}
        <span className="processMain">
          <span className="pid" title={process.pid}>{displayLabel}</span>
          <span className="subtle">{shortProcessId(process.pid)} · {process.image_id}</span>
        </span>
        {process.interrupt_count > 0 ? <span className="badge urgent">{process.interrupt_count}</span> : null}
        {pendingHumanCount > 0 ? (
          <span className="badge human" aria-label={`${pendingHumanLabel}: ${pendingHumanCount}`} title={pendingHumanLabel}>
            <AlertTriangle size={11} aria-hidden="true" />{pendingHumanCount}
          </span>
        ) : null}
        {process.unread_message_count > 0 ? <span className="badge">{process.unread_message_count}</span> : null}
      </button>
      {childProcesses.length ? (
        <div role="group">
          {childProcesses.map((child) => (
            <ProcessNode
              key={child.pid}
              process={child}
              selectedPid={selectedPid}
              focusPid={focusPid}
              taskLabels={taskLabels}
              pendingHumanCounts={pendingHumanCounts}
              pendingHumanLabel={pendingHumanLabel}
              disabled={disabled}
              childrenByPid={childrenByPid}
              onSelect={onSelect}
              onFocusPid={onFocusPid}
              depth={depth + 1}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function moveTreeFocus(event: React.KeyboardEvent<HTMLElement>) {
  if (!new Set(["ArrowDown", "ArrowUp", "Home", "End"]).has(event.key)) return;
  const items = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>("[role='treeitem']"));
  if (!items.length) return;
  const current = items.indexOf(document.activeElement as HTMLButtonElement);
  let next = current;
  if (event.key === "ArrowDown") next = Math.min(items.length - 1, current + 1);
  if (event.key === "ArrowUp") next = Math.max(0, current - 1);
  if (event.key === "Home") next = 0;
  if (event.key === "End") next = items.length - 1;
  if (next < 0) next = 0;
  event.preventDefault();
  items[next]?.focus();
}

function iconForStatus(status: string) {
  if (status === "runnable" || status === "running") return <Play size={14} />;
  if (status.startsWith("waiting")) return <Pause size={14} />;
  if (["exited", "failed", "killed"].includes(status)) return <Square size={14} />;
  return <Circle size={14} />;
}
