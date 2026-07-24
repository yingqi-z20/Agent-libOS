import type { RuntimeProcess } from "./api/types";

const MAX_TASK_LABEL_CHARACTERS = 72;
const MAX_STORED_TASK_LABELS = 200;

export function taskLabelFromGoal(goal: string): string {
  const normalized = goal.trim().replace(/\s+/g, " ");
  const characters = Array.from(normalized);
  if (characters.length <= MAX_TASK_LABEL_CHARACTERS) return normalized;
  return `${characters.slice(0, MAX_TASK_LABEL_CHARACTERS - 1).join("").trimEnd()}…`;
}

export function shortProcessId(pid: string): string {
  if (pid.length <= 18) return pid;
  const prefixLength = pid.startsWith("pid_") ? 12 : 10;
  return `${pid.slice(0, prefixLength)}…${pid.slice(-4)}`;
}

export function taskDisplayLabel(
  process: RuntimeProcess,
  labels: Readonly<Record<string, string>>
): string {
  return labels[process.pid]?.trim() || shortProcessId(process.pid);
}

export function taskLabelsFromStorage(raw: string | null): Record<string, string> {
  if (!raw) return {};
  try {
    const value = JSON.parse(raw) as unknown;
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    return Object.fromEntries(
      Object.entries(value)
        .filter(([pid, label]) => /^pid_[A-Za-z0-9_-]{1,156}$/.test(pid) && typeof label === "string")
        .slice(-MAX_STORED_TASK_LABELS)
        .map(([pid, label]) => [pid, taskLabelFromGoal(label as string)])
        .filter(([, label]) => Boolean(label))
    );
  } catch {
    return {};
  }
}

export function taskLabelsForStorage(labels: Readonly<Record<string, string>>): string {
  return JSON.stringify(Object.fromEntries(
    Object.entries(labels)
      .filter(([pid, label]) => /^pid_[A-Za-z0-9_-]{1,156}$/.test(pid) && Boolean(label.trim()))
      .slice(-MAX_STORED_TASK_LABELS)
      .map(([pid, label]) => [pid, taskLabelFromGoal(label)])
  ));
}
