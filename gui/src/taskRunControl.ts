import type { OptionalQuanta } from "./quanta";
import type { TaskRunDetail, TaskRunSpecV1, TaskRunSummary } from "./api/types";

export type TaskRunStartIntent = {
  fingerprint: string;
  specFingerprint: string;
  runFingerprint: string;
  clientRequestId: string;
  runCommandId: string;
  runId: string | null;
  runExpectedRevision: number | null;
};

export type TaskRunMutationKind =
  | "run"
  | "pause"
  | "resume"
  | "cancel"
  | "recover"
  | "rerun";

/**
 * One immutable revision-fenced HTTP mutation.  This intentionally lives only
 * in renderer memory: request payloads such as recovery receipts must never be
 * copied into browser storage merely to make transport retries idempotent.
 */
export type TaskRunMutationIntent = {
  runId: string;
  action: TaskRunMutationKind;
  expectedRevision: number;
  requestFingerprint: string;
  commandId: string;
};

export type TaskRunFollowUpKind = "normal" | "interrupt";

/**
 * One immutable follow-up request.  In particular, ``expectedRevision`` and
 * ``commandId`` must survive an ambiguous HTTP response and any intervening
 * SSE summary so retrying the same draft replays the server command instead of
 * appending a second requirement.
 */
export type TaskRunFollowUpIntent = {
  runId: string;
  expectedRevision: number;
  body: string;
  kind: TaskRunFollowUpKind;
  required: boolean;
  requestHash: string;
  commandId: string;
};

type TaskRunStartClient = {
  createTaskRun(
    spec: TaskRunSpecV1,
    clientRequestId: string
  ): Promise<TaskRunSummary>;
  runTaskRun(runId: string, expectedRevision: number, commandId: string, maxQuanta: OptionalQuanta): Promise<TaskRunSummary>;
  getTaskRun(runId: string): Promise<TaskRunDetail>;
};

type TaskRunFollowUpClient = {
  followUpTaskRun(
    runId: string,
    body: string,
    expectedRevision: number,
    commandId: string,
    options: { kind: TaskRunFollowUpKind; required: boolean }
  ): Promise<TaskRunSummary>;
};

export function taskRunStartIntent(
  previous: TaskRunStartIntent | null,
  spec: TaskRunSpecV1,
  maxQuanta: OptionalQuanta,
  makeId: (kind: "create" | "run") => string
): TaskRunStartIntent {
  const specFingerprint = stableJson(spec);
  const runFingerprint = stableJson({ max_quanta: maxQuanta });
  const fingerprint = stableJson({ spec, max_quanta: maxQuanta });
  if (
    previous?.specFingerprint === specFingerprint
    && previous.runFingerprint === runFingerprint
  ) return previous;
  if (previous?.specFingerprint === specFingerprint) {
    return {
      ...previous,
      fingerprint,
      runFingerprint,
      runCommandId: makeId("run"),
      runId: null,
      runExpectedRevision: null
    };
  }
  return {
    fingerprint,
    specFingerprint,
    runFingerprint,
    clientRequestId: makeId("create"),
    runCommandId: makeId("run"),
    runId: null,
    runExpectedRevision: null
  };
}

export function bindTaskRunStartIntent(
  intent: TaskRunStartIntent,
  created: TaskRunSummary
): TaskRunStartIntent {
  if (intent.runId !== null || intent.runExpectedRevision !== null) {
    if (intent.runId !== created.run_id || intent.runExpectedRevision === null) {
      throw new Error("TaskRun create replay changed the bound run identity.");
    }
    return intent;
  }
  return {
    ...intent,
    runId: created.run_id,
    runExpectedRevision: created.revision
  };
}

export function rotateUnadmittedTaskRunStartCommand(
  intent: TaskRunStartIntent,
  makeRunCommandId: () => string
): TaskRunStartIntent {
  return {
    ...intent,
    runCommandId: makeRunCommandId(),
    runId: null,
    runExpectedRevision: null
  };
}

/**
 * Reuse the original revision and command id for an exact retry even if an
 * intervening SSE/HTTP snapshot has advanced the visible Run revision.  A
 * changed request body is a new logical command and therefore receives a new
 * id; callers retain the returned intent until an HTTP success is observed.
 */
export function taskRunMutationIntent(
  previous: TaskRunMutationIntent | null,
  input: {
    runId: string;
    action: TaskRunMutationKind;
    expectedRevision: number;
    request?: unknown;
  },
  makeCommandId: () => string
): TaskRunMutationIntent {
  if (!Number.isSafeInteger(input.expectedRevision) || input.expectedRevision < 0) {
    throw new Error("TaskRun mutation expected revision must be a non-negative safe integer.");
  }
  const requestFingerprint = stableJson(input.request ?? {});
  if (
    previous
    && previous.runId === input.runId
    && previous.action === input.action
    && previous.requestFingerprint === requestFingerprint
  ) {
    return previous;
  }
  return {
    runId: input.runId,
    action: input.action,
    expectedRevision: input.expectedRevision,
    requestFingerprint,
    commandId: makeCommandId()
  };
}

/**
 * Standard-user flow: idempotently create queued state, then consume quanta
 * through an independently fenced command. A failed run response is reconciled
 * through authoritative detail before the error is rethrown to the UI.
 */
export async function createAndRunTaskRun(
  client: TaskRunStartClient,
  spec: TaskRunSpecV1,
  intent: TaskRunStartIntent,
  maxQuanta: OptionalQuanta,
  callbacks: {
    onCreated(summary: TaskRunSummary): void;
    onIntent(intent: TaskRunStartIntent): void;
    onSummary(summary: TaskRunSummary): void;
  }
): Promise<TaskRunSummary> {
  const created = await client.createTaskRun(spec, intent.clientRequestId);
  const boundIntent = bindTaskRunStartIntent(intent, created);
  callbacks.onIntent(boundIntent);
  callbacks.onCreated(created);
  callbacks.onSummary(created);
  try {
    const running = await client.runTaskRun(
      boundIntent.runId!,
      boundIntent.runExpectedRevision!,
      boundIntent.runCommandId,
      maxQuanta
    );
    callbacks.onSummary(running);
    return running;
  } catch (error) {
    try {
      const detail = await client.getTaskRun(created.run_id);
      if (detail.summary.revision >= created.revision) callbacks.onSummary(detail.summary);
    } catch {
      // The original mutation error remains the useful outward failure.
    }
    throw error;
  }
}

export async function taskRunFollowUpIntent(
  previous: TaskRunFollowUpIntent | null,
  input: {
    runId: string;
    expectedRevision: number;
    body: string;
    kind: TaskRunFollowUpKind;
    required: boolean;
  },
  makeCommandId: () => string
): Promise<TaskRunFollowUpIntent> {
  const body = input.body.trim();
  if (!body) throw new Error("TaskRun follow-up body must be non-empty.");
  const request = { body, kind: input.kind, required: input.required };
  const requestHash = await sha256CanonicalJson(request);
  if (
    previous
    && previous.runId === input.runId
    && previous.requestHash === requestHash
  ) {
    return previous;
  }
  return {
    runId: input.runId,
    expectedRevision: input.expectedRevision,
    body,
    kind: input.kind,
    required: input.required,
    requestHash,
    commandId: makeCommandId()
  };
}

export function submitTaskRunFollowUp(
  client: TaskRunFollowUpClient,
  intent: TaskRunFollowUpIntent
): Promise<TaskRunSummary> {
  return client.followUpTaskRun(
    intent.runId,
    intent.body,
    intent.expectedRevision,
    intent.commandId,
    { kind: intent.kind, required: intent.required }
  );
}

export function clearTaskRunFollowUpDraft(
  drafts: Readonly<Record<string, string>>,
  intent: TaskRunFollowUpIntent
): Record<string, string> {
  const key = `run:${intent.runId}`;
  if (!(key in drafts) || drafts[key]?.trim() !== intent.body) return drafts;
  const next = { ...drafts };
  delete next[key];
  return next;
}

async function sha256CanonicalJson(value: unknown): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("TaskRun mutation intents require Web Crypto SHA-256 support.");
  const digest = await subtle.digest("SHA-256", new TextEncoder().encode(stableJson(value)));
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${asciiJsonString(key)}:${stableJson(item)}`)
      .join(",")}}`;
  }
  return typeof value === "string" ? asciiJsonString(value) : JSON.stringify(value) ?? "null";
}

function asciiJsonString(value: string): string {
  return JSON.stringify(value).replace(/[\u007f-\uffff]/g, (character) => (
    `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`
  ));
}
