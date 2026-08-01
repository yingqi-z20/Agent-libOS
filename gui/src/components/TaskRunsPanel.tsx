import { AlertTriangle, Ban, Eye, Pause, Play, RefreshCw, Repeat2, Send, ShieldAlert } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { isUnadmittedTaskRunRevisionConflict } from "../api/client";
import type { LibOSClient } from "../api/client";
import { allowedTaskRunActions, assertTaskRunSummary, taskRunActions, taskRunStatuses } from "../api/types";
import type {
  TaskRunAction,
  TaskRunBlockerKind,
  TaskRunDetail,
  TaskRunLedgerItem,
  TaskRunRecoveryOption,
  TaskRunRequirement,
  TaskRunRetention,
  TaskRunStatus,
  TaskRunSummary
} from "../api/types";
import type { ConfirmationRequest, RunGuiAction } from "../adminTypes";
import { useI18n, type TranslationKey } from "../i18n";
import { RequestEpoch } from "../requestEpoch";
import {
  submitTaskRunFollowUp,
  taskRunFollowUpIntent,
  taskRunMutationIntent,
  type TaskRunMutationIntent,
  type TaskRunMutationKind,
  type TaskRunFollowUpIntent
} from "../taskRunControl";
import { CollapsibleJson } from "./CollapsibleJson";

const terminalTaskRunStatuses = new Set<TaskRunStatus>([
  "succeeded",
  "failed",
  "cancelled"
]);

export function TaskRunsPanel({
  runs,
  client,
  runAction,
  confirmAction
}: {
  runs: TaskRunSummary[];
  client: LibOSClient;
  runAction: RunGuiAction;
  confirmAction(request: ConfirmationRequest): void;
}) {
  const { t } = useI18n();
  const [selectedId, setSelectedId] = useState(runs[0]?.run_id ?? "");
  const [ledger, setLedger] = useState<TaskRunLedgerItem[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [followUp, setFollowUp] = useState("");
  const [rerunGoal, setRerunGoal] = useState("");
  const [recoveryReceipt, setRecoveryReceipt] = useState("");
  const [detail, setDetail] = useState<TaskRunDetail | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [result, setResult] = useState<unknown>(null);
  const ledgerRequests = useRef(new RequestEpoch());
  const detailRequests = useRef(new RequestEpoch());
  const followUpIntents = useRef(new Map<string, TaskRunFollowUpIntent>());
  const mutationIntents = useRef(new Map<string, TaskRunMutationIntent>());
  const selected = runs.find((run) => run.run_id === selectedId) ?? null;
  const actions = allowedTaskRunActions(selected);
  const rerunNeedsReplacementGoal = Boolean(
    selected
    && actions.has("rerun")
    && selected.payloads_purged
    && terminalTaskRunStatuses.has(selected.status)
  );
  const hasLifecycleAction = (["run", "pause", "resume", "rerun", "cancel"] as const)
    .some((action) => actions.has(action));
  const visibleDetail = detail
    && selected
    && detail.summary.run_id === selected.run_id
    && detail.summary.revision === selected.revision
    ? detail
    : null;
  const recoveryOptions = useMemo(() => taskRunRecoveryOptions(visibleDetail), [visibleDetail]);

  useEffect(() => {
    if (!runs.some((run) => run.run_id === selectedId)) setSelectedId(runs[0]?.run_id ?? "");
  }, [runs, selectedId]);

  useEffect(() => {
    setRerunGoal("");
    setRecoveryReceipt("");
  }, [client, selectedId]);

  useEffect(() => {
    ledgerRequests.current.invalidate();
    detailRequests.current.invalidate();
    setLedger([]);
    setDetail(null);
    setNextCursor(null);
    setHasMore(false);
    if (selectedId) {
      void loadLedger(false);
      void loadDetail();
    }
    return () => {
      ledgerRequests.current.invalidate();
      detailRequests.current.invalidate();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, selectedId, selected?.revision]);

  async function loadLedger(append: boolean) {
    if (!selectedId) return;
    const epoch = ledgerRequests.current.begin();
    try {
      setLocalError(null);
      const page = await client.listTaskRunLedger(selectedId, 100, append ? nextCursor ?? undefined : undefined);
      if (!ledgerRequests.current.isCurrent(epoch)) return;
      setLedger((current) => append ? [...current, ...page.items] : page.items);
      setNextCursor(page.next_cursor);
      setHasMore(page.has_more);
    } catch (error) {
      if (ledgerRequests.current.isCurrent(epoch)) setLocalError(describe(error));
    }
  }

  async function loadDetail() {
    if (!selectedId) return;
    const epoch = detailRequests.current.begin();
    try {
      const nextDetail = await client.getTaskRun(selectedId);
      if (!detailRequests.current.isCurrent(epoch)) return;
      setDetail(nextDetail);
    } catch (error) {
      if (detailRequests.current.isCurrent(epoch)) setLocalError(describe(error));
    }
  }

  async function loadMoreRequirements() {
    if (!selectedId || !visibleDetail?.requirements.has_more || !visibleDetail.requirements.next_cursor) return;
    const epoch = detailRequests.current.begin();
    try {
      const next = await client.getTaskRun(selectedId, {
        requirementsLimit: 100,
        requirementsCursor: visibleDetail.requirements.next_cursor
      });
      if (!detailRequests.current.isCurrent(epoch)) return;
      setDetail((current) => current ? {
        ...next,
        requirements: {
          ...next.requirements,
          items: [...current.requirements.items, ...next.requirements.items]
        }
      } : next);
    } catch (error) {
      if (detailRequests.current.isCurrent(epoch)) setLocalError(describe(error));
    }
  }

  async function mutate(
    action: TaskRunMutationKind,
    request: unknown,
    operation: (intent: TaskRunMutationIntent) => Promise<TaskRunSummary>
  ): Promise<boolean> {
    if (!selected) return false;
    const run = selected;
    const intent = durableMutationIntent(run, action, request);
    let unadmittedConflict = false;
    const succeeded = await runAction(async () => {
      try {
        const response = await operation(intent);
        setResult(response);
        setSelectedId(response.run_id);
      } catch (error) {
        unadmittedConflict = isUnadmittedTaskRunRevisionConflict(error);
        throw error;
      }
    }, `task_run.${action}`);
    if (succeeded || unadmittedConflict) clearDurableMutationIntent(intent);
    return succeeded;
  }

  function durableMutationIntent(
    run: TaskRunSummary,
    action: TaskRunMutationKind,
    request: unknown = {}
  ): TaskRunMutationIntent {
    const key = mutationIntentKey(run.run_id, action);
    const intent = taskRunMutationIntent(
      mutationIntents.current.get(key) ?? null,
      {
        runId: run.run_id,
        action,
        expectedRevision: run.revision,
        request
      },
      () => newCommandId(action, run.run_id)
    );
    mutationIntents.current.set(key, intent);
    return intent;
  }

  function clearDurableMutationIntent(intent: TaskRunMutationIntent) {
    const key = mutationIntentKey(intent.runId, intent.action);
    if (mutationIntents.current.get(key) === intent) mutationIntents.current.delete(key);
  }

  async function submitFollowUp() {
    if (!selected || !actions.has("follow_up") || !followUp.trim()) return;
    const run = selected;
    let submitted: TaskRunFollowUpIntent | null = null;
    let unadmittedConflict = false;
    const succeeded = await runAction(async () => {
      const intent = await taskRunFollowUpIntent(
        followUpIntents.current.get(run.run_id) ?? null,
        {
          runId: run.run_id,
          expectedRevision: run.revision,
          body: followUp,
          kind: "normal",
          required: true
        },
        () => newCommandId("follow_up", run.run_id)
      );
      submitted = intent;
      followUpIntents.current.set(run.run_id, intent);
      try {
        const response = await submitTaskRunFollowUp(client, intent);
        setResult(response);
        setSelectedId(response.run_id);
      } catch (error) {
        unadmittedConflict = isUnadmittedTaskRunRevisionConflict(error);
        throw error;
      }
    }, "task_run.follow_up");
    const intent = submitted as TaskRunFollowUpIntent | null;
    if (succeeded && intent) {
      if (followUpIntents.current.get(intent.runId) === intent) {
        followUpIntents.current.delete(intent.runId);
      }
      setFollowUp((current) => current.trim() === intent.body ? "" : current);
    } else if (
      unadmittedConflict
      && intent
      && followUpIntents.current.get(intent.runId) === intent
    ) {
      followUpIntents.current.delete(intent.runId);
    }
  }

  function cancel() {
    if (!selected || !actions.has("cancel")) return;
    const run = selected;
    const reason = "cancelled from GUI";
    const intent = durableMutationIntent(run, "cancel", { reason });
    confirmAction({
      title: t("taskRuns.cancelTitle"),
      message: t("taskRuns.cancelMessage"),
      details: { run_id: intent.runId, expected_revision: intent.expectedRevision },
      action: async () => {
        const response = await client.cancelTaskRun(
          intent.runId,
          intent.expectedRevision,
          intent.commandId,
          true,
          reason
        );
        setResult(response);
        clearDurableMutationIntent(intent);
      },
      onErrorReconciled: (error) => {
        if (!isUnadmittedTaskRunRevisionConflict(error)) return false;
        clearDurableMutationIntent(intent);
        return true;
      }
    });
  }

  async function rerun() {
    if (!selected || !actions.has("rerun")) return;
    const run = selected;
    const replacementGoal = rerunNeedsReplacementGoal ? rerunGoal.trim() : null;
    if (rerunNeedsReplacementGoal && !replacementGoal) {
      setLocalError(t("taskRuns.rerunGoalRequired"));
      return;
    }
    const specOverrides = replacementGoal ? { goal: replacementGoal } : undefined;
    const intent = durableMutationIntent(run, "rerun", {
      spec_overrides: specOverrides ?? {}
    });
    const succeeded = await runAction(async () => {
      const response = await client.rerunTaskRun(
        intent.runId,
        intent.expectedRevision,
        intent.commandId,
        specOverrides ? { specOverrides } : undefined
      );
      setResult(response);
      setSelectedId(response.run_id);
    }, "task_run.rerun");
    if (succeeded) {
      clearDurableMutationIntent(intent);
      setRerunGoal("");
    }
  }

  function recover(option: TaskRunRecoveryOption) {
    if (!selected || !actions.has("recover")) return;
    let receipt: Record<string, unknown> | undefined;
    if (option.requires_receipt) {
      try {
        const decoded: unknown = JSON.parse(recoveryReceipt);
        if (!decoded || typeof decoded !== "object" || Array.isArray(decoded) || Object.keys(decoded).length === 0) {
          throw new Error();
        }
        receipt = decoded as Record<string, unknown>;
      } catch {
        setLocalError(t("taskRuns.receiptInvalid"));
        return;
      }
    }
    const run = selected;
    const intent = durableMutationIntent(run, "recover", {
      option_id: option.option_id,
      receipt: receipt ?? {}
    });
    confirmAction({
      title: t("taskRuns.recoverTitle"),
      message: t("taskRuns.recoverMessage"),
      details: {
        run_id: intent.runId,
        expected_revision: intent.expectedRevision,
        option_id: option.option_id,
        ...(receipt ? { receipt_present: true, receipt_field_count: Object.keys(receipt).length } : {})
      },
      action: async () => {
        const response = await client.recoverTaskRun(
          intent.runId,
          option.option_id,
          intent.expectedRevision,
          intent.commandId,
          true,
          receipt
        );
        setResult(response);
        clearDurableMutationIntent(intent);
        setRecoveryReceipt("");
      },
      onErrorReconciled: (error) => {
        if (!isUnadmittedTaskRunRevisionConflict(error)) return false;
        clearDurableMutationIntent(intent);
        return true;
      }
    });
  }

  async function explain(item: TaskRunLedgerItem) {
    const reference = ledgerExplainReference(item);
    if (!reference) return;
    await runAction(async () => {
      setResult(reference.kind === "operation"
        ? await client.explainOperation(reference.id)
        : await client.resolveOperation(reference.kind, reference.id));
    }, "task_run.explain");
  }

  return (
    <section className="adminPanel taskRunsPanel" aria-labelledby="task-runs-panel-title">
      <header className="adminPanelHeader">
        <div>
          <h3 id="task-runs-panel-title"><RefreshCw size={16} />{t("taskRuns.title")}</h3>
          <p>{t("taskRuns.description")}</p>
        </div>
      </header>

      <label className="fieldStack">
        <span>{t("taskRuns.selected")}</span>
        <select value={selectedId} disabled={!runs.length} onChange={(event) => setSelectedId(event.currentTarget.value)}>
          {!runs.length ? <option value="">{t("taskRuns.empty")}</option> : null}
          {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.display_title} · {taskRunStatusLabel(t, run.status)} · r{run.revision}</option>)}
        </select>
      </label>

      {selected ? (
        <>
          {hasLifecycleAction ? (
            <div className="adminActions taskRunLifecycleActions" aria-label={t("taskRuns.summary")}>
              {actions.has("run") ? <button onClick={() => void mutate("run", { max_quanta: null }, (intent) => client.runTaskRun(intent.runId, intent.expectedRevision, intent.commandId, null))}><Play size={14} />{taskRunActionLabel(t, "run")}</button> : null}
              {actions.has("pause") ? <button onClick={() => void mutate("pause", {}, (intent) => client.pauseTaskRun(intent.runId, intent.expectedRevision, intent.commandId))}><Pause size={14} />{taskRunActionLabel(t, "pause")}</button> : null}
              {actions.has("resume") ? <button onClick={() => void mutate("resume", {}, (intent) => client.resumeTaskRun(intent.runId, intent.expectedRevision, intent.commandId))}><Play size={14} />{taskRunActionLabel(t, "resume")}</button> : null}
              {actions.has("rerun") ? <button disabled={rerunNeedsReplacementGoal && !rerunGoal.trim()} onClick={() => void rerun()}><Repeat2 size={14} />{taskRunActionLabel(t, "rerun")}</button> : null}
              {actions.has("cancel") ? <button className="danger" onClick={cancel}><Ban size={14} />{taskRunActionLabel(t, "cancel")}</button> : null}
            </div>
          ) : null}

          {rerunNeedsReplacementGoal ? (
            <label className="fieldStack taskRunRerunGoal">
              <span>{t("taskRuns.rerunGoal")}</span>
              <textarea
                value={rerunGoal}
                onChange={(event) => setRerunGoal(event.currentTarget.value)}
                placeholder={t("taskRuns.rerunGoalPlaceholder")}
              />
              <small className="fieldHint">{t("taskRuns.rerunGoalHint")}</small>
            </label>
          ) : null}

          {taskRunStatusDetail(t, selected.status) ? (
            <div className="inlineWarning taskRunStatusDetail" role="status">
              <ShieldAlert size={15} />
              <span>{taskRunStatusDetail(t, selected.status)}</span>
            </div>
          ) : null}
          {selected.blockers.length ? (
            <section className="taskRunBlockerSection">
              <h4><AlertTriangle size={14} />{t("taskRuns.blockers")}</h4>
              <ul className="taskRunBlockers">
                {selected.blockers.map((blocker, index) => (
                  <li key={`${blocker.kind}:${blocker.effect_id ?? blocker.evidence_ref ?? index}`}>
                    <strong>{taskRunBlockerLabel(t, blocker.kind)}</strong>
                    {blocker.message ? <p>{blocker.message}</p> : null}
                    <dl>
                      {blocker.code ? <><dt>{t("taskRuns.blockerCode")}</dt><dd>{blocker.code}</dd></> : null}
                      {blocker.evidence_ref ? <><dt>{t("taskRuns.evidenceReference")}</dt><dd>{blocker.evidence_ref}</dd></> : null}
                      {blocker.process_id ? <><dt>{t("taskRuns.processReference")}</dt><dd>{blocker.process_id}</dd></> : null}
                      {blocker.effect_id ? <><dt>{t("taskRuns.effectReference")}</dt><dd>{blocker.effect_id}</dd></> : null}
                    </dl>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
          <section className="taskRunRequirements">
            <h4>{t("taskRuns.requirements")}</h4>
            <TaskRunRequirementCounts counts={selected.requirement_counts ?? {}} t={t} />
            {visibleDetail?.requirements.items.length ? (
              <ol>
                {visibleDetail.requirements.items.map((requirement) => (
                  <li key={requirement.requirement_id}>
                    <div>
                      <strong>{requirement.label}</strong>
                      <span>{taskRunRequirementKindLabel(t, requirement.kind)} · {taskRunRequirementStatusLabel(t, requirement.status)}</span>
                    </div>
                    {requirement.content_available && requirement.content_text !== undefined
                      ? <pre>{requirement.content_text}{requirement.content_truncated ? "…" : ""}</pre>
                      : <small>{t("taskRuns.hashOnly")}: {requirement.content_sha256}</small>}
                  </li>
                ))}
              </ol>
            ) : null}
            {visibleDetail?.requirements.has_more ? <button onClick={() => void loadMoreRequirements()}>{t("taskRuns.loadMore")}</button> : null}
          </section>
          <p className="taskRunRetentionPolicy">{t("taskRuns.retentionConfigured")}: <strong>{taskRunRetentionLabel(t, selected.retention)}</strong></p>
          <p className="taskRunPayloadState">{t("taskRuns.payloadState")}: <strong>{t(selected.payloads_purged ? "taskRuns.payloadPurged" : "taskRuns.payloadAvailable")}</strong></p>

          {actions.has("follow_up") ? (
            <div className="adminInlineForm">
              <input aria-label={t("taskRuns.followUp")} value={followUp} placeholder={t("taskRuns.followUpPlaceholder")} onChange={(event) => setFollowUp(event.currentTarget.value)} />
              <button disabled={!followUp.trim()} onClick={() => void submitFollowUp()}><Send size={14} />{taskRunActionLabel(t, "follow_up")}</button>
            </div>
          ) : null}

          {actions.has("recover") ? (
            <div className="taskRunRecoveryControls">
              {recoveryOptions.some((option) => option.requires_receipt) ? (
                <label className="fieldStack">
                  <span>{t("taskRuns.receipt")}</span>
                  <textarea
                    value={recoveryReceipt}
                    onChange={(event) => setRecoveryReceipt(event.currentTarget.value)}
                    placeholder={t("taskRuns.receiptPlaceholder")}
                  />
                  <small className="fieldHint">{t("taskRuns.receiptHint")}</small>
                </label>
              ) : null}
              <div className="adminActions">
                {recoveryOptions.map((option) => (
                  <button
                    className="warning"
                    key={option.option_id}
                    disabled={Boolean(option.requires_receipt && !recoveryReceipt.trim())}
                    onClick={() => recover(option)}
                  >
                    <ShieldAlert size={14} />{taskRunRecoveryOptionLabel(t, option)}
                  </button>
                ))}
                {!recoveryOptions.length ? <span className="inlineError">{t("taskRuns.noRecoveryOptions")}</span> : null}
              </div>
            </div>
          ) : null}

          <CollapsibleJson value={taskRunSummaryForDisplay(t, selected)} label={t("taskRuns.summary")} />
        </>
      ) : null}

      <section>
        <h4>{t("taskRuns.ledger")}</h4>
        {!ledger.length ? <p className="empty">{t("taskRuns.ledgerEmpty")}</p> : (
          <ol className="taskRunLedger">
            {ledger.map((item, index) => {
              const explainable = ledgerExplainReference(item);
              return (
                <li key={item.item_id || `${item.kind}:${item.seq}:${index}`}>
                  <span className="taskRunLedgerIdentity">
                    <strong>{taskRunLedgerKindLabel(t, item.kind)}</strong>
                    {item.label ? <small>{taskRunLedgerValueLabel(t, item.label)}</small> : null}
                  </span>
                  <strong className="taskRunLedgerStatus">{taskRunLedgerValueLabel(t, item.status)}</strong>
                  {explainable ? <button onClick={() => void explain(item)}><Eye size={13} />{t("details.explain")}</button> : null}
                </li>
              );
            })}
          </ol>
        )}
        {hasMore ? <button onClick={() => void loadLedger(true)}>{t("taskRuns.loadMore")}</button> : null}
      </section>
      {localError ? <div className="inlineError" role="alert">{localError}</div> : null}
      {result !== null ? <CollapsibleJson value={taskRunResultForDisplay(t, result)} label={t("tasks.result")} defaultExpanded /> : null}
    </section>
  );
}

export function taskRunRecoveryOptions(detail: TaskRunDetail | null): TaskRunRecoveryOption[] {
  if (!detail) return [];
  const raw = detail.recovery_options;
  if (!Array.isArray(raw)) return [];
  return raw.filter((item): item is TaskRunRecoveryOption => Boolean(
    item && typeof item === "object" && !Array.isArray(item)
      && typeof (item as { option_id?: unknown }).option_id === "string"
      && (item as { option_id: string }).option_id
  ));
}

function ledgerExplainReference(item: TaskRunLedgerItem): { kind: string; id: string } | null {
  if (typeof item.operation_id === "string" && item.operation_id) return { kind: "operation", id: item.operation_id };
  if (typeof item.effect_id === "string" && item.effect_id) return { kind: "effect", id: item.effect_id };
  if (typeof item.human_request_id === "string" && item.human_request_id) return { kind: "request", id: item.human_request_id };
  if (typeof item.llm_call_id === "string" && item.llm_call_id) return { kind: "call", id: item.llm_call_id };
  return null;
}

function newCommandId(action: string, runId: string): string {
  const random = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `gui:${action}:${runId}:${random}`;
}

function mutationIntentKey(runId: string, action: TaskRunMutationKind): string {
  return `${runId}\u0000${action}`;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;

const requirementStatuses = ["pending", "in_progress", "satisfied", "blocked", "waived"] as const;
const requirementKinds = ["initial", "follow_up"] as const;
const retentions = ["purge_on_terminal", "permanent"] as const;

export function taskRunStatusLabel(t: Translate, status: TaskRunStatus): string {
  return t(`taskRun.status.${status}`);
}

export function taskRunRetentionLabel(t: Translate, retention: TaskRunRetention): string {
  return t(`taskRun.retention.${retention}`);
}

export function taskRunRequirementStatusLabel(
  t: Translate,
  status: TaskRunRequirement["status"]
): string {
  return t(`taskRun.requirementStatus.${status}`);
}

export function taskRunRequirementKindLabel(
  t: Translate,
  kind: TaskRunRequirement["kind"]
): string {
  return t(`taskRun.requirementKind.${kind}`);
}

export function taskRunBlockerLabel(t: Translate, kind: TaskRunBlockerKind): string {
  return t(`taskRun.blocker.${kind}`);
}

export function taskRunLedgerKindLabel(t: Translate, kind: TaskRunLedgerItem["kind"]): string {
  return t(`taskRun.ledgerKind.${kind}`);
}

export function taskRunActionLabel(t: Translate, action: TaskRunAction): string {
  return t(`taskRun.action.${action}`);
}

function taskRunStatusDetail(t: Translate, status: TaskRunStatus): string | null {
  if (status === "cancelling" || status === "finalizing" || status === "needs_attention") {
    return t(`taskRun.detail.${status}`);
  }
  return null;
}

function taskRunRequirementStatusValueLabel(t: Translate, status: string): string {
  if (status === "total") return t("taskRun.requirementCount.total");
  if ((requirementStatuses as readonly string[]).includes(status)) {
    return taskRunRequirementStatusLabel(t, status as TaskRunRequirement["status"]);
  }
  return status;
}

function taskRunLedgerValueLabel(t: Translate, value: string): string {
  if (!value) return "—";
  if ((taskRunStatuses as readonly string[]).includes(value)) {
    return taskRunStatusLabel(t, value as TaskRunStatus);
  }
  if ((requirementStatuses as readonly string[]).includes(value)) {
    return taskRunRequirementStatusLabel(t, value as TaskRunRequirement["status"]);
  }
  if ((taskRunActions as readonly string[]).includes(value)) {
    return taskRunActionLabel(t, value as TaskRunAction);
  }
  if ((requirementKinds as readonly string[]).includes(value)) {
    return taskRunRequirementKindLabel(t, value as TaskRunRequirement["kind"]);
  }
  if ((retentions as readonly string[]).includes(value)) {
    return taskRunRetentionLabel(t, value as TaskRunRetention);
  }
  return value;
}

function taskRunRecoveryOptionLabel(t: Translate, option: TaskRunRecoveryOption): string {
  if (option.kind === "effect_receipt") {
    return t("taskRun.recoveryOption.effect_receipt", {
      effectId: option.effect_id ?? "—",
      state: taskRunEffectTransactionStateLabel(t, option.expected_transaction_state),
      epoch: option.runtime_epoch ?? "—"
    });
  }
  if (option.kind === "linked_rerun") return t("taskRun.recoveryOption.linked_rerun");
  if (option.kind === "terminalize") return t("taskRun.recoveryOption.terminalize");
  if (option.kind && (taskRunActions as readonly string[]).includes(option.kind)) {
    return taskRunActionLabel(t, option.kind as TaskRunAction);
  }
  if (option.label) return taskRunLedgerValueLabel(t, option.label);
  return `${taskRunActionLabel(t, "recover")} · ${option.option_id}`;
}

function taskRunEffectTransactionStateLabel(t: Translate, state: string | undefined): string {
  if (state && ["prepared", "authorized", "approved", "dispatched", "committed", "failed", "unknown", "compensated"].includes(state)) {
    return t(`taskRun.effectState.${state}` as TranslationKey);
  }
  return "—";
}

function TaskRunRequirementCounts({
  counts,
  t
}: {
  counts: Record<string, number>;
  t: Translate;
}) {
  const total = Number.isSafeInteger(counts.total) && counts.total >= 0 ? counts.total : 0;
  const satisfied = Number.isSafeInteger(counts.satisfied) && counts.satisfied >= 0
    ? counts.satisfied
    : 0;
  return (
    <p className="taskRunRequirementCounts" aria-label={t("taskRuns.requirementCounts")}>
      {t("taskRuns.requirementProgress", { satisfied, total })}
    </p>
  );
}

function taskRunSummaryForDisplay(t: Translate, summary: TaskRunSummary): Record<string, unknown> {
  return {
    ...summary,
    status: taskRunStatusLabel(t, summary.status),
    allowed_actions: summary.allowed_actions.map((action) => taskRunActionLabel(t, action)),
    blockers: summary.blockers.map((blocker) => ({
      ...blocker,
      kind: taskRunBlockerLabel(t, blocker.kind)
    })),
    retention: taskRunRetentionLabel(t, summary.retention),
    ...(summary.requirement_counts ? {
      requirement_counts: Object.fromEntries(Object.entries(summary.requirement_counts).map(([status, count]) => [
        taskRunRequirementStatusValueLabel(t, status),
        count
      ]))
    } : {})
  };
}

function taskRunResultForDisplay(t: Translate, value: unknown): unknown {
  try {
    assertTaskRunSummary(value);
  } catch {
    return value;
  }
  return taskRunSummaryForDisplay(t, value);
}
