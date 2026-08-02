from __future__ import annotations

import asyncio
import contextvars
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import Any, TYPE_CHECKING, Iterable

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    CapabilityRight,
    CapabilitySpec,
    DataLabels,
    EventPriority,
    EventType,
    HumanRequestStatus,
    ObjectHandle,
    ObjectOwnerKind,
    ObjectRight,
    ObjectTask,
    ObjectTaskNotification,
    ObjectTaskNotificationStatus,
    ObjectTaskOwnerWatch,
    ObjectTaskRecoveryCursor,
    ObjectTaskRecoveryKind,
    ObjectTaskRecoverySummary,
    ObjectTaskStatus,
    ProcessMessageKind,
    ProcessStatus,
    RelationType,
    ToolHandle,
    ToolProcessWait,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProcessError,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    ValidationError,
)
from agent_libos.process_execution import trusted_process_control_mutation
from agent_libos.runtime.object_task_notifications import (
    ObjectTaskNotificationService,
)
from agent_libos.runtime.object_task_state import (
    OBJECT_TASK_TERMINAL_STATUSES,
    ObjectTaskStateService,
)
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)

if TYPE_CHECKING:
    from agent_libos.capability.manager import CapabilityManager
    from agent_libos.human.manager import HumanObjectManager
    from agent_libos.memory.object_memory import ObjectMemoryManager
    from agent_libos.ports import AuditPort, EventPort, OperationPort
    from agent_libos.runtime.authority_manifest_manager import AuthorityManifestManager
    from agent_libos.runtime.message_manager import ProcessMessageManager
    from agent_libos.runtime.process_manager import ProcessManager
    from agent_libos.storage import ObjectRepository, ProcessRepository
    from agent_libos.tools.broker import ToolBroker


_TERMINAL_STATUSES = OBJECT_TASK_TERMINAL_STATUSES
_OWNER_WATCH_EVENTS = {"updated", "linked"}
_MESSAGE_REPLAY_SAFE_TOOLS = {"receive_process_messages"}
_PROCESS_REPLAY_SAFE_TOOLS = {"wait_child_process"}


@dataclass(frozen=True, slots=True)
class _PendingNotifyResultGrant:
    recipient_pid: str
    result_oid: str
    reservation_id: str | None
    used_by: str


@dataclass(slots=True)
class _TaskResultSettlement:
    result_oid: str | None = None
    pending_notify_grant: _PendingNotifyResultGrant | None = None


class _NotifyResultGrantPublicationFailed(RuntimeError):
    pass


class _ObjectTaskLifecycleShutdownHandle:
    """Opaque teardown-only view retained by RuntimeLifecycle."""

    __slots__ = ("__release_recovery_diagnostics", "__shutdown")

    def __init__(
        self,
        *,
        shutdown: Callable[[], bool],
        release_recovery_diagnostics: Callable[[], bool],
    ) -> None:
        self.__shutdown = shutdown
        self.__release_recovery_diagnostics = release_recovery_diagnostics

    def shutdown(self) -> bool:
        return self.__shutdown()

    def release_recovery_diagnostics(self) -> bool:
        """Abandon transient workers without publishing durable cancellation."""

        return self.__release_recovery_diagnostics()


class ObjectTaskManager:
    """Background Object-bound tool tasks.

    Object tasks are host-managed execution records, not model scheduler work.
    Each task runs a single visible tool through a dedicated child process so
    the normal process tool table, capability, resource, event, and audit
    boundaries remain authoritative.
    """

    def __init__(
        self,
        tasks: "ProcessRepository",
        objects: "ObjectRepository",
        process: "ProcessManager",
        tools: "ToolBroker",
        memory: "ObjectMemoryManager",
        capabilities: "CapabilityManager",
        audit: "AuditPort",
        events: "EventPort",
        operations: "OperationPort",
        messages: "ProcessMessageManager",
        authority_manifests: "AuthorityManifestManager",
        human: "HumanObjectManager",
        add_handle_to_process_view: Callable[[str, ObjectHandle], None],
        config: AgentLibOSConfig | None = None,
        *,
        admission: Any,
        require_recovery_lease: Callable[[], None],
        autostart: bool = True,
    ) -> None:
        self._records = tasks
        self._objects = objects
        self._process = process
        self._tools = tools
        self._memory = memory
        self._capabilities = capabilities
        self._audit = audit
        self._events = events
        self._operations = operations
        self._messages = messages
        self._authority_manifests = authority_manifests
        self._human = human
        self._add_handle_to_process_view = add_handle_to_process_view
        self._admission = admission
        self._require_recovery_lease = require_recovery_lease
        self.config = config or DEFAULT_CONFIG
        self._lock = threading.RLock()
        self._notifications = ObjectTaskNotificationService(
            self._records,
            self._messages,
            self._events,
            self._process.TERMINAL_STATUSES,
            self._lock,
        )
        self._state = ObjectTaskStateService(
            self._records,
            self._objects,
            self._process,
            self._memory,
            self._audit,
            self._events,
            self._notifications,
            self._lock,
        )
        self._loop = asyncio.new_event_loop()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.object_tasks.max_running_global,
            thread_name_prefix="agent-libos-object-task-tool",
        )
        self._loop_ready = threading.Event()
        self._worker_cleanup_started = threading.Event()
        self._worker_shutdown_complete = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="agent-libos-object-tasks",
            daemon=True,
        )
        self._futures: dict[str, Future[Any]] = {}
        # concurrent.futures.Future becomes done immediately when cancellation
        # is requested, even if the event-loop coroutine is still unwinding a
        # synchronous section. Track actual coroutine lifetime separately so
        # wait()/shutdown do not expose terminal task state before cleanup.
        self._active_runs: set[str] = set()
        self._grant_result_to_notify: dict[str, bool] = {}
        self._pending_args: dict[str, dict[str, Any]] = {}
        self._closing = False
        self._closed = False
        self._worker_failure: str | None = None
        self.__lifecycle_shutdown_capability = object()
        self._recovered = False
        self._recovery_summary = ObjectTaskRecoverySummary()
        self._started = False
        if autostart:
            try:
                self.recover()
                self.start_worker()
            except BaseException:
                self._cleanup_failed_initialization()
                raise

    def recover(self) -> ObjectTaskRecoverySummary:
        self._require_recovery_lease()
        with self._lock:
            if self._recovered:
                return self._recovery_summary
            if self._started:
                raise RuntimeError("object task recovery must precede worker startup")
            self._recovery_summary = self._recover_persisted_tasks()
            self._recovered = True
            return self._recovery_summary

    def start_worker(self) -> None:
        with self._lock:
            if self._started:
                return
            if not self._recovered:
                raise RuntimeError("object task worker cannot start before recovery")
            if self._closing or self._closed:
                raise RuntimeError("object task manager is shutting down")
            self._thread.start()
            self._started = True
        if not self._loop_ready.wait(timeout=1.0):
            raise RuntimeError("object task event loop did not start")

    def _recover_persisted_tasks(self) -> ObjectTaskRecoverySummary:
        page_size = self.config.runtime.object_task_recovery_page_size
        abandoned_total, abandoned_sample = self._recover_active_tasks(page_size)
        unavailable_total, unavailable_sample = self._recover_missing_results(
            page_size
        )
        notification_total, notification_sample = self._retry_notifications(
            page_size
        )
        return ObjectTaskRecoverySummary(
            abandoned_total=abandoned_total,
            result_unavailable_total=unavailable_total,
            notification_retried_total=notification_total,
            abandoned_sample=tuple(abandoned_sample),
            result_unavailable_sample=tuple(unavailable_sample),
            notification_retried_sample=tuple(notification_sample),
        )

    def _recover_active_tasks(self, page_size: int) -> tuple[int, list[str]]:
        total = 0
        sample: list[str] = []
        cursor: ObjectTaskRecoveryCursor | None = None
        reason = "runtime reopened before object task completed"
        while True:
            page = self._records.query_object_task_recovery(
                kind=ObjectTaskRecoveryKind.ACTIVE,
                after=cursor,
                limit=page_size,
            )
            for task in page.records:
                now = utc_now()
                with self._records.transaction():
                    updated = self._records.abandon_object_task_after_reopen(
                        task.task_id,
                        expected_status=task.status,
                        reason=reason,
                        updated_at=now,
                    )
                    if updated is None:
                        continue
                    updated = self._notifications.begin_phase(updated)
                    self._records.update_object_task(updated)
                    if updated.runner_pid is not None:
                        self._state.terminalize_runner(
                            str(updated.runner_pid),
                            reason="object task abandoned after runtime reopen",
                        )
                    self._state.cleanup_owner_pin_after_terminal(updated)
                updated = self._notifications.notify_terminal(
                    updated,
                    phase="abandoned",
                )
                total += 1
                self._append_recovery_sample(sample, updated.task_id, page_size)
            cursor = page.next_cursor
            if cursor is None:
                break
        self._record_recovery_summary(
            action="object_task.abandon_recovered",
            total=total,
            sample=sample,
        )
        return total, sample

    def _recover_missing_results(self, page_size: int) -> tuple[int, list[str]]:
        total = 0
        sample: list[str] = []
        cursor: ObjectTaskRecoveryCursor | None = None
        while True:
            page = self._records.query_object_task_recovery(
                kind=ObjectTaskRecoveryKind.MISSING_RESULT,
                after=cursor,
                limit=page_size,
            )
            for task in page.records:
                if self._recover_missing_result(task) is None:
                    continue
                total += 1
                self._append_recovery_sample(sample, task.task_id, page_size)
            cursor = page.next_cursor
            if cursor is None:
                break
        self._record_recovery_summary(
            action="object_task.result_unavailable_recovered",
            total=total,
            sample=sample,
        )
        return total, sample

    def _recover_missing_result(self, task: ObjectTask) -> ObjectTask | None:
        result_oid = str(task.result_oid)
        if self._objects.get_object(result_oid) is not None:
            return None
        now = utc_now()
        wait = {
            **task.wait,
            "result_unavailable_after_reopen": True,
            "result_unavailable_at": now,
            "previous_status": task.status.value,
            "previous_result_oid": result_oid,
            "previous_error": task.error,
        }
        with self._records.transaction():
            updated = self._records.mark_object_task_result_unavailable_after_reopen(
                task.task_id,
                expected_result_oid=result_oid,
                wait=wait,
                error=(
                    "result unavailable after runtime reopen: runtime-only Object "
                    "payload was not reconstructable"
                ),
                updated_at=now,
            )
            if updated is None:
                return None
            updated = self._notifications.begin_phase(updated)
            self._records.update_object_task(updated)
        return self._notifications.notify_terminal(
            updated,
            phase="result_unavailable_after_reopen",
        )

    def _retry_notifications(self, page_size: int) -> tuple[int, list[str]]:
        total = 0
        sample: list[str] = []
        cursor: ObjectTaskRecoveryCursor | None = None
        while True:
            page = self._records.query_object_task_recovery(
                kind=ObjectTaskRecoveryKind.NOTIFICATION,
                after=cursor,
                limit=page_size,
            )
            for task in page.records:
                self._notifications.retry_terminal(task)
                total += 1
                self._append_recovery_sample(sample, task.task_id, page_size)
            cursor = page.next_cursor
            if cursor is None:
                break
        self._record_recovery_summary(
            action="object_task.notification_retried_recovered",
            total=total,
            sample=sample,
        )
        return total, sample

    def _record_recovery_summary(
        self,
        *,
        action: str,
        total: int,
        sample: list[str],
    ) -> None:
        if total == 0:
            return
        self._audit.record(
            actor="object_task",
            action=action,
            target="object_tasks",
            decision={
                "task_ids": list(sample),
                "total_count": total,
                "truncated": total > len(sample),
            },
        )

    @staticmethod
    def _append_recovery_sample(
        sample: list[str],
        task_id: str,
        limit: int,
    ) -> None:
        if len(sample) < limit:
            sample.append(str(task_id))

    def _cleanup_failed_initialization(self) -> None:
        self._closing = True
        if self._thread.is_alive():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass
            self._thread.join(timeout=1.0)
        self._executor.shutdown(wait=False, cancel_futures=True)
        if not self._loop.is_running() and not self._loop.is_closed():
            self._loop.close()

    def start(
        self,
        pid: str,
        owner: ObjectHandle,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        notify_pid: str | None = None,
        notify_kind: str | ProcessMessageKind = ProcessMessageKind.NORMAL,
        notify_channel: str | None = None,
        inherit_capabilities: Iterable[CapabilitySpec | dict[str, Any]] | None = None,
        grant_result_to_notify: bool = False,
        owner_watch: bool | dict[str, Any] | ObjectTaskOwnerWatch | None = None,
    ) -> ObjectTask:
        self._ensure_open()
        if type(grant_result_to_notify) is not bool:
            raise ValidationError(
                "object task grant_result_to_notify must be a boolean"
            )
        tool_name = str(tool).strip()
        if not tool_name:
            raise ValidationError("object task tool name is required")
        task_args = dict(args or {})
        selected_kind = self._message_kind(notify_kind, "object task notify_kind")
        selected_notify_pid = notify_pid or pid
        selected_owner_watch = self._normalize_owner_watch(owner_watch)
        selected_notify_channel = self._messages.normalize_channel(
            notify_channel or self.config.object_tasks.notification_channel
        )
        self._require_related_notification_target(pid, selected_notify_pid)
        owner_rights = {
            ObjectRight.READ.value,
            ObjectRight.WRITE.value,
            ObjectRight.LINK.value,
        }
        self._assert_owner_rights(pid, owner, owner_rights)
        if self._objects.get_object(owner.oid) is None:
            raise NotFound(f"object not found: {owner.oid}")
        handle = self._tools.resolve(tool_name, pid=pid)
        if not self._tools.process_has_tool(pid, handle):
            raise ValidationError(f"tool is not in process tool table: {tool_name}")
        self._require_concurrency_capacity(owner.oid)
        spawn_decision = self._capabilities.require(
            pid,
            "process:spawn",
            CapabilityRight.WRITE,
            consume=False,
        )

        task_id = new_id("otask")
        owner_reservations = self._reserve_owner_decisions(
            pid,
            owner,
            owner_rights,
        )
        spawn_reservation = self._capabilities.reserve_decision_use(
            spawn_decision,
            used_by=f"object_task:{task_id}",
            reason="one-time object task spawn permission reserved",
        )
        runner_pid: str | None = None
        task_inserted = False
        try:
            runner_pid = self._process.spawn_child(
                pid,
                goal={"type": "object_task", "task_id": task_id, "owner_oid": owner.oid, "tool": tool_name},
                inherit_capabilities=[dict(spec) if isinstance(spec, dict) else spec.__dict__ for spec in (inherit_capabilities or [])],
                initial_status=ProcessStatus.WAITING_TOOL,
                initial_wait_state=ToolProcessWait(operation_id=task_id),
            )
            with self._memory.ownership_locked():
                self._bootstrap_runner(runner_pid, handle, owner.oid, task_id)

            now = utc_now()
            task = ObjectTask(
                task_id=task_id,
                owner_oid=owner.oid,
                creator_pid=pid,
                runner_pid=runner_pid,
                tool=tool_name,
                tool_id=handle.tool_id,
                status=ObjectTaskStatus.QUEUED,
                notification=ObjectTaskNotification(
                    recipient_pid=selected_notify_pid,
                    kind=selected_kind.value,
                    channel=selected_notify_channel,
                ),
                owner_watch=selected_owner_watch,
                created_at=now,
                updated_at=now,
            )
            # Ownership changes are excluded while the final Store transaction
            # checks for a newly committed restrictive policy and settles the
            # exact finite-use reservations. A reserved ALLOW remains the grant
            # authority; a later independent DENY/ASK still fails publication.
            with self._memory.ownership_locked(), self._records.transaction():
                if self._objects.get_object(owner.oid) is None:
                    raise NotFound(f"object not found: {owner.oid}")
                self._require_concurrency_capacity(owner.oid)
                self._assert_no_restrictive_owner_policy(
                    pid,
                    owner,
                    owner_rights,
                )
                self._records.insert_object_task(task)
                self._operations.link_evidence(
                    "object_task",
                    task.task_id,
                    "result",
                    metadata={"owner_oid": task.owner_oid, "status": task.status.value},
                )
                self._commit_owner_decisions(owner_reservations)
                self._commit_spawn_decision(spawn_reservation, task_id=task_id)
                self._record_start_observability(task, owner, task_args)
            task_inserted = True
        except BaseException:
            if not task_inserted:
                if runner_pid is None or self._cleanup_uncommitted_runner(runner_pid, task_id=task_id):
                    self._restore_owner_decisions(owner_reservations)
                    self._restore_spawn_decision(
                        spawn_reservation,
                        task_id=task_id,
                    )
                else:
                    self._audit.record(
                        actor="object_task",
                        action="object_task.owner_permission_restore_skipped",
                        target=f"object_task:{task_id}",
                        decision={
                            "runner_pid": runner_pid,
                            "reason": "runner cleanup was not confirmed",
                        },
                    )
            raise
        try:
            with self._lock:
                self._grant_result_to_notify[task_id] = grant_result_to_notify
                self._pending_args[task_id] = task_args
                self._schedule_task_locked(task_id)
        except Exception as exc:
            # The durable task record is the authorization commit point, but a
            # failed handoff to the executor must not leave a live queued child
            # or an indefinitely active task behind.
            self._abort_unscheduled_task(task_id, runner_pid, f"object task scheduling failed: {exc}")
            raise
        return task

    def _bootstrap_runner(
        self,
        runner_pid: str,
        handle: ToolHandle,
        owner_oid: str,
        task_id: str,
    ) -> None:
        # ObjectTask runners are Host-managed children.  When start() is called
        # from a scheduler quantum, the creator's execution token is still
        # bound, so these bootstrap writes are intentionally cross-PID.  Keep
        # that exception confined to this new runner and its exact pre-run
        # state; ordinary worker writes remain bound to the creator PID.
        with trusted_process_control_mutation(
            runner_pid,
            allowed_statuses={ProcessStatus.WAITING_TOOL},
            reason="object_task.start bootstraps its waiting runner",
        ):
            self._tools.configure_process_tools(
                runner_pid,
                [handle],
                assigned_by=f"object_task:{task_id}",
            )
            owner_object = self._objects.get_object(owner_oid)
            if owner_object is None:
                raise NotFound(f"object not found: {owner_oid}")
            self._authority_manifests.assert_data_flow_labels(
                runner_pid,
                DataLabels.from_object_metadata(owner_object.metadata),
            )
            self._capabilities.handle_for_object(
                runner_pid,
                owner_oid,
                {ObjectRight.READ.value, ObjectRight.MATERIALIZE.value},
                issued_by=f"object_task:{task_id}",
            )

    def _record_start_observability(
        self,
        task: ObjectTask,
        owner: ObjectHandle,
        task_args: dict[str, Any],
    ) -> None:
        with self._records.transaction():
            self._events.emit(
                EventType.OBJECT_TASK_STARTED,
                source=task.creator_pid,
                target=owner.oid,
                payload={
                    "task_id": task.task_id,
                    "runner_pid": task.runner_pid,
                    "tool": task.tool,
                },
            )
            self._audit.record(
                actor=task.creator_pid,
                action="object_task.start",
                target=f"object:{owner.oid}",
                input_refs=[owner.oid],
                decision={
                    "task_id": task.task_id,
                    "runner_pid": task.runner_pid,
                    "tool": task.tool,
                    "args": sanitize_for_observability(task_args),
                    "notify_pid": task.notification.recipient_pid,
                    "notify_kind": task.notification.kind,
                },
            )
            if task.owner_watch.enabled:
                self._audit.record(
                    actor=task.creator_pid,
                    action="object_task.owner_watch.register",
                    target=f"object_task:{task.task_id}",
                    input_refs=[owner.oid],
                    decision={
                        "owner_oid": owner.oid,
                        "events": task.owner_watch.events,
                        "kind": task.owner_watch.kind,
                        "channel": task.owner_watch.channel,
                    },
                )

    def watch_owner(
        self,
        task_id: str,
        *,
        actor_pid: str,
        enabled: bool = True,
        events: Iterable[str] | None = None,
        channel: str | None = None,
        kind: str | ProcessMessageKind | None = None,
    ) -> ObjectTask:
        # Refresh may take the in-memory task lock and perform Host-owned
        # terminal reconciliation.  Do it before the durable authority lock to
        # preserve the manager's lock ordering.
        self._refresh_task_from_runner(self._get(task_id))
        with self._records.transaction():
            task = self._get(task_id)
            decision = self._require_task_mutable(actor_pid, task)
            if task.status in _TERMINAL_STATUSES:
                raise ValidationError(f"cannot watch owner for terminal object task: {task_id}")
            owner_watch = self._normalize_owner_watch(
                {
                    "enabled": enabled,
                    "events": list(events) if events is not None else task.owner_watch.events,
                    "channel": channel or task.owner_watch.channel,
                    "kind": kind or task.owner_watch.kind,
                }
            )
            with self._capabilities.authority_transaction(
                [decision],
                actor=actor_pid,
                operation="object task owner-watch mutation",
            ):
                updated = replace(task, owner_watch=owner_watch, updated_at=utc_now())
                self._records.update_object_task(updated)
                self._audit.record(
                    actor=actor_pid,
                    action="object_task.owner_watch.register",
                    target=f"object_task:{task_id}",
                    input_refs=[task.owner_oid],
                    decision={
                        "owner_oid": task.owner_oid,
                        "enabled": owner_watch.enabled,
                        "events": owner_watch.events,
                        "kind": owner_watch.kind,
                        "channel": owner_watch.channel,
                    },
                )
            return updated

    def get(self, task_id: str, *, actor_pid: str | None = None) -> ObjectTask:
        self._refresh_task_from_runner(self._get(task_id))
        # retry_terminal follows the task-lock -> store-lock order used by the
        # background worker.  Taking the same lock order here avoids a DB/task
        # lock inversion while keeping visibility and notification retry in
        # the authority transaction.
        with self._lock, self._records.transaction():
            task = self._get(task_id)
            decision = (
                self._require_task_visible(actor_pid, task)
                if actor_pid is not None
                else None
            )
            with self._capabilities.authority_transaction(
                [decision],
                actor=actor_pid or "object_task",
                operation="object task visibility",
            ):
                current = self._notifications.retry_terminal(task)
        if (
            current.status == ObjectTaskStatus.WAITING_MESSAGE
            and not self._has_active_future(current.task_id)
        ):
            try:
                if self._has_matching_waiting_message(current):
                    self._schedule_waiting_message_resume(current.task_id)
            except Exception as exc:
                # Message post is already durable; a later read remains a
                # reconciliation point, but inability to schedule must not make
                # the task itself invisible.
                try:
                    self._audit.record(
                        actor="object_task",
                        action="object_task.message_reconcile_failed",
                        target=f"object_task:{current.task_id}",
                        decision={"error": internal_exception_observation(exc)},
                    )
                except BaseException:
                    pass
        return current

    def list(
        self,
        *,
        actor_pid: str | None = None,
        owner_oid: str | None = None,
        include_terminal: bool = True,
        limit: int | None = None,
    ) -> list[ObjectTask]:
        selected_limit = self._coerce_list_limit(limit)
        if selected_limit == 0:
            return []
        candidate_limit = selected_limit
        if actor_pid is not None:
            # Visibility filtering cannot safely push a simple result LIMIT into
            # SQL, but it must not turn a one-row request into an unbounded
            # payload/authorization scan. Use the configured recovery hard bound
            # as a deterministic candidate ceiling.
            candidate_limit = self.config.runtime.object_task_recovery_page_hard_limit
        snapshots = self._records.list_object_tasks(
            owner_oid=owner_oid,
            include_terminal=include_terminal,
            limit=candidate_limit,
        )
        refreshed = [self._refresh_task_from_runner(candidate) for candidate in snapshots]
        with self._records.transaction():
            selected: list[tuple[ObjectTask, Any | None]] = []
            for snapshot in refreshed:
                task = self._records.get_object_task(snapshot.task_id) or snapshot
                if not include_terminal and task.status in _TERMINAL_STATUSES:
                    continue
                if actor_pid is None or task.creator_pid == actor_pid:
                    selected.append((task, None))
                    continue
                decision = self._capabilities.authorize(
                    actor_pid,
                    f"object:{task.owner_oid}",
                    ObjectRight.READ,
                )
                if decision.allowed:
                    selected.append((task, decision))
            if selected_limit is not None:
                selected = selected[:selected_limit]
            decisions = [decision for _, decision in selected if decision is not None]
            with self._capabilities.authority_transaction(
                decisions,
                actor=actor_pid or "object_task",
                operation="object task list visibility",
            ):
                return [task for task, _ in selected]

    def cancel(self, task_id: str, *, actor_pid: str, reason: str | None = None) -> ObjectTask:
        selected_reason = self._messages.validate_body(str(reason or "cancelled"))
        self._refresh_task_from_runner(self._get(task_id))
        # Tool execution resolves under the registry lifecycle lock and then
        # reads the Store. Never invert that order by resolving a tool while a
        # Store transaction is held: cancellation can otherwise deadlock the
        # worker at registry -> Store versus Store -> registry. The task lock
        # keeps its status/tool binding stable across classification and the
        # following transaction.
        with self._lock:
            task = self._get(task_id)
            running_sync_side_effect = (
                task.status == ObjectTaskStatus.RUNNING
                and self._tools.is_sync_side_effect_tool(
                    task.tool_id or task.tool
                )
            )
            with self._records.transaction():
                task = self._get(task_id)
                decision = self._require_task_mutable(actor_pid, task)
                if (
                    task.status == ObjectTaskStatus.RUNNING
                    and running_sync_side_effect
                ):
                    raise ValidationError(
                        f"running synchronous side-effect object task cannot be safely cancelled: {task_id}"
                    )
                with self._capabilities.authority_transaction(
                    [decision],
                    actor=actor_pid,
                    operation="object task cancellation",
                ):
                    cancelled = (
                        task
                        if task.status in _TERMINAL_STATUSES
                        else self._state.mark_cancelled(
                            task,
                            actor=actor_pid,
                            reason=selected_reason,
                        )
                    )
                if task.status in _TERMINAL_STATUSES:
                    return cancelled
                future = self._futures.get(task_id)
                if future is not None:
                    future.cancel()
                self._cleanup_task_state_locked(cancelled.task_id)
        return cancelled

    def wait(self, task_id: str, *, actor_pid: str | None = None, timeout: float | None = None) -> ObjectTask:
        selected_timeout = self._coerce_wait_timeout(timeout)
        deadline = None if selected_timeout is None else time.monotonic() + selected_timeout
        # A wait is one long-lived read operation.  Linearize and, when
        # applicable, consume its authority once at entry; internal polling
        # must not consume the same one-shot grant again on every iteration.
        task = self.get(task_id, actor_pid=actor_pid)
        while True:
            if (
                task.status == ObjectTaskStatus.WAITING_HUMAN
                and not self._has_active_future(task.task_id)
                and self._schedule_waiting_human_resume(task)
            ):
                task = self.get(task_id)
                continue
            if task.status in _TERMINAL_STATUSES:
                # Terminal task state is written before the worker coroutine
                # delivers its completion notification. Keep waiters on the
                # worker until that final notification write has settled.
                if self._has_active_future(task.task_id):
                    if deadline is not None and time.monotonic() >= deadline:
                        return task
                    time.sleep(0.01)
                    continue
                # The worker may have completed between the row read above and
                # the lifetime check. Re-read so callers observe its final
                # notification/result cleanup writes, not the stale terminal
                # snapshot that preceded coroutine settlement.
                settled = self.get(task_id)
                self._state.cleanup_owner_pin_after_terminal(settled)
                return settled
            if (
                task.status in {
                    ObjectTaskStatus.WAITING_HUMAN,
                    ObjectTaskStatus.WAITING_PROCESS,
                    ObjectTaskStatus.WAITING_MESSAGE,
                }
                and not self._has_active_future(task.task_id)
            ):
                return task
            if deadline is not None and time.monotonic() >= deadline:
                return task
            time.sleep(0.01)
            task = self.get(task_id)

    def _coerce_list_limit(self, limit: int | None) -> int | None:
        if limit is None:
            return None
        if isinstance(limit, bool):
            raise ValidationError("object task list limit must be an integer")
        try:
            selected = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValidationError("object task list limit must be an integer") from exc
        if selected < 0:
            raise ValidationError("object task list limit must be non-negative")
        return selected

    def _coerce_wait_timeout(self, timeout: float | None) -> float | None:
        if timeout is None:
            return None
        if isinstance(timeout, bool):
            raise ValidationError("object task wait timeout must be a number")
        try:
            selected = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValidationError("object task wait timeout must be a number") from exc
        if not math.isfinite(selected):
            raise ValidationError("object task wait timeout must be finite")
        if selected < 0:
            raise ValidationError("object task wait timeout must be non-negative")
        return selected

    def has_active_for_owner(self, owner_oid: str) -> bool:
        for task in self._records.list_object_tasks(
            owner_oid=owner_oid,
            include_terminal=False,
            limit=self.config.runtime.object_task_recovery_page_hard_limit,
        ):
            self._refresh_task_from_runner(task)
        return self.has_published_active_for_owner(owner_oid)

    def has_published_active_for_owner(self, owner_oid: str) -> bool:
        """Return whether a durable, non-terminal task pins ``owner_oid``.

        This predicate deliberately performs no runner reconciliation or other
        state mutation. ObjectMemory calls it while holding the ownership fence,
        so keeping it to a bounded durable read preserves the ownership -> store
        lock order used by task publication.
        """

        return bool(
            self._records.list_object_tasks(
                owner_oid=owner_oid,
                include_terminal=False,
                limit=1,
            )
        )

    def is_runner_pid(self, pid: str) -> bool:
        selected_pid = str(pid)
        return any(
            str(task.runner_pid) == selected_pid
            for task in self._records.list_object_tasks(include_terminal=False)
        )

    def notify_owner_changed(self, owner_oid: str, change: dict[str, Any], actor_pid: str) -> list[str]:
        event = str(change.get("event") or "").strip()
        if event not in _OWNER_WATCH_EVENTS:
            return []
        notified: list[str] = []
        for task in self._records.list_object_tasks(owner_oid=owner_oid, include_terminal=False):
            task = self._refresh_task_from_runner(task)
            if task.status in _TERMINAL_STATUSES:
                continue
            if not task.owner_watch.enabled or event not in task.owner_watch.events:
                continue
            if self._notify_owner_change(task, change, actor_pid):
                notified.append(task.task_id)
        return notified

    def shutdown(self) -> bool:
        """Stop this manager as an admitted public control mutation."""

        return self._shutdown()

    def _issue_lifecycle_shutdown_handle(self) -> object:
        """Return the opaque handle used by RuntimeLifecycle teardown."""

        capability = self.__lifecycle_shutdown_capability
        return _ObjectTaskLifecycleShutdownHandle(
            shutdown=lambda: self._shutdown_for_lifecycle(capability),
            release_recovery_diagnostics=(
                lambda: self._abandon_for_recovery(capability)
            ),
        )

    def _shutdown_for_lifecycle(self, capability: object) -> bool:
        if capability is not self.__lifecycle_shutdown_capability:
            raise RuntimeError("invalid ObjectTask lifecycle shutdown capability")
        return self._shutdown()

    def _abandon_for_recovery(self, capability: object) -> bool:
        """Stop transient execution after a drained persistent recovery fence.

        RuntimeLifecycle owns the opaque capability and verifies both the
        monotonic recovery fence and admission drain before reaching this
        method.  Unlike ordinary shutdown, this path must preserve every
        durable task, runner, audit, and event row for recovery diagnostics.
        """

        if capability is not self.__lifecycle_shutdown_capability:
            raise RuntimeError("invalid ObjectTask lifecycle shutdown capability")
        return self._abandon_transient_workers()

    def _abandon_transient_workers(self) -> bool:
        """Quiesce in-memory worker state without writing durable evidence."""

        with self._lock:
            if self._closed:
                return True
            # A running coroutine owns a lifecycle admission for its entire
            # publication/error path.  Recovery release must wait for that
            # lease to drain; never cancel an admitted run out from under it.
            if self._active_runs:
                return False
            self._closing = True
            started = self._started
            futures = tuple(self._futures.values())

        if not started:
            self._executor.shutdown(wait=True, cancel_futures=True)
            if not self._loop.is_closed():
                self._loop.close()
            self._worker_shutdown_complete.set()
        else:
            # Linearize on the loop after every pre-existing scheduling
            # callback. _closing prevents new submissions while the lifecycle
            # recovery fence prevents a queued coroutine from entering its
            # admitted execution body. If an admitted run is still alive, do
            # not cancel or stop anything: the lifecycle can retry after drain.
            if not self._worker_cleanup_started.is_set():
                stop_decided = threading.Event()
                stop_allowed = False

                def stop_if_quiescent() -> None:
                    nonlocal stop_allowed
                    with self._lock:
                        stop_allowed = not self._active_runs
                    if stop_allowed:
                        self._loop.stop()
                    stop_decided.set()

                try:
                    self._loop.call_soon_threadsafe(stop_if_quiescent)
                except RuntimeError:
                    if self._thread.is_alive():
                        return False
                if self._thread.is_alive() and not stop_decided.wait(
                    timeout=self.config.object_tasks.shutdown_join_timeout_s
                ):
                    return False
                if not stop_allowed and self._thread.is_alive():
                    return False
            self._thread.join(
                timeout=self.config.object_tasks.shutdown_join_timeout_s
            )
            if self._thread.is_alive():
                return False
            if not self._worker_shutdown_complete.is_set():
                return False

            # _run_loop cancels and gathers every pending asyncio Task after
            # stop, then drains the owned default executor before closing the
            # loop. The bridge futures must therefore also be terminal here.
            if any(not future.done() for future in futures):
                return False

        with self._lock:
            # No durable cleanup belongs on this path.  Drop only process-local
            # scheduling inputs and bookkeeping after the worker is gone.
            if self._active_runs:
                return False
            self._futures.clear()
            self._grant_result_to_notify.clear()
            self._pending_args.clear()
            self._closed = True
        return True

    def _shutdown(self) -> bool:
        if self._closed:
            return True
        with self._lock:
            first_shutdown_attempt = not self._closing
            self._closing = True
            started = self._started
        if not started:
            self._executor.shutdown(wait=True, cancel_futures=True)
            if not self._loop.is_closed():
                self._loop.close()
            self._worker_shutdown_complete.set()
            self._closed = True
            return True
        if first_shutdown_attempt:
            active = self._records.list_object_tasks(include_terminal=False)
            for task in active:
                latest = self._records.get_object_task(task.task_id)
                if latest is not None and latest.status not in _TERMINAL_STATUSES:
                    self._mark_cancelled(latest, actor="runtime.shutdown", reason="runtime shutdown")
        self._drain_done_futures()
        with self._lock:
            futures = list(self._futures.values())
        if futures:
            wait(futures, timeout=self.config.object_tasks.shutdown_join_timeout_s)
        self._drain_done_futures()
        with self._lock:
            unfinished_ids = set(self._active_runs)
            unfinished_ids.update(
                task_id
                for task_id, future in self._futures.items()
                if not future.done()
            )
        if unfinished_ids:
            self._audit.record(
                actor="runtime.shutdown",
                action="object_task.shutdown_deferred",
                target="object_tasks",
                decision={"unfinished": len(unfinished_ids)},
            )
            return False
        if not self._worker_cleanup_started.is_set():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                if self._thread.is_alive():
                    self._audit.record(
                        actor="runtime.shutdown",
                        action="object_task.shutdown_deferred",
                        target="object_tasks",
                        decision={"reason": "event loop could not be stopped"},
                    )
                    return False
        self._thread.join(timeout=self.config.object_tasks.shutdown_join_timeout_s)
        if self._thread.is_alive():
            self._audit.record(
                actor="runtime.shutdown",
                action="object_task.shutdown_deferred",
                target="object_tasks",
                decision={"reason": "event loop thread did not stop"},
            )
            return False
        if not self._worker_shutdown_complete.is_set():
            self._audit.record(
                actor="runtime.shutdown",
                action="object_task.shutdown_deferred",
                target="object_tasks",
                decision={"reason": "worker executor cleanup did not complete"},
            )
            return False
        self._closed = True
        return True

    async def _run_task(self, task_id: str) -> None:
        with self._lock:
            self._active_runs.add(task_id)
        try:
            # Scheduling deliberately starts from a clean Context so the
            # caller's short-lived admission cannot escape into this worker.
            # Acquire one fresh lease for the complete background execution,
            # including error handling and terminal notification publication.
            # Nested Tool/Process/DataFlow guards inherit this lease.
            with self._admission.admit():
                try:
                    await self._execute_task(task_id)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    # A task-local BaseException must not escape asyncio's Task
                    # step, where KeyboardInterrupt/SystemExit would terminate
                    # the manager loop. Settle the durable task while its
                    # lifecycle admission is still active.
                    self._fail_task_from_exception(task_id, exc)
        finally:
            self._finish_active_run(task_id)

    async def _execute_task(self, task_id: str) -> None:
        task = self._get(task_id)
        settlement = _TaskResultSettlement()
        with self._lock:
            args = dict(self._pending_args.get(task_id, {}))
        context_metadata = self._context_metadata_for_resume(task)
        try:
            task = self._state.mark_running(task)
            if task.status != ObjectTaskStatus.RUNNING:
                return
            result = await self._tools.acall(
                str(task.runner_pid),
                task.tool,
                args,
                context_metadata=context_metadata,
            )
            self._settle_tool_result(task, result, settlement)
        except asyncio.CancelledError:
            latest = self._records.get_object_task(task_id)
            if latest is not None and latest.status not in _TERMINAL_STATUSES:
                self._mark_cancelled(latest, actor="object_task", reason="cancelled")
            raise
        except HumanApprovalRequired as exc:
            self._state.mark_waiting(task_id, ObjectTaskStatus.WAITING_HUMAN, {"request_id": exc.request_id}, str(exc))
        except ProcessWaitRequired as exc:
            self._state.mark_waiting(task_id, ObjectTaskStatus.WAITING_PROCESS, {"child_pid": exc.child_pid}, str(exc))
        except ProcessMessageWaitRequired as exc:
            self._state.mark_waiting(task_id, ObjectTaskStatus.WAITING_MESSAGE, {"filters": exc.filters}, str(exc))
        except BaseException as exc:
            self._fail_task_from_exception(
                task_id,
                exc,
                runner_pid=str(task.runner_pid),
                result_oid=settlement.result_oid,
            )
        finally:
            self._restore_notify_result_grant(
                settlement.pending_notify_grant,
                reason="object task notify-result publication did not settle",
            )

    def _settle_tool_result(
        self,
        task: ObjectTask,
        result: Any,
        settlement: _TaskResultSettlement,
    ) -> None:
        task_id = task.task_id
        runner_pid = str(task.runner_pid)
        latest_process = self._process.get(runner_pid)
        latest_task = self._records.get_object_task(task_id)
        result_handle = result.result_handle
        if latest_task is None or latest_task.status in _TERMINAL_STATUSES:
            self._state.discard_failed_result(
                runner_pid,
                task_id,
                result_handle.oid if result_handle else None,
            )
            return
        if not result.ok:
            if latest_process.status not in self._process.TERMINAL_STATUSES:
                self._process.exit(
                    runner_pid,
                    failed=True,
                    message=result.error or "object task failed",
                )
            self._state.discard_failed_result(
                runner_pid,
                task_id,
                result_handle.oid if result_handle else None,
            )
            self._state.mark_failed(task_id, result.error or "object task failed")
            return

        settlement.result_oid = result_handle.oid if result_handle is not None else None
        if settlement.result_oid is not None:
            settlement.pending_notify_grant = self._publish_success_result(
                task,
                settlement.result_oid,
            )
        # Serialize the final active-state check with cancellation and terminal
        # task transitions. A cancellation that won while the result was being
        # wired must discard, not publish, that result.
        with self._lock:
            latest_task = self._records.get_object_task(task_id)
            if latest_task is None or latest_task.status in _TERMINAL_STATUSES:
                self._restore_notify_result_grant(
                    settlement.pending_notify_grant,
                    reason="object task completed after cancellation",
                )
                settlement.pending_notify_grant = None
                self._state.discard_failed_result(
                    runner_pid,
                    task_id,
                    settlement.result_oid,
                )
                return
            latest_process = self._process.get(runner_pid)
            if latest_process.status not in self._process.TERMINAL_STATUSES:
                self._process.exit(runner_pid, result=result_handle)
            self._mark_succeeded(
                task_id,
                result,
                settlement.result_oid,
                pending_notify_grant=settlement.pending_notify_grant,
            )
            settlement.pending_notify_grant = None

    def _fail_task_from_exception(
        self,
        task_id: str,
        exc: BaseException,
        *,
        runner_pid: str | None = None,
        result_oid: str | None = None,
    ) -> None:
        latest = self._records.get_object_task(task_id)
        if latest is None or latest.status in _TERMINAL_STATUSES:
            # Once a success row is durable, failures in best-effort
            # post-commit observability must not delete the published result.
            return
        selected_runner_pid = runner_pid or (
            str(latest.runner_pid) if latest.runner_pid is not None else None
        )
        error = self._record_task_failure_diagnostic(task_id, exc)
        if selected_runner_pid is not None:
            self._state.terminalize_runner(
                selected_runner_pid,
                reason=f"object task failed: {error}",
            )
            self._state.discard_failed_result(
                selected_runner_pid,
                task_id,
                result_oid,
            )
        self._state.mark_failed(task_id, error)

    def _record_task_failure_diagnostic(
        self,
        task_id: str,
        exc: BaseException,
    ) -> str:
        public_error = public_error_envelope(exc)
        self._audit.record(
            actor="object_task",
            action="object_task.failure_diagnostic",
            target=f"object_task:{task_id}",
            decision=internal_exception_observation(exc),
            correlation_id=public_error["correlation_id"],
        )
        return public_error["message"]

    @staticmethod
    def _task_error_message(exc: BaseException) -> str:
        return public_error_envelope(exc)["message"]

    def _publish_success_result(
        self,
        task: ObjectTask,
        result_oid: str,
    ) -> _PendingNotifyResultGrant | None:
        task_id = task.task_id
        self._memory.transfer_owner(
            ObjectOwnerKind.PROCESS,
            str(task.runner_pid),
            ObjectOwnerKind.OBJECT_TASK,
            task_id,
            [result_oid],
            actor=f"object_task:{task_id}",
            reason="object_task_result",
        )
        creator = self._records.get_process(task.creator_pid)
        if (
            creator is not None
            and creator.status not in self._process.TERMINAL_STATUSES
        ):
            result_object = self._objects.get_object(result_oid)
            if result_object is None:
                raise NotFound(f"object task result not found: {result_oid}")
            self._authority_manifests.assert_data_flow_labels(
                task.creator_pid,
                DataLabels.from_object_metadata(result_object.metadata),
            )
            creator_handle = self._capabilities.handle_for_object(
                task.creator_pid,
                result_oid,
                {
                    ObjectRight.READ.value,
                    ObjectRight.MATERIALIZE.value,
                    ObjectRight.LINK.value,
                },
                issued_by=f"object_task:{task_id}",
            )
            self._add_handle_to_process_view(task.creator_pid, creator_handle)
        self._memory.link_objects_trusted(
            f"object_task:{task_id}",
            task.owner_oid,
            RelationType.PRODUCED,
            result_oid,
            metadata={"task_id": task_id, "tool": task.tool},
            reason="object_task_result",
        )
        return self._prepare_notify_result_grant(task, result_oid)

    def _run_loop(self) -> None:
        cleanup_complete = True
        failure: BaseException | None = None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.set_default_executor(self._executor)
            self._loop_ready.set()
            self._loop.run_forever()
        except BaseException as exc:
            # The task boundary above contains task-local BaseExceptions. Keep
            # this final guard so an unexpected loop callback still reaches
            # deterministic executor cleanup instead of leaking worker threads.
            failure = exc
        finally:
            self._loop_ready.set()
            self._worker_cleanup_started.set()
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except BaseException as exc:
                cleanup_complete = False
                if failure is None:
                    failure = exc
            try:
                # The object-task event loop owns this explicitly installed
                # executor. Its shutdown coroutine does not return until every
                # underlying worker has exited.
                self._loop.run_until_complete(
                    self._loop.shutdown_default_executor()
                )
            except BaseException as exc:
                cleanup_complete = False
                if failure is None:
                    failure = exc
            try:
                self._loop.close()
            except BaseException as exc:
                cleanup_complete = False
                if failure is None:
                    failure = exc
            with self._lock:
                self._worker_failure = (
                    self._task_error_message(failure)
                    if failure is not None
                    else None
                )
            if cleanup_complete and self._loop.is_closed():
                self._worker_shutdown_complete.set()

    def _schedule_task_locked(self, task_id: str) -> Future[Any]:
        if self._closing or self._closed:
            raise RuntimeError("object task manager is shutting down")
        if not self._started:
            raise RuntimeError("object task worker has not started")
        # run_coroutine_threadsafe copies the submitting thread's ContextVars.
        # A task is submitted from a caller admission scope but outlives that
        # scope, so inheriting its lease races the caller marking it inactive.
        # Start background work from a clean context; public mutations inside
        # the coroutine then acquire their own lifecycle admission lease.
        future = contextvars.Context().run(
            lambda: asyncio.run_coroutine_threadsafe(
                self._run_task(task_id),
                self._loop,
            )
        )
        self._futures[task_id] = future
        future.add_done_callback(lambda _future, task_id=task_id: self._forget_future(task_id))
        return future

    def _has_active_future(self, task_id: str) -> bool:
        with self._lock:
            future = self._futures.get(task_id)
            return task_id in self._active_runs or (future is not None and not future.done())

    def _normalize_owner_watch(self, value: bool | dict[str, Any] | ObjectTaskOwnerWatch | None) -> ObjectTaskOwnerWatch:
        if isinstance(value, ObjectTaskOwnerWatch):
            raw: dict[str, Any] = {
                "enabled": value.enabled,
                "events": value.events,
                "kind": value.kind,
                "channel": value.channel,
            }
        elif isinstance(value, dict):
            raw = dict(value)
            raw.setdefault("enabled", True)
        elif isinstance(value, bool):
            raw = {"enabled": value}
        elif value is None:
            raw = {"enabled": False}
        else:
            raise ValidationError(
                "object task owner_watch must be a boolean or mapping"
            )
        enabled_value = raw.get("enabled", False)
        if type(enabled_value) is not bool:
            raise ValidationError(
                "object task owner_watch.enabled must be a boolean"
            )
        enabled = enabled_value
        raw_events = raw.get("events") or self.config.object_tasks.owner_watch_events
        if isinstance(raw_events, str):
            raw_events = [raw_events]
        events: list[str] = []
        for item in raw_events:
            event = str(item).strip().lower()
            if event not in _OWNER_WATCH_EVENTS:
                raise ValidationError(f"unsupported object task owner watch event: {item}")
            if event not in events:
                events.append(event)
        if not events:
            raise ValidationError("object task owner watch requires at least one event")
        selected_kind = self._message_kind(raw.get("kind") or ProcessMessageKind.NORMAL.value, "object task owner watch kind").value
        channel = str(raw.get("channel") or self.config.object_tasks.owner_watch_channel).strip()
        if not channel:
            raise ValidationError("object task owner watch channel must be non-empty")
        if len(channel) > 128:
            raise ValidationError("object task owner watch channel is too long")
        return ObjectTaskOwnerWatch(enabled=enabled, events=events, kind=selected_kind, channel=channel)

    def _message_kind(self, value: str | ProcessMessageKind, label: str) -> ProcessMessageKind:
        try:
            return ProcessMessageKind(value)
        except ValueError as exc:
            allowed = ", ".join(kind.value for kind in ProcessMessageKind)
            raise ValidationError(f"{label} must be one of: {allowed}") from exc

    def _notify_owner_change(self, task: ObjectTask, change: dict[str, Any], actor_pid: str) -> bool:
        if task.runner_pid is None:
            return False
        payload = {
            "type": "object_task_owner_change",
            "task_id": task.task_id,
            "owner_oid": task.owner_oid,
            "event": change.get("event"),
            "event_id": change.get("event_id"),
            "version": change.get("version"),
            "change": change.get("change") or {},
            "relation": change.get("relation"),
            "dst_oid": change.get("dst_oid"),
            "link_id": change.get("link_id"),
        }
        subject = f"Object owner {payload['event']}: {task.owner_oid}"
        sender = f"object_task_owner_watch:{task.task_id}"
        source_oids = [task.owner_oid]
        if payload["dst_oid"]:
            source_oids.append(str(payload["dst_oid"]))
        try:
            message = self._messages.post(
                sender=sender,
                recipient_pid=task.runner_pid,
                kind=task.owner_watch.kind,
                channel=task.owner_watch.channel,
                correlation_id=task.task_id,
                subject=subject,
                payload=payload,
                source_oids=source_oids,
            )
        except Exception as exc:
            public_error = public_error_envelope(exc)
            internal_error = internal_exception_observation(exc)
            self._events.emit(
                EventType.OBJECT_TASK_OWNER_CHANGE_UNDELIVERED,
                source=actor_pid,
                target=task.owner_oid,
                payload={
                    "task_id": task.task_id,
                    "runner_pid": task.runner_pid,
                    "event": change.get("event"),
                    "error": public_error["message"],
                },
                priority=EventPriority.HIGH,
                correlation_id=public_error["correlation_id"],
            )
            self._audit.record(
                actor="object_task",
                action="object_task.owner_watch.undelivered",
                target=f"object_task:{task.task_id}",
                input_refs=[task.owner_oid],
                decision={
                    "runner_pid": task.runner_pid,
                    "event": change.get("event"),
                    "internal_error": internal_error,
                },
                correlation_id=public_error["correlation_id"],
            )
            return False
        self._events.emit(
            EventType.OBJECT_TASK_OWNER_CHANGE_NOTIFIED,
            source=actor_pid,
            target=task.owner_oid,
            payload={
                "task_id": task.task_id,
                "runner_pid": task.runner_pid,
                "message_id": message.message_id,
                "event": change.get("event"),
            },
        )
        self._audit.record(
            actor="object_task",
            action="object_task.owner_watch.notify",
            target=f"object_task:{task.task_id}",
            input_refs=[task.owner_oid],
            decision={
                "runner_pid": task.runner_pid,
                "message_id": message.message_id,
                "event": change.get("event"),
                "channel": message.channel,
                "kind": message.kind.value,
            },
        )
        if task.status == ObjectTaskStatus.WAITING_MESSAGE and self._message_matches_filters(message, task.wait.get("filters") or {}):
            self._schedule_waiting_message_resume(task.task_id)
        return True

    def notify_process_message(self, message: Any) -> None:
        tasks = [
            task
            for task in self._records.list_object_tasks(include_terminal=False)
            if task.status == ObjectTaskStatus.WAITING_MESSAGE
        ]
        for task in tasks:
            if str(message.recipient_pid) != str(task.runner_pid):
                continue
            if self._message_matches_filters(message, task.wait.get("filters") or {}):
                self._schedule_waiting_message_resume(task.task_id)

    def notify_process_terminal(self, child_pid: str) -> None:
        tasks = [
            task
            for task in self._records.list_object_tasks(include_terminal=False)
            if task.status == ObjectTaskStatus.WAITING_PROCESS and str(task.wait.get("child_pid") or "") == child_pid
        ]
        for task in tasks:
            self._schedule_waiting_process_resume(task.task_id)

    def _schedule_waiting_message_resume(self, task_id: str) -> None:
        with self._lock:
            latest = self._records.get_object_task(task_id)
            if latest is None or latest.status != ObjectTaskStatus.WAITING_MESSAGE:
                return
            future = self._futures.get(task_id)
            if future is not None and not future.done():
                return
            if task_id not in self._pending_args:
                self._audit.record(
                    actor="object_task",
                    action="object_task.owner_watch.resume_missing_pending",
                    target=f"object_task:{task_id}",
                    decision={"status": latest.status.value},
                )
                return
            if not self._can_resume_waiting_message(latest):
                self._audit.record(
                    actor="object_task",
                    action="object_task.owner_watch.resume_unsafe_replay_skipped",
                    target=f"object_task:{task_id}",
                    decision={"status": latest.status.value, "tool": latest.tool, "filters": latest.wait.get("filters")},
                )
                return
            self._audit.record(
                actor="object_task",
                action="object_task.owner_watch.resume",
                target=f"object_task:{task_id}",
                decision={"status": latest.status.value, "filters": latest.wait.get("filters")},
            )
            self._schedule_task_locked(task_id)

    def _schedule_waiting_process_resume(self, task_id: str) -> None:
        with self._lock:
            latest = self._records.get_object_task(task_id)
            if latest is None or latest.status != ObjectTaskStatus.WAITING_PROCESS:
                return
            future = self._futures.get(task_id)
            if future is not None and not future.done():
                return
            if task_id not in self._pending_args:
                self._audit.record(
                    actor="object_task",
                    action="object_task.process_resume_missing_pending",
                    target=f"object_task:{task_id}",
                    decision={"status": latest.status.value, "child_pid": latest.wait.get("child_pid")},
                )
                return
            if latest.tool not in _PROCESS_REPLAY_SAFE_TOOLS:
                self._audit.record(
                    actor="object_task",
                    action="object_task.process_resume_unsafe_replay_skipped",
                    target=f"object_task:{task_id}",
                    decision={"status": latest.status.value, "tool": latest.tool, "child_pid": latest.wait.get("child_pid")},
                )
                return
            self._audit.record(
                actor="object_task",
                action="object_task.process_resume",
                target=f"object_task:{task_id}",
                decision={"status": latest.status.value, "child_pid": latest.wait.get("child_pid")},
            )
            self._schedule_task_locked(task_id)

    def _schedule_waiting_human_resume(self, task: ObjectTask) -> bool:
        request_id = str(task.wait.get("request_id") or "")
        if not request_id:
            self._state.mark_failed(task.task_id, "object task waiting_human state is missing request_id")
            self._cleanup_task_state(task.task_id)
            return True
        try:
            request = self._human.get(request_id)
        except Exception as exc:
            error = self._record_task_failure_diagnostic(task.task_id, exc)
            self._state.mark_failed(task.task_id, error)
            self._cleanup_task_state(task.task_id)
            return True
        if request.status == HumanRequestStatus.PENDING:
            return False
        if request.status == HumanRequestStatus.APPROVED or (
            task.tool == "request_permission" and request.status == HumanRequestStatus.REJECTED
        ):
            with self._lock:
                latest = self._records.get_object_task(task.task_id)
                if latest is None or latest.status != ObjectTaskStatus.WAITING_HUMAN:
                    return False
                future = self._futures.get(task.task_id)
                if future is not None and not future.done():
                    return False
                if task.task_id not in self._pending_args:
                    self._audit.record(
                        actor="object_task",
                        action="object_task.human_resume_missing_pending",
                        target=f"object_task:{task.task_id}",
                        decision={"request_id": request_id, "status": request.status.value},
                    )
                    return False
                self._audit.record(
                    actor="object_task",
                    action="object_task.human_resume",
                    target=f"object_task:{task.task_id}",
                    decision={"request_id": request_id, "status": request.status.value},
                )
                self._schedule_task_locked(task.task_id)
            return True
        self._state.mark_failed(
            task.task_id,
            f"human request was not approved: {request_id} status={request.status.value}",
        )
        self._cleanup_task_state(task.task_id)
        return True

    def _context_metadata_for_resume(self, task: ObjectTask) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        task_operations = self._operations.operation_for_evidence(("object_task",), task.task_id)
        if len(task_operations) == 1:
            metadata["parent_operation_id"] = task_operations[0].operation_id
        if task.status != ObjectTaskStatus.WAITING_HUMAN:
            return metadata
        request_id = task.wait.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return metadata
        metadata["human_resume_request_id"] = request_id
        waiting_operations = self._operations.operation_for_evidence(("human_request",), request_id)
        if len(waiting_operations) == 1:
            metadata["operation_id"] = waiting_operations[0].operation_id
            metadata.pop("parent_operation_id", None)
        return metadata

    def _can_resume_waiting_message(self, task: ObjectTask) -> bool:
        return task.tool in _MESSAGE_REPLAY_SAFE_TOOLS

    def _message_matches_filters(self, message: Any, filters: dict[str, Any]) -> bool:
        if filters.get("kind") is not None and message.kind.value != filters["kind"]:
            return False
        if filters.get("sender") is not None and message.sender != filters["sender"]:
            return False
        if filters.get("channel") is not None and message.channel != filters["channel"]:
            return False
        if filters.get("correlation_id") is not None and message.correlation_id != filters["correlation_id"]:
            return False
        if filters.get("reply_to") is not None and message.reply_to != filters["reply_to"]:
            return False
        message_ids = filters.get("message_ids")
        if message_ids is not None and message.message_id not in set(message_ids):
            return False
        return True

    def _has_matching_waiting_message(self, task: ObjectTask) -> bool:
        if task.runner_pid is None:
            return False
        filters = task.wait.get("filters") or {}
        return bool(
            self._messages.unread(
                str(task.runner_pid),
                kind=filters.get("kind"),
                sender=filters.get("sender"),
                channel=filters.get("channel"),
                correlation_id=filters.get("correlation_id"),
                reply_to=filters.get("reply_to"),
                message_ids=filters.get("message_ids"),
                limit=1,
            )
        )

    def _mark_succeeded(
        self,
        task_id: str,
        result: Any,
        result_oid: str | None,
        *,
        pending_notify_grant: _PendingNotifyResultGrant | None = None,
    ) -> ObjectTask:
        if pending_notify_grant is None:
            return self._state.mark_succeeded(task_id, result, result_oid)
        try:
            with self._memory.ownership_locked(), self._records.transaction():
                result_object = self._objects.get_object(
                    pending_notify_grant.result_oid
                )
                if result_object is None:
                    raise NotFound(
                        f"object task result not found: {pending_notify_grant.result_oid}"
                    )
                self._authority_manifests.assert_data_flow_labels(
                    pending_notify_grant.recipient_pid,
                    DataLabels.from_object_metadata(result_object.metadata),
                )
                self._capabilities.handle_for_object(
                    pending_notify_grant.recipient_pid,
                    pending_notify_grant.result_oid,
                    {
                        ObjectRight.READ.value,
                        ObjectRight.MATERIALIZE.value,
                        ObjectRight.LINK.value,
                    },
                    issued_by=pending_notify_grant.used_by,
                )
                notified = self._state.mark_succeeded(task_id, result, result_oid)
                if (
                    notified.notification.status
                    != ObjectTaskNotificationStatus.DELIVERED
                ):
                    raise _NotifyResultGrantPublicationFailed(
                        "object task result notification was not delivered"
                    )
                committed = self._capabilities.commit_reserved_use(
                    pending_notify_grant.reservation_id,
                    committed_by=pending_notify_grant.used_by,
                    reason="one-time object task notify-result grant committed",
                )
                if pending_notify_grant.reservation_id is not None and not committed:
                    raise RuntimeError(
                        "object task notify-result grant reservation could not be committed"
                    )
            return notified
        except _NotifyResultGrantPublicationFailed:
            self._restore_notify_result_grant(
                pending_notify_grant,
                reason="object task result notification was not delivered",
            )
            return self._state.mark_succeeded(task_id, result, result_oid)

    def _mark_cancelled(self, task: ObjectTask, *, actor: str, reason: str) -> ObjectTask:
        notified = self._state.mark_cancelled(task, actor=actor, reason=reason)
        self._cleanup_task_state(notified.task_id)
        return notified

    def _prepare_notify_result_grant(
        self,
        task: ObjectTask,
        result_oid: str,
    ) -> _PendingNotifyResultGrant | None:
        if not self._grant_result_to_notify.get(task.task_id):
            return None
        recipient = task.notification.recipient_pid
        if recipient is None or recipient == task.creator_pid:
            return None
        result_object = self._objects.get_object(result_oid)
        if result_object is None:
            raise NotFound(f"object task result not found: {result_oid}")
        try:
            self._authority_manifests.assert_data_flow_labels(
                recipient,
                DataLabels.from_object_metadata(result_object.metadata),
            )
        except CapabilityDenied:
            # The normal notification path records the recipient-domain error.
            # Do not reserve GRANT authority or publish a provisional handle.
            return None
        decision = self._capabilities.authorize(task.creator_pid, f"object:{result_oid}", ObjectRight.GRANT)
        if not decision.allowed:
            raise CapabilityDenied(f"{task.creator_pid} cannot grant object task result: {result_oid}")
        used_by = f"object_task:{task.task_id}"
        reservation_id = self._capabilities.reserve_decision_use(
            decision,
            used_by=used_by,
            reason="one-time object task notify-result grant reserved",
        )
        return _PendingNotifyResultGrant(
            recipient_pid=recipient,
            result_oid=result_oid,
            reservation_id=reservation_id,
            used_by=used_by,
        )

    def _restore_notify_result_grant(
        self,
        pending: _PendingNotifyResultGrant | None,
        *,
        reason: str,
    ) -> None:
        if pending is None:
            return
        self._capabilities.restore_reserved_use(
            pending.reservation_id,
            restored_by=pending.used_by,
            reason=reason,
        )

    def _refresh_task_from_runner(self, task: ObjectTask) -> ObjectTask:
        if task.status in _TERMINAL_STATUSES or task.runner_pid is None:
            return task
        if self._has_active_future(task.task_id):
            return task
        process = self._records.get_process(task.runner_pid)
        if process is None or process.status not in self._process.TERMINAL_STATUSES:
            return task
        if process.status == ProcessStatus.KILLED:
            return self._mark_cancelled(task, actor="object_task.runner", reason=process.status_message or "runner killed")
        return self._state.mark_failed(
            task.task_id,
            process.status_message or f"object task runner ended before task completion: {process.status.value}",
        )

    def _assert_owner_rights(self, pid: str, owner: ObjectHandle, rights: set[str]) -> list[Any]:
        decisions = []
        for right in sorted(rights):
            decision = self._capabilities.authorize_handle(pid, owner, right)
            if not decision.allowed:
                raise CapabilityDenied(f"capability lacks {right}: {owner.oid}")
            decisions.append(decision)
        return decisions

    def _assert_no_restrictive_owner_policy(
        self,
        pid: str,
        owner: ObjectHandle,
        rights: set[str],
    ) -> None:
        for right in sorted(rights):
            current = self._capabilities.authorize(
                pid,
                f"object:{owner.oid}",
                right,
            )
            if not current.allowed and current.effect is not None:
                raise CapabilityDenied(
                    f"capability policy now restricts {right}: {owner.oid}"
                )

    def _reserve_owner_decisions(
        self,
        pid: str,
        owner: ObjectHandle,
        rights: set[str],
    ) -> list[str]:
        reservations: list[str] = []
        reserved_capability_ids: set[str] = set()
        with self._capabilities.store.transaction():
            current = self._assert_owner_rights(pid, owner, rights)
            for decision in current:
                cap_id = decision.consume_capability_id
                if cap_id is None or cap_id in reserved_capability_ids:
                    continue
                reservation_id = self._capabilities.reserve_decision_use(
                    decision,
                    used_by="object_task",
                    reason="one-time object task owner permission reserved",
                )
                if reservation_id is not None:
                    reservations.append(reservation_id)
                    reserved_capability_ids.add(str(cap_id))
        return reservations

    def _commit_owner_decisions(self, reservation_ids: list[str]) -> None:
        for reservation_id in reservation_ids:
            committed = self._capabilities.commit_reserved_use(
                reservation_id,
                committed_by="object_task",
                reason="one-time object task owner permission committed",
            )
            if not committed:
                raise CapabilityDenied(
                    "object task owner permission reservation could not be committed"
                )

    def _restore_owner_decisions(self, reservation_ids: list[str]) -> None:
        for reservation_id in reservation_ids:
            self._capabilities.restore_reserved_use(
                reservation_id,
                restored_by="object_task",
                reason="one-time object task owner permission restored before task commit",
            )

    def _commit_spawn_decision(
        self,
        reservation_id: str | None,
        *,
        task_id: str,
    ) -> None:
        if reservation_id is None:
            return
        committed = self._capabilities.commit_reserved_use(
            reservation_id,
            committed_by=f"object_task:{task_id}",
            reason="one-time object task spawn permission committed",
        )
        if not committed:
            raise CapabilityDenied(
                "object task spawn permission reservation could not be committed"
            )

    def _restore_spawn_decision(
        self,
        reservation_id: str | None,
        *,
        task_id: str,
    ) -> None:
        self._capabilities.restore_reserved_use(
            reservation_id,
            restored_by=f"object_task:{task_id}",
            reason="one-time object task spawn permission restored before task commit",
        )

    def _cleanup_uncommitted_runner(self, runner_pid: str, *, task_id: str) -> bool:
        cleanup_error: str | None = None
        release_error: str | None = None
        try:
            self._process.cleanup_failed_launch(runner_pid)
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
        try:
            self._process.release_child_budget(runner_pid)
        except Exception as exc:
            release_error = f"{type(exc).__name__}: {exc}"
        residual_process = self._records.get_process(runner_pid)
        residual_caps = self._capabilities.list_subject(runner_pid, include_inactive=True)
        if (
            residual_process is not None
            or residual_caps
            or cleanup_error is not None
            or release_error is not None
        ):
            self._audit.record(
                actor="object_task",
                action="object_task.runner_cleanup_incomplete",
                target=f"process:{runner_pid}",
                decision={
                    "task_id": task_id,
                    "process_present": residual_process is not None,
                    "capability_count": len(residual_caps),
                    "cleanup_error": cleanup_error,
                    "release_error": release_error,
                },
            )
            return False
        self._audit.record(
            actor="object_task",
            action="object_task.runner_cleanup",
            target=f"process:{runner_pid}",
            decision={"task_id": task_id},
        )
        return True

    def _abort_unscheduled_task(self, task_id: str, runner_pid: str | None, error: str) -> None:
        with self._lock:
            self._grant_result_to_notify.pop(task_id, None)
            self._pending_args.pop(task_id, None)
            self._futures.pop(task_id, None)
        cleanup_confirmed = runner_pid is None or self._cleanup_uncommitted_runner(runner_pid, task_id=task_id)
        try:
            self._state.mark_failed(task_id, error)
        except Exception as cleanup_exc:
            self._audit.record(
                actor="object_task",
                action="object_task.schedule_cleanup_failed",
                target=f"object_task:{task_id}",
                decision={
                    "runner_pid": runner_pid,
                    "runner_cleanup_confirmed": cleanup_confirmed,
                    "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                },
            )

    def _require_related_notification_target(self, actor_pid: str, recipient_pid: str) -> None:
        actor = self._records.get_process(actor_pid)
        recipient = self._records.get_process(recipient_pid)
        if actor is None:
            raise NotFound(f"process not found: {actor_pid}")
        if recipient is None:
            raise NotFound(f"process not found: {recipient_pid}")
        if recipient.status in self._process.TERMINAL_STATUSES:
            raise ProcessError(f"cannot notify terminal process: {recipient_pid}")
        if actor.pid == recipient.pid or actor.parent_pid == recipient.pid or recipient.parent_pid == actor.pid:
            return
        raise ProcessError(f"{actor_pid} can only notify itself, its parent, or its direct children")

    def _require_concurrency_capacity(self, owner_oid: str) -> None:
        active = self._records.list_object_tasks(include_terminal=False)
        if len(active) >= self.config.object_tasks.max_running_global:
            raise ValidationError("object task global concurrency limit exceeded")
        per_object = [task for task in active if task.owner_oid == owner_oid]
        if len(per_object) >= self.config.object_tasks.max_running_per_object:
            raise ValidationError("object task per-object concurrency limit exceeded")

    def _require_task_visible(self, actor_pid: str, task: ObjectTask) -> Any | None:
        if task.creator_pid == actor_pid:
            return None
        decision = self._capabilities.authorize(actor_pid, f"object:{task.owner_oid}", ObjectRight.READ)
        if not decision.allowed:
            raise CapabilityDenied(f"{actor_pid} cannot inspect object task: {task.task_id}")
        return decision

    def _require_task_mutable(self, actor_pid: str, task: ObjectTask) -> Any | None:
        if task.creator_pid == actor_pid:
            return None
        decision = self._capabilities.authorize(actor_pid, f"object:{task.owner_oid}", ObjectRight.WRITE)
        if not decision.allowed:
            raise CapabilityDenied(f"{actor_pid} cannot cancel object task: {task.task_id}")
        return decision

    def _get(self, task_id: str) -> ObjectTask:
        task = self._records.get_object_task(task_id)
        if task is None:
            raise NotFound(f"object task not found: {task_id}")
        return task

    def _forget_future(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._active_runs:
                return
            self._futures.pop(task_id, None)
            latest = self._records.get_object_task(task_id)
            if latest is not None and latest.status not in _TERMINAL_STATUSES:
                return
            self._cleanup_task_state_locked(task_id)

    def _finish_active_run(self, task_id: str) -> None:
        with self._lock:
            self._active_runs.discard(task_id)
            self._futures.pop(task_id, None)
            latest = self._records.get_object_task(task_id)
            if latest is not None and latest.status in _TERMINAL_STATUSES:
                self._cleanup_task_state_locked(task_id)

    def _drain_done_futures(self) -> None:
        with self._lock:
            task_ids = [
                task_id
                for task_id, future in self._futures.items()
                if future.done()
            ]
        for task_id in task_ids:
            self._forget_future(task_id)

    def _cleanup_task_state(self, task_id: str) -> None:
        with self._lock:
            self._cleanup_task_state_locked(task_id)

    def _cleanup_task_state_locked(self, task_id: str) -> None:
        self._grant_result_to_notify.pop(task_id, None)
        self._pending_args.pop(task_id, None)

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("object task manager is shut down")
