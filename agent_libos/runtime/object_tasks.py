from __future__ import annotations

import asyncio
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Any, TYPE_CHECKING, Iterable

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    CapabilityRight,
    CapabilitySpec,
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
    ObjectTaskStatus,
    ProcessMessageKind,
    ProcessStatus,
    RelationType,
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
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.utils.ids import new_id, utc_now

if TYPE_CHECKING:
    from agent_libos.runtime.runtime import Runtime


_ACTIVE_STATUSES = {
    ObjectTaskStatus.QUEUED,
    ObjectTaskStatus.RUNNING,
    ObjectTaskStatus.WAITING_HUMAN,
    ObjectTaskStatus.WAITING_PROCESS,
    ObjectTaskStatus.WAITING_MESSAGE,
}
_TERMINAL_STATUSES = {
    ObjectTaskStatus.SUCCEEDED,
    ObjectTaskStatus.FAILED,
    ObjectTaskStatus.CANCELLED,
    ObjectTaskStatus.ABANDONED,
}
_OWNER_WATCH_EVENTS = {"updated", "linked"}
_MESSAGE_REPLAY_SAFE_TOOLS = {"receive_process_messages"}


class ObjectTaskManager:
    """Background Object-bound tool tasks.

    Object tasks are host-managed execution records, not model scheduler work.
    Each task runs a single visible tool through a dedicated child process so
    the normal process tool table, capability, resource, event, and audit
    boundaries remain authoritative.
    """

    def __init__(self, runtime: "Runtime", config: AgentLibOSConfig | None = None) -> None:
        self.runtime = runtime
        self.config = config or DEFAULT_CONFIG
        self._lock = threading.RLock()
        self._loop = asyncio.new_event_loop()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.object_tasks.max_running_global,
            thread_name_prefix="agent-libos-object-task-tool",
        )
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="agent-libos-object-tasks",
            daemon=True,
        )
        self._futures: dict[str, Future[Any]] = {}
        self._grant_result_to_notify: dict[str, bool] = {}
        self._pending_args: dict[str, dict[str, Any]] = {}
        self._closing = False
        self._closed = False
        self._thread.start()
        self._loop_ready.wait(timeout=1.0)
        active_before_reopen = self.runtime.store.list_object_tasks(include_terminal=False)
        abandoned = self.runtime.store.mark_object_tasks_abandoned("runtime reopened before object task completed")
        if abandoned:
            abandoned_ids = set(abandoned)
            for task in active_before_reopen:
                if task.task_id in abandoned_ids and task.runner_pid is not None:
                    self._terminalize_runner(str(task.runner_pid), reason="object task abandoned after runtime reopen")
                if task.task_id in abandoned_ids:
                    abandoned_task = self.runtime.store.get_object_task(task.task_id)
                    if abandoned_task is not None:
                        self._cleanup_owner_pin_after_terminal(abandoned_task)
            self.runtime.audit.record(
                actor="object_task",
                action="object_task.abandon_recovered",
                target="object_tasks",
                decision={"task_ids": abandoned},
            )

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
        tool_name = str(tool).strip()
        if not tool_name:
            raise ValidationError("object task tool name is required")
        task_args = dict(args or {})
        selected_kind = self._message_kind(notify_kind, "object task notify_kind")
        selected_notify_pid = notify_pid or pid
        selected_owner_watch = self._normalize_owner_watch(owner_watch)
        self._require_related_notification_target(pid, selected_notify_pid)
        self._assert_owner_rights(pid, owner, {ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value})
        if self.runtime.store.get_object(owner.oid) is None:
            raise NotFound(f"object not found: {owner.oid}")
        handle = self.runtime.tools.resolve(tool_name, pid=pid)
        if not self.runtime.tools.process_has_tool(pid, handle):
            raise ValidationError(f"tool is not in process tool table: {tool_name}")
        self._require_concurrency_capacity(owner.oid)
        self.runtime.capability.require(pid, "process:spawn", CapabilityRight.WRITE)

        task_id = new_id("otask")
        runner_pid = self.runtime.process.spawn_child(
            pid,
            goal={"type": "object_task", "task_id": task_id, "owner_oid": owner.oid, "tool": tool_name},
            inherit_capabilities=[dict(spec) if isinstance(spec, dict) else spec.__dict__ for spec in (inherit_capabilities or [])],
            initial_status=ProcessStatus.WAITING_TOOL,
        )
        self.runtime.tools.configure_process_tools(runner_pid, [handle], assigned_by=f"object_task:{task_id}")
        self.runtime.capability.handle_for_object(
            runner_pid,
            owner.oid,
            {ObjectRight.READ.value, ObjectRight.MATERIALIZE.value},
            issued_by=f"object_task:{task_id}",
        )

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
                channel=notify_channel or self.config.object_tasks.notification_channel,
            ),
            owner_watch=selected_owner_watch,
            created_at=now,
            updated_at=now,
        )
        self.runtime.store.insert_object_task(task)
        with self._lock:
            self._grant_result_to_notify[task_id] = bool(grant_result_to_notify)
            self._pending_args[task_id] = task_args
            self._schedule_task_locked(task_id)
        self.runtime.events.emit(
            EventType.OBJECT_TASK_STARTED,
            source=pid,
            target=owner.oid,
            payload={"task_id": task_id, "runner_pid": runner_pid, "tool": tool_name},
        )
        self.runtime.audit.record(
            actor=pid,
            action="object_task.start",
            target=f"object:{owner.oid}",
            input_refs=[owner.oid],
            decision={
                "task_id": task_id,
                "runner_pid": runner_pid,
                "tool": tool_name,
                "args": sanitize_for_observability(task_args),
                "notify_pid": selected_notify_pid,
                "notify_kind": selected_kind.value,
            },
        )
        if selected_owner_watch.enabled:
            self.runtime.audit.record(
                actor=pid,
                action="object_task.owner_watch.register",
                target=f"object_task:{task_id}",
                input_refs=[owner.oid],
                decision={
                    "owner_oid": owner.oid,
                    "events": selected_owner_watch.events,
                    "kind": selected_owner_watch.kind,
                    "channel": selected_owner_watch.channel,
                },
            )
        return task

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
        task = self._refresh_task_from_runner(self._get(task_id))
        self._require_task_mutable(actor_pid, task)
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
        updated = replace(task, owner_watch=owner_watch, updated_at=utc_now())
        self.runtime.store.update_object_task(updated)
        self.runtime.audit.record(
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
        task = self._refresh_task_from_runner(self._get(task_id))
        if actor_pid is not None:
            self._require_task_visible(actor_pid, task)
        return task

    def list(
        self,
        *,
        actor_pid: str | None = None,
        owner_oid: str | None = None,
        include_terminal: bool = True,
        limit: int | None = None,
    ) -> list[ObjectTask]:
        selected_limit = self._coerce_list_limit(limit)
        tasks = [
            self._refresh_task_from_runner(task)
            for task in self.runtime.store.list_object_tasks(
                owner_oid=owner_oid,
                include_terminal=include_terminal,
                limit=None if actor_pid is not None and selected_limit is not None else selected_limit,
            )
        ]
        if actor_pid is not None:
            tasks = [task for task in tasks if self._can_view_task(actor_pid, task)]
        if selected_limit is not None:
            tasks = tasks[:selected_limit]
        return tasks

    def cancel(self, task_id: str, *, actor_pid: str, reason: str | None = None) -> ObjectTask:
        task = self._refresh_task_from_runner(self._get(task_id))
        self._require_task_mutable(actor_pid, task)
        if task.status in _TERMINAL_STATUSES:
            return task
        if task.status == ObjectTaskStatus.RUNNING and self.runtime.tools.is_sync_side_effect_tool(task.tool):
            raise ValidationError(
                f"running synchronous side-effect object task cannot be safely cancelled: {task_id}"
            )
        with self._lock:
            future = self._futures.get(task_id)
            if future is not None:
                future.cancel()
        return self._mark_cancelled(task, actor=actor_pid, reason=reason or "cancelled")

    def wait(self, task_id: str, *, actor_pid: str | None = None, timeout: float | None = None) -> ObjectTask:
        selected_timeout = self._coerce_wait_timeout(timeout)
        deadline = None if selected_timeout is None else time.monotonic() + selected_timeout
        while True:
            task = self.get(task_id, actor_pid=actor_pid)
            if (
                task.status == ObjectTaskStatus.WAITING_HUMAN
                and not self._has_active_future(task.task_id)
                and self._schedule_waiting_human_resume(task)
            ):
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
                self._cleanup_owner_pin_after_terminal(task)
                return task
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
        for task in self.runtime.store.list_object_tasks(owner_oid=owner_oid, include_terminal=False):
            self._refresh_task_from_runner(task)
        return bool(self.runtime.store.list_object_tasks(owner_oid=owner_oid, include_terminal=False, limit=1))

    def is_runner_pid(self, pid: str) -> bool:
        selected_pid = str(pid)
        return any(
            str(task.runner_pid) == selected_pid
            for task in self.runtime.store.list_object_tasks(include_terminal=False)
        )

    def notify_owner_changed(self, owner_oid: str, change: dict[str, Any], actor_pid: str) -> list[str]:
        event = str(change.get("event") or "").strip()
        if event not in _OWNER_WATCH_EVENTS:
            return []
        notified: list[str] = []
        for task in self.runtime.store.list_object_tasks(owner_oid=owner_oid, include_terminal=False):
            task = self._refresh_task_from_runner(task)
            if task.status in _TERMINAL_STATUSES:
                continue
            if not task.owner_watch.enabled or event not in task.owner_watch.events:
                continue
            if self._notify_owner_change(task, change, actor_pid):
                notified.append(task.task_id)
        return notified

    def shutdown(self) -> bool:
        if self._closed:
            return True
        with self._lock:
            first_shutdown_attempt = not self._closing
            self._closing = True
        if first_shutdown_attempt:
            active = self.runtime.store.list_object_tasks(include_terminal=False)
            for task in active:
                latest = self.runtime.store.get_object_task(task.task_id)
                if latest is not None and latest.status not in _TERMINAL_STATUSES:
                    self._mark_cancelled(latest, actor="runtime.shutdown", reason="runtime shutdown")
        self._drain_done_futures()
        with self._lock:
            futures = list(self._futures.values())
        if futures:
            wait(futures, timeout=self.config.object_tasks.shutdown_join_timeout_s)
        self._drain_done_futures()
        with self._lock:
            unfinished = [future for future in self._futures.values() if not future.done()]
        if unfinished:
            self.runtime.audit.record(
                actor="runtime.shutdown",
                action="object_task.shutdown_deferred",
                target="object_tasks",
                decision={"unfinished": len(unfinished)},
            )
            return False
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=self.config.object_tasks.shutdown_join_timeout_s)
        if self._thread.is_alive():
            self.runtime.audit.record(
                actor="runtime.shutdown",
                action="object_task.shutdown_deferred",
                target="object_tasks",
                decision={"reason": "event loop thread did not stop"},
            )
            return False
        self._closed = True
        return True

    async def _run_task(self, task_id: str) -> None:
        task = self._get(task_id)
        with self._lock:
            args = dict(self._pending_args.get(task_id, {}))
        context_metadata = self._context_metadata_for_resume(task)
        try:
            task = self._mark_running(task)
            if task.status != ObjectTaskStatus.RUNNING:
                return
            self._set_runner_status(str(task.runner_pid), ProcessStatus.RUNNING, "object task running")
            result = await self.runtime.tools.acall(
                str(task.runner_pid),
                task.tool,
                args,
                context_metadata=context_metadata,
            )
            latest_process = self.runtime.process.get(str(task.runner_pid))
            latest_task = self.runtime.store.get_object_task(task_id)
            if latest_task is None or latest_task.status in _TERMINAL_STATUSES:
                self._discard_process_owned_result(str(task.runner_pid), result.result_handle.oid if result.result_handle else None)
                return
            if result.ok:
                result_oid = result.result_handle.oid if result.result_handle is not None else None
                if result_oid is not None:
                    self.runtime.memory.transfer_owner(
                        ObjectOwnerKind.PROCESS,
                        str(task.runner_pid),
                        ObjectOwnerKind.OBJECT_TASK,
                        task_id,
                        [result_oid],
                        actor=f"object_task:{task_id}",
                        reason="object_task_result",
                    )
                    creator_handle = self.runtime.capability.handle_for_object(
                        task.creator_pid,
                        result_oid,
                        {ObjectRight.READ.value, ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value},
                        issued_by=f"object_task:{task_id}",
                    )
                    self.runtime._add_handle_to_process_view(task.creator_pid, creator_handle)
                    self.runtime.memory.link_objects_trusted(
                        f"object_task:{task_id}",
                        task.owner_oid,
                        RelationType.PRODUCED,
                        creator_handle.oid,
                        metadata={"task_id": task_id, "tool": task.tool},
                        reason="object_task_result",
                    )
                    self._grant_notify_result_if_requested(task, result_oid)
                if latest_process.status not in self.runtime.process.TERMINAL_STATUSES:
                    self.runtime.process.exit(str(task.runner_pid), result=result.result_handle)
                self._mark_succeeded(task_id, result, result_oid)
                return
            if latest_process.status not in self.runtime.process.TERMINAL_STATUSES:
                self.runtime.process.exit(str(task.runner_pid), failed=True, message=result.error or "object task failed")
            self._mark_failed(task_id, result.error or "object task failed")
        except asyncio.CancelledError:
            latest = self.runtime.store.get_object_task(task_id)
            if latest is not None and latest.status not in _TERMINAL_STATUSES:
                self._mark_cancelled(latest, actor="object_task", reason="cancelled")
            raise
        except HumanApprovalRequired as exc:
            self._mark_waiting(task_id, ObjectTaskStatus.WAITING_HUMAN, {"request_id": exc.request_id}, str(exc))
        except ProcessWaitRequired as exc:
            self._mark_waiting(task_id, ObjectTaskStatus.WAITING_PROCESS, {"child_pid": exc.child_pid}, str(exc))
        except ProcessMessageWaitRequired as exc:
            self._mark_waiting(task_id, ObjectTaskStatus.WAITING_MESSAGE, {"filters": exc.filters}, str(exc))
        except Exception as exc:
            self._mark_failed(task_id, str(exc))

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.set_default_executor(self._executor)
        self._loop_ready.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        # Sync tools run via asyncio.to_thread(), so the object-task loop must
        # drain its default executor before Runtime closes the shared store.
        self._loop.run_until_complete(self._loop.shutdown_default_executor())
        self._loop.close()

    def _schedule_task_locked(self, task_id: str) -> Future[Any]:
        if self._closing or self._closed:
            raise RuntimeError("object task manager is shutting down")
        future = asyncio.run_coroutine_threadsafe(self._run_task(task_id), self._loop)
        self._futures[task_id] = future
        future.add_done_callback(lambda _future, task_id=task_id: self._forget_future(task_id))
        return future

    def _has_active_future(self, task_id: str) -> bool:
        with self._lock:
            future = self._futures.get(task_id)
            return future is not None and not future.done()

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
        elif value:
            raw = {"enabled": True}
        else:
            raw = {"enabled": False}
        enabled = bool(raw.get("enabled", False))
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
        try:
            message = self.runtime.messages.post(
                sender=sender,
                recipient_pid=task.runner_pid,
                kind=task.owner_watch.kind,
                channel=task.owner_watch.channel,
                correlation_id=task.task_id,
                subject=subject,
                payload=payload,
            )
        except Exception as exc:
            self.runtime.events.emit(
                EventType.OBJECT_TASK_OWNER_CHANGE_UNDELIVERED,
                source=actor_pid,
                target=task.owner_oid,
                payload={
                    "task_id": task.task_id,
                    "runner_pid": task.runner_pid,
                    "event": change.get("event"),
                    "error": str(exc),
                },
                priority=EventPriority.HIGH,
            )
            self.runtime.audit.record(
                actor="object_task",
                action="object_task.owner_watch.undelivered",
                target=f"object_task:{task.task_id}",
                input_refs=[task.owner_oid],
                decision={"runner_pid": task.runner_pid, "event": change.get("event"), "error": str(exc)},
            )
            return False
        self.runtime.events.emit(
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
        self.runtime.audit.record(
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

    def _schedule_waiting_message_resume(self, task_id: str) -> None:
        with self._lock:
            latest = self.runtime.store.get_object_task(task_id)
            if latest is None or latest.status != ObjectTaskStatus.WAITING_MESSAGE:
                return
            future = self._futures.get(task_id)
            if future is not None and not future.done():
                return
            if task_id not in self._pending_args:
                self.runtime.audit.record(
                    actor="object_task",
                    action="object_task.owner_watch.resume_missing_pending",
                    target=f"object_task:{task_id}",
                    decision={"status": latest.status.value},
                )
                return
            if not self._can_resume_waiting_message(latest):
                self.runtime.audit.record(
                    actor="object_task",
                    action="object_task.owner_watch.resume_unsafe_replay_skipped",
                    target=f"object_task:{task_id}",
                    decision={"status": latest.status.value, "tool": latest.tool, "filters": latest.wait.get("filters")},
                )
                return
            self.runtime.audit.record(
                actor="object_task",
                action="object_task.owner_watch.resume",
                target=f"object_task:{task_id}",
                decision={"status": latest.status.value, "filters": latest.wait.get("filters")},
            )
            self._schedule_task_locked(task_id)

    def _schedule_waiting_human_resume(self, task: ObjectTask) -> bool:
        request_id = str(task.wait.get("request_id") or "")
        if not request_id:
            self._mark_failed(task.task_id, "object task waiting_human state is missing request_id")
            self._cleanup_task_state(task.task_id)
            return True
        try:
            request = self.runtime.human.get(request_id)
        except Exception as exc:
            self._mark_failed(task.task_id, str(exc))
            self._cleanup_task_state(task.task_id)
            return True
        if request.status == HumanRequestStatus.PENDING:
            return False
        if request.status == HumanRequestStatus.APPROVED or (
            task.tool == "request_permission" and request.status == HumanRequestStatus.REJECTED
        ):
            with self._lock:
                latest = self.runtime.store.get_object_task(task.task_id)
                if latest is None or latest.status != ObjectTaskStatus.WAITING_HUMAN:
                    return False
                future = self._futures.get(task.task_id)
                if future is not None and not future.done():
                    return False
                if task.task_id not in self._pending_args:
                    self.runtime.audit.record(
                        actor="object_task",
                        action="object_task.human_resume_missing_pending",
                        target=f"object_task:{task.task_id}",
                        decision={"request_id": request_id, "status": request.status.value},
                    )
                    return False
                self.runtime.audit.record(
                    actor="object_task",
                    action="object_task.human_resume",
                    target=f"object_task:{task.task_id}",
                    decision={"request_id": request_id, "status": request.status.value},
                )
                self._schedule_task_locked(task.task_id)
            return True
        self._mark_failed(
            task.task_id,
            f"human request was not approved: {request_id} status={request.status.value}",
        )
        self._cleanup_task_state(task.task_id)
        return True

    def _context_metadata_for_resume(self, task: ObjectTask) -> dict[str, Any]:
        if task.status != ObjectTaskStatus.WAITING_HUMAN:
            return {}
        request_id = task.wait.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return {}
        return {"human_resume_request_id": request_id}

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

    def _mark_running(self, task: ObjectTask) -> ObjectTask:
        now = utc_now()
        with self._lock:
            latest = self.runtime.store.get_object_task(task.task_id) or task
            if latest.status in _TERMINAL_STATUSES:
                return latest
            updated = replace(latest, status=ObjectTaskStatus.RUNNING, started_at=latest.started_at or now, updated_at=now)
            self.runtime.store.update_object_task(updated)
        self.runtime.events.emit(
            EventType.OBJECT_TASK_RUNNING,
            source=updated.creator_pid,
            target=updated.owner_oid,
            payload={"task_id": updated.task_id, "runner_pid": updated.runner_pid, "tool": updated.tool},
        )
        self.runtime.audit.record(
            actor="object_task",
            action="object_task.running",
            target=f"object_task:{updated.task_id}",
            decision={"runner_pid": updated.runner_pid, "tool": updated.tool},
        )
        return updated

    def _mark_succeeded(self, task_id: str, result: Any, result_oid: str | None) -> ObjectTask:
        now = utc_now()
        with self._lock:
            task = self._get(task_id)
            if task.status in _TERMINAL_STATUSES:
                return task
            updated = replace(
                task,
                status=ObjectTaskStatus.SUCCEEDED,
                result_oid=result_oid,
                error=None,
                updated_at=now,
                completed_at=now,
            )
            self.runtime.store.update_object_task(updated)
        self.runtime.events.emit(
            EventType.OBJECT_TASK_COMPLETED,
            source=updated.creator_pid,
            target=updated.owner_oid,
            payload={"task_id": task_id, "result_oid": result_oid, "tool": updated.tool},
        )
        self.runtime.audit.record(
            actor="object_task",
            action="object_task.completed",
            target=f"object_task:{task_id}",
            output_refs=[result_oid] if result_oid else [],
            decision={"ok": True, "tool": updated.tool, "call_id": getattr(result, "call_id", None)},
        )
        notified = self._notify(updated, phase="completed")
        self._cleanup_owner_pin_after_terminal(notified)
        return notified

    def _mark_failed(self, task_id: str, error: str) -> ObjectTask:
        now = utc_now()
        with self._lock:
            task = self._get(task_id)
            if task.status in _TERMINAL_STATUSES:
                return task
            updated = replace(
                task,
                status=ObjectTaskStatus.FAILED,
                error=error,
                updated_at=now,
                completed_at=now,
            )
            self.runtime.store.update_object_task(updated)
        self.runtime.events.emit(
            EventType.OBJECT_TASK_FAILED,
            source="object_task",
            target=updated.owner_oid,
            payload={"task_id": task_id, "tool": updated.tool, "error": error},
            priority=EventPriority.HIGH,
        )
        self.runtime.audit.record(
            actor="object_task",
            action="object_task.failed",
            target=f"object_task:{task_id}",
            decision={"tool": updated.tool, "error": sanitize_for_observability(error)},
        )
        notified = self._notify(updated, phase="failed")
        self._cleanup_owner_pin_after_terminal(notified)
        return notified

    def _mark_waiting(
        self,
        task_id: str,
        status: ObjectTaskStatus,
        wait: dict[str, Any],
        message: str,
    ) -> ObjectTask:
        now = utc_now()
        with self._lock:
            task = self._get(task_id)
            if task.status in _TERMINAL_STATUSES:
                return task
            updated = replace(task, status=status, wait=wait, error=message, updated_at=now)
            self.runtime.store.update_object_task(updated)
        self.runtime.events.emit(
            EventType.OBJECT_TASK_WAITING,
            source="object_task",
            target=updated.owner_oid,
            payload={"task_id": task_id, "status": status.value, "wait": wait, "tool": updated.tool},
            priority=EventPriority.HIGH,
        )
        self.runtime.audit.record(
            actor="object_task",
            action="object_task.waiting",
            target=f"object_task:{task_id}",
            decision={"status": status.value, "wait": wait, "tool": updated.tool},
        )
        return self._notify(updated, phase="waiting")

    def _mark_cancelled(self, task: ObjectTask, *, actor: str, reason: str) -> ObjectTask:
        now = utc_now()
        with self._lock:
            latest = self.runtime.store.get_object_task(task.task_id) or task
            if latest.status in _TERMINAL_STATUSES:
                return latest
            updated = replace(
                latest,
                status=ObjectTaskStatus.CANCELLED,
                error=reason,
                updated_at=now,
                completed_at=now,
            )
            self.runtime.store.update_object_task(updated)
        if updated.runner_pid is not None:
            process = self.runtime.store.get_process(updated.runner_pid)
            if process is not None and process.status not in self.runtime.process.TERMINAL_STATUSES:
                try:
                    self.runtime.process.signal(updated.runner_pid, "cancel", {"reason": reason})
                except Exception:
                    pass
        self.runtime.events.emit(
            EventType.OBJECT_TASK_CANCELLED,
            source=actor,
            target=updated.owner_oid,
            payload={"task_id": updated.task_id, "reason": reason},
        )
        self.runtime.audit.record(
            actor=actor,
            action="object_task.cancel",
            target=f"object_task:{updated.task_id}",
            decision={"reason": reason},
        )
        notified = self._notify(updated, phase="cancelled")
        self._cleanup_owner_pin_after_terminal(notified)
        self._cleanup_task_state(updated.task_id)
        return notified

    def _notify(self, task: ObjectTask, *, phase: str) -> ObjectTask:
        notification = task.notification
        if notification.recipient_pid is None:
            return task
        payload = {
            "type": "object_task",
            "phase": phase,
            "task_id": task.task_id,
            "owner_oid": task.owner_oid,
            "tool": task.tool,
            "status": task.status.value,
            "result_oid": task.result_oid,
            "error": task.error,
            "wait": task.wait,
        }
        subject = notification.subject or f"Object task {task.status.value}: {task.tool}"
        body = task.error or ""
        try:
            message = self.runtime.messages.post(
                sender=f"object_task:{task.task_id}",
                recipient_pid=notification.recipient_pid,
                kind=notification.kind,
                channel=notification.channel,
                correlation_id=task.task_id,
                subject=subject,
                body=body,
                payload=payload,
            )
            updated_notification = replace(
                notification,
                message_id=message.message_id,
                status=ObjectTaskNotificationStatus.DELIVERED,
                error=None,
            )
        except ProcessError as exc:
            recipient = self.runtime.store.get_process(notification.recipient_pid)
            status = (
                ObjectTaskNotificationStatus.UNDELIVERED_TERMINAL
                if recipient is not None and recipient.status in self.runtime.process.TERMINAL_STATUSES
                else ObjectTaskNotificationStatus.FAILED
            )
            updated_notification = replace(notification, status=status, error=str(exc))
            if status == ObjectTaskNotificationStatus.UNDELIVERED_TERMINAL:
                self.runtime.events.emit(
                    EventType.OBJECT_TASK_NOTIFICATION_UNDELIVERED,
                    source="object_task",
                    target=notification.recipient_pid,
                    payload={"task_id": task.task_id, "status": task.status.value, "reason": "terminal_process"},
                )
        updated = replace(task, notification=updated_notification, updated_at=utc_now())
        self.runtime.store.update_object_task(updated)
        return updated

    def _grant_notify_result_if_requested(self, task: ObjectTask, result_oid: str) -> None:
        if not self._grant_result_to_notify.get(task.task_id):
            return
        recipient = task.notification.recipient_pid
        if recipient is None or recipient == task.creator_pid:
            return
        decision = self.runtime.capability.authorize(task.creator_pid, f"object:{result_oid}", ObjectRight.GRANT)
        if not decision.allowed:
            raise CapabilityDenied(f"{task.creator_pid} cannot grant object task result: {result_oid}")
        self.runtime.capability.handle_for_object(
            recipient,
            result_oid,
            {ObjectRight.READ.value, ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value},
            issued_by=f"object_task:{task.task_id}",
        )

    def _set_runner_status(self, runner_pid: str, status: ProcessStatus, message: str | None = None) -> None:
        process = self.runtime.store.get_process(runner_pid)
        if process is None or process.status in self.runtime.process.TERMINAL_STATUSES:
            return
        process.status = status
        process.status_message = message
        process.updated_at = utc_now()
        self.runtime.store.update_process(process)

    def _terminalize_runner(self, runner_pid: str, *, reason: str) -> None:
        process = self.runtime.store.get_process(runner_pid)
        if process is None or process.status in self.runtime.process.TERMINAL_STATUSES:
            return
        try:
            self.runtime.process.signal(runner_pid, "cancel", {"reason": reason})
        except Exception:
            process = self.runtime.store.get_process(runner_pid)
            if process is not None and process.status not in self.runtime.process.TERMINAL_STATUSES:
                process.status = ProcessStatus.KILLED
                process.status_message = reason
                process.updated_at = utc_now()
                self.runtime.store.update_process(process)

    def _refresh_task_from_runner(self, task: ObjectTask) -> ObjectTask:
        if task.status in _TERMINAL_STATUSES or task.runner_pid is None:
            return task
        if self._has_active_future(task.task_id):
            return task
        process = self.runtime.store.get_process(task.runner_pid)
        if process is None or process.status not in self.runtime.process.TERMINAL_STATUSES:
            return task
        if process.status == ProcessStatus.KILLED:
            return self._mark_cancelled(task, actor="object_task.runner", reason=process.status_message or "runner killed")
        return self._mark_failed(
            task.task_id,
            process.status_message or f"object task runner ended before task completion: {process.status.value}",
        )

    def _discard_process_owned_result(self, runner_pid: str, result_oid: str | None) -> None:
        if result_oid is None:
            return
        obj = self.runtime.store.get_object(result_oid)
        if obj is None or obj.owner_kind != ObjectOwnerKind.PROCESS or obj.owner_id != runner_pid:
            return
        self.runtime.memory.delete_object_trusted(
            "object_task",
            result_oid,
            reason="cancelled_object_task_result",
        )
        self.runtime.audit.record(
            actor="object_task",
            action="object_task.discard_cancelled_result",
            target=f"object:{result_oid}",
            input_refs=[result_oid],
            decision={"runner_pid": runner_pid},
        )

    def _cleanup_owner_pin_after_terminal(self, task: ObjectTask) -> None:
        creator = self.runtime.store.get_process(task.creator_pid)
        if creator is None or creator.status not in self.runtime.process.TERMINAL_STATUSES:
            return
        self.runtime.memory.release_process_owned(task.creator_pid)

    def _assert_owner_rights(self, pid: str, owner: ObjectHandle, rights: set[str]) -> None:
        decisions = []
        for right in sorted(rights):
            decision = self.runtime.capability.authorize_handle(pid, owner, right)
            if not decision.allowed:
                raise CapabilityDenied(f"capability lacks {right}: {owner.oid}")
            decisions.append(decision)
        consumed: set[str] = set()
        for decision in decisions:
            if decision.consume_capability_id is None or decision.consume_capability_id in consumed:
                continue
            self.runtime.capability.consume_use(
                decision.consume_capability_id,
                used_by="object_task",
                reason="one-time object task owner permission consumed",
            )
            consumed.add(decision.consume_capability_id)

    def _require_related_notification_target(self, actor_pid: str, recipient_pid: str) -> None:
        actor = self.runtime.store.get_process(actor_pid)
        recipient = self.runtime.store.get_process(recipient_pid)
        if actor is None:
            raise NotFound(f"process not found: {actor_pid}")
        if recipient is None:
            raise NotFound(f"process not found: {recipient_pid}")
        if recipient.status in self.runtime.process.TERMINAL_STATUSES:
            raise ProcessError(f"cannot notify terminal process: {recipient_pid}")
        if actor.pid == recipient.pid or actor.parent_pid == recipient.pid or recipient.parent_pid == actor.pid:
            return
        raise ProcessError(f"{actor_pid} can only notify itself, its parent, or its direct children")

    def _require_concurrency_capacity(self, owner_oid: str) -> None:
        active = self.runtime.store.list_object_tasks(include_terminal=False)
        if len(active) >= self.config.object_tasks.max_running_global:
            raise ValidationError("object task global concurrency limit exceeded")
        per_object = [task for task in active if task.owner_oid == owner_oid]
        if len(per_object) >= self.config.object_tasks.max_running_per_object:
            raise ValidationError("object task per-object concurrency limit exceeded")

    def _require_task_visible(self, actor_pid: str, task: ObjectTask) -> None:
        if not self._can_view_task(actor_pid, task):
            raise CapabilityDenied(f"{actor_pid} cannot inspect object task: {task.task_id}")

    def _require_task_mutable(self, actor_pid: str, task: ObjectTask) -> None:
        if task.creator_pid == actor_pid:
            return
        decision = self.runtime.capability.authorize(actor_pid, f"object:{task.owner_oid}", ObjectRight.WRITE)
        if not decision.allowed:
            raise CapabilityDenied(f"{actor_pid} cannot cancel object task: {task.task_id}")

    def _can_view_task(self, actor_pid: str, task: ObjectTask) -> bool:
        if task.creator_pid == actor_pid:
            return True
        return self.runtime.capability.check(actor_pid, f"object:{task.owner_oid}", ObjectRight.READ)

    def _get(self, task_id: str) -> ObjectTask:
        task = self.runtime.store.get_object_task(task_id)
        if task is None:
            raise NotFound(f"object task not found: {task_id}")
        return task

    def _forget_future(self, task_id: str) -> None:
        with self._lock:
            self._futures.pop(task_id, None)
            latest = self.runtime.store.get_object_task(task_id)
            if latest is not None and latest.status not in _TERMINAL_STATUSES:
                return
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
