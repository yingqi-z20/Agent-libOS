from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from typing import Any, AbstractSet

from agent_libos.models import (
    EventType,
    ObjectTask,
    ObjectTaskNotificationStatus,
    ObjectTaskStatus,
    ProcessStatus,
)
from agent_libos.models.exceptions import CapabilityDenied, ProcessError
from agent_libos.ports import EventPort
from agent_libos.runtime.message_manager import ProcessMessageManager
from agent_libos.storage import ProcessRepository
from agent_libos.utils.ids import utc_now
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)


_TERMINAL_NOTIFICATION_PHASES = {
    ObjectTaskStatus.SUCCEEDED: "completed",
    ObjectTaskStatus.FAILED: "failed",
    ObjectTaskStatus.CANCELLED: "cancelled",
    ObjectTaskStatus.ABANDONED: "abandoned",
    ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN: (
        "result_unavailable_after_reopen"
    ),
}
_WAITING_NOTIFICATION_STATUSES = {
    ObjectTaskStatus.WAITING_HUMAN,
    ObjectTaskStatus.WAITING_PROCESS,
    ObjectTaskStatus.WAITING_MESSAGE,
}


class ObjectTaskNotificationService:
    """Durable ObjectTask message delivery and terminal retry policy."""

    def __init__(
        self,
        records: ProcessRepository,
        messages: ProcessMessageManager,
        events: EventPort,
        terminal_process_statuses: AbstractSet[ProcessStatus],
        lock: AbstractContextManager[Any],
    ) -> None:
        self._records = records
        self._messages = messages
        self._events = events
        self._terminal_process_statuses = terminal_process_statuses
        self._lock = lock

    def notify(self, task: ObjectTask, *, phase: str) -> ObjectTask:
        # A delayed waiting delivery must not overwrite a newer terminal row.
        # Reload and validate the phase while holding the same task lock used by
        # state transitions.  The pending/failed check also makes retries for one
        # durable phase idempotent.
        with self._lock, self._records.transaction():
            latest = self._records.get_object_task(task.task_id) or task
            if self._phase_for_status(latest.status) != phase:
                return latest
            if latest.notification.status not in {
                ObjectTaskNotificationStatus.NONE,
                ObjectTaskNotificationStatus.FAILED,
            }:
                return latest
            return self._notify_in_transaction(latest, phase=phase)

    def begin_phase(self, task: ObjectTask) -> ObjectTask:
        """Reset the legacy single notification slot for a new task phase.

        Keeping one slot preserves the persisted representation, while clearing
        the prior message id prevents a delivered waiting notification from
        being mistaken for delivery of a later terminal transition.
        """

        return replace(
            task,
            notification=replace(
                task.notification,
                message_id=None,
                status=ObjectTaskNotificationStatus.NONE,
                error=None,
            ),
        )

    def notify_terminal(self, task: ObjectTask, *, phase: str) -> ObjectTask:
        with self._lock:
            return self._notify_terminal_locked(task, phase=phase)

    def retry_terminal(self, task: ObjectTask) -> ObjectTask:
        # Eligibility is deliberately decided only after notify_terminal has
        # acquired the task lock and reloaded the durable row. A stale NONE
        # snapshot must never overwrite a concurrently delivered message id.
        return self.notify_terminal(task, phase="retry")

    def _notify_in_transaction(self, task: ObjectTask, *, phase: str) -> ObjectTask:
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
        source_oids = [task.owner_oid]
        if task.result_oid is not None:
            source_oids.append(task.result_oid)
        try:
            message = self._messages.post(
                sender=f"object_task:{task.task_id}",
                recipient_pid=notification.recipient_pid,
                kind=notification.kind,
                channel=notification.channel,
                correlation_id=task.task_id,
                subject=(
                    notification.subject
                    or f"Object task {task.status.value}: {task.tool}"
                ),
                body=task.error or "",
                payload=payload,
                source_oids=source_oids,
            )
            updated_notification = replace(
                notification,
                message_id=message.message_id,
                status=ObjectTaskNotificationStatus.DELIVERED,
                error=None,
            )
        except (CapabilityDenied, ProcessError) as exc:
            recipient = self._records.get_process(notification.recipient_pid)
            status = (
                ObjectTaskNotificationStatus.UNDELIVERED_TERMINAL
                if recipient is not None
                and recipient.status in self._terminal_process_statuses
                else ObjectTaskNotificationStatus.FAILED
            )
            public_error = self._record_failure_diagnostic(
                task,
                phase=phase,
                status=status,
                error=exc,
            )
            updated_notification = replace(
                notification,
                status=status,
                error=public_error["message"],
            )
            if status == ObjectTaskNotificationStatus.UNDELIVERED_TERMINAL:
                self._events.emit(
                    EventType.OBJECT_TASK_NOTIFICATION_UNDELIVERED,
                    source="object_task",
                    target=notification.recipient_pid,
                    payload={
                        "task_id": task.task_id,
                        "status": task.status.value,
                        "reason": "terminal_process",
                    },
                    correlation_id=public_error["correlation_id"],
                )
        updated = replace(
            task,
            notification=updated_notification,
            updated_at=utc_now(),
        )
        self._records.update_object_task(updated)
        return updated

    def _notify_terminal_locked(self, task: ObjectTask, *, phase: str) -> ObjectTask:
        """Settle post-commit delivery without hiding a terminal transition."""

        latest = self._records.get_object_task(task.task_id) or task
        selected_phase = _TERMINAL_NOTIFICATION_PHASES.get(latest.status)
        if selected_phase is None or latest.notification.recipient_pid is None:
            return latest
        try:
            with self._records.transaction():
                latest = self._records.get_object_task(task.task_id) or latest
                selected_phase = _TERMINAL_NOTIFICATION_PHASES.get(latest.status)
                if (
                    selected_phase is None
                    or latest.notification.recipient_pid is None
                ):
                    return latest
                if (
                    latest.notification.status
                    == ObjectTaskNotificationStatus.DELIVERED
                ):
                    if self._delivered_message_matches_phase(
                        latest,
                        phase=selected_phase,
                    ):
                        return latest
                    # Legacy rows did not record a notification phase. If their
                    # one delivered slot points at a waiting message, make the
                    # terminal phase pending before publishing its own message.
                    latest = self.begin_phase(latest)
                    self._records.update_object_task(latest)
                elif latest.notification.status not in {
                    ObjectTaskNotificationStatus.NONE,
                    ObjectTaskNotificationStatus.FAILED,
                }:
                    return latest
                return self.notify(latest, phase=selected_phase)
        except Exception as exc:
            with self._records.transaction():
                latest = self._records.get_object_task(task.task_id) or latest
                if (
                    latest.notification.status
                    == ObjectTaskNotificationStatus.DELIVERED
                    and self._delivered_message_matches_phase(
                        latest,
                        phase=selected_phase,
                    )
                ):
                    return latest
                public_error = self._record_failure_diagnostic(
                    latest,
                    phase=selected_phase,
                    status=ObjectTaskNotificationStatus.FAILED,
                    error=exc,
                )
                notification = replace(
                    latest.notification,
                    message_id=None,
                    status=ObjectTaskNotificationStatus.FAILED,
                    error=public_error["message"],
                )
                updated = replace(
                    latest,
                    notification=notification,
                    updated_at=utc_now(),
                )
                self._records.update_object_task(updated)
                return updated

    def _delivered_message_matches_phase(
        self,
        task: ObjectTask,
        *,
        phase: str,
    ) -> bool:
        notification = task.notification
        if (
            notification.status != ObjectTaskNotificationStatus.DELIVERED
            or notification.message_id is None
            or notification.recipient_pid is None
        ):
            return False
        message = self._records.get_process_message(notification.message_id)
        if message is None:
            return False
        payload = message.payload
        return (
            message.sender == f"object_task:{task.task_id}"
            and message.recipient_pid == notification.recipient_pid
            and message.correlation_id == task.task_id
            and payload.get("type") == "object_task"
            and payload.get("task_id") == task.task_id
            and payload.get("phase") == phase
            and payload.get("status") == task.status.value
        )

    @staticmethod
    def _phase_for_status(status: ObjectTaskStatus) -> str | None:
        terminal_phase = _TERMINAL_NOTIFICATION_PHASES.get(status)
        if terminal_phase is not None:
            return terminal_phase
        if status in _WAITING_NOTIFICATION_STATUSES:
            return "waiting"
        return None

    def _record_failure_diagnostic(
        self,
        task: ObjectTask,
        *,
        phase: str,
        status: ObjectTaskNotificationStatus,
        error: BaseException,
    ) -> dict[str, str]:
        public_error = public_error_envelope(
            error,
            code="object_task_notification_failed",
        )
        self._messages.audit.record(
            actor="object_task",
            action="object_task.notification_failed",
            target=f"object_task:{task.task_id}",
            input_refs=[task.owner_oid],
            output_refs=[task.result_oid] if task.result_oid is not None else [],
            decision={
                "phase": phase,
                "recipient_pid": task.notification.recipient_pid,
                "notification_status": status.value,
                "public_error": dict(public_error),
                "internal_error": internal_exception_observation(
                    error,
                    correlation_id=public_error["correlation_id"],
                ),
            },
            correlation_id=public_error["correlation_id"],
        )
        return public_error
