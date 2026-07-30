from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.memory.data_labels import labels_for_explain, metadata_from_labels
from agent_libos.models import (
    DataFlowContext,
    DataLabels,
    EventPriority,
    EventType,
    MessageProcessWait,
    ObjectMetadata,
    ProcessMessage,
    ProcessMessageKind,
    ProcessMessageStatus,
    ProcessStatus,
    ProcessWaitState,
    legacy_status_message,
)
from agent_libos.models.exceptions import NotFound, ProcessError, ProcessMessageWaitRequired, ValidationError
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.storage import ProcessRepository
from agent_libos.tools.observability import ensure_json_size
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.public_errors import internal_exception_observation

if TYPE_CHECKING:
    from agent_libos.runtime.process_manager import ProcessManager


class ProcessMessageManager:
    """Per-process message queues with explicit read/ack semantics."""

    TERMINAL_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}

    def __init__(
        self,
        store: ProcessRepository,
        audit: AuditManager,
        events: EventBus,
        authority_policy: Any,
        *,
        process_manager: ProcessManager,
        transitions: ProcessTransitionService | None = None,
        config: AgentLibOSConfig | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.store = store
        self.audit = audit
        self.events = events
        self.authority_policy = authority_policy
        self._object_tasks: Any | None = None
        self._process_manager = process_manager
        self._transitions = transitions or ProcessTransitionService(store)

    def bind_object_tasks(self, object_tasks: Any) -> None:
        self._object_tasks = object_tasks

    def post(
        self,
        *,
        sender: str,
        recipient_pid: str,
        kind: ProcessMessageKind | str = ProcessMessageKind.NORMAL,
        channel: str = "default",
        correlation_id: str | None = None,
        reply_to: str | None = None,
        subject: str = "",
        body: str = "",
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source_oids: Iterable[str] | None = None,
    ) -> ProcessMessage:
        recipient = self.store.get_process(recipient_pid)
        if recipient is None:
            raise NotFound(f"process not found: {recipient_pid}")
        if recipient.status in self.TERMINAL_STATUSES:
            raise ProcessError(f"cannot post message to terminal process: {recipient_pid}")
        selected_kind = ProcessMessageKind(kind)
        subject_text = str(subject or "")
        body_text = str(body or "")
        payload_dict = dict(payload or {})
        message_metadata = self._message_metadata(sender, metadata, source_oids=source_oids)
        labels = metadata_from_labels(message_metadata)
        if labels is not None:
            self.authority_policy.assert_data_flow_labels(
                recipient_pid,
                DataLabels.from_object_metadata(labels),
            )
        self._validate_text_limit(subject_text, self.config.tools.message_subject_max_chars, "process message subject")
        self._validate_text_limit(body_text, self.config.tools.message_body_max_chars, "process message body")
        selected_correlation_id = self._normalize_optional_identifier(correlation_id, "process message correlation_id")
        selected_reply_to = self._normalize_optional_identifier(reply_to, "process message reply_to")
        ensure_json_size(
            {"payload": payload_dict, "metadata": message_metadata},
            self.config.tools.message_payload_max_bytes,
            "process message payload and metadata",
        )
        now = utc_now()
        message = ProcessMessage(
            message_id=new_id("pmsg"),
            sender=sender,
            recipient_pid=recipient_pid,
            kind=selected_kind,
            channel=self._normalize_channel(channel),
            correlation_id=selected_correlation_id,
            reply_to=selected_reply_to,
            subject=subject_text,
            body=body_text,
            payload=payload_dict,
            metadata=message_metadata,
            status=ProcessMessageStatus.UNREAD,
            created_at=now,
            updated_at=now,
        )
        # Recheck terminal state while holding the same transaction that
        # inserts the message and wakes a matching waiter.  This linearizes
        # post against process exit and makes evidence failures retry-safe.
        with self.store.transaction():
            recipient = self.store.get_process(recipient_pid)
            if recipient is None:
                raise NotFound(f"process not found: {recipient_pid}")
            if recipient.status in self.TERMINAL_STATUSES:
                raise ProcessError(f"cannot post message to terminal process: {recipient_pid}")
            self.store.insert_process_message(message)
            self.events.emit(
                EventType.PROCESS_MESSAGE_POSTED,
                source=sender,
                target=recipient_pid,
                payload={
                    "message_id": message.message_id,
                    "kind": message.kind.value,
                    "channel": message.channel,
                    "correlation_id": message.correlation_id,
                    "reply_to": message.reply_to,
                    "subject": message.subject,
                    "sender": sender,
                    "data_labels": message.metadata.get("data_labels"),
                },
                priority=EventPriority.HIGH if message.kind == ProcessMessageKind.INTERRUPT else EventPriority.NORMAL,
            )
            self.audit.record(
                actor=sender,
                action="process.message.post",
                target=f"process:{recipient_pid}",
                decision={
                    "message_id": message.message_id,
                    "kind": message.kind.value,
                    "channel": message.channel,
                    "correlation_id": message.correlation_id,
                    "reply_to": message.reply_to,
                    "subject": message.subject,
                    "data_labels": message.metadata.get("data_labels"),
                },
            )
            self._wake_if_waiting_for_message(message)
        if self._object_tasks is not None:
            try:
                self._object_tasks.notify_process_message(message)
            except Exception as exc:
                # The message, wake transition, event, and audit are already
                # committed.  A best-effort ObjectTask wake hook must not turn
                # that durable success into an apparent send failure that a
                # caller will retry. ObjectTask reads reconcile matching unread
                # messages, so retaining the row is the retry mechanism.
                try:
                    self.audit.record(
                        actor="process.message",
                        action="process.message.object_task_wake_failed",
                        target=f"process_message:{message.message_id}",
                        decision={
                            "recipient_pid": message.recipient_pid,
                            "error": internal_exception_observation(exc),
                        },
                    )
                except BaseException:
                    # Diagnostics are post-commit too and cannot redefine the
                    # outcome of the already-persisted message.
                    pass
        return message

    def send_from_process(
        self,
        sender_pid: str,
        recipient_pid: str,
        *,
        kind: ProcessMessageKind | str = ProcessMessageKind.NORMAL,
        channel: str = "default",
        correlation_id: str | None = None,
        reply_to: str | None = None,
        subject: str = "",
        body: str = "",
        payload: dict[str, Any] | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> ProcessMessage:
        self._require_related_process(sender_pid, recipient_pid)
        selected_sources = self._process_manager.flow_source_oids(sender_pid, source_oids)
        context = self._process_manager.flow_context(
            sender_pid,
            selected_sources,
            source_labels=source_labels,
            source_context=source_context,
        )
        metadata: dict[str, Any] = {
            "source_oids": selected_sources,
            "data_labels": context.labels.to_dict(),
            "data_flow_context": {
                "labels": context.labels.to_dict(),
                "source_refs": [ref.to_dict() for ref in context.source_refs],
                "materialization_id": context.materialization_id,
            },
        }
        return self.post(
            sender=sender_pid,
            recipient_pid=recipient_pid,
            kind=kind,
            channel=channel,
            correlation_id=correlation_id,
            reply_to=reply_to,
            subject=subject,
            body=body,
            payload=payload,
            metadata=metadata,
        )

    def observe_labels(self, pid: str, messages: Iterable[ProcessMessage]) -> list[str]:
        return self._process_manager.observe_message_labels(pid, messages)

    def unread(
        self,
        pid: str,
        *,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ProcessMessage]:
        self._require_process(pid)
        selected_limit = self._normalize_limit(limit)
        filters = self._filters(
            kind=kind,
            sender=sender,
            channel=channel,
            correlation_id=correlation_id,
            reply_to=reply_to,
            message_ids=message_ids,
        )
        return self.store.list_process_messages(
            pid,
            status=ProcessMessageStatus.UNREAD,
            kind=ProcessMessageKind(filters.get("kind")) if filters.get("kind") is not None else None,
            sender=filters.get("sender"),
            channel=filters.get("channel"),
            correlation_id=filters.get("correlation_id"),
            reply_to=filters.get("reply_to"),
            message_ids=filters.get("message_ids"),
            limit=selected_limit,
        )

    def list(
        self,
        pid: str,
        *,
        include_acked: bool = False,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ProcessMessage]:
        self._require_process(pid)
        selected_limit = self._normalize_limit(limit)
        filters = self._filters(
            kind=kind,
            sender=sender,
            channel=channel,
            correlation_id=correlation_id,
            reply_to=reply_to,
            message_ids=message_ids,
        )
        visible_statuses = (
            (ProcessMessageStatus.UNREAD, ProcessMessageStatus.ACKED)
            if include_acked
            else None
        )
        messages = self.store.list_process_messages(
            pid,
            status=None if include_acked else ProcessMessageStatus.UNREAD,
            statuses=visible_statuses,
            kind=ProcessMessageKind(filters.get("kind")) if filters.get("kind") is not None else None,
            sender=filters.get("sender"),
            channel=filters.get("channel"),
            correlation_id=filters.get("correlation_id"),
            reply_to=filters.get("reply_to"),
            message_ids=filters.get("message_ids"),
            limit=selected_limit,
        )
        return messages

    def list_page(
        self,
        pid: str,
        *,
        include_acked: bool = False,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> tuple[list[ProcessMessage], int]:
        """Return one bounded window and the full matching snapshot count."""

        self._require_process(pid)
        selected_limit = self._normalize_limit(limit)
        filters = self._filters(
            kind=kind,
            sender=sender,
            channel=channel,
            correlation_id=correlation_id,
            reply_to=reply_to,
            message_ids=message_ids,
        )
        status = None if include_acked else ProcessMessageStatus.UNREAD
        statuses = (
            (ProcessMessageStatus.UNREAD, ProcessMessageStatus.ACKED)
            if include_acked
            else None
        )
        with self.store.transaction():
            messages = self.store.list_process_messages(
                pid,
                status=status,
                statuses=statuses,
                kind=ProcessMessageKind(filters.get("kind")) if filters.get("kind") is not None else None,
                sender=filters.get("sender"),
                channel=filters.get("channel"),
                correlation_id=filters.get("correlation_id"),
                reply_to=filters.get("reply_to"),
                message_ids=filters.get("message_ids"),
                limit=selected_limit,
            )
            matching_count = self.store.count_process_messages(
                pid,
                status=status,
                statuses=statuses,
                kind=ProcessMessageKind(filters.get("kind")) if filters.get("kind") is not None else None,
                sender=filters.get("sender"),
                channel=filters.get("channel"),
                correlation_id=filters.get("correlation_id"),
                reply_to=filters.get("reply_to"),
                message_ids=filters.get("message_ids"),
            )
        return messages, matching_count

    def receive(
        self,
        pid: str,
        *,
        block: bool = False,
        include_acked: bool = False,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ProcessMessage]:
        filters = self._filters(
            kind=kind,
            sender=sender,
            channel=channel,
            correlation_id=correlation_id,
            reply_to=reply_to,
            message_ids=message_ids,
        )
        if block and message_ids == []:
            raise ValidationError("blocking process message receive requires a non-empty message id filter")
        if block and limit == 0:
            raise ValidationError("blocking process message receive requires a positive limit")
        # The empty read and WAITING_EVENT registration form one atomic state
        # transition with respect to post(). A post that wins the lock is seen
        # by the read; a post that loses it observes the registered waiter and
        # wakes it. There is no register-after-check lost-wakeup window.
        with self.store.transaction():
            messages = self.list(
                pid,
                include_acked=include_acked,
                kind=filters.get("kind"),
                sender=filters.get("sender"),
                channel=filters.get("channel"),
                correlation_id=filters.get("correlation_id"),
                reply_to=filters.get("reply_to"),
                message_ids=filters.get("message_ids"),
                limit=limit,
            )
            if messages or not block:
                return messages
            process = self._require_process(pid)
            wait_state = self._message_wait_state(filters)
            self._transitions.transition(
                pid,
                ProcessStatus.WAITING_EVENT,
                expected_revision=process.revision,
                expected_status=process.status,
                expected_state_generation=process.state_generation,
                wait_state=wait_state,
            )
            self.audit.record(
                actor=pid,
                action="process.message.wait",
                target=f"process:{pid}",
                decision={"filters": filters, "block": True},
            )
        raise ProcessMessageWaitRequired(
            recipient_pid=pid,
            filters=filters,
            message=f"{pid} is waiting for process message filters={filters}",
        )

    def receive_page(
        self,
        pid: str,
        *,
        block: bool = False,
        include_acked: bool = False,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> tuple[list[ProcessMessage], int]:
        """Receive a bounded window while retaining full-window count evidence."""

        filters = self._filters(
            kind=kind,
            sender=sender,
            channel=channel,
            correlation_id=correlation_id,
            reply_to=reply_to,
            message_ids=message_ids,
        )
        if block and message_ids == []:
            raise ValidationError("blocking process message receive requires a non-empty message id filter")
        if block and limit == 0:
            raise ValidationError("blocking process message receive requires a positive limit")
        with self.store.transaction():
            messages, matching_count = self.list_page(
                pid,
                include_acked=include_acked,
                kind=filters.get("kind"),
                sender=filters.get("sender"),
                channel=filters.get("channel"),
                correlation_id=filters.get("correlation_id"),
                reply_to=filters.get("reply_to"),
                message_ids=filters.get("message_ids"),
                limit=limit,
            )
            if messages or not block:
                return messages, matching_count
            process = self._require_process(pid)
            wait_state = self._message_wait_state(filters)
            self._transitions.transition(
                pid,
                ProcessStatus.WAITING_EVENT,
                expected_revision=process.revision,
                expected_status=process.status,
                expected_state_generation=process.state_generation,
                wait_state=wait_state,
            )
            self.audit.record(
                actor=pid,
                action="process.message.wait",
                target=f"process:{pid}",
                decision={"filters": filters, "block": True},
            )
        raise ProcessMessageWaitRequired(
            recipient_pid=pid,
            filters=filters,
            message=f"{pid} is waiting for process message filters={filters}",
        )

    def ack(
        self,
        pid: str,
        message_ids: list[str] | None = None,
        *,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
    ) -> list[ProcessMessage]:
        if message_ids is not None and not message_ids:
            return []
        selected_ids = set(message_ids or [])
        with self.store.transaction():
            # Explicit message ids already define the bounded ack set. Pass
            # that size down as the query limit so large read windows are not
            # truncated by the default message_read_limit before the id filter
            # is applied. Selection stays in the transaction so concurrent
            # ACK callers cannot both claim the same unread row.
            messages = self.list(
                pid,
                include_acked=False,
                kind=kind,
                sender=sender,
                channel=channel,
                correlation_id=correlation_id,
                reply_to=reply_to,
                message_ids=list(selected_ids) if selected_ids else None,
                limit=len(selected_ids) if selected_ids else None,
            )
            if selected_ids:
                messages = [message for message in messages if message.message_id in selected_ids]
            now = utc_now()
            acked: list[ProcessMessage] = []
            for message in messages:
                if not self.store.ack_process_message(
                    message.message_id,
                    recipient_pid=pid,
                    acked_at=now,
                    updated_at=now,
                ):
                    # A selected row changing before its CAS is an integrity
                    # conflict. Raising rolls back prior rows in this batch so
                    # a multi-message ACK is all-or-none.
                    raise ProcessError(
                        f"process message changed while acknowledging: {message.message_id}"
                    )
                message.status = ProcessMessageStatus.ACKED
                message.acked_at = now
                message.updated_at = now
                acked.append(message)
            if acked:
                acked_ids = [message.message_id for message in acked]
                self.events.emit(
                    EventType.PROCESS_MESSAGE_ACKED,
                    source=pid,
                    target=pid,
                    payload={"message_ids": acked_ids, "count": len(acked)},
                )
                self.audit.record(
                    actor=pid,
                    action="process.message.ack",
                    target=f"process:{pid}",
                    decision={"message_ids": acked_ids, "count": len(acked)},
                )
            return acked

    def notice(
        self,
        pid: str,
        *,
        kind: ProcessMessageKind | str,
        phase: str,
        source: str = "runtime",
        instruction: str | None = None,
    ) -> dict[str, Any] | None:
        messages = self.unread(pid, kind=kind)
        if not messages:
            return None
        selected_kind = ProcessMessageKind(kind)
        payload = {
            "phase": phase,
            "kind": selected_kind.value,
            "count": len(messages),
            "message_ids": [message.message_id for message in messages],
            "channels": sorted({message.channel for message in messages}),
            "correlation_ids": sorted({message.correlation_id for message in messages if message.correlation_id}),
            "instruction": instruction
            or "Call read_process_messages or receive_process_messages to inspect and acknowledge unread process messages.",
        }
        self.events.emit(
            EventType.PROCESS_MESSAGE_NOTICE,
            source=source,
            target=pid,
            payload=payload,
            priority=EventPriority.HIGH if selected_kind == ProcessMessageKind.INTERRUPT else EventPriority.NORMAL,
        )
        self.audit.record(
            actor=source,
            action="process.message.notice",
            target=f"process:{pid}",
            decision=payload,
        )
        return payload

    def _require_related_process(self, sender_pid: str, recipient_pid: str) -> None:
        sender = self._require_process(sender_pid)
        recipient = self.store.get_process(recipient_pid)
        if recipient is not None and sender.pid == recipient.pid:
            return
        if recipient is not None and sender.parent_pid == recipient.pid:
            return
        if recipient is not None and recipient.parent_pid == sender.pid:
            return
        # Missing and unrelated recipient identifiers intentionally share one
        # public denial so the relationship check is not an existence oracle.
        raise ProcessError(
            f"{sender_pid} can only message itself, its parent, or its direct children"
        )

    def _require_process(self, pid: str):
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return process

    def _wake_if_waiting_for_message(self, message: ProcessMessage) -> None:
        process = self.store.get_process(message.recipient_pid)
        if process is None or process.status != ProcessStatus.WAITING_EVENT:
            return
        wait_state = process.wait_state
        if not isinstance(wait_state, MessageProcessWait):
            return
        filters = wait_state.filters
        if not self._message_matches(message, filters):
            return
        token = self._transitions.wait_token(process)
        self._transitions.wake(token)
        self.audit.record(
            actor="process.message",
            action="process.message.wait_wake",
            target=f"process:{process.pid}",
            decision={"message_id": message.message_id, "filters": filters},
        )

    def _message_matches(self, message: ProcessMessage, filters: dict[str, Any]) -> bool:
        if message.status != ProcessMessageStatus.UNREAD:
            return False
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

    def _filters(
        self,
        *,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        filters = {
            "kind": ProcessMessageKind(kind).value if kind is not None else None,
            "sender": sender,
            "channel": self._normalize_channel(channel) if channel is not None else None,
            "correlation_id": self._normalize_optional_identifier(
                correlation_id,
                "process message correlation_id filter",
            ),
            "reply_to": self._normalize_optional_identifier(reply_to, "process message reply_to filter"),
            "message_ids": self._validate_message_ids(message_ids),
        }
        ensure_json_size(filters, self.config.tools.message_filter_json_max_bytes, "process message filters")
        return filters

    def _message_wait_state(self, filters: dict[str, Any]) -> MessageProcessWait:
        wait_state = MessageProcessWait(filters=filters)
        compatibility_message = legacy_status_message(wait_state, None)
        if (
            compatibility_message is not None
            and len(compatibility_message) > self.config.tools.message_wait_status_max_chars
        ):
            raise ValidationError(
                f"process message wait status exceeds {self.config.tools.message_wait_status_max_chars} chars"
            )
        return wait_state

    def has_matching_unread_wait(
        self,
        pid: str,
        wait_state: ProcessWaitState | None,
    ) -> bool:
        """Return whether a persisted message wait already has an unread match."""

        if not isinstance(wait_state, MessageProcessWait):
            return False
        filters = wait_state.filters
        return bool(
            self.unread(
                pid,
                kind=filters.get("kind"),
                sender=filters.get("sender"),
                channel=filters.get("channel"),
                correlation_id=filters.get("correlation_id"),
                reply_to=filters.get("reply_to"),
                message_ids=filters.get("message_ids"),
                limit=1,
            )
        )

    def normalize_channel(self, channel: str | None) -> str:
        """Validate a channel before a caller performs durable side effects."""

        return self._normalize_channel(channel)

    def validate_body(self, body: str) -> str:
        """Validate message body bounds without posting a message."""

        selected = str(body or "")
        self._validate_text_limit(
            selected,
            self.config.tools.message_body_max_chars,
            "process message body",
        )
        return selected

    def _normalize_channel(self, channel: str | None) -> str:
        selected = (channel or "default").strip()
        if not selected:
            raise ProcessError("process message channel must be non-empty")
        if len(selected) > 128:
            raise ProcessError("process message channel is too long")
        return selected

    def _message_metadata(
        self,
        sender: str,
        metadata: dict[str, Any] | None,
        *,
        source_oids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        selected = dict(metadata or {})
        if source_oids is not None:
            if isinstance(source_oids, (str, bytes)):
                raise ValidationError("process message source_oids must be a collection")
            selected_oids = list(dict.fromkeys(str(oid or "").strip() for oid in source_oids))
            if any(not oid for oid in selected_oids):
                raise ValidationError("process message source_oids cannot contain empty Object ids")
            if self._process_manager.data_flow is not None:
                context = self._process_manager.data_flow.context_from_trusted_source_oids(selected_oids)
            else:
                labels = self._process_manager.flow_metadata(selected_oids)
                context = DataFlowContext(labels=DataLabels.from_object_metadata(labels))
            selected["source_oids"] = selected_oids
            selected["data_labels"] = context.labels.to_dict()
            selected["data_flow_context"] = {
                "labels": context.labels.to_dict(),
                "source_refs": [ref.to_dict() for ref in context.source_refs],
                "materialization_id": context.materialization_id,
            }
        source_oids = selected.get("source_oids")
        if source_oids is None:
            selected["source_oids"] = []
        elif isinstance(source_oids, list):
            selected["source_oids"] = [str(oid) for oid in source_oids]
        else:
            raise ValidationError("process message metadata source_oids must be a list")
        labels = metadata_from_labels(selected)
        if labels is None:
            labels = ObjectMetadata(
                origin=sender,
                trust_level="user_asserted" if sender.startswith("human:") else "unknown",
            )
        selected["data_labels"] = labels_for_explain(labels)
        return selected

    def _normalize_limit(self, limit: int | None) -> int:
        if limit is not None and type(limit) is not int:
            raise ValidationError("process message read limit must be an integer")
        selected = self.config.tools.message_read_limit if limit is None else limit
        if selected < 0:
            raise ValidationError("process message read limit must be non-negative")
        if selected > self.config.tools.message_read_hard_limit:
            raise ValidationError(
                f"process message read limit exceeds {self.config.tools.message_read_hard_limit}"
            )
        return selected

    def _validate_message_ids(self, message_ids: list[str] | None) -> list[str] | None:
        if message_ids is None:
            return None
        if len(message_ids) > self.config.tools.message_filter_ids_hard_limit:
            raise ValidationError(
                f"process message id filter exceeds {self.config.tools.message_filter_ids_hard_limit}"
            )
        checked: list[str] = []
        for index, message_id in enumerate(message_ids):
            checked.append(self._normalize_identifier(message_id, f"process message id filter[{index}]"))
        return checked

    def _validate_text_limit(self, value: str, limit: int, label: str) -> None:
        if len(value) > limit:
            raise ValidationError(f"{label} exceeds {limit} chars")

    def _normalize_optional_identifier(self, value: str | None, label: str) -> str | None:
        if value is None:
            return None
        return self._normalize_identifier(value, label)

    def _normalize_identifier(self, value: Any, label: str) -> str:
        if not isinstance(value, str):
            raise ValidationError(f"{label} must be a string")
        if not value:
            raise ValidationError(f"{label} must be non-empty")
        if "\x00" in value:
            raise ValidationError(f"{label} cannot contain NUL bytes")
        if len(value) > self.config.tools.message_id_max_chars:
            raise ValidationError(f"{label} exceeds {self.config.tools.message_id_max_chars} chars")
        return value
