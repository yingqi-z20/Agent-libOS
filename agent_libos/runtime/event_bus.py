from __future__ import annotations

import builtins
from typing import Any

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.ports import OperationPort
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.models import Event, EventPriority, EventType
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage import EvidenceRepository


_EVENT_QUERY_LIMIT = DEFAULT_CONFIG.gui.event_buffer_limit


class EventBus:
    def __init__(
        self,
        store: EvidenceRepository,
        operations: OperationPort | None = None,
    ) -> None:
        self.store = store
        self.operations = operations

    def emit(
        self,
        event_type: EventType | str,
        source: str,
        target: str | None = None,
        payload: dict[str, Any] | None = None,
        priority: EventPriority | str = EventPriority.NORMAL,
        correlation_id: str | None = None,
        causality: dict[str, Any] | None = None,
    ) -> Event:
        event = self._validated_event(
            event_id=new_id("evt"),
            event_type=event_type,
            source=source,
            target=target,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            causality=causality,
        )
        with self.store.transaction():
            self.store.insert_event(event)
            if self.operations is not None:
                self.operations.link_evidence("event", event.event_id, "event")
                if event.type == EventType.RESOURCE_CHARGED:
                    self.operations.link_evidence("event", event.event_id, "resource_charge")
        return event

    def emit_once(
        self,
        event_id: str,
        event_type: EventType | str,
        source: str,
        target: str | None = None,
        payload: dict[str, Any] | None = None,
        priority: EventPriority | str = EventPriority.NORMAL,
        correlation_id: str | None = None,
        causality: dict[str, Any] | None = None,
    ) -> Event:
        """Publish one Host-identified event, or return its exact prior row.

        The caller owns the stable semantic identity.  Reusing that identity
        for different event fields is an integrity error, while an exact retry
        is idempotent across threads and Runtime reopen.  Insertion and evidence
        linking share one transaction so a failed link leaves no orphan event.
        """

        if (
            type(event_id) is not str
            or not event_id
            or len(event_id) > 256
            or "\x00" in event_id
        ):
            raise ValidationError(
                "idempotent event id must be a non-empty string of at most 256 chars"
            )
        event = self._validated_event(
            event_id=event_id,
            event_type=event_type,
            source=source,
            target=target,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            causality=causality,
        )
        with self.store.transaction():
            existing = self.store.get_event(event_id)
            if existing is not None:
                self._assert_same_idempotent_event(existing, event)
                return existing
            self.store.insert_event(event)
            if self.operations is not None:
                self.operations.link_evidence("event", event.event_id, "event")
                if event.type == EventType.RESOURCE_CHARGED:
                    self.operations.link_evidence(
                        "event",
                        event.event_id,
                        "resource_charge",
                    )
        return event

    @staticmethod
    def _validated_event(
        *,
        event_id: str,
        event_type: EventType | str,
        source: str,
        target: str | None,
        payload: dict[str, Any] | None,
        priority: EventPriority | str,
        correlation_id: str | None,
        causality: dict[str, Any] | None,
    ) -> Event:
        if type(event_type) not in {EventType, str}:
            raise ValidationError("event type must be an EventType or string")
        if type(priority) not in {EventPriority, str}:
            raise ValidationError("event priority must be an EventPriority or string")
        if type(source) is not str:
            raise ValidationError("event source must be a string")
        if target is not None and type(target) is not str:
            raise ValidationError("event target must be a string or null")
        if correlation_id is not None and type(correlation_id) is not str:
            raise ValidationError("event correlation_id must be a string or null")
        if payload is not None and type(payload) is not dict:
            raise ValidationError("event payload must be an object or null")
        if causality is not None and type(causality) is not dict:
            raise ValidationError("event causality must be an object or null")
        try:
            selected_type = EventType(event_type)
        except (TypeError, ValueError) as exc:
            raise ValidationError("event type is invalid") from exc
        try:
            selected_priority = EventPriority(priority)
        except (TypeError, ValueError) as exc:
            raise ValidationError("event priority is invalid") from exc
        return Event(
            event_id=event_id,
            type=selected_type,
            source=source,
            target=target,
            payload=dict(payload) if payload is not None else {},
            priority=selected_priority,
            created_at=utc_now(),
            correlation_id=correlation_id,
            causality=dict(causality) if causality is not None else {},
        )

    @staticmethod
    def _assert_same_idempotent_event(existing: Event, proposed: Event) -> None:
        if (
            existing.type != proposed.type
            or existing.source != proposed.source
            or existing.target != proposed.target
            or existing.payload != proposed.payload
            or existing.priority != proposed.priority
            or existing.correlation_id != proposed.correlation_id
            or existing.causality != proposed.causality
        ):
            raise ValidationError(
                f"idempotent event identity collision: {proposed.event_id}"
            )

    def list(
        self,
        target: str | None = None,
        limit: int | None = None,
        before_event_id: str | None = None,
        after_event_id: str | None = None,
        *,
        include_gui_presentation: bool = True,
    ) -> builtins.list[Event]:
        if target is not None and type(target) is not str:
            raise ValidationError("event query target must be a string or null")
        if limit is not None and (
            type(limit) is not int
            or limit <= 0
            or limit > _EVENT_QUERY_LIMIT
        ):
            raise ValidationError(
                "event list limit must be a positive integer no greater than "
                f"{_EVENT_QUERY_LIMIT}"
            )
        if before_event_id is not None and type(before_event_id) is not str:
            raise ValidationError(
                "event query before_event_id must be a string or null"
            )
        if after_event_id is not None and type(after_event_id) is not str:
            raise ValidationError(
                "event query after_event_id must be a string or null"
            )
        if type(include_gui_presentation) is not bool:
            raise ValidationError(
                "event query include_gui_presentation must be a boolean"
            )
        if before_event_id is not None and after_event_id is not None:
            raise ValidationError(
                "event query cannot use before_event_id and after_event_id together"
            )
        filters = {
            "target": target,
            "limit": limit,
            "before_event_id": before_event_id,
            "after_event_id": after_event_id,
        }
        if not include_gui_presentation:
            filters["include_gui_presentation"] = False
        return self.store.list_events(**filters)
