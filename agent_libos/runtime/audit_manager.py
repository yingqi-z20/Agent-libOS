from __future__ import annotations

from typing import Any

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import ValidationError
from agent_libos.ports import OperationPort
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.models import AuditRecord
from agent_libos.storage import EvidenceRepository


_AUDIT_QUERY_LIMIT = DEFAULT_CONFIG.gui.event_buffer_limit


class AuditManager:
    def __init__(
        self,
        store: EvidenceRepository,
        operations: OperationPort | None = None,
        *,
        query_limit: int = _AUDIT_QUERY_LIMIT,
    ) -> None:
        if type(query_limit) is not int or query_limit <= 0:
            raise ValidationError("audit query limit must be a positive integer")
        self.store = store
        self.operations = operations
        self._query_limit = query_limit

    def record(
        self,
        actor: str,
        action: str,
        target: str | None = None,
        input_refs: list[str] | None = None,
        output_refs: list[str] | None = None,
        capability_refs: list[str] | None = None,
        decision: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        parent_record_id: str | None = None,
    ) -> AuditRecord:
        record = AuditRecord(
            record_id=new_id("audit"),
            timestamp=utc_now(),
            actor=actor,
            action=action,
            target=target,
            input_refs=input_refs or [],
            output_refs=output_refs or [],
            capability_refs=capability_refs or [],
            decision=decision,
            correlation_id=correlation_id,
            parent_record_id=parent_record_id,
        )
        with self.store.transaction():
            self.store.insert_audit(record)
            if self.operations is not None:
                self.operations.link_evidence("audit", record.record_id, "audit")
                semantic_role = self._semantic_role(action)
                if semantic_role is not None:
                    self.operations.link_evidence("audit", record.record_id, semantic_role)
        return record

    @staticmethod
    def _semantic_role(action: str) -> str | None:
        if action == "capability.authorize":
            return "decision"
        if action.startswith("capability.") and any(
            marker in action for marker in ("reserve", "consume", "restore")
        ):
            return "reservation"
        if action.startswith("human.") and any(
            marker in action for marker in ("approve", "reject", "response", "terminal")
        ):
            return "approval"
        if action == "resource.charge":
            return "resource_charge"
        if action in {"tool.call", "syscall.result"}:
            return "result"
        return None

    def trace(
        self,
        limit: int | None = None,
        *,
        actor: str | None = None,
        target: str | None = None,
        match_any: bool = False,
        include_gui_presentation: bool = True,
        before_record_id: str | None = None,
    ) -> list[AuditRecord]:
        if actor is not None and type(actor) is not str:
            raise ValidationError("audit trace actor must be a string or null")
        if target is not None and type(target) is not str:
            raise ValidationError("audit trace target must be a string or null")
        selected_limit = limit
        if selected_limit is not None and (
            type(selected_limit) is not int
            or selected_limit <= 0
            or selected_limit > self._query_limit
        ):
            raise ValidationError(
                "audit trace limit must be a positive integer no greater than "
                f"{self._query_limit}"
            )
        if before_record_id is not None and type(before_record_id) is not str:
            raise ValidationError(
                "audit trace before_record_id must be a string or null"
            )
        if type(match_any) is not bool:
            raise ValidationError("audit trace match_any must be a boolean")
        if type(include_gui_presentation) is not bool:
            raise ValidationError(
                "audit trace include_gui_presentation must be a boolean"
            )
        filters = {
            "limit": selected_limit,
            "actor": actor,
            "target": target,
            "match_any": match_any,
        }
        if not include_gui_presentation:
            filters["include_gui_presentation"] = False
        if before_record_id is not None:
            filters["before_record_id"] = before_record_id
        return self.store.list_audit(**filters)
