from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_libos.models import DataFlowContext
from agent_libos.ports.data_flow import DataFlowPort
from agent_libos.ports.operations import OperationPort
from agent_libos.storage.repositories import EvidenceRepository, ProcessRepository
from agent_libos.utils.ids import new_id

PENDING_METADATA_FILTER_KEY = "__agent_libos_pending_metadata__"
PENDING_TASK_RUN_TRANSCRIPT_KEY = "task_run_transcript_call_id"


class LLMPendingActionService:
    """Own durable and in-memory state for resumable LLM actions."""

    def __init__(
        self,
        *,
        processes: ProcessRepository,
        evidence: EvidenceRepository,
        operations: OperationPort,
        data_flow: DataFlowPort,
        restore_child_goal: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._processes = processes
        self._evidence = evidence
        self._operations = operations
        self._data_flow = data_flow
        self._restore_child_goal = restore_child_goal
        self._human_actions: dict[str, dict[str, Any]] = {}
        self._llm_release_actions: dict[str, dict[str, Any]] = {}
        self._wait_actions: dict[str, dict[str, Any]] = {}
        self._message_actions: dict[str, dict[str, Any]] = {}

    def has_memory(self, pid: str, wait_type: str) -> bool:
        selected = self._mapping_for(wait_type)
        return selected is not None and pid in selected

    def remember(
        self,
        pid: str,
        wait_type: str,
        action: dict[str, Any],
    ) -> None:
        selected = self._mapping_for(wait_type)
        if selected is None:
            raise ValueError(f"unsupported pending LLM wait type: {wait_type}")
        selected[pid] = dict(action)

    def require_memory(self, pid: str, wait_type: str) -> dict[str, Any]:
        selected = self._mapping_for(wait_type)
        if selected is None:
            raise ValueError(f"unsupported pending LLM wait type: {wait_type}")
        return selected[pid]

    def forget_generation(
        self,
        pid: str,
        wait_type: str,
        resume_token: str,
    ) -> None:
        selected = self._mapping_for(wait_type)
        if selected is None:
            raise ValueError(f"unsupported pending LLM wait type: {wait_type}")
        current = selected.get(pid)
        if current is not None and pending_resume_token(current) == resume_token:
            selected.pop(pid, None)

    def get(self, pid: str) -> dict[str, Any] | None:
        return self._processes.get_llm_pending_action(pid)

    def list(self, *, status: str | None = None) -> list[dict[str, Any]]:
        return self._processes.list_llm_pending_actions(status=status)

    def claim(self, pid: str, *, resume_token: str) -> dict[str, Any] | None:
        return self._processes.claim_llm_pending_action(pid, resume_token=resume_token)

    def complete(self, pid: str, *, resume_token: str) -> None:
        if not self._processes.complete_llm_pending_action(pid, resume_token=resume_token):
            raise RuntimeError(f"pending LLM action was not claimed before completion: {pid}")

    def persist(
        self,
        pid: str,
        *,
        wait_type: str,
        action: dict[str, Any],
        content_preview: str,
        tool_call_count: int,
        request_id: str | None = None,
        child_pid: str | None = None,
        filters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        response_id: str | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        resume_token: str | None = None,
        status: str = "pending",
    ) -> str:
        if status not in {"pending", "completed"}:
            raise ValueError(f"unsupported pending LLM action status: {status}")
        selected_token = resume_token or new_id("llmwait")
        llm_operation = self._operations.current()
        tool_operation_id = self._waiting_tool_operation(
            llm_operation,
            request_id=request_id,
            child_pid=child_pid,
        )
        stored_filters = dict(filters or {})
        stored_filters.pop(PENDING_METADATA_FILTER_KEY, None)
        if metadata:
            stored_filters[PENDING_METADATA_FILTER_KEY] = dict(metadata)
        self._processes.upsert_llm_pending_action(
            pid,
            {
                "resume_token": selected_token,
                "llm_operation_id": llm_operation.operation_id if llm_operation is not None else None,
                "tool_operation_id": tool_operation_id,
                "wait_type": wait_type,
                "request_id": request_id,
                "child_pid": child_pid,
                "response_id": response_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "filters": stored_filters,
                "action": dict(action),
                "data_flow_context": serialize_data_flow_context(self._data_flow.current_context()),
                "content_preview": content_preview,
                "tool_call_count": tool_call_count,
                "status": status,
            },
        )
        return selected_token

    def _waiting_tool_operation(
        self,
        llm_operation: Any | None,
        *,
        request_id: str | None,
        child_pid: str | None,
    ) -> str | None:
        if llm_operation is None:
            return None
        evidence_id = request_id or child_pid
        evidence_types = ("human_request",) if request_id else (("process",) if child_pid else ())
        candidates: list[Any] = []
        if evidence_id is not None and evidence_types:
            candidates = [
                candidate
                for candidate in self._operations.operation_for_evidence(evidence_types, evidence_id)
                if candidate.root_operation_id == llm_operation.root_operation_id
                and candidate.operation_id != llm_operation.operation_id
            ]
        selected = select_waiting_tool_operation(candidates)
        if selected is not None:
            return selected
        waiting = [
            candidate
            for candidate in self._evidence.list_operations(
                root_operation_id=llm_operation.root_operation_id,
                state="waiting",
            )
            if candidate.operation_id != llm_operation.operation_id
        ]
        return select_waiting_tool_operation(waiting)

    def synchronize(self, pid: str) -> dict[str, Any] | None:
        pending = self.get(pid)
        if pending is None or pending.get("status") != "pending":
            self.clear_memory(pid)
            return pending
        selected = self._mapping_for(str(pending.get("wait_type") or ""))
        token = pending_resume_token(pending)
        current = selected.get(pid) if selected is not None else None
        other_present = any(
            pid in mapping
            for mapping in self._mappings()
            if mapping is not selected
        )
        if (
            selected is not None
            and current is not None
            and pending_resume_token(current) == token
            and not other_present
        ):
            return pending
        self.clear_memory(pid)
        self.hydrate(pending)
        return pending

    def clear_memory(self, pid: str) -> None:
        for mapping in self._mappings():
            mapping.pop(pid, None)

    def hydrate(self, pending: dict[str, Any]) -> None:
        pid = str(pending["pid"])
        wait_type = str(pending["wait_type"])
        common = _hydrated_common(pending)
        if wait_type == "llm_release" and pending.get("request_id"):
            self._llm_release_actions[pid] = {**common, "request_id": str(pending["request_id"])}
            return
        if wait_type == "human" and pending.get("request_id"):
            self._human_actions[pid] = {**common, "request_id": str(pending["request_id"])}
            return
        if wait_type == "child" and pending.get("child_pid"):
            restored = {**common, "child_pid": str(pending["child_pid"])}
            self._wait_actions[pid] = restored
            if self._restore_child_goal is not None:
                self._restore_child_goal({**pending, **restored})
            return
        if wait_type == "message":
            self._message_actions[pid] = {
                **common,
                "filters": pending_message_filters(pending),
            }
            return
        raise RuntimeError(f"invalid durable pending LLM action for {pid}: wait_type={wait_type!r}")

    def _mappings(self) -> tuple[dict[str, dict[str, Any]], ...]:
        return (
            self._llm_release_actions,
            self._human_actions,
            self._wait_actions,
            self._message_actions,
        )

    def _mapping_for(self, wait_type: str) -> dict[str, dict[str, Any]] | None:
        return {
            "llm_release": self._llm_release_actions,
            "human": self._human_actions,
            "child": self._wait_actions,
            "message": self._message_actions,
        }.get(wait_type)


def select_waiting_tool_operation(candidates: list[Any]) -> str | None:
    selected = [
        candidate
        for candidate in candidates
        if candidate.kind.value == "tool_call" and candidate.state.value == "waiting"
    ]
    if len(selected) == 1:
        return selected[0].operation_id
    if len(candidates) == 1:
        return candidates[0].operation_id
    return None


def serialize_data_flow_context(context: DataFlowContext) -> dict[str, Any]:
    return {
        "labels": context.labels.to_dict(),
        "source_refs": [item.to_dict() for item in context.source_refs],
        "materialization_id": context.materialization_id,
    }


def pending_resume_token(pending: dict[str, Any]) -> str:
    token = str(pending.get("resume_token") or "").strip()
    if not token:
        raise RuntimeError("durable pending LLM action is missing its resume token")
    return token


def pending_data_flow_metadata(pending: dict[str, Any]) -> dict[str, Any]:
    return {"data_flow_context": dict(pending.get("data_flow_context") or {})}


def pending_metadata(pending: dict[str, Any]) -> dict[str, Any]:
    selected = _raw_pending_metadata(pending)
    selected.pop(PENDING_TASK_RUN_TRANSCRIPT_KEY, None)
    return selected


def pending_task_run_transcript_call_id(pending: dict[str, Any]) -> str | None:
    direct = pending.get(PENDING_TASK_RUN_TRANSCRIPT_KEY)
    value = (
        direct
        if direct is not None
        else _raw_pending_metadata(pending).get(PENDING_TASK_RUN_TRANSCRIPT_KEY)
    )
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RuntimeError("durable TaskRun pending transcript call id is invalid")
    return value


def _raw_pending_metadata(pending: dict[str, Any]) -> dict[str, Any]:
    direct = pending.get("pending_metadata")
    if isinstance(direct, dict):
        return dict(direct)
    filters = pending.get("filters")
    if not isinstance(filters, dict):
        return {}
    value = filters.get(PENDING_METADATA_FILTER_KEY)
    return dict(value) if isinstance(value, dict) else {}


def pending_message_filters(pending: dict[str, Any]) -> dict[str, Any]:
    filters = dict(pending.get("filters") or {})
    filters.pop(PENDING_METADATA_FILTER_KEY, None)
    return filters


def _hydrated_common(pending: dict[str, Any]) -> dict[str, Any]:
    return {
        "resume_token": pending_resume_token(pending),
        "llm_operation_id": pending.get("llm_operation_id"),
        "tool_operation_id": pending.get("tool_operation_id"),
        "action": dict(pending.get("action") or {}),
        "content_preview": str(pending.get("content_preview") or ""),
        "tool_call_count": int(pending.get("tool_call_count") or 0),
        "response_id": str(pending["response_id"]) if pending.get("response_id") else None,
        "tool_call_id": str(pending["tool_call_id"]) if pending.get("tool_call_id") else None,
        "tool_name": str(pending["tool_name"]) if pending.get("tool_name") else None,
        "data_flow_context": dict(pending.get("data_flow_context") or {}),
        "pending_metadata": pending_metadata(pending),
        PENDING_TASK_RUN_TRANSCRIPT_KEY: pending_task_run_transcript_call_id(
            pending
        ),
    }
