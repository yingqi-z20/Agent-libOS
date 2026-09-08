"""Checkpoint references to completed local hosted-tool results, never sessions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_libos.llm.provider_continuation import (
    continuation_source_sha256,
    validated_continuation_marker,
)
from agent_libos.models import LLMCallRecord
from agent_libos.models.data_flow import DataFlowContext
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.snapshot import LocalProviderContinuationReference
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, to_jsonable


def _sha256(value: Any) -> str:
    return hashlib.sha256(dumps(to_jsonable(value)).encode("utf-8")).hexdigest()


class CheckpointContinuationAdapter:
    """Reuse checkpoint source validation for immutable local result references."""

    def __init__(
        self, *, processes: Any, config: Any,
        profile_scope: Callable[..., Any], validate_sources: Callable[..., None],
        fork_context: Callable[..., DataFlowContext], validate_fork_sources: Callable[..., None],
    ) -> None:
        self.processes = processes
        self.config = config
        self._profile_scope = profile_scope
        self._validate_sources = validate_sources
        self._fork_context = fork_context
        self._validate_fork_sources = validate_fork_sources

    def capture(self, pids: Sequence[str]) -> dict[str, dict[str, Any]]:
        references: dict[str, dict[str, Any]] = {}
        for pid in pids:
            process = self.processes.get_process(pid)
            # Durable TaskRun content cannot be exported through checkpoints.
            if process is None or process.task_run_id is not None:
                continue
            marker = self.processes.get_latest_llm_call(pid=pid, purpose="provider_continuation")
            if marker is None:
                continue
            manifest = validated_continuation_marker(marker)
            if manifest["state"] != "pending":
                continue
            if manifest["context_generation"] != self.processes.get_llm_context_generation(pid):
                continue
            reference = LocalProviderContinuationReference(
                pid=pid, marker_call_id=marker.call_id, marker_sha256=_sha256(marker),
                source_call_id=manifest["call_id"], source_sha256=manifest["source_sha256"],
                source_pid=manifest.get("source_pid", pid),
                profile_identity_sha256=manifest["profile_identity_sha256"],
                context_generation=manifest["context_generation"],
                payload_sha256=manifest["payload_sha256"],
            )
            self._payload(reference)
            references[pid] = reference.to_mapping()
        return references

    @staticmethod
    def _entries(snapshot: Mapping[str, Any]) -> dict[str, LocalProviderContinuationReference]:
        return {
            pid: LocalProviderContinuationReference.from_mapping(value)
            for pid, value in snapshot.get("provider_continuation_refs", {}).items()
        }

    def _payload(self, reference: LocalProviderContinuationReference) -> dict[str, Any]:
        if not self.config.llm.persist_full_io:
            raise ValidationError("checkpoint provider continuation is unavailable under payload retention")
        marker = self.processes.get_llm_call(reference.marker_call_id)
        if marker is None or marker.pid != reference.pid or _sha256(marker) != reference.marker_sha256:
            raise ValidationError("checkpoint provider continuation marker is unavailable or changed")
        manifest = validated_continuation_marker(marker)
        expected = {
            "state": "pending", "call_id": reference.source_call_id,
            "source_sha256": reference.source_sha256,
            "profile_identity_sha256": reference.profile_identity_sha256,
            "context_generation": reference.context_generation,
            "payload_sha256": reference.payload_sha256,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValidationError("checkpoint provider continuation reference changed")
        if manifest.get("source_pid", marker.pid) != reference.source_pid:
            raise ValidationError("checkpoint provider continuation source process changed")
        self._validate_source(reference, marker)
        payload = {"message": marker.messages, "flow_context": marker.raw_response}
        if _sha256(payload) != reference.payload_sha256:
            raise ValidationError("checkpoint provider continuation content is unavailable or changed")
        message = payload["message"]
        if (
            not isinstance(message, dict) or set(message) != {"role", "content"}
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str) or not message["content"].strip()
        ):
            raise ValidationError("checkpoint provider continuation transcript is invalid")
        DataFlowContext.from_dict(payload["flow_context"])
        return payload

    def _validate_source(
        self, reference: LocalProviderContinuationReference, marker: LLMCallRecord,
    ) -> None:
        source = self.processes.get_llm_call(reference.source_call_id)
        if (
            source is None or source.pid != reference.source_pid
            or source.status != "ok" or not source.completed_at
            or source.image_id != marker.image_id
            or source.request_options.get("provider_tools_enabled") is not True
            or source.tool_calls
            or continuation_source_sha256(source) != reference.source_sha256
        ):
            raise ValidationError("checkpoint provider continuation source is unavailable or changed")
        source_process = self.processes.get_process(reference.source_pid)
        if source_process is None or source_process.task_run_id is not None:
            raise ValidationError("checkpoint provider continuation cannot export TaskRun state")

    def validate(self, snapshot: Mapping[str, Any], *, remapped: Mapping[str, Any] | None) -> None:
        rows = {str(row["pid"]): row for row in snapshot["rows"]["processes"]}
        for pid, reference in self._entries(snapshot).items():
            row = rows.get(pid)
            if row is None or row.get("task_run_id") is not None or reference.pid != pid:
                raise ValidationError("checkpoint provider continuation is outside its process scope")
            if self._profile_scope(row)[0] != reference.profile_identity_sha256:
                raise ValidationError("checkpoint provider continuation profile changed")
            payload = self._payload(reference)
            context = DataFlowContext.from_dict(payload["flow_context"])
            self._validate_sources(pid, context, snapshot=snapshot)
            if remapped is not None:
                self._validate_fork_sources(remapped["pid_map"][pid], context, remapped=remapped)

    def publish_restore(self, snapshot: Mapping[str, Any]) -> None:
        for pid, reference in self._entries(snapshot).items():
            self._publish(reference, pid, self._payload(reference), snapshot=snapshot)

    def publish_fork(self, snapshot: Mapping[str, Any], *, remapped: Mapping[str, Any]) -> None:
        for pid, reference in self._entries(snapshot).items():
            payload = self._payload(reference)
            context = self._fork_context(DataFlowContext.from_dict(payload["flow_context"]), remapped)
            self._publish(reference, remapped["pid_map"][pid], {
                "message": payload["message"], "flow_context": context.to_dict(),
            }, snapshot=snapshot)

    def _publish(
        self, reference: LocalProviderContinuationReference, pid: str,
        payload: Mapping[str, Any], *, snapshot: Mapping[str, Any],
    ) -> None:
        process = self.processes.get_process(pid)
        if process is None:
            raise ValidationError("checkpoint provider continuation target process is missing")
        manifest = {
            "schema_version": 2, "state": "pending", "call_id": reference.source_call_id,
            "source_pid": reference.source_pid, "source_sha256": reference.source_sha256,
            "profile_identity_sha256": reference.profile_identity_sha256,
            "context_generation": self.processes.get_llm_context_generation(pid),
            "payload_sha256": _sha256(payload),
        }
        now = utc_now()
        self.processes.insert_llm_call(LLMCallRecord(
            call_id=new_id("llmcontinuation"), pid=pid, image_id=process.image_id,
            purpose="provider_continuation", status="ok", tool_calls=[], tools=[],
            messages=payload["message"], raw_response=payload["flow_context"],
            request_options={
                "provider_continuation": manifest,
                "checkpoint_provider_continuation": {
                    "checkpoint_id": snapshot["checkpoint_id"],
                    "marker_call_id": reference.marker_call_id,
                },
            },
            created_at=now, completed_at=now,
        ))
