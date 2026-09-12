"""Host-only, certified context transitions for pending local provider results."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any

from agent_libos.models import DataFlowContext, DataLabels, DataSourceRef, LLMCallRecord
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, to_jsonable


def _validate_sources(
    pid: str, context: DataFlowContext, *, data_flow: Any,
    file_resource_resolver: Callable[[str], str] | None,
) -> None:
    def resolve_file(path: str) -> str:
        if file_resource_resolver is None:
            raise ValidationError("provider continuation file source resolver is unavailable")
        return file_resource_resolver(path)

    data_flow.validate_replay_sources(
        pid, context, file_resource_resolver=resolve_file,
        allow_recovered_source_snapshots=True,
    )


@contextmanager
def provider_continuation_context_update_scope(
    pid: str, *, processes: Any, unit_of_work: Any,
    pending_data: Callable[[str], Mapping[str, Any] | None],
    context_memory: Any, data_flow: Any,
    file_resource_resolver: Callable[[str], str] | None,
    persist_full_io: bool, volatile_payloads: MutableMapping[str, dict[str, Any]],
) -> Iterator[None]:
    """Follow one authenticated Host append without accepting arbitrary writes."""

    oid = context_memory.context_oid(pid)
    pending = pending_data(pid) if oid is not None else None
    if pending is None:
        yield
        return
    result = None
    with unit_of_work.transaction(include_object_payloads=True):
        current_pending = pending_data(pid)
        if current_pending is None or current_pending["marker"].call_id != pending["marker"].call_id:
            raise ValidationError("provider continuation changed before Host append")
        prior_flow = DataFlowContext.from_dict(pending["flow_context"])
        _validate_sources(pid, prior_flow, data_flow=data_flow, file_resource_resolver=file_resource_resolver)
        previous = data_flow.context_from_source_oids(pid, [oid], include_current=False)
        generation = processes.get_llm_context_generation(pid)
        yield
        if processes.get_llm_context_generation(pid) != generation:
            raise ValidationError("provider continuation Host append changed context generation")
        current = data_flow.context_from_source_oids(pid, [oid], include_current=False)
        flow = _advanced_source(prior_flow, previous.source_refs[0], current)
        if flow is not None:
            _validate_sources(pid, flow, data_flow=data_flow, file_resource_resolver=file_resource_resolver)
            result = _publish_continuation(
                pid, pending=pending, flow=flow, generation=generation, processes=processes,
                persist_full_io=persist_full_io,
                rebind_generation=False,
                provenance={"provider_continuation_context_update": {
                    "source_marker_call_id": pending["marker"].call_id,
                    "context_oid": oid, "source_version": previous.source_refs[0].version,
                    "context_version": current.source_refs[0].version,
                }},
            )
    if result is not None and not persist_full_io:
        marker_id, payload = result
        volatile_payloads[marker_id] = payload
        volatile_payloads.pop(pending["marker"].call_id, None)


def _advanced_source(
    flow: DataFlowContext, previous: DataSourceRef, current: DataFlowContext,
) -> DataFlowContext | None:
    retained = tuple(ref for ref in flow.source_refs if ref.oid == previous.oid)
    if not retained:
        return None
    if retained != (previous,):
        raise ValidationError("provider continuation context changed before Host append")
    if len(current.source_refs) != 1 or current.source_refs[0].oid != previous.oid:
        raise ValidationError("provider continuation Host append source is invalid")
    replacement = current.source_refs[0]
    if replacement == previous:
        return None
    if replacement.version != previous.version + 1:
        raise ValidationError("provider continuation Host append skipped a source version")
    return DataFlowContext.aggregate((
        DataFlowContext(labels=flow.labels, source_refs=tuple(
            ref for ref in flow.source_refs if ref.oid != previous.oid
        )), current,
    ))


@contextmanager
def provider_continuation_compaction_scope(
    pid: str, *, processes: Any, unit_of_work: Any,
    pending_data: Callable[[str], Mapping[str, Any] | None],
    context_memory: Any, data_flow: Any,
    file_resource_resolver: Callable[[str], str] | None,
    persist_full_io: bool,
) -> Iterator[None]:
    """Rebind only a locally certified compaction, atomically with its payload.

    Calls with no pending result retain the existing compaction transaction
    behavior. TaskRun and non-retained continuations must first consume their
    local action; neither can silently lose its resume contract to compaction.
    """

    pending = pending_data(pid)
    if pending is None:
        yield
        return
    process = processes.get_process(pid)
    if process is None or process.task_run_id is not None:
        raise ValidationError("pending TaskRun provider result must select a local action before compaction")
    if not persist_full_io:
        raise ValidationError("pending provider result requires retained payloads before compaction")
    prior_flow = DataFlowContext.from_dict(pending["flow_context"])
    with unit_of_work.transaction(include_object_payloads=True):
        current = pending_data(pid)
        if current is None or current["marker"].call_id != pending["marker"].call_id:
            raise ValidationError("provider continuation changed before compaction")
        old_generation = processes.get_llm_context_generation(pid)
        _validate_sources(
            pid, prior_flow, data_flow=data_flow,
            file_resource_resolver=file_resource_resolver,
        )
        yield
        certificate = context_memory.latest_validated_compaction(pid)
        generation = processes.get_llm_context_generation(pid)
        if (
            certificate is None or generation == old_generation
            or certificate.get("context_generation") != generation
        ):
            raise ValidationError("provider continuation compaction lacks a current certificate")
        _publish_compacted_continuation(
            pid, pending=pending, prior_flow=prior_flow, certificate=certificate,
            generation=generation, processes=processes, data_flow=data_flow,
            file_resource_resolver=file_resource_resolver,
        )


def _publish_compacted_continuation(
    pid: str, *, pending: Mapping[str, Any], prior_flow: DataFlowContext,
    certificate: Mapping[str, Any], generation: str, processes: Any, data_flow: Any,
    file_resource_resolver: Callable[[str], str] | None,
) -> None:
    context_oid = certificate["context_oid"]
    summary_flow = data_flow.context_from_source_oids(pid, [context_oid], include_current=False)
    summary_refs = [ref for ref in summary_flow.source_refs if ref.oid == context_oid]
    if len(summary_refs) != 1 or summary_refs[0].version != certificate["context_version"]:
        raise ValidationError("provider continuation compaction source changed after certification")
    if any(ref.oid == context_oid and ref.version > certificate["source_version"] for ref in prior_flow.source_refs):
        raise ValidationError("provider continuation compaction does not cover its retained source")
    flow = DataFlowContext.aggregate((
        DataFlowContext(
            labels=prior_flow.labels,
            source_refs=tuple(ref for ref in prior_flow.source_refs if ref.oid != context_oid),
        ),
        summary_flow,
        DataFlowContext(labels=DataLabels.from_dict(certificate["data_labels"])),
    ))
    _validate_sources(pid, flow, data_flow=data_flow, file_resource_resolver=file_resource_resolver)
    _publish_continuation(
        pid, pending=pending, flow=flow, generation=generation, processes=processes,
        persist_full_io=True,
        provenance={"provider_continuation_compaction": {
            "source_marker_call_id": pending["marker"].call_id,
            "source_context_generation": pending["manifest"]["context_generation"],
            "context_oid": context_oid, "summary_sha256": certificate["summary_sha256"],
        }},
    )


def _publish_continuation(
    pid: str, *, pending: Mapping[str, Any], flow: DataFlowContext,
    generation: str, processes: Any, persist_full_io: bool,
    provenance: Mapping[str, Any],
    rebind_generation: bool = True,
) -> tuple[str, dict[str, Any]]:
    payload = {"message": pending["message"], "flow_context": flow.to_dict()}
    marker = pending["marker"]
    manifest = {
        **pending["manifest"],
        "context_generation": generation,
        "payload_sha256": hashlib.sha256(dumps(to_jsonable(payload)).encode("utf-8")).hexdigest(),
    }
    if rebind_generation:
        manifest.update(schema_version=2, source_pid=pending["manifest"].get("source_pid", pid))
    now = utc_now()
    marker_id = new_id("llmcontinuation")
    processes.insert_llm_call(LLMCallRecord(
        call_id=marker_id, pid=pid, image_id=marker.image_id,
        purpose="provider_continuation", status="ok", tool_calls=[], tools=[],
        messages=payload["message"] if persist_full_io else None,
        raw_response=payload["flow_context"] if persist_full_io else None,
        request_options={
            **marker.request_options,
            "provider_continuation": manifest,
            **provenance,
        }, created_at=now, completed_at=now,
    ))
    return marker_id, payload
