from __future__ import annotations

import json
from copy import deepcopy
from math import ceil
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.context_memory import context_object_name
from agent_libos.llm.tool_protocol import tool_call_to_action
from agent_libos.models import ObjectMetadata, ObjectPatch, ObjectRight, ObjectType, ProcessStatus
from agent_libos.models.exceptions import NotFound, ProcessWaitRequired, ValidationError as LibOSValidationError
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.utils.ids import estimate_tokens, utc_now

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools

CONTEXT_COMPRESSOR_IMAGE_ID = "context-compressor:v0"
_JOB_NAME_PREFIX = "context_compaction_job"
_SUMMARY_FIELDS = {
    "goal",
    "constraints",
    "user_preferences",
    "completed",
    "pending",
    "key_references",
    "recent_decisions",
    "risks",
    "uncertainties",
    "next_steps",
}


class _ChildSummaryUnavailable(Exception):
    pass


class CompactProcessContextArgs(BaseModel):
    target_tokens: int = Field(default=4000, ge=256, le=64_000)
    preserve_recent_entries: int = Field(default=8, ge=0, le=128)
    max_chunks: int = Field(default=8, ge=1, le=64)
    force: bool = Field(default=False)
    resume_job: dict[str, Any] | None = Field(
        default=None,
        alias="_resume_job",
        exclude=True,
        json_schema_extra={"x-agent-libos-internal": True},
    )

    model_config = ConfigDict(populate_by_name=True)


class CompactProcessContextOutput(BaseModel):
    compacted: bool
    context_oid: str | None = None
    old_version: int | None = None
    new_version: int | None = None
    compressor_pids: list[str] = Field(default_factory=list)
    source_tokens: int = 0
    compacted_tokens: int = 0
    preserved_recent_entries: int = 0
    reason: str


class CompactProcessContextTool(SyncAgentTool[CompactProcessContextArgs]):
    name = "compact_process_context"
    description = (
        "Compress this AgentProcess LLM context object by delegating summarization to "
        "a context-compressor child image, then replace the current llm_context object."
    )
    args_schema = CompactProcessContextArgs
    output_schema = CompactProcessContextOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.read", "object.write", "process.spawn", "process.wait"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["context", "llm_context", "compaction", "process"]

    def run(self, args: CompactProcessContextArgs, ctx: ToolContext) -> CompactProcessContextOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        resume_job = _normalize_resume_job(args.resume_job)
        if resume_job is not None:
            _require_authorized_resume_job(runtime, ctx.pid, resume_job, ctx.metadata)
        context = _load_context(runtime, ctx.pid, required=resume_job is None)
        if context is not None:
            rendered = runtime.llm.context_memory.render(context.payload)
            source_tokens = estimate_tokens(rendered)
        elif resume_job is not None:
            source_tokens = int(resume_job["source_tokens"])
        else:
            raise ToolExecutionError("Current LLM context is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        job_obj, job_handle = _load_job(runtime, ctx.pid)
        job = resume_job or (dict(job_obj.payload) if job_obj is not None else {})
        if not job or job.get("status") != "active":
            if context is None:
                raise ToolExecutionError(
                    "Cannot start context compaction without a live LLM context.",
                    code=ToolErrorCode.EXECUTION_ERROR,
                )
            if not args.force and source_tokens <= args.target_tokens:
                return CompactProcessContextOutput(
                    compacted=False,
                    context_oid=context.oid,
                    old_version=context.version,
                    new_version=context.version,
                    source_tokens=source_tokens,
                    compacted_tokens=source_tokens,
                    reason="context_under_target",
                )
            job = _new_job_payload(ctx.pid, args, context.oid, context.version, context.payload, source_tokens)
            if job_handle is None:
                job_obj, job_handle = _create_job(runtime, ctx.pid, job)
            else:
                _update_job(runtime, ctx.pid, job_handle, job)
        elif job_handle is not None:
            _update_job(runtime, ctx.pid, job_handle, job)

        try:
            _assert_source_unchanged(context, job)
        except ToolExecutionError as exc:
            _fail_job(runtime, ctx.pid, job_handle, job, str(exc))
            raise
        chunks = _entry_chunks(job["source_payload"], int(job["max_chunks"]))
        if not chunks:
            chunks = [[]]

        while int(job.get("stage_index", 0)) < len(chunks):
            child_pid = str(job.get("current_child_pid") or "")
            if not child_pid:
                try:
                    child_pid = _spawn_stage(runtime, ctx.pid, job, chunks)
                except Exception as exc:
                    _fail_job(runtime, ctx.pid, job_handle, job, str(exc))
                    raise
                job["current_child_pid"] = child_pid
                job.setdefault("compressor_pids", []).append(child_pid)
                job["updated_at"] = utc_now()
                _update_job(runtime, ctx.pid, job_handle, job)
            try:
                wait_result = runtime.process.wait(ctx.pid, child_pid, timeout=None)
            except ProcessWaitRequired as exc:
                raise ProcessWaitRequired(
                    child_pid=exc.child_pid,
                    message=str(exc),
                    resume_action=_resume_action(job),
                ) from exc
            if wait_result.status != ProcessStatus.EXITED or wait_result.result is None:
                reason = (
                    "context compressor child ended without a result: "
                    f"{child_pid} status={wait_result.status.value}"
                )
                _fail_job(runtime, ctx.pid, job_handle, job, reason)
                raise ToolExecutionError(reason, code=ToolErrorCode.EXECUTION_ERROR)
            try:
                summary = _read_child_summary(runtime, ctx.pid, wait_result.result, child_pid=child_pid)
            except _ChildSummaryUnavailable:
                job["current_child_pid"] = None
                job.setdefault("discarded_compressor_pids", []).append(child_pid)
                job["updated_at"] = utc_now()
                _update_job(runtime, ctx.pid, job_handle, job)
                continue
            _record_stage_summary(runtime, ctx.pid, job_handle, job, summary)

        compact_summary = _latest_stage_summary(job.get("summaries") or [])
        try:
            result = runtime.llm.context_memory.replace_with_compacted_summary(
                ctx.pid,
                context_oid=str(job["context_oid"]),
                expected_version=int(job["source_version"]),
                summary=compact_summary,
                compaction_method="agent_image_child",
                compaction_metadata={
                    "tool_name": self.name,
                    "compressor_image_id": CONTEXT_COMPRESSOR_IMAGE_ID,
                    "stage_count": int(job.get("stage_index", 0)),
                    "chunk_count": len(chunks),
                    "requested_max_chunks": int(job["max_chunks"]),
                    "discarded_compressor_pids": [str(pid) for pid in job.get("discarded_compressor_pids", [])],
                },
                preserve_recent_entries=int(job["preserve_recent_entries"]),
                source_tokens=int(job["source_tokens"]),
                target_tokens=int(job["target_tokens"]),
                compressor_pids=[str(pid) for pid in job.get("compressor_pids", [])],
                source_payload=job.get("source_payload"),
            )
        except LibOSValidationError as exc:
            _fail_job(runtime, ctx.pid, job_handle, job, str(exc))
            raise ToolExecutionError(str(exc), code=ToolErrorCode.VALIDATION_ERROR) from exc
        job["status"] = "completed"
        job["completed_at"] = utc_now()
        job["result"] = dict(result)
        _update_job(runtime, ctx.pid, job_handle, job)
        return CompactProcessContextOutput(
            compacted=True,
            context_oid=str(result["context_oid"]),
            old_version=int(result["old_version"]),
            new_version=int(result["new_version"]),
            compressor_pids=[str(pid) for pid in job.get("compressor_pids", [])],
            source_tokens=int(result["source_tokens"]),
            compacted_tokens=int(result["compacted_tokens"]),
            preserved_recent_entries=int(result["preserved_recent_entries"]),
            reason="compacted",
        )


def _load_context(runtime: Any, pid: str, *, required: bool = True) -> Any | None:
    name = context_object_name(pid, config=runtime.config)
    try:
        handle = runtime.memory.handle_for_name(
            pid,
            name,
            rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.MATERIALIZE.value},
            issued_by="compact_process_context",
        )
    except NotFound:
        if not required:
            return None
        raise
    obj = runtime.memory.get_object(pid, handle)
    if not isinstance(obj.payload, dict) or obj.payload.get("kind") != "llm_context":
        raise ToolExecutionError("Current context object is not an LLM context.", code=ToolErrorCode.VALIDATION_ERROR)
    return obj


def _load_job(runtime: Any, pid: str) -> tuple[Any | None, Any | None]:
    name = _job_name(pid)
    try:
        handle = runtime.memory.handle_for_name(
            pid,
            name,
            rights={ObjectRight.READ.value, ObjectRight.WRITE.value},
            issued_by="compact_process_context.job",
        )
        return runtime.memory.get_object(pid, handle), handle
    except NotFound:
        return None, None


def _create_job(runtime: Any, pid: str, payload: dict[str, Any]) -> tuple[Any, Any]:
    handle = runtime.memory.create_object(
        pid=pid,
        object_type=ObjectType.PROCESS_STATE,
        payload=payload,
        metadata=ObjectMetadata(title=f"Context compaction job for {pid}", tags=["context_compaction"]),
        immutable=False,
        name=_job_name(pid),
    )
    return runtime.memory.get_object(pid, handle), handle


def _new_job_payload(
    pid: str,
    args: CompactProcessContextArgs,
    context_oid: str,
    context_version: int,
    context_payload: dict[str, Any],
    source_tokens: int,
) -> dict[str, Any]:
    return {
        "kind": "context_compaction_job",
        "schema_version": 1,
        "status": "active",
        "caller_pid": pid,
        "context_oid": context_oid,
        "source_version": context_version,
        "source_payload": deepcopy(context_payload),
        "source_tokens": source_tokens,
        "target_tokens": args.target_tokens,
        "preserve_recent_entries": args.preserve_recent_entries,
        "max_chunks": args.max_chunks,
        "force": args.force,
        "stage_index": 0,
        "current_child_pid": None,
        "compressor_pids": [],
        "summaries": [],
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


def _update_job(runtime: Any, pid: str, handle: Any, payload: dict[str, Any]) -> None:
    if handle is None:
        return
    runtime.memory.update_object(
        pid,
        handle,
        ObjectPatch(
            payload=payload,
            metadata=ObjectMetadata(
                title=f"Context compaction job for {pid}",
                tags=["context_compaction", str(payload.get("status") or "unknown")],
                token_estimate=estimate_tokens(payload),
            ),
        ),
    )


def _fail_job(runtime: Any, pid: str, handle: Any, payload: dict[str, Any], reason: str) -> None:
    payload["status"] = "failed"
    payload["error"] = reason
    payload["updated_at"] = utc_now()
    _update_job(runtime, pid, handle, payload)


def _assert_source_unchanged(context: Any | None, job: dict[str, Any]) -> None:
    if context is None:
        return
    if context.oid != job.get("context_oid"):
        raise ToolExecutionError(
            "Active compaction job targets a different context object.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    if context.version != int(job.get("source_version") or 0):
        raise ToolExecutionError(
            "LLM context changed during compaction.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"expected_version": job.get("source_version"), "actual_version": context.version},
        )


def _entry_chunks(payload: dict[str, Any], max_chunks: int) -> list[list[dict[str, Any]]]:
    entries = [entry for entry in payload.get("entries", []) if isinstance(entry, dict)]
    if not entries:
        return []
    chunk_count = max(1, min(max_chunks, len(entries)))
    chunk_size = max(1, ceil(len(entries) / chunk_count))
    return [entries[index : index + chunk_size] for index in range(0, len(entries), chunk_size)]


def _stage_goal(pid: str, job: dict[str, Any], chunks: list[list[dict[str, Any]]], index: int) -> dict[str, Any]:
    previous_summary = _latest_stage_summary(job.get("summaries") or []) if job.get("summaries") else {}
    return {
        "kind": "context_compaction_stage",
        "caller_pid": pid,
        "context_oid": job["context_oid"],
        "source_version": job["source_version"],
        "stage_index": index + 1,
        "total_stages": len(chunks),
        "target_tokens": job["target_tokens"],
        "previous_summary": previous_summary,
        "entries": chunks[index],
        "instructions": (
            "Summarize only the supplied entries and previous_summary. Preserve exact ids, "
            "tool names, process ids, object ids, constraints, open tasks, risks, and next steps."
        ),
    }


def _spawn_stage(runtime: Any, pid: str, job: dict[str, Any], chunks: list[list[dict[str, Any]]]) -> str:
    index = int(job.get("stage_index", 0))
    goal = _stage_goal(pid, job, chunks, index)
    return runtime.spawn_child_process(
        parent=pid,
        goal=goal,
        image=CONTEXT_COMPRESSOR_IMAGE_ID,
    )


def _read_child_summary(runtime: Any, pid: str, handle: Any, *, child_pid: str) -> dict[str, Any]:
    try:
        obj = runtime.memory.get_object(pid, handle)
        payload = obj.payload
    except NotFound:
        payload = _read_child_summary_from_llm_calls(runtime, child_pid)
    if not isinstance(payload, dict):
        raise ToolExecutionError(
            "Context compressor returned a non-object payload.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    missing = sorted(_SUMMARY_FIELDS - set(payload))
    if missing:
        raise ToolExecutionError(
            "Context compressor output is missing required fields.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"missing": missing},
        )
    if not any(payload.get(key) for key in _SUMMARY_FIELDS):
        raise ToolExecutionError("Context compressor returned an empty summary.", code=ToolErrorCode.VALIDATION_ERROR)
    return {key: payload.get(key) for key in sorted(_SUMMARY_FIELDS)}


def _record_stage_summary(
    runtime: Any,
    pid: str,
    job_handle: Any,
    job: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    # Each stage is cumulative. Retaining and re-merging every intermediate
    # would duplicate earlier content and grow quadratically with chunk count.
    job["summaries"] = [summary]
    job["stage_index"] = int(job.get("stage_index", 0)) + 1
    job["current_child_pid"] = None
    job["updated_at"] = utc_now()
    _update_job(runtime, pid, job_handle, job)


def _read_child_summary_from_llm_calls(runtime: Any, child_pid: str) -> dict[str, Any]:
    calls = runtime.store.list_llm_calls(pid=child_pid, limit=1000)
    for call in reversed(calls):
        tool_calls = call.tool_calls
        if isinstance(tool_calls, dict) and isinstance(tool_calls.get("preview"), str):
            try:
                tool_calls = json.loads(tool_calls["preview"])
            except Exception:
                continue
        if not isinstance(tool_calls, list):
            continue
        for tool_call in reversed(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            try:
                action = tool_call_to_action(tool_call)
            except Exception:
                continue
            if action.get("action") != "process_exit":
                continue
            payload = action.get("payload")
            if isinstance(payload, dict):
                return payload
    raise _ChildSummaryUnavailable(
        "Context compressor result object is unavailable and no durable process_exit payload was found."
    )


def _latest_stage_summary(summaries: list[Any]) -> dict[str, Any]:
    """Return the latest cumulative summary, including from legacy jobs."""
    if not summaries:
        return {key: [] for key in sorted(_SUMMARY_FIELDS)}
    latest = next(
        (summary for summary in reversed(summaries) if isinstance(summary, dict)),
        None,
    )
    if latest is None:
        return {key: [] for key in sorted(_SUMMARY_FIELDS)}
    return {key: latest.get(key, []) for key in sorted(_SUMMARY_FIELDS)}


def _job_name(pid: str) -> str:
    return f"{_JOB_NAME_PREFIX}:{pid}"


def _resume_action(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "compact_process_context",
        "target_tokens": int(job["target_tokens"]),
        "preserve_recent_entries": int(job["preserve_recent_entries"]),
        "max_chunks": int(job["max_chunks"]),
        "force": bool(job.get("force")),
        "_resume_job": deepcopy(job),
    }


def _normalize_resume_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if job is None:
        return None
    if not isinstance(job, dict):
        raise ToolExecutionError("Invalid context compaction resume state.", code=ToolErrorCode.VALIDATION_ERROR)
    selected = deepcopy(job)
    required = {
        "kind",
        "schema_version",
        "status",
        "caller_pid",
        "context_oid",
        "source_version",
        "source_payload",
        "source_tokens",
        "target_tokens",
        "preserve_recent_entries",
        "max_chunks",
        "stage_index",
        "compressor_pids",
        "summaries",
    }
    missing = sorted(required - set(selected))
    if missing:
        raise ToolExecutionError(
            "Invalid context compaction resume state.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"missing": missing},
        )
    if selected.get("kind") != "context_compaction_job" or selected.get("status") != "active":
        raise ToolExecutionError("Invalid context compaction resume state.", code=ToolErrorCode.VALIDATION_ERROR)
    if not isinstance(selected.get("source_payload"), dict) or selected["source_payload"].get("kind") != "llm_context":
        raise ToolExecutionError(
            "Resume state source payload is not an LLM context.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    return selected


def _require_authorized_resume_job(
    runtime: Any,
    pid: str,
    job: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    if metadata.get("pending_child_resume") is not True:
        raise ToolExecutionError(
            "Internal context compaction resume state is not callable directly.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    pending_child_pid = str(metadata.get("pending_child_pid") or "")
    current_child_pid = str(job.get("current_child_pid") or "")
    if not pending_child_pid or pending_child_pid != current_child_pid:
        raise ToolExecutionError(
            "Context compaction resume state does not match the pending child.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    if str(job.get("caller_pid") or "") != pid:
        raise ToolExecutionError(
            "Context compaction resume state belongs to a different process.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    pending = runtime.store.get_llm_pending_action(pid)
    if not isinstance(pending, dict) or str(pending.get("status") or "") not in {"pending", "resuming"}:
        raise ToolExecutionError(
            "Context compaction resume state has no durable pending action.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    if str(pending.get("wait_type") or "") != "child" or str(pending.get("child_pid") or "") != pending_child_pid:
        raise ToolExecutionError(
            "Context compaction resume state does not match the durable child wait.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    action = dict(pending.get("action") or {})
    if action.get("action") != "compact_process_context":
        raise ToolExecutionError(
            "Durable pending action is not a context compaction resume.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    pending_job = _normalize_resume_job(action.get("_resume_job"))
    if pending_job != job:
        raise ToolExecutionError(
            "Context compaction resume state does not match the durable pending action.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )


def restore_pending_compaction_child_goal(
    pending_action: dict[str, Any],
    *,
    processes: Any,
    objects: Any,
    memory: Any,
) -> None:
    action = dict(pending_action.get("action") or {})
    if action.get("action") != "compact_process_context":
        return
    job = _normalize_resume_job(action.get("_resume_job"))
    if job is None:
        return
    child_pid = str(pending_action.get("child_pid") or job.get("current_child_pid") or "")
    if not child_pid:
        return
    child = processes.get_process(child_pid)
    if child is None or child.status in {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}:
        return
    if child.goal_oid and objects.get_object(child.goal_oid) is not None:
        return
    chunks = _entry_chunks(job["source_payload"], int(job["max_chunks"])) or [[]]
    index = int(job.get("stage_index", 0))
    if index >= len(chunks):
        return
    goal = _stage_goal(str(job["caller_pid"]), job, chunks, index)
    handle = memory.create_object(
        pid=child_pid,
        object_type=ObjectType.GOAL,
        payload=goal,
        metadata=ObjectMetadata(
            title=f"Restored context compaction stage for {child_pid}",
            tags=["context_compaction", "goal"],
        ),
        immutable=False,
        name=f"context_compaction_stage:{child_pid}",
    )
    child.goal_oid = handle.oid
    child.memory_view = memory.create_view(child_pid, [handle])
    child.updated_at = utc_now()
    processes.patch_process(
        child_pid,
        {
            "goal_oid": child.goal_oid,
            "memory_view": child.memory_view,
            "updated_at": child.updated_at,
        },
        expected_revision=child.revision,
    )
