"""Local, non-executable projections of completed provider-hosted work."""

from __future__ import annotations

import hashlib
from typing import Any

from agent_libos.models import DataFlowContext, LLMCallRecord
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import dumps, to_jsonable


def continuation_message(record: LLMCallRecord) -> dict[str, str]:
    """Derive assistant history from the retained, selected provider result.

    Only readable tool results are projected. Native provider items, encrypted
    reasoning and session handles never become executable replay inputs here.
    """
    from agent_libos.llm.provider_tools import provider_tool_result_text

    trace = record.reasoning
    detail: dict[str, Any] = {}
    if isinstance(trace, dict):
        selected = trace.get("selected_attempt")
        attempts = trace.get("attempts")
        if (
            type(selected) is int
            and isinstance(attempts, list)
            and 0 < selected <= len(attempts)
            and isinstance(attempts[selected - 1], dict)
        ):
            value = attempts[selected - 1].get("provider_tools")
            if isinstance(value, dict):
                detail = value
    text = provider_tool_result_text(
        record.response_content,
        detail.get("activities", []),
        detail.get("citations", []),
        detail.get("artifacts", []),
    )
    if not text.strip():
        raise ValueError("provider continuation result content is unavailable")
    return {"role": "assistant", "content": text}


def continuation_source_sha256(record: LLMCallRecord) -> str:
    """Bind a continuation to its successful local call and retained result."""
    return hashlib.sha256(dumps(to_jsonable({
        "call_id": record.call_id,
        "pid": record.pid,
        "image_id": record.image_id,
        "purpose": record.purpose,
        "status": record.status,
        "request_options": record.request_options,
        "response_content": record.response_content,
        "tool_calls": record.tool_calls,
        "reasoning": record.reasoning,
    })).encode("utf-8")).hexdigest()


def validated_continuation_marker(record: LLMCallRecord) -> dict[str, Any]:
    value = record.request_options.get("provider_continuation")
    fields = {
        "schema_version", "state", "call_id", "source_sha256",
        "profile_identity_sha256", "context_generation", "payload_sha256",
    }
    if isinstance(value, dict) and value.get("schema_version") == 2:
        fields.add("source_pid")
        if not isinstance(value.get("source_pid"), str) or not 0 < len(value["source_pid"]) <= 256:
            raise ValidationError("provider continuation source process is invalid")
    if (
        record.status != "ok" or not record.completed_at
        or not isinstance(value, dict)
        or set(value) != fields
        or type(value.get("schema_version")) is not int
        or value["schema_version"] not in {1, 2}
        or value.get("state") not in {"pending", "consumed"}
    ):
        raise ValidationError("provider continuation manifest is invalid")
    return value


def pending_continuation_marker(
    processes: Any,
    *,
    pid: str,
    marker: LLMCallRecord | None,
) -> dict[str, Any] | None:
    """Discard settled or obsolete markers before resolving provider state."""

    if marker is None:
        return None
    if marker.pid != pid or marker.purpose != "provider_continuation":
        raise ValidationError("provider continuation owner is invalid")
    manifest = validated_continuation_marker(marker)
    if manifest["state"] == "consumed":
        return None
    if manifest["context_generation"] != processes.get_llm_context_generation(pid):
        # A discarded checkpoint/context generation cannot retain authority.
        return None
    return manifest


def load_continuation_data(
    processes: Any,
    *,
    pid: str,
    marker: LLMCallRecord | None,
    profile_identity_sha256: str,
    volatile_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate retained result bytes before dispatch or startup READ retention."""

    manifest = pending_continuation_marker(processes, pid=pid, marker=marker)
    if manifest is None:
        return None
    assert marker is not None
    source = processes.get_llm_call(str(manifest.get("call_id") or ""))
    _validate_continuation_source(source, marker=marker, manifest=manifest, pid=pid)
    if manifest["profile_identity_sha256"] != profile_identity_sha256:
        raise ValidationError("provider continuation profile changed; start a new task")
    payload = volatile_payload if volatile_payload is not None else {
        "message": marker.messages, "flow_context": marker.raw_response,
    }
    if hashlib.sha256(dumps(to_jsonable(payload)).encode("utf-8")).hexdigest() != manifest["payload_sha256"]:
        raise ValidationError("provider continuation content is unavailable under the retention policy")
    message = payload.get("message")
    if (
        not isinstance(message, dict) or set(message) != {"role", "content"}
        or message["role"] != "assistant" or not isinstance(message["content"], str)
        or not message["content"].strip()
    ):
        raise ValidationError("provider continuation transcript is invalid")
    DataFlowContext.from_dict(payload["flow_context"])
    return {"marker": marker, "manifest": manifest, **payload}


def _validate_continuation_source(
    source: LLMCallRecord | None,
    *,
    marker: LLMCallRecord,
    manifest: dict[str, Any],
    pid: str,
) -> None:
    if (
        source is None or source.pid != manifest.get("source_pid", pid)
        or source.status != "ok" or not source.completed_at
        or source.image_id != marker.image_id
        or source.purpose != "action_selection"
        or source.request_options.get("provider_tools_enabled") is not True
        or (source.tool_calls != [] and source.request_options.get("provider_tools_function_call_count") != 0)
        or continuation_source_sha256(source) != manifest["source_sha256"]
    ):
        raise ValidationError("provider continuation source is unavailable or changed")


def has_provider_continuation_result(completion: Any) -> bool:
    return bool(
        not completion.tool_calls
        and (
            completion.content.strip()
            or getattr(completion, "provider_tool_activities", [])
            or getattr(completion, "citations", [])
            or getattr(completion, "artifacts", [])
        )
    )


def is_provider_continuation_completion(record: LLMCallRecord | None, completion: Any) -> bool:
    return bool(
        record is not None
        and record.request_options.get("provider_tools_enabled") is True
        and has_provider_continuation_result(completion)
    )


def provider_repair_messages(record: LLMCallRecord | None, completion: Any) -> list[dict[str, str]]:
    if record is None or not record.request_options.get("provider_tools_configured"):
        return []
    from agent_libos.llm.provider_tools import provider_tool_result_text

    return [{"role": "assistant", "content": provider_tool_result_text(
        completion.content,
        getattr(completion, "provider_tool_activities", []),
        getattr(completion, "citations", []),
        getattr(completion, "artifacts", []),
    )}]
