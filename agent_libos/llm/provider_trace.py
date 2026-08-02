from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice
from typing import Any, Literal

PROVIDER_TRACE_SCHEMA_VERSION = 1
PROVIDER_TRACE_MAX_ATTEMPTS = 256
PROVIDER_TRACE_MAX_BYTES = 4 * 1024 * 1024
PROVIDER_TRACE_MAX_DEPTH = 32
PROVIDER_TRACE_MAX_NODES = 4_096
PROVIDER_TRACE_TEXT_MAX_CHARS = 262_144
PROVIDER_TRACE_TEXT_MAX_BYTES = 1_048_576
PROVIDER_TRACE_SAFE_INTEGER_MAX = (1 << 53) - 1

ProviderAttemptKind = Literal[
    "initial",
    "transport_retry",
    "compatibility_retry",
    "responses_to_chat",
    "json_action_fallback",
    "non_thinking_retry",
]

_TRACE_ERROR_ATTR = "_agent_libos_provider_trace_v1"
_OPAQUE_KEY_FRAGMENTS = (
    "encrypted",
    "encryption",
    "cipher",
    "signature",
    "signed",
    "blob",
    "opaque",
)
_CREDENTIAL_KEY_MARKERS = (
    "api_key",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
)
_CREDENTIAL_TOKEN_KEYS = frozenset(
    {
        "access_token",
        "accesstoken",
        "apikey",
        "api_token",
        "apitoken",
        "auth_token",
        "authtoken",
        "id_token",
        "idtoken",
        "refresh_token",
        "refreshtoken",
        "session_token",
        "sessiontoken",
        "token",
    }
)
_CREDENTIAL_COOKIE_KEYS = frozenset({"cookie", "cookies", "set_cookie", "setcookie"})


class ProviderTraceAttemptLimitExceeded(RuntimeError):
    """The Host trace bound stopped a physical Provider dispatch."""


@dataclass(slots=True)
class _AttemptState:
    value: dict[str, Any]
    monotonic_started: float


@dataclass(slots=True)
class _ReasoningProjectionState:
    blocks: list[dict[str, Any]] = field(default_factory=list)
    nodes: int = 0
    limited: bool = False
    text_bytes: int = 0


@dataclass(slots=True)
class _RawProjectionState:
    nodes: int = 0
    limited: bool = False
    text_bytes: int = 0


@dataclass(slots=True)
class ProviderTraceBuilder:
    """Collect one bounded, terminal-only logical Provider call trace."""

    coverage: str = "complete"
    attempts: list[_AttemptState] = field(default_factory=list)
    selected_attempt: int | None = None
    limited: bool = False
    omitted_attempts: int = 0

    def start_attempt(
        self,
        *,
        api: Literal["responses", "chat", "custom"],
        kind: ProviderAttemptKind,
    ) -> int:
        if len(self.attempts) >= PROVIDER_TRACE_MAX_ATTEMPTS:
            self.limited = True
            raise ProviderTraceAttemptLimitExceeded(
                "provider attempt trace limit reached before dispatch"
            )
        sequence = len(self.attempts) + 1
        now = _utc_now()
        self.attempts.append(
            _AttemptState(
                value={
                    "sequence": sequence,
                    "kind": kind,
                    "api": api,
                    "status": "error",
                    "reasoning": _empty_reasoning(),
                    "output": "",
                    "tool_calls": [],
                    "usage": {},
                    "model": None,
                    "request_id": None,
                    "response_id": None,
                    "started_at": now,
                    "completed_at": now,
                    "duration_ms": 0,
                    "error": None,
                },
                monotonic_started=time.monotonic(),
            )
        )
        return sequence

    def finish_response(self, sequence: int, response: Any) -> None:
        attempt = self._attempt(sequence)
        attempt["status"] = "ok"
        attempt["completed_at"] = _utc_now()
        attempt["duration_ms"] = self._duration_ms(sequence)
        attempt["request_id"] = _bounded_identifier(
            _get_attr_or_key(response, "_request_id")
        )
        attempt["response_id"] = _bounded_identifier(
            _get_attr_or_key(response, "id")
        )
        attempt["model"] = _bounded_identifier(_get_attr_or_key(response, "model"))
        attempt["usage"] = project_provider_usage(_get_attr_or_key(response, "usage"))
        attempt["error"] = None

    def finish_error(self, sequence: int, error: BaseException) -> None:
        self.reject_response(sequence, error)
        attempt = self._attempt(sequence)
        attempt["usage"] = {}
        attempt["model"] = None
        attempt["request_id"] = None
        attempt["response_id"] = None

    def reject_response(self, sequence: int, error: BaseException) -> None:
        """Reject a received response while retaining safe diagnostic metadata."""

        attempt = self._attempt(sequence)
        attempt["status"] = "error"
        attempt["completed_at"] = _utc_now()
        attempt["duration_ms"] = self._duration_ms(sequence)
        attempt["reasoning"] = _empty_reasoning()
        attempt["output"] = ""
        attempt.pop("output_limited", None)
        attempt["tool_calls"] = []
        attempt["error"] = safe_provider_error(error)

    def enrich_response(
        self,
        sequence: int,
        *,
        reasoning: Any,
        output: Any,
        tool_calls: Any,
        usage: Any,
        model: Any = None,
        request_id: Any = None,
        response_id: Any = None,
    ) -> None:
        attempt = self._attempt(sequence)
        attempt["reasoning"] = provider_reasoning_view(reasoning)
        selected_output, limited_output = _bounded_readable_text(output)
        attempt["output"] = selected_output
        if limited_output is None:
            attempt.pop("output_limited", None)
        else:
            attempt["output_limited"] = limited_output
            self.limited = True
        attempt["tool_calls"] = project_provider_tool_calls(tool_calls)
        attempt["usage"] = project_provider_usage(usage)
        attempt["model"] = _bounded_identifier(model) or attempt.get("model")
        attempt["request_id"] = _bounded_identifier(request_id) or attempt.get(
            "request_id"
        )
        attempt["response_id"] = _bounded_identifier(response_id) or attempt.get(
            "response_id"
        )
        if attempt["reasoning"].get("availability") == "limited":
            self.limited = True

    def mark_selected(self, sequence: int | None) -> None:
        if sequence is None:
            self.selected_attempt = None
            return
        self._attempt(sequence)
        self.selected_attempt = sequence

    def to_dict(self) -> dict[str, Any]:
        trace = {
            "kind": "provider_trace",
            "schema_version": PROVIDER_TRACE_SCHEMA_VERSION,
            "coverage": self.coverage,
            "selected_attempt": self.selected_attempt,
            "limited": self.limited,
            "omitted_attempts": self.omitted_attempts,
            "attempts": [copy.deepcopy(item.value) for item in self.attempts],
        }
        return _fit_trace_aggregate(trace)

    def _attempt(self, sequence: int) -> dict[str, Any]:
        if sequence <= 0 or sequence > len(self.attempts):
            raise ValueError("provider attempt sequence is out of range")
        return self.attempts[sequence - 1].value

    def _duration_ms(self, sequence: int) -> int:
        state = self.attempts[sequence - 1]
        return max(0, int(round((time.monotonic() - state.monotonic_started) * 1000)))


def provider_trace_summary(trace: Any) -> dict[str, Any]:
    if not is_provider_trace(trace):
        return {
            "schema_version": PROVIDER_TRACE_SCHEMA_VERSION,
            "coverage": "legacy_final_only",
            "attempt_count": 0,
            "recorded_attempt_count": 0,
            "selected_attempt": None,
            "status_counts": {"ok": 0, "error": 0},
            "limited": False,
            "omitted_attempts": 0,
        }
    attempts = trace.get("attempts")
    selected_attempts = attempts if isinstance(attempts, list) else []
    ok_count = sum(
        1
        for attempt in selected_attempts
        if isinstance(attempt, dict) and attempt.get("status") == "ok"
    )
    error_count = sum(
        1
        for attempt in selected_attempts
        if isinstance(attempt, dict) and attempt.get("status") == "error"
    )
    omitted = trace.get("omitted_attempts")
    selected_omitted = omitted if type(omitted) is int and omitted >= 0 else 0
    return {
        "schema_version": PROVIDER_TRACE_SCHEMA_VERSION,
        "coverage": trace.get("coverage"),
        "attempt_count": len(selected_attempts) + selected_omitted,
        "recorded_attempt_count": len(selected_attempts),
        "selected_attempt": trace.get("selected_attempt"),
        "status_counts": {"ok": ok_count, "error": error_count},
        "limited": bool(trace.get("limited")),
        "omitted_attempts": selected_omitted,
    }


def is_provider_trace(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("kind") == "provider_trace"
        and value.get("schema_version") == PROVIDER_TRACE_SCHEMA_VERSION
        and isinstance(value.get("attempts"), list)
    )


def attach_provider_trace(error: BaseException, trace: dict[str, Any]) -> None:
    if not is_provider_trace(trace):
        return
    try:
        object.__setattr__(error, _TRACE_ERROR_ATTR, trace)
    except BaseException:
        return


def provider_trace_from_error(error: BaseException) -> dict[str, Any] | None:
    selected: BaseException | None = error
    seen: set[int] = set()
    while selected is not None and id(selected) not in seen:
        seen.add(id(selected))
        try:
            trace = object.__getattribute__(selected, _TRACE_ERROR_ATTR)
        except BaseException:
            trace = None
        if is_provider_trace(trace):
            return trace
        cause = getattr(selected, "__cause__", None)
        selected = cause if isinstance(cause, BaseException) else None
    return None


def custom_provider_trace(
    completion: Any | None = None,
    *,
    error: BaseException | None = None,
) -> dict[str, Any]:
    builder = ProviderTraceBuilder(coverage="custom_client_incomplete")
    if completion is None:
        return builder.to_dict()
    sequence = builder.start_attempt(api="custom", kind="initial")
    builder.finish_response(sequence, completion)
    builder.enrich_response(
        sequence,
        reasoning=getattr(completion, "reasoning", None),
        output=getattr(completion, "content", ""),
        tool_calls=getattr(completion, "tool_calls", []) or [],
        usage=getattr(completion, "usage", {}) or {},
        model=getattr(completion, "model", None),
        request_id=getattr(completion, "request_id", None),
        response_id=getattr(completion, "response_id", None),
    )
    if error is not None:
        builder.finish_error(sequence, error)
    else:
        builder.mark_selected(sequence)
    return builder.to_dict()


def provider_reasoning_view(value: Any) -> dict[str, Any]:
    state = _ReasoningProjectionState()
    _visit_reasoning_value(value, source="reasoning", depth=0, state=state)
    if state.limited:
        availability = "limited"
    elif state.blocks:
        availability = "returned"
    else:
        availability = "not_returned"
    return {"availability": availability, "blocks": state.blocks}


def _visit_reasoning_value(
    item: Any,
    *,
    source: str,
    depth: int,
    state: _ReasoningProjectionState,
) -> None:
    if item is None or (isinstance(item, str) and item == ""):
        return
    state.nodes += 1
    if depth > PROVIDER_TRACE_MAX_DEPTH or state.nodes > PROVIDER_TRACE_MAX_NODES:
        state.limited = True
        state.blocks.append(_omitted_value(item, reason="structure_limit"))
        return
    if isinstance(item, str):
        _append_reasoning_text(item, source=source, state=state)
        return
    if isinstance(item, (bytes, bytearray)):
        state.blocks.append(_opaque_block(item, source=source))
        return
    if isinstance(item, (list, tuple)):
        _visit_reasoning_sequence(item, source=source, depth=depth, state=state)
        return
    if isinstance(item, dict):
        _visit_reasoning_mapping(item, source=source, depth=depth, state=state)
        return
    object_view = _safe_provider_object_view(
        item,
        max_items=max(1, PROVIDER_TRACE_MAX_NODES - state.nodes),
    )
    if object_view is not item:
        _visit_reasoning_value(
            object_view,
            source=source,
            depth=depth + 1,
            state=state,
        )


def _append_reasoning_text(
    text: str,
    *,
    source: str,
    state: _ReasoningProjectionState,
    block_type: str | None = None,
) -> None:
    selected, omitted = _bounded_readable_text(text)
    selected_bytes = len(selected.encode("utf-8")) if selected else 0
    if omitted is None and state.text_bytes + selected_bytes <= PROVIDER_TRACE_MAX_BYTES:
        state.text_bytes += selected_bytes
        state.blocks.append(
            {
                "type": block_type or (
                    "summary_text" if "summary" in source else "reasoning_text"
                ),
                "text": selected,
                "source": source[:128],
            }
        )
        return
    state.blocks.append(
        {
            "type": "omitted",
            "reason": "aggregate_limit" if omitted is None else "bounds",
            **(omitted or _text_digest(text)),
        }
    )
    state.limited = True


def _visit_reasoning_sequence(
    items: list[Any] | tuple[Any, ...],
    *,
    source: str,
    depth: int,
    state: _ReasoningProjectionState,
) -> None:
    for index, child in enumerate(items):
        if state.nodes >= PROVIDER_TRACE_MAX_NODES:
            state.blocks.append(
                _omitted_sequence_tail(items, index, reason="node_limit")
            )
            state.limited = True
            return
        _visit_reasoning_value(
            child,
            source=source,
            depth=depth + 1,
            state=state,
        )


def _visit_reasoning_mapping(
    item: dict[Any, Any],
    *,
    source: str,
    depth: int,
    state: _ReasoningProjectionState,
) -> None:
    if item.get("_provider_projection_limited") is True:
        state.limited = True
    if _append_reasoning_descriptor(item, source=source, state=state):
        return
    availability = item.get("availability")
    children = item.get("blocks")
    if availability in {"returned", "not_returned", "limited"} and isinstance(
        children, list
    ):
        _visit_reasoning_sequence(children, source=source, depth=depth, state=state)
        state.limited = state.limited or availability == "limited"
        return
    _visit_reasoning_fields(item, source=source, depth=depth, state=state)


def _append_reasoning_descriptor(
    item: dict[Any, Any],
    *,
    source: str,
    state: _ReasoningProjectionState,
) -> bool:
    item_type = item.get("type")
    if item_type == "omitted":
        descriptor = _safe_omitted_descriptor(item)
        state.blocks.append(
            {
                "type": "omitted",
                "reason": _safe_key_text(item.get("reason") or "bounds")[:64],
                **descriptor,
            }
        )
        state.limited = True
        return True
    if item_type in {"summary_text", "reasoning_text"} and isinstance(
        item.get("text"), str
    ):
        _append_reasoning_text(
            item["text"],
            source=_safe_key_text(item.get("source") or source),
            block_type=item_type,
            state=state,
        )
        return True
    if item_type == "opaque" and type(item.get("bytes")) is int:
        state.blocks.append(_existing_opaque_block(item, source=source))
        return True
    return False


def _visit_reasoning_fields(
    item: dict[Any, Any],
    *,
    source: str,
    depth: int,
    state: _ReasoningProjectionState,
) -> None:
    readable_keys = {
        "summary",
        "content",
        "text",
        "reasoning",
        "reasoning_content",
        "thinking",
        "thinking_content",
    }
    for key, child in item.items():
        if state.nodes >= PROVIDER_TRACE_MAX_NODES:
            state.blocks.append(_omitted_value(item, reason="node_limit"))
            state.limited = True
            return
        key_text = _safe_key_text(key).lower()
        if _is_opaque_key(key_text) and child is not None:
            state.blocks.append(_reasoning_opaque_value(child, source=key_text))
            continue
        if key_text in readable_keys:
            _visit_reasoning_value(
                child,
                source=source if key_text == "text" else key_text,
                depth=depth + 1,
                state=state,
            )


def _reasoning_opaque_value(value: Any, *, source: str) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("type") == "opaque":
        if type(value.get("bytes")) is int:
            return _existing_opaque_block(value, source=source)
    return _opaque_block(value, source=source)


def _existing_opaque_block(value: dict[Any, Any], *, source: str) -> dict[str, Any]:
    byte_count = value.get("bytes")
    selected_bytes = (
        byte_count
        if type(byte_count) is int
        and 0 <= byte_count <= PROVIDER_TRACE_SAFE_INTEGER_MAX
        else 0
    )
    return {
        "type": "opaque",
        "source": _safe_key_text(value.get("source") or source)[:128],
        "bytes": selected_bytes,
        "sha256": _safe_key_text(value.get("sha256") or "")[:64],
        **(
            {"digest_scope": _safe_key_text(value["digest_scope"])[:64]}
            if value.get("digest_scope")
            else {}
        ),
    }


def _safe_omitted_descriptor(value: dict[Any, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in ("chars", "bytes"):
        item = value.get(key)
        if type(item) is int and 0 <= item <= PROVIDER_TRACE_SAFE_INTEGER_MAX:
            selected[key] = item
    if "sha256" in value:
        selected["sha256"] = _safe_key_text(value.get("sha256") or "")[:64]
    if value.get("digest_scope"):
        selected["digest_scope"] = _safe_key_text(value["digest_scope"])[:64]
    return selected


def project_provider_raw_response(value: Any) -> Any:
    """Return a bounded raw-response projection with opaque bytes hashed."""

    state = _RawProjectionState()
    projected = _project_provider_value(value, key="", depth=0, state=state)
    if isinstance(projected, dict) and state.limited:
        projected.setdefault("_provider_projection_limited", True)
    if _json_bytes(projected) > PROVIDER_TRACE_MAX_BYTES:
        encoded = _json_encode(projected)
        return {
            "_provider_projection_limited": True,
            "type": "omitted",
            "reason": "aggregate_limit",
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return projected


def _project_provider_value(
    item: Any,
    *,
    key: str,
    depth: int,
    state: _RawProjectionState,
) -> Any:
    state.nodes += 1
    if depth > PROVIDER_TRACE_MAX_DEPTH or state.nodes > PROVIDER_TRACE_MAX_NODES:
        state.limited = True
        return _omitted_value(item, reason="structure_limit")
    if item is None:
        return None
    if _is_sensitive_provider_key(key):
        return _opaque_block(item, source=key or "opaque")
    if isinstance(item, bool):
        return item
    if isinstance(item, int):
        return _project_provider_integer(item, state=state)
    if isinstance(item, float):
        return _project_provider_float(item, state=state)
    if isinstance(item, str):
        return _project_provider_text(item, state=state)
    if isinstance(item, (bytes, bytearray)):
        return _opaque_block(item, source=key or "bytes")
    if isinstance(item, dict):
        return _project_provider_mapping(item, key=key, depth=depth, state=state)
    if isinstance(item, (list, tuple)):
        return _project_provider_sequence(item, depth=depth, state=state)
    object_view = _safe_provider_object_view(
        item,
        max_items=max(1, PROVIDER_TRACE_MAX_NODES - state.nodes),
    )
    if object_view is item:
        return {"type": type(item).__name__[:128]}
    return _project_provider_value(
        object_view,
        key=key,
        depth=depth + 1,
        state=state,
    )


def _project_provider_integer(value: int, *, state: _RawProjectionState) -> Any:
    if abs(value) <= PROVIDER_TRACE_SAFE_INTEGER_MAX:
        return value
    state.limited = True
    return _omitted_value(value, reason="integer_out_of_range")


def _project_provider_float(value: float, *, state: _RawProjectionState) -> Any:
    if math.isfinite(value):
        return value
    state.limited = True
    return _omitted_value(value, reason="non_finite_number")


def _project_provider_text(value: str, *, state: _RawProjectionState) -> Any:
    selected, omitted = _bounded_readable_text(value)
    selected_bytes = len(selected.encode("utf-8")) if selected else 0
    if omitted is None and state.text_bytes + selected_bytes <= PROVIDER_TRACE_MAX_BYTES:
        state.text_bytes += selected_bytes
        return selected
    state.limited = True
    return {
        "type": "omitted",
        "reason": "aggregate_limit" if omitted is None else "bounds",
        **(omitted or _text_digest(value)),
    }


def _project_provider_mapping(
    item: dict[Any, Any],
    *,
    key: str,
    depth: int,
    state: _RawProjectionState,
) -> dict[str, Any]:
    marker = _safe_key_text(item.get("type") or "").lower()
    if marker in {"opaque", "encrypted", "ciphertext", "signature"}:
        return _opaque_block(item, source=key or marker)
    result: dict[str, Any] = {}
    sensitive_named_value = _sensitive_named_value_mapping(item)
    for index, (raw_key, child) in enumerate(item.items()):
        if state.nodes >= PROVIDER_TRACE_MAX_NODES:
            state.limited = True
            result["_provider_projection_limited"] = True
            result["_omitted"] = _omitted_mapping_tail(
                item,
                index,
                reason="node_limit",
            )
            return result
        full_key = _safe_key_text(raw_key)
        selected_key = full_key[:256]
        normalized_key = full_key.strip().lower().replace("-", "_")
        if _is_sensitive_provider_key(full_key) or (
            sensitive_named_value
            and normalized_key
            not in {"name", "key", "header", "header_name", "headername"}
        ):
            result[selected_key] = (
                None
                if child is None
                else _opaque_block(child, source=selected_key or "opaque")
            )
            continue
        result[selected_key] = _project_provider_value(
            child,
            key=selected_key.lower(),
            depth=depth + 1,
            state=state,
        )
    return result


def _project_provider_sequence(
    item: list[Any] | tuple[Any, ...],
    *,
    depth: int,
    state: _RawProjectionState,
) -> list[Any]:
    result: list[Any] = []
    sensitive_pair_key = (
        _safe_key_text(item[0])
        if len(item) >= 2
        and isinstance(item[0], str)
        and _is_sensitive_provider_key(item[0])
        else None
    )
    for index, child in enumerate(item):
        if state.nodes >= PROVIDER_TRACE_MAX_NODES:
            state.limited = True
            result.append(_omitted_sequence_tail(item, index, reason="node_limit"))
            return result
        result.append(
            _project_provider_value(
                child,
                key=(sensitive_pair_key or "") if index > 0 else "",
                depth=depth + 1,
                state=state,
            )
        )
    return result


def project_provider_usage(value: Any) -> dict[str, Any]:
    jsonable = _safe_provider_object_view(value, max_items=512)
    if not isinstance(jsonable, dict):
        return {}

    nodes = [0]
    invalid_fields: list[str] = []

    def visit(item: Any, depth: int, path: str) -> Any:
        nodes[0] += 1
        if depth > 8 or nodes[0] > 512:
            invalid_fields.append(path)
            return None
        if item is None:
            return item
        if type(item) is int:
            if abs(item) <= PROVIDER_TRACE_SAFE_INTEGER_MAX:
                return item
            invalid_fields.append(path)
            return None
        if type(item) is float:
            if math.isfinite(item):
                return item
            invalid_fields.append(path)
            return None
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in islice(item.items(), 256):
                selected_key = _safe_key_text(key)[:128]
                selected_path = f"{path}.{selected_key}" if path else selected_key
                result[selected_key] = visit(child, depth + 1, selected_path)
            if len(item) > 256:
                invalid_fields.append(f"{path}.*" if path else "*")
            return result
        object_view = _safe_provider_object_view(
            item,
            max_items=max(1, 512 - nodes[0]),
        )
        if object_view is not item:
            return visit(object_view, depth + 1, path)
        invalid_fields.append(path)
        return None

    selected = visit(jsonable, 0, "")
    if not isinstance(selected, dict):
        return {}
    if invalid_fields:
        selected["_provider_projection_invalid_fields"] = sorted(
            set(invalid_fields)
        )[:256]
    return selected


def project_provider_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    projected: list[dict[str, Any]] = []
    for raw in list(value)[:256]:
        if not isinstance(raw, dict):
            continue
        call: dict[str, Any] = {}
        for key in ("id", "call_id", "name"):
            selected = _bounded_identifier(raw.get(key))
            if selected is not None:
                call[key] = selected
        arguments = raw.get("arguments")
        if isinstance(arguments, str):
            selected_arguments, omitted = _bounded_readable_text(arguments)
            if omitted is None:
                call["arguments"] = selected_arguments
            else:
                call["arguments_limited"] = omitted
        projected.append(call)
    return projected


def safe_provider_error(error: BaseException) -> dict[str, Any]:
    try:
        message = str(error)
    except BaseException:
        message = "provider error text unavailable"
    encoded = message.encode("utf-8", errors="replace")
    result = {
        "error_type": type(error).__name__[:128],
        "message_bytes": len(encoded),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    status_code = getattr(error, "status_code", None)
    if type(status_code) is int and 100 <= status_code <= 599:
        result["status_code"] = status_code
    return result


def _fit_trace_aggregate(trace: dict[str, Any]) -> dict[str, Any]:
    original_encoded = _json_encode(trace)
    if len(original_encoded) <= PROVIDER_TRACE_MAX_BYTES:
        return trace
    original_observation = {
        "bytes": len(original_encoded),
        "sha256": hashlib.sha256(original_encoded).hexdigest(),
    }
    trace["limited"] = True
    attempts = trace.get("attempts")
    if not isinstance(attempts, list):
        return trace
    selected = trace.get("selected_attempt")
    ordered = sorted(
        (item for item in attempts if isinstance(item, dict)),
        key=lambda item: item.get("sequence") == selected,
    )
    for attempt in ordered:
        _limit_attempt_readable_content(attempt)
        if _json_bytes(trace) <= PROVIDER_TRACE_MAX_BYTES:
            return trace
    for attempt in ordered:
        _limit_attempt_diagnostics(attempt)
        if _json_bytes(trace) <= PROVIDER_TRACE_MAX_BYTES:
            return trace
    return _omitted_trace_envelope(
        trace,
        attempts=attempts,
        original_observation=original_observation,
    )


def _limit_attempt_readable_content(attempt: dict[str, Any]) -> None:
    reasoning = attempt.get("reasoning")
    if isinstance(reasoning, dict):
        reasoning["availability"] = "limited"
        reasoning["blocks"] = _limited_reasoning_blocks(reasoning.get("blocks"))
    output = attempt.get("output")
    if isinstance(output, str) and output:
        attempt["output_limited"] = _text_digest(output)
        attempt["output"] = ""
    calls = attempt.get("tool_calls")
    if not isinstance(calls, list):
        return
    for call in calls:
        if isinstance(call, dict) and isinstance(call.get("arguments"), str):
            call["arguments_limited"] = _text_digest(call.pop("arguments"))


def _limited_reasoning_blocks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    limited: list[dict[str, Any]] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            limited.append(
                {
                    "type": "omitted",
                    "reason": "aggregate_limit",
                    **_text_digest(text),
                }
            )
        else:
            limited.append(block)
    return limited


def _limit_attempt_diagnostics(attempt: dict[str, Any]) -> None:
    calls = attempt.get("tool_calls")
    if isinstance(calls, list) and calls:
        attempt["tool_calls"] = []
        attempt["tool_calls_limited"] = {
            "count": len(calls),
            **_omitted_value(calls, reason="aggregate_limit"),
        }
    usage = attempt.get("usage")
    if isinstance(usage, dict) and usage:
        attempt["usage"] = {}
        attempt["usage_limited"] = _omitted_value(
            usage,
            reason="aggregate_limit",
        )


def _omitted_trace_envelope(
    trace: dict[str, Any],
    *,
    attempts: list[Any],
    original_observation: dict[str, Any],
) -> dict[str, Any]:
    omitted = trace.get("omitted_attempts")
    omitted_count = omitted if type(omitted) is int and omitted >= 0 else 0
    return {
        "kind": "provider_trace",
        "schema_version": PROVIDER_TRACE_SCHEMA_VERSION,
        "coverage": trace.get("coverage"),
        "selected_attempt": trace.get("selected_attempt"),
        "limited": True,
        "omitted_attempts": len(attempts) + omitted_count,
        "attempts": [],
        "omitted_trace": original_observation,
    }


def _bounded_readable_text(value: Any) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(value, str):
        return "", None
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        encoded = value.encode("utf-8", errors="replace")
        return "", {
            "chars": len(value),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if (
        len(value) <= PROVIDER_TRACE_TEXT_MAX_CHARS
        and len(encoded) <= PROVIDER_TRACE_TEXT_MAX_BYTES
    ):
        return value, None
    return "", {
        "chars": len(value),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _opaque_block(value: Any, *, source: str) -> dict[str, Any]:
    byte_count, sha256, digest_scope = _value_digest(value)
    result = {
        "type": "opaque",
        "source": source[:128],
        "bytes": byte_count,
        "sha256": sha256,
    }
    if digest_scope != "exact":
        result["digest_scope"] = digest_scope
    return result


def _omitted_value(value: Any, *, reason: str) -> dict[str, Any]:
    opaque = _opaque_block(value, source=reason)
    result = {
        "type": "omitted",
        "reason": reason,
        "bytes": opaque["bytes"],
        "sha256": opaque["sha256"],
    }
    if "digest_scope" in opaque:
        result["digest_scope"] = opaque["digest_scope"]
    return result


def _omitted_sequence_tail(
    value: list[Any] | tuple[Any, ...],
    start: int,
    *,
    reason: str,
) -> dict[str, Any]:
    omitted_items = max(0, len(value) - start)
    sample = list(islice(value, start, start + 32))
    result = _omitted_value(
        {"omitted_items": omitted_items, "sample": sample},
        reason=reason,
    )
    result["items"] = omitted_items
    return result


def _omitted_mapping_tail(
    value: dict[Any, Any],
    start: int,
    *,
    reason: str,
) -> dict[str, Any]:
    omitted_items = max(0, len(value) - start)
    sample = list(islice(value.items(), start, start + 32))
    result = _omitted_value(
        {"omitted_items": omitted_items, "sample": sample},
        reason=reason,
    )
    result["items"] = omitted_items
    return result


def _value_digest(value: Any) -> tuple[int, str, str]:
    digest = hashlib.sha256()
    byte_count = 0

    def feed(data: bytes) -> None:
        nonlocal byte_count
        byte_count += len(data)
        digest.update(data)

    if isinstance(value, str):
        for offset in range(0, len(value), 8_192):
            feed(value[offset : offset + 8_192].encode("utf-8", errors="replace"))
        return byte_count, digest.hexdigest(), "exact"
    if isinstance(value, (bytes, bytearray)):
        selected = memoryview(value)
        for offset in range(0, len(selected), 65_536):
            feed(bytes(selected[offset : offset + 65_536]))
        return byte_count, digest.hexdigest(), "exact"

    # Structural omissions must not defeat the depth/node limits by recursively
    # serializing the entire rejected subtree. Hash a deterministic bounded
    # summary instead and label its scope explicitly.
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending and nodes < 256:
        item, depth = pending.pop()
        nodes += 1
        if item is None or isinstance(item, (bool, float)):
            feed(f"scalar:{type(item).__name__}:{item!r};".encode())
            continue
        if isinstance(item, int):
            feed(f"int:bits={item.bit_length()}:sign={item < 0};".encode())
            continue
        if isinstance(item, str):
            prefix = item[:512].encode("utf-8", errors="replace")
            suffix = item[-512:].encode("utf-8", errors="replace")
            feed(f"str:{len(item)}:".encode())
            feed(prefix)
            feed(b":")
            feed(suffix)
            continue
        if isinstance(item, (bytes, bytearray)):
            selected = memoryview(item)
            feed(f"bytes:{len(selected)}:".encode())
            feed(bytes(selected[:512]))
            feed(b":")
            feed(bytes(selected[-512:]))
            continue
        object_id = id(item)
        if object_id in seen:
            feed(b"cycle;")
            continue
        seen.add(object_id)
        if depth >= 4:
            feed(f"depth:{type(item).__name__};".encode())
            continue
        if isinstance(item, dict):
            feed(f"dict:{len(item)};".encode())
            selected_items = list(islice(item.items(), 32))
            for key, child in reversed(selected_items):
                pending.append((child, depth + 1))
                pending.append((_safe_key_text(key), depth + 1))
            continue
        if isinstance(item, (list, tuple)):
            feed(f"sequence:{len(item)};".encode())
            for child in reversed(item[:32]):
                pending.append((child, depth + 1))
            continue
        feed(f"object:{type(item).__module__}.{type(item).__qualname__};".encode())
    if pending:
        feed(f"nodes_omitted:{len(pending)};".encode())
    return byte_count, digest.hexdigest(), "bounded_summary"


def _text_digest(value: str) -> dict[str, Any]:
    encoded = value.encode("utf-8", errors="replace")
    return {
        "chars": len(value),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _bounded_identifier(value: Any) -> str | None:
    if value is None:
        return None
    try:
        selected = str(value)
    except BaseException:
        return None
    if not selected:
        return None
    try:
        selected.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return selected[:512]


def _empty_reasoning() -> dict[str, Any]:
    return {"availability": "not_returned", "blocks": []}


def _is_opaque_key(key: str) -> bool:
    selected = key.lower()
    return selected == "bytes" or any(fragment in selected for fragment in _OPAQUE_KEY_FRAGMENTS)


def _is_sensitive_provider_key(key: str) -> bool:
    selected = key.strip().lower().replace("-", "_")
    compact = selected.replace("_", "")
    if _is_opaque_key(selected) or any(
        marker in selected for marker in _CREDENTIAL_KEY_MARKERS
    ):
        return True
    if (
        selected in _CREDENTIAL_TOKEN_KEYS
        or compact in _CREDENTIAL_TOKEN_KEYS
        or selected in _CREDENTIAL_COOKIE_KEYS
        or compact in _CREDENTIAL_COOKIE_KEYS
    ):
        return True
    return selected.endswith(("_token", "_cookie", "_cookies"))


def _sensitive_named_value_mapping(value: dict[Any, Any]) -> bool:
    for raw_key, item in value.items():
        selected_key = _safe_key_text(raw_key).strip().lower().replace("-", "_")
        if selected_key not in {"name", "key", "header", "header_name", "headername"}:
            continue
        if isinstance(item, str) and _is_sensitive_provider_key(item):
            return True
    return False


def _safe_key_text(value: Any) -> str:
    try:
        selected = str(value)
    except BaseException:
        selected = f"<{type(value).__name__}>"
    return selected.encode("utf-8", errors="replace").decode("utf-8")


def _get_attr_or_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _safe_provider_object_view(value: Any, *, max_items: int) -> Any:
    """Expose at most one object layer; bounded walkers own all recursion."""

    try:
        attributes = object.__getattribute__(value, "__dict__")
    except BaseException:
        attributes = None
    try:
        extras = object.__getattribute__(value, "__pydantic_extra__")
    except BaseException:
        extras = None
    if isinstance(attributes, dict) and not isinstance(extras, dict):
        return attributes

    if isinstance(attributes, dict) or isinstance(extras, dict):
        return _bounded_object_attributes(
            attributes if isinstance(attributes, dict) else {},
            extras if isinstance(extras, dict) else {},
            max_items=max_items,
        )

    try:
        slots = getattr(type(value), "__slots__", ())
    except BaseException:
        return value
    if isinstance(slots, str):
        slots = (slots,)
    if not isinstance(slots, (list, tuple)):
        return value
    selected: dict[str, Any] = {}
    for name in islice(slots, max(1, max_items)):
        if not isinstance(name, str) or name.startswith("_"):
            continue
        try:
            selected[name] = object.__getattribute__(value, name)
        except BaseException:
            continue
    return selected or value


def _bounded_object_attributes(
    attributes: dict[Any, Any],
    extras: dict[Any, Any],
    *,
    max_items: int,
) -> dict[Any, Any]:
    selected: dict[Any, Any] = {}
    budget = max(1, max_items)
    attribute_budget = budget if not extras else max(1, budget - min(256, budget // 2))
    for key, item in islice(attributes.items(), attribute_budget):
        selected[key] = item
    for key, item in islice(extras.items(), max(0, budget - len(selected))):
        selected[key] = item
    observed = len(selected)
    total = len(attributes) + len(extras)
    if total > observed:
        selected["_provider_projection_limited"] = True
        selected["_omitted"] = _omitted_value(
            {
                "attribute_count": len(attributes),
                "extra_count": len(extras),
                "observed_count": observed,
            },
            reason="object_field_limit",
        )
    return selected


def _json_bytes(value: Any) -> int:
    return len(_json_encode(value))


def _json_encode(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
        allow_nan=False,
    ).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
