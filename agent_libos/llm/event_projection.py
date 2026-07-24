from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import Event, EventType, ResourceUsage


_RESOURCE_USAGE_FIELDS = frozenset(ResourceUsage.__dataclass_fields__)
_MESSAGE_EVENT_TYPES = frozenset(
    {
        EventType.PROCESS_MESSAGE_POSTED,
        EventType.PROCESS_MESSAGE_ACKED,
    }
)
_ACTIONABLE_FIELD_ORDER = (
    "outcome",
    "reason",
    "code",
    "status",
    "path",
    "sink",
    "direction",
    "labels",
    "trust_id",
    "release_capability_id",
    "result_oid",
    "error",
    "kind",
    "count",
    "phase",
)


@dataclass(frozen=True)
class ProjectedEventBatch:
    """A bounded, provider-neutral view of durable runtime events.

    Event rows and their original payloads remain untouched in the evidence
    store.  This value is only the model-facing projection.  ``summary``
    accounts for every input row so a caller may safely advance its durable
    event cursor after persisting or rendering the projection.
    """

    visible_records: list[dict[str, Any]]
    represented_through_event_id: str | None
    omitted_counts: dict[str, int]
    resource_usage_delta: dict[str, int | float]
    summary: dict[str, Any]

    @property
    def events(self) -> list[dict[str, Any]]:
        """Compatibility spelling for context-object entry construction."""

        return self.visible_records

    @property
    def model_records(self) -> list[dict[str, Any]]:
        """Visible event records plus one bounded accounting record.

        Source-only prompt assembly should use this property so policy-omitted
        rows are still explicitly represented to the model without teaching
        the executor a second projection format.
        """

        if not self.summary.get("input_event_count"):
            return []
        summary_payload = {
            key: self.summary[key]
            for key in (
                "input_event_count",
                "represented_event_count",
                "omitted_event_count",
                "payload_truncated_event_count",
                "represented_type_counts",
                "omitted_reason_counts",
                "omitted_events_sha256",
            )
            if key in self.summary
        }
        if self.resource_usage_delta:
            summary_payload["resource_usage_delta"] = self.resource_usage_delta
        return [
            *self.visible_records,
            {
                "type": "event_projection_summary",
                "payload": summary_payload,
            },
        ]

    def to_dict(self) -> dict[str, Any]:
        projected: dict[str, Any] = {
            "events": self.visible_records,
            "projection_summary": self.summary,
        }
        if self.omitted_counts:
            projected["omitted_evidence_event_counts"] = self.omitted_counts
        if self.resource_usage_delta:
            projected["resource_usage_delta"] = self.resource_usage_delta
        return projected

    def canonical_json(self) -> str:
        return canonical_prompt_json(self.to_dict())


def project_prompt_events(
    events: Iterable[Event],
    *,
    context_object_name: str | None = None,
    payload_max_chars: int = DEFAULT_CONFIG.llm_context.prompt_event_payload_max_chars,
) -> ProjectedEventBatch:
    """Project audit events into deterministic, bounded prompt data.

    ``payload_max_chars`` applies independently to each represented event.
    Repetitive bookkeeping is summarized by type/reason and a digest instead
    of being copied into the prompt.  Process-message metadata is deliberately
    delegated to the mediated message-read directive and never projected.
    """

    if payload_max_chars < 512:
        raise ValueError("payload_max_chars must be at least 512")

    selected_events = list(events)
    projected: list[dict[str, Any]] = []
    omitted_counts: dict[str, int] = {}
    omitted_descriptors: list[dict[str, Any]] = []
    resource_usage_delta: dict[str, int | float] = {}
    represented_type_counts: dict[str, int] = {}
    payload_truncated_count = 0

    for event in selected_events:
        omission_reason = _prompt_event_omission_reason(
            event,
            context_object_name=context_object_name,
        )
        if omission_reason is not None:
            omitted_counts[omission_reason] = omitted_counts.get(omission_reason, 0) + 1
            omitted_descriptors.append(_event_digest_descriptor(event))
            if event.type == EventType.RESOURCE_CHARGED:
                _accumulate_resource_usage(
                    resource_usage_delta,
                    event.payload.get("usage"),
                )
            continue

        record, was_truncated = _project_visible_event(
            event,
            max_chars=payload_max_chars,
        )
        if was_truncated:
            payload_truncated_count += 1
        event_type = event.type.value
        represented_type_counts[event_type] = represented_type_counts.get(event_type, 0) + 1
        projected.append(record)

    represented_through_event_id = selected_events[-1].event_id if selected_events else None
    summary = _projection_summary(
        selected_events,
        projected_count=len(projected),
        payload_truncated_count=payload_truncated_count,
        represented_type_counts=represented_type_counts,
        omitted_counts=omitted_counts,
        omitted_descriptors=omitted_descriptors,
        represented_through_event_id=represented_through_event_id,
    )

    return ProjectedEventBatch(
        visible_records=projected,
        represented_through_event_id=represented_through_event_id,
        omitted_counts=dict(sorted(omitted_counts.items())),
        resource_usage_delta=dict(sorted(resource_usage_delta.items())),
        summary=summary,
    )


def _prompt_event_omission_reason(
    event: Event,
    *,
    context_object_name: str | None,
) -> str | None:
    if event.type == EventType.RESOURCE_CHARGED:
        return event.type.value
    if (
        context_object_name
        and event.type in {EventType.OBJECT_CREATED, EventType.OBJECT_UPDATED}
        and event.payload.get("name") == context_object_name
    ):
        return f"llm_context_{event.type.value}"
    if event.type in _MESSAGE_EVENT_TYPES:
        # The live process-message directive instructs the model to cross the
        # mediated read boundary, so its identifying metadata stays omitted.
        return f"delegated_{event.type.value}"
    if event.type == EventType.TOOL_COMPLETED:
        # Tool output is already present through the memory/result channel.
        return event.type.value
    if (
        event.type == EventType.DATA_FLOW_DECISION
        and event.payload.get("outcome") == "allow"
    ):
        return "allowed_data_flow_decision"
    return None


def _project_visible_event(
    event: Event,
    *,
    max_chars: int,
) -> tuple[dict[str, Any], bool]:
    if event.type == EventType.PROCESS_MESSAGE_NOTICE:
        return {
            "type": event.type.value,
            "payload": _project_process_message_notice(event.payload),
        }, False

    raw_payload: Any = event.payload
    if event.type == EventType.DATA_FLOW_DECISION:
        raw_payload = {
            key: event.payload[key]
            for key in (
                "decision_id",
                "direction",
                "outcome",
                "reason",
                "sink",
                "labels",
                "trust_id",
                "release_capability_id",
            )
            if key in event.payload
        }
    elif event.type == EventType.PROCESS_SIGNAL:
        raw_payload = _project_process_signal(event.payload)

    payload, was_truncated = _bounded_payload_projection(
        raw_payload,
        max_chars=max_chars,
    )
    return {
        "event_id": _bounded_identifier(event.event_id),
        "type": event.type.value,
        "source": _bounded_identifier(event.source),
        "target": _bounded_identifier(event.target) if event.target is not None else None,
        "payload": payload,
    }, was_truncated


def _projection_summary(
    selected_events: list[Event],
    *,
    projected_count: int,
    payload_truncated_count: int,
    represented_type_counts: dict[str, int],
    omitted_counts: dict[str, int],
    omitted_descriptors: list[dict[str, Any]],
    represented_through_event_id: str | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "input_event_count": len(selected_events),
        "represented_event_count": projected_count,
        "omitted_event_count": len(selected_events) - projected_count,
        "payload_truncated_event_count": payload_truncated_count,
        "represented_type_counts": dict(sorted(represented_type_counts.items())),
        "omitted_reason_counts": dict(sorted(omitted_counts.items())),
        "input_event_ids_sha256": _digest_json([event.event_id for event in selected_events]),
        "represented_through_event_id": represented_through_event_id,
    }
    if omitted_descriptors:
        summary["omitted_events_sha256"] = _digest_json(omitted_descriptors)
    return summary


# Kept as an import-compatible alias while callers migrate to the more explicit
# batch name.
PromptEventProjection = ProjectedEventBatch


def canonical_prompt_json(value: Any) -> str:
    """Render projection data as canonical JSON accepted by any provider."""

    return json.dumps(
        _normalize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _project_process_signal(payload: dict[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    signal = payload.get("signal")
    if isinstance(signal, str) and signal:
        projected["signal"] = signal
    else:
        projected["signal"] = "runtime_control"

    nested = payload.get("payload")
    reason_ref = payload.get("reason_oid")
    if not reason_ref and isinstance(nested, dict):
        reason_ref = nested.get("reason_oid")
    if isinstance(reason_ref, str) and reason_ref:
        projected["reason_ref"] = _bounded_identifier(reason_ref)
    return projected


def _project_process_message_notice(payload: dict[str, Any]) -> dict[str, Any]:
    kind = payload.get("kind")
    count = payload.get("count")
    projected: dict[str, Any] = {
        "control": "read_pending_process_messages",
        "count": count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 1,
    }
    if isinstance(kind, str) and kind:
        projected["kind"] = _bounded_identifier(kind, max_chars=64)
    return projected


def _accumulate_resource_usage(
    totals: dict[str, int | float],
    usage: Any,
) -> None:
    if not isinstance(usage, dict):
        return
    for name in sorted(_RESOURCE_USAGE_FIELDS):
        value = usage.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
            and value != 0
        ):
            total = totals.get(name, 0) + value
            if total == 0:
                totals.pop(name, None)
            else:
                totals[name] = total


def _bounded_payload_projection(
    payload: Any,
    *,
    max_chars: int,
) -> tuple[Any, bool]:
    normalized = _normalize_json(payload)
    full_json = canonical_prompt_json(normalized)
    if len(full_json) <= max_chars:
        return normalized, False

    full_digest = hashlib.sha256(full_json.encode("utf-8")).hexdigest()
    envelope: dict[str, Any] = {
        "_projection": {
            "truncated": True,
            "payload_sha256": full_digest,
            "original_chars": len(full_json),
        },
        "fields": {},
    }
    fields = envelope["fields"]
    assert isinstance(fields, dict)

    if isinstance(normalized, dict):
        ordered_keys = _ordered_payload_keys(normalized)
        for key in ordered_keys:
            candidate = _compact_value(normalized[key], max_chars=max_chars)
            fields[key] = candidate
            if len(canonical_prompt_json(envelope)) > max_chars:
                fields[key] = _value_summary(normalized[key])
            if len(canonical_prompt_json(envelope)) > max_chars:
                fields.pop(key, None)
        omitted_fields = len(normalized) - len(fields)
        if omitted_fields:
            envelope["_projection"]["omitted_field_count"] = omitted_fields
    else:
        fields["value"] = _compact_value(normalized, max_chars=max_chars)
        if len(canonical_prompt_json(envelope)) > max_chars:
            fields["value"] = _value_summary(normalized)
        if len(canonical_prompt_json(envelope)) > max_chars:
            fields.clear()

    rendered_chars = len(canonical_prompt_json(envelope))
    envelope["_projection"]["omitted_chars"] = max(len(full_json) - rendered_chars, 0)
    if len(canonical_prompt_json(envelope)) > max_chars:
        # ``payload_max_chars`` is validated above, so this compact metadata
        # envelope always fits even when the input has millions of fields.
        envelope = {
            "_projection": {
                "truncated": True,
                "payload_sha256": full_digest,
                "original_chars": len(full_json),
                "omitted_chars": len(full_json),
            }
        }
    return envelope, True


def _ordered_payload_keys(payload: dict[str, Any]) -> list[str]:
    priority = {name: index for index, name in enumerate(_ACTIONABLE_FIELD_ORDER)}
    return sorted(payload, key=lambda key: (priority.get(key, len(priority)), key))


def _compact_value(value: Any, *, max_chars: int) -> Any:
    string_limit = min(512, max(64, max_chars // 4))
    collection_limit = 12
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        omitted = value[string_limit:]
        return {
            "_text_preview": value[:string_limit],
            "_omitted_chars": len(omitted),
            "_omitted_sha256": hashlib.sha256(omitted.encode("utf-8")).hexdigest(),
        }
    if isinstance(value, list):
        retained = [
            _compact_value(item, max_chars=max_chars)
            for item in value[:collection_limit]
        ]
        if len(value) <= collection_limit:
            return retained
        return {
            "_items": retained,
            "_omitted_item_count": len(value) - collection_limit,
            "_omitted_sha256": _digest_json(value[collection_limit:]),
        }
    if isinstance(value, dict):
        keys = sorted(value)
        retained = {
            key: _compact_value(value[key], max_chars=max_chars)
            for key in keys[:collection_limit]
        }
        if len(keys) <= collection_limit:
            return retained
        return {
            "_fields": retained,
            "_omitted_field_count": len(keys) - collection_limit,
            "_omitted_sha256": _digest_json({key: value[key] for key in keys[collection_limit:]}),
        }
    return value


def _value_summary(value: Any) -> dict[str, Any]:
    rendered = canonical_prompt_json(value)
    summary: dict[str, Any] = {
        "_type": _json_type(value),
        "_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "_chars": len(rendered),
    }
    if isinstance(value, (dict, list)):
        summary["_item_count"] = len(value)
    return summary


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _event_digest_descriptor(event: Event) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "type": event.type.value,
        "source": event.source,
        "target": event.target,
        "payload": _normalize_json(event.payload),
    }


def _bounded_identifier(value: Any, *, max_chars: int = 256) -> str:
    selected = str(value)
    if len(selected) <= max_chars:
        return selected
    return (
        selected[: max_chars - 81]
        + "...[sha256:"
        + hashlib.sha256(selected.encode("utf-8")).hexdigest()
        + "]"
    )


def _digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_prompt_json(value).encode("utf-8")).hexdigest()


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"_non_finite_number": str(value)}
    if isinstance(value, dict):
        normalized_items = sorted(
            ((str(key), _normalize_json(item)) for key, item in value.items()),
            key=lambda item: item[0],
        )
        return {key: item for key, item in normalized_items}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "_binary_bytes": len(raw),
            "_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {"_unsupported_type": f"{type(value).__module__}.{type(value).__qualname__}"}
