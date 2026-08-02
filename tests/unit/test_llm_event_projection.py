from __future__ import annotations

import json

from agent_libos.llm.event_projection import (
    canonical_prompt_json,
    project_prompt_events,
)
from agent_libos.models import Event, EventPriority, EventType


def _event(
    index: int,
    event_type: EventType,
    payload: dict[str, object],
    *,
    source: str = "runtime",
) -> Event:
    return Event(
        event_id=f"event-{index:02d}",
        type=event_type,
        source=source,
        target="pid-test",
        payload=payload,
        priority=EventPriority.NORMAL,
        created_at="2026-07-25T00:00:00Z",
    )


def test_message_projection_delegates_without_leaking_metadata() -> None:
    sentinel = "MESSAGE_METADATA_MUST_NOT_REACH_THE_MODEL"
    events = [
        _event(
            1,
            EventType.PROCESS_MESSAGE_POSTED,
            {
                "message_id": f"pmsg-{sentinel}",
                "kind": "interrupt",
                "subject": sentinel,
                "body": sentinel,
                "correlation_id": sentinel,
                "sender": sentinel,
            },
            source=sentinel,
        ),
        _event(
            2,
            EventType.PROCESS_MESSAGE_NOTICE,
            {
                "kind": "interrupt",
                "count": 1,
                "message_ids": [sentinel],
                "correlation_ids": [sentinel],
                "instruction": sentinel,
            },
        ),
        _event(
            3,
            EventType.PROCESS_MESSAGE_ACKED,
            {"message_ids": [sentinel], "count": 1},
        ),
    ]

    projection = project_prompt_events(
        events,
        context_object_name="llm_context_pid-test",
        payload_max_chars=2_048,
    )
    rendered = projection.canonical_json()

    assert projection.visible_records == [
        {
            "type": "process_message_notice",
            "payload": {
                "control": "read_pending_process_messages",
                "count": 1,
                "kind": "interrupt",
            },
        }
    ]
    assert projection.summary["input_event_count"] == 3
    assert projection.summary["represented_event_count"] == 1
    assert projection.summary["omitted_event_count"] == 2
    assert projection.represented_through_event_id == "event-03"
    assert sum(projection.omitted_counts.values()) == 2
    assert projection.model_records[-1]["type"] == "event_projection_summary"
    assert projection.model_records[-1]["payload"]["omitted_event_count"] == 2
    assert sentinel not in rendered
    assert "message_id" not in rendered
    assert "correlation" not in rendered


def test_process_signal_keeps_only_control_type_and_reason_reference() -> None:
    sentinel = "RAW_SIGNAL_REASON_MUST_NOT_REACH_THE_MODEL"
    event = _event(
        1,
        EventType.PROCESS_SIGNAL,
        {
            "signal": "pause",
            "payload": {
                "reason_oid": "oid-reason",
                "reason": sentinel,
                "nested": {"secret": sentinel},
            },
            "reason": sentinel,
        },
    )

    projection = project_prompt_events(
        [event],
        context_object_name="llm_context_pid-test",
        payload_max_chars=2_048,
    )

    assert projection.visible_records[0]["payload"] == {
        "signal": "pause",
        "reason_ref": "oid-reason",
    }
    assert sentinel not in projection.canonical_json()


def test_large_payload_is_bounded_canonical_and_deterministic() -> None:
    huge_value = "x" * 100_000 + "LARGE_PAYLOAD_TAIL_SENTINEL"
    event = _event(
        1,
        EventType.EXTERNAL_WRITE,
        {
            "path": "result.json",
            "body": huge_value,
            "rows": [{"value": huge_value, "index": index} for index in range(20)],
        },
    )

    first = project_prompt_events(
        [event],
        context_object_name="llm_context_pid-test",
        payload_max_chars=768,
    )
    second = project_prompt_events(
        [event],
        context_object_name="llm_context_pid-test",
        payload_max_chars=768,
    )
    payload = first.visible_records[0]["payload"]
    payload_json = canonical_prompt_json(payload)

    assert len(payload_json) <= 768
    assert payload["_projection"]["truncated"] is True
    assert payload["_projection"]["payload_sha256"]
    assert payload["_projection"]["omitted_chars"] > 0
    assert first.summary["payload_truncated_event_count"] == 1
    assert "LARGE_PAYLOAD_TAIL_SENTINEL" not in payload_json
    assert first == second
    assert first.canonical_json() == second.canonical_json()
    assert json.loads(first.canonical_json()) == first.to_dict()


def test_twenty_row_window_is_fully_accounted_for() -> None:
    events: list[Event] = []
    for index in range(5):
        events.append(
            _event(
                index,
                EventType.RESOURCE_CHARGED,
                {"usage": {"tool_calls": 1}, "noise": "x" * 10_000},
            )
        )
    for index in range(5, 10):
        events.append(
            _event(
                index,
                EventType.TOOL_COMPLETED,
                {"call_id": f"call-{index}", "result_oid": f"oid-{index}"},
            )
        )
    for index in range(10, 15):
        events.append(
            _event(
                index,
                EventType.PROCESS_MESSAGE_NOTICE,
                {"kind": "normal", "count": 1, "message_ids": [f"pmsg-{index}"]},
            )
        )
    for index in range(15, 20):
        events.append(
            _event(
                index,
                EventType.EXTERNAL_WRITE,
                {"path": f"result-{index}.json", "bytes_written": index},
            )
        )

    projection = project_prompt_events(
        events,
        context_object_name="llm_context_pid-test",
        payload_max_chars=2_048,
    )
    summary = projection.summary

    assert summary["input_event_count"] == 20
    assert summary["represented_event_count"] == 10
    assert summary["omitted_event_count"] == 10
    assert summary["represented_event_count"] + summary["omitted_event_count"] == 20
    assert sum(summary["omitted_reason_counts"].values()) == 10
    assert summary["omitted_events_sha256"]
    assert summary["input_event_ids_sha256"]
    assert projection.resource_usage_delta == {"tool_calls": 5}
    model_summary = projection.model_records[-1]
    assert model_summary["type"] == "event_projection_summary"
    assert model_summary["payload"]["resource_usage_delta"] == {"tool_calls": 5}


def test_resource_usage_summary_stays_sparse_and_drops_zero_totals() -> None:
    events = [
        _event(
            1,
            EventType.RESOURCE_CHARGED,
            {"usage": {"tool_calls": 0, "llm_calls": 2}},
        ),
        _event(
            2,
            EventType.RESOURCE_CHARGED,
            {"usage": {"tool_calls": 0, "llm_calls": -2}},
        ),
    ]

    projection = project_prompt_events(events)

    assert projection.resource_usage_delta == {}
    assert "resource_usage_delta" not in projection.model_records[-1]["payload"]


def test_denied_data_flow_projection_drops_source_payloads() -> None:
    sentinel = "DENIED_SOURCE_PAYLOAD_SENTINEL"
    event = _event(
        1,
        EventType.DATA_FLOW_DECISION,
        {
            "decision_id": "flow-deny",
            "direction": "egress",
            "outcome": "deny",
            "reason": "sink is not trusted",
            "sink": "llm:test",
            "labels": {"sensitivity": "secret"},
            "source_refs": [{"payload": sentinel}],
            "payload_sha256": "f" * 64,
        },
    )

    projection = project_prompt_events(
        [event],
        context_object_name="llm_context_pid-test",
        payload_max_chars=2_048,
    )
    payload = projection.visible_records[0]["payload"]

    assert payload == {
        "decision_id": "flow-deny",
        "direction": "egress",
        "outcome": "deny",
        "reason": "sink is not trusted",
        "sink": "llm:test",
        "labels": {"sensitivity": "secret"},
    }
    assert sentinel not in projection.canonical_json()
