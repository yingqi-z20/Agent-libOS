from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig
from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    DataSensitivity,
    EventType,
    HumanRequestStatus,
    ProcessSignal,
    ProcessStatus,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError


def _reservation_rows(runtime: Runtime) -> list[dict[str, object]]:
    return runtime.store.select_table_rows(
        "capability_use_reservations",
        order_by="reservation_id",
    )


def _requesting_process(runtime: Runtime) -> str:
    return runtime.process.spawn(
        image="review-agent:v0",
        goal="exercise atomic Human authority",
        authority_manifest={
            "authorized_capabilities": [],
            "approval_policy": {
                "requestable_capabilities": [
                    {
                        "resource": "filesystem:workspace:*",
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ]
            },
        },
    )


@pytest.mark.parametrize("request_kind", ["ask", "permission"])
@pytest.mark.parametrize("failed_sink", ["return_false", "audit", "evidence"])
def test_human_request_reservation_settlement_failure_rolls_back_composite_unit(
    monkeypatch: pytest.MonkeyPatch,
    request_kind: str,
    failed_sink: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = _requesting_process(runtime)
        authority = runtime.capability.grant_once(
            pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        before_reservations = _reservation_rows(runtime)
        if failed_sink == "return_false":
            monkeypatch.setattr(
                runtime.capability,
                "commit_reserved_use",
                lambda *_args, **_kwargs: False,
            )
            expected_error: type[BaseException] = CapabilityDenied
            expected_message = "could not be committed"
        elif failed_sink == "audit":
            original_record = runtime.audit.record

            def fail_after_settlement_audit(*args: Any, **kwargs: Any) -> object:
                record = original_record(*args, **kwargs)
                if kwargs.get("action") == "capability.commit_reserved_use":
                    raise RuntimeError("injected Human settlement audit failure")
                return record

            monkeypatch.setattr(runtime.audit, "record", fail_after_settlement_audit)
            expected_error = RuntimeError
            expected_message = "settlement audit failure"
        else:
            original_link = runtime.operations.link_evidence

            def fail_after_settlement_evidence(
                evidence_type: str,
                evidence_id: str,
                role: str,
                **kwargs: Any,
            ) -> object:
                result = original_link(
                    evidence_type,
                    evidence_id,
                    role,
                    **kwargs,
                )
                if evidence_type == "capability_reservation" and role == "result":
                    raise RuntimeError("injected Human settlement evidence failure")
                return result

            monkeypatch.setattr(
                runtime.operations,
                "link_evidence",
                fail_after_settlement_evidence,
            )
            expected_error = RuntimeError
            expected_message = "settlement evidence failure"

        with pytest.raises(expected_error, match=expected_message):
            if request_kind == "ask":
                runtime.human.ask(pid, "Should this request roll back?")
            else:
                runtime.human.request_permission(
                    pid,
                    "owner",
                    "filesystem:workspace:agent_outputs/atomic.txt",
                    [CapabilityRight.WRITE.value],
                    "test atomic settlement",
                )

        current = runtime.store.get_capability(authority.cap_id)
        assert current is not None and current.uses_remaining == 1
        assert current.active
        assert runtime.human.list(pid) == []
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert _reservation_rows(runtime) == before_reservations
        assert not [
            event
            for event in runtime.events.list(target=pid)
            if event.type == EventType.HUMAN_QUERY
        ]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "signal",
    [ProcessSignal.PAUSE, ProcessSignal.CANCEL, ProcessSignal.TERMINATE],
)
@pytest.mark.parametrize("failed_sink", ["event", "audit"])
def test_human_interrupt_evidence_failure_rolls_back_state_cancellation_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
    signal: ProcessSignal,
    failed_sink: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="atomic Human cancel")
        request_id = runtime.human.query(
            pid,
            "owner",
            {"type": "question", "question": "Remain pending?"},
            blocking=True,
        )
        before_event_ids = {event.event_id for event in runtime.events.list()}
        before_audit_ids = {record.record_id for record in runtime.audit.trace()}
        if failed_sink == "event":
            original_emit = runtime.events.emit

            def fail_after_interrupt_event(*args: Any, **kwargs: Any) -> object:
                event = original_emit(*args, **kwargs)
                event_type = args[0] if args else kwargs.get("event_type")
                if EventType(event_type) == EventType.PROCESS_SIGNAL:
                    raise RuntimeError("injected Human interrupt event failure")
                return event

            monkeypatch.setattr(runtime.events, "emit", fail_after_interrupt_event)
        else:
            original_record = runtime.audit.record

            def fail_after_interrupt_audit(*args: Any, **kwargs: Any) -> object:
                record = original_record(*args, **kwargs)
                if kwargs.get("action") == "human.interrupt":
                    raise RuntimeError("injected Human interrupt audit failure")
                return record

            monkeypatch.setattr(runtime.audit, "record", fail_after_interrupt_audit)

        with pytest.raises(RuntimeError, match=f"interrupt {failed_sink} failure"):
            runtime.human.interrupt(
                pid,
                signal,
                {"reason": "must roll back"},
            )

        assert runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
        assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        assert {event.event_id for event in runtime.events.list()} == before_event_ids
        assert {record.record_id for record in runtime.audit.trace()} == before_audit_ids
    finally:
        runtime.close()


def test_human_interrupt_commits_state_cancellation_and_evidence_together() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="Human cancel control")
        request_id = runtime.human.query(
            pid,
            "owner",
            {"type": "question", "question": "Cancel this request?"},
            blocking=True,
        )

        event_id = runtime.human.interrupt(
            pid,
            ProcessSignal.CANCEL,
            {"reason": "approved control"},
        )

        assert runtime.process.get(pid).status == ProcessStatus.KILLED
        assert runtime.human.get(request_id).status == HumanRequestStatus.CANCELLED
        assert any(
            event.event_id == event_id and event.type == EventType.PROCESS_SIGNAL
            for event in runtime.events.list(target=pid)
        )
        assert any(
            record.action == "human.interrupt"
            for record in runtime.audit.trace(target=f"process:{pid}")
        )
    finally:
        runtime.close()


def test_human_message_audit_failure_cannot_raise_after_durable_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="atomic Human message")
        before_event_ids = {event.event_id for event in runtime.events.list()}
        before_audit_ids = {record.record_id for record in runtime.audit.trace()}
        original_record = runtime.audit.record

        def fail_after_human_message_audit(*args: Any, **kwargs: Any) -> object:
            record = original_record(*args, **kwargs)
            if kwargs.get("action") == "human.message":
                raise RuntimeError("injected Human message audit failure")
            return record

        monkeypatch.setattr(runtime.audit, "record", fail_after_human_message_audit)

        with pytest.raises(RuntimeError, match="message audit failure"):
            runtime.human.send_process_message(pid, "must not become visible")

        assert runtime.messages.unread(pid) == []
        assert {event.event_id for event in runtime.events.list()} == before_event_ids
        assert {record.record_id for record in runtime.audit.trace()} == before_audit_ids
    finally:
        runtime.close()


def test_human_message_success_commits_delivery_and_both_evidence_records() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="Human message control")

        message = runtime.human.send_process_message(pid, "committed message")

        assert [item.message_id for item in runtime.messages.unread(pid)] == [
            message.message_id
        ]
        actions = {record.action for record in runtime.audit.trace()}
        assert {"process.message.post", "human.message"} <= actions
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("tool_overrides", "decision", "error"),
    [
        (
            {"human_response_payload_max_bytes": 64},
            {"approved": True, "answer": "x" * 80},
            "exceeds 64 bytes",
        ),
        (
            {"human_response_max_depth": 2},
            {"approved": True, "answer": "yes", "extra": {"nested": {}}},
            "maximum JSON depth=2",
        ),
        (
            {"human_response_max_nodes": 5},
            {"approved": True, "answer": "yes", "extra": [1, 2, 3]},
            "maximum JSON nodes=5",
        ),
        (
            {},
            {"approved": True, "answer": "yes", "extra": {1: "value"}},
            "must use string JSON object keys",
        ),
        (
            {},
            {"approved": True, "answer": "yes", "extra": float("nan")},
            "contains a non-finite JSON number",
        ),
        (
            {},
            {"approved": True, "answer": "yes", "extra": {"not-json"}},
            "contains a non-JSON value: set",
        ),
    ],
)
def test_human_decision_limits_reject_before_request_or_capability_side_effects(
    tool_overrides: dict[str, int],
    decision: dict[str, object],
    error: str,
) -> None:
    base = AgentLibOSConfig()
    runtime = Runtime.open(
        "local",
        config=replace(base, tools=replace(base.tools, **tool_overrides)),
    )
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bounded decision")
        resource = "object:bounded-human-decision"
        request_id = runtime.human.query(
            pid,
            "owner",
            {
                "type": "question",
                "question": "Approve bounded authority?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [CapabilityRight.READ.value],
                },
            },
            blocking=True,
        )

        with pytest.raises(ValidationError, match=error):
            runtime.human.approve(request_id, decision)

        assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        assert runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
        assert not runtime.capability.check(pid, resource, CapabilityRight.READ)
    finally:
        runtime.close()


def test_bounded_human_decision_control_still_applies_approved_side_effects() -> None:
    base = AgentLibOSConfig()
    runtime = Runtime.open(
        "local",
        config=replace(
            base,
            tools=replace(
                base.tools,
                human_response_payload_max_bytes=128,
                human_response_max_depth=3,
                human_response_max_nodes=16,
            ),
        ),
    )
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bounded decision control")
        resource = "object:bounded-human-decision-control"
        request_id = runtime.human.query(
            pid,
            "owner",
            {
                "type": "question",
                "question": "Approve bounded authority?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [CapabilityRight.READ.value],
                },
            },
            blocking=True,
        )

        approved = runtime.human.approve(
            request_id,
            {"approved": True, "answer": "yes"},
        )

        assert approved.status == HumanRequestStatus.APPROVED
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert runtime.capability.check(pid, resource, CapabilityRight.READ)
    finally:
        runtime.close()


def test_oversized_terminal_provider_answer_is_rejected_without_approval() -> None:
    base = AgentLibOSConfig()
    runtime = Runtime.open(
        "local",
        config=replace(
            base,
            tools=replace(base.tools, human_response_payload_max_bytes=64),
        ),
    )
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bounded provider answer")
        resource = "object:oversized-provider-answer"
        request_id = runtime.human.query(
            pid,
            "owner",
            {
                "type": "question",
                "question": "Return a bounded answer?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [CapabilityRight.READ.value],
                },
            },
            blocking=True,
        )
        oversized_answer = "x" * 80
        runtime.substrate.human.input_reader = lambda _prompt: oversized_answer

        with pytest.raises(ValidationError, match="human provider response exceeds 64 bytes"):
            runtime.human.process_next_terminal()

        request = runtime.human.get(request_id)
        assert request.status == HumanRequestStatus.PENDING
        assert request.decision is None
        assert runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
        assert not runtime.capability.check(pid, resource, CapabilityRight.READ)
        protected_read_evidence = json.dumps(
            {
                "events": [
                    event.payload
                    for event in runtime.events.list()
                    if event.type == EventType.HUMAN_RESPONSE
                ],
                "audit": [
                    record.decision
                    for record in runtime.audit.trace()
                    if record.action == "human.terminal.read"
                ],
                "effects": [
                    effect.provider_metadata
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.provider == "human"
                ],
            },
            sort_keys=True,
        )
        assert "response_bounds" in protected_read_evidence
        assert oversized_answer not in protected_read_evidence

        runtime.substrate.human.input_reader = lambda _prompt: "yes"
        approved = runtime.human.process_next_terminal()
        assert approved is not None and approved.status == HumanRequestStatus.APPROVED
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert runtime.capability.check(pid, resource, CapabilityRight.READ)
    finally:
        runtime.close()


@pytest.mark.parametrize("provider_error", [False, True])
def test_output_delivery_tail_preserves_concurrent_request_metadata(
    provider_error: bool,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="preserve output metadata")
        runtime.capability.grant(
            pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )

        def interleaving_sink(_message: str) -> None:
            output_request = next(
                request
                for request in runtime.human.list(pid)
                if request.payload.get("type") == "output"
            )
            output_request.payload = dict(output_request.payload)
            output_request.payload["_agent_libos_data_release_visible"] = {
                "gui": {"release_request_id": "release-concurrent"}
            }
            runtime.store.update_human_request(output_request)
            if provider_error:
                raise RuntimeError("injected output provider failure")

        runtime.substrate.human.output_sink = interleaving_sink
        if provider_error:
            with pytest.raises(RuntimeError, match="output provider failure"):
                runtime.human.output(pid, "deliver once")
        else:
            runtime.human.output(pid, "deliver once")

        persisted = next(
            request
            for request in runtime.human.list(pid)
            if request.payload.get("type") == "output"
        )
        assert persisted.payload["_agent_libos_data_release_visible"] == {
            "gui": {"release_request_id": "release-concurrent"}
        }
    finally:
        runtime.close()


def test_ambiguous_provider_marker_failure_uses_minimal_retry_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="fence ambiguous Human I/O")
        request_id = runtime.human.query(
            pid,
            "owner",
            {"type": "question", "question": "Ask exactly once"},
            blocking=True,
        )
        provider_calls = 0

        def fail_provider(_prompt: str) -> str:
            nonlocal provider_calls
            provider_calls += 1
            raise RuntimeError("ambiguous provider failure")

        runtime.substrate.human.input_reader = fail_provider
        original_update = runtime.human.requests.update
        failed_marker = False

        def fail_first_unknown_marker(request: Any) -> None:
            nonlocal failed_marker
            if (
                not failed_marker
                and request.status == HumanRequestStatus.CANCELLED
                and (request.decision or {}).get("provider_outcome") == "unknown"
            ):
                failed_marker = True
                raise RuntimeError("transient marker persistence failure")
            original_update(request)

        monkeypatch.setattr(runtime.human.requests, "update", fail_first_unknown_marker)

        with pytest.raises(RuntimeError, match="ambiguous provider failure"):
            runtime.human.process_next_terminal()

        persisted = runtime.human.get(request_id)
        assert failed_marker is True
        assert persisted.status == HumanRequestStatus.CANCELLED
        assert persisted.decision is not None
        assert persisted.decision["automatic_retry_disabled"] is True
        assert persisted.decision["process_reconciliation_required"] is False
        assert runtime.process.get(pid).status == ProcessStatus.PAUSED
        assert runtime.human.process_next_terminal() is None
        assert provider_calls == 1
    finally:
        runtime.close()


def test_terminal_question_fallback_excludes_host_only_payload_metadata() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="format safe fallback")
        request_id = runtime.human.query(
            pid,
            "owner",
            {"type": "approval", "question": ""},
            blocking=False,
        )

        rendered = runtime.human.format_terminal_request(runtime.human.get(request_id))

        assert "_agent_libos_data_flow_context" not in rendered
        assert json.loads(rendered)["type"] == "approval"
    finally:
        runtime.close()


def test_reserved_data_release_does_not_unredact_before_terminal_settlement() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="withhold reserved release")
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity=DataSensitivity.SECRET,
            ),
            actor="test",
            require_capability=False,
        )
        secret = "RESERVED_RELEASE_SECRET"
        request_id = runtime.human.query(
            pid,
            human,
            {"type": "question", "question": secret},
            blocking=False,
        )
        parent = runtime.human.get(request_id)
        parent.payload = dict(parent.payload)
        parent.payload["_agent_libos_data_flow_context"] = DataFlowContext(
            labels=DataLabels(sensitivity=DataSensitivity.SECRET)
        ).to_dict()
        runtime.store.update_human_request(parent)

        release_resource = f"data_release:test:{request_id}"
        token = runtime.human._data_release_parent_request.set(request_id)
        try:
            release_id = runtime.human.request_data_release(
                pid=pid,
                human=human,
                request={
                    "type": "data_release_approval",
                    "question": "Release exact Human request?",
                    "context": {"sink": f"human:{human}:{channel}"},
                    "requested_once_capability": {
                        "subject": pid,
                        "resource": release_resource,
                        "rights": [CapabilityRight.EXECUTE.value],
                        "constraints": {},
                    },
                },
                blocking=False,
            )
        finally:
            runtime.human._data_release_parent_request.reset(token)
        runtime.human.approve(release_id, {"approved": True})
        release_capability = next(
            capability
            for capability in runtime.capability.capabilities_for(pid)
            if capability.resource == release_resource
        )
        reservation_id = runtime.capability.reserve_use(
            release_capability.cap_id,
            reserved_by=pid,
            reason="simulate in-flight terminal release",
        )

        reserved_view = runtime.human.public_request_view(
            runtime.human.get(request_id)
        )
        assert secret not in json.dumps(reserved_view, sort_keys=True)
        reservation = runtime.store.get_capability_use_reservation(reservation_id)
        assert reservation is not None and reservation["status"] == "reserved"

        assert runtime.capability.commit_reserved_use(
            reservation_id,
            committed_by=pid,
            reason="simulate completed terminal release",
        )
        runtime.human._mark_terminal_release_completed(request_id)
        completed_view = runtime.human.public_request_view(
            runtime.human.get(request_id)
        )
        assert completed_view["payload"]["question"] == secret
    finally:
        runtime.close()


def test_scoped_terminal_drain_never_settles_requests_from_other_processes() -> None:
    runtime = Runtime.open("local")
    try:
        outside_pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="remain outside scoped Human drain",
        )
        inside_pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="run inside scoped Human drain",
        )
        outside_id = runtime.human.query(
            outside_pid,
            "owner",
            {"type": "question", "question": "outside"},
            blocking=False,
        )
        inside_id = runtime.human.query(
            inside_pid,
            "owner",
            {"type": "question", "question": "inside"},
            blocking=False,
        )

        processed = runtime.human.drain_terminal_queue(
            auto_answer="inside-answer",
            pids=frozenset({inside_pid}),
        )

        assert [request.request_id for request in processed] == [inside_id]
        assert runtime.human.get(inside_id).status == HumanRequestStatus.APPROVED
        assert runtime.human.get(outside_id).status == HumanRequestStatus.PENDING

        remaining = runtime.human.drain_terminal_queue(auto_answer="outside-answer")
        assert [request.request_id for request in remaining] == [outside_id]
    finally:
        runtime.close()
