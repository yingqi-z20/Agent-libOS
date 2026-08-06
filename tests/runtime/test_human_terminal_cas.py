from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.models import (
    CapabilityRight,
    EventType,
    HumanRequestStatus,
    ProcessStatus,
)
from agent_libos.models.exceptions import ValidationError


def _pending_approval(runtime: Runtime, *, with_authority: bool = False) -> tuple[str, str]:
    pid = runtime.process.spawn(
        image="base-agent:v0",
        goal="exercise Human terminal revision CAS",
    )
    payload: dict[str, Any] = {"type": "approval", "reason": "CAS test"}
    if with_authority:
        payload["requested_once_capability"] = {
            "subject": pid,
            "resource": "object:human-cas-authority",
            "rights": [CapabilityRight.READ.value],
        }
    request_id = runtime.human.query(pid, "owner", payload, blocking=True)
    return pid, request_id


def test_concurrent_human_approval_and_rejection_have_one_terminal_winner() -> None:
    runtime = Runtime.open("local")
    try:
        pid, request_id = _pending_approval(runtime)
        barrier = threading.Barrier(2)

        def settle(approved: bool) -> tuple[str, object]:
            barrier.wait(timeout=2)
            try:
                if approved:
                    return "ok", runtime.human.approve(
                        request_id,
                        {"approved": True},
                        responder="human:gui",
                    )
                return "ok", runtime.human.reject(
                    request_id,
                    {"approved": False},
                    responder="human:terminal",
                )
            except Exception as exc:  # the loser is asserted below
                return "error", exc

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(settle, (True, False)))

        assert [kind for kind, _value in outcomes].count("ok") == 1
        errors = [value for kind, value in outcomes if kind == "error"]
        assert len(errors) == 1
        assert isinstance(errors[0], ValidationError)
        assert "not pending" in str(errors[0]) or "changed concurrently" in str(errors[0])

        persisted = runtime.human.get(request_id)
        assert persisted.status in {
            HumanRequestStatus.APPROVED,
            HumanRequestStatus.REJECTED,
        }
        assert persisted.revision == 1
        assert runtime.process.get(pid).status in {
            ProcessStatus.RUNNABLE,
            ProcessStatus.PAUSED,
        }
        assert len(
            [
                event
                for event in runtime.events.list(target=pid)
                if event.type == EventType.HUMAN_RESPONSE
                and event.payload.get("request_id") == request_id
            ]
        ) == 1
        assert len(
            [
                record
                for record in runtime.audit.trace(target=f"human_request:{request_id}")
                if record.action == "human.response"
            ]
        ) == 1
    finally:
        runtime.close()


def test_gui_cli_terminal_and_cancel_race_has_one_request_terminal_winner() -> None:
    runtime = Runtime.open("local")
    try:
        pid, request_id = _pending_approval(runtime)
        barrier = threading.Barrier(4)

        def settle(channel: str) -> tuple[str, bool, object]:
            barrier.wait(timeout=3)
            try:
                if channel == "gui":
                    value = runtime.human.approve_for_presentation(
                        request_id,
                        presentation="gui",
                        decision={"approved": True, "source": "gui"},
                        responder="human:gui",
                    )
                    return channel, value.request_id == request_id, value
                if channel == "cli":
                    value = runtime.human.reject(
                        request_id,
                        {"approved": False, "source": "cli"},
                        responder="human:cli",
                    )
                    return channel, value.request_id == request_id, value
                if channel == "terminal":
                    value = runtime.human.process_next_terminal(auto_approve=True)
                    return (
                        channel,
                        value is not None and value.request_id == request_id,
                        value,
                    )
                runtime.human.interrupt(
                    pid,
                    "cancel",
                    {"reason": "concurrent Host cancellation"},
                )
                value = runtime.human.get(request_id)
                return (
                    channel,
                    value.status is HumanRequestStatus.CANCELLED,
                    value,
                )
            except Exception as exc:
                return channel, False, exc

        with ThreadPoolExecutor(max_workers=4) as executor:
            outcomes = list(
                executor.map(settle, ("gui", "cli", "terminal", "cancel"))
            )

        assert sum(transitioned for _channel, transitioned, _value in outcomes) == 1
        persisted = runtime.human.get(request_id)
        assert persisted.status in {
            HumanRequestStatus.APPROVED,
            HumanRequestStatus.REJECTED,
            HumanRequestStatus.CANCELLED,
        }
        assert persisted.revision == 1
        response_events = [
            event
            for event in runtime.events.list(target=pid)
            if event.type == EventType.HUMAN_RESPONSE
            and event.payload.get("request_id") == request_id
        ]
        response_audits = [
            record
            for record in runtime.audit.trace(target=f"human_request:{request_id}")
            if record.action in {"human.response", "human.request_cancelled"}
        ]
        assert len(response_events) == 1
        assert len(response_audits) == 1
    finally:
        runtime.close()


def test_persisted_request_capture_is_failure_isolated() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="isolate Human request capture",
        )
        observed: list[tuple[str, int, HumanRequestStatus]] = []

        def fail_after_observation(request: Any) -> None:
            observed.append((request.request_id, request.revision, request.status))
            raise RuntimeError("capture is unavailable")

        runtime.human.set_request_capture(fail_after_observation)
        request_id = runtime.human.query(
            pid,
            "owner",
            {"type": "approval", "reason": "capture must not affect Human"},
            blocking=True,
        )

        assert observed == [(request_id, 0, HumanRequestStatus.PENDING)]
        assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        assert runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
    finally:
        runtime.close()


@pytest.mark.parametrize("failed_step", ["request_cas", "process", "event", "audit"])
def test_human_terminal_failure_rolls_back_request_authority_wait_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
    failed_step: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid, request_id = _pending_approval(runtime, with_authority=True)
        before_event_ids = {event.event_id for event in runtime.events.list()}
        before_audit_ids = {record.record_id for record in runtime.audit.trace()}

        if failed_step == "request_cas":
            monkeypatch.setattr(
                runtime.human.requests._processes,
                "compare_and_set_human_request",
                lambda _expected, _target: False,
            )
            expected_error = "changed concurrently"
        elif failed_step == "process":
            monkeypatch.setattr(
                runtime.human._transitions,
                "transition",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("injected process transition failure")
                ),
            )
            expected_error = "process transition failure"
        elif failed_step == "event":
            original_emit = runtime.events.emit

            def fail_after_event(*args: Any, **kwargs: Any) -> object:
                result = original_emit(*args, **kwargs)
                event_type = args[0] if args else kwargs.get("event_type")
                if EventType(event_type) == EventType.HUMAN_RESPONSE:
                    raise RuntimeError("injected response event failure")
                return result

            monkeypatch.setattr(runtime.events, "emit", fail_after_event)
            expected_error = "response event failure"
        else:
            original_record = runtime.audit.record

            def fail_after_audit(*args: Any, **kwargs: Any) -> object:
                result = original_record(*args, **kwargs)
                if kwargs.get("action") == "human.response":
                    raise RuntimeError("injected response audit failure")
                return result

            monkeypatch.setattr(runtime.audit, "record", fail_after_audit)
            expected_error = "response audit failure"

        with pytest.raises((RuntimeError, ValidationError), match=expected_error):
            runtime.human.approve(request_id, {"approved": True})

        persisted = runtime.human.get(request_id)
        assert persisted.status == HumanRequestStatus.PENDING
        assert persisted.revision == 0
        assert persisted.decision is None
        assert runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
        assert not runtime.capability.check(
            pid,
            "object:human-cas-authority",
            CapabilityRight.READ,
        )
        assert {event.event_id for event in runtime.events.list()} == before_event_ids
        assert {record.record_id for record in runtime.audit.trace()} == before_audit_ids
    finally:
        runtime.close()


@pytest.mark.parametrize("failed_sink", ["event", "audit"])
def test_permission_policy_response_failure_rolls_back_exact_process_and_authority_state(
    monkeypatch: pytest.MonkeyPatch,
    failed_sink: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="rollback Human permission policy")
        resource = "object:human-permission-policy-rollback"
        request_id = runtime.human.query(
            pid,
            "owner",
            {
                "type": "permission_request",
                "reason": "exercise permission side-effect rollback",
                "requested_permission": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [CapabilityRight.READ.value],
                    "constraints": {},
                },
            },
            blocking=True,
        )
        before_request = runtime.human.get(request_id)
        before_process = runtime.process.get(pid)
        before_capabilities = tuple(runtime.capability.list_subject(pid))
        before_event_ids = {event.event_id for event in runtime.events.list()}
        before_audit_ids = {record.record_id for record in runtime.audit.trace()}

        if failed_sink == "event":
            original_emit = runtime.events.emit

            def fail_after_response_event(*args: Any, **kwargs: Any) -> object:
                result = original_emit(*args, **kwargs)
                event_type = args[0] if args else kwargs.get("event_type")
                if EventType(event_type) == EventType.HUMAN_RESPONSE:
                    raise RuntimeError("permission response event failure")
                return result

            monkeypatch.setattr(runtime.events, "emit", fail_after_response_event)
        else:
            original_record = runtime.audit.record

            def fail_after_response_audit(*args: Any, **kwargs: Any) -> object:
                result = original_record(*args, **kwargs)
                if kwargs.get("action") == "human.response":
                    raise RuntimeError("permission response audit failure")
                return result

            monkeypatch.setattr(runtime.audit, "record", fail_after_response_audit)

        with pytest.raises(RuntimeError, match=f"permission response {failed_sink} failure"):
            runtime.human.approve(
                request_id,
                {
                    "approved": True,
                    "policy": CapabilityManager.ALWAYS_ALLOW,
                },
            )

        assert runtime.human.get(request_id) == before_request
        assert runtime.process.get(pid) == before_process
        assert tuple(runtime.capability.list_subject(pid)) == before_capabilities
        assert not runtime.capability.check(
            pid,
            resource,
            CapabilityRight.READ,
        )
        assert {event.event_id for event in runtime.events.list()} == before_event_ids
        assert {record.record_id for record in runtime.audit.trace()} == before_audit_ids
    finally:
        runtime.close()
