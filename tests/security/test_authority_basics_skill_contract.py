from __future__ import annotations

import pytest

from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.models import CapabilityRight, EventType


def test_unbridged_clock_ask_is_denied_without_human_wait() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="verify unbridged ASK handling",
        )
        runtime.tools.configure_process_tools(
            pid,
            ["get_current_time"],
            assigned_by="test",
        )
        runtime.capability.set_permission_policy(
            subject=pid,
            resource="clock:now",
            rights=[CapabilityRight.READ],
            policy=CapabilityManager.ASK_EACH_TIME,
            issued_by="test.host",
        )

        result = runtime.tools.call(
            pid,
            "get_current_time",
            {"timezone": "UTC"},
        )

        assert result.ok is False
        assert (result.error or "").startswith(
            "permission_denied: CapabilityDenied"
        )
        assert runtime.human.pending() == []
        decisions = [
            record
            for record in runtime.audit.trace()
            if record.action == "capability.authorize"
            and record.target == "clock:now"
        ]
        assert decisions
        assert decisions[-1].decision["effect"] == "ask"
        assert decisions[-1].decision["allowed"] is False
    finally:
        runtime.close()


def test_permission_request_queue_commit_consumes_finite_human_write() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="review-agent:v0",
            goal="verify finite Human write settlement",
            authority_manifest={
                "authorized_capabilities": [],
                "approval_policy": {
                    "requestable_capabilities": [
                        {
                            "resource": "object:permission-target",
                            "rights": [CapabilityRight.READ.value],
                        }
                    ]
                },
            },
        )
        human_write = runtime.capability.grant_once(
            pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        request_id = runtime.human.request_permission(
            pid,
            "owner",
            "object:permission-target",
            [CapabilityRight.READ.value],
            "read the requested object",
            blocking=False,
        )

        queued = runtime.store.get_capability(human_write.cap_id)
        assert queued is not None and queued.uses_remaining == 0
        assert any(
            event.type == EventType.HUMAN_QUERY
            and event.source == pid
            and event.payload.get("request_id") == request_id
            for event in runtime.events.list()
        )

        runtime.human.reject(
            request_id,
            {
                "approved": False,
                "policy": CapabilityManager.ALWAYS_DENY,
            },
        )

        rejected = runtime.store.get_capability(human_write.cap_id)
        assert rejected is not None and rejected.uses_remaining == 0
    finally:
        runtime.close()
