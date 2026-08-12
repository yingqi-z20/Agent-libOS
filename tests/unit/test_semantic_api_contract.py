from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.api.gui.server import GuiRequestHandler, GuiServerError
from agent_libos.models import HumanRequestStatus


class _HumanFenceRecorder:
    def __init__(self, current: Any, terminal: Any) -> None:
        self.current = current
        self.terminal = terminal
        self.calls: list[dict[str, Any]] = []

    def get(self, _request_id: str) -> Any:
        return self.current

    def approve_for_presentation(self, request_id: str, **kwargs: Any) -> Any:
        self.calls.append({"request_id": request_id, **kwargs})
        return self.terminal


class _SchedulerRecorder:
    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []

    def maybe_start(self, **kwargs: Any) -> None:
        self.started.append(kwargs)

    def status(self) -> dict[str, bool]:
        return {"running": False}


def test_gui_external_approval_forwards_display_time_fence_and_returns_policy_conflict() -> None:
    current = SimpleNamespace(
        request_id="request-1",
        status=HumanRequestStatus.PENDING,
        payload={"type": "external_operation_approval"},
        decision=None,
    )
    terminal = SimpleNamespace(
        request_id="request-1",
        status=HumanRequestStatus.REJECTED,
        payload={"type": "external_operation_approval"},
        decision={
            "source": "machine_policy",
            "settlement_receipt": {"settlement_id": "settlement-1"},
        },
    )
    human = _HumanFenceRecorder(current, terminal)
    scheduler = _SchedulerRecorder()
    published: list[str] = []
    service = SimpleNamespace(
        runtime=SimpleNamespace(human=human),
        scheduler=scheduler,
        publish_runtime_changes=published.append,
        human_request_view=lambda request: {
            "request_id": request.request_id,
            "status": request.status.value,
            "decision": request.decision,
        },
    )
    handler = object.__new__(GuiRequestHandler)
    handler.server = SimpleNamespace(service=service)
    handler._read_body = lambda: {
        "approved": True,
        "expected_revision": 4,
        "preview_sha256": "a" * 64,
        "auto_run": True,
        "max_quanta": 2,
    }

    with pytest.raises(GuiServerError) as raised:
        handler._respond_to_human_request("request-1")

    assert raised.value.status == 409
    assert raised.value.details == {
        "code": "semantic_policy_rejected",
        "request": {
            "request_id": "request-1",
            "status": "rejected",
            "decision": terminal.decision,
        },
    }
    assert human.calls == [
        {
            "request_id": "request-1",
            "presentation": "gui",
            "decision": {"approved": True, "source": "gui"},
            "expected_revision": 4,
            "preview_sha256": "a" * 64,
        }
    ]
    assert published == ["human.respond"]
    assert scheduler.started == [
        {"max_quanta": 2, "reason": "human:request-1"}
    ]


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("expected_revision", True, "invalid_human_request_revision"),
        ("expected_revision", -1, "invalid_human_request_revision"),
        ("preview_sha256", "not-a-digest", "invalid_human_request_preview"),
    ),
)
def test_gui_external_approval_rejects_malformed_fence_before_human_mutation(
    field: str,
    value: Any,
    code: str,
) -> None:
    current = SimpleNamespace(
        request_id="request-1",
        status=HumanRequestStatus.PENDING,
        payload={"type": "external_operation_approval"},
        decision=None,
    )
    human = _HumanFenceRecorder(current, current)
    service = SimpleNamespace(
        runtime=SimpleNamespace(human=human),
        scheduler=_SchedulerRecorder(),
    )
    handler = object.__new__(GuiRequestHandler)
    handler.server = SimpleNamespace(service=service)
    body = {
        "approved": True,
        "expected_revision": 0,
        "preview_sha256": "a" * 64,
        field: value,
    }
    handler._read_body = lambda: body

    with pytest.raises(GuiServerError) as raised:
        handler._respond_to_human_request("request-1")

    assert raised.value.status == 400
    assert raised.value.details["code"] == code
    assert human.calls == []
