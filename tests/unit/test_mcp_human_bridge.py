from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.mcp.human import (
    HumanObjectManagerMcpBridge,
    mcp_human_preview,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError


class _HumanManager:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            runtime=SimpleNamespace(default_human="owner")
        )
        self.ask_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.request: Any | None = None

    def ask(self, owner_id: str, question: str, **kwargs: Any) -> str:
        self.ask_calls.append(
            {"owner_id": owner_id, "question": question, **kwargs}
        )
        raise CapabilityDenied(f"{owner_id} lacks write on human:owner")

    def query(self, **kwargs: Any) -> str:
        self.query_calls.append(kwargs)
        request = kwargs["request"]
        self.request = SimpleNamespace(
            request_id="hreq-host",
            revision=0,
            status="pending",
            payload=request,
            decision=None,
        )
        return "hreq-host"

    def get(self, request_id: str) -> Any:
        assert request_id == "hreq-host"
        return self.request


def _preview() -> tuple[dict[str, Any], str]:
    return mcp_human_preview(
        server_id="modern",
        operation="resources/read",
        local_ref="continuation-local",
        input_requests=(),
    )


def test_mcp_host_question_requires_active_protected_authorizer() -> None:
    manager = _HumanManager()
    bridge = HumanObjectManagerMcpBridge(manager)
    preview, digest = _preview()

    with pytest.raises(ValidationError, match="authorizer is unavailable"):
        bridge.create_question(
            owner_id="gui",
            server_id="modern",
            operation="resources/read",
            local_ref="continuation-local",
            preview=preview,
            preview_sha256=digest,
            expires_at=None,
        )
    assert manager.ask_calls == []
    assert manager.query_calls == []


def test_mcp_host_question_uses_only_preview_bound_authorized_query() -> None:
    manager = _HumanManager()
    authorizations: list[dict[str, Any]] = []

    def authorize(**kwargs: Any) -> None:
        authorizations.append(kwargs)

    bridge = HumanObjectManagerMcpBridge(
        manager,
        host_question_authorizer=authorize,
    )
    preview, digest = _preview()

    receipt = bridge.create_question(
        owner_id="gui",
        server_id="modern",
        operation="resources/read",
        local_ref="continuation-local",
        preview=preview,
        preview_sha256=digest,
        expires_at=None,
    )

    assert receipt.request_id == "hreq-host"
    assert manager.ask_calls == []
    assert authorizations == [
        {
            "owner_id": "gui",
            "server_id": "modern",
            "operation": "resources/read",
            "local_ref": "continuation-local",
            "preview": preview,
            "preview_sha256": digest,
        }
    ]
    assert manager.query_calls[0]["pid"] == "gui"
    assert manager.query_calls[0]["human"] == "owner"
    assert manager.query_calls[0]["request"]["context"]["mcp_preview"] == preview


def test_mcp_cli_question_uses_the_same_preview_bound_host_seam() -> None:
    manager = _HumanManager()
    authorizations: list[dict[str, Any]] = []
    bridge = HumanObjectManagerMcpBridge(
        manager,
        host_question_authorizer=lambda **kwargs: authorizations.append(kwargs),
    )
    preview, digest = _preview()

    receipt = bridge.create_question(
        owner_id="cli",
        server_id="modern",
        operation="resources/read",
        local_ref="continuation-local",
        preview=preview,
        preview_sha256=digest,
        expires_at=None,
    )

    assert receipt.request_id == "hreq-host"
    assert manager.ask_calls == []
    assert manager.query_calls[0]["pid"] == "cli"
    assert authorizations[0]["owner_id"] == "cli"
    assert authorizations[0]["preview_sha256"] == digest


def test_mcp_process_question_still_requires_human_write_capability() -> None:
    manager = _HumanManager()
    authorizations: list[dict[str, Any]] = []
    bridge = HumanObjectManagerMcpBridge(
        manager,
        host_question_authorizer=lambda **kwargs: authorizations.append(kwargs),
    )
    preview, digest = _preview()

    with pytest.raises(CapabilityDenied, match="lacks write"):
        bridge.create_question(
            owner_id="pid-agent",
            server_id="modern",
            operation="resources/read",
            local_ref="continuation-local",
            preview=preview,
            preview_sha256=digest,
            expires_at=None,
        )
    assert len(manager.ask_calls) == 1
    assert manager.query_calls == []
    assert authorizations == []


def test_mcp_host_authorizer_cannot_mutate_preview_before_persistence() -> None:
    manager = _HumanManager()

    def mutate_preview(**kwargs: Any) -> None:
        kwargs["preview"]["serverId"] = "changed"

    bridge = HumanObjectManagerMcpBridge(
        manager,
        host_question_authorizer=mutate_preview,
    )
    preview, digest = _preview()

    with pytest.raises(ValidationError, match="preview binding changed"):
        bridge.create_question(
            owner_id="runtime",
            server_id="modern",
            operation="resources/read",
            local_ref="continuation-local",
            preview=preview,
            preview_sha256=digest,
            expires_at=None,
        )
    assert manager.query_calls == []
