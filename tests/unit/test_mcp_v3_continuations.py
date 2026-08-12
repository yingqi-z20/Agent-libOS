from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from agent_libos.mcp.continuations import (
    McpContinuationBinding,
    McpContinuationCaptureSettlement,
    McpContinuationDispatchNotStarted,
    McpContinuationManager,
    McpContinuationStatus,
    McpSdkContinuationCaptureAdapter,
)
from agent_libos.mcp.human import McpHumanRequestReceipt
from agent_libos.mcp.types import (
    McpComplete,
    McpInputRequestKind,
    McpInputRequired,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.human import HumanRequest, HumanRequestStatus
from agent_libos.storage import SQLiteStore, UnitOfWork


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _Broker:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.enabled = True
        self.counter = 0

    def reserve_secret_ref(self, namespace: str) -> str:
        assert self.enabled
        self.counter += 1
        return f"secret-{self.counter}"

    def put_secret_at(
        self,
        secret_ref: str,
        namespace: str,
        value: bytes,
        *,
        expires_at: str | None,
    ) -> None:
        assert self.enabled
        self.values[secret_ref] = bytes(value)

    def put_secret(
        self, namespace: str, value: bytes, *, expires_at: str | None
    ) -> str:
        ref = self.reserve_secret_ref(namespace)
        self.put_secret_at(ref, namespace, value, expires_at=expires_at)
        return ref

    def get_secret(self, secret_ref: str) -> bytes:
        if not self.enabled or secret_ref not in self.values:
            raise KeyError(secret_ref)
        return self.values[secret_ref]

    def delete_secret(self, secret_ref: str) -> None:
        self.values.pop(secret_ref, None)

    def available(self) -> bool:
        return self.enabled


class _Repo:
    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}

    def insert(self, record: Any) -> None:
        if record.continuation_id in self.rows:
            raise RuntimeError("duplicate")
        self.rows[record.continuation_id] = record

    def get(self, continuation_id: str) -> Any | None:
        return self.rows.get(continuation_id)

    def list(self, **_filters: object) -> list[Any]:
        return list(self.rows.values())

    def count_active(self, *, owner_id: str | None = None) -> int:
        terminal = {"complete", "cancelled", "expired", "needs_attention"}
        return sum(
            row.status not in terminal
            and (owner_id is None or row.owner_id == owner_id)
            for row in self.rows.values()
        )

    def list_terminal(
        self, *, owner_id: str | None = None, limit: int = 100
    ) -> tuple[Any, ...]:
        terminal = {"complete", "cancelled", "expired", "needs_attention"}
        return tuple(
            row
            for row in self.rows.values()
            if row.status in terminal
            and (owner_id is None or row.owner_id == owner_id)
        )[:limit]

    def delete_terminal(
        self, continuation_id: str, *, expected_revision: int
    ) -> bool:
        current = self.rows.get(continuation_id)
        if (
            current is None
            or current.revision != expected_revision
            or current.status not in {"complete", "cancelled", "expired", "needs_attention"}
        ):
            return False
        del self.rows[continuation_id]
        return True

    def compare_and_swap(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        replacement: Any,
    ) -> bool:
        current = self.rows.get(continuation_id)
        if current is None or current.revision != expected_revision:
            return False
        self.rows[continuation_id] = replacement
        return True


class _SideEffects:
    def __init__(self, operation_repository: Any) -> None:
        self.operation_repository = operation_repository
        self.rows: dict[str, Any] = {}

    def insert(self, record: Any) -> Any:
        if record.preparation_id in self.rows:
            raise RuntimeError("duplicate preparation")
        self.rows[record.preparation_id] = record
        return record

    def get(self, preparation_id: str) -> Any | None:
        return self.rows.get(preparation_id)

    def list(self, **filters: object) -> tuple[Any, ...]:
        selected = tuple(self.rows.values())
        for name in ("operation_kind", "status"):
            value = filters.get(name)
            if value is not None:
                selected = tuple(row for row in selected if getattr(row, name) == value)
        limit = filters.get("limit", 100)
        assert type(limit) is int
        return selected[:limit]

    def compare_and_swap(
        self,
        preparation_id: str,
        *,
        expected_revision: int,
        replacement: Any,
    ) -> bool:
        current = self.rows.get(preparation_id)
        if current is None or current.revision != expected_revision:
            return False
        self.rows[preparation_id] = replacement
        return True

    def delete(self, preparation_id: str, *, expected_revision: int) -> bool:
        current = self.rows.get(preparation_id)
        if current is None or current.revision != expected_revision:
            return False
        del self.rows[preparation_id]
        return True

    def commit(
        self,
        preparation_id: str,
        *,
        expected_revision: int,
        replacement: Any,
    ) -> bool:
        preparation = self.rows.get(preparation_id)
        if preparation is None or preparation.revision != expected_revision:
            return False
        if preparation.operation_revision is None:
            self.operation_repository.insert(replacement)
        elif not self.operation_repository.compare_and_swap(
            preparation.operation_id,
            expected_revision=preparation.operation_revision,
            replacement=replacement,
        ):
            return False
        retirement_metadata = {
            "automatic_retry_disabled": True,
            "cleanup_mode": "retire",
            "retire_refs": tuple(preparation.metadata.get("retire_refs", ())),
        }
        for key in (
            "retire_human_request_id",
            "retire_human_preview_sha256",
        ):
            if key in preparation.metadata:
                retirement_metadata[key] = preparation.metadata[key]
        self.rows[preparation_id] = replace(
            preparation,
            status="cleaning",
            revision=preparation.revision + 1,
            metadata=retirement_metadata,
            updated_at=replacement.updated_at,
        )
        return True

    def commit_terminal(
        self,
        preparation_id: str,
        *,
        expected_revision: int,
    ) -> bool:
        preparation = self.rows.get(preparation_id)
        if preparation is None or preparation.revision != expected_revision:
            return False
        current = self.operation_repository.get(preparation.operation_id)
        if current is None or current.revision != preparation.operation_revision:
            return False
        if not self.operation_repository.delete_terminal(
            preparation.operation_id,
            expected_revision=current.revision,
        ):
            return False
        retirement_metadata = dict(preparation.metadata)
        retirement_metadata["cleanup_mode"] = "retire"
        self.rows[preparation_id] = replace(
            preparation,
            status="cleaning",
            revision=preparation.revision + 1,
            metadata=retirement_metadata,
        )
        return True


class _Human:
    def __init__(self, *, durable_store: SQLiteStore | None = None) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.counter = 0
        self.durable_store = durable_store

    def reserve_question_id(self) -> str:
        self.counter += 1
        return f"human-real-{self.counter}"

    def create_question(self, **kwargs: Any) -> McpHumanRequestReceipt:
        request_id = kwargs.get("request_id") or self.reserve_question_id()
        self.rows[request_id] = {
            **kwargs,
            "revision": 0,
            "status": "pending",
            "answer": None,
            "presented_revision": None,
        }
        if self.durable_store is not None:
            self.durable_store.insert_human_request(
                HumanRequest(
                    request_id=request_id,
                    pid=kwargs["owner_id"],
                    human="owner",
                    payload={"type": "question", "preview": kwargs["preview"]},
                    status=HumanRequestStatus.PENDING,
                    decision=None,
                    blocking=True,
                    created_at="2030-01-01T00:00:00+00:00",
                    updated_at="2030-01-01T00:00:00+00:00",
                )
            )
        return McpHumanRequestReceipt(request_id, 0, kwargs["preview_sha256"])

    def inspect_question(
        self, request_id: str, *, preview_sha256: str
    ) -> McpHumanRequestReceipt:
        row = self.rows[request_id]
        assert row["preview_sha256"] == preview_sha256
        return McpHumanRequestReceipt(request_id, row["revision"], preview_sha256)

    def settle(
        self,
        public: McpInputRequired,
        responses: dict[str, Any],
    ) -> dict[str, Any]:
        assert public.human_request_id is not None
        assert public.human_revision is not None
        assert public.human_preview_sha256 is not None
        row = self.rows[public.human_request_id]
        assert row["status"] == "pending"
        assert row["revision"] == public.human_revision
        row["status"] = "approved"
        row["answer"] = responses
        row["presented_revision"] = public.human_revision
        row["revision"] += 1
        return {
            "human_request_id": public.human_request_id,
            "human_expected_revision": public.human_revision,
            "human_preview_sha256": public.human_preview_sha256,
        }

    def consume_approved_answer(
        self,
        request_id: str,
        *,
        presented_revision: int,
        preview_sha256: str,
    ) -> dict[str, Any]:
        row = self.rows[request_id]
        assert row["status"] == "approved"
        assert row["presented_revision"] == presented_revision
        assert row["preview_sha256"] == preview_sha256
        return dict(row["answer"])

    def cancel_question(
        self,
        request_id: str,
        *,
        preview_sha256: str,
        reason: str,
    ) -> None:
        row = self.rows.get(request_id)
        if row is None:
            return
        assert row["preview_sha256"] == preview_sha256
        if row["status"] == "pending":
            row["status"] = "cancelled"
            row["reason"] = reason
            row["revision"] += 1

    def cancel_question_for_recovery(self, request_id: str, *, reason: str) -> None:
        row = self.rows.get(request_id)
        if row is None:
            return
        self.cancel_question(
            request_id,
            preview_sha256=row["preview_sha256"],
            reason=reason,
        )

    def question_preview_sha256_for_recovery(self, request_id: str) -> str:
        row = self.rows[request_id]
        return str(row["preview_sha256"])


class _Boundary:
    def __init__(self, *results: dict[str, Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []
        self.fail_not_started = False

    async def continue_request(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail_not_started:
            raise McpContinuationDispatchNotStarted("capability denied")
        raw = self.results.pop(0)
        public, settlement = kwargs["result_settler"](
            raw,
            f"continuation-response-effect-{len(self.calls)}",
        )
        try:
            settlement.commit_deferred()
        except Exception:
            settlement.abort(reason="test_boundary_commit_failed")
            raise
        try:
            settlement.finalize()
        except Exception:
            # Production finalization runs after the authoritative outer
            # commit and leaves a cleaning receipt for restart on failure.
            pass
        return public

    async def cancel_continuation(self, **_kwargs: Any) -> None:
        return None


def _binding(**overrides: Any) -> McpContinuationBinding:
    selected: dict[str, Any] = {
        "server_id": "calendar",
        "server_spec_sha256": _sha("spec"),
        "server_generation": 4,
        "owner_id": "process-1",
        "auth_principal_sha256": _sha("principal"),
        "auth_scope_sha256": _sha("scope"),
        "canonical_request": {
            "method": "tools/call",
            "params": {"name": "delete", "arguments": {"event": "evt-7"}},
        },
        "effect_id": "effect-1",
        "capability_sha256": _sha("capability"),
        "data_flow_sha256": _sha("flow"),
    }
    selected.update(overrides)
    return McpContinuationBinding(**selected)


def _input_required(
    *, state: str = "round-state", message: str = "Confirm delete"
) -> dict[str, Any]:
    return {
        "resultType": "input_required",
        "inputRequests": {
            "remote-request-key": {
                "method": "elicitation/create",
                "params": {
                    "mode": "form",
                    "message": message,
                    "requestedSchema": {
                        "type": "object",
                        "properties": {"confirm": {"type": "boolean"}},
                        "required": ["confirm"],
                    },
                },
            }
        },
        "requestState": state,
    }


def _manager(
    repo: _Repo,
    broker: _Broker,
    boundary: _Boundary,
    *,
    now: datetime | None = None,
    human: _Human | None = None,
    **settings: Any,
) -> McpContinuationManager:
    selected_now = now or datetime(2030, 1, 1, tzinfo=timezone.utc)
    return McpContinuationManager(
        repository=repo,
        side_effects=_SideEffects(repo),
        broker=broker,
        human_requests=human or _Human(),
        boundary=boundary,
        now=lambda: selected_now,
        id_factory=lambda: "continuation-local-1",
        sensitive_values=("credential-value",),
        **settings,
    )


def _approved(
    manager: McpContinuationManager,
    public: McpInputRequired,
    responses: dict[str, Any],
) -> dict[str, Any]:
    assert isinstance(manager.human_requests, _Human)
    return manager.human_requests.settle(public, responses)


def test_capture_creates_real_human_request_and_sanitizes_broker_state() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)

    public = manager.capture_input_required(
        _binding(),
        _input_required(
            state="opaque-state",
            message="credential-value reflected",
        ),
        expires_at="2030-01-01T00:05:00Z",
    )

    assert public.continuation_id == "continuation-local-1"
    assert public.revision == 0
    assert public.input_requests[0].request_id == "input-1"
    assert public.human_request_id == "human-real-1"
    assert public.human_revision == 0
    assert public.human_preview_sha256
    assert "credential-value" not in public.input_requests[0].prompt
    persisted = repo.rows[public.continuation_id]
    assert "credential-value" not in repr(persisted)
    assert "remote-request-key" not in repr(persisted)
    assert persisted.broker_ref in broker.values
    assert b"credential-value" not in broker.values[persisted.broker_ref]
    assert b"remote-request-key" in broker.values[persisted.broker_ref]

    with pytest.raises(ValidationError, match="reflected an operation secret"):
        manager.capture_input_required(
            _binding(),
            _input_required(state="credential-value:opaque-state"),
            expires_at=None,
        )
    assert len(repo.rows) == 1


def test_multi_round_response_uses_bound_continuation_and_never_initial_dispatch() -> None:
    repo, broker = _Repo(), _Broker()
    boundary = _Boundary(
        _input_required(state="state-two", message="One more value"),
        {"resultType": "complete", "content": [{"type": "text", "text": "done"}]},
    )
    manager = _manager(repo, broker, boundary)
    first = manager.capture_input_required(
        _binding(), _input_required(state="state-one"), expires_at=None
    )

    second = asyncio.run(
        manager.respond(
            first.continuation_id,
            expected_revision=first.revision,
            binding=_binding(),
            **_approved(
                manager,
                first,
                {"input-1": {"action": "accept", "content": {"confirm": True}}},
            ),
            deadline=100.0,
        )
    )
    assert isinstance(second, McpInputRequired)
    assert second.revision == 2
    assert len(boundary.calls) == 1
    assert boundary.calls[0]["request_state"] == "state-one"
    assert boundary.calls[0]["input_responses"] == {
        "remote-request-key": {
            "action": "accept",
            "content": {"confirm": True},
        }
    }
    assert boundary.calls[0]["original_request"] == _binding().canonical_request

    completed = asyncio.run(
        manager.respond(
            second.continuation_id,
            expected_revision=second.revision,
            binding=_binding(),
            **_approved(manager, second, {"input-1": {"action": "decline"}}),
            deadline=101.0,
        )
    )
    assert isinstance(completed, McpComplete)
    assert completed.value == {
        "content": [{"type": "text", "text": "done"}]
    }
    assert len(boundary.calls) == 2
    assert boundary.calls[1]["request_state"] == "state-two"
    assert repo.rows[first.continuation_id].status is McpContinuationStatus.COMPLETE


def test_human_preview_binds_complete_multi_request_set_and_cross_id_is_denied() -> None:
    raw = _input_required()
    raw["inputRequests"]["another-remote-key"] = {
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": "Choose a label",
            "requestedSchema": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        },
    }
    repo, broker = _Repo(), _Broker()
    boundary = _Boundary({"resultType": "complete", "ok": True})
    human = _Human()
    manager = _manager(repo, broker, boundary, human=human)
    pending = manager.capture_input_required(_binding(), raw, expires_at=None)

    assert pending.human_request_id is not None
    preview = human.rows[pending.human_request_id]["preview"]
    public_requests = preview["inputRequests"]
    assert {item["requestId"] for item in public_requests} == {"input-1", "input-2"}
    assert all(len(item["schemaSha256"]) == 64 for item in public_requests)
    assert "remote-request-key" not in repr(preview)
    fence = human.settle(
        pending,
        {
            "input-1": {"action": "accept", "content": {"label": "safe"}},
            "input-2": {"action": "decline"},
        },
    )
    other = human.create_question(
        owner_id="process-1",
        server_id="calendar",
        operation="tools/call",
        local_ref="other-continuation",
        preview={"other": True},
        preview_sha256="a" * 64,
        expires_at=None,
    )
    with pytest.raises(CapabilityDenied, match="another continuation"):
        asyncio.run(
            manager.respond(
                pending.continuation_id,
                expected_revision=pending.revision,
                binding=_binding(),
                human_request_id=other.request_id,
                human_expected_revision=fence["human_expected_revision"],
                human_preview_sha256=fence["human_preview_sha256"],
                deadline=100.0,
            )
        )
    assert boundary.calls == []

    result = asyncio.run(
        manager.respond(
            pending.continuation_id,
            expected_revision=pending.revision,
            binding=_binding(),
            **fence,
            deadline=100.0,
        )
    )
    assert result == McpComplete(value={"ok": True})
    assert set(boundary.calls[0]["input_responses"]) == {
        "another-remote-key",
        "remote-request-key",
    }


def test_multi_round_response_atomically_rebinds_human_request_in_sql_store() -> None:
    store = SQLiteStore(":memory:")
    try:
        broker = _Broker()
        boundary = _Boundary(
            _input_required(state="state-two", message="One more value"),
        )
        human = _Human(durable_store=store)
        manager = _manager(
            UnitOfWork(store).mcp_continuations,  # type: ignore[arg-type]
            broker,
            boundary,
            human=human,
        )
        first = manager.capture_input_required(
            _binding(), _input_required(state="state-one"), expires_at=None
        )
        second = asyncio.run(
            manager.respond(
                first.continuation_id,
                expected_revision=first.revision,
                binding=_binding(),
                **_approved(
                    manager,
                    first,
                    {"input-1": {"action": "accept", "content": {"confirm": True}}},
                ),
                deadline=100.0,
            )
        )

        assert isinstance(second, McpInputRequired)
        assert second.human_request_id == "human-real-2"
        persisted = UnitOfWork(store).mcp_continuations.get(first.continuation_id)
        assert persisted is not None
        assert persisted.status == "input_required"
        assert persisted.revision == 2
        assert persisted.human_request_id == second.human_request_id
    finally:
        store.close()


def test_duplicate_or_stale_response_cannot_dispatch_twice() -> None:
    repo, broker = _Repo(), _Broker()
    boundary = _Boundary({"resultType": "complete", "ok": True})
    manager = _manager(repo, broker, boundary)
    pending = manager.capture_input_required(_binding(), _input_required(), expires_at=None)
    human_fence = _approved(
        manager,
        pending,
        {"input-1": {"action": "decline"}},
    )

    asyncio.run(
        manager.respond(
            pending.continuation_id,
            expected_revision=0,
            binding=_binding(),
            **human_fence,
            deadline=100.0,
        )
    )
    with pytest.raises(ValidationError, match="revision|terminal"):
        asyncio.run(
            manager.respond(
                pending.continuation_id,
                expected_revision=0,
                binding=_binding(),
                **human_fence,
                deadline=100.0,
            )
        )
    assert len(boundary.calls) == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"server_id": "other"},
        {"server_generation": 5},
        {"auth_principal_sha256": _sha("other-principal")},
        {"auth_scope_sha256": _sha("other-scope")},
        {"effect_id": "other-effect"},
        {"capability_sha256": _sha("other-cap")},
        {"data_flow_sha256": _sha("other-flow")},
        {"canonical_request": {"method": "tools/call", "params": {}}},
    ],
)
def test_cross_binding_reuse_fails_before_dispatch(changed: dict[str, Any]) -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    pending = manager.capture_input_required(_binding(), _input_required(), expires_at=None)
    human_fence = _approved(manager, pending, {"input-1": {"action": "decline"}})
    with pytest.raises((CapabilityDenied, ValidationError), match="binding"):
        asyncio.run(
            manager.respond(
                pending.continuation_id,
                expected_revision=0,
                binding=_binding(**changed),
                **human_fence,
                deadline=100.0,
            )
        )
    assert boundary.calls == []


def test_expired_or_brokerless_continuation_never_dispatches() -> None:
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary, now=now)
    expired = manager.capture_input_required(
        _binding(),
        _input_required(),
        expires_at=(now - timedelta(seconds=1)).isoformat(),
    )
    expired_fence = _approved(
        manager,
        expired,
        {"input-1": {"action": "decline"}},
    )
    with pytest.raises(ValidationError, match="expired"):
        asyncio.run(
            manager.respond(
                expired.continuation_id,
                expected_revision=0,
                binding=_binding(),
                **expired_fence,
                deadline=100.0,
            )
        )
    assert repo.rows[expired.continuation_id].status is McpContinuationStatus.EXPIRED

    repo2, broker2, boundary2 = _Repo(), _Broker(), _Boundary()
    manager2 = _manager(repo2, broker2, boundary2)
    pending = manager2.capture_input_required(_binding(), _input_required(), expires_at=None)
    pending_fence = _approved(
        manager2,
        pending,
        {"input-1": {"action": "decline"}},
    )
    broker2.enabled = False
    with pytest.raises(ValidationError, match="credential broker"):
        asyncio.run(
            manager2.respond(
                pending.continuation_id,
                expected_revision=0,
                binding=_binding(),
                **pending_fence,
                deadline=100.0,
            )
        )
    assert repo2.rows[pending.continuation_id].status is McpContinuationStatus.NEEDS_ATTENTION
    assert boundary.calls == [] and boundary2.calls == []


def test_not_started_authority_failure_releases_claim_for_explicit_retry() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    boundary.fail_not_started = True
    manager = _manager(repo, broker, boundary)
    pending = manager.capture_input_required(_binding(), _input_required(), expires_at=None)
    human_fence = _approved(manager, pending, {"input-1": {"action": "decline"}})
    with pytest.raises(McpContinuationDispatchNotStarted):
        asyncio.run(
            manager.respond(
                pending.continuation_id,
                expected_revision=0,
                binding=_binding(),
                **human_fence,
                deadline=100.0,
            )
        )
    restored = repo.rows[pending.continuation_id]
    assert restored.status is McpContinuationStatus.INPUT_REQUIRED
    assert restored.revision == 2


def test_sampling_and_roots_are_typed_unsupported_and_not_fulfillable() -> None:
    raw = {
        "resultType": "input_required",
        "inputRequests": {
            "sample": {"method": "sampling/createMessage", "params": {}},
            "roots": {"method": "roots/list", "params": {}},
        },
        "requestState": "opaque",
    }
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    pending = manager.capture_input_required(_binding(), raw, expires_at=None)
    assert [item.kind for item in pending.input_requests] == [
        McpInputRequestKind.ROOTS_UNSUPPORTED,
        McpInputRequestKind.SAMPLING_UNSUPPORTED,
    ]
    assert pending.respondable is False
    assert pending.continuation_id == ""
    assert pending.human_request_id is None
    assert repo.rows == {}
    assert boundary.calls == []


def test_restart_reconciles_dispatching_to_attention_without_replay() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    human = _Human()
    manager = _manager(repo, broker, boundary, human=human)
    pending = manager.capture_input_required(_binding(), _input_required(), expires_at=None)
    claimed = replace(
        repo.rows[pending.continuation_id],
        status=McpContinuationStatus.DISPATCHING,
        revision=1,
    )
    assert repo.compare_and_swap(
        pending.continuation_id, expected_revision=0, replacement=claimed
    )

    # Construction is the startup boundary and reconciles without dispatch.
    _manager(repo, broker, boundary, human=human)
    assert repo.rows[pending.continuation_id].status is McpContinuationStatus.NEEDS_ATTENTION
    assert boundary.calls == []


def test_request_state_only_requires_explicit_empty_response() -> None:
    repo, broker = _Repo(), _Broker()
    boundary = _Boundary({"resultType": "complete", "ok": True})
    manager = _manager(repo, broker, boundary)
    pending = manager.capture_input_required(
        _binding(),
        {"resultType": "input_required", "requestState": "load-shed-state"},
        expires_at=None,
    )
    assert pending.input_requests == ()
    result = asyncio.run(
        manager.respond(
            pending.continuation_id,
            expected_revision=0,
            binding=_binding(),
            **_approved(manager, pending, {}),
            deadline=100.0,
        )
    )
    assert result == McpComplete(value={"ok": True})
    assert boundary.calls[0]["request_state"] == "load-shed-state"


def test_invalid_deadline_is_local_preflight_and_does_not_claim() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    pending = manager.capture_input_required(_binding(), _input_required(), expires_at=None)
    human_fence = _approved(
        manager,
        pending,
        {"input-1": {"action": "decline"}},
    )
    with pytest.raises(ValidationError, match="deadline"):
        asyncio.run(
            manager.respond(
                pending.continuation_id,
                expected_revision=0,
                binding=_binding(),
                **human_fence,
                deadline=0,
            )
        )
    assert repo.rows[pending.continuation_id].revision == 0
    assert boundary.calls == []


def test_original_request_is_detached_from_later_caller_mutation() -> None:
    repo, broker = _Repo(), _Broker()
    boundary = _Boundary({"resultType": "complete", "ok": True})
    manager = _manager(repo, broker, boundary)
    binding = _binding()
    pending = manager.capture_input_required(binding, _input_required(), expires_at=None)
    binding.canonical_request["params"] = {"name": "attacker-changed"}
    human_fence = _approved(
        manager,
        pending,
        {"input-1": {"action": "decline"}},
    )

    asyncio.run(
        manager.respond(
            pending.continuation_id,
            expected_revision=0,
            binding=binding,
            **human_fence,
            deadline=100.0,
        )
    )
    assert boundary.calls[0]["original_request"] == {
        "method": "tools/call",
        "params": {"name": "delete", "arguments": {"event": "evt-7"}},
    }


def test_binding_material_recovers_broker_only_original_request_and_fails_closed_on_tamper() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    binding = _binding()
    pending = manager.capture_input_required(binding, _input_required(), expires_at=None)

    recovered = manager.binding_material(pending.continuation_id)
    assert recovered == binding
    durable_text = repr(repo.rows[pending.continuation_id])
    assert "evt-7" not in durable_text

    record = repo.rows[pending.continuation_id]
    assert record.broker_ref is not None
    broker.values[record.broker_ref] = broker.values[record.broker_ref].replace(
        b'"evt-7"',
        b'"evt-8"',
    )
    # Updating the outer broker digest cannot bypass the independently durable
    # canonical request hash fence.
    tampered = replace(
        record,
        broker_value_sha256=hashlib.sha256(broker.values[record.broker_ref]).hexdigest(),
    )
    repo.rows[pending.continuation_id] = tampered
    with pytest.raises(ValidationError, match="binding changed"):
        manager.binding_material(pending.continuation_id)
    assert repo.rows[pending.continuation_id].status is McpContinuationStatus.NEEDS_ATTENTION


def test_titled_elicitation_enum_is_validated_before_dispatch() -> None:
    raw = _input_required()
    raw["inputRequests"]["remote-request-key"]["params"]["requestedSchema"] = {
        "type": "object",
        "properties": {
            "color": {
                "type": "string",
                "oneOf": [
                    {"const": "r", "title": "Red"},
                    {"const": "b", "title": "Blue"},
                ],
            }
        },
        "required": ["color"],
    }
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    pending = manager.capture_input_required(_binding(), raw, expires_at=None)
    human_fence = _approved(
        manager,
        pending,
        {"input-1": {"action": "accept", "content": {"color": "x"}}},
    )
    with pytest.raises(ValidationError, match="enum"):
        asyncio.run(
            manager.respond(
                pending.continuation_id,
                expected_revision=0,
                binding=_binding(),
                **human_fence,
                deadline=100.0,
            )
        )
    assert repo.rows[pending.continuation_id].revision == 0
    assert boundary.calls == []


def test_host_mrtr_limits_attenuate_requests_state_ttl_and_rounds() -> None:
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    repo, broker, boundary, human = _Repo(), _Broker(), _Boundary(), _Human()
    limited = _manager(
        repo,
        broker,
        boundary,
        now=now,
        human=human,
        max_input_requests=1,
        request_state_max_bytes=4,
        continuation_ttl_s=2.0,
    )
    too_many = _input_required(state="1234")
    too_many["inputRequests"]["second"] = too_many["inputRequests"][
        "remote-request-key"
    ]
    with pytest.raises(ValidationError, match="bounded"):
        limited.capture_input_required(_binding(), too_many, expires_at=None)
    with pytest.raises(ValidationError, match="requestState"):
        limited.capture_input_required(
            _binding(),
            _input_required(state="12345"),
            expires_at=None,
        )
    assert repo.rows == {} and human.rows == {}

    pending = limited.capture_input_required(
        _binding(),
        _input_required(state="1234"),
        expires_at=None,
    )
    assert datetime.fromisoformat(pending.expires_at) == now + timedelta(seconds=2)

    round_repo, round_broker = _Repo(), _Broker()
    round_boundary = _Boundary(_input_required(state="next"))
    round_manager = _manager(
        round_repo,
        round_broker,
        round_boundary,
        max_rounds=1,
    )
    first = round_manager.capture_input_required(
        _binding(), _input_required(), expires_at=None
    )
    with pytest.raises(ValidationError, match="unknown outcome"):
        asyncio.run(
            round_manager.respond(
                first.continuation_id,
                expected_revision=first.revision,
                binding=_binding(),
                **_approved(
                    round_manager,
                    first,
                    {"input-1": {"action": "decline"}},
                ),
                deadline=100.0,
            )
        )
    assert round_repo.rows[first.continuation_id].status is McpContinuationStatus.NEEDS_ATTENTION
    assert len(round_boundary.calls) == 1


def test_post_provider_malformed_result_is_unknown_and_never_retryable() -> None:
    repo, broker = _Repo(), _Broker()
    boundary = _Boundary({"resultType": "not-a-result"})
    manager = _manager(repo, broker, boundary)
    pending = manager.capture_input_required(
        _binding(),
        _input_required(),
        expires_at=None,
    )

    with pytest.raises(ValidationError, match="unknown outcome"):
        asyncio.run(
            manager.respond(
                pending.continuation_id,
                expected_revision=pending.revision,
                binding=_binding(),
                **_approved(
                    manager,
                    pending,
                    {"input-1": {"action": "decline"}},
                ),
                deadline=100.0,
            )
        )

    assert len(boundary.calls) == 1
    record = repo.rows[pending.continuation_id]
    assert record.status is McpContinuationStatus.NEEDS_ATTENTION
    assert record.metadata["automatic_retry_disabled"] is True


def test_sdk_capture_is_prepared_until_exact_public_result_is_claimed() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    adapter = McpSdkContinuationCaptureAdapter(
        manager,
        lambda _server, _operation, _logical: _binding(),
    )

    public = adapter.capture_input_required(
        server_id="calendar",
        operation="tools/call",
        logical_id="delete",
        request_state="round-state",
        input_requests=_input_required()["inputRequests"],
        deadline=100.0,
        sensitive_values=(),
    )

    assert repo.rows == {}
    side_effects = manager.side_effects
    prepared = tuple(side_effects.rows.values())
    assert len(prepared) == 1 and prepared[0].status == "prepared"
    settlement = manager.claim_initial_capture(public, _binding())
    assert isinstance(settlement, McpContinuationCaptureSettlement)
    settlement.commit_deferred()
    assert repo.rows[public.continuation_id].effect_id == "effect-1"
    assert next(iter(side_effects.rows.values())).status == "cleaning"
    settlement.finalize()
    assert side_effects.rows == {}


def test_prepared_capture_rejects_mutated_public_binding_and_sidecar() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    public = manager.prepare_initial_input_required(
        _binding(), _input_required(), expires_at=None
    )
    exact_public = deepcopy(public)
    public.input_requests[0].schema["properties"]["confirm"]["type"] = "string"
    with pytest.raises(CapabilityDenied, match="result changed"):
        manager.claim_initial_capture(public, _binding())
    with pytest.raises(CapabilityDenied, match="binding changed"):
        manager.claim_initial_capture(
            exact_public,
            _binding(server_generation=5),
        )
    side_effects = manager.side_effects
    preparation = next(iter(side_effects.rows.values()))
    side_effects.rows[preparation.preparation_id] = replace(
        preparation,
        owner_id="another-owner",
    )
    with pytest.raises(ValidationError, match="sidecar is unavailable"):
        manager.claim_initial_capture(exact_public, _binding())
    side_effects.rows[preparation.preparation_id] = preparation
    assert manager.abort_prepared_effect("effect-1") == 1
    assert repo.rows == {} and side_effects.rows == {} and broker.values == {}


def test_initial_capture_claim_is_single_winner_and_abort_is_idempotent() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    public = manager.prepare_initial_input_required(
        _binding(), _input_required(), expires_at=None
    )

    def claim() -> object:
        try:
            return manager.claim_initial_capture(public, _binding())
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: claim(), range(2)))
    winner = next(item for item in results if isinstance(item, McpContinuationCaptureSettlement))
    assert sum(isinstance(item, ValidationError) for item in results) == 1
    winner.abort()
    winner.abort()
    assert repo.rows == {} and manager.side_effects.rows == {} and broker.values == {}


def test_initial_capture_exception_cleans_but_baseexception_waits_for_restart() -> None:
    class Crash(BaseException):
        pass

    class FailingBroker(_Broker):
        def __init__(self, error: BaseException) -> None:
            super().__init__()
            self.error = error

        def put_secret_at(self, *args: Any, **kwargs: Any) -> None:
            super().put_secret_at(*args, **kwargs)
            raise self.error

    error_broker = FailingBroker(RuntimeError("write failed"))
    error_manager = _manager(_Repo(), error_broker, _Boundary())
    with pytest.raises(ValidationError, match="broker write failed"):
        error_manager.prepare_initial_input_required(
            _binding(), _input_required(), expires_at=None
        )
    assert error_manager.side_effects.rows == {} and error_broker.values == {}

    crash_broker = FailingBroker(Crash())
    crash_manager = _manager(_Repo(), crash_broker, _Boundary())
    with pytest.raises(Crash):
        crash_manager.prepare_initial_input_required(
            _binding(), _input_required(), expires_at=None
        )
    assert len(crash_manager.side_effects.rows) == 1
    assert next(iter(crash_manager.side_effects.rows.values())).status == "prepared"
    crash_manager.reconcile_after_restart()
    assert crash_manager.side_effects.rows == {} and crash_broker.values == {}


def test_eager_capture_returns_recoverable_ref_after_finalize_failure() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    side_effects = manager.side_effects
    original_delete = side_effects.delete

    def fail_delete(_preparation_id: str, *, expected_revision: int) -> bool:
        del expected_revision
        return False

    side_effects.delete = fail_delete
    public = manager.capture_input_required(
        _binding(), _input_required(), expires_at=None
    )
    assert repo.rows[public.continuation_id].effect_id == "effect-1"
    assert next(iter(side_effects.rows.values())).status == "cleaning"
    side_effects.delete = original_delete
    manager.reconcile_after_restart()
    assert side_effects.rows == {}
