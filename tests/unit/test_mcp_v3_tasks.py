from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from agent_libos.mcp.manifest import MCP_TASKS_EXTENSION_ID
from agent_libos.mcp.supervisor import McpConnectionFence
from agent_libos.mcp.tasks import (
    McpContinuationRemoteTaskCaptureAdapter,
    McpRemoteTaskBinding,
    McpRemoteTaskDispatchNotStarted,
    McpRemoteTaskManager,
    McpRemoteTaskRecordStatus,
)
from agent_libos.mcp.types import McpRemoteTaskStatus, McpSubscriptionEvent
from agent_libos.models.exceptions import CapabilityDenied, ValidationError

from tests.unit.test_mcp_v3_continuations import _Broker, _Human, _SideEffects, _sha
from tests.unit.test_mcp_v3_continuations import _binding as _continuation_binding


_TASKS_DIGEST = hashlib.sha256(b"pinned tasks extension schema").hexdigest()


class _Repo:
    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}

    def insert(self, record: Any) -> None:
        self.rows[record.task_ref] = record

    def get(self, task_ref: str) -> Any | None:
        return self.rows.get(task_ref)

    def get_by_remote_id_sha256(
        self,
        server_id: str,
        remote_id_sha256: str,
    ) -> Any | None:
        return next(
            (
                row
                for row in self.rows.values()
                if row.server_id == server_id
                and row.remote_id_sha256 == remote_id_sha256
            ),
            None,
        )

    def list(self, **_filters: object) -> list[Any]:
        return list(self.rows.values())

    def count(self, *, owner_id: str | None = None) -> int:
        if owner_id is None:
            return len(self.rows)
        return sum(row.owner_id == owner_id for row in self.rows.values())

    def count_active(self, *, owner_id: str | None = None) -> int:
        terminal = {"completed", "failed", "cancelled", "needs_attention"}
        return sum(
            row.status not in terminal
            and (owner_id is None or row.owner_id == owner_id)
            for row in self.rows.values()
        )

    def list_terminal(
        self, *, owner_id: str | None = None, limit: int = 100
    ) -> tuple[Any, ...]:
        terminal = {"completed", "failed", "cancelled", "needs_attention"}
        return tuple(
            row
            for row in self.rows.values()
            if row.status in terminal
            and (owner_id is None or row.owner_id == owner_id)
        )[:limit]

    def delete_terminal(self, task_ref: str, *, expected_revision: int) -> bool:
        current = self.rows.get(task_ref)
        if (
            current is None
            or current.revision != expected_revision
            or current.status not in {"completed", "failed", "cancelled", "needs_attention"}
        ):
            return False
        del self.rows[task_ref]
        return True

    def compare_and_swap(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        replacement: Any,
    ) -> bool:
        current = self.rows.get(task_ref)
        if current is None or current.revision != expected_revision:
            return False
        self.rows[task_ref] = replacement
        return True


class _Boundary:
    def __init__(self) -> None:
        self.get_results: list[dict[str, Any]] = []
        self.update_results: list[dict[str, Any]] = []
        self.cancel_results: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self.fail_not_started = False

    async def get_remote_task(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        if self.fail_not_started:
            raise McpRemoteTaskDispatchNotStarted("denied")
        return self.get_results.pop(0)

    async def update_remote_task(self, **kwargs: Any) -> dict[str, Any]:
        self.update_calls.append(kwargs)
        if self.fail_not_started:
            raise McpRemoteTaskDispatchNotStarted("denied")
        return self.update_results.pop(0) if self.update_results else {"resultType": "complete"}

    async def cancel_remote_task(self, **kwargs: Any) -> dict[str, Any]:
        self.cancel_calls.append(kwargs)
        if self.fail_not_started:
            raise McpRemoteTaskDispatchNotStarted("denied")
        return self.cancel_results.pop(0) if self.cancel_results else {"resultType": "complete"}


def _binding(**overrides: Any) -> McpRemoteTaskBinding:
    selected: dict[str, Any] = {
        "server_id": "builder",
        "server_spec_sha256": _sha("server-spec"),
        "server_generation": 7,
        "owner_id": "process-7",
        "auth_principal_sha256": _sha("principal"),
        "auth_scope_sha256": _sha("scope"),
        "origin_request_sha256": _sha("original request"),
        "origin_effect_id": "effect-initial",
        "extension_id": MCP_TASKS_EXTENSION_ID,
        "tasks_extension_sha256": _TASKS_DIGEST,
        "host_tasks_extension_sha256": _TASKS_DIGEST,
    }
    selected.update(overrides)
    return McpRemoteTaskBinding(**selected)


def _fence(**overrides: Any) -> McpConnectionFence:
    selected: dict[str, Any] = {
        "server_id": "builder",
        "server_spec_sha256": _sha("server-spec"),
        "registry_generation": 7,
        "owner": "process-7",
        "auth_principal_sha256": _sha("principal"),
        "auth_scope_sha256": _sha("scope"),
    }
    selected.update(overrides)
    return McpConnectionFence(**selected)


def _task_notification(**overrides: Any) -> McpSubscriptionEvent:
    payload = _task_result()
    payload.pop("resultType")
    payload.update(overrides)
    return McpSubscriptionEvent(
        sequence=19,
        event_type="taskStatus",
        payload=payload,
        received_at="2030-01-01T00:00:02Z",
    )


def _task_result(
    *,
    status: str = "working",
    task_id: str = "remote-bearer-id",
    **extra: Any,
) -> dict[str, Any]:
    selected: dict[str, Any] = {
        "resultType": "task" if status == "working" else "complete",
        "taskId": task_id,
        "status": status,
        "statusMessage": "still running",
        "createdAt": "2030-01-01T00:00:00Z",
        "lastUpdatedAt": "2030-01-01T00:00:01Z",
        "ttlMs": 60_000,
        "pollIntervalMs": 250,
    }
    selected.update(extra)
    return selected


def _manager(
    repo: _Repo,
    broker: _Broker,
    boundary: _Boundary,
    *,
    now: datetime | None = None,
    human: _Human | None = None,
    **settings: Any,
) -> McpRemoteTaskManager:
    selected_now = now or datetime(2030, 1, 1, tzinfo=timezone.utc)
    return McpRemoteTaskManager(
        repository=repo,
        side_effects=_SideEffects(repo),
        broker=broker,
        human_requests=human or _Human(),
        boundary=boundary,
        now=lambda: selected_now,
        id_factory=lambda: "task-local-1",
        sensitive_values=("credential-value",),
        **settings,
    )


def test_create_task_requires_exact_extension_pin_and_hides_remote_id() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    public = manager.capture_task(
        _binding(),
        _task_result(task_id="remote-bearer-id"),
    )
    assert public.task_ref == "task-local-1"
    assert public.revision == 0
    assert "remote-bearer-id" not in repr(public)
    persisted = repo.rows[public.task_ref]
    assert "remote-bearer-id" not in repr(persisted)
    assert persisted.broker_ref in broker.values
    assert broker.values[persisted.broker_ref] == b"remote-bearer-id"

    recovered = manager.binding_material(
        public.task_ref,
        tasks_extension_sha256=_TASKS_DIGEST,
        host_tasks_extension_sha256=_TASKS_DIGEST,
    )
    assert recovered == _binding()
    with pytest.raises(ValidationError, match="extension|pin"):
        manager.binding_material(
            public.task_ref,
            tasks_extension_sha256=_sha("wrong extension"),
            host_tasks_extension_sha256=_TASKS_DIGEST,
        )

    with pytest.raises(ValidationError, match="reflected an operation secret"):
        manager.capture_task(
            _binding(),
            _task_result(task_id="credential-value"),
        )
    assert len(repo.rows) == 1

    for changed in (
        {"extension_id": "vendor/tasks"},
        {"tasks_extension_sha256": _sha("wrong")},
        {"host_tasks_extension_sha256": _sha("different")},
    ):
        with pytest.raises(ValidationError, match="extension|pin"):
            manager.capture_task(
                _binding(**changed),
                _task_result(task_id=f"unused-{len(broker.values)}"),
            )


def test_initial_task_prepare_is_not_published_until_deferred_settlement() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)

    public = manager.prepare_initial_task(_binding(), _task_result())

    assert repo.rows == {}
    assert manager.has_prepared_effect("effect-initial") is True
    settlement = manager.claim_initial_capture(public, binding=_binding())
    settlement.commit_deferred()
    assert repo.rows[public.task_ref].origin_effect_id == "effect-initial"
    settlement.finalize()
    assert manager.has_prepared_effect("effect-initial") is False


def test_initial_task_claim_rejects_mutated_public_projection_and_aborts() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    public = manager.prepare_initial_task(
        _binding(),
        _task_result(
            resultType="task",
            status="completed",
            result={"nested": {"ok": True}},
        ),
    )
    assert isinstance(public.result, dict)
    public.result["nested"] = {"ok": False}

    with pytest.raises(ValidationError, match="provenance"):
        manager.claim_initial_capture(public, binding=_binding())
    manager.abort_prepared_effect("effect-initial")

    assert repo.rows == {}
    assert manager.has_prepared_effect("effect-initial") is False


def test_task_notification_projects_only_known_local_ref_without_mutation() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    before = repo.rows[created.task_ref]

    projected = manager.project_task_notification(
        event=_task_notification(
            status="completed",
            statusMessage="remote-bearer-id finished",
        ),
        fence=_fence(),
        sensitive_values=(),
    )

    assert projected.event_type == "taskStatus"
    assert projected.payload["task_ref"] == created.task_ref
    assert projected.payload["status"] == "completed"
    assert "remote-bearer-id" not in repr(projected)
    assert "result" not in projected.payload
    assert "inputRequests" not in projected.payload
    assert repo.rows[created.task_ref] == before
    assert boundary.get_calls == boundary.update_calls == boundary.cancel_calls == []
    assert manager.subscription_targets(fence=_fence()) == ("remote-bearer-id",)


def test_task_notification_unknown_tampered_or_cross_fence_fails_closed() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    persisted = repo.rows[created.task_ref]

    with pytest.raises(ValidationError, match="unknown"):
        manager.project_task_notification(
            event=_task_notification(taskId="unregistered-bearer"),
            fence=_fence(),
            sensitive_values=(),
        )
    for changed in (
        {"server_spec_sha256": _sha("replacement")},
        {"registry_generation": 8},
        {"owner": "other-owner"},
        {"auth_principal_sha256": _sha("other-principal")},
        {"auth_scope_sha256": _sha("other-scope")},
    ):
        with pytest.raises((CapabilityDenied, ValidationError)):
            manager.project_task_notification(
                event=_task_notification(),
                fence=_fence(**changed),
                sensitive_values=(),
            )
    broker.values[persisted.broker_ref] = b"tampered-bearer"
    with pytest.raises(ValidationError, match="integrity"):
        manager.project_task_notification(
            event=_task_notification(),
            fence=_fence(),
            sensitive_values=(),
        )
    assert repo.rows[created.task_ref] == persisted


def test_task_notification_rejects_payloads_apps_and_dynamic_secrets() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    manager.capture_task(_binding(), _task_result())
    for extra in (
        {"result": {"ok": True}},
        {"error": {"code": "boom"}},
        {"inputRequests": {}},
        {"resultType": "complete"},
        {"ui/resourceUri": "ui://remote"},
    ):
        with pytest.raises(ValidationError, match="unsupported"):
            manager.project_task_notification(
                event=_task_notification(**extra),
                fence=_fence(),
                sensitive_values=(),
            )
    with pytest.raises(ValidationError, match="secret"):
        manager.project_task_notification(
            event=_task_notification(taskId="refreshed-oauth-token"),
            fence=_fence(),
            sensitive_values=("refreshed-oauth-token",),
        )


def test_remote_task_id_is_sensitive_for_all_sibling_projections() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    boundary.get_results.append(
        _task_result(
            status="completed",
            statusMessage="remote-bearer-id status",
            result={
                "echo": "remote-bearer-id",
                "ui/resourceUri": "ui://forbidden-app",
            },
        )
    )
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(
        _binding(),
        _task_result(statusMessage="remote-bearer-id working"),
    )
    assert "remote-bearer-id" not in repr(created)
    completed = asyncio.run(
        manager.get(
            created.task_ref,
            expected_revision=created.revision,
            binding=_binding(),
            deadline=100.0,
        )
    )
    assert "remote-bearer-id" not in repr(completed)
    assert "ui/" not in repr(completed)
    assert "remote-bearer-id" not in repr(repo.rows[created.task_ref])


def test_continuation_task_capture_preserves_original_effect_fence_without_replay() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    capture = McpContinuationRemoteTaskCaptureAdapter(
        manager,
        lambda binding, *, origin_effect_id: McpRemoteTaskBinding(
            server_id=binding.server_id,
            server_spec_sha256=binding.server_spec_sha256,
            server_generation=binding.server_generation,
            owner_id=binding.owner_id,
            auth_principal_sha256=binding.auth_principal_sha256,
            auth_scope_sha256=binding.auth_scope_sha256,
            origin_request_sha256=binding.request_sha256,
            origin_effect_id=origin_effect_id,
            extension_id=MCP_TASKS_EXTENSION_ID,
            tasks_extension_sha256=_TASKS_DIGEST,
            host_tasks_extension_sha256=_TASKS_DIGEST,
        ),
    )
    continuation_binding = _continuation_binding()
    public = capture(continuation_binding, _task_result())

    assert public.task_ref == "task-local-1"
    record = repo.rows[public.task_ref]
    assert record.origin_request_sha256 == continuation_binding.request_sha256
    assert record.origin_effect_id == continuation_binding.effect_id
    assert boundary.get_calls == boundary.update_calls == boundary.cancel_calls == []


def test_get_terminal_result_is_sanitized_and_status_is_monotonic() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    boundary.get_results.append(
        _task_result(
            status="completed",
            result={"text": "credential-value leaked"},
            statusMessage="credential-value status",
        )
    )
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    completed = asyncio.run(
        manager.get(
            created.task_ref,
            expected_revision=0,
            binding=_binding(),
            deadline=100.0,
        )
    )
    assert completed.status is McpRemoteTaskStatus.COMPLETED
    assert "credential-value" not in repr(completed)
    assert completed.revision == 1
    assert boundary.get_calls[0]["remote_task_id"] == "remote-bearer-id"
    assert "credential-value" not in repr(repo.rows[created.task_ref])

    # A terminal task is served from the durable/local projection and cannot regress.
    again = asyncio.run(
        manager.get(
            created.task_ref,
            expected_revision=1,
            binding=_binding(),
            deadline=101.0,
        )
    )
    assert again.status is McpRemoteTaskStatus.COMPLETED
    assert len(boundary.get_calls) == 1


def test_task_input_update_maps_local_request_ids_and_dispatches_once() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    boundary.get_results.append(
        _task_result(
            status="input_required",
            inputRequests={
                "remote-input-key": {
                    "method": "elicitation/create",
                    "params": {
                        "message": "Approve?",
                        "requestedSchema": {
                            "type": "object",
                            "properties": {"approved": {"type": "boolean"}},
                            "required": ["approved"],
                        },
                    },
                }
            },
        )
    )
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    waiting = asyncio.run(
        manager.get(
            created.task_ref,
            expected_revision=0,
            binding=_binding(),
            deadline=100.0,
        )
    )
    assert waiting.status is McpRemoteTaskStatus.INPUT_REQUIRED
    assert waiting.input_requests[0].request_id == "input-1"
    assert waiting.human_request_id == "human-real-1"
    assert waiting.human_revision == 0
    assert waiting.human_preview_sha256
    assert isinstance(manager.human_requests, _Human)
    human_fence = manager.human_requests.settle(
        waiting,
        {"input-1": {"action": "accept", "content": {"approved": True}}},
    )

    working = asyncio.run(
        manager.update(
            created.task_ref,
            expected_revision=waiting.revision,
            binding=_binding(),
            **human_fence,
            deadline=101.0,
        )
    )
    assert working.status is McpRemoteTaskStatus.WORKING
    assert boundary.update_calls[0]["input_responses"] == {
        "remote-input-key": {
            "action": "accept",
            "content": {"approved": True},
        }
    }
    with pytest.raises(ValidationError, match="revision|state"):
        asyncio.run(
            manager.update(
                created.task_ref,
                expected_revision=waiting.revision,
                binding=_binding(),
                **human_fence,
                deadline=102.0,
            )
        )
    assert len(boundary.update_calls) == 1


def test_each_task_input_round_gets_a_new_real_human_request() -> None:
    input_requests = {
        "remote-input-key": {
            "method": "elicitation/create",
            "params": {
                "message": "Approve?",
                "requestedSchema": {
                    "type": "object",
                    "properties": {"approved": {"type": "boolean"}},
                    "required": ["approved"],
                },
            },
        }
    }
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    boundary.get_results.extend(
        [
            _task_result(status="input_required", inputRequests=input_requests),
            _task_result(
                status="input_required",
                inputRequests=input_requests,
                lastUpdatedAt="2030-01-01T00:00:02Z",
            ),
            _task_result(
                status="input_required",
                inputRequests=input_requests,
                lastUpdatedAt="2030-01-01T00:00:03Z",
            ),
        ]
    )
    human = _Human()
    manager = _manager(repo, broker, boundary, human=human)
    created = manager.capture_task(_binding(), _task_result())
    first = asyncio.run(
        manager.get(
            created.task_ref,
            expected_revision=created.revision,
            binding=_binding(),
            deadline=100.0,
        )
    )
    first_human_id = first.human_request_id
    assert first_human_id == "human-real-1"

    # Re-observing the same outstanding round reuses its exact question.
    repeated = asyncio.run(
        manager.get(
            created.task_ref,
            expected_revision=first.revision,
            binding=_binding(),
            deadline=101.0,
        )
    )
    assert repeated.human_request_id == first_human_id
    assert human.counter == 1
    fence = human.settle(
        repeated,
        {"input-1": {"action": "decline"}},
    )
    working = asyncio.run(
        manager.update(
            created.task_ref,
            expected_revision=repeated.revision,
            binding=_binding(),
            **fence,
            deadline=102.0,
        )
    )

    later = asyncio.run(
        manager.get(
            created.task_ref,
            expected_revision=working.revision,
            binding=_binding(),
            deadline=103.0,
        )
    )
    assert later.human_request_id == "human-real-2"
    assert later.human_request_id != first_human_id
    assert human.counter == 2


def test_cancel_ack_is_only_cancel_requested_and_is_exactly_once() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    requested = asyncio.run(
        manager.cancel(
            created.task_ref,
            expected_revision=0,
            binding=_binding(),
            deadline=100.0,
        )
    )
    assert requested.status is McpRemoteTaskStatus.CANCEL_REQUESTED
    assert requested.status is not McpRemoteTaskStatus.CANCELLED
    with pytest.raises(ValidationError, match="revision|state"):
        asyncio.run(
            manager.cancel(
                created.task_ref,
                expected_revision=0,
                binding=_binding(),
                deadline=100.0,
            )
        )
    assert len(boundary.cancel_calls) == 1


def test_cancel_requested_can_later_observe_actual_terminal_state() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    requested = asyncio.run(
        manager.cancel(
            created.task_ref,
            expected_revision=0,
            binding=_binding(),
            deadline=100.0,
        )
    )
    boundary.get_results.append(
        _task_result(
            status="completed",
            statusMessage="completed despite cancellation",
            result={"ok": True},
        )
    )
    completed = asyncio.run(
        manager.get(
            created.task_ref,
            expected_revision=requested.revision,
            binding=_binding(),
            deadline=101.0,
        )
    )
    assert completed.status is McpRemoteTaskStatus.COMPLETED
    assert completed.result == {"ok": True}


def test_task_invalid_deadline_is_local_preflight() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    with pytest.raises(ValidationError, match="deadline"):
        asyncio.run(
            manager.cancel(
                created.task_ref,
                expected_revision=0,
                binding=_binding(),
                deadline=0,
            )
        )
    assert repo.rows[created.task_ref].revision == 0
    assert boundary.cancel_calls == []


def test_task_observation_rejects_backwards_remote_timestamp() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    boundary.get_results.append(
        _task_result(
            lastUpdatedAt="2030-01-01T00:00:00Z",
            resultType="complete",
        )
    )
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    with pytest.raises(ValidationError, match="backwards"):
        asyncio.run(
            manager.get(
                created.task_ref,
                expected_revision=0,
                binding=_binding(),
                deadline=100.0,
            )
        )
    assert repo.rows[created.task_ref].revision == 0


@pytest.mark.parametrize(
    "changed",
    [
        {"server_id": "other"},
        {"server_generation": 8},
        {"auth_principal_sha256": _sha("other")},
        {"auth_scope_sha256": _sha("other")},
        {"origin_request_sha256": _sha("other")},
        {"origin_effect_id": "other-effect"},
    ],
)
def test_task_cross_binding_reuse_is_denied_before_provider(changed: dict[str, Any]) -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    with pytest.raises((CapabilityDenied, ValidationError), match="binding"):
        asyncio.run(
            manager.get(
                created.task_ref,
                expected_revision=0,
                binding=_binding(**changed),
                deadline=100.0,
            )
        )
    assert boundary.get_calls == []


def test_missing_broker_and_expiry_need_attention_without_provider() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    broker.enabled = False
    with pytest.raises(ValidationError, match="credential broker"):
        asyncio.run(
            manager.get(
                created.task_ref,
                expected_revision=0,
                binding=_binding(),
                deadline=100.0,
            )
        )
    assert repo.rows[created.task_ref].status is McpRemoteTaskRecordStatus.NEEDS_ATTENTION
    assert boundary.get_calls == []

    old = datetime(2030, 1, 1, tzinfo=timezone.utc)
    repo2, broker2, boundary2 = _Repo(), _Broker(), _Boundary()
    manager2 = _manager(repo2, broker2, boundary2, now=old + timedelta(minutes=2))
    stale = manager2.capture_task(
        _binding(),
        _task_result(
            createdAt=old.isoformat(),
            lastUpdatedAt=old.isoformat(),
            ttlMs=1_000,
        ),
    )
    with pytest.raises(ValidationError, match="expired"):
        asyncio.run(
            manager2.get(
                stale.task_ref,
                expected_revision=0,
                binding=_binding(),
                deadline=100.0,
            )
        )
    assert boundary2.get_calls == []


def test_crash_reconciliation_never_replays_update_or_initial_call() -> None:
    repo, broker, boundary = _Repo(), _Broker(), _Boundary()
    manager = _manager(repo, broker, boundary)
    created = manager.capture_task(_binding(), _task_result())
    current = repo.rows[created.task_ref]
    repo.rows[created.task_ref] = replace(
        current,
        status=McpRemoteTaskRecordStatus.UPDATE_DISPATCHING,
        revision=1,
    )
    _manager(repo, broker, boundary)
    assert repo.rows[created.task_ref].status is McpRemoteTaskRecordStatus.NEEDS_ATTENTION
    assert boundary.update_calls == []
    assert not hasattr(manager, "list")


def test_host_task_limits_bound_inputs_lifetime_polling_and_records() -> None:
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    repo, broker, boundary, human = _Repo(), _Broker(), _Boundary(), _Human()
    manager = _manager(
        repo,
        broker,
        boundary,
        now=now,
        human=human,
        max_input_requests=1,
        max_wait_s=2.0,
        max_records=1,
    )
    too_many = {
        key: {
            "method": "elicitation/create",
            "params": {
                "message": key,
                "requestedSchema": {"type": "object", "properties": {}},
            },
        }
        for key in ("one", "two")
    }
    with pytest.raises(ValidationError, match="bounded"):
        manager.capture_task(
            _binding(),
            _task_result(
                status="input_required",
                resultType="task",
                inputRequests=too_many,
            ),
        )
    assert repo.rows == {} and human.rows == {}

    created = manager.capture_task(
        _binding(),
        _task_result(ttlMs=60_000),
    )
    persisted = repo.rows[created.task_ref]
    assert datetime.fromisoformat(persisted.expires_at) == now + timedelta(seconds=2)
    with pytest.raises(ValidationError, match="record limit"):
        manager.capture_task(_binding(), _task_result(task_id="another-id"))

    poll_repo, poll_broker, poll_boundary = _Repo(), _Broker(), _Boundary()
    poll_human = _Human()
    initial = _manager(
        poll_repo,
        poll_broker,
        poll_boundary,
        now=now,
        human=poll_human,
    ).capture_task(_binding(), _task_result())
    too_soon = _manager(
        poll_repo,
        poll_broker,
        poll_boundary,
        now=now + timedelta(milliseconds=100),
        human=poll_human,
        poll_min_interval_s=0.25,
    )
    with pytest.raises(ValidationError, match="poll interval"):
        asyncio.run(
            too_soon.get(
                initial.task_ref,
                expected_revision=initial.revision,
                binding=_binding(),
                deadline=100.0,
            )
        )
    assert poll_boundary.get_calls == []
