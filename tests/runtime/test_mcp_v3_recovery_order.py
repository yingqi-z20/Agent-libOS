from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.mcp.continuations import McpContinuationManager
from agent_libos.mcp.subscriptions import McpSubscriptionManager
from agent_libos.mcp.tasks import McpRemoteTaskManager
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.human import HumanRequest, HumanRequestStatus
from agent_libos.runtime.builder import RuntimeBuilder
from agent_libos.runtime.lifecycle import RuntimeLifecycle
from agent_libos.runtime.task_runs import TaskRunManager
from agent_libos.storage import (
    McpContinuationRecord,
    McpRemoteTaskRecord,
    McpSubscriptionRecord,
    SQLiteStore,
    UnitOfWork,
)


_CREATED_AT = "2026-08-11T00:00:00Z"
_EXPIRES_AT = "2026-08-12T00:00:00Z"


def _tasks_config() -> AgentLibOSConfig:
    return AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256="f" * 64,
        )
    )


def _interrupted_continuation() -> McpContinuationRecord:
    return McpContinuationRecord(
        continuation_id="continuation-interrupted-before-preflight",
        server_id="server.v3",
        server_spec_sha256="0" * 64,
        server_generation=3,
        owner_id="runtime",
        auth_principal_sha256="1" * 64,
        auth_scope_sha256="2" * 64,
        request_sha256="3" * 64,
        effect_id="effect-mcp-recovery",
        capability_sha256="4" * 64,
        data_flow_sha256="5" * 64,
        human_request_id="human-mcp-recovery",
        broker_ref="mcp-continuation:opaque-recovery",
        broker_value_sha256="6" * 64,
        status="dispatching",
        revision=0,
        expires_at=_EXPIRES_AT,
        metadata={"automatic_retry_disabled": True},
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )


def _interrupted_remote_task() -> McpRemoteTaskRecord:
    return McpRemoteTaskRecord(
        task_ref="remote-task-interrupted-before-preflight",
        server_id="server.v3",
        server_spec_sha256="0" * 64,
        server_generation=3,
        owner_id="runtime",
        auth_principal_sha256="1" * 64,
        auth_scope_sha256="2" * 64,
        origin_request_sha256="3" * 64,
        origin_effect_id="effect-mcp-recovery",
        human_request_id=None,
        broker_ref="mcp-task:opaque-recovery",
        remote_id_sha256="6" * 64,
        status="update_dispatching",
        revision=0,
        expires_at=_EXPIRES_AT,
        poll_interval_ms=500,
        status_message_sha256=None,
        result_ref=None,
        result_sha256=None,
        metadata={"automatic_retry_disabled": True},
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )


def _interrupted_subscription() -> McpSubscriptionRecord:
    return McpSubscriptionRecord(
        subscription_id="subscription-interrupted-before-preflight",
        server_id="server.v3",
        server_spec_sha256="0" * 64,
        server_generation=3,
        owner_id="runtime",
        auth_principal_sha256="1" * 64,
        auth_scope_sha256="2" * 64,
        requested_filter_sha256="3" * 64,
        acknowledged_filter_sha256="4" * 64,
        status="active",
        queue_limit=2,
        event_max_bytes=4096,
        received_count=0,
        dropped_count=0,
        revision=0,
        last_event_at=None,
        metadata={"automatic_retry_disabled": True},
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )


def _insert_interrupted_records(
    store: SQLiteStore,
) -> tuple[McpContinuationRecord, McpRemoteTaskRecord, McpSubscriptionRecord]:
    store.insert_human_request(
        HumanRequest(
            request_id="human-mcp-recovery",
            pid="runtime",
            human="owner",
            payload={"type": "question", "context": {}},
            status=HumanRequestStatus.PENDING,
            decision=None,
            blocking=True,
            created_at=_CREATED_AT,
            updated_at=_CREATED_AT,
        )
    )
    uow = UnitOfWork(store)
    continuation = uow.mcp_continuations.insert(_interrupted_continuation())
    remote_task = uow.mcp_remote_tasks.insert(_interrupted_remote_task())
    subscription = uow.mcp_subscriptions.insert(_interrupted_subscription())
    return continuation, remote_task, subscription


def test_task_run_preflight_failure_performs_zero_mcp_recovery_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    originals = _insert_interrupted_records(store)
    cas_calls = {"continuation": 0, "remote_task": 0, "subscription": 0}
    original_continuation_cas = store.compare_and_swap_mcp_continuation
    original_remote_task_cas = store.compare_and_swap_mcp_remote_task
    original_subscription_cas = store.compare_and_swap_mcp_subscription

    def counted_continuation_cas(*args: object, **kwargs: object) -> bool:
        cas_calls["continuation"] += 1
        return original_continuation_cas(*args, **kwargs)

    def counted_remote_task_cas(*args: object, **kwargs: object) -> bool:
        cas_calls["remote_task"] += 1
        return original_remote_task_cas(*args, **kwargs)

    def counted_subscription_cas(*args: object, **kwargs: object) -> bool:
        cas_calls["subscription"] += 1
        return original_subscription_cas(*args, **kwargs)

    def reject_task_run_payloads(_manager: TaskRunManager) -> None:
        raise ValidationError("injected TaskRun integrity failure")

    monkeypatch.setattr(
        store,
        "compare_and_swap_mcp_continuation",
        counted_continuation_cas,
    )
    monkeypatch.setattr(
        store,
        "compare_and_swap_mcp_remote_task",
        counted_remote_task_cas,
    )
    monkeypatch.setattr(
        store,
        "compare_and_swap_mcp_subscription",
        counted_subscription_cas,
    )
    monkeypatch.setattr(
        TaskRunManager,
        "validate_recoverable_payloads",
        reject_task_run_payloads,
    )
    try:
        builder = RuntimeBuilder.configured(Runtime, config=_tasks_config())
        with pytest.raises(ValidationError, match="TaskRun integrity failure"):
            builder.from_store(store)

        uow = UnitOfWork(store)
        assert cas_calls == {
            "continuation": 0,
            "remote_task": 0,
            "subscription": 0,
        }
        assert uow.mcp_continuations.get(originals[0].continuation_id) == originals[0]
        assert uow.mcp_remote_tasks.get(originals[1].task_ref) == originals[1]
        assert uow.mcp_subscriptions.get(originals[2].subscription_id) == originals[2]
        assert store._admission_commit_guard is None  # noqa: SLF001
    finally:
        store.close()


def test_all_mcp_reconciliation_follows_preflight_in_one_recovery_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    observed_lease_tokens: list[int] = []
    active_leases: list[tuple[int, RuntimeLifecycle]] = []
    next_lease_token = 0
    original_lease = RuntimeLifecycle.recovery_lease
    original_validate = TaskRunManager.validate_recoverable_payloads
    original_continuation = McpContinuationManager.reconcile_after_restart
    original_remote_task = McpRemoteTaskManager.reconcile_after_restart
    original_subscription = McpSubscriptionManager.reconcile_after_restart

    @contextmanager
    def tracked_lease(lifecycle: RuntimeLifecycle) -> Iterator[None]:
        nonlocal next_lease_token
        next_lease_token += 1
        token = next_lease_token
        with original_lease(lifecycle):
            active_leases.append((token, lifecycle))
            try:
                yield
            finally:
                assert active_leases.pop() == (token, lifecycle)

    def observe(name: str) -> None:
        assert active_leases, f"{name} ran outside the startup recovery lease"
        token, lifecycle = active_leases[-1]
        lifecycle.require_recovery_lease()
        order.append(name)
        observed_lease_tokens.append(token)

    def tracked_validate(manager: TaskRunManager) -> None:
        observe("task-run-integrity")
        original_validate(manager)

    def tracked_continuation(manager: McpContinuationManager) -> int:
        observe("continuation-reconcile")
        return original_continuation(manager)

    def tracked_remote_task(manager: McpRemoteTaskManager) -> int:
        observe("remote-task-reconcile")
        return original_remote_task(manager)

    def tracked_subscription(manager: McpSubscriptionManager) -> int:
        observe("subscription-reconcile")
        return original_subscription(manager)

    monkeypatch.setattr(RuntimeLifecycle, "recovery_lease", tracked_lease)
    monkeypatch.setattr(
        TaskRunManager,
        "validate_recoverable_payloads",
        tracked_validate,
    )
    monkeypatch.setattr(
        McpContinuationManager,
        "reconcile_after_restart",
        tracked_continuation,
    )
    monkeypatch.setattr(
        McpRemoteTaskManager,
        "reconcile_after_restart",
        tracked_remote_task,
    )
    monkeypatch.setattr(
        McpSubscriptionManager,
        "reconcile_after_restart",
        tracked_subscription,
    )

    runtime = Runtime.open(":memory:", config=_tasks_config())
    try:
        assert order == [
            "task-run-integrity",
            "continuation-reconcile",
            "remote-task-reconcile",
            "subscription-reconcile",
        ]
        assert len(set(observed_lease_tokens)) == 1
        assert runtime.recovered_mcp_continuations == 0
        assert runtime.recovered_mcp_remote_tasks == 0
        assert runtime.recovered_mcp_subscriptions == 0
    finally:
        runtime.close()
