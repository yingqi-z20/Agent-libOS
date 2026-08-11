from __future__ import annotations

import asyncio
import threading
import time
from contextvars import ContextVar
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.mcp import (
    McpPage,
    McpResource,
    McpResourceSpec,
    McpServerManifestV3,
    McpSubscription,
    McpSubscriptionEvent,
    McpSubscriptionSession,
    McpSubscriptionStatus,
)
from agent_libos.mcp.client import (
    McpClientBinding,
    current_mcp_client_binding,
    mcp_transport_spec_from_v3,
)
from agent_libos.mcp.runtime_bridge import mcp_connection_fence
from agent_libos.mcp.subscriptions import McpSubscriptionManager
from agent_libos.primitives.mcp import _McpSubscriptionLoopRunner
from agent_libos.models import (
    CapabilityRight,
    McpHeaderSpec,
    McpHttpTransportSpec,
    McpProtocolMode,
    ResourceBudget,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.runtime.builder import RuntimeBuilder
from agent_libos.runtime.task_runs import TaskRunManager
from agent_libos.storage import McpSubscriptionRecord, SQLiteStore, UnitOfWork


_SERVER_ID = "modern-subscriptions"
_SECRET_ENV = "AGENT_LIBOS_MCP_SUBSCRIPTION_SECRET"
_SECRET = "subscription-provider-secret"


def _manifest() -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id=_SERVER_ID,
        transport="streamable_http",
        http=McpHttpTransportSpec(
            url="http://127.0.0.1:8765/mcp",
            headers={"Authorization": McpHeaderSpec(env=_SECRET_ENV)},
        ),
        timeout_s=1.0,
        max_request_bytes=4096,
        max_response_bytes=16_384,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        resources=(
            McpResourceSpec(
                resource_id="status",
                remote_uri="file:///provider/status",
            ),
        ),
        subscriptions=("resourcesListChanged",),
    )


class _SubscriptionProvider:
    def __init__(self) -> None:
        self.listen_count = 0
        self.receive_count = 0
        self.close_count = 0
        self.owner: str | None = None
        self.listen_thread_id: int | None = None
        self.close_thread_id: int | None = None
        self.on_listen: Callable[[], None] | None = None
        self.expected_filters = ("resourcesListChanged",)
        self._events: list[McpSubscriptionEvent] = [
            McpSubscriptionEvent(
                sequence=0,
                event_type="resourcesListChanged",
                payload={"credential": _SECRET, "changed": True},
                received_at="2026-08-11T00:00:00+00:00",
            )
        ]

    async def listen(
        self,
        server: Any,
        filters: tuple[str, ...],
        *,
        deadline: float,
    ) -> McpSubscriptionSession:
        assert server.server_id == _SERVER_ID
        assert filters == self.expected_filters
        assert deadline > time.monotonic()
        binding = current_mcp_client_binding()
        assert binding.manifest.server_id == _SERVER_ID
        assert _SECRET in binding.sensitive_values
        self.owner = binding.owner_id
        self.listen_thread_id = threading.get_ident()
        self.listen_count += 1
        if self.on_listen is not None:
            self.on_listen()
        owner_task = asyncio.create_task(
            asyncio.Event().wait(),
            name="test-mcp-subscription-owner",
        )
        return McpSubscriptionSession(
            handle=object(),
            owner_task=owner_task,
            acknowledged_filters=filters,
        )

    async def receive(self, _handle: Any, *, deadline: float) -> McpSubscriptionEvent:
        assert deadline > time.monotonic()
        self.receive_count += 1
        if self._events:
            return self._events.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def close(self, _handle: Any) -> None:
        self.close_thread_id = threading.get_ident()
        self.close_count += 1


class _TwoStepSubscriptionProvider(_SubscriptionProvider):
    def __init__(self) -> None:
        super().__init__()
        self._events.clear()
        self.releases = (threading.Event(), threading.Event())
        self.delivered = 0

    async def receive(self, _handle: Any, *, deadline: float) -> McpSubscriptionEvent:
        assert deadline > time.monotonic()
        if self.delivered >= len(self.releases):
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        release = self.releases[self.delivered]
        while not release.is_set():
            await asyncio.sleep(0.001)
        self.delivered += 1
        return McpSubscriptionEvent(
            sequence=0,
            event_type="resourcesListChanged",
            payload={"number": self.delivered},
            received_at="2026-08-11T00:00:00+00:00",
        )


class _OneShotInvalidationProvider(_SubscriptionProvider):
    def __init__(self, event_type: str, subscription_filter: str) -> None:
        super().__init__()
        self._events.clear()
        self.event_type = event_type
        self.expected_filters = (subscription_filter,)
        self.release = threading.Event()
        self.delivered = False

    async def receive(self, _handle: Any, *, deadline: float) -> McpSubscriptionEvent:
        assert deadline > time.monotonic()
        if self.delivered:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        while not self.release.is_set():
            await asyncio.sleep(0.001)
        self.delivered = True
        return McpSubscriptionEvent(
            sequence=0,
            event_type=self.event_type,
            payload={"changed": True},
            received_at="2026-08-11T00:00:00+00:00",
        )


class _PagingResourceProvider:
    def __init__(self) -> None:
        self.cursors: list[str | None] = []

    async def list_resources(
        self,
        _server: Any,
        cursor: str | None,
        *,
        deadline: float,
    ) -> McpPage[McpResource]:
        assert deadline > time.monotonic()
        self.cursors.append(cursor)
        return McpPage(
            items=(
                McpResource(
                    resource_id="file:///provider/status",
                    name="Status",
                ),
            ),
            next_cursor="provider-page-2" if cursor is None else None,
        )


@pytest.fixture
def subscription_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Runtime, _SubscriptionProvider]:
    monkeypatch.setenv(_SECRET_ENV, _SECRET)
    runtime = Runtime.open(":memory:")
    runtime.mcp.register_server(
        _manifest(),
        actor="runtime",
        require_capability=False,
    )
    provider = _SubscriptionProvider()
    runtime.mcp._modern_subscription_provider = provider  # noqa: SLF001
    try:
        yield runtime, provider
    finally:
        runtime.close()


def _spawn(runtime: Runtime) -> str:
    return runtime.process.spawn(
        image="base-agent:v0",
        goal="consume bounded MCP notifications",
        resource_budget=ResourceBudget(max_mcp_bytes=1_000_000),
    )


def test_host_subscription_lifecycle_is_protected_pending_first_and_secret_safe(
    subscription_runtime: tuple[Runtime, _SubscriptionProvider],
) -> None:
    runtime, provider = subscription_runtime

    def assert_pending() -> None:
        effects = [
            item
            for item in runtime.store.list_external_effects()
            if item.provider == "mcp"
            and item.operation == "subscriptions.start"
        ]
        assert effects
        assert effects[-1].effect_state == "pending"

    provider.on_listen = assert_pending
    active = runtime.mcp.start_subscription(
        _SERVER_ID,
        filters=("resourcesListChanged",),
        actor="gui",
    )

    assert active.status is McpSubscriptionStatus.ACTIVE
    assert provider.owner == "gui"
    assert provider.listen_count == 1
    assert runtime.mcp.subscription_status(
        active.subscription_id,
        actor="gui",
    ).status is McpSubscriptionStatus.ACTIVE

    events: tuple[McpSubscriptionEvent, ...] = ()
    for _ in range(100):
        events = runtime.mcp.subscription_events(
            active.subscription_id,
            actor="gui",
        )
        if events:
            break
        time.sleep(0.005)
    assert len(events) == 1
    assert events[0].sequence == 1
    assert events[0].provenance == "untrusted_mcp_notification"
    assert _SECRET not in repr(events[0].payload)

    closed = runtime.mcp.stop_subscription(
        active.subscription_id,
        actor="gui",
    )
    assert closed.status is McpSubscriptionStatus.CLOSED
    assert provider.close_count == 1

    effects = {
        item.operation: item
        for item in runtime.store.list_external_effects()
        if item.provider == "mcp" and item.operation.startswith("subscriptions.")
    }
    assert {
        "subscriptions.start",
        "subscriptions.status",
        "subscriptions.events",
        "subscriptions.stop",
    } <= set(effects)
    assert effects["subscriptions.start"].rollback_class.value == "rollbackable"
    assert effects["subscriptions.stop"].rollback_class.value == "irreversible"
    assert effects["subscriptions.start"].provider_metadata[
        "protected_operation"
    ]["contract_name"] == "primitive.mcp.subscriptions.start.internal"
    assert effects["subscriptions.stop"].provider_metadata[
        "protected_operation"
    ]["contract_name"] == "primitive.mcp.subscriptions.stop.internal"
    assert any(
        item.action == "primitive.mcp.subscriptions.start"
        for item in runtime.audit.trace()
    )


class _InjectedSubscriptionFatal(BaseException):
    pass


@pytest.mark.parametrize(
    "error_type",
    (RuntimeError, _InjectedSubscriptionFatal),
    ids=("exception", "base-exception"),
)
def test_post_listen_evidence_failure_aborts_unpublished_subscription(
    subscription_runtime: tuple[Runtime, _SubscriptionProvider],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    runtime, provider = subscription_runtime
    manager = runtime._mcp_subscription_manager  # noqa: SLF001

    def reject_evidence(**_kwargs: Any) -> None:
        rows = runtime.uow.mcp_subscriptions.list(server_id=_SERVER_ID)
        assert len(rows) == 1 and rows[0].status == "starting"
        assert manager._live == {}  # noqa: SLF001
        assert len(manager._opening) == 1  # noqa: SLF001
        assert provider.receive_count == 0
        raise error_type("injected subscription evidence failure")

    monkeypatch.setattr(runtime.mcp, "_modern_success_evidence", reject_evidence)
    with pytest.raises(error_type, match="injected subscription evidence failure"):
        runtime.mcp.start_subscription(
            _SERVER_ID,
            filters=("resourcesListChanged",),
            actor="gui",
        )

    rows = runtime.uow.mcp_subscriptions.list(server_id=_SERVER_ID)
    assert len(rows) == 1 and rows[0].status == "lost"
    assert manager._live == {}  # noqa: SLF001
    assert manager._opening == {}  # noqa: SLF001
    assert provider.receive_count == 0
    assert provider.close_count == 1
    assert provider.close_thread_id == provider.listen_thread_id


def test_post_listen_deadline_aborts_before_subscription_publication(
    subscription_runtime: tuple[Runtime, _SubscriptionProvider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = subscription_runtime
    manager = runtime._mcp_subscription_manager  # noqa: SLF001
    armed = threading.Event()
    provider.on_listen = armed.set
    original_remaining = runtime.mcp._remaining_timeout  # noqa: SLF001

    def expire_after_listen(deadline: float) -> float:
        if armed.is_set():
            raise TimeoutError("injected post-listen deadline")
        return original_remaining(deadline)

    monkeypatch.setattr(runtime.mcp, "_remaining_timeout", expire_after_listen)
    with pytest.raises(TimeoutError, match="injected post-listen deadline"):
        runtime.mcp.start_subscription(
            _SERVER_ID,
            filters=("resourcesListChanged",),
            actor="gui",
        )

    rows = runtime.uow.mcp_subscriptions.list(server_id=_SERVER_ID)
    assert len(rows) == 1 and rows[0].status == "lost"
    assert manager._live == {}  # noqa: SLF001
    assert manager._opening == {}  # noqa: SLF001
    assert provider.receive_count == 0
    assert provider.close_count == 1
    assert provider.close_thread_id == provider.listen_thread_id


def test_subscription_active_cas_rolls_back_with_failed_effect_settlement(
    subscription_runtime: tuple[Runtime, _SubscriptionProvider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = subscription_runtime
    manager = runtime._mcp_subscription_manager  # noqa: SLF001
    original_commit = manager.commit_prepared_start_deferred
    observed_inside_transaction: list[str] = []

    def commit_then_fail(settlement: Any) -> None:
        original_commit(settlement)
        current = runtime.uow.mcp_subscriptions.get(settlement.subscription_id)
        assert current is not None
        observed_inside_transaction.append(current.status)
        raise RuntimeError("injected subscription settlement failure")

    monkeypatch.setattr(manager, "commit_prepared_start_deferred", commit_then_fail)
    with pytest.raises(RuntimeError, match="injected subscription settlement failure"):
        runtime.mcp.start_subscription(
            _SERVER_ID,
            filters=("resourcesListChanged",),
            actor="gui",
        )

    assert observed_inside_transaction == ["active"]
    rows = runtime.uow.mcp_subscriptions.list(server_id=_SERVER_ID)
    assert len(rows) == 1 and rows[0].status == "lost"
    assert manager._live == {}  # noqa: SLF001
    assert manager._opening == {}  # noqa: SLF001
    assert provider.receive_count == 0
    assert provider.close_count == 1


def test_post_commit_finalize_failure_closes_and_marks_subscription_lost(
    subscription_runtime: tuple[Runtime, _SubscriptionProvider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = subscription_runtime
    manager = runtime._mcp_subscription_manager  # noqa: SLF001

    async def reject_finalize(_settlement: Any) -> McpSubscription:
        raise RuntimeError("injected subscription finalize failure")

    monkeypatch.setattr(manager, "finalize_prepared_start", reject_finalize)
    with pytest.raises(RuntimeError, match="injected subscription finalize failure"):
        runtime.mcp.start_subscription(
            _SERVER_ID,
            filters=("resourcesListChanged",),
            actor="gui",
        )

    rows = runtime.uow.mcp_subscriptions.list(server_id=_SERVER_ID)
    assert len(rows) == 1 and rows[0].status == "lost"
    assert manager._live == {}  # noqa: SLF001
    assert manager._opening == {}  # noqa: SLF001
    assert provider.receive_count == 0
    assert provider.close_count == 1


def test_process_subscription_capabilities_owner_and_async_reads(
    subscription_runtime: tuple[Runtime, _SubscriptionProvider],
) -> None:
    runtime, provider = subscription_runtime
    pid = _spawn(runtime)
    runtime.capability.grant(
        pid,
        f"mcp:{_SERVER_ID}:subscription:catalog",
        [CapabilityRight.WRITE],
        issued_by="test",
    )
    runtime.capability.grant(
        pid,
        f"mcp_server:{_SERVER_ID}",
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )
    active = runtime.mcp.start_subscription(
        _SERVER_ID,
        filters=("resourcesListChanged",),
        actor=pid,
    )
    handle = f"mcp_subscription:{active.subscription_id}"
    runtime.capability.grant(
        pid,
        handle,
        [CapabilityRight.READ, CapabilityRight.WRITE],
        issued_by="test",
    )

    status = asyncio.run(
        runtime.mcp.asubscription_status(active.subscription_id, actor=pid)
    )
    assert status.status is McpSubscriptionStatus.ACTIVE
    process = runtime.process.get(pid)
    assert process.resource_usage.mcp_request_bytes > 0
    assert process.resource_usage.mcp_response_bytes > 0

    other = _spawn(runtime)
    runtime.capability.grant(
        other,
        handle,
        [CapabilityRight.READ],
        issued_by="test",
    )
    with pytest.raises(CapabilityDenied, match="another owner"):
        runtime.mcp.subscription_status(active.subscription_id, actor=other)

    runtime.mcp.stop_subscription(active.subscription_id, actor=pid)
    assert provider.close_count == 1


def test_runtime_shutdown_closes_live_stream_and_reopen_marks_handle_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Shutdown never lets a background stream reuse a fenced caller lease."""

    monkeypatch.setenv(_SECRET_ENV, _SECRET)
    database = tmp_path / "mcp-subscription-restart.sqlite"
    runtime = Runtime.open(database)
    provider = _SubscriptionProvider()
    provider._events.clear()  # keep the stream live until Runtime shutdown
    runtime.mcp.register_server(
        _manifest(),
        actor="runtime",
        require_capability=False,
    )
    runtime.mcp._modern_subscription_provider = provider  # noqa: SLF001
    active = runtime.mcp.start_subscription(
        _SERVER_ID,
        filters=("resourcesListChanged",),
        actor="gui",
    )
    durable = runtime.uow.mcp_subscriptions.get(active.subscription_id)
    assert durable is not None and durable.status == "active"
    runner = runtime.mcp._modern_subscription_runner  # noqa: SLF001
    assert runner is not None
    owner_thread = runner._thread  # noqa: SLF001
    owner_loop = runner._loop  # noqa: SLF001

    receipt = runtime.close()
    assert receipt == {
        "ok": True,
        "already_shutdown": False,
        "reason": "runtime.close",
    }
    assert provider.close_count == 1
    assert not owner_thread.is_alive()
    assert owner_loop.is_closed()
    assert asyncio.run(runtime._mcp_connection_supervisor.snapshot()) == ()

    reopened = Runtime.open(database)
    try:
        recovered = reopened.uow.mcp_subscriptions.get(active.subscription_id)
        assert recovered is not None
        assert recovered.status == "lost"
        assert recovered.metadata["reason_code"] == "runtime_restart"
        public = reopened.mcp.subscription_status(
            active.subscription_id,
            actor="gui",
        )
        assert public.status is McpSubscriptionStatus.LOST
        assert public.lost_reason == "runtime_restart"
        with pytest.raises(
            NotFound,
            match=f"MCP subscription events unavailable: {active.subscription_id}",
        ):
            reopened.mcp.subscription_events(
                active.subscription_id,
                actor="gui",
            )
    finally:
        reopened.close()
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_runtime_poll_acknowledges_events_and_reclaims_single_slot_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SECRET_ENV, _SECRET)
    config = dataclass_replace(
        DEFAULT_CONFIG,
        mcp=dataclass_replace(
            DEFAULT_CONFIG.mcp,
            subscription_queue_events=1,
        ),
    )
    runtime = Runtime.open(":memory:", config=config)
    provider = _TwoStepSubscriptionProvider()
    try:
        runtime.mcp.register_server(
            _manifest(),
            actor="runtime",
            require_capability=False,
        )
        runtime.mcp._modern_subscription_provider = provider  # noqa: SLF001
        active = runtime.mcp.start_subscription(
            _SERVER_ID,
            filters=("resourcesListChanged",),
            actor="gui",
        )

        provider.releases[0].set()
        first: tuple[McpSubscriptionEvent, ...] = ()
        for _ in range(200):
            first = runtime.mcp.subscription_events(
                active.subscription_id,
                after=0,
                limit=1,
                actor="gui",
            )
            if first:
                break
            time.sleep(0.005)
        assert [event.sequence for event in first] == [1]

        with pytest.raises(
            ValidationError,
            match="cursor is stale or has multiple readers",
        ):
            runtime.mcp.subscription_events(
                active.subscription_id,
                after=0,
                actor="gui",
            )

        provider.releases[1].set()
        second: tuple[McpSubscriptionEvent, ...] = ()
        for _ in range(200):
            second = runtime.mcp.subscription_events(
                active.subscription_id,
                after=1,
                limit=1,
                actor="gui",
            )
            if second:
                break
            time.sleep(0.005)
        assert [event.sequence for event in second] == [2]
        assert runtime.mcp.subscription_status(
            active.subscription_id,
            actor="gui",
        ).status is McpSubscriptionStatus.ACTIVE
        durable = runtime.uow.mcp_subscriptions.get(active.subscription_id)
        assert durable is not None
        assert (durable.received_count, durable.dropped_count) == (2, 0)
        runtime.mcp.stop_subscription(active.subscription_id, actor="gui")
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("subscription_filter", "event_type"),
    (
        ("toolsListChanged", "toolsListChanged"),
        ("promptsListChanged", "promptsListChanged"),
        ("resourcesListChanged", "resourcesListChanged"),
        ("resourceSubscriptions", "resourceUpdated"),
    ),
)
def test_subscription_change_event_invalidates_opaque_catalog_cursors(
    subscription_filter: str,
    event_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SECRET_ENV, _SECRET)
    runtime = Runtime.open(":memory:")
    provider = _OneShotInvalidationProvider(event_type, subscription_filter)
    resources = _PagingResourceProvider()
    try:
        runtime.mcp.register_server(
            dataclass_replace(
                _manifest(),
                subscriptions=(subscription_filter,),
            ),
            actor="runtime",
            require_capability=False,
        )
        runtime.mcp._modern_subscription_provider = provider  # noqa: SLF001
        runtime.mcp._modern_client.resource_provider = resources  # noqa: SLF001

        first_page = runtime.mcp.list_resources(_SERVER_ID, actor="gui")
        stale_cursor = first_page.next_cursor
        assert isinstance(stale_cursor, str)
        active = runtime.mcp.start_subscription(
            _SERVER_ID,
            filters=(subscription_filter,),
            actor="gui",
        )
        provider.release.set()
        events: tuple[McpSubscriptionEvent, ...] = ()
        for _ in range(200):
            events = runtime.mcp.subscription_events(
                active.subscription_id,
                after=0,
                actor="gui",
            )
            if events:
                break
            time.sleep(0.005)
        assert [event.event_type for event in events] == [event_type]

        with pytest.raises(
            ValidationError,
            match="cursor is expired or unknown",
        ):
            runtime.mcp.list_resources(
                _SERVER_ID,
                cursor=stale_cursor,
                actor="gui",
            )
        assert resources.cursors == [None]

        refreshed = runtime.mcp.list_resources(_SERVER_ID, actor="gui")
        assert len(refreshed.items) == 1
        assert resources.cursors == [None, None]
        runtime.mcp.stop_subscription(active.subscription_id, actor="gui")
    finally:
        runtime.close()


def _interrupted_subscription_record() -> McpSubscriptionRecord:
    timestamp = datetime.now(UTC).isoformat(timespec="microseconds")
    return McpSubscriptionRecord(
        subscription_id="subscription-interrupted-before-preflight",
        server_id=_SERVER_ID,
        server_spec_sha256="a" * 64,
        server_generation=1,
        owner_id="gui",
        auth_principal_sha256="b" * 64,
        auth_scope_sha256="c" * 64,
        requested_filter_sha256="d" * 64,
        acknowledged_filter_sha256="e" * 64,
        status="active",
        queue_limit=2,
        event_max_bytes=4096,
        received_count=0,
        dropped_count=0,
        revision=0,
        last_event_at=None,
        metadata={"automatic_retry_disabled": True},
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_failed_task_run_preflight_does_not_reconcile_subscriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    repository = UnitOfWork(store).mcp_subscriptions
    original = _interrupted_subscription_record()
    repository.insert(original)
    cas_calls = 0
    original_cas = store.compare_and_swap_mcp_subscription

    def counted_cas(
        subscription_id: str,
        *,
        expected_revision: int,
        replacement: McpSubscriptionRecord,
    ) -> bool:
        nonlocal cas_calls
        cas_calls += 1
        return original_cas(
            subscription_id,
            expected_revision=expected_revision,
            replacement=replacement,
        )

    def reject_task_run_payloads(_manager: TaskRunManager) -> None:
        raise ValidationError("injected TaskRun integrity failure")

    monkeypatch.setattr(store, "compare_and_swap_mcp_subscription", counted_cas)
    monkeypatch.setattr(
        TaskRunManager,
        "validate_recoverable_payloads",
        reject_task_run_payloads,
    )
    try:
        with pytest.raises(ValidationError, match="TaskRun integrity failure"):
            RuntimeBuilder.configured(Runtime).from_store(store)

        assert cas_calls == 0
        assert repository.get(original.subscription_id) == original
        assert store._admission_commit_guard is None  # noqa: SLF001
    finally:
        store.close()


def test_subscription_recovery_runs_after_task_run_integrity_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    original_validate = TaskRunManager.validate_recoverable_payloads
    original_reconcile = McpSubscriptionManager.reconcile_after_restart

    def tracked_validate(manager: TaskRunManager) -> None:
        order.append("task-run-integrity")
        original_validate(manager)

    def tracked_reconcile(manager: McpSubscriptionManager) -> int:
        order.append("subscription-reconcile")
        return original_reconcile(manager)

    monkeypatch.setattr(
        TaskRunManager,
        "validate_recoverable_payloads",
        tracked_validate,
    )
    monkeypatch.setattr(
        McpSubscriptionManager,
        "reconcile_after_restart",
        tracked_reconcile,
    )

    runtime = Runtime.open(":memory:")
    try:
        assert order == ["task-run-integrity", "subscription-reconcile"]
        assert runtime.recovered_mcp_subscriptions == 0
    finally:
        runtime.close()


def test_subscription_denial_precedes_store_registry_and_provider(
    subscription_runtime: tuple[Runtime, _SubscriptionProvider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = subscription_runtime
    pid = _spawn(runtime)
    monkeypatch.setattr(
        runtime.uow.mcp_subscriptions,
        "get",
        lambda _subscription_id: (_ for _ in ()).throw(
            AssertionError("denied caller must not inspect subscription Store state")
        ),
    )
    monkeypatch.setattr(
        runtime.uow.extensions,
        "get_mcp_v3_server",
        lambda _server_id: (_ for _ in ()).throw(
            AssertionError("denied caller must not inspect the MCP registry")
        ),
    )

    with pytest.raises(CapabilityDenied):
        runtime.mcp.subscription_status("mcp-subscription-local", actor=pid)
    assert provider.listen_count == 0


def test_subscription_filters_must_be_global_and_manifest_allowlisted(
    subscription_runtime: tuple[Runtime, _SubscriptionProvider],
) -> None:
    runtime, provider = subscription_runtime
    with pytest.raises(ValidationError, match="unsupported"):
        runtime.mcp.start_subscription(
            _SERVER_ID,
            filters=("resources/updated",),
            actor="gui",
        )
    with pytest.raises(ValidationError, match="not declared"):
        runtime.mcp.start_subscription(
            _SERVER_ID,
            filters=("toolsListChanged",),
            actor="gui",
        )
    assert provider.listen_count == 0


def test_subscription_runner_cleans_context_and_owns_runtime_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SECRET_ENV, _SECRET)
    loop_failures: list[dict[str, object]] = []
    original_new_event_loop = asyncio.events.new_event_loop

    def monitored_new_event_loop() -> asyncio.AbstractEventLoop:
        loop = original_new_event_loop()
        loop.set_exception_handler(
            lambda _loop, context: loop_failures.append(dict(context))
        )
        return loop

    monkeypatch.setattr(asyncio.events, "new_event_loop", monitored_new_event_loop)
    caller_context: ContextVar[str | None] = ContextVar(
        "test_mcp_subscription_caller_context",
        default=None,
    )
    observed_context: list[str | None] = []
    runtime = Runtime.open(":memory:")
    closed = False
    provider = _SubscriptionProvider()
    provider.on_listen = lambda: observed_context.append(caller_context.get())
    try:
        runtime.mcp.register_server(
            _manifest(),
            actor="runtime",
            require_capability=False,
        )
        runtime.mcp._modern_subscription_provider = provider  # noqa: SLF001
        token = caller_context.set("must-not-escape")
        try:
            active = runtime.mcp.start_subscription(
                _SERVER_ID,
                filters=("resourcesListChanged",),
                actor="gui",
            )
        finally:
            caller_context.reset(token)
        assert active.status is McpSubscriptionStatus.ACTIVE
        assert observed_context == [None]
        assert provider.listen_thread_id is not None
        for _ in range(100):
            if runtime.lifecycle._active_leases == 0:  # noqa: SLF001
                break
            time.sleep(0.001)
        assert runtime.lifecycle._active_leases == 0  # noqa: SLF001

        runtime.close()
        closed = True
        assert provider.close_count == 1
        assert provider.close_thread_id == provider.listen_thread_id
        assert loop_failures == []
    finally:
        if not closed:
            runtime.close()


def test_subscription_runner_timeout_cancels_and_drains_without_loop_noise(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    loop_failures: list[dict[str, object]] = []
    original_new_event_loop = asyncio.events.new_event_loop

    def monitored_new_event_loop() -> asyncio.AbstractEventLoop:
        loop = original_new_event_loop()
        loop.set_exception_handler(
            lambda _loop, context: loop_failures.append(dict(context))
        )
        return loop

    monkeypatch.setattr(asyncio.events, "new_event_loop", monitored_new_event_loop)

    class Manager:
        async def close(self) -> None:
            return None

    dispatch: ContextVar[dict[str, Any] | None] = ContextVar(
        "test_mcp_subscription_timeout_dispatch",
        default=None,
    )
    binding = McpClientBinding(
        manifest=_manifest(),
        registry_generation=1,
        owner_id="gui",
        sensitive_values=(_SECRET,),
        runtime_environment={_SECRET_ENV: _SECRET},
    )
    runner = _McpSubscriptionLoopRunner(
        Manager(),
        dispatch_context_var=dispatch,
    )
    cancelled = threading.Event()
    token = dispatch.set({"binding": binding})

    async def blocked() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    try:
        with pytest.raises(TimeoutError, match="MCP subscription deadline exceeded"):
            runner.run(
                blocked,
                deadline=time.monotonic() + 0.02,
                binding=binding,
            )
        assert cancelled.is_set()
    finally:
        dispatch.reset(token)
        runner.close()

    assert not runner._thread.is_alive()  # noqa: SLF001
    assert runner._loop.is_closed()  # noqa: SLF001
    assert loop_failures == []
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
