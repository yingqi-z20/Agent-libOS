from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_libos.mcp import (
    McpConnectionFence,
    McpConnectionSupervisor,
    McpSubscriptionEvent,
    McpSubscriptionManager,
    McpSubscriptionPolicy,
    McpSubscriptionSession,
    McpSubscriptionStatus,
    McpTasksSubscriptionFence,
)
from agent_libos.mcp.manifest import MCP_TASKS_EXTENSION_ID
from agent_libos.mcp.client import (
    McpClientBinding,
    bind_mcp_client_binding,
    mcp_transport_spec_from_v3,
)
from agent_libos.mcp.manifest import McpResourceSpec, McpServerManifestV3
from agent_libos.mcp.manifest import McpTasksExtensionSpec
from agent_libos.mcp.runtime_bridge import McpSupervisedSdkSessionFactory
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import McpProtocolMode, McpServerSpec, McpStdioTransportSpec
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.runtime import Runtime
from agent_libos.primitives.mcp import McpPrimitive
from agent_libos.storage.mcp_v7 import McpSubscriptionRecord
from agent_libos.models.exceptions import ValidationError


@dataclass
class _Session:
    acknowledged_filters: Any = ()
    close_count: int = 0

    async def aclose(self) -> None:
        self.close_count += 1


def _fence(**values: object) -> McpConnectionFence:
    selected: dict[str, object] = {
        "server_id": "server",
        "server_spec_sha256": "a" * 64,
        "registry_generation": 1,
        "owner": "pid:1",
        "auth_principal_sha256": "b" * 64,
        "auth_generation": 1,
    }
    selected.update(values)
    return McpConnectionFence(**selected)  # type: ignore[arg-type]


def _server() -> McpServerSpec:
    return McpServerSpec(
        schema_version=2,
        server_id="server",
        transport="stdio",
        stdio=McpStdioTransportSpec(command="server"),
        tools=[],
        timeout_s=1.0,
        max_request_bytes=1024,
        max_response_bytes=4096,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
    )


def _async_factory(value: Any):
    async def factory() -> Any:
        return value

    return factory


def test_supervisor_reuses_only_exact_read_fence_and_never_mutation() -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor()
        sessions: list[_Session] = []

        async def factory() -> _Session:
            session = _Session()
            sessions.append(session)
            return session

        first = await supervisor.acquire(_fence(), "read", factory, reusable=True)
        second = await supervisor.acquire(_fence(), "read", factory, reusable=True)
        assert first.connection_id == second.connection_id
        assert first.lease_token != second.lease_token
        await supervisor.release(first)
        assert sessions[0].close_count == 0
        assert (await supervisor.snapshot())[0].lease_count == 1
        # A duplicate release cannot consume the other caller's lease.
        await supervisor.release(first)
        assert sessions[0].close_count == 0
        mutation_a = await supervisor.acquire(_fence(), "mutation", factory, reusable=True)
        mutation_b = await supervisor.acquire(_fence(), "mutation", factory, reusable=True)
        assert mutation_a.connection_id != mutation_b.connection_id
        assert len(sessions) == 3
        await supervisor.close()
        assert [item.close_count for item in sessions] == [1, 1, 1]

    asyncio.run(exercise())


def test_registry_and_auth_fence_changes_close_without_reconnect() -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor()
        session = _Session()
        await supervisor.acquire(
            _fence(), "read", _async_factory(session), reusable=True
        )
        await supervisor.invalidate_server(
            "server", current_spec_sha256="c" * 64, current_registry_generation=2
        )
        assert session.close_count == 1
        assert await supervisor.snapshot() == ()

        auth_session = _Session()
        await supervisor.acquire(
            _fence(), "read", _async_factory(auth_session), reusable=True
        )
        await supervisor.invalidate_auth("b" * 64, current_generation=2)
        assert auth_session.close_count == 1
        assert await supervisor.snapshot() == ()

    asyncio.run(exercise())


class _EventsProvider:
    def __init__(self, events: list[McpSubscriptionEvent | BaseException]) -> None:
        self.events = events
        self.listen_count = 0
        self.close_count = 0
        self.handles: list[_Session] = []
        self.owner_tasks: list[asyncio.Task[None]] = []

    async def listen(self, server, filters, *, deadline):
        del server, deadline
        self.listen_count += 1
        handle = _Session(acknowledged_filters=filters)
        self.handles.append(handle)

        async def own() -> None:
            await asyncio.Future()

        owner_task = asyncio.create_task(own())
        self.owner_tasks.append(owner_task)
        return McpSubscriptionSession(
            handle=handle,
            owner_task=owner_task,
            acknowledged_filters=filters,
        )

    async def receive(self, handle, *, deadline):
        del handle, deadline
        if not self.events:
            await asyncio.Future()
        selected = self.events.pop(0)
        if isinstance(selected, BaseException):
            raise selected
        return selected

    async def close(self, handle) -> None:
        self.close_count += 1
        await handle.aclose()


class _PushEventsProvider(_EventsProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.pending: asyncio.Queue[McpSubscriptionEvent | BaseException] = (
            asyncio.Queue()
        )

    async def receive(self, handle, *, deadline):
        del handle, deadline
        selected = await self.pending.get()
        if isinstance(selected, BaseException):
            raise selected
        return selected


def _event(payload: object) -> McpSubscriptionEvent:
    return McpSubscriptionEvent(
        sequence=0,
        event_type="resourcesListChanged",
        payload=payload,  # type: ignore[arg-type]
        received_at="2026-08-11T00:00:00+00:00",
    )


async def _wait_for_status(
    manager: McpSubscriptionManager,
    subscription_id: str,
    status: McpSubscriptionStatus,
) -> None:
    for _ in range(100):
        if (await manager.status(subscription_id)).status is status:
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"subscription did not reach {status}")


def test_subscription_disconnect_marks_lost_without_reconnect_or_replay() -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor()
        provider = _EventsProvider([ConnectionError("wire lost")])
        store = _MemorySubscriptionStore()
        manager = McpSubscriptionManager(supervisor, store=store)
        record = await manager.start(
            _server(), _fence(), provider, ("resourcesListChanged",)
        )
        await _wait_for_status(manager, record.subscription_id, McpSubscriptionStatus.LOST)
        durable = store.get(record.subscription_id)
        assert durable is not None
        assert durable.metadata["reason_code"] == "subscription_connection_lost"
        assert "wire lost" not in repr(durable.to_dict())
        assert provider.listen_count == 1
        await _wait_until(lambda: provider.close_count == 1)
        assert provider.close_count == 1
        assert provider.handles[0].close_count == 1
        assert await supervisor.snapshot() == ()

    asyncio.run(exercise())


def test_subscription_oversize_and_backpressure_fail_closed() -> None:
    async def exercise() -> None:
        for policy, events in (
            (McpSubscriptionPolicy(event_max_bytes=8), [_event("x" * 32)]),
            (
                McpSubscriptionPolicy(queue_events=1),
                [_event({"n": 1}), _event({"n": 2})],
            ),
        ):
            supervisor = McpConnectionSupervisor()
            provider = _EventsProvider(events)
            manager = McpSubscriptionManager(supervisor, policy=policy)
            record = await manager.start(
                _server(), _fence(), provider, ("resourcesListChanged",)
            )
            await _wait_for_status(manager, record.subscription_id, McpSubscriptionStatus.LOST)
            assert provider.listen_count == 1
            await manager.close()

    asyncio.run(exercise())


def test_subscription_poll_acknowledges_batch_and_reclaims_queue_capacity() -> None:
    async def exercise() -> None:
        provider = _PushEventsProvider()
        manager = McpSubscriptionManager(
            McpConnectionSupervisor(),
            policy=McpSubscriptionPolicy(queue_events=1),
        )
        public = await manager.start(
            _server(), _fence(), provider, ("resourcesListChanged",)
        )

        provider.pending.put_nowait(_event({"number": 1}))
        first = await _wait_for_event(manager, public.subscription_id)
        assert first.sequence == 1
        assert first.payload == {"number": 1}

        provider.pending.put_nowait(_event({"number": 2}))
        second: tuple[McpSubscriptionEvent, ...] = ()
        for _ in range(200):
            second = await manager.events(
                public.subscription_id,
                after=first.sequence,
                limit=1,
            )
            if second:
                break
            await asyncio.sleep(0)
        assert len(second) == 1
        assert second[0].sequence == 2
        assert second[0].payload == {"number": 2}
        assert (
            await manager.status(public.subscription_id)
        ).status is McpSubscriptionStatus.ACTIVE
        await manager.stop(public.subscription_id)

    asyncio.run(exercise())


def test_subscription_event_cursor_rejects_stale_and_future_positions() -> None:
    async def exercise() -> None:
        provider = _PushEventsProvider()
        manager = McpSubscriptionManager(McpConnectionSupervisor())
        public = await manager.start(
            _server(), _fence(), provider, ("resourcesListChanged",)
        )
        provider.pending.put_nowait(_event({"number": 1}))
        first = await _wait_for_event(manager, public.subscription_id)

        for cursor in (0, first.sequence + 1):
            with pytest.raises(
                ValidationError,
                match="cursor is stale or has multiple readers",
            ):
                await manager.events(public.subscription_id, after=cursor)
        assert await manager.events(
            public.subscription_id,
            after=first.sequence,
        ) == ()
        await manager.stop(public.subscription_id)

    asyncio.run(exercise())


def test_subscription_event_cursor_fails_closed_for_concurrent_readers() -> None:
    async def exercise() -> None:
        provider = _PushEventsProvider()
        manager = McpSubscriptionManager(McpConnectionSupervisor())
        public = await manager.start(
            _server(), _fence(), provider, ("resourcesListChanged",)
        )
        provider.pending.put_nowait(_event({"number": 1}))
        await _wait_until(
            lambda: manager._live[public.subscription_id].queue.qsize() == 1  # noqa: SLF001
        )

        outcomes = await asyncio.gather(
            manager.events(public.subscription_id, after=0),
            manager.events(public.subscription_id, after=0),
            return_exceptions=True,
        )
        batches = [item for item in outcomes if isinstance(item, tuple)]
        failures = [item for item in outcomes if isinstance(item, BaseException)]
        assert len(batches) == 1 and len(batches[0]) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], ValidationError)
        assert "multiple readers" in str(failures[0])
        await manager.stop(public.subscription_id)

    asyncio.run(exercise())


def test_subscription_stop_is_explicit_and_closes_stream() -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor()
        provider = _EventsProvider([])
        manager = McpSubscriptionManager(supervisor)
        record = await manager.start(
            _server(), _fence(), provider, ("resourcesListChanged",)
        )
        closed = await manager.stop(record.subscription_id)
        assert closed.status is McpSubscriptionStatus.CLOSED
        assert provider.listen_count == 1
        assert provider.close_count == 1
        assert provider.handles[0].close_count == 1
        assert await supervisor.snapshot() == ()

    asyncio.run(exercise())


def test_active_leases_are_not_idle_expired_and_absolute_expiry_is_lost() -> None:
    async def exercise() -> None:
        session = _Session()
        reasons: list[str] = []
        supervisor = McpConnectionSupervisor(
            idle_ttl_s=0.01,
            absolute_ttl_s=0.08,
        )
        first = await supervisor.acquire(
            _fence(),
            "read",
            _async_factory(session),
            reusable=True,
            on_lost=lambda _connection_id, reason: reasons.append(reason),
        )
        second = await supervisor.acquire(
            _fence(), "read", _async_factory(_Session()), reusable=True
        )
        await supervisor.release(first)
        await asyncio.sleep(0.03)
        snapshot = await supervisor.snapshot()
        assert len(snapshot) == 1
        assert snapshot[0].lease_count == 1
        assert session.close_count == 0
        await _wait_until(
            lambda: session.close_count == 1
            and reasons == ["absolute_ttl_expired"]
        )
        assert reasons == ["absolute_ttl_expired"]
        assert await supervisor.snapshot() == ()
        # The lease was revoked by the absolute fence, so release is harmless.
        await supervisor.release(second)

    asyncio.run(exercise())


def test_supervisor_open_and_close_io_never_holds_global_lock() -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor(close_timeout_s=0.02)
        factory_entered = asyncio.Event()
        factory_release = asyncio.Event()
        late_session = _Session()

        async def slow_factory() -> _Session:
            factory_entered.set()
            try:
                await factory_release.wait()
            except asyncio.CancelledError:
                # A cancellation-resistant Provider may still hand back a
                # handle.  The supervisor must close that late result without
                # leaving the opener in its catalog.
                await factory_release.wait()
            return late_session

        opening = asyncio.create_task(
            supervisor.acquire(
                _fence(), "read", slow_factory, task_affine=False
            )
        )
        await factory_entered.wait()
        # close() must invalidate an in-flight factory without waiting for it.
        await asyncio.wait_for(supervisor.close(), timeout=0.1)
        with pytest.raises(RuntimeError, match="invalidated"):
            await opening
        factory_release.set()
        await _wait_until(lambda: late_session.close_count == 1)

        close_release = asyncio.Event()
        close_entered = asyncio.Event()

        class SlowClose:
            async def aclose(self) -> None:
                close_entered.set()
                try:
                    await close_release.wait()
                except asyncio.CancelledError:
                    await close_release.wait()

        second = McpConnectionSupervisor(close_timeout_s=0.02)
        managed = await second.acquire(
            _fence(), "read", _async_factory(SlowClose())
        )
        invalidating = asyncio.create_task(
            second.invalidate_server("server", current_spec_sha256="c" * 64)
        )
        await close_entered.wait()
        # A different connection can open while the first Provider close stalls.
        other = await asyncio.wait_for(
            second.acquire(
                _fence(server_id="other"), "read", _async_factory(_Session())
            ),
            timeout=0.05,
        )
        await asyncio.wait_for(invalidating, timeout=0.1)
        close_release.set()
        await second.release(other)
        await second.release(managed)

    asyncio.run(exercise())


def test_invalid_subscription_deadline_creates_no_slot_or_durable_row() -> None:
    async def exercise() -> None:
        store = _MemorySubscriptionStore()
        supervisor = McpConnectionSupervisor()
        manager = McpSubscriptionManager(
            supervisor,
            store=store,
            policy=McpSubscriptionPolicy(max_open=1),
        )
        provider = _EventsProvider([])
        with pytest.raises(TimeoutError, match="opening deadline"):
            await manager.start(
                _server(),
                _fence(),
                provider,
                ("resourcesListChanged",),
                deadline=asyncio.get_running_loop().time() - 1,
            )
        assert provider.listen_count == 0
        assert store.records == {}

        opened = await manager.start(
            _server(), _fence(), provider, ("resourcesListChanged",)
        )
        assert opened.status is McpSubscriptionStatus.ACTIVE
        await manager.close()

    asyncio.run(exercise())


def test_subscription_absolute_expiry_marks_lost_and_closes_exactly_once() -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor(
            idle_ttl_s=0.005,
            absolute_ttl_s=0.03,
        )
        provider = _EventsProvider([])
        manager = McpSubscriptionManager(
            supervisor,
            policy=McpSubscriptionPolicy(exchange_timeout_s=1.0),
        )
        record = await manager.start(
            _server(), _fence(), provider, ("resourcesListChanged",)
        )
        await _wait_for_status(manager, record.subscription_id, McpSubscriptionStatus.LOST)
        assert (await manager.status(record.subscription_id)).lost_reason == (
            "absolute_ttl_expired"
        )
        await _wait_until(lambda: provider.close_count == 1)
        assert provider.close_count == 1
        assert provider.handles[0].close_count == 1
        assert await supervisor.snapshot() == ()

    asyncio.run(exercise())


def test_terminal_subscriptions_release_live_capacity_but_status_remains_readable() -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor()
        manager = McpSubscriptionManager(
            supervisor,
            policy=McpSubscriptionPolicy(max_open=1),
        )
        first_provider = _EventsProvider([])
        first = await manager.start(
            _server(), _fence(), first_provider, ("resourcesListChanged",)
        )
        closed = await manager.stop(first.subscription_id)
        assert (await manager.status(first.subscription_id)) == closed

        second_provider = _EventsProvider([ConnectionError("lost")])
        second = await manager.start(
            _server(), _fence(), second_provider, ("resourcesListChanged",)
        )
        await _wait_for_status(manager, second.subscription_id, McpSubscriptionStatus.LOST)
        assert (await manager.status(second.subscription_id)).status is McpSubscriptionStatus.LOST

        third_provider = _EventsProvider([])
        third = await manager.start(
            _server(), _fence(), third_provider, ("resourcesListChanged",)
        )
        assert third.status is McpSubscriptionStatus.ACTIVE
        await manager.close()

    asyncio.run(exercise())


def test_sync_registry_invalidation_immediately_fences_and_persists_lost() -> None:
    async def exercise() -> None:
        store = _MemorySubscriptionStore()
        supervisor = McpConnectionSupervisor()
        manager = McpSubscriptionManager(supervisor, store=store)
        provider = _EventsProvider([_event({"before_invalidation": True})])
        public = await manager.start(
            _server(),
            _fence(auth_scope_sha256="c" * 64),
            provider,
            ("resourcesListChanged",),
        )
        await _wait_until(
            lambda: manager._live[public.subscription_id].queue.qsize() == 1  # noqa: SLF001
        )
        manager.invalidate_server_nowait("server")
        supervisor.invalidate_server_nowait("server")

        # The post-commit hook is synchronous: authority/catalog state and
        # durable status are revoked before it returns, while Provider close
        # is scheduled and cannot make registry commit appear rolled back.
        assert await supervisor.snapshot() == ()
        terminal = await manager.status(public.subscription_id)
        assert terminal.status is McpSubscriptionStatus.LOST
        assert terminal.lost_reason == "registry_fence_changed"
        retained = await manager.events(public.subscription_id, after=0)
        assert len(retained) == 1
        assert retained[0].payload == {"before_invalidation": True}
        with pytest.raises(ValidationError, match="cursor is stale"):
            await manager.events(public.subscription_id, after=0)
        assert await manager.events(
            public.subscription_id,
            after=retained[-1].sequence,
        ) == ()
        durable = store.get(public.subscription_id)
        assert durable is not None
        assert durable.status == "lost"
        assert durable.metadata["reason_code"] == "subscription_connection_lost"
        await _wait_until(lambda: provider.close_count == 1)
        assert provider.handles[0].close_count == 1
        assert provider.owner_tasks[0].done()

        # Terminal rows no longer consume max_open capacity.
        replacement = _EventsProvider([])
        reopened = await manager.start(
            _server(),
            _fence(auth_scope_sha256="c" * 64),
            replacement,
            ("resourcesListChanged",),
        )
        assert reopened.status is McpSubscriptionStatus.ACTIVE
        await manager.close()

    asyncio.run(exercise())


class _MemorySubscriptionStore:
    def __init__(self, records: tuple[McpSubscriptionRecord, ...] = ()) -> None:
        self.records = {record.subscription_id: record for record in records}

    def insert(self, record: McpSubscriptionRecord) -> McpSubscriptionRecord:
        if record.subscription_id in self.records:
            raise RuntimeError("duplicate")
        self.records[record.subscription_id] = record
        return record

    def get(self, subscription_id: str) -> McpSubscriptionRecord | None:
        return self.records.get(subscription_id)

    def list(self, **filters: Any) -> tuple[McpSubscriptionRecord, ...]:
        status = filters.get("status")
        limit = filters.get("limit", 100)
        return tuple(
            record
            for record in self.records.values()
            if status is None or record.status == status
        )[:limit]

    def compare_and_swap(
        self,
        subscription_id: str,
        *,
        expected_revision: int,
        replacement: McpSubscriptionRecord,
    ) -> bool:
        current = self.records.get(subscription_id)
        if current is None or current.revision != expected_revision:
            return False
        self.records[subscription_id] = replacement
        return True


def test_background_event_store_mutation_uses_fresh_admission_and_clean_context() -> None:
    caller_lease = contextvars.ContextVar("mcp_test_caller_lease", default=None)

    class Admission:
        def __init__(self) -> None:
            self.active = contextvars.ContextVar(
                "mcp_test_mutation_admission", default=False
            )
            self.count = 0

        @contextlib.contextmanager
        def admit(self, *, read_only: bool = False):
            assert read_only is False
            self.count += 1
            token = self.active.set(True)
            try:
                yield
            finally:
                self.active.reset(token)

    class GuardedStore(_MemorySubscriptionStore):
        def __init__(self, admission: Admission) -> None:
            super().__init__()
            self.admission = admission
            self.event_contexts: list[str | None] = []

        def insert(self, record: McpSubscriptionRecord) -> McpSubscriptionRecord:
            assert self.admission.active.get() is True
            return super().insert(record)

        def compare_and_swap(
            self,
            subscription_id: str,
            *,
            expected_revision: int,
            replacement: McpSubscriptionRecord,
        ) -> bool:
            assert self.admission.active.get() is True
            current = self.records.get(subscription_id)
            if current is not None and replacement.received_count > current.received_count:
                self.event_contexts.append(caller_lease.get())
            return super().compare_and_swap(
                subscription_id,
                expected_revision=expected_revision,
                replacement=replacement,
            )

    async def exercise() -> None:
        admission = Admission()
        store = GuardedStore(admission)
        manager = McpSubscriptionManager(
            McpConnectionSupervisor(),
            store=store,
            admission=admission,
        )
        token = caller_lease.set("short-lived-caller-admission")
        try:
            public = await manager.start(
                _server(),
                _fence(),
                _EventsProvider([_event({"ok": True})]),
                ("resourcesListChanged",),
            )
            await _wait_for_event(manager, public.subscription_id)
        finally:
            caller_lease.reset(token)
        assert store.event_contexts == [None]
        assert admission.count >= 3  # insert, ACTIVE CAS, event counter CAS
        await manager.stop(public.subscription_id)

    asyncio.run(exercise())


def _durable_subscription(
    subscription_id: str,
    status: str,
    *,
    revision: int = 0,
) -> McpSubscriptionRecord:
    now = datetime.now(UTC).isoformat(timespec="microseconds")
    return McpSubscriptionRecord(
        subscription_id=subscription_id,
        server_id="server",
        server_spec_sha256="a" * 64,
        server_generation=1,
        owner_id="pid:1",
        auth_principal_sha256="b" * 64,
        auth_scope_sha256="c" * 64,
        requested_filter_sha256="d" * 64,
        acknowledged_filter_sha256=None,
        status=status,
        queue_limit=2,
        event_max_bytes=1024,
        received_count=0,
        dropped_count=0,
        revision=revision,
        last_event_at=None,
        metadata={"automatic_retry_disabled": True},
        created_at=now,
        updated_at=now,
    )


def test_reopen_cas_marks_every_nonterminal_durable_record_lost() -> None:
    async def exercise() -> None:
        records = tuple(
            _durable_subscription(f"subscription-{status}", status)
            for status in ("starting", "active", "stopping")
        ) + (_durable_subscription("subscription-stopped", "stopped"),)
        store = _MemorySubscriptionStore(records)
        manager = McpSubscriptionManager(McpConnectionSupervisor(), store=store)
        for original in records[:3]:
            durable = store.get(original.subscription_id)
            assert durable is not None
            assert durable.status == "lost"
            assert durable.revision == original.revision + 1
            assert durable.metadata["reason_code"] == "runtime_restart"
            public = await manager.status(original.subscription_id)
            assert public.status is McpSubscriptionStatus.LOST
            assert public.lost_reason == "runtime_restart"
        assert (
            await manager.status("subscription-stopped")
        ).status is McpSubscriptionStatus.CLOSED

    asyncio.run(exercise())


def test_subscription_restart_reconciliation_can_be_deferred_and_is_counted() -> None:
    records = tuple(
        _durable_subscription(f"subscription-deferred-{status}", status)
        for status in ("starting", "active", "stopping")
    )
    store = _MemorySubscriptionStore(records)
    manager = McpSubscriptionManager(
        McpConnectionSupervisor(),
        store=store,
        reconcile_on_start=False,
    )

    assert tuple(store.records.values()) == records
    assert manager.reconcile_after_restart() == 3
    assert manager.reconcile_after_restart() == 0
    assert all(record.status == "lost" for record in store.records.values())


def test_subscription_reconcile_on_start_requires_an_exact_boolean() -> None:
    with pytest.raises(
        ValidationError,
        match="subscription reconcile_on_start is invalid",
    ):
        McpSubscriptionManager(
            McpConnectionSupervisor(),
            reconcile_on_start=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("subscription_filter", "event_type"),
    (
        ("toolsListChanged", "toolsListChanged"),
        ("promptsListChanged", "promptsListChanged"),
        ("resourcesListChanged", "resourcesListChanged"),
        ("resourceSubscriptions", "resourceUpdated"),
    ),
)
def test_projected_change_events_invalidate_only_local_server_state(
    subscription_filter: str,
    event_type: str,
) -> None:
    async def exercise() -> None:
        store = _MemorySubscriptionStore()
        invalidated: list[str] = []

        def invalidate_after_receipt_commit(server_id: str) -> None:
            records = tuple(store.records.values())
            assert len(records) == 1
            assert records[0].received_count == 1
            invalidated.append(server_id)

        event = dataclass_replace(
            _event({"changed": True}),
            event_type=event_type,
        )
        manager = McpSubscriptionManager(
            McpConnectionSupervisor(),
            store=store,
            local_cache_invalidator=invalidate_after_receipt_commit,
        )
        public = await manager.start(
            _server(),
            _fence(),
            _EventsProvider([event]),
            (subscription_filter,),
        )
        await _wait_for_event(manager, public.subscription_id)
        assert invalidated == ["server"]
        await manager.stop(public.subscription_id)

    asyncio.run(exercise())


@pytest.mark.parametrize("event_type", ("unknownChanged", "ui/resourceUpdated"))
def test_unknown_and_apps_event_types_never_reach_local_invalidator(
    event_type: str,
) -> None:
    async def exercise() -> None:
        invalidated: list[str] = []
        manager = McpSubscriptionManager(
            McpConnectionSupervisor(),
            local_cache_invalidator=invalidated.append,
        )
        public = await manager.start(
            _server(),
            _fence(),
            _EventsProvider(
                [
                    dataclass_replace(
                        _event({"changed": True}),
                        event_type=event_type,
                    )
                ]
            ),
            ("resourcesListChanged",),
        )
        await _wait_for_status(
            manager,
            public.subscription_id,
            McpSubscriptionStatus.LOST,
        )
        assert invalidated == []
        await manager.stop(public.subscription_id)

    asyncio.run(exercise())


def test_subscription_local_cache_invalidator_must_be_callable() -> None:
    with pytest.raises(TypeError, match="local cache invalidator"):
        McpSubscriptionManager(
            McpConnectionSupervisor(),
            local_cache_invalidator=object(),  # type: ignore[arg-type]
        )


def test_resource_update_event_maps_only_to_resource_subscription_filter() -> None:
    async def exercise() -> None:
        update = McpSubscriptionEvent(
            sequence=0,
            event_type="resourceUpdated",
            payload={"resource_handle": "mcp-resource:test"},
            received_at="provider-time",
        )
        accepted_manager = McpSubscriptionManager(McpConnectionSupervisor())
        accepted = await accepted_manager.start(
            _server(),
            _fence(),
            _EventsProvider([update]),
            ("resourceSubscriptions",),
        )
        projected = await _wait_for_event(accepted_manager, accepted.subscription_id)
        assert projected.event_type == "resourceUpdated"
        await accepted_manager.stop(accepted.subscription_id)

        rejected_manager = McpSubscriptionManager(McpConnectionSupervisor())
        rejected = await rejected_manager.start(
            _server(),
            _fence(),
            _EventsProvider([update]),
            ("resourcesListChanged",),
        )
        await _wait_for_status(
            rejected_manager,
            rejected.subscription_id,
            McpSubscriptionStatus.LOST,
        )

    asyncio.run(exercise())


def test_task_subscription_requires_exact_manifest_and_host_extension_pins() -> None:
    async def exercise() -> None:
        digest = "d" * 64
        valid_fence = McpTasksSubscriptionFence(
            extension_id=MCP_TASKS_EXTENSION_ID,
            manifest_spec_sha256=digest,
            host_spec_sha256=digest,
        )
        for selected_fence in (
            None,
            McpTasksSubscriptionFence(
                extension_id=MCP_TASKS_EXTENSION_ID,
                manifest_spec_sha256=digest,
                host_spec_sha256="e" * 64,
            ),
        ):
            provider = _EventsProvider([])
            manager = McpSubscriptionManager(McpConnectionSupervisor())
            with pytest.raises(ValidationError, match="Tasks extension"):
                await manager.start(
                    _server(),
                    _fence(),
                    provider,
                    ("taskIds",),
                    tasks_extension_fence=selected_fence,
                )
            assert provider.listen_count == 0

        no_projector = _EventsProvider([])
        with pytest.raises(ValidationError, match="result-claim projector"):
            await McpSubscriptionManager(McpConnectionSupervisor()).start(
                _server(),
                _fence(),
                no_projector,
                ("taskIds",),
                tasks_extension_fence=valid_fence,
            )
        assert no_projector.listen_count == 0

        remote_bearer = "remote-task-bearer-must-not-escape"
        projected: list[str] = []

        class Projector:
            def project_task_notification(self, *, event, fence, sensitive_values):
                assert fence == _fence()
                assert event.payload == {
                    "taskId": remote_bearer,
                    "status": "working",
                }
                assert sensitive_values == ()
                projected.append(remote_bearer)
                return dataclass_replace(
                    event,
                    payload={
                        "task_ref": "mcp-task-local-ref",
                        "status": "working",
                    },
                )

        provider = _EventsProvider(
            [
                McpSubscriptionEvent(
                    sequence=0,
                    event_type="taskStatus",
                    payload={"taskId": remote_bearer, "status": "working"},
                    received_at="provider-time",
                )
            ]
        )
        manager = McpSubscriptionManager(
            McpConnectionSupervisor(),
            task_event_projector=Projector(),
        )
        public = await manager.start(
            _server(),
            _fence(),
            provider,
            ("taskIds",),
            tasks_extension_fence=valid_fence,
        )
        event = await _wait_for_event(manager, public.subscription_id)
        assert event.payload == {
            "task_ref": "mcp-task-local-ref",
            "status": "working",
        }
        assert remote_bearer not in repr(event)
        assert projected == [remote_bearer]
        await manager.stop(public.subscription_id)

    asyncio.run(exercise())


def test_primitive_revalidates_task_subscription_pin_at_dispatch_time() -> None:
    digest = "d" * 64
    primitive = object.__new__(McpPrimitive)
    primitive.config = dataclass_replace(
        DEFAULT_CONFIG,
        mcp=dataclass_replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
        ),
    )
    manifest = McpServerManifestV3(
        schema_version=3,
        server_id="task-subscriptions",
        transport="stdio",
        stdio=McpStdioTransportSpec(command="server"),
        timeout_s=1.0,
        max_request_bytes=1024,
        max_response_bytes=4096,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        resources=(
            McpResourceSpec(
                resource_id="status",
                remote_uri="fixture://status",
            ),
        ),
        subscriptions=("taskIds",),
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=digest,
        ),
    )
    fence = primitive._tasks_subscription_fence(  # noqa: SLF001
        manifest,
        ("taskIds",),
    )
    assert fence == McpTasksSubscriptionFence(
        extension_id=MCP_TASKS_EXTENSION_ID,
        manifest_spec_sha256=digest,
        host_spec_sha256=digest,
    )

    primitive.config = dataclass_replace(
        primitive.config,
        mcp=dataclass_replace(
            primitive.config.mcp,
            tasks_extension_spec_sha256="e" * 64,
        ),
    )
    with pytest.raises(ValidationError, match="Host-pinned Tasks extension"):
        primitive._tasks_subscription_fence(  # noqa: SLF001
            manifest,
            ("taskIds",),
        )


def test_durable_lost_cas_failure_cannot_erase_terminal_latch() -> None:
    class LostRaceStore(_MemorySubscriptionStore):
        fail_lost = False

        def compare_and_swap(
            self,
            subscription_id: str,
            *,
            expected_revision: int,
            replacement: McpSubscriptionRecord,
        ) -> bool:
            if self.fail_lost and replacement.status == "lost":
                return False
            return super().compare_and_swap(
                subscription_id,
                expected_revision=expected_revision,
                replacement=replacement,
            )

    async def exercise() -> None:
        store = LostRaceStore()
        manager = McpSubscriptionManager(McpConnectionSupervisor(), store=store)
        provider = _EventsProvider([ConnectionError("wire lost")])
        public = await manager.start(
            _server(), _fence(), provider, ("resourcesListChanged",)
        )
        store.fail_lost = True
        await _wait_for_status(
            manager, public.subscription_id, McpSubscriptionStatus.LOST
        )
        assert await manager.events(public.subscription_id) == ()
        assert (
            await manager.stop(public.subscription_id)
        ).status is McpSubscriptionStatus.LOST
        await _wait_until(lambda: provider.close_count == 1)

    asyncio.run(exercise())


async def _wait_for_event(
    manager: McpSubscriptionManager,
    subscription_id: str,
) -> McpSubscriptionEvent:
    for _ in range(200):
        selected = await manager.events(subscription_id)
        if selected:
            return selected[0]
        await asyncio.sleep(0)
    raise AssertionError("subscription did not receive an event")


def test_notifications_are_strict_redacted_app_free_and_payload_free_in_store() -> None:
    async def exercise() -> None:
        secret = "opaque-provider-credential"
        store = _MemorySubscriptionStore()
        provider = _EventsProvider(
            [
                McpSubscriptionEvent(
                    sequence=999,
                    event_type="resourcesListChanged",
                    payload={
                        secret: f"prefix {secret} suffix",
                        "nested": {
                            "UI/widget": {"html": secret},
                            "ui/resourceUri": "ui://host/private",
                            "Ui/Visibility": ["model", "app"],
                            "UI/CSP": {"connectDomains": ["example.invalid"]},
                            "io.modelcontextprotocol/ui/card": secret,
                        },
                    },
                    received_at="2000-01-01T00:00:00+00:00",
                    provenance="untrusted_mcp_notification",
                )
            ]
        )
        manager = McpSubscriptionManager(
            McpConnectionSupervisor(),
            store=store,
        )
        public = await manager.start(
            _server(), _fence(auth_scope_sha256="c" * 64), provider,
            ("resourcesListChanged",),
            sensitive_values=(secret,),
        )
        event = await _wait_for_event(manager, public.subscription_id)
        assert event.sequence == 1
        assert event.received_at != "2000-01-01T00:00:00+00:00"
        assert secret not in repr(event.payload)
        assert event.payload["nested"] == {}
        durable = store.get(public.subscription_id)
        assert durable is not None and durable.received_count == 1
        durable_projection = durable.to_dict()
        assert secret not in repr(durable_projection)
        assert set(durable_projection) == {
            "subscription_id", "server_id", "server_spec_sha256",
            "server_generation", "owner_id", "auth_principal_sha256",
            "auth_scope_sha256", "requested_filter_sha256",
            "acknowledged_filter_sha256", "status", "queue_limit",
            "event_max_bytes", "received_count", "dropped_count", "revision",
            "last_event_at", "metadata", "created_at", "updated_at",
        }
        await manager.stop(public.subscription_id)

        for bad_event in (
            _event(("not", "strict", "json")),
            _event({"uri": " \tUI://host/app"}),
            McpSubscriptionEvent(
                sequence=0,
                event_type="promptsListChanged",
                payload={},
                received_at="provider-time",
            ),
            _event({"mime": "Text/HTML ; profile=\"MCP-APP\""}),
        ):
            bad_provider = _EventsProvider([bad_event])
            bad_manager = McpSubscriptionManager(McpConnectionSupervisor())
            bad = await bad_manager.start(
                _server(), _fence(), bad_provider, ("resourcesListChanged",)
            )
            await _wait_for_status(
                bad_manager, bad.subscription_id, McpSubscriptionStatus.LOST
            )
            await _wait_until(lambda: bad_provider.close_count == 1)
            assert bad_provider.close_count == 1

    asyncio.run(exercise())
def test_acknowledgement_rejects_unbounded_iterables_without_iterating() -> None:
    class ExplodingIterable:
        def __iter__(self):
            raise AssertionError("unbounded acknowledgement was iterated")

    class BadAckProvider(_EventsProvider):
        async def listen(self, server, filters, *, deadline):
            session = await super().listen(server, filters, deadline=deadline)
            return McpSubscriptionSession(
                handle=session.handle,
                owner_task=session.owner_task,
                acknowledged_filters=ExplodingIterable(),  # type: ignore[arg-type]
            )

    async def exercise() -> None:
        provider = BadAckProvider([])
        manager = McpSubscriptionManager(McpConnectionSupervisor())
        with pytest.raises(ValueError, match="bounded array"):
            await manager.start(
                _server(), _fence(), provider, ("resourcesListChanged",)
            )
        assert provider.close_count == 1
        assert provider.handles[0].close_count == 1

    asyncio.run(exercise())


def test_invalid_subscription_owner_task_still_closes_provider_handle() -> None:
    class BadOwnerProvider(_EventsProvider):
        async def listen(self, server, filters, *, deadline):
            session = await super().listen(server, filters, deadline=deadline)
            return McpSubscriptionSession(
                handle=session.handle,
                owner_task=object(),  # type: ignore[arg-type]
                acknowledged_filters=filters,
            )

    async def exercise() -> None:
        provider = BadOwnerProvider([])
        manager = McpSubscriptionManager(McpConnectionSupervisor())
        with pytest.raises(TypeError, match="owner task"):
            await manager.start(
                _server(),
                _fence(),
                provider,
                ("resourcesListChanged",),
            )
        assert provider.close_count == 1
        assert provider.handles[0].close_count == 1
        for owner_task in provider.owner_tasks:
            owner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await owner_task

    asyncio.run(exercise())


def test_start_and_stop_provider_io_is_lock_free_and_stop_is_bounded() -> None:
    class BlockingProvider(_EventsProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.listen_entered = asyncio.Event()
            self.listen_release = asyncio.Event()
            self.receive_entered = asyncio.Event()
            self.receive_release = asyncio.Event()

        async def listen(self, server, filters, *, deadline):
            self.listen_entered.set()
            await self.listen_release.wait()
            return await super().listen(server, filters, deadline=deadline)

        async def receive(self, handle, *, deadline):
            del handle, deadline
            self.receive_entered.set()
            try:
                await self.receive_release.wait()
            except asyncio.CancelledError:
                await self.receive_release.wait()
            return _event({"late": True})

        async def close(self, handle) -> None:
            self.receive_release.set()
            await super().close(handle)

    async def exercise() -> None:
        pending_provider = BlockingProvider()
        pending_manager = McpSubscriptionManager(McpConnectionSupervisor())
        opening = asyncio.create_task(
            pending_manager.start(
                _server(), _fence(), pending_provider, ("resourcesListChanged",)
            )
        )
        await pending_provider.listen_entered.wait()
        await asyncio.wait_for(pending_manager.close(), timeout=0.05)
        pending_provider.listen_release.set()
        with pytest.raises(RuntimeError, match="cancelled"):
            await opening
        assert pending_provider.close_count == 1

        provider = BlockingProvider()
        provider.listen_release.set()
        manager = McpSubscriptionManager(
            McpConnectionSupervisor(close_timeout_s=0.05),
            policy=McpSubscriptionPolicy(exchange_timeout_s=0.01),
        )
        public = await manager.start(
            _server(), _fence(), provider, ("resourcesListChanged",)
        )
        await provider.receive_entered.wait()
        closed = await asyncio.wait_for(manager.stop(public.subscription_id), timeout=0.2)
        assert closed.status is McpSubscriptionStatus.CLOSED
        assert provider.close_count == 1
        assert provider.handles[0].close_count == 1

    asyncio.run(exercise())


async def _wait_until(predicate: Any) -> None:
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not reached")


def test_supervised_sdk_permit_keeps_context_task_affine_and_cancels_only_owner() -> None:
    async def exercise() -> None:
        manifest = McpServerManifestV3(
            schema_version=3,
            server_id="modern",
            transport="stdio",
            stdio=McpStdioTransportSpec(command="modern-server"),
            timeout_s=1.0,
            max_request_bytes=4096,
            max_response_bytes=4096,
            protocol_mode=McpProtocolMode.REVISION_2026_07_28,
            resources=(
                McpResourceSpec(
                    resource_id="status",
                    remote_uri="fixture://status",
                ),
            ),
        )
        binding = McpClientBinding(
            manifest=manifest,
            registry_generation=1,
            owner_id="pid:1",
        )
        server = mcp_transport_spec_from_v3(manifest)
        supervisor = McpConnectionSupervisor()
        entered = asyncio.Event()
        unblock = asyncio.Event()
        task_ids: list[asyncio.Task[Any] | None] = []
        loop_errors: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

        @contextlib.asynccontextmanager
        async def raw_context(
            server_value,
            *,
            deadline,
            binding,
            task_notification_ingress=None,
        ):
            del deadline
            assert task_notification_ingress is None
            assert server_value == server
            assert binding.owner_id == "pid:1"
            task_ids.append(asyncio.current_task())
            entered.set()
            try:
                yield object()
            finally:
                task_ids.append(asyncio.current_task())

        factory = McpSupervisedSdkSessionFactory(supervisor, raw_context)

        async def provider_operation() -> None:
            with bind_mcp_client_binding(binding):
                async with factory(
                    server,
                    deadline=asyncio.get_running_loop().time() + 1,
                ):
                    task_ids.append(asyncio.current_task())
                    await unblock.wait()

        outer_task = asyncio.current_task()
        operation_task = asyncio.create_task(provider_operation())
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        await asyncio.wait_for(
            supervisor.invalidate_server(
                "modern",
                current_spec_sha256="f" * 64,
                current_registry_generation=2,
            ),
            timeout=1.0,
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation_task, timeout=1.0)
        await asyncio.sleep(0)
        loop.set_exception_handler(previous_handler)
        assert task_ids == [operation_task, operation_task, operation_task]
        assert outer_task is not operation_task
        assert not outer_task.cancelled()
        assert loop_errors == []
        assert await supervisor.snapshot() == ()

    asyncio.run(exercise())


def test_release_invalidate_race_closes_session_once() -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor()
        session = _Session()
        managed = await supervisor.acquire(
            _fence(), "read", _async_factory(session)
        )
        await asyncio.gather(
            supervisor.release(managed),
            supervisor.release(managed),
            supervisor.invalidate_server(
                "server",
                current_spec_sha256="f" * 64,
                current_registry_generation=2,
            ),
        )
        assert session.close_count == 1
        assert await supervisor.snapshot() == ()

    asyncio.run(exercise())


def test_connection_owner_revocation_latches_before_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor()
        session = _Session()
        await supervisor.acquire(
            _fence(),
            "read",
            _async_factory(session),
            reusable=True,
        )
        original_invalidate = supervisor._invalidate_nowait  # noqa: SLF001
        secret = "SECRET /private/provider-close"

        def fail_invalidation(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(secret)

        monkeypatch.setattr(supervisor, "_invalidate_nowait", fail_invalidation)
        with pytest.raises(RuntimeError, match="provider-close"):
            supervisor.close_owner_nowait("pid:1")

        factory_called = False

        async def forbidden_factory() -> _Session:
            nonlocal factory_called
            factory_called = True
            return _Session()

        with pytest.raises(RuntimeError, match="owner is closed"):
            await supervisor.acquire(
                _fence(),
                "read",
                forbidden_factory,
                reusable=True,
            )
        assert factory_called is False

        # A terminal-cleanup retry can still detach the old handle, but the
        # monotonic owner denial does not depend on that retry succeeding.
        monkeypatch.setattr(supervisor, "_invalidate_nowait", original_invalidate)
        supervisor.close_owner_nowait("pid:1")
        assert await supervisor.snapshot() == ()
        await _wait_until(lambda: session.close_count == 1)

    asyncio.run(exercise())


def test_subscription_owner_revocation_runs_connection_branch_after_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor()
        manager = McpSubscriptionManager(supervisor)
        provider = _EventsProvider([])
        active = await manager.start(
            _server(),
            _fence(),
            provider,
            ("resourcesListChanged",),
        )
        original_invalidate = manager._invalidate_nowait  # noqa: SLF001

        def fail_local_invalidation(**_kwargs: Any) -> None:
            raise RuntimeError("injected subscription revocation failure")

        monkeypatch.setattr(manager, "_invalidate_nowait", fail_local_invalidation)
        with pytest.raises(RuntimeError, match="subscription revocation"):
            manager.close_owner_nowait("pid:1")

        # The independent connection branch still detaches and schedules the
        # Provider close, while both managers deny all later reuse.
        assert await supervisor.snapshot() == ()
        with pytest.raises(RuntimeError, match="owner is closed"):
            await manager.status(active.subscription_id)
        with pytest.raises(RuntimeError, match="owner is closed"):
            await manager.events(active.subscription_id)
        replacement = _EventsProvider([])
        with pytest.raises(RuntimeError, match="owner is closed"):
            await manager.start(
                _server(),
                _fence(),
                replacement,
                ("resourcesListChanged",),
            )
        assert replacement.listen_count == 0
        with pytest.raises(RuntimeError, match="owner is closed"):
            await supervisor.acquire(
                _fence(),
                "read",
                _async_factory(_Session()),
            )

        monkeypatch.setattr(manager, "_invalidate_nowait", original_invalidate)
        manager.close_owner_nowait("pid:1")
        await _wait_for_status(
            manager,
            active.subscription_id,
            McpSubscriptionStatus.LOST,
        )
        await _wait_until(lambda: provider.close_count == 1)

    asyncio.run(exercise())


def test_subscription_owner_revocation_latches_supervisor_before_local_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        supervisor = McpConnectionSupervisor()
        manager = McpSubscriptionManager(supervisor)
        local_cleanup_entered = threading.Event()
        release_local_cleanup = threading.Event()

        def block_local_cleanup(**_kwargs: Any) -> None:
            local_cleanup_entered.set()
            assert release_local_cleanup.wait(timeout=5.0)

        monkeypatch.setattr(manager, "_invalidate_nowait", block_local_cleanup)
        revocation = asyncio.create_task(
            asyncio.to_thread(manager.close_owner_nowait, "pid:1")
        )
        assert await asyncio.to_thread(local_cleanup_entered.wait, 5.0)

        factory_called = False

        async def forbidden_factory() -> _Session:
            nonlocal factory_called
            factory_called = True
            return _Session()

        try:
            with pytest.raises(RuntimeError, match="owner is closed"):
                await supervisor.acquire(
                    _fence(),
                    "read",
                    forbidden_factory,
                )
            assert factory_called is False
        finally:
            release_local_cleanup.set()
            await revocation

    asyncio.run(exercise())


def test_owner_revocation_keeps_authority_closed_when_provider_close_fails() -> None:
    async def exercise() -> None:
        secret = "SECRET provider close failure"

        class FailingClose:
            close_count = 0

            async def aclose(self) -> None:
                self.close_count += 1
                raise RuntimeError(secret)

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop_errors: list[dict[str, Any]] = []
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            supervisor = McpConnectionSupervisor(close_timeout_s=0.02)
            session = FailingClose()
            await supervisor.acquire(
                _fence(),
                "read",
                _async_factory(session),
                task_affine=False,
            )
            supervisor.close_owner_nowait("pid:1")
            await _wait_until(lambda: session.close_count == 1)
            assert await supervisor.snapshot() == ()

            factory_called = False

            async def forbidden_factory() -> _Session:
                nonlocal factory_called
                factory_called = True
                return _Session()

            with pytest.raises(RuntimeError, match="owner is closed"):
                await supervisor.acquire(_fence(), "read", forbidden_factory)
            supervisor.close_owner_nowait("pid:1")
            await asyncio.sleep(0)
            assert factory_called is False
            assert session.close_count == 1
        finally:
            loop.set_exception_handler(previous_handler)
        assert loop_errors == []

    asyncio.run(exercise())


def test_owner_revocation_keeps_authority_closed_when_provider_close_times_out() -> None:
    async def exercise() -> None:
        class HangingClose:
            def __init__(self) -> None:
                self.close_count = 0
                self.entered = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def aclose(self) -> None:
                self.close_count += 1
                self.entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop_errors: list[dict[str, Any]] = []
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            supervisor = McpConnectionSupervisor(close_timeout_s=0.01)
            session = HangingClose()
            await supervisor.acquire(
                _fence(),
                "read",
                _async_factory(session),
                task_affine=False,
            )
            supervisor.close_owner_nowait("pid:1")
            await asyncio.wait_for(session.entered.wait(), timeout=1.0)
            await asyncio.wait_for(session.cancelled.wait(), timeout=1.0)
            assert await supervisor.snapshot() == ()
            with pytest.raises(RuntimeError, match="owner is closed"):
                await supervisor.acquire(
                    _fence(),
                    "read",
                    _async_factory(_Session()),
                )
            supervisor.close_owner_nowait("pid:1")
            await asyncio.sleep(0)
            assert session.close_count == 1
        finally:
            loop.set_exception_handler(previous_handler)
        assert loop_errors == []

    asyncio.run(exercise())


def test_process_terminal_revokes_mcp_owner_before_other_notifications() -> None:
    calls: list[tuple[str, str]] = []

    class Subscriptions:
        def close_owner_nowait(self, owner: str) -> None:
            calls.append(("mcp", owner))

    class Human:
        def cancel_pending_for_process(self, pid: str, **_keywords: Any) -> None:
            calls.append(("human", pid))

    class ObjectTasks:
        def notify_process_terminal(self, pid: str) -> None:
            calls.append(("object_tasks", pid))

    runtime = object.__new__(Runtime)
    runtime._mcp_subscription_manager = Subscriptions()
    runtime.human = Human()
    runtime.object_tasks = ObjectTasks()
    runtime._notify_process_terminal("pid:terminal")

    assert calls == [
        ("mcp", "pid:terminal"),
        ("human", "pid:terminal"),
        ("object_tasks", "pid:terminal"),
    ]


def test_process_terminal_reports_mcp_revocation_failure_and_continues() -> None:
    calls: list[tuple[str, str]] = []
    secret = "SECRET /private/mcp-revocation"

    class Subscriptions:
        def close_owner_nowait(self, owner: str) -> None:
            calls.append(("mcp", owner))
            raise RuntimeError(secret)

    class Human:
        def cancel_pending_for_process(self, pid: str, **_keywords: Any) -> None:
            calls.append(("human", pid))

    class ObjectTasks:
        def notify_process_terminal(self, pid: str) -> None:
            calls.append(("object_tasks", pid))

    runtime = object.__new__(Runtime)
    runtime._mcp_subscription_manager = Subscriptions()
    runtime.human = Human()
    runtime.object_tasks = ObjectTasks()

    with pytest.raises(RuntimeError, match="terminal_cleanup_failed") as caught:
        runtime._notify_process_terminal("pid:terminal")

    assert calls == [
        ("mcp", "pid:terminal"),
        ("human", "pid:terminal"),
        ("object_tasks", "pid:terminal"),
    ]
    assert secret not in str(caught.value)
    assert caught.value.cleanup_failures[0]["phase"] == "mcp"
    assert caught.value.cleanup_failures[0]["error_type"] == "RuntimeError"
    assert len(
        caught.value.cleanup_failures[0]["exception_text"]["sha256"]
    ) == 64
