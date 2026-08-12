"""Bounded Host manager for MCP ``subscriptions/listen`` streams."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import math
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Callable, Protocol

from agent_libos.mcp._input import canonical_json_bytes, json_sha256, strict_json_value
from agent_libos.mcp.manifest import (
    MCP_TASKS_EXTENSION_ID,
    MCP_V3_SUBSCRIPTION_FILTERS,
)
from agent_libos.mcp.providers import (
    McpSubscriptionProvider,
    McpSubscriptionSession,
)
from agent_libos.mcp.resources import (
    is_mcp_app_mime,
    sanitize_provider_json,
)
from agent_libos.mcp.supervisor import (
    McpConnectionFence,
    McpConnectionSupervisor,
    McpManagedConnection,
)
from agent_libos.mcp.types import (
    McpSubscription,
    McpSubscriptionEvent,
    McpSubscriptionStatus,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.mcp import McpServerSpec
from agent_libos.storage.mcp_v7 import McpSubscriptionRecord


@dataclass(frozen=True)
class McpSubscriptionPolicy:
    max_open: int = 8
    queue_events: int = 256
    event_max_bytes: int = 64 * 1024
    max_lifetime_s: float = 3600.0
    exchange_timeout_s: float = 30.0
    terminal_status_records: int = 256


@dataclass(frozen=True, slots=True)
class McpTasksSubscriptionFence:
    """Exact manifest/Host double pin for the optional Tasks filter."""

    extension_id: str
    manifest_spec_sha256: str
    host_spec_sha256: str


class McpSubscriptionStore(Protocol):
    """Payload-free RuntimeStore v7 repository surface."""

    def insert(self, record: McpSubscriptionRecord) -> McpSubscriptionRecord: ...

    def get(self, subscription_id: str) -> McpSubscriptionRecord | None: ...

    def list(self, **filters: Any) -> tuple[McpSubscriptionRecord, ...]: ...

    def compare_and_swap(
        self,
        subscription_id: str,
        *,
        expected_revision: int,
        replacement: McpSubscriptionRecord,
    ) -> bool: ...


class McpRuntimeAdmission(Protocol):
    """Runtime lifecycle seam for one background Store mutation."""

    def admit(self, *, read_only: bool = False) -> AbstractContextManager[None]: ...


class McpTaskSubscriptionProjector(Protocol):
    """Host result-claim seam for bearer-bearing Tasks notifications."""

    def project_task_notification(
        self,
        *,
        event: McpSubscriptionEvent,
        fence: McpConnectionFence,
        sensitive_values: tuple[str, ...],
    ) -> McpSubscriptionEvent: ...


@dataclass
class _ProviderSession:
    """Supervisor-owned close-once wrapper around a Provider-owned handle."""

    provider: McpSubscriptionProvider
    session: McpSubscriptionSession
    cleanup_done: asyncio.Event
    close_started: bool = False

    async def aclose(self) -> None:
        if self.close_started:
            await self.cleanup_done.wait()
            return
        self.close_started = True
        selected_owner = self.session.owner_task
        owner_task = (
            selected_owner
            if isinstance(selected_owner, asyncio.Task)
            and selected_owner.get_loop() is asyncio.get_running_loop()
            else None
        )
        try:
            if (
                owner_task is not None
                and owner_task is not asyncio.current_task()
                and not owner_task.done()
            ):
                owner_task.cancel()
            close = getattr(self.provider, "close", None)
            if not callable(close) or not inspect.iscoroutinefunction(close):
                raise TypeError("MCP subscription provider close must be asynchronous")
            await close(self.session.handle)
        finally:
            try:
                if (
                    owner_task is not None
                    and owner_task is not asyncio.current_task()
                    and not owner_task.done()
                ):
                    owner_task.cancel()
                    try:
                        await owner_task
                    except asyncio.CancelledError:
                        pass
            finally:
                self.cleanup_done.set()


@dataclass
class _LiveSubscription:
    public: McpSubscription
    connection: McpManagedConnection
    queue: asyncio.Queue[McpSubscriptionEvent]
    task: asyncio.Task[None] | None
    stop: asyncio.Event
    provider: McpSubscriptionProvider
    provider_handle: Any
    durable: McpSubscriptionRecord | None
    sensitive_values: tuple[str, ...]
    listener_task: asyncio.Task[Any]
    cleanup_done: asyncio.Event
    task_event_projector: McpTaskSubscriptionProjector | None
    receive_task: asyncio.Task[Any] | None = None
    next_sequence: int = 1
    event_cursor: int = 0


@dataclass(frozen=True)
class _TerminalSubscription:
    public: McpSubscription
    events: tuple[McpSubscriptionEvent, ...]
    event_cursor: int
    cleanup_done: asyncio.Event


@dataclass
class _OpeningSubscription:
    public: McpSubscription
    fence: McpConnectionFence
    cleanup_done: asyncio.Event
    origin_effect_id: str | None = None
    prepared: _LiveSubscription | None = None
    committed: bool = False
    closed: bool = False


@dataclass(frozen=True)
class McpSubscriptionStartSettlement:
    """Host-only token binding one opened listener to its effect commit.

    The Provider session is already open, but it remains private and cannot
    consume notifications until ``commit_deferred`` advances the durable row in
    the originating protected-effect transaction and ``finalize`` publishes the
    in-memory handle on the owner loop.
    """

    manager: "McpSubscriptionManager"
    opening: _OpeningSubscription

    @property
    def subscription_id(self) -> str:
        return self.opening.public.subscription_id

    def commit_deferred(self) -> None:
        self.manager.commit_prepared_start_deferred(self)

    async def finalize(self) -> McpSubscription:
        return await self.manager.finalize_prepared_start(self)

    async def abort(self, *, reason: str = "subscription_publication_failed") -> None:
        await self.manager.abort_prepared_start(self, reason=reason)


SensitiveValues = Iterable[str] | Callable[[], Iterable[str]]

_EVENT_TYPES_BY_FILTER: dict[str, frozenset[str]] = {
    "toolsListChanged": frozenset({"toolsListChanged"}),
    "promptsListChanged": frozenset({"promptsListChanged"}),
    "resourcesListChanged": frozenset({"resourcesListChanged"}),
    # subscriptions/listen acknowledges the subscription class, while the
    # modern wire notification names the concrete resource change.
    "resourceSubscriptions": frozenset({"resourceUpdated"}),
    # The extension filter is logical ``taskIds`` while the wire method is
    # ``notifications/tasks/status``.  Adapters expose the normalized event
    # name only after a Host result-claim projection hides the remote bearer.
    "taskIds": frozenset({"taskStatus"}),
}

_LOCAL_CACHE_INVALIDATING_EVENT_TYPES = frozenset(
    {
        "toolsListChanged",
        "promptsListChanged",
        "resourcesListChanged",
        "resourceUpdated",
    }
)


class McpSubscriptionManager:
    """Own streams without exposing provider iterators to processes/models.

    Events are inert, untrusted records.  This class deliberately has no
    callback capable of launching a model, Tool, or TaskRun.  RuntimeStore
    receives lifecycle metadata, counters, and digests only--never payloads.
    """

    def __init__(
        self,
        supervisor: McpConnectionSupervisor,
        *,
        policy: McpSubscriptionPolicy | None = None,
        store: McpSubscriptionStore | None = None,
        sanitize_event: Callable[[McpSubscriptionEvent], McpSubscriptionEvent]
        | None = None,
        sensitive_values: SensitiveValues = (),
        admission: McpRuntimeAdmission | None = None,
        task_event_projector: McpTaskSubscriptionProjector | None = None,
        local_cache_invalidator: Callable[[str], None] | None = None,
        reconcile_on_start: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._supervisor = supervisor
        self._policy = policy or McpSubscriptionPolicy()
        self._store = store
        self._sanitize_event = sanitize_event
        self._sensitive_values = sensitive_values
        if admission is not None and not callable(getattr(admission, "admit", None)):
            raise TypeError("MCP subscription Runtime admission seam is invalid")
        self._admission = admission
        self._task_event_projector = _validate_task_event_projector(
            task_event_projector
        )
        if local_cache_invalidator is not None and not callable(
            local_cache_invalidator
        ):
            raise TypeError("MCP subscription local cache invalidator is invalid")
        self._local_cache_invalidator = local_cache_invalidator
        self._clock = clock
        # Registry/process invalidation is a synchronous post-commit hook.
        # This lock guards in-memory state only; Provider I/O stays outside.
        self._lock = threading.RLock()
        self._live: dict[str, _LiveSubscription] = {}
        self._opening: dict[str, _OpeningSubscription] = {}
        self._stopping: dict[str, _LiveSubscription] = {}
        self._terminal: OrderedDict[str, _TerminalSubscription] = OrderedDict()
        self._event_reads: set[str] = set()
        self._closed = False
        self._validate_policy(reconcile_on_start=reconcile_on_start)
        if reconcile_on_start:
            self.reconcile_after_restart()

    def bind_task_event_projector(
        self,
        projector: McpTaskSubscriptionProjector,
    ) -> None:
        selected = _validate_task_event_projector(projector)
        assert selected is not None
        with self._lock:
            if self._live or self._opening or self._stopping:
                raise RuntimeError(
                    "MCP Tasks subscription projector cannot change while streams exist"
                )
            if (
                self._task_event_projector is not None
                and self._task_event_projector is not selected
            ):
                raise RuntimeError("MCP Tasks subscription projector is already bound")
            self._task_event_projector = selected

    async def start(
        self,
        server: McpServerSpec,
        fence: McpConnectionFence,
        provider: McpSubscriptionProvider,
        filters: tuple[str, ...],
        *,
        sensitive_values: Iterable[str] = (),
        tasks_extension_fence: McpTasksSubscriptionFence | None = None,
        deadline: float | None = None,
    ) -> McpSubscription:
        public, settlement = await self.prepare_start(
            server,
            fence,
            provider,
            filters,
            sensitive_values=sensitive_values,
            tasks_extension_fence=tasks_extension_fence,
            deadline=deadline,
        )
        try:
            settlement.commit_deferred()
            return await settlement.finalize()
        except BaseException as error:
            try:
                await settlement.abort(reason="subscription_start_failed")
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "MCP subscription start cleanup failed",
                    [error, cleanup_error],
                )
            raise

    async def prepare_start(
        self,
        server: McpServerSpec,
        fence: McpConnectionFence,
        provider: McpSubscriptionProvider,
        filters: tuple[str, ...],
        *,
        sensitive_values: Iterable[str] = (),
        tasks_extension_fence: McpTasksSubscriptionFence | None = None,
        deadline: float | None = None,
        origin_effect_id: str | None = None,
    ) -> tuple[McpSubscription, McpSubscriptionStartSettlement]:
        """Open a private listener without publishing an ACTIVE handle."""

        if origin_effect_id is not None and (
            type(origin_effect_id) is not str or not origin_effect_id
        ):
            raise ValidationError("MCP subscription origin effect id is invalid")
        selected_filters = _validate_filters(
            filters,
            tasks_extension_fence=tasks_extension_fence,
        )
        operation_secrets = _validate_sensitive_values(sensitive_values)
        task_event_projector = self._selected_task_event_projector(selected_filters)
        now = self._clock()
        if deadline is None:
            deadline = now + self._policy.exchange_timeout_s
        elif (
            type(deadline) not in {int, float}
            or not math.isfinite(deadline)
            or deadline <= now
        ):
            raise TimeoutError("MCP subscription opening deadline exceeded")
        else:
            deadline = min(float(deadline), now + self._policy.exchange_timeout_s)
        subscription_id = _subscription_id()
        opened_at = _utc_now()
        opening_public = McpSubscription(
            subscription_id=subscription_id,
            server_id=fence.server_id,
            status=McpSubscriptionStatus.OPENING,
            requested_filters=selected_filters,
            opened_at=opened_at,
        )
        # Finish local record construction before reserving live capacity.  A
        # malformed fence must not strand an OPENING slot or durable row.
        durable = self._new_durable_record(
            opening_public,
            fence,
            selected_filters,
            opened_at,
        )
        provider_cleanup_done = asyncio.Event()
        opening = _OpeningSubscription(
            opening_public,
            fence,
            provider_cleanup_done,
            origin_effect_id=origin_effect_id,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP subscription manager is closed")
            if len(self._live) + len(self._opening) >= self._policy.max_open:
                raise RuntimeError("MCP subscription limit reached")
            self._opening[subscription_id] = opening
        if self._store is not None:
            try:
                with self._mutation_admission():
                    durable = self._store.insert(durable)
            except BaseException:
                with self._lock:
                    self._opening.pop(subscription_id, None)
                raise

        connection: McpManagedConnection | None = None
        try:
            async def open_provider() -> _ProviderSession:
                listen = getattr(provider, "listen", None)
                if not callable(listen) or not inspect.iscoroutinefunction(listen):
                    raise TypeError(
                        "MCP subscription provider listen must be asynchronous"
                    )
                result = await listen(server, selected_filters, deadline=deadline)
                if type(result) is not McpSubscriptionSession:
                    raise TypeError(
                        "MCP subscription provider returned an invalid session"
                    )
                selected_session = _ProviderSession(
                    provider=provider,
                    session=result,
                    cleanup_done=provider_cleanup_done,
                )
                if (
                    not isinstance(result.owner_task, asyncio.Task)
                    or result.owner_task.done()
                    or result.owner_task.get_loop() is not asyncio.get_running_loop()
                ):
                    await selected_session.aclose()
                    raise TypeError(
                        "MCP subscription session owner task is invalid"
                    )
                return selected_session

            async def lost_callback(_connection_id: str, reason: str) -> None:
                await self._connection_lost(subscription_id, reason)

            connection = await self._supervisor.acquire(
                fence,
                "subscription",
                open_provider,
                deadline=deadline,
                on_lost=lost_callback,
                # Built-in adapters own their SDK context in a dedicated
                # listener task.  The outer supervisor lease owns only their
                # opaque handle, so a malicious listen() remains hard-bounded.
                task_affine=False,
            )
            if not isinstance(connection.session, _ProviderSession):
                raise TypeError("MCP subscription session ownership is invalid")
            provider_session = connection.session
            acknowledged = _acknowledged_filters(
                provider_session.session.acknowledged_filters,
                selected_filters,
            )
            active_public = replace(
                opening_public,
                status=McpSubscriptionStatus.ACTIVE,
                acknowledged_filters=acknowledged,
            )
            with self._lock:
                if self._closed or self._opening.get(subscription_id) is not opening:
                    raise RuntimeError("MCP subscription opening was cancelled")
                queue: asyncio.Queue[McpSubscriptionEvent] = asyncio.Queue(
                    maxsize=self._policy.queue_events
                )
                live = _LiveSubscription(
                    public=active_public,
                    connection=connection,
                    queue=queue,
                    task=None,
                    stop=asyncio.Event(),
                    provider=provider,
                    provider_handle=provider_session.session.handle,
                    durable=durable,
                    sensitive_values=operation_secrets,
                    listener_task=provider_session.session.owner_task,
                    cleanup_done=provider_cleanup_done,
                    task_event_projector=task_event_projector,
                )
                opening.prepared = live
                return active_public, McpSubscriptionStartSettlement(self, opening)
        except BaseException as exc:
            if connection is not None:
                await self._supervisor.mark_lost(
                    connection,
                    _diagnostic_reason(exc, fallback="subscription_start_failed"),
                )
            else:
                provider_cleanup_done.set()
            with self._lock:
                self._opening.pop(subscription_id, None)
                if durable is not None and self._store is not None:
                    durable = self._transition_to_lost(
                        durable,
                        "subscription_start_failed",
                    )
                public = replace(
                    opening_public,
                    status=McpSubscriptionStatus.LOST,
                    closed_at=_utc_now(),
                    lost_reason=_diagnostic_reason(
                        exc, fallback="subscription_start_failed"
                    ),
                )
                self._remember_terminal(
                    public,
                    (),
                    event_cursor=0,
                    cleanup_done=provider_cleanup_done,
                )
            raise

    def commit_prepared_start_deferred(
        self,
        settlement: McpSubscriptionStartSettlement,
    ) -> None:
        """Publish durable ACTIVE inside the originating effect transaction."""

        if (
            type(settlement) is not McpSubscriptionStartSettlement
            or settlement.manager is not self
        ):
            raise TypeError("MCP subscription start settlement is invalid")
        opening = settlement.opening
        # Do not acquire ``_lock`` here: ProtectedOperation.complete already
        # owns the RuntimeStore transaction, while background loss/event paths
        # intentionally use the opposite manager-lock -> Store order.  This
        # token is private to the one in-flight facade call; finalize performs
        # the exact registry/identity check back on the owner loop.
        if opening.closed:
            raise ValidationError("MCP subscription start settlement is unavailable")
        if opening.committed:
            raise ValidationError("MCP subscription start was already committed")
        live = opening.prepared
        if live is None:
            raise ValidationError("MCP subscription start is not prepared")
        if live.durable is not None and self._store is not None:
            acknowledged = live.public.acknowledged_filters
            live.durable = self._cas_record(
                live.durable,
                status="active",
                acknowledged_filter_sha256=(
                    json_sha256(
                        list(acknowledged),
                        label="MCP acknowledged subscription filters",
                    )
                    if acknowledged
                    else None
                ),
                metadata={"automatic_retry_disabled": True},
            )
        opening.committed = True

    async def finalize_prepared_start(
        self,
        settlement: McpSubscriptionStartSettlement,
    ) -> McpSubscription:
        """Publish one committed listener and start inert event consumption."""

        with self._lock:
            opening = self._opening_for_start_settlement(settlement)
            if not opening.committed:
                raise ValidationError("MCP subscription start was not committed")
            if self._closed:
                raise RuntimeError("MCP subscription manager is closed")
            live = opening.prepared
            if live is None:
                raise ValidationError("MCP subscription start is not prepared")
            subscription_id = opening.public.subscription_id
            if live.durable is not None and self._store is not None:
                current = self._store.get(subscription_id)
                if current != live.durable or current.status != "active":
                    raise ValidationError(
                        "MCP subscription durable publication changed"
                    )
            self._opening.pop(subscription_id)
            self._live[subscription_id] = live
            try:
                # This task outlives the caller's protected-operation lease. A
                # clean Context prevents that soon-inactive lease (and any
                # operation-local secrets) from escaping into the stream.
                live.task = contextvars.Context().run(
                    lambda: asyncio.create_task(
                        self._consume(subscription_id),
                        name=f"agent-libos-mcp-subscription-{subscription_id}",
                    )
                )
            except BaseException:
                self._live.pop(subscription_id, None)
                self._opening[subscription_id] = opening
                raise
            opening.closed = True
            return live.public

    async def abort_prepared_start(
        self,
        settlement: McpSubscriptionStartSettlement,
        *,
        reason: str = "subscription_publication_failed",
        persist: bool = True,
    ) -> None:
        """Fail closed and release a listener that was never returned."""

        selected_reason = _bounded_diagnostic(reason)
        subscription_id = settlement.subscription_id
        live: _LiveSubscription | None = None
        cleanup_done: asyncio.Event | None = None
        with self._lock:
            live, cleanup_done = self._detach_aborted_start_locked(settlement)
            if live is not None:
                self._latch_aborted_start_locked(
                    live,
                    reason=selected_reason,
                    persist=persist,
                )
        if live is not None:
            await self._release_aborted_start(live, reason=selected_reason)
        elif cleanup_done is not None:
            await _await_cleanup_done(
                cleanup_done,
                timeout=self._policy.exchange_timeout_s,
            )

    def _latch_aborted_start_locked(
        self,
        live: _LiveSubscription,
        *,
        reason: str,
        persist: bool,
    ) -> None:
        live.stop.set()
        public = replace(
            live.public,
            status=McpSubscriptionStatus.LOST,
            closed_at=_utc_now(),
            lost_reason=reason,
        )
        if persist and live.durable is not None and self._store is not None:
            try:
                live.durable = self._transition_to_lost(
                    live.durable,
                    "subscription_start_failed",
                )
            except BaseException:
                # Provider cleanup remains mandatory. A rolled-back starting or
                # active row is fenced to LOST by restart reconciliation.
                pass
        self._remember_terminal(
            public,
            tuple(live.queue._queue),
            event_cursor=live.event_cursor,
            cleanup_done=live.cleanup_done,
        )
        for task in (live.task, live.receive_task, live.listener_task):
            if (
                task is not None
                and task is not asyncio.current_task()
                and not task.done()
            ):
                task.cancel()

    async def _release_aborted_start(
        self,
        live: _LiveSubscription,
        *,
        reason: str,
    ) -> None:
        await self._supervisor.mark_lost(live.connection, reason)
        await _await_cleanup_done(
            live.cleanup_done,
            timeout=self._policy.exchange_timeout_s,
        )
        for task in (live.task, live.receive_task, live.listener_task):
            if task is not None and task is not asyncio.current_task():
                await _drain_cancelled(
                    task,
                    timeout=self._policy.exchange_timeout_s,
                )

    def _detach_aborted_start_locked(
        self,
        settlement: McpSubscriptionStartSettlement,
    ) -> tuple[_LiveSubscription | None, asyncio.Event | None]:
        if (
            type(settlement) is not McpSubscriptionStartSettlement
            or settlement.manager is not self
        ):
            raise TypeError("MCP subscription start settlement is invalid")
        subscription_id = settlement.subscription_id
        opening = self._opening.get(subscription_id)
        if opening is settlement.opening:
            self._opening.pop(subscription_id)
            opening.closed = True
            return opening.prepared, opening.cleanup_done
        if self._live.get(subscription_id) is settlement.opening.prepared:
            live = self._live.pop(subscription_id)
            return live, live.cleanup_done
        terminal = self._terminal.get(subscription_id)
        if terminal is not None:
            return None, terminal.cleanup_done
        if settlement.opening.closed:
            return None, settlement.opening.cleanup_done
        raise ValidationError("MCP subscription start settlement is unavailable")

    def _opening_for_start_settlement(
        self,
        settlement: McpSubscriptionStartSettlement,
    ) -> _OpeningSubscription:
        if (
            type(settlement) is not McpSubscriptionStartSettlement
            or settlement.manager is not self
        ):
            raise TypeError("MCP subscription start settlement is invalid")
        opening = settlement.opening
        current = self._opening.get(opening.public.subscription_id)
        if current is not opening or opening.closed:
            raise ValidationError("MCP subscription start settlement is unavailable")
        return opening

    async def status(self, subscription_id: str) -> McpSubscription:
        _validate_subscription_id(subscription_id)
        with self._lock:
            selected = self._live.get(subscription_id)
            if selected is not None:
                return selected.public
            opening = self._opening.get(subscription_id)
            if opening is not None:
                return opening.public
            stopping = self._stopping.get(subscription_id)
            if stopping is not None:
                return stopping.public
            terminal = self._terminal.get(subscription_id)
            if terminal is not None:
                self._terminal.move_to_end(subscription_id)
                return terminal.public
        if self._store is not None:
            record = self._store.get(subscription_id)
            if record is not None:
                return _public_from_record(record)
        raise KeyError(subscription_id)

    async def events(
        self, subscription_id: str, *, after: int = 0, limit: int = 100
    ) -> tuple[McpSubscriptionEvent, ...]:
        """Consume one prefix from the exact single-reader local cursor."""

        _validate_subscription_id(subscription_id)
        if (
            type(after) is not int
            or after < 0
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise ValueError("invalid MCP subscription event window")
        with self._lock:
            if subscription_id in self._event_reads:
                raise ValidationError(
                    "MCP subscription event cursor is stale or has multiple readers"
                )
            self._event_reads.add(subscription_id)
        try:
            # Establish a real single-reader exclusion window even when the
            # current queue is empty. Runtime calls remain one-shot reads, not
            # long polling, but concurrently submitted readers fail closed.
            await asyncio.sleep(0)
            with self._lock:
                selected = self._live.get(subscription_id)
                if selected is not None:
                    if after != selected.event_cursor:
                        raise ValidationError(
                            "MCP subscription event cursor is stale or has multiple readers"
                        )
                    snapshot = tuple(
                        selected.queue._queue  # bounded Host-local snapshot
                    )
                    batch = tuple(
                        event for event in snapshot if event.sequence > after
                    )[:limit]
                    if batch:
                        for expected in batch:
                            consumed = selected.queue.get_nowait()
                            if consumed is not expected:
                                raise RuntimeError(
                                    "MCP subscription event queue order is invalid"
                                )
                            selected.queue.task_done()
                        selected.event_cursor = batch[-1].sequence
                else:
                    terminal = self._terminal.get(subscription_id)
                    if terminal is None:
                        raise KeyError(subscription_id)
                    if after != terminal.event_cursor:
                        raise ValidationError(
                            "MCP subscription event cursor is stale or has multiple readers"
                        )
                    batch = tuple(
                        event for event in terminal.events if event.sequence > after
                    )[:limit]
                    if batch:
                        terminal = replace(
                            terminal,
                            events=terminal.events[len(batch):],
                            event_cursor=batch[-1].sequence,
                        )
                        self._terminal[subscription_id] = terminal
                    self._terminal.move_to_end(subscription_id)
                return batch
        finally:
            with self._lock:
                self._event_reads.discard(subscription_id)

    async def stop(self, subscription_id: str) -> McpSubscription:
        return await self._stop(subscription_id, persist=True)

    async def _stop(
        self,
        subscription_id: str,
        *,
        persist: bool,
    ) -> McpSubscription:
        _validate_subscription_id(subscription_id)
        selected, terminal = self._claim_stop(subscription_id, persist=persist)
        if terminal is not None:
            await _await_cleanup_done(
                terminal.cleanup_done,
                timeout=self._policy.exchange_timeout_s,
            )
            return terminal.public
        assert selected is not None
        task = selected.task
        receive_task = selected.receive_task
        listener_task = selected.listener_task

        if task is not None:
            await _drain_cancelled(task, timeout=self._policy.exchange_timeout_s)
        await self._supervisor.release(selected.connection)
        await _await_cleanup_done(
            selected.cleanup_done,
            timeout=self._policy.exchange_timeout_s,
        )
        if receive_task is not None:
            await _drain_cancelled(
                receive_task,
                timeout=self._policy.exchange_timeout_s,
            )
        if listener_task is not asyncio.current_task():
            await _drain_cancelled(
                listener_task,
                timeout=self._policy.exchange_timeout_s,
            )

        closed = replace(
            selected.public,
            status=McpSubscriptionStatus.CLOSED,
            closed_at=_utc_now(),
        )
        with self._lock:
            if selected.durable is not None and selected.durable.status == "lost":
                self._stopping.pop(subscription_id, None)
                terminal = self._terminal.get(subscription_id)
                if terminal is not None:
                    return terminal.public
            if persist and selected.durable is not None and self._store is not None:
                selected.durable = self._cas_record(
                    selected.durable,
                    status="stopped",
                    metadata={"automatic_retry_disabled": True},
                )
            self._stopping.pop(subscription_id, None)
            self._remember_terminal(
                closed,
                tuple(selected.queue._queue),
                event_cursor=selected.event_cursor,
                cleanup_done=selected.cleanup_done,
            )
            return closed

    def _claim_stop(
        self,
        subscription_id: str,
        *,
        persist: bool,
    ) -> tuple[_LiveSubscription | None, _TerminalSubscription | None]:
        with self._lock:
            selected = self._live.pop(subscription_id, None)
            if selected is None:
                terminal = self._terminal.get(subscription_id)
                if terminal is None:
                    raise KeyError(subscription_id)
                self._terminal.move_to_end(subscription_id)
                return None, terminal
            selected.stop.set()
            self._stopping[subscription_id] = selected
            try:
                if persist and selected.durable is not None and self._store is not None:
                    selected.durable = self._cas_record(
                        selected.durable,
                        status="stopping",
                        metadata={"automatic_retry_disabled": True},
                    )
            except BaseException:
                # No Provider I/O has started yet. Restore the exact live state
                # so a rejected/fenced durable mutation cannot orphan the
                # handle or hand its cleanup to another event loop.
                self._stopping.pop(subscription_id, None)
                selected.stop.clear()
                self._live[subscription_id] = selected
                raise
            for task in (selected.task, selected.receive_task, selected.listener_task):
                if task is not None and task is not asyncio.current_task() and not task.done():
                    task.cancel()
            return selected, None

    async def close(self) -> None:
        with self._lock:
            self._closed = True
            identifiers = tuple(self._live)
            prepared = tuple(
                McpSubscriptionStartSettlement(self, opening)
                for opening in self._opening.values()
                if opening.prepared is not None
            )
        for subscription_id in identifiers:
            try:
                # Runtime shutdown has already closed ordinary mutation
                # admission. Keep cleanup on this owner loop, but deliberately
                # leave the durable row nonterminal so the next Runtime OPEN
                # reconciles it to LOST rather than claiming an explicit stop.
                await self._stop(subscription_id, persist=False)
            except KeyError:  # pragma: no cover - concurrent Host stop/loss
                pass
        for settlement in prepared:
            try:
                await self.abort_prepared_start(
                    settlement,
                    reason="runtime_shutdown",
                    persist=False,
                )
            except (KeyError, ValidationError):
                pass

    async def _consume(self, subscription_id: str) -> None:
        lifetime_deadline = self._clock() + self._policy.max_lifetime_s
        try:
            while True:
                event = await self._next_projected_event(
                    subscription_id,
                    lifetime_deadline,
                )
                if event is None:
                    return
                event_bytes = canonical_json_bytes(
                    event.payload,
                    label="MCP subscription event",
                    max_bytes=self._policy.event_max_bytes,
                )
                if len(event_bytes) > self._policy.event_max_bytes:
                    raise ValidationError("MCP subscription event exceeds byte limit")
                accepted = self._enqueue_event(subscription_id, event)
                if accepted is None:
                    return
                if not accepted:
                    raise RuntimeError("MCP subscription queue overflow")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            with self._lock:
                selected = self._live.get(subscription_id)
                connection = selected.connection if selected is not None else None
            reason = _diagnostic_reason(
                exc, fallback="subscription_receive_failed"
            )
            # Claim the terminal transition before asking the supervisor to
            # close.  Otherwise a concurrent explicit stop can remove _live
            # after this snapshot but before mark_lost(), misclassifying the
            # normal local close as a connection loss.
            claimed = await self._connection_lost(subscription_id, reason)
            if claimed and connection is not None:
                await self._supervisor.mark_lost(
                    connection,
                    reason,
                )

    async def _next_projected_event(
        self,
        subscription_id: str,
        lifetime_deadline: float,
    ) -> McpSubscriptionEvent | None:
        with self._lock:
            selected = self._live.get(subscription_id)
            if selected is None or selected.stop.is_set():
                return None
            provider = selected.provider
            handle = selected.provider_handle
            requested = selected.public.requested_filters
            operation_secrets = selected.sensitive_values
            task_event_projector = selected.task_event_projector
            fence = selected.connection.fence
        now = self._clock()
        if now >= lifetime_deadline:
            raise TimeoutError("MCP subscription lifetime expired")
        deadline = min(lifetime_deadline, now + self._policy.exchange_timeout_s)
        receive_task = _provider_receive_task(provider, handle, deadline=deadline)
        with self._lock:
            selected = self._live.get(subscription_id)
            if selected is None or selected.stop.is_set():
                receive_task.cancel()
                return None
            selected.receive_task = receive_task
        event = await _await_receive_event(
            receive_task,
            deadline=deadline,
            clock=self._clock,
        )
        with self._lock:
            selected = self._live.get(subscription_id)
            if selected is not None and selected.receive_task is receive_task:
                selected.receive_task = None
        return self._project_event(
            event,
            requested,
            operation_secrets,
            fence=fence,
            task_event_projector=task_event_projector,
        )

    def _enqueue_event(
        self,
        subscription_id: str,
        event: McpSubscriptionEvent,
    ) -> bool | None:
        with self._lock:
            selected = self._live.get(subscription_id)
            if selected is None or selected.stop.is_set():
                return None
            event = replace(
                event,
                sequence=selected.next_sequence,
                received_at=_utc_now(),
                provenance="untrusted_mcp_notification",
            )
            selected.next_sequence += 1
            overflow = selected.queue.full()
            if not overflow:
                selected.queue.put_nowait(event)
            if selected.durable is not None and self._store is not None:
                selected.durable = self._cas_record(
                    selected.durable,
                    received_count=selected.durable.received_count + 1,
                    dropped_count=(
                        selected.durable.dropped_count + (1 if overflow else 0)
                    ),
                    last_event_at=event.received_at,
                    metadata={"automatic_retry_disabled": True},
                )
            if (
                self._local_cache_invalidator is not None
                and event.event_type in _LOCAL_CACHE_INVALIDATING_EVENT_TYPES
            ):
                # Exact, synchronous Runtime-local revocation only: no remote
                # I/O and no model, Tool, TaskRun, or subscription dispatch.
                self._local_cache_invalidator(selected.public.server_id)
            return not overflow

    async def _connection_lost(self, subscription_id: str, reason: str) -> bool:
        task: asyncio.Task[None] | None = None
        receive_task: asyncio.Task[Any] | None = None
        listener_task: asyncio.Task[Any] | None = None
        with self._lock:
            selected = self._live.pop(subscription_id, None)
            if selected is None:
                opening = self._opening.get(subscription_id)
                if opening is None or opening.prepared is None:
                    return False
                self._opening.pop(subscription_id)
                opening.closed = True
                selected = opening.prepared
            selected.stop.set()
            task = selected.task
            receive_task = selected.receive_task
            listener_task = selected.listener_task
            public = replace(
                selected.public,
                status=McpSubscriptionStatus.LOST,
                closed_at=_utc_now(),
                lost_reason=_bounded_diagnostic(reason),
            )
            # Latch the Host-visible terminal state and bounded queued events
            # before touching the durable repository.  A CAS race or Store
            # outage must never leave a removed live subscription invisible to
            # status/events/stop.
            self._remember_terminal(
                public,
                tuple(selected.queue._queue),
                event_cursor=selected.event_cursor,
                cleanup_done=selected.cleanup_done,
            )
            if selected.durable is not None and self._store is not None:
                try:
                    selected.durable = self._transition_to_lost(
                        selected.durable,
                        "subscription_connection_lost",
                    )
                except BaseException:
                    # The public latch is already fail-closed.  Reopen
                    # reconciliation will CAS any remaining nonterminal row to
                    # LOST; never erase the terminal state or skip Provider
                    # cleanup because durable evidence raced.
                    pass
            if task is not None and task is not asyncio.current_task():
                task.cancel()
            if receive_task is not None and receive_task is not asyncio.current_task():
                receive_task.cancel()
            if listener_task is not asyncio.current_task() and not listener_task.done():
                listener_task.cancel()
        if task is not None and task is not asyncio.current_task():
            task.add_done_callback(_consume_task_result)
        if receive_task is not None and receive_task is not asyncio.current_task():
            await _drain_cancelled(
                receive_task,
                timeout=self._policy.exchange_timeout_s,
            )
        if listener_task is not None and listener_task is not asyncio.current_task():
            await _drain_cancelled(
                listener_task,
                timeout=self._policy.exchange_timeout_s,
            )
        return True

    def _project_event(
        self,
        event: Any,
        requested: tuple[str, ...],
        operation_secrets: tuple[str, ...],
        *,
        fence: McpConnectionFence,
        task_event_projector: McpTaskSubscriptionProjector | None,
    ) -> McpSubscriptionEvent:
        if type(event) is not McpSubscriptionEvent:
            raise TypeError("MCP subscription provider returned an invalid event")
        _validate_event_type(event.event_type, requested)
        strict_json_value(event.payload, label="MCP subscription event")
        secrets_snapshot = tuple(
            dict.fromkeys((*self._resolved_sensitive_values(), *operation_secrets))
        )
        if event.event_type == "taskStatus":
            if task_event_projector is None:
                raise ValidationError(
                    "MCP task notification has no Host result-claim projector"
                )
            with self._mutation_admission():
                event = task_event_projector.project_task_notification(
                    event=event,
                    fence=fence,
                    sensitive_values=secrets_snapshot,
                )
            if (
                type(event) is not McpSubscriptionEvent
                or event.event_type != "taskStatus"
            ):
                raise ValidationError(
                    "MCP Tasks result-claim projector returned an invalid event"
                )
            strict_json_value(
                event.payload,
                label="MCP claimed Task subscription event",
            )
        payload = sanitize_provider_json(
            event.payload,
            sensitive_values=secrets_snapshot,
        )
        _reject_apps_json(payload)
        selected = replace(
            event,
            sequence=0,
            payload=payload,
            received_at=_utc_now(),
            provenance="untrusted_mcp_notification",
        )
        if self._sanitize_event is not None:
            selected = self._sanitize_event(selected)
            if type(selected) is not McpSubscriptionEvent:
                raise TypeError("MCP subscription sanitizer returned an invalid event")
            _validate_event_type(selected.event_type, requested)
            strict_json_value(selected.payload, label="sanitized MCP subscription event")
            payload = sanitize_provider_json(
                selected.payload,
                sensitive_values=secrets_snapshot,
            )
            _reject_apps_json(payload)
            selected = replace(selected, payload=payload)
        return replace(
            selected,
            sequence=0,
            received_at=_utc_now(),
            provenance="untrusted_mcp_notification",
        )

    def _selected_task_event_projector(
        self,
        filters: tuple[str, ...],
    ) -> McpTaskSubscriptionProjector | None:
        with self._lock:
            selected = self._task_event_projector
        if "taskIds" in filters and selected is None:
            raise ValidationError(
                "MCP taskIds subscription requires a Host result-claim projector"
            )
        return selected

    def _resolved_sensitive_values(self) -> tuple[str, ...]:
        values = (
            self._sensitive_values()
            if callable(self._sensitive_values)
            else self._sensitive_values
        )
        return _validate_sensitive_values(values)

    def _new_durable_record(
        self,
        public: McpSubscription,
        fence: McpConnectionFence,
        filters: tuple[str, ...],
        now: str,
    ) -> McpSubscriptionRecord:
        empty_sha256 = sha256(b"").hexdigest()
        return McpSubscriptionRecord(
            subscription_id=public.subscription_id,
            server_id=fence.server_id,
            server_spec_sha256=fence.server_spec_sha256,
            server_generation=fence.registry_generation,
            owner_id=fence.owner,
            auth_principal_sha256=fence.auth_principal_sha256 or empty_sha256,
            auth_scope_sha256=fence.auth_scope_sha256 or empty_sha256,
            requested_filter_sha256=json_sha256(
                list(filters), label="MCP requested subscription filters"
            ),
            acknowledged_filter_sha256=None,
            status="starting",
            queue_limit=self._policy.queue_events,
            event_max_bytes=self._policy.event_max_bytes,
            received_count=0,
            dropped_count=0,
            revision=0,
            last_event_at=None,
            metadata={"automatic_retry_disabled": True},
            created_at=now,
            updated_at=now,
        )

    def _cas_record(
        self,
        record: McpSubscriptionRecord,
        **changes: Any,
    ) -> McpSubscriptionRecord:
        if self._store is None:
            return record
        replacement = replace(
            record,
            revision=record.revision + 1,
            updated_at=_utc_now(),
            **changes,
        )
        with self._mutation_admission():
            swapped = self._store.compare_and_swap(
                record.subscription_id,
                expected_revision=record.revision,
                replacement=replacement,
            )
        if not swapped:
            raise RuntimeError("MCP subscription durable state changed concurrently")
        return replacement

    def _mutation_admission(self) -> AbstractContextManager[None]:
        if self._admission is None:
            return nullcontext()
        return self._admission.admit()

    def _transition_to_lost(
        self,
        record: McpSubscriptionRecord,
        reason_code: str,
    ) -> McpSubscriptionRecord:
        selected = record
        for _ in range(8):
            if selected.status in {"lost", "stopped"}:
                return selected
            try:
                return self._cas_record(
                    selected,
                    status="lost",
                    metadata={
                        "automatic_retry_disabled": True,
                        # Durable metadata has a closed Host-owned vocabulary.  The
                        # provider/supervisor reason remains only in the bounded
                        # in-memory public status and cannot become a Store payload.
                        "reason_code": _durable_reason(reason_code),
                        "retry_class": "not_retryable",
                    },
                )
            except RuntimeError:
                assert self._store is not None
                current = self._store.get(selected.subscription_id)
                if current is None:
                    raise
                selected = current
        raise RuntimeError("MCP subscription LOST transition CAS was unstable")

    def invalidate_server(self, server_id: str) -> None:
        """Post-commit synchronous registry invalidation seam."""

        self.invalidate_server_nowait(server_id)

    def invalidate_server_nowait(self, server_id: str) -> None:
        if type(server_id) is not str or not server_id:
            return
        self._invalidate_nowait(
            live_predicate=lambda selected: selected.public.server_id == server_id,
            opening_predicate=lambda selected: selected.public.server_id == server_id,
            reason="registry_fence_changed",
        )

    def close_owner_nowait(self, owner: str) -> None:
        if type(owner) is not str or not owner:
            return
        self._invalidate_nowait(
            live_predicate=lambda selected: selected.connection.fence.owner == owner,
            opening_predicate=lambda selected: selected.fence.owner == owner,
            reason="owner_closed",
        )
        self._supervisor.close_owner_nowait(owner)

    def _invalidate_nowait(
        self,
        *,
        live_predicate: Callable[[_LiveSubscription], bool],
        opening_predicate: Callable[[_OpeningSubscription], bool],
        reason: str,
    ) -> None:
        cancelled: list[asyncio.Task[Any]] = []
        try:
            with self._lock:
                for subscription_id, selected in tuple(self._live.items()):
                    if not live_predicate(selected):
                        continue
                    self._live.pop(subscription_id, None)
                    self._lose_selected_nowait(selected, reason)
                    cancelled.extend(
                        task
                        for task in (
                            selected.task,
                            selected.receive_task,
                            selected.listener_task,
                        )
                        if task is not None
                    )
                for subscription_id, selected in tuple(self._stopping.items()):
                    if not live_predicate(selected):
                        continue
                    self._stopping.pop(subscription_id, None)
                    self._lose_selected_nowait(selected, reason)
                    cancelled.extend(
                        task
                        for task in (
                            selected.task,
                            selected.receive_task,
                            selected.listener_task,
                        )
                        if task is not None
                    )
                for subscription_id, selected in tuple(self._opening.items()):
                    if not opening_predicate(selected):
                        continue
                    self._opening.pop(subscription_id, None)
                    selected.closed = True
                    prepared = selected.prepared
                    if prepared is not None:
                        prepared.stop.set()
                        durable = prepared.durable
                        cancelled.extend(
                            task
                            for task in (
                                prepared.task,
                                prepared.receive_task,
                                prepared.listener_task,
                            )
                            if task is not None
                        )
                        public = prepared.public
                        events = tuple(prepared.queue._queue)
                        event_cursor = prepared.event_cursor
                    else:
                        durable = (
                            self._store.get(subscription_id)
                            if self._store is not None
                            else None
                        )
                        public = selected.public
                        events = ()
                        event_cursor = 0
                    if durable is not None:
                        try:
                            self._transition_to_lost(durable, reason)
                        except BaseException:
                            pass
                    self._remember_terminal(
                        replace(
                            public,
                            status=McpSubscriptionStatus.LOST,
                            closed_at=_utc_now(),
                            lost_reason=reason,
                        ),
                        events,
                        event_cursor=event_cursor,
                        cleanup_done=selected.cleanup_done,
                    )
        except BaseException:
            # Registry/OAuth state is already committed.  Keep the in-memory
            # revocation latch and never report a false rollback.
            pass
        for task in dict.fromkeys(cancelled):
            _cancel_task_threadsafe(task)

    def _lose_selected_nowait(
        self,
        selected: _LiveSubscription,
        reason: str,
    ) -> None:
        public = replace(
            selected.public,
            status=McpSubscriptionStatus.LOST,
            closed_at=_utc_now(),
            lost_reason=reason,
        )
        if selected.durable is not None and self._store is not None:
            try:
                selected.durable = self._transition_to_lost(
                    selected.durable,
                    reason,
                )
            except BaseException:
                pass
        self._remember_terminal(
            public,
            tuple(selected.queue._queue),
            event_cursor=selected.event_cursor,
            cleanup_done=selected.cleanup_done,
        )

    def _remember_terminal(
        self,
        public: McpSubscription,
        events: tuple[McpSubscriptionEvent, ...],
        *,
        event_cursor: int,
        cleanup_done: asyncio.Event,
    ) -> None:
        self._terminal[public.subscription_id] = _TerminalSubscription(
            public=public,
            events=events,
            event_cursor=event_cursor,
            cleanup_done=cleanup_done,
        )
        self._terminal.move_to_end(public.subscription_id)
        while len(self._terminal) > self._policy.terminal_status_records:
            self._terminal.popitem(last=False)

    def _validate_policy(self, *, reconcile_on_start: bool) -> None:
        integers = (
            self._policy.max_open,
            self._policy.queue_events,
            self._policy.event_max_bytes,
            self._policy.terminal_status_records,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("MCP subscription integer bounds must be positive")
        durations = (
            self._policy.max_lifetime_s,
            self._policy.exchange_timeout_s,
        )
        if any(
            type(value) not in {int, float}
            or not math.isfinite(value)
            or value <= 0
            for value in durations
        ):
            raise ValueError("MCP subscription time bounds must be positive")
        if type(reconcile_on_start) is not bool:
            raise ValidationError(
                "MCP subscription reconcile_on_start is invalid"
            )

    def reconcile_after_restart(self) -> int:
        """Fence pre-restart streams without reconnecting or replaying them."""

        if self._store is None:
            return 0
        with self._lock:
            if self._live or self._opening or self._stopping:
                raise RuntimeError(
                    "MCP subscription restart reconciliation requires no live streams"
                )
        changed = 0
        for status in ("starting", "active", "stopping"):
            while True:
                records = self._store.list(status=status, limit=500)
                if not records:
                    break
                for record in records:
                    if self._reconcile_interrupted_record(record):
                        changed += 1
                if len(records) < 500:
                    break
        return changed

    def _reconcile_interrupted_record(self, record: McpSubscriptionRecord) -> bool:
        assert self._store is not None
        selected = record
        for _ in range(8):
            if selected.status not in {"starting", "active", "stopping"}:
                return False
            replacement = replace(
                selected,
                status="lost",
                revision=selected.revision + 1,
                metadata={
                    "automatic_retry_disabled": True,
                    "reason_code": "runtime_restart",
                    "retry_class": "not_retryable",
                },
                updated_at=_utc_now(),
            )
            with self._mutation_admission():
                swapped = self._store.compare_and_swap(
                    selected.subscription_id,
                    expected_revision=selected.revision,
                    replacement=replacement,
                )
            if swapped:
                return True
            current = self._store.get(selected.subscription_id)
            if current is None:
                return False
            selected = current
        raise RuntimeError("MCP subscription restart reconciliation CAS was unstable")


def _validate_filters(
    filters: tuple[str, ...],
    *,
    tasks_extension_fence: McpTasksSubscriptionFence | None,
) -> tuple[str, ...]:
    if type(filters) is not tuple or not filters or any(
        type(item) is not str for item in filters
    ):
        raise ValueError("MCP subscription filters must be a non-empty string tuple")
    if len(filters) > len(MCP_V3_SUBSCRIPTION_FILTERS):
        raise ValueError("MCP subscription filters exceed the maximum count")
    if len(set(filters)) != len(filters):
        raise ValueError("MCP subscription filters must be unique")
    unknown = set(filters) - MCP_V3_SUBSCRIPTION_FILTERS
    if unknown:
        raise ValueError(f"unsupported MCP subscription filters: {sorted(unknown)}")
    if "taskIds" in filters:
        if type(tasks_extension_fence) is not McpTasksSubscriptionFence:
            raise ValidationError(
                "MCP taskIds subscription requires an exact Tasks extension fence"
            )
        if (
            tasks_extension_fence.extension_id != MCP_TASKS_EXTENSION_ID
            or not _is_sha256(tasks_extension_fence.manifest_spec_sha256)
            or not _is_sha256(tasks_extension_fence.host_spec_sha256)
            or tasks_extension_fence.manifest_spec_sha256
            != tasks_extension_fence.host_spec_sha256
        ):
            raise ValidationError(
                "MCP taskIds subscription Tasks extension pin changed"
            )
    elif tasks_extension_fence is not None:
        raise ValidationError(
            "MCP Tasks extension fence requires the taskIds subscription filter"
        )
    return filters


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _acknowledged_filters(
    value: tuple[str, ...],
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) > len(requested):
        raise ValueError("MCP subscription acknowledgement is not a bounded array")
    if (
        any(type(item) is not str or item not in requested for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("MCP server acknowledged unrequested subscription filters")
    return value


def _provider_receive_task(
    provider: McpSubscriptionProvider,
    handle: Any,
    *,
    deadline: float,
) -> asyncio.Task[Any]:
    async def invoke() -> Any:
        receive = getattr(provider, "receive", None)
        if not callable(receive) or not inspect.iscoroutinefunction(receive):
            raise TypeError("MCP subscription provider receive must be asynchronous")
        return await receive(handle, deadline=deadline)

    return asyncio.create_task(
        invoke(),
        name="agent-libos-mcp-subscription-receive",
    )


async def _await_receive_event(
    task: asyncio.Task[Any],
    *,
    deadline: float,
    clock: Callable[[], float],
) -> Any:
    try:
        done, _ = await asyncio.wait(
            {task},
            timeout=max(0.0, deadline - clock()),
        )
    except asyncio.CancelledError:
        task.cancel()
        raise
    if task not in done:
        task.cancel()
        raise TimeoutError("MCP subscription receive deadline exceeded")
    return task.result()


def _validate_event_type(event_type: Any, requested: tuple[str, ...]) -> None:
    allowed = frozenset(
        event_type
        for selected_filter in requested
        for event_type in _EVENT_TYPES_BY_FILTER.get(selected_filter, ())
    )
    if type(event_type) is not str or event_type not in allowed:
        raise ValidationError("MCP subscription event type was not allowlisted")


def _reject_apps_json(value: Any) -> None:
    pending = [value]
    while pending:
        selected = pending.pop()
        if type(selected) is str:
            if selected.lstrip().casefold().startswith("ui:"):
                raise ValidationError("MCP Apps ui:// resources are unsupported")
            if is_mcp_app_mime(selected):
                raise ValidationError("MCP Apps content is unsupported")
        elif type(selected) is list:
            pending.extend(selected)
        elif type(selected) is dict:
            pending.extend(selected.keys())
            pending.extend(selected.values())


async def _drain_cancelled(task: asyncio.Task[Any], *, timeout: float) -> None:
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if task not in done:
        task.add_done_callback(_consume_task_result)
        return
    _consume_task_result(task)


async def _await_cleanup_done(event: asyncio.Event, *, timeout: float) -> None:
    if event.is_set():
        return
    try:
        async with asyncio.timeout(timeout):
            await event.wait()
    except TimeoutError as exc:
        raise RuntimeError(
            "MCP subscription Provider cleanup did not complete"
        ) from exc


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass


def _cancel_task_threadsafe(task: asyncio.Task[Any]) -> None:
    if task.done():
        return
    loop = task.get_loop()
    if loop.is_closed():
        return
    try:
        if _running_loop_or_none() is loop:
            task.cancel()
        else:
            loop.call_soon_threadsafe(task.cancel)
    except RuntimeError:
        return


def _running_loop_or_none() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _validate_sensitive_values(values: Iterable[str]) -> tuple[str, ...]:
    if type(values) not in {tuple, list, set, frozenset}:
        raise ValidationError("MCP subscription sensitive values are invalid")
    selected = tuple(values)
    if len(selected) > 128 or any(
        type(value) is not str or len(value) > 4096 for value in selected
    ):
        raise ValidationError("MCP subscription sensitive values are invalid")
    return selected


def _validate_task_event_projector(
    value: McpTaskSubscriptionProjector | None,
) -> McpTaskSubscriptionProjector | None:
    if value is None:
        return None
    if not callable(getattr(value, "project_task_notification", None)):
        raise TypeError("MCP Tasks subscription result-claim projector is invalid")
    return value


def _durable_reason(reason: str) -> str:
    if reason in {
        "runtime_restart",
        "subscription_connection_lost",
        "subscription_failure",
        "subscription_receive_failed",
        "subscription_start_failed",
    }:
        return reason
    if reason in {
        "subscription_receive_failed",
        "timeout_error",
        "validation_error",
        "runtime_error",
    }:
        return "subscription_receive_failed"
    if reason in {
        "absolute_ttl_expired",
        "auth_fence_changed",
        "owner_closed",
        "registry_fence_changed",
        "supervisor_closed",
    }:
        return "subscription_connection_lost"
    return "subscription_failure"


def _subscription_id() -> str:
    return f"mcp-subscription-{secrets.token_urlsafe(18)}"


def _validate_subscription_id(value: Any) -> None:
    if type(value) is not str or not value or len(value) > 512:
        raise ValueError("invalid MCP subscription id")


def _public_from_record(record: McpSubscriptionRecord) -> McpSubscription:
    if record.status in {"starting", "active", "stopping"}:
        # Constructor reconciliation should have removed this state.  If a
        # concurrent writer created it later, never represent it as resumable.
        status = McpSubscriptionStatus.LOST
        reason = "unrecoverable_runtime_state"
    elif record.status == "stopped":
        status = McpSubscriptionStatus.CLOSED
        reason = None
    else:
        status = McpSubscriptionStatus.LOST
        reason_value = record.metadata.get("reason_code")
        reason = reason_value if type(reason_value) is str else record.status
    return McpSubscription(
        subscription_id=record.subscription_id,
        server_id=record.server_id,
        status=status,
        requested_filters=(),
        acknowledged_filters=(),
        opened_at=record.created_at,
        closed_at=record.updated_at,
        lost_reason=reason,
    )


def _diagnostic_reason(exc: BaseException, *, fallback: str) -> str:
    name = type(exc).__name__
    if not name or name == "Exception":
        return fallback
    chars: list[str] = []
    for character in name:
        if character.isupper() and chars:
            chars.append("_")
        if character.isalnum() or character in "_.:-":
            chars.append(character.lower())
    selected = "".join(chars).strip("_")
    return _bounded_diagnostic(selected or fallback)


def _bounded_diagnostic(reason: Any) -> str:
    selected = reason if type(reason) is str else type(reason).__name__
    selected = selected.casefold()
    sanitized = "".join(
        character if character.isalnum() or character in "_.:-" else "_"
        for character in selected
    ).strip("_")
    if not sanitized or not sanitized[0].isalpha():
        sanitized = f"reason_{sanitized}" if sanitized else "unknown"
    return sanitized[:128]


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="microseconds")
