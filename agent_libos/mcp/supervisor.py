"""Runtime-owned MCP connection lifecycle fencing.

The supervisor never reconnects, replays an operation, or supplies
``Last-Event-ID``.  Each acquisition receives a distinct lease token.  A
caller must explicitly create a new connection after any loss or
registry/auth generation change.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal


@dataclass(frozen=True)
class McpConnectionFence:
    server_id: str
    server_spec_sha256: str
    registry_generation: int
    owner: str
    auth_principal_sha256: str | None = None
    auth_scope_sha256: str | None = None
    auth_generation: int = 0


@dataclass(frozen=True)
class McpManagedConnection:
    """A single, non-transferable lease on a supervised session."""

    connection_id: str
    lease_token: str
    fence: McpConnectionFence
    purpose: Literal["read", "mutation", "subscription", "oauth"]
    session: Any
    created_monotonic: float
    last_used_monotonic: float
    absolute_expires_monotonic: float
    idle_expires_monotonic: float
    lease_count: int = 1
    state: Literal["active", "lost", "closed"] = "active"
    lost_reason: str | None = None


SessionFactory = Callable[[], Awaitable[Any]]
LossCallback = Callable[[str, str], Any | Awaitable[Any]]


@dataclass
class _ConnectionState:
    connection_id: str
    fence: McpConnectionFence
    purpose: Literal["read", "mutation", "subscription", "oauth"]
    session: Any
    created_monotonic: float
    last_used_monotonic: float
    absolute_expires_monotonic: float
    idle_expires_monotonic: float
    leases: set[str] = field(default_factory=set)
    loss_callbacks: tuple[LossCallback, ...] = ()
    expiry_task: asyncio.Task[None] | None = None
    state: Literal["active", "lost", "closed"] = "active"
    lost_reason: str | None = None
    close_started: bool = False
    task_affine: bool = True
    owner_task: asyncio.Task[Any] | None = None
    owner_loop: asyncio.AbstractEventLoop | None = None


@dataclass
class _OpeningState:
    opening_id: str
    fence: McpConnectionFence
    purpose: Literal["read", "mutation", "subscription", "oauth"]
    invalidated: asyncio.Event = field(default_factory=asyncio.Event)
    invalidated_reason: str | None = None
    task_affine: bool = True
    owner_task: asyncio.Task[Any] | None = None
    owner_loop: asyncio.AbstractEventLoop | None = None


class McpConnectionSupervisor:
    """Own bounded sessions and close them on every authority fence change.

    Provider/factory/close callbacks are never awaited while the supervisor
    lock is held.  Close is best effort but bounded: a Provider that ignores
    cancellation cannot prevent another caller from acquiring or closing.
    """

    def __init__(
        self,
        *,
        idle_ttl_s: float = 30.0,
        absolute_ttl_s: float = 300.0,
        max_connections: int = 64,
        open_timeout_s: float = 30.0,
        close_timeout_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        bounds = (idle_ttl_s, absolute_ttl_s, open_timeout_s, close_timeout_s)
        if any(
            type(value) not in {int, float}
            or not math.isfinite(value)
            or value <= 0
            for value in bounds
        ):
            raise ValueError("MCP connection timeouts must be positive")
        if type(max_connections) is not int or max_connections <= 0:
            raise ValueError("MCP max_connections must be a positive exact integer")
        self._idle_ttl_s = float(idle_ttl_s)
        self._absolute_ttl_s = float(absolute_ttl_s)
        self._open_timeout_s = float(open_timeout_s)
        self._close_timeout_s = float(close_timeout_s)
        self._max_connections = max_connections
        self._clock = clock
        # The registry/OAuth/process lifecycle hooks are synchronous and may
        # run outside the connection's event-loop thread.  No I/O is ever
        # performed while this short state lock is held.
        self._lock = threading.RLock()
        self._connections: dict[str, _ConnectionState] = {}
        self._openings: dict[str, _OpeningState] = {}
        self._sequence = 0
        self._closed = False

    async def acquire(
        self,
        fence: McpConnectionFence,
        purpose: Literal["read", "mutation", "subscription", "oauth"],
        factory: SessionFactory,
        *,
        reusable: bool = False,
        deadline: float | None = None,
        on_lost: LossCallback | None = None,
        task_affine: bool = True,
    ) -> McpManagedConnection:
        """Create or reuse an exact-fence connection without retrying factory."""

        selected_deadline = self._validate_deadline(deadline)
        expired = await self._detach_expired()
        await self._close_many(expired)

        opening: _OpeningState | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP connection supervisor is closed")
            if reusable and purpose != "mutation":
                selected = self._find_reusable_locked(fence, purpose)
                if selected is not None:
                    lease_token = _lease_token()
                    selected.leases.add(lease_token)
                    if on_lost is not None:
                        selected.loss_callbacks = (*selected.loss_callbacks, on_lost)
                    now = self._clock()
                    selected.last_used_monotonic = now
                    selected.idle_expires_monotonic = min(
                        selected.absolute_expires_monotonic,
                        now + self._idle_ttl_s,
                    )
                    return self._lease(selected, lease_token)
            if len(self._connections) + len(self._openings) >= self._max_connections:
                raise RuntimeError("MCP connection limit reached")
            opening_id = _lease_token()
            opening = _OpeningState(
                opening_id,
                fence,
                purpose,
                task_affine=task_affine,
                owner_task=asyncio.current_task(),
                owner_loop=asyncio.get_running_loop(),
            )
            self._openings[opening_id] = opening

        assert opening is not None
        session: Any = None
        try:
            session = await self._run_factory(factory, opening, selected_deadline)
            if session is None:
                raise TypeError("MCP session factory returned None")
            with self._lock:
                self._openings.pop(opening.opening_id, None)
                invalidated_reason = opening.invalidated_reason
                if self._closed and invalidated_reason is None:
                    invalidated_reason = "supervisor_closed"
                if invalidated_reason is None:
                    now = self._clock()
                    self._sequence += 1
                    connection_id = f"mcp-connection-{self._sequence}"
                    lease_token = _lease_token()
                    selected = _ConnectionState(
                        connection_id=connection_id,
                        fence=fence,
                        purpose=purpose,
                        session=session,
                        created_monotonic=now,
                        last_used_monotonic=now,
                        absolute_expires_monotonic=now + self._absolute_ttl_s,
                        idle_expires_monotonic=min(
                            now + self._idle_ttl_s,
                            now + self._absolute_ttl_s,
                        ),
                        leases={lease_token},
                        loss_callbacks=(() if on_lost is None else (on_lost,)),
                        task_affine=task_affine,
                        owner_task=opening.owner_task,
                        owner_loop=opening.owner_loop,
                    )
                    self._connections[connection_id] = selected
                    selected.expiry_task = asyncio.create_task(
                        self._expiry_worker(connection_id),
                        name=f"agent-libos-mcp-connection-expiry-{connection_id}",
                    )
                    return self._lease(selected, lease_token)
            await self._close_session(session, inline=task_affine)
            raise RuntimeError(
                f"MCP connection opening invalidated: {_bounded_reason(invalidated_reason)}"
            )
        except BaseException:
            with self._lock:
                self._openings.pop(opening.opening_id, None)
            raise

    async def release(
        self,
        connection: McpManagedConnection | str,
        *,
        lease_token: str | None = None,
        keep_alive: bool = False,
    ) -> None:
        """Release exactly one lease; duplicate/foreign tokens cannot consume another."""

        connection_id, selected_token = _release_identity(connection, lease_token)
        detached: list[tuple[_ConnectionState, str | None]] = []
        with self._lock:
            selected = self._connections.get(connection_id)
            if selected is None:
                return
            if selected_token is None:
                if len(selected.leases) != 1:
                    raise ValueError("MCP connection release requires an exact lease token")
                selected_token = next(iter(selected.leases))
            if selected_token not in selected.leases:
                return
            selected.leases.remove(selected_token)
            selected.last_used_monotonic = self._clock()
            if selected.leases:
                return
            if (
                keep_alive
                and selected.purpose != "mutation"
                and selected.state == "active"
                and self._clock() < selected.absolute_expires_monotonic
            ):
                selected.idle_expires_monotonic = min(
                    selected.absolute_expires_monotonic,
                    self._clock() + self._idle_ttl_s,
                )
                self._reschedule_expiry_locked(selected)
                return
            detached.append(self._detach_locked(selected, state="closed", reason=None))
        await self._close_many(detached)

    async def mark_lost(self, connection: McpManagedConnection | str, reason: str) -> None:
        connection_id = (
            connection.connection_id
            if isinstance(connection, McpManagedConnection)
            else connection
        )
        detached: list[tuple[_ConnectionState, str | None]] = []
        with self._lock:
            selected = self._connections.get(connection_id)
            if selected is None:
                return
            detached.append(
                self._detach_locked(selected, state="lost", reason=_bounded_reason(reason))
            )
        await self._close_many(detached)

    async def invalidate_server(
        self,
        server_id: str,
        *,
        current_spec_sha256: str | None = None,
        current_registry_generation: int | None = None,
    ) -> None:
        def matches(fence: McpConnectionFence) -> bool:
            return fence.server_id == server_id and (
                current_spec_sha256 is None
                or fence.server_spec_sha256 != current_spec_sha256
                or (
                    current_registry_generation is not None
                    and fence.registry_generation != current_registry_generation
                )
            )

        await self._invalidate(matches, "registry_fence_changed")

    def invalidate_server_nowait(
        self,
        server_id: str,
        *,
        current_spec_sha256: str | None = None,
        current_registry_generation: int | None = None,
    ) -> None:
        """Synchronously revoke a registry fence and schedule bounded cleanup.

        This seam is safe after an already-committed registry transaction: it
        never waits for or propagates a Provider failure.  Connections are
        removed from the reusable catalog before the method returns.
        """

        if type(server_id) is not str or not server_id:
            return

        def matches(fence: McpConnectionFence) -> bool:
            return fence.server_id == server_id and (
                current_spec_sha256 is None
                or fence.server_spec_sha256 != current_spec_sha256
                or (
                    current_registry_generation is not None
                    and fence.registry_generation != current_registry_generation
                )
            )

        self._invalidate_nowait(matches, "registry_fence_changed")

    async def invalidate_auth(
        self, auth_principal_sha256: str, *, current_generation: int | None = None
    ) -> None:
        await self._invalidate(
            lambda fence: fence.auth_principal_sha256 == auth_principal_sha256
            and (
                current_generation is None
                or fence.auth_generation != current_generation
            ),
            "auth_fence_changed",
        )

    def invalidate_auth_nowait(
        self,
        auth_principal_sha256: str,
        *,
        current_generation: int | None = None,
    ) -> None:
        if type(auth_principal_sha256) is not str or not auth_principal_sha256:
            return
        self._invalidate_nowait(
            lambda fence: fence.auth_principal_sha256 == auth_principal_sha256
            and (
                current_generation is None
                or fence.auth_generation != current_generation
            ),
            "auth_fence_changed",
        )

    async def close_owner(self, owner: str) -> None:
        await self._invalidate(lambda fence: fence.owner == owner, "owner_closed")

    def close_owner_nowait(self, owner: str) -> None:
        if type(owner) is not str or not owner:
            return
        self._invalidate_nowait(lambda fence: fence.owner == owner, "owner_closed")

    async def close(self) -> None:
        detached: list[tuple[_ConnectionState, str | None]] = []
        with self._lock:
            self._closed = True
            for opening in self._openings.values():
                opening.invalidated_reason = "supervisor_closed"
                self._wake_opening(opening)
            for selected in list(self._connections.values()):
                detached.append(
                    self._detach_locked(
                        selected,
                        state="lost",
                        reason="supervisor_closed",
                    )
                )
        await self._close_many(detached)

    async def snapshot(self) -> tuple[McpManagedConnection, ...]:
        expired = await self._detach_expired()
        await self._close_many(expired)
        with self._lock:
            return tuple(self._snapshot(selected) for selected in self._connections.values())

    async def _invalidate(
        self,
        predicate: Callable[[McpConnectionFence], bool],
        reason: str,
    ) -> None:
        detached: list[tuple[_ConnectionState, str | None]] = []
        with self._lock:
            for opening in self._openings.values():
                if predicate(opening.fence):
                    opening.invalidated_reason = reason
                    self._wake_opening(opening)
            for selected in list(self._connections.values()):
                if predicate(selected.fence):
                    detached.append(
                        self._detach_locked(selected, state="lost", reason=reason)
                    )
        await self._close_many(detached)

    def _invalidate_nowait(
        self,
        predicate: Callable[[McpConnectionFence], bool],
        reason: str,
    ) -> None:
        detached: list[tuple[_ConnectionState, str | None]] = []
        try:
            with self._lock:
                for opening in self._openings.values():
                    if predicate(opening.fence):
                        opening.invalidated_reason = reason
                        self._wake_opening(opening)
                for selected in list(self._connections.values()):
                    if predicate(selected.fence):
                        detached.append(
                            self._detach_locked(
                                selected,
                                state="lost",
                                reason=reason,
                            )
                        )
        except BaseException:
            # This is a post-commit revocation latch.  Never report a false
            # registry/OAuth rollback to the caller.
            return
        self._schedule_close_many(detached)

    def _find_reusable_locked(
        self, fence: McpConnectionFence, purpose: str
    ) -> _ConnectionState | None:
        now = self._clock()
        for selected in self._connections.values():
            if (
                selected.state == "active"
                and selected.fence == fence
                and selected.purpose == purpose
                and now < selected.absolute_expires_monotonic
            ):
                return selected
        return None

    async def _detach_expired(
        self,
    ) -> list[tuple[_ConnectionState, str | None]]:
        detached: list[tuple[_ConnectionState, str | None]] = []
        with self._lock:
            now = self._clock()
            for selected in list(self._connections.values()):
                if now >= selected.absolute_expires_monotonic:
                    detached.append(
                        self._detach_locked(
                            selected,
                            state="lost",
                            reason="absolute_ttl_expired",
                        )
                    )
                elif not selected.leases and now >= selected.idle_expires_monotonic:
                    detached.append(
                        self._detach_locked(selected, state="closed", reason=None)
                    )
        return detached

    def _detach_locked(
        self,
        selected: _ConnectionState,
        *,
        state: Literal["lost", "closed"],
        reason: str | None,
    ) -> tuple[_ConnectionState, str | None]:
        self._connections.pop(selected.connection_id, None)
        selected.state = state
        selected.lost_reason = _bounded_reason(reason) if reason is not None else None
        expiry = selected.expiry_task
        if expiry is not None and expiry is not _current_task_or_none():
            _cancel_task_threadsafe(expiry, selected.owner_loop)
        return selected, selected.lost_reason

    def _reschedule_expiry_locked(self, selected: _ConnectionState) -> None:
        expiry = selected.expiry_task
        if expiry is not None and expiry is not _current_task_or_none():
            expiry.cancel()
        selected.expiry_task = asyncio.create_task(
            self._expiry_worker(selected.connection_id),
            name=(
                "agent-libos-mcp-connection-expiry-"
                f"{selected.connection_id}"
            ),
        )

    async def _expiry_worker(self, connection_id: str) -> None:
        try:
            while True:
                with self._lock:
                    selected = self._connections.get(connection_id)
                    if selected is None:
                        return
                    now = self._clock()
                    if now >= selected.absolute_expires_monotonic:
                        reason = "absolute_ttl_expired"
                        delay = 0.0
                    elif not selected.leases and now >= selected.idle_expires_monotonic:
                        reason = "idle_ttl_expired"
                        delay = 0.0
                    else:
                        reason = None
                        expires = selected.absolute_expires_monotonic
                        if not selected.leases:
                            expires = min(expires, selected.idle_expires_monotonic)
                        delay = max(0.0, expires - now)
                if reason is not None:
                    if reason == "absolute_ttl_expired":
                        await self.mark_lost(connection_id, reason)
                    else:
                        await self._close_idle(connection_id)
                    return
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

    async def _close_idle(self, connection_id: str) -> None:
        detached: list[tuple[_ConnectionState, str | None]] = []
        with self._lock:
            selected = self._connections.get(connection_id)
            if (
                selected is not None
                and not selected.leases
                and self._clock() >= selected.idle_expires_monotonic
            ):
                detached.append(
                    self._detach_locked(selected, state="closed", reason=None)
                )
        await self._close_many(detached)

    async def _run_factory(
        self,
        factory: SessionFactory,
        opening: _OpeningState,
        deadline: float,
    ) -> Any:
        if opening.task_affine:
            return await self._run_task_affine_factory(factory, opening, deadline)
        return await self._run_detached_factory(factory, opening, deadline)

    async def _run_task_affine_factory(
        self,
        factory: SessionFactory,
        opening: _OpeningState,
        deadline: float,
    ) -> Any:
        timeout = max(0.0, deadline - self._clock())
        if not inspect.iscoroutinefunction(factory):
            raise TypeError("MCP session factory must be asynchronous")
        try:
            async with asyncio.timeout(timeout):
                selected = await factory()
        except asyncio.CancelledError:
            if opening.invalidated_reason is not None:
                raise RuntimeError(
                    "MCP connection opening invalidated: "
                    f"{_bounded_reason(opening.invalidated_reason)}"
                ) from None
            raise
        if opening.invalidated_reason is not None:
            await self._close_session(selected, inline=True)
            raise RuntimeError(
                "MCP connection opening invalidated: "
                f"{_bounded_reason(opening.invalidated_reason)}"
            )
        if self._clock() >= deadline:
            await self._close_session(selected, inline=True)
            raise TimeoutError("MCP connection opening deadline exceeded")
        return selected

    async def _run_detached_factory(
        self,
        factory: SessionFactory,
        opening: _OpeningState,
        deadline: float,
    ) -> Any:
        if not inspect.iscoroutinefunction(factory):
            raise TypeError("MCP session factory must be asynchronous")

        async def invoke() -> Any:
            return await factory()

        task = asyncio.create_task(invoke(), name="agent-libos-mcp-session-open")
        invalidated = asyncio.create_task(opening.invalidated.wait())
        try:
            timeout = max(0.0, deadline - self._clock())
            done, _ = await asyncio.wait(
                {task, invalidated},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if invalidated in done:
                self._close_late_factory_result(task)
                raise RuntimeError(
                    "MCP connection opening invalidated: "
                    f"{_bounded_reason(opening.invalidated_reason or 'fence_changed')}"
                )
            if task not in done:
                self._close_late_factory_result(task)
                raise TimeoutError("MCP connection opening deadline exceeded")
            return task.result()
        except asyncio.CancelledError:
            self._close_late_factory_result(task)
            raise
        finally:
            invalidated.cancel()

    def _close_late_factory_result(self, task: asyncio.Task[Any]) -> None:
        # The opener is an owned Provider task.  Cancellation bounds the
        # ordinary timeout/invalidation path; the completion callback still
        # closes a late handle when a hostile Provider suppresses cancellation
        # and eventually returns one.
        if not task.done():
            task.cancel()

        def finish(completed: asyncio.Task[Any]) -> None:
            if completed.cancelled():
                return
            try:
                session = completed.result()
            except BaseException:
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - loop teardown
                return
            loop.create_task(self._close_session(session, inline=False))

        task.add_done_callback(finish)

    async def _close_many(
        self, selected: list[tuple[_ConnectionState, str | None]]
    ) -> None:
        if not selected:
            return
        closing = asyncio.gather(
            *(
                self._close_detached_on_owner(state, reason)
                for state, reason in selected
            ),
            return_exceptions=True,
        )
        try:
            # A loss callback may intentionally cancel the operation task that
            # initiated mark_lost().  Detached sessions are already outside
            # the reusable catalog, so their close-once cleanup must survive
            # that caller cancellation.
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            closing.add_done_callback(_consume_task_result)
            raise

    async def _close_detached_on_owner(
        self,
        selected: _ConnectionState,
        reason: str | None,
    ) -> None:
        owner_loop = selected.owner_loop
        current_loop = asyncio.get_running_loop()
        if owner_loop is None or owner_loop is current_loop:
            await self._close_detached(selected, reason)
            return
        if owner_loop.is_closed():
            return
        cleanup = self._close_detached(selected, reason)
        try:
            submitted = asyncio.run_coroutine_threadsafe(cleanup, owner_loop)
        except BaseException:
            cleanup.close()
            return
        try:
            async with asyncio.timeout(self._close_timeout_s * 2):
                await asyncio.wrap_future(submitted)
        except BaseException:
            submitted.cancel()

    def _schedule_close_many(
        self,
        selected: list[tuple[_ConnectionState, str | None]],
    ) -> None:
        for state, reason in selected:
            loop = state.owner_loop
            if loop is None or loop.is_closed():
                continue

            def start_close(
                selected_state: _ConnectionState = state,
                selected_reason: str | None = reason,
                selected_loop: asyncio.AbstractEventLoop = loop,
            ) -> None:
                task = selected_loop.create_task(
                    self._close_detached(selected_state, selected_reason),
                    name=(
                        "agent-libos-mcp-connection-revoke-"
                        f"{selected_state.connection_id}"
                    ),
                )
                task.add_done_callback(_consume_task_result)

            try:
                if _running_loop_or_none() is loop:
                    start_close()
                else:
                    loop.call_soon_threadsafe(start_close)
            except (RuntimeError, AttributeError):
                continue

    @staticmethod
    def _wake_opening(opening: _OpeningState) -> None:
        loop = opening.owner_loop
        if loop is None or loop.is_closed():
            return

        def wake() -> None:
            opening.invalidated.set()
            if (
                opening.task_affine
                and opening.owner_task is not None
                and opening.owner_task is not asyncio.current_task()
            ):
                opening.owner_task.cancel()

        try:
            if _running_loop_or_none() is loop:
                wake()
            else:
                loop.call_soon_threadsafe(wake)
        except RuntimeError:
            return

    async def _close_detached(
        self, selected: _ConnectionState, reason: str | None
    ) -> None:
        # Publish loss before potentially slow Provider cleanup.  In
        # particular, absolute expiry must become observable as LOST at the
        # authority boundary, not only after a close timeout elapses.
        if reason is not None:
            await asyncio.gather(
                *(
                    self._run_callback(
                        callback,
                        selected.connection_id,
                        reason,
                    )
                    for callback in selected.loss_callbacks
                ),
                return_exceptions=True,
            )
        if not selected.close_started:
            selected.close_started = True
            await self._close_session(
                selected.session,
                inline=(
                    selected.task_affine
                    and selected.owner_task is asyncio.current_task()
                ),
            )

    async def _close_session(self, session: Any, *, inline: bool = False) -> None:
        close = getattr(session, "aclose", None)
        if not callable(close):
            close = getattr(session, "close", None)
        if not callable(close):
            return
        if not inspect.iscoroutinefunction(close):
            return

        async def invoke() -> None:
            await close()

        if inline:
            try:
                async with asyncio.timeout(self._close_timeout_s):
                    await invoke()
            except BaseException:
                return
            return

        task = asyncio.create_task(invoke(), name="agent-libos-mcp-session-close")
        done, _ = await asyncio.wait({task}, timeout=self._close_timeout_s)
        if task not in done:
            task.cancel()
            task.add_done_callback(_consume_task_result)
            return
        try:
            task.result()
        except BaseException:
            # Close is best effort and must not mask the operation/loss reason.
            return

    async def _run_callback(
        self, callback: LossCallback, connection_id: str, reason: str
    ) -> None:
        async def invoke() -> None:
            result = callback(connection_id, reason)
            if inspect.isawaitable(result):
                await result

        task = asyncio.create_task(invoke())
        done, _ = await asyncio.wait({task}, timeout=self._close_timeout_s)
        if task not in done:
            task.cancel()
            task.add_done_callback(_consume_task_result)
            return
        try:
            task.result()
        except BaseException:
            return

    def _validate_deadline(self, deadline: float | None) -> float:
        now = self._clock()
        if deadline is None:
            return now + self._open_timeout_s
        if (
            type(deadline) not in {int, float}
            or not math.isfinite(deadline)
            or deadline <= now
        ):
            raise TimeoutError("MCP connection opening deadline exceeded")
        return float(deadline)

    @staticmethod
    def _lease(selected: _ConnectionState, lease_token: str) -> McpManagedConnection:
        return McpManagedConnection(
            connection_id=selected.connection_id,
            lease_token=lease_token,
            fence=selected.fence,
            purpose=selected.purpose,
            session=selected.session,
            created_monotonic=selected.created_monotonic,
            last_used_monotonic=selected.last_used_monotonic,
            absolute_expires_monotonic=selected.absolute_expires_monotonic,
            idle_expires_monotonic=selected.idle_expires_monotonic,
            lease_count=len(selected.leases),
            state=selected.state,
            lost_reason=selected.lost_reason,
        )

    @staticmethod
    def _snapshot(selected: _ConnectionState) -> McpManagedConnection:
        return McpManagedConnection(
            connection_id=selected.connection_id,
            lease_token="",
            fence=selected.fence,
            purpose=selected.purpose,
            session=selected.session,
            created_monotonic=selected.created_monotonic,
            last_used_monotonic=selected.last_used_monotonic,
            absolute_expires_monotonic=selected.absolute_expires_monotonic,
            idle_expires_monotonic=selected.idle_expires_monotonic,
            lease_count=len(selected.leases),
            state=selected.state,
            lost_reason=selected.lost_reason,
        )


def _release_identity(
    connection: McpManagedConnection | str,
    lease_token: str | None,
) -> tuple[str, str | None]:
    if isinstance(connection, McpManagedConnection):
        if lease_token is not None and lease_token != connection.lease_token:
            raise ValueError("MCP connection lease token does not match the lease")
        return connection.connection_id, connection.lease_token
    if type(connection) is not str or not connection:
        raise ValueError("MCP connection id must be a non-empty string")
    return connection, lease_token


def _lease_token() -> str:
    return secrets.token_urlsafe(24)


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass


def _running_loop_or_none() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _current_task_or_none() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _cancel_task_threadsafe(
    task: asyncio.Task[Any],
    loop: asyncio.AbstractEventLoop | None,
) -> None:
    if task.done() or loop is None or loop.is_closed():
        return
    try:
        if _running_loop_or_none() is loop:
            task.cancel()
        else:
            loop.call_soon_threadsafe(task.cancel)
    except RuntimeError:
        return


def _bounded_reason(reason: object) -> str:
    selected = reason if type(reason) is str else type(reason).__name__
    return selected[:256]
