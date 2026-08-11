"""Python SDK v2 adapter for modern MCP ``subscriptions/listen``.

The adapter deliberately owns one SDK listen stream per returned handle.  A
single background task enters and exits both the governed SDK session and the
SDK listen context; this is important because the SDK uses anyio cancel scopes
which must be unwound by the task that entered them.  There is no reconnect,
re-listen, replay, or ``Last-Event-ID`` path in this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, AsyncContextManager, Literal, Protocol

from agent_libos.mcp._input import strict_json_value
from agent_libos.mcp.client import (
    McpClientBinding,
    current_mcp_client_binding,
)
from agent_libos.mcp.manifest import (
    MCP_V3_PROTOCOL_REVISION,
    MCP_V3_SUBSCRIPTION_FILTERS,
)
from agent_libos.mcp.providers import (
    McpSubscriptionProvider,
    McpSubscriptionSession,
)
from agent_libos.mcp.runtime_bridge import mcp_connection_fence
from agent_libos.mcp.supervisor import McpConnectionFence
from agent_libos.mcp.resources import (
    bounded_public_size,
    inert_resource_handle,
    reject_mcp_app_selector,
    sanitize_provider_json,
)
from agent_libos.mcp.types import McpSubscriptionEvent
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.mcp import McpProtocolMode, McpServerSpec
from agent_libos.substrate.base import ProviderEffectNotStarted
from agent_libos.utils.redaction import redact_sensitive_text


_RESOURCE_FILTER = "resourceSubscriptions"
_TASK_FILTER = "taskIds"

McpSdkSubscriptionLossReason = Literal[
    "backpressure",
    "disconnected",
    "invalid_event",
    "open_failed",
]


class McpSdkSubscriptionLost(ConnectionError):
    """A listen stream ended and must be explicitly reopened and refetched."""

    def __init__(
        self,
        message: str,
        *,
        reason: McpSdkSubscriptionLossReason,
    ) -> None:
        self.reason = reason
        super().__init__(message)


class McpSdkSubscriptionClosed(EOFError):
    """The server gracefully closed a modern subscription stream."""


@dataclass(frozen=True)
class McpSdkV2SubscriptionLimits:
    """Provider-local bounds applied before events reach the Host manager."""

    queue_events: int = 256
    event_max_bytes: int = 64 * 1024
    max_resource_subscriptions: int = 256
    max_resource_selector_bytes: int = 4096
    max_task_subscriptions: int = 1_000
    max_task_id_bytes: int = 8 * 1024

    def validate(self) -> None:
        for name, value in (
            ("queue_events", self.queue_events),
            ("event_max_bytes", self.event_max_bytes),
            ("max_resource_subscriptions", self.max_resource_subscriptions),
            ("max_resource_selector_bytes", self.max_resource_selector_bytes),
            ("max_task_subscriptions", self.max_task_subscriptions),
            ("max_task_id_bytes", self.max_task_id_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValidationError(
                    f"MCP SDK subscription {name} must be a positive integer"
                )


@dataclass(frozen=True)
class _Terminal:
    kind: Literal["closed", "lost"]
    message: str
    reason: McpSdkSubscriptionLossReason | None = None


@dataclass(frozen=True)
class _TaskIngressFailure:
    reason: McpSdkSubscriptionLossReason


@dataclass(eq=False, repr=False)
class _TaskNotificationIngress:
    """Synchronous SDK-dispatcher ingress for one exact taskIds stream."""

    request_id: str
    requested_ids: frozenset[str]
    queue: asyncio.Queue[McpSubscriptionEvent | _TaskIngressFailure]
    route: Any = None

    def __call__(self, params: Mapping[str, Any] | None) -> None:
        try:
            event = self._validated_event(params)
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.fail("backpressure")
        except (TypeError, ValueError, ValidationError):
            self.fail("invalid_event")

    def fail(self, reason: McpSdkSubscriptionLossReason) -> None:
        if self.route is not None:
            self.route.settle("lost")
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        try:
            self.queue.put_nowait(_TaskIngressFailure(reason))
        except asyncio.QueueFull:  # pragma: no cover - drained immediately above
            pass

    def _validated_event(
        self,
        params: Mapping[str, Any] | None,
    ) -> McpSubscriptionEvent:
        if type(params) is not dict:
            raise ValidationError("MCP Tasks notification params are invalid")
        strict_json_value(params, label="MCP Tasks subscription notification")
        metadata = params.get("_meta")
        if type(metadata) is not dict:
            raise ValidationError("MCP Tasks notification metadata is invalid")
        subscription_id = metadata.get(
            "io.modelcontextprotocol/subscriptionId"
        )
        if subscription_id != self.request_id:
            raise ValidationError("MCP Tasks notification belongs to another stream")
        remote_id = params.get("taskId")
        if type(remote_id) is not str or remote_id not in self._honored_ids():
            raise ValidationError("MCP Tasks notification identity was not honored")
        payload = {key: value for key, value in params.items() if key != "_meta"}
        return McpSubscriptionEvent(
            sequence=0,
            event_type="taskStatus",
            payload=payload,
            received_at=_utc_now(),
        )

    def _honored_ids(self) -> frozenset[str]:
        honored = getattr(self.route, "honored", None)
        if honored is None:
            raise ValidationError("MCP Tasks notification arrived before acknowledgement")
        dumped = honored.model_dump(by_alias=True, mode="json", exclude_none=True)
        raw_ids = dumped.get(_TASK_FILTER)
        if not isinstance(raw_ids, list) or any(type(item) is not str for item in raw_ids):
            raise ValidationError("MCP Tasks acknowledgement is invalid")
        selected = frozenset(raw_ids)
        if not selected or len(selected) != len(raw_ids):
            raise ValidationError("MCP Tasks acknowledgement is invalid")
        if not selected.issubset(self.requested_ids):
            raise ValidationError("MCP Tasks acknowledgement exceeded its request")
        return selected


@dataclass(eq=False, repr=False)
class _SdkSubscriptionHandle:
    """Opaque, provider-owned live state returned through the public SPI."""

    provider_token: object
    queue: asyncio.Queue[McpSubscriptionEvent]
    ready: asyncio.Future[None]
    handoff: asyncio.Event
    done: asyncio.Event
    close_lock: asyncio.Lock
    acknowledged_filters: tuple[str, ...] = ()
    terminal: _Terminal | None = None
    driver: asyncio.Task[None] | None = None
    close_requested: bool = False

    async def aclose(self) -> None:
        """Cancel the owner task at most once and wait for both contexts to exit."""

        async with self.close_lock:
            if not self.close_requested:
                self.close_requested = True
                self.handoff.set()
                # Cancel opening readiness before cancelling the owner. Python
                # 3.14 deliberately reports a shielded inner Future that gains
                # an exception after its outer waiter was cancelled. The
                # closing caller already owns the cancellation/deadline result,
                # so readiness must become cancellation-only in this path.
                if not self.ready.done():
                    self.ready.cancel()
                if self.driver is not None and not self.driver.done():
                    self.driver.cancel()
            driver = self.driver
        if driver is not None and driver is not asyncio.current_task():
            with contextlib.suppress(asyncio.CancelledError):
                await driver
        # A caller can be cancelled in the same event-loop slice in which the
        # owner records an opening failure.  Retrieve that private future's
        # exception during close so it cannot surface later as ``Future
        # exception was never retrieved``; the caller still receives its
        # original cancellation/deadline error from ``listen``.
        if self.ready.done() and not self.ready.cancelled():
            with contextlib.suppress(BaseException):
                self.ready.exception()


class McpSdkV2SubscriptionSessionFactory(Protocol):
    """Enter the primitive-owned raw governed SDK context.

    This factory must not acquire an :class:`McpConnectionSupervisor` itself:
    ``McpSubscriptionManager`` registers the provider-owned handle with the
    one outer supervisor fence.  The explicit binding is the exact protected
    operation snapshot captured before the owner task is started.
    """

    def __call__(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        binding: McpClientBinding,
        task_notification_ingress: Callable[[Mapping[str, Any] | None], None]
        | None = None,
    ) -> AsyncContextManager[Any]: ...


ResourceSubscriptionsResolver = Callable[
    [McpServerSpec, McpClientBinding], Sequence[str]
]


class TaskSubscriptionsResolver(Protocol):
    def __call__(
        self,
        *,
        fence: McpConnectionFence,
    ) -> Sequence[str]: ...


class McpSdkV2SubscriptionProvider(McpSubscriptionProvider):
    """Adapt an already-governed Python SDK v2 session to the Host SPI.

    ``resourceSubscriptions`` maps only to Host-registered selectors returned
    by ``resource_subscriptions_resolver``.  The default reads the captured
    exact v3 binding and therefore cannot accept an ad-hoc URI from a caller.  The
    core SDK has no ``taskIds`` filter; requesting it fails closed instead of
    falsely acknowledging an unsupported Tasks extension surface.
    """

    mcp_manifest_schema_version: Literal[3] = 3
    mcp_protocol_revision: Literal["2026-07-28"] = MCP_V3_PROTOCOL_REVISION

    def __init__(
        self,
        session_factory: McpSdkV2SubscriptionSessionFactory,
        *,
        binding_resolver: Callable[[], McpClientBinding] | None = None,
        sensitive_values_resolver: Callable[[str], tuple[str, ...]] | None = None,
        resource_subscriptions_resolver: ResourceSubscriptionsResolver | None = None,
        task_subscriptions_resolver: TaskSubscriptionsResolver | None = None,
        limits: McpSdkV2SubscriptionLimits | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], str] | None = None,
    ) -> None:
        if not callable(session_factory):
            raise ValidationError("MCP SDK subscription session factory is invalid")
        self.session_factory = session_factory
        self.binding_resolver = (
            current_mcp_client_binding
            if binding_resolver is None
            else binding_resolver
        )
        self.sensitive_values_resolver = (
            (lambda _id: ())
            if sensitive_values_resolver is None
            else sensitive_values_resolver
        )
        self.resource_subscriptions_resolver = (
            _registered_resource_selectors
            if resource_subscriptions_resolver is None
            else resource_subscriptions_resolver
        )
        self.task_subscriptions_resolver = task_subscriptions_resolver
        self.limits = McpSdkV2SubscriptionLimits() if limits is None else limits
        if not isinstance(self.limits, McpSdkV2SubscriptionLimits):
            raise ValidationError("MCP SDK subscription limits are invalid")
        self.limits.validate()
        self.monotonic = monotonic
        self.utc_now = _utc_now if utc_now is None else utc_now
        for label, callback in (
            ("binding resolver", self.binding_resolver),
            ("sensitive-value resolver", self.sensitive_values_resolver),
            ("Resource resolver", self.resource_subscriptions_resolver),
            ("monotonic clock", self.monotonic),
            ("UTC clock", self.utc_now),
        ):
            if not callable(callback):
                raise ValidationError(
                    f"MCP SDK subscription {label} is invalid"
                )
        self._provider_token = object()

    def bind_task_subscriptions_resolver(
        self,
        resolver: TaskSubscriptionsResolver,
    ) -> None:
        if not callable(resolver):
            raise ValidationError("MCP SDK Tasks subscription resolver is invalid")
        if (
            self.task_subscriptions_resolver is not None
            and self.task_subscriptions_resolver is not resolver
        ):
            raise RuntimeError("MCP SDK Tasks subscription resolver is already bound")
        self.task_subscriptions_resolver = resolver

    async def listen(
        self,
        server: McpServerSpec,
        filters: tuple[str, ...],
        *,
        deadline: float,
    ) -> McpSubscriptionSession:
        _require_exact_server(server)
        selected_filters = _validate_filters(filters)
        selected_deadline = _validate_deadline(deadline, self.monotonic())
        binding = self.binding_resolver()
        _validate_binding(binding, server)
        sensitive_values = _sensitive_values(
            tuple(
                dict.fromkeys(
                    (
                        *binding.sensitive_values,
                        *self.sensitive_values_resolver(server.server_id),
                    )
                )
            )
        )
        resource_selectors: tuple[str, ...] = ()
        if _RESOURCE_FILTER in selected_filters:
            resource_selectors = _resource_selectors(
                self.resource_subscriptions_resolver(server, binding),
                limits=self.limits,
            )
        task_targets: tuple[str, ...] = ()
        if _TASK_FILTER in selected_filters:
            resolver = self.task_subscriptions_resolver
            if resolver is None:
                raise ValidationError(
                    "MCP taskIds subscription target resolver is unavailable"
                )
            task_targets = _task_targets(
                resolver(fence=mcp_connection_fence(binding)),
                limits=self.limits,
            )

        loop = asyncio.get_running_loop()
        handle = _SdkSubscriptionHandle(
            provider_token=self._provider_token,
            queue=asyncio.Queue(maxsize=self.limits.queue_events),
            ready=loop.create_future(),
            handoff=asyncio.Event(),
            done=asyncio.Event(),
            close_lock=asyncio.Lock(),
        )
        handle.driver = asyncio.create_task(
            self._drive(
                handle,
                server,
                selected_filters,
                resource_selectors,
                task_targets,
                selected_deadline,
                sensitive_values,
                binding,
            ),
            name=f"agent-libos-mcp-sdk-listen-{server.server_id}",
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(handle.ready),
                timeout=_remaining(selected_deadline, self.monotonic()),
            )
        except BaseException:
            await handle.aclose()
            raise
        # Keep the exact owner task alive until the caller that awaited
        # ``listen`` has synchronously validated the returned session.  A
        # server may acknowledge, emit, and close in one event-loop slice;
        # releasing on the next loop turn avoids misclassifying that legitimate
        # terminal stream as a forged already-done owner task.
        loop.call_soon(handle.handoff.set)
        return McpSubscriptionSession(
            handle=handle,
            owner_task=handle.driver,
            acknowledged_filters=handle.acknowledged_filters,
        )

    async def receive(
        self,
        handle: Any,
        *,
        deadline: float,
    ) -> McpSubscriptionEvent:
        selected = self._require_handle(handle)
        selected_deadline = _validate_deadline(deadline, self.monotonic())
        while True:
            try:
                return selected.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if selected.done.is_set():
                _raise_terminal(selected.terminal)

            event_waiter = asyncio.create_task(selected.queue.get())
            done_waiter = asyncio.create_task(selected.done.wait())
            try:
                await asyncio.wait_for(
                    asyncio.wait(
                        (event_waiter, done_waiter),
                        return_when=asyncio.FIRST_COMPLETED,
                    ),
                    timeout=_remaining(selected_deadline, self.monotonic()),
                )
                # Prefer an event if event delivery and stream termination land
                # in the same event-loop slice; terminal state follows after
                # the already-received bounded backlog is drained.
                if event_waiter.done():
                    return event_waiter.result()
            finally:
                for waiter in (event_waiter, done_waiter):
                    if not waiter.done():
                        waiter.cancel()
                await asyncio.gather(
                    event_waiter,
                    done_waiter,
                    return_exceptions=True,
                )

    async def close(self, handle: Any) -> None:
        await self._require_handle(handle).aclose()

    def _require_handle(self, handle: Any) -> _SdkSubscriptionHandle:
        if type(handle) is McpSubscriptionSession:
            handle = handle.handle
        if (
            not isinstance(handle, _SdkSubscriptionHandle)
            or handle.provider_token is not self._provider_token
        ):
            raise ValidationError("MCP SDK subscription handle is invalid")
        return handle

    async def _drive(
        self,
        handle: _SdkSubscriptionHandle,
        server: McpServerSpec,
        filters: tuple[str, ...],
        resource_selectors: tuple[str, ...],
        task_targets: tuple[str, ...],
        deadline: float,
        sensitive_values: tuple[str, ...],
        binding: McpClientBinding,
    ) -> None:
        terminal: _Terminal | None = None
        try:
            _remaining(deadline, self.monotonic())
            task_ingress = _new_task_ingress(task_targets, limits=self.limits)
            if task_ingress is None:
                session_context = self.session_factory(
                    server,
                    deadline=deadline,
                    binding=binding,
                )
            else:
                session_context = self.session_factory(
                    server,
                    deadline=deadline,
                    binding=binding,
                    task_notification_ingress=task_ingress,
                )
            async with session_context as selected:
                # OAuth access material is intentionally obtained only while
                # entering the raw governed context.  Refresh the exact-secret
                # snapshot after entry so notifications cannot reflect the
                # operation token through an event or disconnect diagnostic.
                sensitive_values = _sensitive_values(
                    tuple(
                        dict.fromkeys(
                            (
                                *sensitive_values,
                                *self.sensitive_values_resolver(server.server_id),
                            )
                        )
                    )
                )
                session = _exact_modern_session(selected)
                _remaining(deadline, self.monotonic())
                listen_context = _sdk_listen_context(
                    selected,
                    session,
                    filters=filters,
                    resource_selectors=resource_selectors,
                    task_targets=task_targets,
                    task_ingress=task_ingress,
                    deadline=deadline,
                )
                async with listen_context as subscription:
                    handle.acknowledged_filters = _acknowledged_filters(
                        subscription,
                        requested=filters,
                        requested_resource_selectors=resource_selectors,
                    )
                    if not handle.ready.done():
                        handle.ready.set_result(None)
                    stream_terminal: _Terminal | None = None
                    try:
                        async for raw_event in subscription:
                            event = _project_event(
                                raw_event,
                                server_id=server.server_id,
                                sensitive_values=sensitive_values,
                                received_at=self.utc_now(),
                            )
                            try:
                                bounded_public_size(
                                    event,
                                    maximum=min(
                                        self.limits.event_max_bytes,
                                        server.max_response_bytes,
                                    ),
                                    label="MCP SDK subscription event",
                                )
                            except ValidationError as exc:
                                raise _AdapterLoss(
                                    "backpressure",
                                    "MCP SDK subscription event exceeds its byte bound",
                                ) from exc
                            try:
                                handle.queue.put_nowait(event)
                            except asyncio.QueueFull as exc:
                                raise _AdapterLoss(
                                    "backpressure",
                                    "MCP SDK subscription event queue overflowed",
                                ) from exc
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # Consume stream failures before leaving the governed
                        # context.  Its boundary intentionally erases unknown
                        # provider diagnostics; here we still have the exact
                        # operation secret snapshot needed for safe typing and
                        # redaction.
                        stream_terminal = _safe_loss(
                            exc,
                            sensitive_values=sensitive_values,
                            default_reason="disconnected",
                        )
                terminal = stream_terminal or _Terminal(
                    "closed", "MCP server gracefully closed the subscription"
                )
        except asyncio.CancelledError:
            terminal = _Terminal("closed", "MCP subscription was closed locally")
            if not handle.ready.done():
                handle.ready.cancel()
            raise
        except TimeoutError:
            terminal = _Terminal(
                "lost",
                "MCP subscription opening exceeded its absolute deadline",
                "open_failed",
            )
            if not handle.ready.done():
                handle.ready.set_exception(
                    TimeoutError("MCP SDK subscription deadline expired")
                )
        except ProviderEffectNotStarted as exc:
            if not handle.ready.done():
                handle.ready.set_exception(exc)
            else:
                terminal = _safe_loss(
                    exc,
                    sensitive_values=sensitive_values,
                    default_reason="disconnected",
                )
        except Exception as exc:
            terminal = _safe_loss(
                exc,
                sensitive_values=sensitive_values,
                default_reason=(
                    "disconnected" if handle.ready.done() else "open_failed"
                ),
            )
            if not handle.ready.done():
                handle.ready.set_exception(
                    McpSdkSubscriptionLost(
                        terminal.message,
                        reason=terminal.reason or "open_failed",
                    )
                )
        finally:
            if terminal is None:
                terminal = _Terminal("closed", "MCP subscription was closed locally")
            await handle.handoff.wait()
            handle.terminal = terminal
            handle.done.set()


class _AdapterLoss(RuntimeError):
    def __init__(self, reason: McpSdkSubscriptionLossReason, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class _ExtendedSdkSubscription:
    """Merge the SDK core route with the exact Tasks extension ingress."""

    def __init__(self, route: Any, ingress: _TaskNotificationIngress) -> None:
        self._route = route
        self._ingress = ingress
        self.honored = route.honored

    def __aiter__(self) -> _ExtendedSdkSubscription:
        return self

    async def __anext__(self) -> Any:
        try:
            selected = self._ingress.queue.get_nowait()
        except asyncio.QueueEmpty:
            selected = await self._wait_next()
        if isinstance(selected, _TaskIngressFailure):
            raise _AdapterLoss(
                selected.reason,
                "MCP Tasks subscription ingress failed",
            )
        if isinstance(selected, str):
            if selected == "lost":
                raise _AdapterLoss(
                    "disconnected",
                    "MCP Tasks subscription connection was lost",
                )
            raise StopAsyncIteration
        if isinstance(selected, McpSubscriptionEvent):
            return selected
        self._route.consume(selected)
        return selected

    async def _wait_next(self) -> Any:
        route_waiter = asyncio.create_task(self._route.next_event())
        task_waiter = asyncio.create_task(self._ingress.queue.get())
        try:
            done, _pending = await asyncio.wait(
                (route_waiter, task_waiter),
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Invalid/flooded Tasks ingress is fail-closed even if a core event
            # landed in the same event-loop slice.
            if task_waiter in done:
                return task_waiter.result()
            return route_waiter.result()
        finally:
            for waiter in (route_waiter, task_waiter):
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(
                route_waiter,
                task_waiter,
                return_exceptions=True,
            )


def _sdk_listen_context(
    selected: Any,
    session: Any,
    *,
    filters: tuple[str, ...],
    resource_selectors: tuple[str, ...],
    task_targets: tuple[str, ...],
    task_ingress: _TaskNotificationIngress | None,
    deadline: float,
) -> Any:
    if task_targets:
        if task_ingress is None:
            raise ValidationError("MCP Tasks subscription ingress is unavailable")
        return _sdk_task_listen_context(
            session,
            filters=filters,
            resource_selectors=resource_selectors,
            task_targets=task_targets,
            task_ingress=task_ingress,
            deadline=deadline,
        )
    keywords = {
        "tools_list_changed": "toolsListChanged" in filters,
        "prompts_list_changed": "promptsListChanged" in filters,
        "resources_list_changed": "resourcesListChanged" in filters,
        "resource_subscriptions": resource_selectors,
    }
    listen = getattr(selected, "listen", None)
    if callable(listen):
        context = listen(**keywords)
    else:
        try:
            from mcp.client.subscriptions import listen as sdk_listen
        except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
            raise ValidationError("MCP Python SDK v2 is unavailable") from exc
        context = sdk_listen(session, **keywords)
    if not hasattr(context, "__aenter__") or not hasattr(context, "__aexit__"):
        raise ValidationError("MCP SDK listen returned an invalid context manager")
    return context


@contextlib.asynccontextmanager
async def _sdk_task_listen_context(
    session: Any,
    *,
    filters: tuple[str, ...],
    resource_selectors: tuple[str, ...],
    task_targets: tuple[str, ...],
    task_ingress: _TaskNotificationIngress,
    deadline: float,
) -> Any:
    """Open one extension filter through the SDK's exact raw listen route."""

    try:
        import anyio
        import mcp_types as types
        from mcp.shared.dispatcher import CallOptions
        from mcp.shared.exceptions import MCPError
    except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
        raise ValidationError("MCP Python SDK v2 is unavailable") from exc

    raw_filter: dict[str, Any] = {
        "toolsListChanged": True if "toolsListChanged" in filters else None,
        "promptsListChanged": True if "promptsListChanged" in filters else None,
        "resourcesListChanged": (
            True if "resourcesListChanged" in filters else None
        ),
        "resourceSubscriptions": list(resource_selectors) or None,
        _TASK_FILTER: list(task_targets),
    }
    notifications = types.SubscriptionFilter.model_validate(raw_filter)
    request = types.SubscriptionsListenRequest(
        params=types.SubscriptionsListenRequestParams(
            notifications=notifications,
        )
    )
    data = request.model_dump(by_alias=True, mode="json", exclude_none=True)
    request_id = task_ingress.request_id
    options: CallOptions = {"request_id": request_id}
    session._stamp(data, options)
    task_group = getattr(session, "_task_group", None)
    if task_group is None:
        raise ValidationError("MCP Tasks listen requires an entered SDK session")
    route = session._register_listen_route(request_id)
    task_ingress.route = route
    driver_scope = anyio.CancelScope()

    async def drive() -> None:
        with driver_scope:
            try:
                await session._dispatcher.send_raw_request(
                    data["method"], data.get("params"), options
                )
            except (MCPError, ValueError):
                route.settle("lost")
                return
            route.set_acked(types.SubscriptionFilter())
            route.settle("graceful")

    try:
        task_group.start_soon(drive)
        with anyio.fail_after(_remaining(deadline, time.monotonic())):
            await route.acked.wait()
        if route.honored is None:
            raise _AdapterLoss(
                "open_failed",
                "MCP Tasks subscription ended before acknowledgement",
            )
        _task_acknowledgement(
            route.honored,
            requested=frozenset(task_targets),
        )
        yield _ExtendedSdkSubscription(route, task_ingress)
    finally:
        route.settle("local")
        driver_scope.cancel()
        session._unregister_listen_route(request_id)


def _exact_modern_session(selected: Any) -> Any:
    session = getattr(selected, "session", selected)
    if str(getattr(session, "protocol_version", "")) != MCP_V3_PROTOCOL_REVISION:
        raise ValidationError(
            "MCP SDK subscriptions require exact protocol 2026-07-28"
        )
    return session


def _project_event(
    value: Any,
    *,
    server_id: str,
    sensitive_values: tuple[str, ...],
    received_at: str,
) -> McpSubscriptionEvent:
    if type(value) is McpSubscriptionEvent:
        if value.event_type != "taskStatus" or type(value.payload) is not dict:
            raise _AdapterLoss(
                "invalid_event", "MCP SDK Tasks subscription event is invalid"
            )
        return McpSubscriptionEvent(
            sequence=0,
            event_type="taskStatus",
            payload=sanitize_provider_json(
                value.payload,
                sensitive_values=sensitive_values,
            ),
            received_at=redact_sensitive_text(
                received_at,
                sensitive_values=sensitive_values,
            ),
        )
    try:
        from mcp.client.subscriptions import (
            PromptsListChanged,
            ResourcesListChanged,
            ResourceUpdated,
            ToolsListChanged,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
        raise ValidationError("MCP Python SDK v2 is unavailable") from exc

    event_type: str
    payload: Any
    if isinstance(value, ToolsListChanged):
        event_type, payload = "toolsListChanged", {}
    elif isinstance(value, PromptsListChanged):
        event_type, payload = "promptsListChanged", {}
    elif isinstance(value, ResourcesListChanged):
        event_type, payload = "resourcesListChanged", {}
    elif isinstance(value, ResourceUpdated):
        uri = getattr(value, "uri", None)
        if type(uri) is not str or not uri or "\x00" in uri:
            raise _AdapterLoss(
                "invalid_event", "MCP SDK resource update selector is invalid"
            )
        # The remote URI is transport state, not a model/GUI selector.  Expose
        # only the same inert one-way handle used by ResourceLink projections.
        event_type = "resourceUpdated"
        payload = {"resource_handle": inert_resource_handle(server_id, uri)}
    else:
        raise _AdapterLoss(
            "invalid_event", "MCP SDK subscription returned an unsupported event"
        )
    public_payload = sanitize_provider_json(
        payload,
        sensitive_values=sensitive_values,
    )
    return McpSubscriptionEvent(
        sequence=0,
        event_type=event_type,
        payload=public_payload,
        received_at=redact_sensitive_text(
            received_at,
            sensitive_values=sensitive_values,
        ),
    )


def _acknowledged_filters(
    subscription: Any,
    *,
    requested: tuple[str, ...],
    requested_resource_selectors: tuple[str, ...],
) -> tuple[str, ...]:
    honored = getattr(subscription, "honored", None)
    if honored is None:
        raise ValidationError("MCP SDK subscription omitted filter acknowledgement")
    selected = _core_acknowledgements(honored)
    if _resources_acknowledged(honored, requested_resource_selectors):
        selected.add(_RESOURCE_FILTER)
    if _tasks_acknowledged(honored, requested):
        selected.add(_TASK_FILTER)
    if not selected.issubset(requested):
        raise ValidationError("MCP server acknowledged unrequested subscription filters")
    return tuple(item for item in requested if item in selected)


def _core_acknowledgements(honored: Any) -> set[str]:
    selected: set[str] = set()
    for public, attribute in (
        ("toolsListChanged", "tools_list_changed"),
        ("promptsListChanged", "prompts_list_changed"),
        ("resourcesListChanged", "resources_list_changed"),
    ):
        value = getattr(honored, attribute, None)
        if value not in {None, False, True}:
            raise ValidationError("MCP SDK subscription acknowledgement is invalid")
        if value is True:
            selected.add(public)
    return selected


def _resources_acknowledged(
    honored: Any,
    requested_resource_selectors: tuple[str, ...],
) -> bool:
    raw_resources = getattr(honored, "resource_subscriptions", None)
    if raw_resources is None:
        return False
    if isinstance(raw_resources, str) or not isinstance(raw_resources, Sequence):
        raise ValidationError("MCP SDK resource acknowledgement is invalid")
    honored_resources = tuple(raw_resources)
    if any(type(item) is not str for item in honored_resources):
        raise ValidationError("MCP SDK resource acknowledgement is invalid")
    if not set(honored_resources).issubset(requested_resource_selectors):
        raise ValidationError(
            "MCP server acknowledged unrequested resource subscriptions"
        )
    return bool(honored_resources)


def _tasks_acknowledged(honored: Any, requested: tuple[str, ...]) -> bool:
    if _TASK_FILTER not in requested:
        return False
    model_dump = getattr(honored, "model_dump", None)
    if not callable(model_dump):
        raise ValidationError("MCP SDK Tasks acknowledgement is invalid")
    dumped = model_dump(by_alias=True, mode="json", exclude_none=True)
    raw_tasks = dumped.get(_TASK_FILTER)
    if raw_tasks is None:
        return False
    if (
        not isinstance(raw_tasks, list)
        or not raw_tasks
        or any(type(item) is not str or not item for item in raw_tasks)
        or len(set(raw_tasks)) != len(raw_tasks)
    ):
        raise ValidationError("MCP SDK Tasks acknowledgement is invalid")
    return True


def _validate_filters(filters: tuple[str, ...]) -> tuple[str, ...]:
    if type(filters) is not tuple or not filters:
        raise ValidationError("MCP SDK subscription filters must be a non-empty tuple")
    if any(type(item) is not str for item in filters):
        raise ValidationError("MCP SDK subscription filters must contain strings")
    if len(set(filters)) != len(filters):
        raise ValidationError("MCP SDK subscription filters must be unique")
    unknown = set(filters) - MCP_V3_SUBSCRIPTION_FILTERS
    if unknown:
        raise ValidationError(
            f"unsupported MCP SDK subscription filters: {sorted(unknown)}"
        )
    return filters


def _registered_resource_selectors(
    server: McpServerSpec,
    binding: McpClientBinding,
) -> tuple[str, ...]:
    if binding.manifest.server_id != server.server_id:
        raise ValidationError("MCP subscription binding belongs to another server")
    return tuple(resource.remote_uri for resource in binding.manifest.resources)


def _resource_selectors(
    value: Sequence[str],
    *,
    limits: McpSdkV2SubscriptionLimits,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValidationError("MCP resource subscription resolver returned an invalid value")
    selected = tuple(value)
    if not selected:
        raise ValidationError(
            "MCP resourceSubscriptions requires a registered Resource selector"
        )
    if len(selected) > limits.max_resource_subscriptions:
        raise ValidationError("MCP resource subscription selector limit exceeded")
    if len(set(selected)) != len(selected):
        raise ValidationError("MCP resource subscription selectors must be unique")
    for selector in selected:
        if (
            type(selector) is not str
            or not selector
            or "\x00" in selector
            or len(selector.encode("utf-8")) > limits.max_resource_selector_bytes
        ):
            raise ValidationError("MCP resource subscription selector is invalid")
        reject_mcp_app_selector(
            selector,
            label="subscription resource selector",
        )
    return selected


def _task_targets(
    value: Sequence[str],
    *,
    limits: McpSdkV2SubscriptionLimits,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValidationError("MCP Tasks subscription resolver returned an invalid value")
    selected = tuple(value)
    if not selected:
        raise ValidationError("MCP taskIds subscription has no current Task targets")
    if len(selected) > limits.max_task_subscriptions:
        raise ValidationError("MCP Tasks subscription target limit exceeded")
    if len(set(selected)) != len(selected):
        raise ValidationError("MCP Tasks subscription targets must be unique")
    for remote_id in selected:
        if (
            type(remote_id) is not str
            or not remote_id
            or "\x00" in remote_id
            or len(remote_id.encode("utf-8")) > limits.max_task_id_bytes
        ):
            raise ValidationError("MCP Tasks subscription target is invalid")
    return selected


def _new_task_ingress(
    task_targets: tuple[str, ...],
    *,
    limits: McpSdkV2SubscriptionLimits,
) -> _TaskNotificationIngress | None:
    if not task_targets:
        return None
    return _TaskNotificationIngress(
        request_id=f"agent-libos-task-listen-{secrets.token_urlsafe(18)}",
        requested_ids=frozenset(task_targets),
        queue=asyncio.Queue(maxsize=limits.queue_events),
    )


def _task_acknowledgement(
    honored: Any,
    *,
    requested: frozenset[str],
) -> tuple[str, ...]:
    dumped = honored.model_dump(by_alias=True, mode="json", exclude_none=True)
    raw = dumped.get(_TASK_FILTER)
    if (
        not isinstance(raw, list)
        or not raw
        or any(type(item) is not str or not item for item in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValidationError("MCP Tasks acknowledgement is invalid")
    selected = tuple(raw)
    if not set(selected).issubset(requested):
        raise ValidationError("MCP Tasks acknowledgement exceeded its request")
    return selected


def _sensitive_values(values: tuple[str, ...]) -> tuple[str, ...]:
    if type(values) is not tuple or any(
        type(item) is not str or not item for item in values
    ):
        raise ValidationError("MCP subscription sensitive-value snapshot is invalid")
    return tuple(dict.fromkeys(values))


def _safe_loss(
    error: Exception,
    *,
    sensitive_values: tuple[str, ...],
    default_reason: McpSdkSubscriptionLossReason = "open_failed",
) -> _Terminal:
    if isinstance(error, _AdapterLoss):
        reason = error.reason
    elif type(error).__name__ == "SubscriptionLost" or isinstance(
        error, (ConnectionError, EOFError)
    ):
        reason = "disconnected"
    else:
        reason = default_reason
    # Provider exception text may contain undeclared credentials, URLs, local
    # paths, or raw wire bytes.  Exact-secret redaction still runs on event
    # payloads, but diagnostics expose only this Host-owned reason allowlist.
    del error, sensitive_values
    messages = {
        "backpressure": "MCP subscription exceeded its bounded ingress capacity",
        "disconnected": "MCP subscription connection was lost",
        "invalid_event": "MCP subscription received an invalid event",
        "open_failed": "MCP subscription could not be opened",
    }
    return _Terminal("lost", messages[reason], reason)


def _raise_terminal(terminal: _Terminal | None) -> None:
    if terminal is None:
        raise McpSdkSubscriptionLost(
            "MCP subscription ended without terminal state",
            reason="disconnected",
        )
    if terminal.kind == "closed":
        raise McpSdkSubscriptionClosed(terminal.message)
    raise McpSdkSubscriptionLost(
        terminal.message,
        reason=terminal.reason or "disconnected",
    )


def _require_exact_server(server: McpServerSpec) -> None:
    if not isinstance(server, McpServerSpec):
        raise ValidationError("MCP SDK subscription server is invalid")
    if server.protocol_mode is not McpProtocolMode.REVISION_2026_07_28:
        raise ValidationError(
            "MCP SDK subscriptions require exact protocol 2026-07-28"
        )
    if type(server.max_response_bytes) is not int or server.max_response_bytes <= 0:
        raise ValidationError("MCP SDK subscription response bound is invalid")


def _validate_binding(binding: McpClientBinding, server: McpServerSpec) -> None:
    if not isinstance(binding, McpClientBinding):
        raise ValidationError("MCP SDK subscription binding is invalid")
    if binding.manifest.server_id != server.server_id:
        raise ValidationError("MCP subscription binding belongs to another server")
    if (
        binding.manifest.protocol_mode
        is not McpProtocolMode.REVISION_2026_07_28
    ):
        raise ValidationError("MCP subscription binding is not exact modern protocol")
    if binding.owner_id is None:
        raise ValidationError("MCP subscription binding requires an operation owner")


def _validate_deadline(value: float, now: float) -> float:
    if (
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= now
    ):
        raise TimeoutError("MCP SDK subscription deadline expired")
    return float(value)


def _remaining(deadline: float, now: float) -> float:
    remaining = deadline - now
    if remaining <= 0:
        raise TimeoutError("MCP SDK subscription deadline expired")
    return remaining


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "McpSdkSubscriptionClosed",
    "McpSdkSubscriptionLossReason",
    "McpSdkSubscriptionLost",
    "McpSdkV2SubscriptionLimits",
    "McpSdkV2SubscriptionProvider",
    "McpSdkV2SubscriptionSessionFactory",
]
