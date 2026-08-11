from __future__ import annotations

import asyncio
import contextlib
import gc
import time
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.mcp.client import McpClientBinding
from agent_libos.mcp.manifest import (
    MCP_TASKS_EXTENSION_ID,
    McpResourceSpec,
    McpServerManifestV3,
)
from agent_libos.mcp.resources import inert_resource_handle
from agent_libos.mcp.runtime_bridge import mcp_connection_fence
from agent_libos.mcp.sdk_subscriptions import (
    McpSdkSubscriptionClosed,
    McpSdkSubscriptionLost,
    McpSdkV2SubscriptionLimits,
    McpSdkV2SubscriptionProvider,
)
from agent_libos.mcp.subscriptions import (
    McpSubscriptionManager,
    McpTasksSubscriptionFence,
)
from agent_libos.mcp.supervisor import McpConnectionFence, McpConnectionSupervisor
from agent_libos.mcp.types import McpSubscriptionEvent, McpSubscriptionStatus
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.mcp import (
    McpProtocolMode,
    McpServerSpec,
    McpStdioTransportSpec,
)
from agent_libos.substrate.base import ProviderEffectNotStarted


pytestmark = [pytest.mark.mcp, pytest.mark.mcp_transport]

_SECRET = "opaque-subscription-credential"


def _server() -> McpServerSpec:
    return McpServerSpec(
        schema_version=2,
        server_id="modern",
        transport="stdio",
        stdio=McpStdioTransportSpec(command="modern-server"),
        tools=[],
        timeout_s=1.0,
        max_request_bytes=4096,
        max_response_bytes=64 * 1024,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
    )


def _binding() -> McpClientBinding:
    return McpClientBinding(
        manifest=McpServerManifestV3(
            schema_version=3,
            server_id="modern",
            transport="stdio",
            stdio=McpStdioTransportSpec(command="modern-server"),
            timeout_s=1.0,
            max_request_bytes=4096,
            max_response_bytes=64 * 1024,
            protocol_mode=McpProtocolMode.REVISION_2026_07_28,
            resources=(
                McpResourceSpec(
                    resource_id="current",
                    remote_uri="fixture://document/current",
                ),
            ),
        ),
        registry_generation=1,
        owner_id="pid:1",
    )


class _FakeSubscription:
    def __init__(self, events: list[Any], honored: Any) -> None:
        self.events = events
        self.honored = honored
        self.block = asyncio.Event()

    def __aiter__(self) -> _FakeSubscription:
        return self

    async def __anext__(self) -> Any:
        if not self.events:
            await self.block.wait()
            raise StopAsyncIteration
        selected = self.events.pop(0)
        if selected is StopAsyncIteration:
            raise StopAsyncIteration
        if isinstance(selected, BaseException):
            raise selected
        return selected


class _TrackedContext:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.enter_count = 0
        self.exit_count = 0
        self.owner: asyncio.Task[Any] | None = None

    async def __aenter__(self) -> Any:
        self.enter_count += 1
        self.owner = asyncio.current_task()
        return self.value

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        assert asyncio.current_task() is self.owner
        self.exit_count += 1


class _FakeClient:
    protocol_version = "2026-07-28"

    def __init__(self, subscription: _FakeSubscription) -> None:
        self.subscription = subscription
        self.listen_count = 0
        self.listen_arguments: dict[str, Any] | None = None
        self.listen_context = _TrackedContext(subscription)

    def listen(self, **keywords: Any) -> _TrackedContext:
        self.listen_count += 1
        self.listen_arguments = keywords
        return self.listen_context


class _Factory:
    def __init__(
        self,
        client: _FakeClient | None = None,
        *,
        enter_error: BaseException | None = None,
    ) -> None:
        self.client = client
        self.enter_error = enter_error
        self.call_count = 0
        self.context: _TrackedContext | None = None

    def __call__(self, server, *, deadline, binding):
        del server, deadline
        assert binding == _binding()
        self.call_count += 1
        if self.enter_error is not None:
            error = self.enter_error

            @contextlib.asynccontextmanager
            async def failing():
                raise error
                yield  # pragma: no cover

            return failing()
        assert self.client is not None
        self.context = _TrackedContext(self.client)
        return self.context


class _TaskDispatcher:
    def __init__(
        self,
        client: "_TaskClient",
        *,
        acknowledged_ids: tuple[str, ...],
        notification: dict[str, Any] | None,
    ) -> None:
        self.client = client
        self.acknowledged_ids = acknowledged_ids
        self.notification = notification

    async def send_raw_request(
        self,
        method: str,
        params: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        import anyio
        import mcp_types as types

        assert method == "subscriptions/listen"
        request_id = options["request_id"]
        self.client.request_params = params
        route = self.client.routes[request_id]
        route.set_acked(
            types.SubscriptionFilter.model_validate(
                {"taskIds": list(self.acknowledged_ids)}
            )
        )
        if self.notification is not None:
            selected = dict(self.notification)
            metadata = dict(selected.get("_meta", {}))
            metadata.setdefault(
                "io.modelcontextprotocol/subscriptionId",
                request_id,
            )
            selected["_meta"] = metadata
            self.client.task_notification_ingress(selected)
        await anyio.sleep_forever()


class _TaskClient:
    protocol_version = "2026-07-28"

    def __init__(
        self,
        *,
        task_notification_ingress: Any,
        acknowledged_ids: tuple[str, ...],
        notification: dict[str, Any] | None,
    ) -> None:
        self.task_notification_ingress = task_notification_ingress
        self.routes: dict[str, Any] = {}
        self.request_params: dict[str, Any] | None = None
        self._task_group: Any = None
        self._dispatcher = _TaskDispatcher(
            self,
            acknowledged_ids=acknowledged_ids,
            notification=notification,
        )

    def _stamp(self, _data: dict[str, Any], _options: dict[str, Any]) -> None:
        return None

    def _register_listen_route(self, request_id: str) -> Any:
        from mcp.client.subscriptions import ListenRoute

        route = ListenRoute()
        self.routes[request_id] = route
        return route

    def _unregister_listen_route(self, request_id: str) -> None:
        self.routes.pop(request_id, None)


class _TaskFactory:
    def __init__(
        self,
        *,
        acknowledged_ids: tuple[str, ...],
        notification: dict[str, Any] | None,
    ) -> None:
        self.acknowledged_ids = acknowledged_ids
        self.notification = notification
        self.client: _TaskClient | None = None

    @contextlib.asynccontextmanager
    async def __call__(
        self,
        _server: Any,
        *,
        deadline: float,
        binding: McpClientBinding,
        task_notification_ingress: Any = None,
    ) -> Any:
        import anyio

        del deadline
        assert binding == _binding()
        assert callable(task_notification_ingress)
        client = _TaskClient(
            task_notification_ingress=task_notification_ingress,
            acknowledged_ids=self.acknowledged_ids,
            notification=self.notification,
        )
        self.client = client
        async with anyio.create_task_group() as task_group:
            client._task_group = task_group
            try:
                yield client
            finally:
                task_group.cancel_scope.cancel()


def _honored(**values: Any) -> Any:
    defaults = {
        "tools_list_changed": None,
        "prompts_list_changed": None,
        "resources_list_changed": None,
        "resource_subscriptions": None,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


async def _wait_done(handle: Any) -> None:
    handle = getattr(handle, "handle", handle)
    for _ in range(100):
        if handle.done.is_set():
            return
        await asyncio.sleep(0)
    raise AssertionError("subscription driver did not stop")


def _task_notification(remote_id: str) -> dict[str, Any]:
    return {
        "taskId": remote_id,
        "status": "working",
        "statusMessage": "still working",
        "createdAt": "2026-08-11T00:00:00+00:00",
        "lastUpdatedAt": "2026-08-11T00:00:01+00:00",
        "ttlMs": 60_000,
        "pollIntervalMs": 250,
    }


class _TaskProjector:
    def __init__(self, remote_id: str) -> None:
        self.remote_id = remote_id
        self.seen: list[str] = []

    def project_task_notification(
        self,
        *,
        event: McpSubscriptionEvent,
        fence: McpConnectionFence,
        sensitive_values: tuple[str, ...],
    ) -> McpSubscriptionEvent:
        del fence, sensitive_values
        assert type(event.payload) is dict
        assert event.payload["taskId"] == self.remote_id
        self.seen.append(self.remote_id)
        return McpSubscriptionEvent(
            sequence=0,
            event_type="taskStatus",
            payload={"task_ref": "mcp-task-local", "status": "working"},
            received_at="provider-controlled-time",
        )


def test_sdk_subscription_receives_one_event_and_closes_owned_contexts_once() -> None:
    async def exercise() -> None:
        from mcp.client.subscriptions import ToolsListChanged

        subscription = _FakeSubscription(
            [ToolsListChanged()],
            _honored(tools_list_changed=True),
        )
        client = _FakeClient(subscription)
        factory = _Factory(client)
        provider = McpSdkV2SubscriptionProvider(
            factory,
            binding_resolver=_binding,
            utc_now=lambda: "2026-08-11T00:00:00+00:00",
        )
        handle = await provider.listen(
            _server(),
            ("toolsListChanged",),
            deadline=time.monotonic() + 1,
        )
        assert handle.acknowledged_filters == ("toolsListChanged",)
        event = await provider.receive(handle, deadline=time.monotonic() + 1)
        assert event.event_type == "toolsListChanged"
        assert event.payload == {}
        assert event.provenance == "untrusted_mcp_notification"
        assert event.sequence == 0

        await provider.close(handle)
        await provider.close(handle)
        await handle.handle.aclose()  # supervisor invalidation uses this same seam
        assert factory.call_count == 1
        assert client.listen_count == 1
        assert factory.context is not None
        assert factory.context.enter_count == factory.context.exit_count == 1
        assert client.listen_context.enter_count == client.listen_context.exit_count == 1

    asyncio.run(exercise())


def test_resource_subscription_uses_only_host_resolved_selectors_and_opaque_event() -> None:
    async def exercise() -> None:
        from mcp.client.subscriptions import ResourceUpdated

        selector = f"https://provider.invalid/private/{_SECRET}"
        subscription = _FakeSubscription(
            [ResourceUpdated(uri=selector)],
            _honored(resource_subscriptions=[selector]),
        )
        client = _FakeClient(subscription)
        provider = McpSdkV2SubscriptionProvider(
            _Factory(client),
            binding_resolver=_binding,
            sensitive_values_resolver=lambda _id: (_SECRET,),
            resource_subscriptions_resolver=lambda _server, _binding: (selector,),
        )
        handle = await provider.listen(
            _server(),
            ("resourceSubscriptions",),
            deadline=time.monotonic() + 1,
        )
        assert handle.acknowledged_filters == ("resourceSubscriptions",)
        assert client.listen_arguments == {
            "tools_list_changed": False,
            "prompts_list_changed": False,
            "resources_list_changed": False,
            "resource_subscriptions": (selector,),
        }
        event = await provider.receive(handle, deadline=time.monotonic() + 1)
        assert event.event_type == "resourceUpdated"
        assert event.payload == {
            "resource_handle": inert_resource_handle("modern", selector)
        }
        assert selector not in repr(event)
        assert _SECRET not in repr(event)
        await provider.close(handle)

    asyncio.run(exercise())


def test_disconnect_is_typed_and_exact_secret_is_redacted_without_relisten() -> None:
    async def exercise() -> None:
        subscription = _FakeSubscription(
            [ConnectionError(f"wire lost with {_SECRET}")],
            _honored(resources_list_changed=True),
        )
        client = _FakeClient(subscription)
        factory = _Factory(client)
        provider = McpSdkV2SubscriptionProvider(
            factory,
            binding_resolver=_binding,
            sensitive_values_resolver=lambda _id: (_SECRET,),
        )
        handle = await provider.listen(
            _server(),
            ("resourcesListChanged",),
            deadline=time.monotonic() + 1,
        )
        await _wait_done(handle)
        with pytest.raises(McpSdkSubscriptionLost) as raised:
            await provider.receive(handle, deadline=time.monotonic() + 1)
        assert raised.value.reason == "disconnected"
        assert _SECRET not in str(raised.value)
        assert str(raised.value) == "MCP subscription connection was lost"
        assert factory.call_count == 1
        assert client.listen_count == 1
        await provider.close(handle)

    asyncio.run(exercise())


def test_secret_acquired_during_governed_session_entry_is_redacted() -> None:
    async def exercise() -> None:
        dynamic_secret = "oauth-token-acquired-in-provider-phase"
        active = False
        subscription = _FakeSubscription(
            [ConnectionError(f"peer reflected {dynamic_secret}")],
            _honored(resources_list_changed=True),
        )
        client = _FakeClient(subscription)

        @contextlib.asynccontextmanager
        async def governed_context(server, *, deadline, binding):
            nonlocal active
            del server, deadline
            assert binding == _binding()
            active = True
            try:
                yield client
            finally:
                active = False

        provider = McpSdkV2SubscriptionProvider(
            governed_context,
            binding_resolver=_binding,
            sensitive_values_resolver=(
                lambda _id: (dynamic_secret,) if active else ()
            ),
        )
        handle = await provider.listen(
            _server(),
            ("resourcesListChanged",),
            deadline=time.monotonic() + 1,
        )
        await _wait_done(handle)
        with pytest.raises(McpSdkSubscriptionLost) as raised:
            await provider.receive(handle, deadline=time.monotonic() + 1)
        assert dynamic_secret not in str(raised.value)
        assert str(raised.value) == "MCP subscription connection was lost"
        await provider.close(handle)

    asyncio.run(exercise())


def test_queue_overflow_and_oversize_event_fail_closed() -> None:
    async def exercise() -> None:
        from mcp.client.subscriptions import ToolsListChanged

        for limits, events in (
            (
                McpSdkV2SubscriptionLimits(queue_events=1),
                [ToolsListChanged(), ToolsListChanged()],
            ),
            (
                McpSdkV2SubscriptionLimits(event_max_bytes=1),
                [ToolsListChanged()],
            ),
        ):
            subscription = _FakeSubscription(
                events,
                _honored(tools_list_changed=True),
            )
            provider = McpSdkV2SubscriptionProvider(
                _Factory(_FakeClient(subscription)),
                binding_resolver=_binding,
                limits=limits,
            )
            handle = await provider.listen(
                _server(),
                ("toolsListChanged",),
                deadline=time.monotonic() + 1,
            )
            await _wait_done(handle)
            if limits.queue_events == 1 and limits.event_max_bytes > 1:
                await provider.receive(handle, deadline=time.monotonic() + 1)
            with pytest.raises(McpSdkSubscriptionLost) as raised:
                await provider.receive(handle, deadline=time.monotonic() + 1)
            assert raised.value.reason == "backpressure"
            await provider.close(handle)

    asyncio.run(exercise())


def test_graceful_server_close_has_distinct_typed_end() -> None:
    async def exercise() -> None:
        subscription = _FakeSubscription(
            [StopAsyncIteration],
            _honored(prompts_list_changed=True),
        )
        provider = McpSdkV2SubscriptionProvider(
            _Factory(_FakeClient(subscription)),
            binding_resolver=_binding,
        )
        handle = await provider.listen(
            _server(),
            ("promptsListChanged",),
            deadline=time.monotonic() + 1,
        )
        # The exact session handoff guarantees a Manager can synchronously
        # validate the owner task even when the server acknowledges and closes
        # before ``listen`` returns to its caller.
        assert not handle.owner_task.done()
        await _wait_done(handle)
        with pytest.raises(McpSdkSubscriptionClosed):
            await provider.receive(handle, deadline=time.monotonic() + 1)
        await provider.close(handle)

    asyncio.run(exercise())


def test_pre_dispatch_certificate_is_preserved_and_task_filter_fails_locally() -> None:
    async def exercise() -> None:
        certificate = ProviderEffectNotStarted("not started")
        provider = McpSdkV2SubscriptionProvider(
            _Factory(enter_error=certificate),
            binding_resolver=_binding,
        )
        with pytest.raises(ProviderEffectNotStarted) as raised:
            await provider.listen(
                _server(),
                ("toolsListChanged",),
                deadline=time.monotonic() + 1,
            )
        assert raised.value is certificate

        unused = _Factory(_FakeClient(_FakeSubscription([], _honored())))
        provider = McpSdkV2SubscriptionProvider(
            unused,
            binding_resolver=_binding,
        )
        with pytest.raises(ValidationError, match="taskIds"):
            await provider.listen(
                _server(),
                ("taskIds",),
                deadline=time.monotonic() + 1,
            )
        assert unused.call_count == 0

        apps = _Factory(_FakeClient(_FakeSubscription([], _honored())))
        provider = McpSdkV2SubscriptionProvider(
            apps,
            binding_resolver=_binding,
            resource_subscriptions_resolver=(
                lambda _server, _binding: ("ui://remote-app",)
            ),
        )
        with pytest.raises(ValidationError, match="Apps"):
            await provider.listen(
                _server(),
                ("resourceSubscriptions",),
                deadline=time.monotonic() + 1,
            )
        assert apps.call_count == 0

    asyncio.run(exercise())


def test_task_ids_raw_wire_projects_only_local_ref_through_host_manager() -> None:
    async def exercise() -> None:
        remote_id = "remote-task-bearer-must-not-escape"
        factory = _TaskFactory(
            acknowledged_ids=(remote_id,),
            notification=_task_notification(remote_id),
        )
        provider = McpSdkV2SubscriptionProvider(
            factory,
            binding_resolver=_binding,
            task_subscriptions_resolver=lambda *, fence: (remote_id,),
        )
        projector = _TaskProjector(remote_id)
        supervisor = McpConnectionSupervisor()
        manager = McpSubscriptionManager(
            supervisor,
            task_event_projector=projector,
        )
        public = await manager.start(
            _server(),
            mcp_connection_fence(_binding()),
            provider,
            ("taskIds",),
            tasks_extension_fence=McpTasksSubscriptionFence(
                extension_id=MCP_TASKS_EXTENSION_ID,
                manifest_spec_sha256="d" * 64,
                host_spec_sha256="d" * 64,
            ),
        )
        assert public.acknowledged_filters == ("taskIds",)
        events: tuple[McpSubscriptionEvent, ...] = ()
        for _ in range(100):
            events = await manager.events(public.subscription_id)
            if events:
                break
            await asyncio.sleep(0)
        assert len(events) == 1
        assert events[0].payload == {
            "task_ref": "mcp-task-local",
            "status": "working",
        }
        assert remote_id not in repr(events[0])
        assert projector.seen == [remote_id]
        assert factory.client is not None
        assert remote_id in repr(factory.client.request_params)
        await manager.stop(public.subscription_id)
        await manager.close()
        await supervisor.close()

    asyncio.run(exercise())


def test_task_ids_rejects_acknowledgement_outside_host_targets() -> None:
    async def exercise() -> None:
        remote_id = "requested-remote-task-bearer"
        attacker_id = "unrequested-remote-task-bearer"
        provider = McpSdkV2SubscriptionProvider(
            _TaskFactory(
                acknowledged_ids=(attacker_id,),
                notification=None,
            ),
            binding_resolver=_binding,
            task_subscriptions_resolver=lambda *, fence: (remote_id,),
        )
        with pytest.raises(McpSdkSubscriptionLost) as raised:
            await provider.listen(
                _server(),
                ("taskIds",),
                deadline=time.monotonic() + 1,
            )
        assert raised.value.reason == "open_failed"
        assert remote_id not in str(raised.value)
        assert attacker_id not in str(raised.value)

    asyncio.run(exercise())


def test_task_ids_cross_stream_notification_fails_without_bearer_reflection() -> None:
    async def exercise() -> None:
        remote_id = "cross-stream-remote-task-bearer"
        notification = _task_notification(remote_id)
        notification["_meta"] = {
            "io.modelcontextprotocol/subscriptionId": "another-stream"
        }
        provider = McpSdkV2SubscriptionProvider(
            _TaskFactory(
                acknowledged_ids=(remote_id,),
                notification=notification,
            ),
            binding_resolver=_binding,
            task_subscriptions_resolver=lambda *, fence: (remote_id,),
        )
        handle = await provider.listen(
            _server(),
            ("taskIds",),
            deadline=time.monotonic() + 1,
        )
        await _wait_done(handle)
        with pytest.raises(McpSdkSubscriptionLost) as raised:
            await provider.receive(handle, deadline=time.monotonic() + 1)
        assert raised.value.reason == "invalid_event"
        assert remote_id not in str(raised.value)
        await provider.close(handle)

    asyncio.run(exercise())


def test_receive_deadline_and_foreign_handle_fail_without_closing_stream() -> None:
    async def exercise() -> None:
        subscription = _FakeSubscription(
            [],
            _honored(resources_list_changed=True),
        )
        first = McpSdkV2SubscriptionProvider(
            _Factory(_FakeClient(subscription)),
            binding_resolver=_binding,
        )
        second = McpSdkV2SubscriptionProvider(
            _Factory(_FakeClient(_FakeSubscription([], _honored()))),
            binding_resolver=_binding,
        )
        handle = await first.listen(
            _server(),
            ("resourcesListChanged",),
            deadline=time.monotonic() + 1,
        )
        with pytest.raises(ValidationError, match="handle"):
            await second.receive(handle, deadline=time.monotonic() + 1)
        with pytest.raises(TimeoutError):
            await first.receive(handle, deadline=time.monotonic() + 0.001)
        assert not handle.handle.done.is_set()
        await first.close(handle)

    asyncio.run(exercise())


def test_listen_deadline_cancels_owner_and_exits_raw_context() -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        session_exits = 0

        class SlowClient:
            protocol_version = "2026-07-28"

            @contextlib.asynccontextmanager
            async def listen(self, **keywords):
                del keywords
                entered.set()
                await asyncio.Future()
                yield  # pragma: no cover

        @contextlib.asynccontextmanager
        async def governed_context(server, *, deadline, binding):
            nonlocal session_exits
            del server, deadline
            assert binding == _binding()
            try:
                yield SlowClient()
            finally:
                session_exits += 1

        provider = McpSdkV2SubscriptionProvider(
            governed_context,
            binding_resolver=_binding,
        )
        with pytest.raises(TimeoutError):
            await provider.listen(
                _server(),
                ("toolsListChanged",),
                deadline=time.monotonic() + 0.01,
            )
        assert entered.is_set()
        assert session_exits == 1

    asyncio.run(exercise())


def test_cancelled_listen_consumes_concurrent_owner_open_failure() -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        loop_failures: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(
            lambda _loop, context: loop_failures.append(dict(context))
        )

        @contextlib.asynccontextmanager
        async def governed_context(server, *, deadline, binding):
            del server, deadline
            assert binding == _binding()
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                # Model an owner that reports an opening deadline in the same
                # slice in which its caller is cancelled.
                raise TimeoutError("owner deadline") from None
            yield  # pragma: no cover

        provider = McpSdkV2SubscriptionProvider(
            governed_context,
            binding_resolver=_binding,
        )
        opening = asyncio.create_task(
            provider.listen(
                _server(),
                ("toolsListChanged",),
                deadline=time.monotonic() + 1,
            )
        )
        await entered.wait()
        opening.cancel()
        with pytest.raises(asyncio.CancelledError):
            await opening
        gc.collect()
        await asyncio.sleep(0)
        assert loop_failures == []

    asyncio.run(exercise())


def test_real_python_sdk_v2_listen_event_and_close() -> None:
    async def exercise() -> None:
        from mcp.client import Client
        from mcp.server.mcpserver import MCPServer
        from mcp.server.subscriptions import (
            InMemorySubscriptionBus,
            ResourceUpdated,
        )

        selector = "fixture://document/current"
        bus = InMemorySubscriptionBus()
        sdk_server = MCPServer(
            "agent-libos-subscription-adapter-fixture",
            subscriptions=bus,
        )
        context_exits = 0

        @contextlib.asynccontextmanager
        async def governed_context(server, *, deadline, binding):
            nonlocal context_exits
            del server, deadline
            assert binding == _binding()
            try:
                async with Client(
                    sdk_server,
                    mode="2026-07-28",
                    raise_exceptions=True,
                ) as client:
                    yield client
            finally:
                context_exits += 1

        provider = McpSdkV2SubscriptionProvider(
            governed_context,
            binding_resolver=_binding,
            resource_subscriptions_resolver=lambda _server, _binding: (selector,),
        )
        handle = await provider.listen(
            _server(),
            ("resourceSubscriptions",),
            deadline=time.monotonic() + 2,
        )
        assert handle.acknowledged_filters == ("resourceSubscriptions",)
        await bus.publish(ResourceUpdated(uri=selector))
        event = await provider.receive(handle, deadline=time.monotonic() + 2)
        assert event.event_type == "resourceUpdated"
        assert event.payload == {
            "resource_handle": inert_resource_handle("modern", selector)
        }
        await provider.close(handle)
        assert context_exits == 1

    asyncio.run(exercise())


def test_host_manager_owns_one_outer_supervisor_lease_and_closes_adapter_once() -> None:
    async def exercise() -> None:
        from mcp.client.subscriptions import ToolsListChanged

        subscription = _FakeSubscription(
            [ToolsListChanged()],
            _honored(tools_list_changed=True),
        )
        client = _FakeClient(subscription)
        factory = _Factory(client)
        provider = McpSdkV2SubscriptionProvider(
            factory,
            binding_resolver=_binding,
        )
        supervisor = McpConnectionSupervisor()
        manager = McpSubscriptionManager(supervisor)
        binding = _binding()
        record = await manager.start(
            _server(),
            McpConnectionFence(
                server_id="modern",
                server_spec_sha256=binding.manifest_sha256,
                registry_generation=binding.registry_generation,
                owner=binding.owner_id or "",
            ),
            provider,
            ("toolsListChanged",),
        )
        events = ()
        for _ in range(100):
            events = await manager.events(record.subscription_id, after=0)
            if events:
                break
            await asyncio.sleep(0)
        assert len(events) == 1
        assert events[0].event_type == "toolsListChanged"

        closed = await manager.stop(record.subscription_id)
        assert closed.status is McpSubscriptionStatus.CLOSED
        assert factory.call_count == 1
        assert client.listen_count == 1
        assert factory.context is not None
        assert factory.context.exit_count == 1
        assert client.listen_context.exit_count == 1
        for _ in range(200):
            if await supervisor.snapshot() == ():
                break
            await asyncio.sleep(0)
        assert await supervisor.snapshot() == ()
        await manager.close()
        await supervisor.close()

    asyncio.run(exercise())


def test_manager_retains_event_and_terminal_handle_when_server_closes_gracefully() -> None:
    async def exercise() -> None:
        from mcp.client.subscriptions import ToolsListChanged

        subscription = _FakeSubscription(
            [ToolsListChanged()],
            _honored(tools_list_changed=True),
        )
        client = _FakeClient(subscription)
        factory = _Factory(client)
        provider = McpSdkV2SubscriptionProvider(
            factory,
            binding_resolver=_binding,
        )
        supervisor = McpConnectionSupervisor()
        manager = McpSubscriptionManager(supervisor)
        binding = _binding()
        public = await manager.start(
            _server(),
            McpConnectionFence(
                server_id="modern",
                server_spec_sha256=binding.manifest_sha256,
                registry_generation=binding.registry_generation,
                owner=binding.owner_id or "",
            ),
            provider,
            ("toolsListChanged",),
        )
        subscription.block.set()

        terminal = None
        events = ()
        cursor = 0
        for _ in range(200):
            # All three methods must retain a stable local handle throughout
            # the Provider-done/loss-callback race; a transient KeyError would
            # make graceful wire closure indistinguishable from an unknown id.
            terminal = await manager.status(public.subscription_id)
            batch = await manager.events(
                public.subscription_id,
                after=cursor,
            )
            if batch:
                events += batch
                cursor = batch[-1].sequence
            if events and terminal.status is not McpSubscriptionStatus.ACTIVE:
                break
            await asyncio.sleep(0)
        assert terminal is not None
        assert terminal.status is McpSubscriptionStatus.LOST
        assert len(events) == 1
        assert events[0].event_type == "toolsListChanged"
        stopped = await manager.stop(public.subscription_id)
        assert stopped == terminal
        assert await manager.events(
            public.subscription_id,
            after=cursor,
        ) == ()
        assert factory.context is not None
        assert factory.context.exit_count == 1
        assert client.listen_context.exit_count == 1
        for _ in range(200):
            if await supervisor.snapshot() == ():
                break
            await asyncio.sleep(0)
        assert await supervisor.snapshot() == ()

    asyncio.run(exercise())
