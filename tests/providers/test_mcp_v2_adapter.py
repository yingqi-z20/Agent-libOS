from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    McpExchangePhase,
    McpExchangeReceipt,
    McpHttpTransportSpec,
    McpProtocolEra,
    McpProtocolMode,
    McpServerSpec,
    McpToolSpec,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.primitives.mcp import McpPrimitive
from agent_libos.substrate.base import CommandMetrics, SubprocessLimits, SubprocessTimeoutExpired
from agent_libos.substrate.local import (
    SdkMcpProvider,
    _McpAbsoluteDeadlineExceeded,
    _mcp_negotiate_sdk_v2_session,
    _mcp_sdk_v2_client,
)
from agent_libos.substrate.local import (
    _McpWireLedger,
    _McpPolicyAsyncHTTPTransport,
    _mcp_await_with_deadline,
    _mcp_legacy_wire_bytes,
    _strict_stdio_client,
)

pytestmark = pytest.mark.mcp


class _RawWireTransport:
    """Small in-memory peer that records exactly what ClientSession writes."""

    def __init__(
        self,
        *,
        supported_versions: list[str] | None = None,
        listed_tool: dict[str, Any] | None = None,
        call_result: dict[str, Any] | None = None,
    ) -> None:
        self.outbound: list[Any] = []
        self.outbound_metadata: list[Any] = []
        self.supported_versions = supported_versions or ["2026-07-28"]
        self.listed_tool = listed_tool or {
            "name": "demo.echo",
            "inputSchema": {"type": "object"},
        }
        self.call_result = call_result or {
            "content": [],
            "structuredContent": {"ok": True},
            "isError": False,
        }
        self._next_request_id = 10_000
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._drive_task: asyncio.Task[None] | None = None
        self._modern = False

    async def __aenter__(self) -> tuple[Any, Any]:
        import anyio

        self._server_send, client_read = anyio.create_memory_object_stream(128)
        client_write, self._server_receive = anyio.create_memory_object_stream(128)
        self._drive_task = asyncio.create_task(self._drive())
        return client_read, client_write

    async def __aexit__(self, *_exc: object) -> None:
        await self._server_send.aclose()
        if self._drive_task is not None:
            self._drive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drive_task

    async def _drive(self) -> None:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCError, JSONRPCRequest, JSONRPCResponse

        async with self._server_receive:
            async for item in self._server_receive:
                message = item.message
                self.outbound.append(message)
                self.outbound_metadata.append(item.metadata)
                if isinstance(message, JSONRPCRequest):
                    if message.method == "server/discover":
                        self._modern = True
                        result = {
                            "supportedVersions": self.supported_versions,
                            "capabilities": {"tools": {}},
                            "resultType": "complete",
                            "ttlMs": 0,
                            "cacheScope": "private",
                        }
                    elif message.method == "initialize":
                        result = {
                            "protocolVersion": "2025-11-25",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "raw-wire", "version": "1"},
                        }
                    elif message.method == "tools/list":
                        result = {"tools": [self.listed_tool]}
                        if self._modern:
                            result.update(
                                {
                                    "resultType": "complete",
                                    "ttlMs": 0,
                                    "cacheScope": "private",
                                }
                            )
                    elif message.method == "tools/call":
                        result = dict(self.call_result)
                        if self._modern:
                            result.setdefault("resultType", "complete")
                    else:
                        continue
                    await self._server_send.send(
                        SessionMessage(
                            JSONRPCResponse(
                                jsonrpc="2.0",
                                id=message.id,
                                result=result,
                            )
                        )
                    )
                elif isinstance(message, (JSONRPCResponse, JSONRPCError)):
                    pending = self._pending.pop(int(message.id), None)
                    if pending is not None and not pending.done():
                        pending.set_result(message)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCRequest

        request_id = self._next_request_id
        self._next_request_id += 1
        pending = asyncio.get_running_loop().create_future()
        self._pending[request_id] = pending
        await self._server_send.send(
            SessionMessage(
                JSONRPCRequest(
                    jsonrpc="2.0",
                    id=request_id,
                    method=method,
                    params=params,
                )
            )
        )
        return await asyncio.wait_for(pending, timeout=1)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCNotification

        await self._server_send.send(
            SessionMessage(
                JSONRPCNotification(
                    jsonrpc="2.0",
                    method=method,
                    params=params,
                )
            )
        )


class _Transport:
    async def __aenter__(self) -> tuple[object, object]:
        return object(), object()

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _McpError(Exception):
    def __init__(self, code: int, data: Any = None) -> None:
        super().__init__(f"MCP error {code}")
        self.code = code
        self.error = SimpleNamespace(code=code, data=data)


class _NegotiationSession:
    discover_error: Exception | None = None
    initialize_calls = 0

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        from mcp.types import Implementation

        self.protocol_version: str | None = None
        self.server_info = SimpleNamespace(name="fixture", version="1")
        self.server_capabilities = SimpleNamespace(tools={})
        self._client_info = Implementation(name="fixture", version="1")

    async def __aenter__(self) -> "_NegotiationSession":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def send_discover(self, _version: str) -> dict[str, Any]:
        if type(self).discover_error is not None:
            raise type(self).discover_error
        return {
            "supportedVersions": ["2026-07-28"],
            "capabilities": {"tools": {}},
            "resultType": "complete",
            "ttlMs": 0,
            "cacheScope": "public",
        }

    def _build_capabilities(self, _version: str) -> Any:
        from mcp.types import ClientCapabilities

        return ClientCapabilities()

    def adopt(self, result: Any) -> None:
        self.protocol_version = (
            "2026-07-28"
            if hasattr(result, "supported_versions")
            else str(result.protocol_version)
        )

    async def send_request(self, _request: Any, _result_type: Any) -> Any:
        from mcp.types import Implementation, InitializeResult, ServerCapabilities

        type(self).initialize_calls += 1
        return InitializeResult(
            protocol_version="2025-11-25",
            capabilities=ServerCapabilities(tools={}),
            server_info=Implementation(name="fixture", version="1"),
        )

    async def send_notification(self, _notification: Any) -> None:
        return None

    async def initialize(self) -> None:
        raise AssertionError("the adapter must use its release-locked initialize path")


def _http_server(mode: McpProtocolMode = McpProtocolMode.AUTO) -> McpServerSpec:
    return McpServerSpec(
        schema_version=2,
        server_id="v2-http",
        transport="streamable_http",
        tools=[],
        timeout_s=1,
        max_request_bytes=65_536,
        max_response_bytes=1_048_576,
        http=McpHttpTransportSpec(url="https://mcp.example.test/mcp"),
        protocol_mode=mode,
    )


def _stdio_server(mode: McpProtocolMode = McpProtocolMode.AUTO) -> McpServerSpec:
    server = _http_server(mode)
    return McpServerSpec(
        **{
            **server.__dict__,
            "server_id": "v2-stdio",
            "transport": "stdio",
            "http": None,
        }
    )


@pytest.mark.parametrize(
    ("operation_deadline", "probe_timeout_s", "expected_probe_deadline"),
    (
        (110.0, 5.0, 105.0),
        (103.0, 5.0, 103.0),
        (110.0, 0.125, 100.125),
    ),
)
def test_auto_probe_uses_release_cap_and_shorter_operation_deadline(
    monkeypatch: pytest.MonkeyPatch,
    operation_deadline: float,
    probe_timeout_s: float,
    expected_probe_deadline: float,
) -> None:
    import agent_libos.substrate.local as local_substrate

    captured: list[tuple[float, str]] = []

    class ProbeSession:
        async def send_discover(self, _version: str) -> dict[str, Any]:
            raise AssertionError("capturing await helper must stop before dispatch")

    async def capture_deadline(
        awaitable: Any,
        *,
        deadline: float,
        stage: str,
    ) -> Any:
        awaitable.close()
        captured.append((deadline, stage))
        raise _McpAbsoluteDeadlineExceeded(stage)

    monkeypatch.setattr(
        local_substrate,
        "_mcp_await_with_deadline",
        capture_deadline,
    )

    async def exercise() -> None:
        with pytest.raises(_McpAbsoluteDeadlineExceeded):
            await _mcp_negotiate_sdk_v2_session(
                ProbeSession(),
                server=_http_server(),
                mode=McpProtocolMode.AUTO,
                deadline=operation_deadline,
                negotiation_started=100.0,
                http_policy_transport=None,
                mcp_types=SimpleNamespace(),
                protocol_probe_timeout_s=probe_timeout_s,
            )

    asyncio.run(exercise())
    assert captured == [(expected_probe_deadline, "server/discover probe")]
    assert probe_timeout_s <= DEFAULT_CONFIG.mcp.protocol_probe_timeout_s


def test_sdk_provider_classifies_discover_as_protected_external_read() -> None:
    provider = SdkMcpProvider()

    classification = provider.classify_external_effect(
        "discover",
        {"server_id": "v2-http", "transport": "streamable_http"},
        {},
    )

    assert classification.rollback_class.value == "no_rollback_required"
    assert classification.rollback_status.value == "not_required"
    assert not classification.state_mutation
    assert classification.information_flow
    assert classification.metadata["operation"] == "discover"


@pytest.mark.parametrize("status", [401, 403, 500])
def test_auto_http_does_not_fallback_on_auth_or_server_error(status: int) -> None:
    _NegotiationSession.discover_error = _McpError(-32603)
    _NegotiationSession.initialize_calls = 0
    policy = SimpleNamespace(
        last_response_status=status,
        last_request_method="server/discover",
    )

    async def exercise() -> None:
        import time

        async with _mcp_sdk_v2_client(
            _NegotiationSession,
            _Transport(),
            server=_http_server(),
            mode=McpProtocolMode.AUTO,
            sdk_mode="auto",
            deadline=time.monotonic() + 10,
            max_response_bytes=1_048_576,
            http_policy_transport=policy,
        ):
            raise AssertionError("ambiguous HTTP failures must not connect")

    import asyncio

    with pytest.raises(_McpError):
        asyncio.run(exercise())
    assert _NegotiationSession.initialize_calls == 0


def test_auto_http_falls_back_only_on_legacy_400_signal() -> None:
    _NegotiationSession.discover_error = _McpError(-32603)
    _NegotiationSession.initialize_calls = 0
    policy = SimpleNamespace(
        last_response_status=400,
        last_request_method="server/discover",
    )

    async def exercise() -> Any:
        import time

        async with _mcp_sdk_v2_client(
            _NegotiationSession,
            _Transport(),
            server=_http_server(),
            mode=McpProtocolMode.AUTO,
            sdk_mode="auto",
            deadline=time.monotonic() + 1,
            max_response_bytes=1_048_576,
            http_policy_transport=policy,
        ) as client:
            return client._agent_libos_connection, tuple(client._agent_libos_receipts)

    import asyncio

    connection, receipts = asyncio.run(exercise())
    assert connection.protocol_era is McpProtocolEra.LEGACY
    assert connection.fallback_used
    assert [item.phase for item in receipts] == [
        McpExchangePhase.SERVER_DISCOVER,
        McpExchangePhase.INITIALIZE,
    ]
    assert _NegotiationSession.initialize_calls == 1


def test_builtin_v2_http_auth_failure_returns_wire_certified_pre_call_result() -> None:
    class Handler(BaseHTTPRequestHandler):
        methods: list[str] = []

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            type(self).methods.append(str(body.get("method")))
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        tool = McpToolSpec(
            **{
                **_tool().__dict__,
                "right": "write",
                "rollback_class": "unknown",
                "state_mutation": True,
            }
        )
        server = McpServerSpec(
            **{
                **_http_server().__dict__,
                "server_id": "auth-pre-call",
                "tools": [tool],
                "http": McpHttpTransportSpec(
                    url=f"http://127.0.0.1:{httpd.server_port}/mcp"
                ),
            }
        )

        result = SdkMcpProvider().validate_and_call(
            server,
            tool,
            {},
            timeout_s=1,
            max_response_bytes=server.max_response_bytes,
        )

        assert Handler.methods == ["server/discover"]
        assert result.error_type == "McpPreCallFailure"
        assert result.connection is None
        assert not result.call_started
        assert result.call_request_bytes == 0
        assert result.call_response_bytes == 0
        assert result.response_bytes == 0
        assert [item.phase for item in result.receipts] == [
            McpExchangePhase.SERVER_DISCOVER,
        ]
        assert all(
            item.phase is not McpExchangePhase.TOOLS_CALL
            for item in result.receipts
        )
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def test_modern_pin_rejects_legacy_method_not_found_without_fallback() -> None:
    _NegotiationSession.discover_error = _McpError(-32601)
    _NegotiationSession.initialize_calls = 0
    server = _http_server(McpProtocolMode.REVISION_2026_07_28)
    server = McpServerSpec(
        **{
            **server.__dict__,
            "transport": "stdio",
            "http": None,
        }
    )

    async def exercise() -> None:
        import time

        async with _mcp_sdk_v2_client(
            _NegotiationSession,
            _Transport(),
            server=server,
            mode=McpProtocolMode.REVISION_2026_07_28,
            sdk_mode="auto",
            deadline=time.monotonic() + 1,
            max_response_bytes=1_048_576,
        ):
            raise AssertionError("modern pin must not fall back")

    import asyncio

    with pytest.raises(_McpError):
        asyncio.run(exercise())
    assert _NegotiationSession.initialize_calls == 0


def test_stdio_auto_32022_falls_back_only_for_release_locked_legacy_versions() -> None:
    _NegotiationSession.discover_error = _McpError(
        -32022,
        {"supported": ["2025-11-25", "2025-06-18"]},
    )
    _NegotiationSession.initialize_calls = 0

    async def exercise() -> Any:
        async with _mcp_sdk_v2_client(
            _NegotiationSession,
            _Transport(),
            server=_stdio_server(),
            mode=McpProtocolMode.AUTO,
            sdk_mode="auto",
            deadline=time.monotonic() + 1,
            max_response_bytes=1_048_576,
        ) as client:
            return client._agent_libos_connection

    connection = asyncio.run(exercise())
    assert connection.protocol_revision == "2025-11-25"
    assert connection.fallback_used
    assert _NegotiationSession.initialize_calls == 1


@pytest.mark.parametrize(
    "supported",
    [[], ["2099-01-01"], ["2025-11-25", "2099-01-01"], "2025-11-25"],
)
def test_stdio_auto_32022_rejects_malformed_or_future_only_versions(
    supported: Any,
) -> None:
    _NegotiationSession.discover_error = _McpError(
        -32022,
        {"supported": supported},
    )
    _NegotiationSession.initialize_calls = 0

    async def exercise() -> None:
        async with _mcp_sdk_v2_client(
            _NegotiationSession,
            _Transport(),
            server=_stdio_server(),
            mode=McpProtocolMode.AUTO,
            sdk_mode="auto",
            deadline=time.monotonic() + 1,
            max_response_bytes=1_048_576,
        ):
            raise AssertionError("unsafe -32022 must not connect")

    with pytest.raises(_McpError):
        asyncio.run(exercise())
    assert _NegotiationSession.initialize_calls == 0


def test_auto_rejects_future_sdk_revision_before_any_tools_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.client import ClientSession
    import mcp.client.session as sdk_session

    monkeypatch.setattr(
        sdk_session,
        "MODERN_PROTOCOL_VERSIONS",
        ("2026-07-28", "2099-01-01"),
    )
    transport = _RawWireTransport(
        supported_versions=["2026-07-28", "2099-01-01"],
    )

    async def exercise() -> None:
        async with _mcp_sdk_v2_client(
            ClientSession,
            transport,
            server=_http_server(),
            mode=McpProtocolMode.AUTO,
            sdk_mode="auto",
            deadline=time.monotonic() + 1,
            max_response_bytes=1_048_576,
        ):
            raise AssertionError("a future SDK revision must not be adopted")

    with pytest.raises(ValidationError, match="release-locked supported set"):
        asyncio.run(exercise())
    assert [item.method for item in transport.outbound] == ["server/discover"]


def test_stdio_provider_timeout_is_not_misclassified_as_legacy_probe_timeout() -> None:
    _NegotiationSession.discover_error = SubprocessTimeoutExpired(
        "provider resource timeout",
        metrics=CommandMetrics(killed=True, limit_kind="subprocess_timeout"),
    )
    _NegotiationSession.initialize_calls = 0

    async def exercise() -> None:
        async with _mcp_sdk_v2_client(
            _NegotiationSession,
            _Transport(),
            server=_stdio_server(),
            mode=McpProtocolMode.AUTO,
            sdk_mode="auto",
            deadline=time.monotonic() + 1,
            max_response_bytes=1_048_576,
        ):
            raise AssertionError("provider timeout must not connect")

    with pytest.raises(SubprocessTimeoutExpired, match="provider resource timeout"):
        asyncio.run(exercise())
    assert _NegotiationSession.initialize_calls == 0


def test_deadline_wrapper_retrieves_child_when_parent_is_cancelled() -> None:
    child_closed = asyncio.Event()

    async def child() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            child_closed.set()

    async def exercise() -> None:
        parent = asyncio.create_task(
            _mcp_await_with_deadline(
                child(),
                deadline=time.monotonic() + 10,
                stage="parent cancellation regression",
            )
        )
        await asyncio.sleep(0)
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
        assert child_closed.is_set()

    asyncio.run(exercise())


def test_manifest_v1_locked_initialize_and_wire_frames_match_sdk_v1_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.client import ClientSession
    import mcp.client.session as sdk_session

    monkeypatch.setattr(sdk_session, "LATEST_HANDSHAKE_VERSION", "2099-01-01")
    server = McpServerSpec(
        **{
            **_http_server(McpProtocolMode.LEGACY).__dict__,
            "schema_version": 1,
            "protocol_mode": None,
        }
    )

    async def exercise() -> list[Any]:
        transport = _RawWireTransport()
        async with _mcp_sdk_v2_client(
            ClientSession,
            transport,
            server=server,
            mode=McpProtocolMode.LEGACY,
            sdk_mode="legacy",
            deadline=time.monotonic() + 2,
            max_response_bytes=1_048_576,
        ) as client:
            await client.list_tools()
            await client.session.call_tool(
                "demo.echo",
                {
                    "tiny": 1e-7,
                    "unicode": "你好",
                    "nested": {"value": -2.5e20},
                },
            )
            return list(transport.outbound)

    outbound = asyncio.run(exercise())
    frames = [
        _mcp_legacy_wire_bytes(
            (item.model_dump_json(by_alias=True, exclude_none=True) + "\n").encode(),
            newline=True,
        )
        for item in outbound
    ]
    assert frames == [
        b'{"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"mcp","version":"0.1.0"}},"jsonrpc":"2.0","id":0}\n',
        b'{"method":"notifications/initialized","jsonrpc":"2.0"}\n',
        b'{"method":"tools/list","jsonrpc":"2.0","id":1}\n',
        b'{"method":"tools/call","params":{"name":"demo.echo","arguments":{"tiny":1e-7,"unicode":"\xe4\xbd\xa0\xe5\xa5\xbd","nested":{"value":-2.5e+20}}},"jsonrpc":"2.0","id":2}\n',
    ]


@pytest.mark.parametrize("newline", [False, True])
def test_manifest_v1_wire_rewrite_preserves_json_value_tokens(newline: bool) -> None:
    encoded = (
        b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":'
        b'{"_meta":{},"name":"demo.\\u0065cho","arguments":'
        b'{"tiny":1e-7,"nested":{"_meta":{},"unicode":"\\u4f60\\u597d"}}}}'
    )
    expected = (
        b'{"method":"tools/call","params":{"name":"demo.\\u0065cho",'
        b'"arguments":{"tiny":1e-7,"nested":{"_meta":{},'
        b'"unicode":"\\u4f60\\u597d"}}},"jsonrpc":"2.0","id":7}'
        + (b"\n" if newline else b"")
    )

    assert _mcp_legacy_wire_bytes(encoded, newline=newline) == expected


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",',
        b'[1e-7,{"_meta":{}}]',
        b'{"id":1,"id":"1","method":"tools/list"}',
    ],
)
def test_manifest_v1_wire_rewrite_leaves_unsafe_shapes_untouched(
    encoded: bytes,
) -> None:
    assert _mcp_legacy_wire_bytes(encoded, newline=True) == encoded


def test_wire_ledger_counts_raw_frames_and_does_not_close_on_reverse_id_collision() -> None:
    from mcp.types import JSONRPCRequest, JSONRPCResponse

    ledger = _McpWireLedger()
    request = b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{}}\n'
    reverse = b'{"jsonrpc":"2.0","id":7,"method":"roots/list"}\n'
    response = b'{"jsonrpc":"2.0","id":7,"result":{"ok":true}}\n'
    ledger.record_stdio_request(request)
    ledger.record_stdio_response(
        reverse,
        JSONRPCRequest(jsonrpc="2.0", id=7, method="roots/list"),
    )
    during = ledger.receipts()[0]
    assert during.request_bytes == len(request)
    assert during.response_bytes == len(reverse)
    ledger.record_stdio_response(
        response,
        JSONRPCResponse(jsonrpc="2.0", id=7, result={"ok": True}),
    )
    completed = ledger.receipts()[0]
    assert completed.request_bytes == len(request)
    assert completed.response_bytes == len(reverse) + len(response)
    assert completed.call_started


def test_wire_ledger_keeps_jsonrpc_string_and_number_ids_distinct() -> None:
    from mcp.types import JSONRPCResponse

    ledger = _McpWireLedger()
    request = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{}}\n'
    wrong = b'{"jsonrpc":"2.0","id":"1","result":{"wrong":true}}\n'
    correct = b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n'

    ledger.record_stdio_request(request)
    ledger.record_stdio_response(
        wrong,
        JSONRPCResponse(jsonrpc="2.0", id="1", result={"wrong": True}),
    )
    during = ledger.receipts()[0]
    assert during.phase is McpExchangePhase.TOOLS_CALL
    assert during.response_bytes == len(wrong)

    ledger.record_stdio_response(
        correct,
        JSONRPCResponse(jsonrpc="2.0", id=1, result={"ok": True}),
    )
    completed = ledger.receipts()[0]
    assert completed.phase is McpExchangePhase.TOOLS_CALL
    assert completed.response_bytes == len(wrong) + len(correct)


def test_transport_failure_call_started_comes_only_from_call_wire_receipt() -> None:
    provider = SdkMcpProvider()
    server = McpServerSpec(**{**_http_server().__dict__, "tools": [_tool()]})
    list_only = (
        McpExchangeReceipt(
            phase=McpExchangePhase.TOOLS_LIST,
            request_bytes=51,
            response_bytes=67,
            call_started=True,
        ),
    )
    before_call = provider._mcp_transport_failure_result(  # noqa: SLF001
        server,
        _tool(),
        {},
        message="MCP HTTP operation exceeded max_response_bytes=100",
        started_at=time.monotonic(),
        max_response_bytes=100,
        receipts=list_only,
    )
    assert not before_call.call_started
    assert before_call.call_request_bytes == 0
    call_receipt = McpExchangeReceipt(
        phase=McpExchangePhase.TOOLS_CALL,
        request_bytes=73,
        response_bytes=89,
        call_started=True,
    )
    during_call = provider._mcp_transport_failure_result(  # noqa: SLF001
        server,
        _tool(),
        {},
        message="MCP HTTP operation exceeded max_response_bytes=100",
        started_at=time.monotonic(),
        max_response_bytes=100,
        receipts=(*list_only, call_receipt),
    )
    assert during_call.call_started
    assert during_call.list_request_bytes == 51
    assert during_call.call_request_bytes == 73
    assert during_call.call_response_bytes == 89


def _tool() -> McpToolSpec:
    return McpToolSpec(
        tool_id="echo",
        mcp_name="demo.echo",
        right="read",
        rollback_class="no_rollback_required",
        state_mutation=False,
        information_flow=True,
    )


def test_manifest_v2_pagination_is_bounded_and_receipted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SdkMcpProvider()
    tool = _tool()
    server = _http_server()
    server = McpServerSpec(**{**server.__dict__, "tools": [tool]})

    class Session:
        _agent_libos_sdk_v2 = True
        _agent_libos_receipts: list[Any] = []
        _agent_libos_connection = None

        def __init__(self) -> None:
            self.session = self
            self.list_calls: list[str | None] = []
            self.call_calls = 0

        async def list_tools(self, *, cursor: str | None = None, **_kwargs: Any) -> Any:
            self.list_calls.append(cursor)
            if cursor is None:
                return SimpleNamespace(
                    tools=[SimpleNamespace(name="demo.other", inputSchema={})],
                    nextCursor="page-2",
                )
            return SimpleNamespace(
                tools=[SimpleNamespace(name="demo.echo", inputSchema={})],
                nextCursor=None,
            )

        async def call_tool(
            self,
            _name: str,
            _arguments: dict[str, Any],
            **_kwargs: Any,
        ) -> Any:
            self.call_calls += 1
            return SimpleNamespace(content=[], structuredContent={"ok": True}, isError=False)

    session = Session()

    @contextlib.asynccontextmanager
    async def fake_session(*_args: Any, **_kwargs: Any):
        yield session

    monkeypatch.setattr(provider, "_session", fake_session)
    result = provider.validate_and_call(
        server,
        tool,
        {},
        timeout_s=1,
        max_response_bytes=server.max_response_bytes,
    )

    assert result.error is None
    assert session.list_calls == [None, "page-2"]
    assert session.call_calls == 1
    assert [item.phase for item in result.receipts] == [
        McpExchangePhase.TOOLS_LIST,
        McpExchangePhase.TOOLS_LIST,
        McpExchangePhase.TOOLS_CALL,
    ]


def test_v2_schema_drift_uses_canonical_json_and_has_no_call_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.client import ClientSession

    provider = SdkMcpProvider()
    manifest_schema = {
        "type": "object",
        "properties": {"flag": {"const": True}},
    }
    live_schema = {
        "type": "object",
        "properties": {"flag": {"const": 1}},
    }
    tool = McpToolSpec(**{**_tool().__dict__, "input_schema": manifest_schema})
    server = McpServerSpec(**{**_http_server().__dict__, "tools": [tool]})
    transport = _RawWireTransport(
        listed_tool={
            "name": tool.mcp_name,
            "inputSchema": live_schema,
        }
    )

    @contextlib.asynccontextmanager
    async def raw_session(*_args: Any, **kwargs: Any):
        async with _mcp_sdk_v2_client(
            ClientSession,
            transport,
            server=server,
            mode=McpProtocolMode.AUTO,
            sdk_mode="auto",
            deadline=kwargs["deadline"],
            max_response_bytes=kwargs["max_response_bytes"],
        ) as client:
            yield client

    monkeypatch.setattr(provider, "_session", raw_session)
    result = provider.validate_and_call(
        server,
        tool,
        {"flag": True},
        timeout_s=1,
        max_response_bytes=server.max_response_bytes,
    )

    assert result.error_type == "LiveToolValidationError"
    assert not result.call_started
    assert result.call_request_bytes == 0
    assert result.call_response_bytes == 0
    assert result.response_bytes == 0
    assert all(
        item.phase is not McpExchangePhase.TOOLS_CALL
        for item in result.receipts
    )
    assert [item.method for item in transport.outbound] == [
        "server/discover",
        "tools/list",
    ]
    # The primitive must accept the built-in provider's exact phase evidence;
    # it must not turn a safe pre-call denial into a malformed/unknown result.
    primitive = object.__new__(McpPrimitive)
    validated = primitive._validated_provider_call_result(server, result)  # noqa: SLF001
    assert not validated.call_started


def test_v2_output_schema_and_hints_are_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.client import ClientSession

    provider = SdkMcpProvider()
    tool = _tool()
    server = McpServerSpec(**{**_http_server().__dict__, "tools": [tool]})
    transport = _RawWireTransport(
        listed_tool={
            "name": tool.mcp_name,
            "inputSchema": {"type": "object"},
            "outputSchema": {
                "type": "object",
                "required": ["requiredByHintOnly"],
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": True,
            },
            "_meta": {"cacheHint": "diagnostic-only"},
        },
        call_result={
            "resultType": "complete",
            "content": [],
            "structuredContent": {"ok": True},
            "isError": False,
        },
    )

    @contextlib.asynccontextmanager
    async def raw_session(*_args: Any, **kwargs: Any):
        async with _mcp_sdk_v2_client(
            ClientSession,
            transport,
            server=server,
            mode=McpProtocolMode.AUTO,
            sdk_mode="auto",
            deadline=kwargs["deadline"],
            max_response_bytes=kwargs["max_response_bytes"],
        ) as client:
            yield client

    monkeypatch.setattr(provider, "_session", raw_session)
    result = provider.validate_and_call(
        server,
        tool,
        {},
        timeout_s=1,
        max_response_bytes=server.max_response_bytes,
    )

    assert result.error is None
    assert result.structured_content == {"ok": True}
    assert result.call_started
    assert [item.method for item in transport.outbound] == [
        "server/discover",
        "tools/list",
        "tools/call",
    ]
    assert [item.phase for item in result.receipts] == [
        McpExchangePhase.SERVER_DISCOVER,
        McpExchangePhase.TOOLS_LIST,
        McpExchangePhase.TOOLS_CALL,
    ]


def test_input_required_is_returned_once_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SdkMcpProvider()
    tool = _tool()
    server = McpServerSpec(**{**_http_server().__dict__, "tools": [tool]})

    class InputRequiredResult:
        result_type = "input_required"
        request_state = "must-not-be-projected"

    class Session:
        _agent_libos_sdk_v2 = True
        _agent_libos_receipts: list[Any] = []
        _agent_libos_connection = None

        def __init__(self) -> None:
            self.session = self
            self.call_calls = 0

        async def list_tools(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                tools=[SimpleNamespace(name="demo.echo", inputSchema={})],
                nextCursor=None,
            )

        async def call_tool(self, *_args: Any, **_kwargs: Any) -> Any:
            self.call_calls += 1
            return InputRequiredResult()

    session = Session()

    @contextlib.asynccontextmanager
    async def fake_session(*_args: Any, **_kwargs: Any):
        yield session

    monkeypatch.setattr(provider, "_session", fake_session)
    result = provider.validate_and_call(
        server,
        tool,
        {},
        timeout_s=1,
        max_response_bytes=server.max_response_bytes,
    )

    assert result.error_type == "mcp_input_required_unsupported"
    assert result.is_error
    assert result.call_started
    assert session.call_calls == 1
    assert "must-not-be-projected" not in str(result)


def test_tools_only_adapter_advertises_no_reverse_or_extension_capabilities() -> None:
    from mcp.client import ClientSession

    async def exercise() -> tuple[dict[str, Any], Any]:
        transport = _RawWireTransport()
        async with _mcp_sdk_v2_client(
            ClientSession,
            transport,
            server=_http_server(McpProtocolMode.REVISION_2026_07_28),
            mode=McpProtocolMode.REVISION_2026_07_28,
            sdk_mode="auto",
            deadline=time.monotonic() + 1,
            max_response_bytes=1_048_576,
        ) as client:
            session = client.session
            capabilities = session._build_capabilities(  # noqa: SLF001 - adapter contract
                "2026-07-28"
            ).model_dump(by_alias=True, mode="json", exclude_none=True)
            assert session._notification_bindings == {}  # noqa: SLF001
            assert session._binding_queues == {}  # noqa: SLF001
            assert session._listen_routes == {}  # noqa: SLF001
            return capabilities, transport.outbound[0]

    capabilities, discover_request = asyncio.run(exercise())

    assert capabilities == {}
    wire = discover_request.model_dump(by_alias=True, mode="json", exclude_none=True)
    advertised = wire["params"]["_meta"][
        "io.modelcontextprotocol/clientCapabilities"
    ]
    assert advertised == {}


def test_modern_server_reverse_requests_are_method_not_found_and_do_not_dispatch() -> None:
    from mcp.client import ClientSession
    from mcp.types import JSONRPCError

    reverse_requests = (
        (
            "sampling/createMessage",
            {
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": "run"}}
                ],
                "maxTokens": 1,
            },
        ),
        ("roots/list", None),
        (
            "elicitation/create",
            {
                "mode": "form",
                "message": "enter a secret",
                "requestedSchema": {"type": "object", "properties": {}},
            },
        ),
        ("logging/setLevel", {"level": "debug"}),
        ("subscriptions/listen", {"notifications": []}),
        ("tasks/get", {"taskId": "attacker-controlled"}),
    )

    async def exercise() -> tuple[list[Any], dict[str, Any]]:
        transport = _RawWireTransport()
        async with _mcp_sdk_v2_client(
            ClientSession,
            transport,
            server=_http_server(McpProtocolMode.REVISION_2026_07_28),
            mode=McpProtocolMode.REVISION_2026_07_28,
            sdk_mode="auto",
            deadline=time.monotonic() + 2,
            max_response_bytes=1_048_576,
        ) as client:
            responses = [
                await transport.request(method, params)
                for method, params in reverse_requests
            ]
            session_state = {
                "bindings": dict(client.session._binding_queues),  # noqa: SLF001
                "listens": dict(client.session._listen_routes),  # noqa: SLF001
                "extensions": client.session._extensions,  # noqa: SLF001
            }
            return responses, session_state

    responses, session_state = asyncio.run(exercise())

    assert all(isinstance(item, JSONRPCError) for item in responses)
    assert [item.error.code for item in responses] == [-32601] * len(reverse_requests)
    assert session_state == {"bindings": {}, "listens": {}, "extensions": None}


def test_legacy_reverse_callbacks_decline_sampling_roots_and_elicitation() -> None:
    from mcp.client import ClientSession
    from mcp.types import JSONRPCError

    reverse_requests = (
        (
            "sampling/createMessage",
            {
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": "run"}}
                ],
                "maxTokens": 1,
            },
        ),
        ("roots/list", None),
        (
            "elicitation/create",
            {
                "mode": "form",
                "message": "enter a secret",
                "requestedSchema": {"type": "object", "properties": {}},
            },
        ),
    )

    async def exercise() -> list[Any]:
        transport = _RawWireTransport()
        async with _mcp_sdk_v2_client(
            ClientSession,
            transport,
            server=_http_server(McpProtocolMode.LEGACY),
            mode=McpProtocolMode.LEGACY,
            sdk_mode="legacy",
            deadline=time.monotonic() + 2,
            max_response_bytes=1_048_576,
        ):
            return [
                await transport.request(method, params)
                for method, params in reverse_requests
            ]

    responses = asyncio.run(exercise())

    assert all(isinstance(item, JSONRPCError) for item in responses)
    assert [item.error.code for item in responses] == [-32600, -32600, -32600]
    assert [item.error.message for item in responses] == [
        "Sampling not supported",
        "List roots not supported",
        "Elicitation not supported",
    ]


def test_notification_flood_is_dropped_without_adapter_queues_or_runtime_actions() -> None:
    from mcp.client import ClientSession

    class CountingClientSession(ClientSession):
        notification_dispatches = 0

        async def _on_notify(self, *args: Any, **kwargs: Any) -> None:
            type(self).notification_dispatches += 1
            await super()._on_notify(*args, **kwargs)

    async def exercise() -> tuple[int, int, dict[str, Any]]:
        transport = _RawWireTransport()
        async with _mcp_sdk_v2_client(
            CountingClientSession,
            transport,
            server=_http_server(McpProtocolMode.REVISION_2026_07_28),
            mode=McpProtocolMode.REVISION_2026_07_28,
            sdk_mode="auto",
            deadline=time.monotonic() + 2,
            max_response_bytes=1_048_576,
        ) as client:
            outbound_before = len(transport.outbound)
            notifications = (
                ("notifications/message", {"level": "info", "data": "ignored"}),
                (
                    "notifications/subscriptions/acknowledged",
                    {"notifications": {}},
                ),
                ("notifications/tools/list_changed", None),
                ("notifications/attacker/flood", {"blob": "x"}),
            )
            for index in range(512):
                method, params = notifications[index % len(notifications)]
                await transport.notify(method, params)
            # The default SDK handlers checkpoint once.  Give those bounded,
            # no-op tasks an opportunity to leave the task group.
            await asyncio.sleep(0.05)
            return (
                len(transport.outbound) - outbound_before,
                CountingClientSession.notification_dispatches,
                {
                    "bindings": dict(client.session._binding_queues),  # noqa: SLF001
                    "listens": dict(client.session._listen_routes),  # noqa: SLF001
                    "extensions": client.session._extensions,  # noqa: SLF001
                },
            )

    outbound_messages, notification_dispatches, state = asyncio.run(exercise())

    assert outbound_messages == 0
    assert notification_dispatches == 0
    assert state == {"bindings": {}, "listens": {}, "extensions": None}


def test_stdio_notification_flood_is_counted_before_sdk_dispatch_and_bounded() -> None:
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-c",
            "import os\n"
            "line = b'{\"jsonrpc\":\"2.0\",\"method\":\"notifications/attacker/flood\"}\\n'\n"
            "while True: os.write(1, line)",
        ],
        env={},
    )

    async def exercise() -> None:
        async with _strict_stdio_client(
            params,
            max_frame_bytes=256,
            stdout_limit_bytes=2_048,
            deadline=time.monotonic() + 1,
            limits=SubprocessLimits(wall_seconds=1),
        ) as (read, _write):
            while True:
                item = await read.receive()
                if isinstance(item, BaseException):
                    raise item

    with pytest.raises(Exception) as caught:
        asyncio.run(exercise())

    messages = _exception_messages(caught.value)
    assert any(
        "MCP stdio stdout exceeded max_output_bytes=2048" in item
        for item in messages
    )


def test_safe_stdio_stdout_limit_matches_manifest_transport_budget() -> None:
    v2_server = McpServerSpec(
        **{
            **_stdio_server().__dict__,
            "max_response_bytes": 512,
        }
    )
    v1_server = McpServerSpec(
        **{
            **v2_server.__dict__,
            "schema_version": 1,
            "protocol_mode": None,
        }
    )

    assert McpPrimitive._safe_transport_error_message(  # noqa: SLF001
        v2_server,
        "McpStdioStdoutTooLarge",
    ) == "MCP stdio stdout exceeded max_output_bytes=512"
    assert McpPrimitive._safe_transport_error_message(  # noqa: SLF001
        v1_server,
        "McpStdioStdoutTooLarge",
    ) == "MCP stdio stdout exceeded max_output_bytes=2048"


def test_http_notification_bytes_use_one_operation_cumulative_budget() -> None:
    frame = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "notifications/attacker/flood",
            "params": {"blob": "x"},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    transport = _McpPolicyAsyncHTTPTransport(
        max_response_bytes=len(frame) * 2,
        max_request_bytes=1_024,
    )
    exchange = transport.wire_ledger.begin_http("subscriptions/listen")

    transport._count_response_bytes(  # noqa: SLF001 - wire edge contract
        len(frame), exchange=exchange
    )
    transport._count_response_bytes(  # noqa: SLF001 - wire edge contract
        len(frame), exchange=exchange
    )
    with pytest.raises(RuntimeError, match="MCP HTTP operation exceeded"):
        transport._count_response_bytes(  # noqa: SLF001 - wire edge contract
            1, exchange=exchange
        )

    assert transport.response_bytes == len(frame) * 2
    assert exchange.response_bytes == len(frame) * 2
    assert transport.limit_error is not None


def test_ambient_otel_trace_and_baggage_do_not_reach_modern_wire() -> None:
    from mcp.client import ClientSession
    from opentelemetry import baggage, context, trace
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
        TraceState,
    )

    trace_id = int("1234567890abcdef1234567890abcdef", 16)
    span_id = int("1234567890abcdef", 16)
    ambient = trace.set_span_in_context(
        NonRecordingSpan(
            SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
                trace_state=TraceState(),
            )
        )
    )
    ambient = baggage.set_baggage("tenant-secret", "must-not-leak", context=ambient)
    token = context.attach(ambient)
    try:
        async def exercise() -> tuple[dict[str, Any], dict[str, str]]:
            transport = _RawWireTransport()
            async with _mcp_sdk_v2_client(
                ClientSession,
                transport,
                server=_http_server(McpProtocolMode.REVISION_2026_07_28),
                mode=McpProtocolMode.REVISION_2026_07_28,
                sdk_mode="auto",
                deadline=time.monotonic() + 1,
                max_response_bytes=1_048_576,
            ):
                message = transport.outbound[0].model_dump(
                    by_alias=True,
                    mode="json",
                    exclude_none=True,
                )
                metadata = transport.outbound_metadata[0]
                return message, dict(metadata.headers or {})

        message, headers = asyncio.run(exercise())
        assert baggage.get_baggage("tenant-secret") == "must-not-leak"
    finally:
        context.detach(token)

    serialized = json.dumps(
        {"message": message, "headers": headers},
        sort_keys=True,
    ).lower()
    for forbidden in (
        "must-not-leak",
        "tenant-secret",
        "traceparent",
        "tracestate",
        "baggage",
        "1234567890abcdef1234567890abcdef",
    ):
        assert forbidden not in serialized


def test_modern_connection_redacts_reflected_exact_and_common_credentials() -> None:
    opaque_secret = "opaque-provider-credential-without-a-known-prefix"
    github_token = "ghp_0123456789abcdefghijklmnop"

    class ReflectingSession(_NegotiationSession):
        discover_error = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.server_info = SimpleNamespace(
                name=f"fixed-server/{opaque_secret}",
                version=f"release/{github_token}",
            )
            self.server_capabilities = {
                "tools": {"enabled": True},
                f"extension/{opaque_secret}": {"enabled": True},
                github_token: {"enabled": True},
            }

    async def exercise() -> Any:
        async with _mcp_sdk_v2_client(
            ReflectingSession,
            _Transport(),
            server=_http_server(McpProtocolMode.REVISION_2026_07_28),
            mode=McpProtocolMode.REVISION_2026_07_28,
            sdk_mode="auto",
            deadline=time.monotonic() + 1,
            max_response_bytes=1_048_576,
            sensitive_values=(opaque_secret,),
        ) as client:
            return client._agent_libos_connection

    connection = asyncio.run(exercise())

    assert connection.protocol_era is McpProtocolEra.MODERN
    assert connection.server_name == "fixed-server/[redacted]"
    assert connection.server_version == "release/[redacted]"
    assert set(connection.capabilities) == {
        "tools",
        "extension/[redacted]",
        "[redacted]",
    }
    assert set(connection.unsupported_capabilities) == {
        "extension/[redacted]",
        "[redacted]",
    }
    serialized = repr(connection)
    assert opaque_secret not in serialized
    assert github_token not in serialized


def _legacy_tools_http_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        get_count = 0
        delete_count = 0
        methods: list[str] = []
        requests: list[dict[str, Any]] = []
        response_body_bytes: list[int] = []
        get_started = threading.Event()
        malformed_phase: str | None = None

        def do_GET(self) -> None:  # noqa: N802
            type(self).get_count += 1
            type(self).requests.append(
                {
                    "method": "GET",
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": b"",
                }
            )
            type(self).get_started.set()
            self.send_response(405)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_DELETE(self) -> None:  # noqa: N802
            type(self).delete_count += 1
            type(self).requests.append(
                {
                    "method": "DELETE",
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": b"",
                }
            )
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) or b"{}"
            request = json.loads(raw_body)
            method = str(request.get("method", ""))
            type(self).methods.append(method)
            type(self).requests.append(
                {
                    "method": "POST",
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": raw_body,
                }
            )
            if method == "notifications/initialized":
                # Give an SDK-created background GET a deterministic chance to
                # reach this independent handler thread.
                type(self).get_started.wait(timeout=0.25)
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                type(self).response_body_bytes.append(0)
                return
            if method == "initialize":
                result: dict[str, Any] = {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "legacy-http", "version": "1"},
                }
            elif method == "server/discover":
                result = (
                    {"supportedVersions": "not-a-version-list"}
                    if type(self).malformed_phase == "server/discover"
                    else {
                        "supportedVersions": ["2026-07-28"],
                        "capabilities": {"tools": {}},
                        "resultType": "complete",
                        "ttlMs": 0,
                        "cacheScope": "private",
                    }
                )
            elif method == "tools/list":
                result = (
                    {"tools": "not-a-tool-list"}
                    if type(self).malformed_phase == "tools/list"
                    else {
                        "tools": [
                            {
                                "name": "demo.echo",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    }
                )
                if self.headers.get("Mcp-Protocol-Version") == "2026-07-28":
                    result.update(
                        {
                            "resultType": "complete",
                            "ttlMs": 0,
                            "cacheScope": "private",
                        }
                    )
            elif method == "tools/call":
                result = {
                    "content": [],
                    "structuredContent": {"ok": True},
                    "isError": False,
                }
            else:
                result = {}
            response = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"], "result": result},
                separators=(",", ":"),
            ).encode("utf-8")
            type(self).response_body_bytes.append(len(response))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if method in {"initialize", "server/discover"}:
                self.send_header("Mcp-Session-Id", "fixture-session")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    return Handler


def test_manifest_v1_http_preserves_sdk_v1_post_get_and_cleanup_sequence() -> None:
    handler = _legacy_tools_http_handler()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tool = _tool()
        spec = McpServerSpec(
            schema_version=1,
            server_id="legacy-http-network",
            transport="streamable_http",
            tools=[tool],
            # This test validates wire ordering and byte budgets, not a
            # two-second startup SLA.  Importing and initializing the optional
            # SDK can exceed that on cold Windows/Python 3.14 runners.
            timeout_s=10,
            max_request_bytes=160,
            max_response_bytes=250,
            http=McpHttpTransportSpec(
                url=f"http://127.0.0.1:{server.server_port}/mcp"
            ),
            protocol_mode=None,
        )

        result = SdkMcpProvider().validate_and_call(
            spec,
            tool,
            {
                "tiny": 1e-7,
                "unicode": "你好",
                "nested": {"value": -2.5e20},
            },
            timeout_s=spec.timeout_s,
            max_response_bytes=spec.max_response_bytes,
        )

        assert result.error is None
        assert result.structured_content == {"ok": True}
        assert handler.get_count == 1
        assert handler.delete_count == 1
        assert handler.methods.count("initialize") == 1
        assert handler.methods.count("notifications/initialized") == 1
        assert handler.methods.count("tools/list") == 1
        assert handler.methods.count("tools/call") == 1
        request_methods = [item["method"] for item in handler.requests]
        # The SDK starts the legacy SSE reader concurrently with its
        # initialized notification.  Their relative wire order is scheduler
        # dependent (notably across Python 3.11 and 3.14), while initialization
        # must still lead and cleanup must still trail every request.
        assert [method for method in request_methods if method != "GET"] == [
            "POST",
            "POST",
            "POST",
            "POST",
            "DELETE",
        ]
        assert 0 < request_methods.index("GET") < request_methods.index("DELETE")
        assert [
            item["body"]
            for item in handler.requests
            if item["method"] == "POST"
        ] == [
            b'{"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"mcp","version":"0.1.0"}},"jsonrpc":"2.0","id":0}',
            b'{"method":"notifications/initialized","jsonrpc":"2.0"}',
            b'{"method":"tools/list","jsonrpc":"2.0","id":1}',
            b'{"method":"tools/call","params":{"name":"demo.echo","arguments":{"tiny":1e-07,"unicode":"\xe4\xbd\xa0\xe5\xa5\xbd","nested":{"value":-2.5e+20}}},"jsonrpc":"2.0","id":2}',
        ]
        get_request = next(
            item for item in handler.requests if item["method"] == "GET"
        )
        delete_request = next(
            item for item in handler.requests if item["method"] == "DELETE"
        )
        assert get_request["headers"]["mcp-session-id"] == "fixture-session"
        assert "text/event-stream" in get_request["headers"]["accept"]
        assert delete_request["headers"]["mcp-session-id"] == "fixture-session"
        post_sizes = [
            len(item["body"])
            for item in handler.requests
            if item["method"] == "POST"
        ]
        assert max(post_sizes) <= spec.max_request_bytes < sum(post_sizes)
        assert (
            max(handler.response_body_bytes)
            <= spec.max_response_bytes
            < sum(handler.response_body_bytes)
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_manifest_v2_modern_http_never_starts_legacy_get() -> None:
    handler = _legacy_tools_http_handler()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spec = McpServerSpec(
            schema_version=2,
            server_id="modern-http-no-get",
            transport="streamable_http",
            tools=[],
            timeout_s=2,
            max_request_bytes=65_536,
            max_response_bytes=1_048_576,
            http=McpHttpTransportSpec(
                url=f"http://127.0.0.1:{server.server_port}/mcp"
            ),
            protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        )

        result = SdkMcpProvider().list_tools(
            spec,
            timeout_s=2,
            max_response_bytes=spec.max_response_bytes,
        )

        assert [tool.name for tool in result.tools] == ["demo.echo"]
        assert handler.methods == ["server/discover", "tools/list"]
        assert handler.get_count == 0
        assert handler.delete_count == 0
        assert [item["method"] for item in handler.requests] == [
            "POST",
            "POST",
        ]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_manifest_v2_http_keeps_cumulative_operation_response_budget() -> None:
    handler = _legacy_tools_http_handler()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spec = McpServerSpec(
            schema_version=2,
            server_id="modern-http-cumulative-budget",
            transport="streamable_http",
            tools=[],
            timeout_s=2,
            max_request_bytes=2_048,
            max_response_bytes=200,
            http=McpHttpTransportSpec(
                url=f"http://127.0.0.1:{server.server_port}/mcp"
            ),
            protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        )

        with pytest.raises(RuntimeError, match="MCP HTTP operation exceeded"):
            SdkMcpProvider().list_tools(
                spec,
                timeout_s=2,
                max_response_bytes=spec.max_response_bytes,
            )

        assert max(handler.response_body_bytes) <= spec.max_response_bytes
        assert sum(handler.response_body_bytes) > spec.max_response_bytes
        assert handler.methods == ["server/discover", "tools/list"]
        assert handler.get_count == 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_manifest_v2_http_call_limit_returns_valid_wire_failure() -> None:
    handler = _legacy_tools_http_handler()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tool = McpToolSpec(
            **{
                **_tool().__dict__,
                "right": "write",
                "rollback_class": "unknown",
                "state_mutation": True,
            }
        )
        spec = McpServerSpec(
            schema_version=2,
            server_id="modern-http-call-cumulative-budget",
            transport="streamable_http",
            tools=[tool],
            timeout_s=2,
            max_request_bytes=2_048,
            max_response_bytes=350,
            http=McpHttpTransportSpec(
                url=f"http://127.0.0.1:{server.server_port}/mcp"
            ),
            protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        )

        result = SdkMcpProvider().validate_and_call(
            spec,
            tool,
            {},
            timeout_s=2,
            max_response_bytes=spec.max_response_bytes,
        )

        assert handler.methods == ["server/discover", "tools/list", "tools/call"]
        assert result.error_type == "McpHttpOperationTooLarge"
        assert result.connection is not None
        assert result.call_started
        assert (
            sum(item.response_bytes for item in result.receipts)
            <= spec.max_response_bytes
        )
        assert result.receipts[-1].phase is McpExchangePhase.TOOLS_CALL
        assert result.receipts[-1].response_bytes == 0

        primitive = object.__new__(McpPrimitive)
        validated = primitive._validated_provider_call_result(  # noqa: SLF001
            spec,
            result,
        )
        public_result = primitive._call_result_from_provider(  # noqa: SLF001
            spec,
            tool,
            validated,
        )
        assert public_result.status.value == "transport_error"
        assert public_result.error is not None
        assert public_result.error["message"] == (
            "MCP HTTP operation exceeded max_response_bytes=350"
        )

        classification = primitive._post_call_failure_classification_override(  # noqa: SLF001
            tool,
            validated,
        )
        assert classification is not None
        assert classification.rollback_class.value == "unknown"
        assert classification.rollback_status.value == "unknown"
        assert classification.metadata["call_started"] is True
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize(
    ("malformed_phase", "expected_methods", "has_connection"),
    [
        ("server/discover", ["server/discover"], False),
        ("tools/list", ["server/discover", "tools/list"], True),
    ],
)
def test_builtin_v2_malformed_pre_call_response_is_wire_certified(
    malformed_phase: str,
    expected_methods: list[str],
    has_connection: bool,
) -> None:
    handler = _legacy_tools_http_handler()
    handler.malformed_phase = malformed_phase
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tool = McpToolSpec(
            **{
                **_tool().__dict__,
                "right": "write",
                "rollback_class": "unknown",
                "state_mutation": True,
            }
        )
        spec = McpServerSpec(
            schema_version=2,
            server_id=f"malformed-{malformed_phase.replace('/', '-')}",
            transport="streamable_http",
            tools=[tool],
            timeout_s=2,
            max_request_bytes=65_536,
            max_response_bytes=1_048_576,
            http=McpHttpTransportSpec(
                url=f"http://127.0.0.1:{server.server_port}/mcp"
            ),
            protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        )

        result = SdkMcpProvider().validate_and_call(
            spec,
            tool,
            {},
            timeout_s=2,
            max_response_bytes=spec.max_response_bytes,
        )

        assert handler.methods == expected_methods
        assert result.error_type == "McpPreCallFailure"
        assert (result.connection is not None) is has_connection
        assert not result.call_started
        assert result.call_request_bytes == 0
        assert result.call_response_bytes == 0
        assert all(
            receipt.phase is not McpExchangePhase.TOOLS_CALL
            for receipt in result.receipts
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _sse_resume_http_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        methods: list[str] = []
        get_last_event_ids: list[str | None] = []
        delete_count = 0
        call_sse_body = b"id: resume-1\nretry: 0\nevent: message\ndata:\n\n"

        def do_GET(self) -> None:  # noqa: N802
            last_event_id = self.headers.get("Last-Event-ID")
            type(self).get_last_event_ids.append(last_event_id)
            if last_event_id is None:
                self.send_response(405)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 999,
                    "result": {
                        "content": [],
                        "structuredContent": {"ok": True},
                        "isError": False,
                    },
                },
                separators=(",", ":"),
            )
            body = f"id: resume-2\nevent: message\ndata: {payload}\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_DELETE(self) -> None:  # noqa: N802
            type(self).delete_count += 1
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            method = str(request.get("method", ""))
            type(self).methods.append(method)
            if method == "notifications/initialized":
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if method == "tools/call":
                body = type(self).call_sse_body
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if method == "initialize":
                result: dict[str, Any] = {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "sse-legacy", "version": "1"},
                }
            elif method == "server/discover":
                result = {
                    "supportedVersions": ["2026-07-28"],
                    "capabilities": {"tools": {}},
                    "resultType": "complete",
                    "ttlMs": 0,
                    "cacheScope": "private",
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "demo.echo",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                }
                if self.headers.get("Mcp-Protocol-Version") == "2026-07-28":
                    result.update(
                        {
                            "resultType": "complete",
                            "ttlMs": 0,
                            "cacheScope": "private",
                        }
                    )
            else:
                result = {}
            body = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"], "result": result},
                separators=(",", ":"),
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if method in {"initialize", "server/discover"}:
                self.send_header("Mcp-Session-Id", "sse-session")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    return Handler


def test_manifest_v2_sse_disconnect_never_resumes_with_get() -> None:
    handler = _sse_resume_http_handler()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = SdkMcpProvider()
    try:
        tool = McpToolSpec(
            **{
                **_tool().__dict__,
                "right": "write",
                "rollback_class": "unknown",
                "state_mutation": True,
            }
        )
        spec = McpServerSpec(
            schema_version=2,
            server_id="modern-sse-no-resume",
            transport="streamable_http",
            tools=[tool],
            timeout_s=2,
            max_request_bytes=65_536,
            max_response_bytes=1_048_576,
            http=McpHttpTransportSpec(
                url=f"http://127.0.0.1:{server.server_port}/mcp"
            ),
            protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        )

        with pytest.raises(Exception) as caught:
            provider.validate_and_call(
                spec,
                tool,
                {},
                timeout_s=2,
                max_response_bytes=spec.max_response_bytes,
            )

        assert handler.methods == ["server/discover", "tools/list", "tools/call"]
        assert handler.get_last_event_ids == []
        receipts = provider._mcp_transport_receipts(caught.value)  # noqa: SLF001
        certified, evidence_receipts, connection = provider._mcp_wire_failure_evidence(  # noqa: SLF001
            caught.value
        )
        assert certified
        assert connection is not None
        assert evidence_receipts == receipts
        call_receipts = [
            item for item in receipts if item.phase is McpExchangePhase.TOOLS_CALL
        ]
        assert len(call_receipts) == 1
        assert receipts[-1] is call_receipts[0]
        assert call_receipts[0].call_started
        assert call_receipts[0].response_bytes == len(handler.call_sse_body)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_manifest_v1_sse_disconnect_preserves_sdk_v1_get_resumption() -> None:
    handler = _sse_resume_http_handler()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tool = _tool()
        spec = McpServerSpec(
            schema_version=1,
            server_id="legacy-sse-resume",
            transport="streamable_http",
            tools=[tool],
            timeout_s=2,
            max_request_bytes=65_536,
            max_response_bytes=1_048_576,
            http=McpHttpTransportSpec(
                url=f"http://127.0.0.1:{server.server_port}/mcp"
            ),
            protocol_mode=None,
        )

        result = SdkMcpProvider().validate_and_call(
            spec,
            tool,
            {},
            timeout_s=2,
            max_response_bytes=spec.max_response_bytes,
        )

        assert result.error is None
        assert result.structured_content == {"ok": True}
        assert handler.methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
        assert "resume-1" in handler.get_last_event_ids
        assert handler.get_last_event_ids.count("resume-1") == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _exception_messages(error: BaseException) -> list[str]:
    if isinstance(error, BaseExceptionGroup):
        return [
            message
            for nested in error.exceptions
            for message in _exception_messages(nested)
        ]
    return [str(error)]
