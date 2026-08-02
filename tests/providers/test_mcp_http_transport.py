from __future__ import annotations

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
import time
from typing import Any, Iterator

import pytest

from agent_libos.substrate.local import (
    _bounded_mcp_http_stream,
    _McpDeadlineNetworkStream,
    _McpPolicyAsyncHTTPTransport,
    _McpPolicyNetworkBackend,
)


pytestmark = pytest.mark.mcp


def _handler(
    body: bytes,
    *,
    content_encoding: str | None = None,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        accept_encodings: list[str] = []

        def do_GET(self) -> None:  # noqa: N802
            type(self).accept_encodings.append(self.headers.get("Accept-Encoding", ""))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if content_encoding is not None:
                self.send_header("Content-Encoding", content_encoding)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    return Handler


@contextmanager
def _serve(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/mcp"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


async def _get(url: str, *, max_response_bytes: int) -> bytes:
    # Keep the optional SDK import inside the marked test path.  Pytest still
    # imports this module while collecting unrelated lanes (including the
    # PostgreSQL --fail-on-skip gate), where an importorskip at module scope
    # would turn an unselected MCP dependency into a session-wide skip.
    import httpx2 as httpx

    transport = _McpPolicyAsyncHTTPTransport(
        max_response_bytes=max_response_bytes,
    )
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get(url)
        return bytes(response.content)


def test_public_httpcore_transport_round_trip_forces_identity_encoding() -> None:
    handler = _handler(b'{"ok":true}')

    with _serve(handler) as url:
        body = asyncio.run(_get(url, max_response_bytes=1024))

    assert body == b'{"ok":true}'
    assert handler.accept_encodings == ["identity"]


def test_http_transport_bounds_body_before_materialization() -> None:
    handler = _handler(b"x" * 32)

    with _serve(handler) as url:
        with pytest.raises(
            RuntimeError,
            match="MCP HTTP response exceeded max_response_bytes=16",
        ):
            asyncio.run(_get(url, max_response_bytes=16))


def test_http_transport_rejects_content_encoding_before_decode() -> None:
    handler = _handler(b"encoded", content_encoding="gzip")

    with _serve(handler) as url:
        with pytest.raises(
            RuntimeError,
            match="unsupported Content-Encoding=gzip",
        ):
            asyncio.run(_get(url, max_response_bytes=1024))


@pytest.mark.parametrize(
    ("body", "allowed"),
    [
        (b"", True),
        (b'{"jsonrpc":"2.0","id":0,"error":{"code":-32601,"message":"missing"}}', True),
        (b'{"jsonrpc":"2.0","id":0,"error":{"code":-32022,"message":"modern"}}', False),
        (b"not-json", False),
        (b'{"jsonrpc":"2.0","id":0,"result":{}}', False),
    ],
)
def test_http_400_fallback_requires_bounded_unambiguous_legacy_signal(
    body: bytes,
    allowed: bool,
) -> None:
    transport = _McpPolicyAsyncHTTPTransport(max_response_bytes=1024)
    exchange = transport.wire_ledger.begin_http("server/discover")
    exchange.call_started = True
    exchange.started_at = time.monotonic()
    exchange.response_body = bytearray(body)
    exchange.response_declared_bytes = len(body)
    transport.last_request_method = "server/discover"
    transport.last_response_status = 400

    transport._finish_http_response(exchange)  # noqa: SLF001 - fallback edge

    assert transport.last_legacy_400_signal is allowed


def test_network_backend_bounds_slow_dns_by_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver_started = threading.Event()
    release_resolver = threading.Event()
    resolver_completed = threading.Event()

    def slow_getaddrinfo(*_args: Any, **_kwargs: Any) -> list[Any]:
        resolver_started.set()
        release_resolver.wait(timeout=3.0)
        try:
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("1.1.1.1", 443),
                )
            ]
        finally:
            resolver_completed.set()

    monkeypatch.setattr(socket, "getaddrinfo", slow_getaddrinfo)

    async def exercise() -> float:
        backend = _McpPolicyNetworkBackend(deadline=time.monotonic() + 0.2)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await backend.connect_tcp("deadline.example", 443, timeout=10.0)
        return time.monotonic() - started

    try:
        elapsed = asyncio.run(exercise())
        assert resolver_started.wait(timeout=1.0)
        assert not resolver_completed.is_set()
        assert elapsed < 1.5
    finally:
        release_resolver.set()

    assert resolver_completed.wait(timeout=1.0)


def test_network_backend_shares_deadline_across_resolved_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def resolved_addresses(*_args: Any, **_kwargs: Any) -> list[str]:
        return ["1.1.1.1", "8.8.8.8"]

    class FailingBackend:
        def __init__(self) -> None:
            self.timeouts: list[float | None] = []

        async def connect_tcp(
            self,
            _host: str,
            _port: int,
            *,
            timeout: float | None,
            local_address: str | None,
            socket_options: Any,
        ) -> Any:
            del local_address, socket_options
            self.timeouts.append(timeout)
            raise OSError("synthetic connection failure")

    monkeypatch.setattr(
        "agent_libos.substrate.local.run_blocking_once",
        resolved_addresses,
    )

    async def exercise() -> list[float | None]:
        backend = _McpPolicyNetworkBackend(deadline=time.monotonic() + 10.0)
        failing = FailingBackend()
        backend._backend = failing
        with pytest.raises(OSError, match="synthetic connection failure"):
            await backend.connect_tcp("multi.example", 443, timeout=10.0)
        return failing.timeouts

    timeouts = asyncio.run(exercise())

    assert len(timeouts) == 2
    assert timeouts[0] is not None and timeouts[1] is not None
    assert 0 < timeouts[1] <= timeouts[0] <= 10.0


def test_network_stream_clamps_write_tls_and_read_to_one_deadline() -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.timeouts: list[float | None] = []

        async def write(self, _buffer: bytes, timeout: float | None = None) -> None:
            self.timeouts.append(timeout)

        async def start_tls(
            self,
            _ssl_context: Any,
            *,
            server_hostname: str | None,
            timeout: float | None,
        ) -> "FakeStream":
            del server_hostname
            self.timeouts.append(timeout)
            return self

        async def read(self, _max_bytes: int, timeout: float | None = None) -> bytes:
            self.timeouts.append(timeout)
            return b"ok"

        async def aclose(self) -> None:
            return None

        def get_extra_info(self, _info: str) -> Any:
            return None

    async def exercise() -> list[float | None]:
        raw = FakeStream()
        stream = _McpDeadlineNetworkStream(
            raw,
            deadline=time.monotonic() + 10.0,
        )
        await stream.write(b"request", timeout=10.0)
        stream = await stream.start_tls(
            object(),
            server_hostname="deadline.example",
            timeout=10.0,
        )
        assert await stream.read(16, timeout=10.0) == b"ok"
        return raw.timeouts

    timeouts = asyncio.run(exercise())

    selected_timeouts = [float(timeout) for timeout in timeouts if timeout is not None]
    assert len(selected_timeouts) == 3
    assert all(0 < timeout <= 10.0 for timeout in selected_timeouts)
    assert selected_timeouts[0] >= selected_timeouts[1] >= selected_timeouts[2]


def test_http_response_chunks_cannot_reset_absolute_deadline() -> None:
    class SlowChunks:
        def __init__(self) -> None:
            self.index = 0

        def __aiter__(self) -> "SlowChunks":
            return self

        async def __anext__(self) -> bytes:
            if self.index >= 2:
                raise StopAsyncIteration
            self.index += 1
            await asyncio.sleep(0.12)
            return b"x"

        async def aclose(self) -> None:
            return None

    async def exercise() -> None:
        bounded = _bounded_mcp_http_stream(
            SlowChunks(),
            max_response_bytes=64,
            is_sse=False,
            fail=RuntimeError,
            deadline=time.monotonic() + 0.2,
        )
        with pytest.raises(TimeoutError):
            async for _chunk in bounded:
                pass

    asyncio.run(asyncio.wait_for(exercise(), timeout=1.5))


def test_http_response_iteration_and_close_share_one_task() -> None:
    class OneChunk:
        def __init__(self) -> None:
            self.iteration_task: asyncio.Task[Any] | None = None
            self.close_task: asyncio.Task[Any] | None = None
            self.sent = False

        def __aiter__(self) -> "OneChunk":
            return self

        async def __anext__(self) -> bytes:
            self.iteration_task = asyncio.current_task()
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return b"too large"

        async def aclose(self) -> None:
            self.close_task = asyncio.current_task()

    async def exercise() -> OneChunk:
        source = OneChunk()
        bounded = _bounded_mcp_http_stream(
            source,
            max_response_bytes=1,
            is_sse=False,
            fail=RuntimeError,
            deadline=time.monotonic() + 1,
        )
        with pytest.raises(RuntimeError, match="MCP HTTP response exceeded"):
            async for _chunk in bounded:
                pass
        await bounded.aclose()
        return source

    source = asyncio.run(exercise())

    assert source.iteration_task is not None
    assert source.close_task is source.iteration_task
