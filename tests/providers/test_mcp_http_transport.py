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


httpx = pytest.importorskip("httpx")


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
