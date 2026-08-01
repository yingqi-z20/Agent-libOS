from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    CapabilityRight,
    McpCallStatus,
    McpExchangePhase,
    McpHttpTransportSpec,
    McpProtocolEra,
    McpServerSpec,
    McpToolSpec,
)
from agent_libos.substrate import LocalResourceProviderSubstrate, SdkMcpProvider
from agent_libos.substrate.base import SubprocessLimits
from agent_libos.substrate.local import (
    _McpHttpResponseLimiter,
    _MCP_STABLE_CWD_SUPPORTED,
    _mcp_stdio_read_size,
    _strict_stdio_client,
)

pytestmark = pytest.mark.mcp


def _grant_stdio_spawn(
    runtime: Runtime,
    pid: str,
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> None:
    runtime.capability.grant(pid, "process:spawn", [CapabilityRight.WRITE], issued_by="test")
    runtime.capability.grant(
        pid,
        runtime.mcp.stdio_resource_for_argv(command, args, env=env, cwd=cwd),
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )


class TestMcpSdkIntegration:
    def test_stdio_receive_size_never_reads_past_frame_limit_sentinel(self) -> None:
        assert _mcp_stdio_read_size(0, 2_048) == 2_049
        assert _mcp_stdio_read_size(2_048, 2_048) == 1
        assert _mcp_stdio_read_size(0, 1_000_000) == 64 * 1_024

    def test_http_sse_limiter_resets_each_crlf_frame_and_bounds_cross_chunk_event(self) -> None:
        limiter = _McpHttpResponseLimiter(max_response_bytes=18, is_sse=True)

        assert limiter.feed(b'data: first\r') is None
        assert limiter.feed(b'\n\r\n') is None
        assert limiter.feed(b'data: second\r\n\r') is None
        assert limiter.feed(b'\n') is None
        assert limiter.feed(b'data: oversized-') is None
        assert limiter.feed(b'event') == 'MCP HTTP SSE frame exceeded max_response_bytes=18'

    def test_stdio_fastmcp_tool_call(self, tmp_path: Path) -> None:
        server_path = _write_fastmcp_stdio_server(tmp_path)
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="mcp stdio integration")
            args = [str(server_path)]
            runtime.mcp.register_server(_server_spec("stdio-it", sys.executable, args), actor="cli", require_capability=False)
            runtime.capability.grant(pid, "mcp:stdio-it:echo", [CapabilityRight.READ], issued_by="test")
            _grant_stdio_spawn(runtime, pid, sys.executable, args)

            result = runtime.mcp.call_tool(pid, "stdio-it", "echo", {"text": "hello"})

            assert result.ok
            assert "hello" in json.dumps(result.result, sort_keys=True)
        finally:
            runtime.close()

    def test_stdio_modern_discovery_call_and_phase_receipts(self, tmp_path: Path) -> None:
        server_path = _write_fastmcp_stdio_server(tmp_path)
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="mcp modern stdio")
            args = [str(server_path)]
            spec = _server_spec("stdio-modern-it", sys.executable, args)
            spec["schema_version"] = 2
            spec["protocol_mode"] = "2026-07-28"
            runtime.mcp.register_server(spec, actor="cli", require_capability=False)
            runtime.capability.grant(
                pid,
                "mcp:stdio-modern-it:echo",
                [CapabilityRight.READ],
                issued_by="test",
            )
            _grant_stdio_spawn(runtime, pid, sys.executable, args)

            result = runtime.mcp.call_tool(
                pid,
                "stdio-modern-it",
                "echo",
                {"text": "modern"},
            )

            assert result.ok
            assert result.connection is not None
            assert result.connection.protocol_era is McpProtocolEra.MODERN
            assert result.connection.protocol_revision == "2026-07-28"
            assert not result.connection.fallback_used
            assert [item.phase for item in result.receipts] == [
                McpExchangePhase.SERVER_DISCOVER,
                McpExchangePhase.TOOLS_LIST,
                McpExchangePhase.TOOLS_CALL,
            ]
        finally:
            runtime.close()

    @pytest.mark.skipif(
        not _MCP_STABLE_CWD_SUPPORTED,
        reason="stable configured cwd dispatch requires Linux /proc/self/fd",
    )
    def test_stdio_uses_workspace_cwd_and_exact_allowlisted_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace = tmp_path / "workspace"
        cwd = workspace / "server-cwd"
        cwd.mkdir(parents=True)
        server_path = _write_fastmcp_env_server(tmp_path)
        monkeypatch.setenv("AGENT_LIBOS_MCP_ALLOWED_TOKEN", "allowed-token")
        monkeypatch.setenv("OPENAI_API_KEY", "should-not-inherit")
        runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(workspace))
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="mcp stdio env integration")
            args = [str(server_path)]
            spec = _server_spec("stdio-env-it", sys.executable, args)
            spec["stdio"]["cwd"] = "server-cwd"
            spec["stdio"]["env"] = {"DEMO_TOKEN": "AGENT_LIBOS_MCP_ALLOWED_TOKEN"}
            spec["tools"] = [_tool_spec(tool_id="envcwd", mcp_name="demo.envcwd")]
            runtime.mcp.register_server(spec, actor="cli", require_capability=False)
            runtime.capability.grant(pid, "mcp:stdio-env-it:envcwd", [CapabilityRight.READ], issued_by="test")
            _grant_stdio_spawn(
                runtime,
                pid,
                sys.executable,
                args,
                env={"DEMO_TOKEN": "AGENT_LIBOS_MCP_ALLOWED_TOKEN"},
                cwd="server-cwd",
            )

            result = runtime.mcp.call_tool(pid, "stdio-env-it", "envcwd", {})

            assert result.ok
            structured = result.result["structured_content"]
            assert Path(structured["cwd"]).resolve() == cwd.resolve()
            assert structured["allowed"] == "allowed-token"
            assert structured["secret"] is None
        finally:
            runtime.close()

    def test_stdio_raw_response_frame_exceeding_limit_is_rejected(self, tmp_path: Path) -> None:
        server_path = _write_fastmcp_large_stdio_server(tmp_path)
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='mcp stdio raw frame limit')
            args = [str(server_path)]
            spec = _server_spec('stdio-frame-limit', sys.executable, args)
            spec['tools'] = [_tool_spec(tool_id='large', mcp_name='demo.large')]
            spec['max_response_bytes'] = 2_048
            runtime.mcp.register_server(spec, actor='cli', require_capability=False)
            runtime.capability.grant(pid, 'mcp:stdio-frame-limit:large', [CapabilityRight.READ], issued_by='test')
            _grant_stdio_spawn(runtime, pid, sys.executable, args)

            result = runtime.mcp.call_tool(pid, 'stdio-frame-limit', 'large', {})

            assert not result.ok
            assert result.status == McpCallStatus.TRANSPORT_ERROR
            assert 'MCP stdio frame exceeded max_response_bytes=2048' in result.error['message']
        finally:
            runtime.close()

    def test_stdio_infinite_small_frames_hit_aggregate_stdout_limit(self) -> None:
        from mcp.client.stdio import StdioServerParameters

        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-c",
                "import os\n"
                "line = b'{\"jsonrpc\":\"2.0\",\"method\":\"tick\"}\\n'\n"
                "while True: os.write(1, line)",
            ],
            env={},
        )

        async def exercise() -> None:
            async with _strict_stdio_client(
                params,
                max_frame_bytes=256,
                deadline=time.monotonic() + 1.0,
                limits=SubprocessLimits(wall_seconds=1.0),
            ) as (read, _write):
                while True:
                    item = await read.receive()
                    if isinstance(item, BaseException):
                        raise item

        with pytest.raises(Exception) as caught:
            asyncio.run(exercise())

        assert "MCP stdio stdout exceeded max_output_bytes=" in "\n".join(
            _exception_messages(caught.value)
        )

    @pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup regression")
    def test_stdio_cleanup_kills_child_after_direct_parent_exits(
        self,
        tmp_path: Path,
    ) -> None:
        from mcp.client.stdio import StdioServerParameters

        pid_file = tmp_path / "stdio-child.pid"
        script = (
            "import pathlib, subprocess, sys\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        )
        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", script],
            env={},
        )

        async def exercise() -> None:
            async with _strict_stdio_client(
                params,
                max_frame_bytes=1_024,
                deadline=time.monotonic() + 1.0,
                limits=SubprocessLimits(wall_seconds=1.0),
            ):
                until = time.monotonic() + 0.75
                while not pid_file.exists() and time.monotonic() < until:
                    await asyncio.sleep(0.01)
                assert pid_file.exists()
                await asyncio.sleep(0.05)

        asyncio.run(asyncio.wait_for(exercise(), timeout=2.0))
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        try:
            until = time.monotonic() + 1.0
            while _pid_is_running(child_pid) and time.monotonic() < until:
                time.sleep(0.02)
            assert not _pid_is_running(child_pid)
        finally:
            if _pid_is_running(child_pid):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child_pid, 9)

    def test_streamable_http_fastmcp_tool_call(self, tmp_path: Path) -> None:
        port = _free_local_port()
        server_path = _write_fastmcp_http_server(tmp_path)
        proc = subprocess.Popen(
            [sys.executable, str(server_path), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_port(port)
            runtime = Runtime.open("local")
            try:
                pid = runtime.process.spawn(image="base-agent:v0", goal="mcp http integration")
                runtime.mcp.register_server(_http_server_spec("http-it", f"http://127.0.0.1:{port}/mcp"), actor="cli", require_capability=False)
                runtime.capability.grant(pid, "mcp:http-it:echo", [CapabilityRight.READ], issued_by="test")

                result = runtime.mcp.call_tool(pid, "http-it", "echo", {"text": "hello"})

                assert result.ok
                assert "hello" in json.dumps(result.result, sort_keys=True)
            finally:
                runtime.close()
        finally:
            _stop_http_server_process(proc)

    def test_streamable_http_raw_response_exceeding_limit_is_rejected(self, tmp_path: Path) -> None:
        port = _free_local_port()
        server_path = _write_fastmcp_http_server(tmp_path)
        proc = subprocess.Popen(
            [sys.executable, str(server_path), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_port(port)
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='mcp http raw response limit')
                spec = _http_server_spec('http-limit-it', f'http://127.0.0.1:{port}/mcp')
                spec['tools'] = [_tool_spec(tool_id='large', mcp_name='demo.large')]
                spec['max_response_bytes'] = 2_048
                runtime.mcp.register_server(spec, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'mcp:http-limit-it:large', [CapabilityRight.READ], issued_by='test')

                result = runtime.mcp.call_tool(pid, 'http-limit-it', 'large', {})

                assert not result.ok
                assert result.status == McpCallStatus.TRANSPORT_ERROR
                assert 'MCP HTTP' in result.error['message']
                assert 'max_response_bytes=2048' in result.error['message']
            finally:
                runtime.close()
        finally:
            _stop_http_server_process(proc)

    def test_streamable_http_provider_honors_call_limit_below_manifest_limit(self, tmp_path: Path) -> None:
        port = _free_local_port()
        server_path = _write_fastmcp_http_server(tmp_path)
        proc = subprocess.Popen(
            [sys.executable, str(server_path), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_port(port)
            tool = McpToolSpec(
                tool_id='large',
                mcp_name='demo.large',
                right='read',
                rollback_class='no_rollback_required',
                state_mutation=False,
                information_flow=True,
            )
            spec = McpServerSpec(
                schema_version=1,
                server_id='http-provider-limit-it',
                transport='streamable_http',
                http=McpHttpTransportSpec(url=f'http://127.0.0.1:{port}/mcp'),
                tools=[tool],
                timeout_s=10,
                max_request_bytes=65536,
                max_response_bytes=1048576,
            )

            with pytest.raises(RuntimeError, match='max_response_bytes=2048'):
                SdkMcpProvider().call_tool(
                    spec,
                    tool,
                    {},
                    timeout_s=10,
                    max_response_bytes=2_048,
                )
        finally:
            _stop_http_server_process(proc)

    def test_streamable_http_rejects_content_encoding_before_decode(self) -> None:
        port = _free_local_port()
        handler = _encoded_response_handler()
        server = ThreadingHTTPServer(('127.0.0.1', port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        provider = SdkMcpProvider()
        try:
            spec = McpServerSpec(
                schema_version=1,
                server_id='encoded-it',
                transport='streamable_http',
                http=McpHttpTransportSpec(url=f'http://127.0.0.1:{port}/mcp'),
                tools=[],
                timeout_s=5,
                max_request_bytes=65536,
                max_response_bytes=2048,
            )

            async def request_encoded() -> None:
                async with provider._http_client(
                    spec,
                    timeout_s=5,
                    max_response_bytes=spec.max_response_bytes,
                ) as client:
                    await client.get(f'http://127.0.0.1:{port}/encoded')

            with pytest.raises(RuntimeError, match='unsupported Content-Encoding=gzip'):
                asyncio.run(request_encoded())
            assert handler.accept_encodings == ['identity']
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_streamable_http_json_body_is_bounded_before_materialization(self) -> None:
        port = _free_local_port()
        handler = _large_json_response_handler()
        server = ThreadingHTTPServer(('127.0.0.1', port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        provider = SdkMcpProvider()
        try:
            spec = McpServerSpec(
                schema_version=1,
                server_id='large-json-it',
                transport='streamable_http',
                http=McpHttpTransportSpec(url=f'http://127.0.0.1:{port}/mcp'),
                tools=[],
                timeout_s=5,
                max_request_bytes=65536,
                max_response_bytes=2048,
            )

            async def request_large_json() -> None:
                async with provider._http_client(
                    spec,
                    timeout_s=5,
                    max_response_bytes=spec.max_response_bytes,
                ) as client:
                    await client.get(f'http://127.0.0.1:{port}/large-json')

            with pytest.raises(RuntimeError, match='MCP HTTP response exceeded max_response_bytes=2048'):
                asyncio.run(request_large_json())
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_streamable_http_client_does_not_follow_redirects(self) -> None:
        port = _free_local_port()
        handler = _redirect_handler(f"http://127.0.0.1:{port}/private")
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        provider = SdkMcpProvider()
        try:
            spec = McpServerSpec(
                schema_version=1,
                server_id="redirect-it",
                transport="streamable_http",
                http=McpHttpTransportSpec(url=f"http://127.0.0.1:{port}/mcp"),
                tools=[],
                timeout_s=5,
                max_request_bytes=65536,
                max_response_bytes=1048576,
            )

            async def request_redirect() -> int:
                async with provider._http_client(
                    spec,
                    timeout_s=5,
                    max_response_bytes=spec.max_response_bytes,
                ) as client:
                    response = await client.get(f"http://127.0.0.1:{port}/redirect")
                    return int(response.status_code)

            status = asyncio.run(request_redirect())

            assert status == 307
            assert handler.paths == ["/redirect"]
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


def _stop_http_server_process(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            stream.close()


def _write_fastmcp_stdio_server(root: Path) -> Path:
    path = root / "stdio_server.py"
    path.write_text(
        """
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("stdio-it")

@mcp.tool(name="demo.echo")
def echo(text: str) -> dict[str, str]:
    return {"echo": text}

if __name__ == "__main__":
    mcp.run("stdio")
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_fastmcp_env_server(root: Path) -> Path:
    path = root / "env_server.py"
    path.write_text(
        """
from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("stdio-env-it")

@mcp.tool(name="demo.envcwd")
def envcwd() -> dict[str, str | None]:
    return {
        "cwd": os.getcwd(),
        "allowed": os.environ.get("DEMO_TOKEN"),
        "secret": os.environ.get("OPENAI_API_KEY"),
    }

if __name__ == "__main__":
    mcp.run("stdio")
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_fastmcp_large_stdio_server(root: Path) -> Path:
    path = root / 'large_stdio_server.py'
    path.write_text(
        """
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("stdio-frame-limit")

@mcp.tool(name="demo.large")
def large() -> dict[str, str]:
    return {"blob": "x" * 16384}

if __name__ == "__main__":
    mcp.run("stdio")
""".strip(),
        encoding='utf-8',
    )
    return path


def _write_fastmcp_http_server(root: Path) -> Path:
    path = root / "http_server.py"
    path.write_text(
        """
from __future__ import annotations

import sys

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("http-it")

@mcp.tool(name="demo.echo")
def echo(text: str) -> dict[str, str]:
    return {"echo": text}

@mcp.tool(name="demo.large")
def large() -> dict[str, str]:
    return {"blob": "x" * 16384}

if __name__ == "__main__":
    mcp.run(
        "streamable-http",
        host="127.0.0.1",
        port=int(sys.argv[1]),
        streamable_http_path="/mcp",
        stateless_http=True,
    )
""".strip(),
        encoding="utf-8",
    )
    return path


def _server_spec(server_id: str, command: str, args: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "server_id": server_id,
        "transport": "stdio",
        "stdio": {"command": command, "args": args},
        "tools": [_tool_spec()],
        "timeout_s": 10,
        "max_request_bytes": 65536,
        "max_response_bytes": 1048576,
    }


def _http_server_spec(server_id: str, url: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "server_id": server_id,
        "transport": "streamable_http",
        "http": {"url": url},
        "tools": [_tool_spec()],
        "timeout_s": 10,
        "max_request_bytes": 65536,
        "max_response_bytes": 1048576,
    }


def _tool_spec(tool_id: str = "echo", mcp_name: str = "demo.echo") -> dict[str, Any]:
    return {
        "tool_id": tool_id,
        "mcp_name": mcp_name,
        "right": "read",
        "rollback_class": "no_rollback_required",
        "state_mutation": False,
        "information_flow": True,
    }


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise AssertionError(f"MCP test server did not listen on port {port}")


def _exception_messages(error: BaseException) -> list[str]:
    messages = [str(error)]
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            messages.extend(_exception_messages(nested))
    return messages


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        import psutil

        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True


def _redirect_handler(location: str) -> type[BaseHTTPRequestHandler]:
    class RedirectHandler(BaseHTTPRequestHandler):
        paths: list[str] = []

        def do_GET(self) -> None:
            type(self).paths.append(self.path)
            if self.path == "/redirect":
                self.send_response(307)
                self.send_header("Location", location)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"redirect followed")

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return RedirectHandler


def _encoded_response_handler() -> type[BaseHTTPRequestHandler]:
    class EncodedResponseHandler(BaseHTTPRequestHandler):
        accept_encodings: list[str] = []

        def do_GET(self) -> None:
            type(self).accept_encodings.append(self.headers.get('Accept-Encoding', ''))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            self.wfile.write(b'not-actually-compressed')

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return EncodedResponseHandler


def _large_json_response_handler() -> type[BaseHTTPRequestHandler]:
    class LargeJsonResponseHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"blob":"' + (b'x' * 16384) + b'"}')

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return LargeJsonResponseHandler
