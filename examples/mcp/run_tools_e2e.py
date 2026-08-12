#!/usr/bin/env python3
"""Register Manifest v3 and run governed Tools over stdio and loopback HTTP."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from agent_libos import Runtime
from agent_libos.mcp.manifest import (
    McpServerManifestV3,
    parse_mcp_v3_manifest_yaml_text,
)
from agent_libos.mcp.types import McpComplete
from agent_libos.models import CapabilityRight
from agent_libos.utils.serde import to_jsonable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "mcp"


def main() -> int:
    stdio_v3 = _load_manifest(EXAMPLE_ROOT / "stdio-v3.yaml")
    http_v3 = _load_manifest(EXAMPLE_ROOT / "http-v3.yaml")
    port = _free_loopback_port()
    http_process = subprocess.Popen(
        [
            sys.executable,
            str(EXAMPLE_ROOT / "http_server.py"),
            "--port",
            str(port),
        ],
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    # Keep the tutorial isolated from any developer database or migration
    # state. The registry/effect evidence still exercises the complete Runtime
    # path for the lifetime of this deterministic smoke.
    runtime = Runtime.open(":memory:")
    try:
        _wait_for_loopback(port, http_process)
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="deterministic MCP stdio and loopback HTTP Tool smoke",
        )

        assert stdio_v3.stdio is not None
        stdio_spec = replace(
            stdio_v3,
            stdio=replace(
                stdio_v3.stdio,
                command=sys.executable,
                args=[str(EXAMPLE_ROOT / "stdio_server.py")],
            ),
        )
        assert http_v3.http is not None
        http_spec = replace(
            http_v3,
            http=replace(
                http_v3.http,
                url=f"http://127.0.0.1:{port}/mcp",
            ),
        )

        runtime.mcp.register_server(
            stdio_spec,
            actor="examples.mcp.run_tools_e2e",
            require_capability=False,
        )
        runtime.mcp.register_server(
            http_spec,
            actor="examples.mcp.run_tools_e2e",
            require_capability=False,
        )
        for server_id in (stdio_v3.server_id, http_v3.server_id):
            runtime.capability.grant(
                pid,
                f"mcp:{server_id}:echo",
                [CapabilityRight.READ],
                issued_by="examples.mcp.run_tools_e2e",
            )
            runtime.capability.grant(
                pid,
                f"mcp_server:{server_id}",
                [CapabilityRight.EXECUTE],
                issued_by="examples.mcp.run_tools_e2e",
            )
        runtime.capability.grant(
            pid,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="examples.mcp.run_tools_e2e",
        )
        runtime.capability.grant(
            pid,
            runtime.mcp.stdio_resource_for_argv(
                sys.executable,
                [str(EXAMPLE_ROOT / "stdio_server.py")],
            ),
            [CapabilityRight.EXECUTE],
            issued_by="examples.mcp.run_tools_e2e",
        )

        stdio_result = runtime.mcp.call_tool(
            pid,
            stdio_v3.server_id,
            "echo",
            {"text": "deterministic-stdio"},
        )
        http_result = runtime.mcp.call_tool(
            pid,
            http_v3.server_id,
            "echo",
            {"text": "deterministic-http"},
        )
        if not isinstance(stdio_result, McpComplete) or not isinstance(
            http_result, McpComplete
        ):
            raise RuntimeError("one or more governed MCP demo calls did not complete")
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "stdio": to_jsonable(stdio_result),
                    "streamable_http": to_jsonable(http_result),
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    finally:
        runtime.close()
        _stop_process(http_process)


def _load_manifest(path: Path) -> McpServerManifestV3:
    return parse_mcp_v3_manifest_yaml_text(path.read_text(encoding="utf-8"))


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as selected:
        selected.bind(("127.0.0.1", 0))
        return int(selected.getsockname()[1])


def _wait_for_loopback(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("loopback MCP demo exited before accepting connections")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as selected:
            selected.settimeout(0.1)
            if selected.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.02)
    raise RuntimeError("loopback MCP demo did not start within 15 seconds")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
