from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from agent_libos.primitives.mcp import McpPrimitive


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts" / "desktop_mcp_fixture.py"
TASKS_SCHEMA = ROOT / "tests" / "fixtures" / "mcp_sdk_v2" / "tasks_extension_schema.json"


def _target_layout(root: Path) -> tuple[str, Path, Path, Path, str]:
    if sys.platform == "darwin":
        app = root / "mac-arm64" / "Agent libOS.app"
        resources = app / "Contents" / "Resources"
        return (
            "darwin-arm64",
            app / "Contents" / "MacOS" / "Agent libOS",
            resources / "backend" / "agent-libos-gui-server",
            resources / "bin" / "deno",
            "aarch64",
        )
    if sys.platform == "win32":
        app = root / "win-unpacked"
        resources = app / "resources"
        return (
            "win32-x64",
            app / "Agent libOS.exe",
            resources / "backend" / "agent-libos-gui-server.exe",
            resources / "bin" / "deno.exe",
            "x86_64",
        )
    app = root / "linux-unpacked"
    resources = app / "resources"
    return (
        "linux-x64",
        app / "agent-libos",
        resources / "backend" / "agent-libos-gui-server",
        resources / "bin" / "deno",
        "x86_64",
    )


def _clean_runtime_environment() -> dict[str, str]:
    selected = dict(os.environ)
    for name in (
        "AGENT_LIBOS_GUI_SERVER_BIN",
        "DENO_DIR",
        "NODE_PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "UV",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
    ):
        selected.pop(name, None)
    if os.name == "nt":
        selected["Path"] = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")
        selected.pop("PATH", None)
    else:
        selected["PATH"] = "/usr/bin:/bin"
    selected["DENO_NO_UPDATE_CHECK"] = "1"
    return selected


def _require_regular(*paths: Path) -> None:
    for path in paths:
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f"desktop smoke input is unavailable: {path}")


def _run_deno(deno: Path, expected_arch: str, root: Path) -> dict[str, Any]:
    source = root / "offline-jit.ts"
    source.write_text(
        "const marker: string = 'desktop-deno-jit';\n"
        "console.log(JSON.stringify({marker, version: Deno.version.deno, arch: Deno.build.arch}));\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(deno), "run", "--cached-only", "--no-config", "--no-lock", str(source)],
        cwd=root,
        env=_clean_runtime_environment(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    value = json.loads(result.stdout)
    if value != {"marker": "desktop-deno-jit", "version": "2.9.5", "arch": expected_arch}:
        raise RuntimeError(f"bundled Deno identity/JIT smoke failed: {value!r}")
    return value


def _read_ready_line(process: subprocess.Popen[str], timeout_s: float) -> dict[str, Any]:
    selected: queue.Queue[str] = queue.Queue(maxsize=1)

    def read() -> None:
        if process.stdout is not None:
            selected.put(process.stdout.readline())

    threading.Thread(target=read, daemon=True).start()
    try:
        line = selected.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise RuntimeError("frozen GUI backend did not publish a ready receipt") from exc
    if not line:
        error = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"frozen GUI backend exited before ready: {error[-4000:]}")
    value = json.loads(line)
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("url"), str)
        or not isinstance(value.get("token"), str)
    ):
        raise RuntimeError("frozen GUI backend ready receipt is invalid")
    return value


def _request(
    ready: dict[str, Any],
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = urlsplit(ready["url"])
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=15)
    raw = None if body is None else json.dumps(body, separators=(",", ":"))
    headers = {"Authorization": f"Bearer {ready['token']}"}
    if raw is not None:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=raw, headers=headers)
    response = connection.getresponse()
    payload = response.read().decode("utf-8")
    connection.close()
    value = json.loads(payload) if payload else {}
    if response.status < 200 or response.status >= 300:
        raise RuntimeError(f"{method} {path} failed with {response.status}: {value!r}")
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path} returned a non-object")
    return value


def _manifest(python: Path, task_state: Path) -> dict[str, Any]:
    tasks_sha256 = hashlib.sha256(TASKS_SCHEMA.read_bytes()).hexdigest()
    # Keep a virtual-environment launcher's lexical path.  The Runtime pins
    # the resolved executable identity separately, while Python itself needs
    # the lexical path to discover the adjacent pyvenv.cfg and MCP SDK.
    command = str(python.absolute())
    arguments = [str(FIXTURE.resolve()), "--task-state-file", str(task_state.resolve())]
    return {
        "schema_version": 3,
        "server_id": "desktop-frozen-smoke",
        "transport": "stdio",
        "protocol_mode": "2026-07-28",
        "stdio": {"command": command, "args": arguments},
        "tools": [
            {
                "tool_id": "echo",
                "mcp_name": "desktop_echo",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "rollback_status": "not_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
            {
                "tool_id": "review",
                "mcp_name": "desktop_review_mrtr",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "rollback_status": "not_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": {
                    "type": "object",
                    "properties": {"document": {"type": "string"}},
                    "required": ["document"],
                    "additionalProperties": False,
                },
            },
            {
                "tool_id": "task",
                "mcp_name": "desktop_begin_task",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "rollback_status": "not_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": {
                    "type": "object",
                    "properties": {"mode": {"const": "input"}},
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            },
        ],
        "resources": [
            {
                "resource_id": "status",
                "remote_uri": "desktop://status",
                "right": "read",
                "information_flow": True,
                "model_visible": False,
                "mime_types": ["text/plain"],
            }
        ],
        "resource_templates": [
            {
                "template_id": "greeting",
                "remote_uri_template": "desktop://greeting/{name}",
                "variables": ["name"],
                "right": "read",
                "information_flow": True,
                "model_visible": False,
                "mime_types": ["text/plain"],
            }
        ],
        "prompts": [
            {
                "prompt_id": "review-prompt",
                "mcp_name": "desktop_review",
                "argument_names": ["subject"],
            }
        ],
        "subscriptions": [],
        "tasks_extension": {
            "extension_id": "io.modelcontextprotocol/tasks",
            "spec_sha256": tasks_sha256,
        },
        "timeout_s": 10,
        "max_request_bytes": 65_536,
        "max_response_bytes": 1_048_576,
    }


def _grant(ready: dict[str, Any], pid: str, resource: str, right: str) -> None:
    _request(
        ready,
        "POST",
        "/api/capabilities/grant",
        {
            "subject": pid,
            "resource": resource,
            "rights": [right],
            "confirmed": True,
        },
    )


def _responses(pending: dict[str, Any]) -> dict[str, Any]:
    requests = pending.get("input_requests")
    if not isinstance(requests, list) or len(requests) != 1:
        raise RuntimeError("frozen MCP input request projection is invalid")
    selected = requests[0]
    if not isinstance(selected, dict) or not isinstance(selected.get("request_id"), str):
        raise RuntimeError("frozen MCP input request identifier is invalid")
    local_id = selected["request_id"]
    return {local_id: {"action": "accept", "content": {"approved": True}}}


def _exercise_frozen_mcp(backend: Path, root: Path) -> dict[str, Any]:
    database = root / "frozen-backend.sqlite"
    profiles = root / "llm-profiles.json"
    tasks_sha256 = hashlib.sha256(TASKS_SCHEMA.read_bytes()).hexdigest()
    config = root / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "mcp": {
                    "remote_task_poll_min_interval_s": 0.001,
                    "tasks_extension_enabled": True,
                    "tasks_extension_spec_sha256": tasks_sha256,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            str(backend),
            "--config",
            str(config),
            "--db",
            str(database),
            "--port",
            "0",
            "--llm-profiles-file",
            str(profiles),
            "--no-auto-run",
        ],
        cwd=root,
        env=_clean_runtime_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _read_ready_line(process, 30)
        health = _request(ready, "GET", "/api/health")
        if health.get("ok") is not True:
            raise RuntimeError("frozen GUI backend health check failed")
        manifest = _manifest(Path(sys.executable), root / "task-state.json")
        _request(
            ready,
            "POST",
            "/api/mcp/register",
            {"manifest_text": json.dumps(manifest), "confirmed": True},
        )
        spawned = _request(
            ready,
            "POST",
            "/api/processes",
            {"goal": "frozen desktop MCP smoke", "auto_run": False},
        )
        pid = str(spawned["pid"])
        command = manifest["stdio"]["command"]
        arguments = manifest["stdio"]["args"]
        for resource, right in (
            ("process:spawn", "write"),
            ("mcp_server:desktop-frozen-smoke", "execute"),
            (McpPrimitive.stdio_resource_for_argv(command, arguments), "execute"),
            ("mcp:desktop-frozen-smoke:echo", "read"),
            ("mcp:desktop-frozen-smoke:review", "read"),
            ("mcp:desktop-frozen-smoke:task", "read"),
            ("human:owner", "write"),
        ):
            _grant(ready, pid, resource, right)

        resource = _request(
            ready,
            "POST",
            "/api/mcp/desktop-frozen-smoke/resources/read",
            {"resource_id": "status"},
        )
        template = _request(
            ready,
            "POST",
            "/api/mcp/desktop-frozen-smoke/resources/read",
            {"resource_id": "greeting", "variables": {"name": "Ada"}},
        )
        prompt_preview = _request(
            ready,
            "POST",
            "/api/mcp/desktop-frozen-smoke/prompts/get",
            {"prompt_id": "review-prompt", "arguments": {"subject": "release"}},
        )
        prompt = _request(
            ready,
            "POST",
            "/api/mcp/desktop-frozen-smoke/prompts/get",
            {
                "prompt_id": "review-prompt",
                "arguments": {"subject": "release"},
                "confirmed": True,
                "expected_preview_sha256": prompt_preview["preview_sha256"],
            },
        )
        completion = _request(
            ready,
            "POST",
            "/api/mcp/desktop-frozen-smoke/completion",
            {
                "reference_type": "prompt",
                "reference_id": "review-prompt",
                "argument": {"name": "subject", "value": "release"},
                "context": {},
            },
        )
        tool = _request(
            ready,
            "POST",
            "/api/mcp/desktop-frozen-smoke/call",
            {
                "pid": pid,
                "tool_id": "echo",
                "arguments": {"text": "ready"},
                "confirmed": True,
            },
        )
        pending = _request(
            ready,
            "POST",
            "/api/mcp/desktop-frozen-smoke/call",
            {
                "pid": pid,
                "tool_id": "review",
                "arguments": {"document": "release"},
                "confirmed": True,
            },
        )
        continued = _request(
            ready,
            "POST",
            f"/api/mcp/continuations/{pending['continuation_id']}/respond",
            {
                "expected_revision": pending["revision"],
                "responses": _responses(pending),
                "human_request_id": pending["human_request_id"],
                "human_expected_revision": pending["human_revision"],
                "human_preview_sha256": pending["human_preview_sha256"],
                "confirmed": True,
            },
        )
        task = _request(
            ready,
            "POST",
            "/api/mcp/desktop-frozen-smoke/call",
            {
                "pid": pid,
                "tool_id": "task",
                "arguments": {"mode": "input"},
                "confirmed": True,
            },
        )
        _grant(ready, pid, f"mcp_task:{task['task_ref']}", "read")
        _grant(ready, pid, f"mcp_task:{task['task_ref']}", "write")
        task_pending = _request(
            ready,
            "POST",
            f"/api/mcp/remote-tasks/{task['task_ref']}/get",
            {"expected_revision": task["revision"]},
        )
        task_updated = _request(
            ready,
            "POST",
            f"/api/mcp/remote-tasks/{task['task_ref']}/update",
            {
                "expected_revision": task_pending["revision"],
                "responses": _responses(task_pending),
                "human_request_id": task_pending["human_request_id"],
                "human_expected_revision": task_pending["human_revision"],
                "human_preview_sha256": task_pending["human_preview_sha256"],
                "confirmed": True,
            },
        )
        task_complete = _request(
            ready,
            "POST",
            f"/api/mcp/remote-tasks/{task['task_ref']}/get",
            {"expected_revision": task_updated["revision"]},
        )
        kinds = {
            "resource": resource.get("kind"),
            "template": template.get("kind"),
            "prompt": prompt.get("kind"),
            "completion": completion.get("kind"),
            "tool": tool.get("kind"),
            "continuation": continued.get("kind"),
            "task": task_complete.get("status"),
        }
        expected = {
            "resource": "complete",
            "template": "complete",
            "prompt": "complete",
            "completion": "complete",
            "tool": "complete",
            "continuation": "complete",
            "task": "completed",
        }
        if kinds != expected:
            raise RuntimeError(f"frozen MCP surface smoke failed: {kinds!r}")
        shutdown = _request(ready, "POST", "/api/shutdown", {})
        if shutdown != {"ok": True, "status": "stopped"}:
            raise RuntimeError("frozen GUI backend shutdown was incomplete")
        process.wait(timeout=15)
        if process.returncode != 0:
            raise RuntimeError("frozen GUI backend exited unsuccessfully")
        private_values = (b"desktop-frozen-mrtr-private-state", b"desktop-private-task-")
        for selected in database.parent.glob(f"{database.name}*"):
            if selected.is_file() and any(value in selected.read_bytes() for value in private_values):
                raise RuntimeError("frozen MCP private state leaked into SQLite")
        return kinds
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def _electron_smoke(
    executable: Path,
    backend: Path | None,
    root: Path,
    *,
    window: bool,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    user_data = root / "electron-user-data"
    user_data.mkdir(mode=0o700)
    observed_database: str | None = None
    for attempt in (1, 2):
        log_path = root / f"electron-{attempt}.jsonl"
        env = _clean_runtime_environment()
        env.update(
            {
                "AGENT_LIBOS_GUI_SERVER_BIN": str(root / "forbidden-backend-override"),
                "AGENT_LIBOS_GUI_SMOKE": "1",
                "AGENT_LIBOS_GUI_SMOKE_LOG": str(log_path),
                "AGENT_LIBOS_GUI_SMOKE_PERSIST": "1",
                "AGENT_LIBOS_GUI_SMOKE_USER_DATA": str(user_data),
            }
        )
        env.update(extra_env or {})
        if window:
            env["AGENT_LIBOS_GUI_SMOKE_WINDOW"] = "1"
        subprocess.run(
            [str(executable)],
            cwd=root,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90,
        )
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        stages = {entry["stage"]: entry for entry in entries}
        required = {"server.command", "window.server.ready", "server.stop.completed", "smoke.exiting"}
        if window:
            required.update({"window.loaded", "window.renderer.checked", "smoke.complete"})
        else:
            required.add("server.health.checked")
        if not required.issubset(stages):
            raise RuntimeError(f"packaged Electron smoke stages are incomplete: {sorted(stages)}")
        command = Path(str(stages["server.command"]["command"])).resolve()
        if backend is not None:
            if command != backend.resolve():
                raise RuntimeError("packaged Electron accepted a backend override")
        elif command.name not in {"agent-libos-gui-server", "agent-libos-gui-server.exe"} or (
            command.parent.name != "backend"
        ):
            raise RuntimeError("packaged Electron did not use its bundled backend")
        ready = stages["window.server.ready"]
        database = str(ready.get("db"))
        if observed_database is None:
            observed_database = database
        elif database != observed_database:
            raise RuntimeError("packaged Electron did not reopen the same database")
        stopped = stages["server.stop.completed"]
        if stopped.get("gracefulAcknowledged") is not True or stopped.get("forced") is not False:
            raise RuntimeError("packaged Electron did not stop the backend gracefully")
        if stages["smoke.exiting"].get("code") != 0:
            raise RuntimeError("packaged Electron smoke exited unsuccessfully")
        if window:
            renderer = stages["window.renderer.checked"]
            if renderer != {
                "stage": "window.renderer.checked",
                "preloadReady": True,
                "apiReady": True,
                "origin": "agent-libos://app",
                "originReady": True,
            }:
                raise RuntimeError(f"packaged renderer/preload smoke failed: {renderer!r}")
    database_path = user_data / "runtime" / "agent-libos.sqlite"
    if not database_path.is_file() or database_path.stat().st_size == 0:
        raise RuntimeError("packaged Electron persistent database was not created")
    return {"database": str(database_path), "reopens": 2, "window": window}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke one native unpacked desktop bundle.")
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args(argv)
    root = args.artifact_dir.expanduser().resolve()
    target, electron, backend, deno, deno_arch = _target_layout(root)
    _require_regular(electron, backend, deno, FIXTURE, TASKS_SCHEMA)
    with tempfile.TemporaryDirectory(prefix="agent-libos-desktop-smoke-") as temporary:
        smoke_root = Path(temporary)
        deno_result = _run_deno(deno, deno_arch, smoke_root)
        mcp_result = _exercise_frozen_mcp(backend, smoke_root)
        electron_result = _electron_smoke(
            electron,
            backend,
            smoke_root,
            window=not args.no_window,
        )
    print(
        json.dumps(
            {
                "deno": deno_result,
                "electron": electron_result,
                "mcp": mcp_result,
                "target": target,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
