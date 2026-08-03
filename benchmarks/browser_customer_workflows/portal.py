from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_STARTUP_TIMEOUT_SECONDS = 30.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_MAX_STARTUP_LINE_CHARS = 16_384


class PlaywrightPortalHarness:
    """Own one isolated Chromium customer portal and its fixed JSON-RPC bridge."""

    evidence_mode = "browser-live"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "browser-state.json"
        self._process: subprocess.Popen[str] | None = None
        self.rpc_url = ""
        self.portal_url = ""
        self.browser_engine = ""

    def __enter__(self) -> PlaywrightPortalHarness:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Playwright portal harness is already started")
        repository_root = Path(__file__).resolve().parents[2]
        script = repository_root / "gui" / "e2e" / "customer-operations-bridge.mjs"
        if not script.is_file():
            raise RuntimeError("customer browser bridge script is missing")
        process = subprocess.Popen(
            ["node", str(script), "--state", str(self.state_path)],
            cwd=repository_root,
            env=_browser_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self._process = process
        try:
            startup = _read_startup_line(process)
        except Exception:
            self.close()
            raise
        self.rpc_url = _loopback_url(startup.get("rpc_url"), "rpc_url", path="/rpc")
        self.portal_url = _loopback_url(
            startup.get("portal_url"),
            "portal_url",
            path="/portal",
        )
        engine = startup.get("browser_engine")
        if not isinstance(engine, str) or not engine.startswith("chromium/"):
            self.close()
            raise RuntimeError("browser bridge did not report a Chromium engine")
        self.browser_engine = engine

    def state_snapshot(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("browser bridge state is unavailable or invalid") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise RuntimeError("browser bridge state has an unsupported schema")
        return payload

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def _read_startup_line(process: subprocess.Popen[str]) -> dict[str, Any]:
    assert process.stdout is not None
    assert process.stderr is not None
    output: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=2)

    def read_stdout() -> None:
        output.put(("stdout", process.stdout.readline(_MAX_STARTUP_LINE_CHARS + 1)))

    thread = threading.Thread(target=read_stdout, daemon=True)
    thread.start()
    try:
        source, line = output.get(timeout=_STARTUP_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        process.terminate()
        raise RuntimeError("browser bridge startup timed out") from exc
    if source != "stdout" or not line or len(line) > _MAX_STARTUP_LINE_CHARS:
        returncode = process.poll()
        diagnostic = process.stderr.readline(_MAX_STARTUP_LINE_CHARS + 1)
        process.terminate()
        raise RuntimeError(
            "browser bridge startup failed"
            + (f" with exit code {returncode}" if returncode is not None else "")
            + (f" ({_safe_diagnostic(diagnostic)})" if diagnostic else "")
        )
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        process.terminate()
        raise RuntimeError("browser bridge emitted invalid startup JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        process.terminate()
        raise RuntimeError("browser bridge emitted an unsupported startup schema")
    return payload


def _browser_environment() -> dict[str, str]:
    allowed = (
        "HOME",
        "PATH",
        "TMPDIR",
        "TEMP",
        "TMP",
        "PLAYWRIGHT_BROWSERS_PATH",
        "NODE_EXTRA_CA_CERTS",
    )
    environment = {
        name: os.environ[name]
        for name in allowed
        if isinstance(os.environ.get(name), str) and os.environ[name]
    }
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = "127.0.0.1,localhost"
    return environment


def _loopback_url(value: Any, name: str, *, path: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"browser bridge {name} is missing")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != path
    ):
        raise RuntimeError(f"browser bridge {name} is not a fixed loopback URL")
    return value


def _safe_diagnostic(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized[:300] if normalized else "no diagnostic"
