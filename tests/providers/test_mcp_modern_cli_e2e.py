from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.mcp import (
    InMemoryMcpCredentialBroker,
    McpOAuthHttpResponse,
)
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.runtime import Runtime
from agent_libos.substrate import LocalResourceProviderSubstrate


pytestmark = [pytest.mark.mcp, pytest.mark.mcp_transport]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "mcp"


def test_modern_cli_runs_real_stdio_and_loopback_http_without_stderr(
    tmp_path: Path,
) -> None:
    port = _free_loopback_port()
    stdio_manifest = tmp_path / "stdio-v3.yaml"
    stdio_manifest.write_text(
        (EXAMPLE_ROOT / "stdio-v3.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "command: .venv/bin/python",
            f"command: {json.dumps(sys.executable)}",
        ),
        encoding="utf-8",
    )
    http_manifest = tmp_path / "http-v3.yaml"
    http_manifest.write_text(
        (EXAMPLE_ROOT / "http-v3.yaml")
        .read_text(encoding="utf-8")
        .replace("http://127.0.0.1:8765/mcp", f"http://127.0.0.1:{port}/mcp"),
        encoding="utf-8",
    )
    server = subprocess.Popen(
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
    database = tmp_path / "runtime.sqlite"
    try:
        _wait_for_loopback(port, server)

        probe = _run_cli(
            database,
            "mcp",
            "probe",
            str(stdio_manifest),
            "--confirm-probe",
            "--reviewer",
            "cli-e2e",
            "--reason",
            "verify the unregistered full catalog",
        )
        assert probe["catalog_scope"] == "full_catalog"
        assert probe["complete"] is True
        for catalog_name in (
            "tools",
            "resources",
            "resource_templates",
            "prompts",
        ):
            assert probe[catalog_name]
        probe_json = tmp_path / "stdio-probe.json"
        probe_json.write_text(json.dumps(probe), encoding="utf-8")
        candidate = _run_cli(
            database,
            "mcp",
            "scaffold",
            "create",
            str(stdio_manifest),
            str(probe_json),
            "--confirm-scaffold",
            "--reviewer",
            "cli-e2e",
            "--reason",
            "review all discovered authority",
        )
        assert candidate["catalog_scope"] == "full_catalog"
        assert candidate["source_manifest_sha256"] == probe["manifest_sha256"]
        for catalog_name in (
            "tools",
            "resources",
            "resource_templates",
            "prompts",
        ):
            assert candidate["manifest"][catalog_name]

        _run_cli(database, "mcp", "register", str(stdio_manifest))
        _run_cli(database, "mcp", "register", str(http_manifest))

        for server_id, transport in (
            ("demo-stdio-v3", "stdio"),
            ("demo-http-v3", "streamable_http"),
        ):
            resources = _run_cli(
                database,
                "mcp",
                "resources",
                "list",
                server_id,
            )
            assert resources["has_more"] is False
            assert resources["resources"][0]["resource_id"] == "status"

            templates = _run_cli(
                database,
                "mcp",
                "resources",
                "templates",
                server_id,
            )
            assert templates["has_more"] is False
            assert templates["resource_templates"][0]["template_id"] == "greeting"

            resource = _run_cli(
                database,
                "mcp",
                "resources",
                "read",
                server_id,
                "status",
            )
            assert resource["kind"] == "complete"
            assert resource["value"]["provenance"] == "untrusted_mcp_resource"
            assert transport in resource["value"]["contents"][0]["text"]

            prompts = _run_cli(
                database,
                "mcp",
                "prompts",
                "list",
                server_id,
            )
            assert prompts["has_more"] is False
            assert prompts["prompts"][0]["prompt_id"] == "review"

            prompt = _run_cli(
                database,
                "mcp",
                "prompts",
                "get",
                server_id,
                "review",
                "--arguments-json",
                '{"subject":"the MCP contract"}',
            )
            assert prompt["kind"] == "complete"
            assert prompt["preview_only"] is True
            assert prompt["user_confirmation_required"] is True
            assert prompt["system_or_developer_injection_allowed"] is False
            assert prompt["preview_sha256"]

            completion = _run_cli(
                database,
                "mcp",
                "prompts",
                "complete",
                server_id,
                "prompt",
                "review",
                "subject",
                "cont",
            )
            assert completion["kind"] == "complete"
            assert completion["preview_only"] is True
            assert completion["user_confirmation_required"] is True
            assert completion["value"]["values"]
            assert set(completion["value"]) == {"values", "total", "has_more"}
    finally:
        _stop_process(server)


def test_oauth_foreground_login_rehydrates_in_fresh_cli_runtime_and_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_libos.api import cli as cli_module

    port = _free_loopback_port()
    resource_uri = f"http://127.0.0.1:{port}/mcp"
    issuer = "https://auth.example.test/tenant"
    redirect_uri = "http://127.0.0.1:49152/oauth/callback"
    resource_metadata = (
        f"http://127.0.0.1:{port}/.well-known/oauth-protected-resource/mcp"
    )
    authorization_metadata = (
        "https://auth.example.test/.well-known/oauth-authorization-server/tenant"
    )
    authorization_endpoint = "https://auth.example.test/authorize"
    token_endpoint = "https://auth.example.test/token"
    access_token = "cli-access-token-never-project"
    refresh_token = "cli-refresh-token-never-project"
    authorization_code = "cli-authorization-code-never-project"

    profile_file = tmp_path / "oauth-profile.json"
    profile_file.write_text(
        json.dumps(
            {
                "profile_id": "work-oauth",
                "server_id": "oauth-http-v3",
                "resource_uri": resource_uri,
                "expected_issuer": issuer,
                "redirect_uri": redirect_uri,
                "client_id": "agent-libos-cli",
                "registration_mode": "preregistered",
                "allowed_scopes": ["resources.read"],
                "default_scopes": ["resources.read"],
                "protected_resource_metadata_url": resource_metadata,
                "authorization_server_metadata_url": authorization_metadata,
                "allow_loopback_http": True,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "oauth-http-v3.yaml"
    manifest.write_text(
        (EXAMPLE_ROOT / "http-v3.yaml")
        .read_text(encoding="utf-8")
        .replace("server_id: demo-http-v3", "server_id: oauth-http-v3")
        .replace("http://127.0.0.1:8765/mcp", resource_uri)
        + "\nauth_profile_id: work-oauth\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "oauth-config.yaml"
    config_file.write_text("mcp:\n  oauth_enabled: true\n", encoding="utf-8")
    transport = _CliOAuthTransport(
        {
            ("GET", resource_metadata): [
                _oauth_json_response(
                    {
                        "resource": resource_uri,
                        "authorization_servers": [issuer],
                        "scopes_supported": ["resources.read"],
                    }
                )
            ],
            ("GET", authorization_metadata): [
                _oauth_json_response(
                    {
                        "issuer": issuer,
                        "authorization_endpoint": authorization_endpoint,
                        "token_endpoint": token_endpoint,
                        "code_challenge_methods_supported": ["S256"],
                        "response_types_supported": ["code"],
                        "grant_types_supported": [
                            "authorization_code",
                            "refresh_token",
                        ],
                        "token_endpoint_auth_methods_supported": ["none"],
                        "authorization_response_iss_parameter_supported": True,
                    }
                )
            ],
            ("POST", token_endpoint): [
                _oauth_json_response(
                    {
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "resources.read",
                        "resource": resource_uri,
                    }
                )
            ],
        }
    )
    broker = InMemoryMcpCredentialBroker()
    real_open = Runtime.open

    def open_runtime(
        target: str | Path | None = None,
        substrate: Any | None = None,
        config: AgentLibOSConfig | None = None,
        **kwargs: Any,
    ) -> Runtime:
        assert substrate is None
        workspace = tmp_path / "workspace"
        workspace.mkdir(exist_ok=True)
        selected = LocalResourceProviderSubstrate(workspace)
        selected.mcp_credential_broker = broker
        selected.mcp_oauth_transport = transport
        return real_open(target, substrate=selected, config=config, **kwargs)

    monkeypatch.setattr(cli_module.Runtime, "open", open_runtime)
    server = subprocess.Popen(
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
    database = tmp_path / "oauth-cli.sqlite"
    base = ["--config", str(config_file), "--db", str(database)]
    try:
        _wait_for_loopback(port, server)
        registered, register_stderr = _run_in_process_cli(
            cli_module,
            [
                *base,
                "mcp",
                "--oauth-profile-file",
                str(profile_file),
                "register",
                str(manifest),
            ],
        )
        assert registered["server_id"] == "oauth-http-v3"
        assert register_stderr == ""

        login_stdout = io.StringIO()
        login_stderr = io.StringIO()
        callback_input = _OAuthCallbackInput(
            login_stderr,
            redirect_uri=redirect_uri,
            issuer=issuer,
            code=authorization_code,
        )
        with (
            patch.object(sys, "stdin", callback_input),
            redirect_stdout(login_stdout),
            redirect_stderr(login_stderr),
        ):
            cli_module.cli(
                [
                    *base,
                    "mcp",
                    "auth",
                    "login",
                    "work-oauth",
                    "--profile-file",
                    str(profile_file),
                    "--callback-stdin",
                ]
            )
        login = json.loads(login_stdout.getvalue())
        assert login["status"] == "authorized"
        assert authorization_endpoint in login_stderr.getvalue()

        status, status_stderr = _run_in_process_cli(
            cli_module,
            [
                *base,
                "mcp",
                "--oauth-profile-file",
                str(profile_file),
                "auth",
                "status",
                "work-oauth",
            ],
        )
        assert status["status"] == "authorized"
        assert status_stderr == ""

        resources, resources_stderr = _run_in_process_cli(
            cli_module,
            [
                *base,
                "mcp",
                "--oauth-profile-file",
                str(profile_file),
                "resources",
                "list",
                "oauth-http-v3",
            ],
        )
        assert resources["resources"][0]["resource_id"] == "status"
        assert resources_stderr == ""

        logout, logout_stderr = _run_in_process_cli(
            cli_module,
            [
                *base,
                "mcp",
                "auth",
                "logout",
                "work-oauth",
                "--profile-file",
                str(profile_file),
            ],
        )
        assert logout["status"] == "revoked"
        assert logout_stderr == ""
        after_logout, after_logout_stderr = _run_in_process_cli(
            cli_module,
            [
                *base,
                "mcp",
                "--oauth-profile-file",
                str(profile_file),
                "auth",
                "status",
                "work-oauth",
            ],
        )
        assert after_logout["status"] == "authorization_required"
        assert after_logout_stderr == ""

        outward = "".join(
            (
                login_stdout.getvalue(),
                login_stderr.getvalue(),
                json.dumps(status),
                json.dumps(resources),
                json.dumps(logout),
                json.dumps(after_logout),
            )
        )
        for secret in (access_token, refresh_token, authorization_code):
            assert secret not in outward
            assert secret.encode() not in database.read_bytes()
    finally:
        _stop_process(server)


def test_modern_cli_oauth_facade_failures_are_structured_and_secret_free(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(":memory:")
    callback_url = (
        "https://client.invalid/callback?code=fake-sensitive-code"
        "&state=fake-sensitive-state"
    )
    try:
        with pytest.raises(ValidationError, match="OAuth profile is unavailable"):
            runtime.mcp.auth_status("missing-profile")
        with pytest.raises(ValidationError, match="OAuth profile is unavailable"):
            runtime.mcp.auth_begin(
                "missing-profile",
                scopes=("resources.read",),
            )
        with pytest.raises(
            ValidationError,
            match="authorization challenge is unavailable",
        ) as complete_error:
            runtime.mcp.auth_complete(
                "missing-challenge",
                callback_url,
            )
        assert "fake-sensitive-code" not in str(complete_error.value)
        assert "fake-sensitive-state" not in str(complete_error.value)
        with pytest.raises(ValidationError, match="OAuth profile is unavailable"):
            runtime.mcp.auth_logout("missing-profile")
    finally:
        runtime.shutdown()

    database = tmp_path / "runtime.sqlite"

    status = _run_cli_failure(database, "mcp", "auth", "status", "missing-profile")
    assert status["error"]["message"] == (
        "MCP auth_status failed; sensitive request details were omitted"
    )


def test_modern_cli_subscription_facade_failures_are_real_and_structured(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(":memory:")
    try:
        with pytest.raises(NotFound, match="Manifest v3 server not found"):
            runtime.mcp.start_subscription(
                "missing-server",
                filters=("resourcesListChanged",),
                actor="cli",
            )
        with pytest.raises(NotFound, match="subscription not found"):
            runtime.mcp.subscription_status(
                "missing-subscription",
                actor="cli",
            )
        with pytest.raises(NotFound, match="subscription not found"):
            runtime.mcp.subscription_events(
                "missing-subscription",
                after=0,
                limit=1,
                actor="cli",
            )
        with pytest.raises(NotFound, match="subscription not found"):
            runtime.mcp.stop_subscription(
                "missing-subscription",
                actor="cli",
            )
    finally:
        runtime.shutdown()

    database = tmp_path / "runtime.sqlite"
    failure = _run_cli_failure(
        database,
        "mcp",
        "subscriptions",
        "listen",
        "missing-server",
        "--filter",
        "resourcesListChanged",
        "--max-seconds",
        "0.1",
    )
    assert failure["error"]["message"] == (
        "MCP start_subscription failed; sensitive request details were omitted"
    )


def test_subscription_listen_streams_and_stops_on_one_real_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_libos.api import cli as cli_module
    from tests.runtime.test_mcp_v3_subscription_facade import (
        _SECRET,
        _SECRET_ENV,
        _SubscriptionProvider,
        _manifest,
    )

    monkeypatch.setenv(_SECRET_ENV, _SECRET)
    runtime = Runtime.open(":memory:")
    provider = _SubscriptionProvider()
    runtime.mcp.register_server(
        _manifest(),
        actor="runtime",
        require_capability=False,
    )
    runtime.mcp._modern_subscription_provider = provider  # noqa: SLF001
    lease = SimpleNamespace(
        **runtime.__dict__,
        shutdown=lambda **_kwargs: {"ok": True},
    )

    class RuntimeFactory:
        @staticmethod
        def open(*_args: Any, **_kwargs: Any) -> Any:
            return lease

    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with (
            patch.object(cli_module, "Runtime", RuntimeFactory),
            patch.object(sys, "stdin", io.StringIO()),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            cli_module.cli(
                [
                    "--db",
                    ":memory:",
                    "mcp",
                    "subscriptions",
                    "listen",
                    "modern-subscriptions",
                    "--filter",
                    "resourcesListChanged",
                    "--max-events",
                    "1",
                    "--max-seconds",
                    "2",
                ]
            )
        result = json.loads(stdout.getvalue())
        streamed = [json.loads(line) for line in stderr.getvalue().splitlines()]
        assert result["events_seen"] == 1
        assert result["stopped"]["status"] == "closed"
        assert streamed[0]["mcp_subscription"] == "opened"
        assert streamed[1]["mcp_subscription_event"]["provenance"] == (
            "untrusted_mcp_notification"
        )
        assert _SECRET not in stdout.getvalue()
        assert _SECRET not in stderr.getvalue()
        assert provider.owner == "cli"
        assert provider.listen_count == 1
        assert provider.close_count == 1
    finally:
        runtime.close()


def test_modern_cli_durable_facades_use_exact_host_signatures_and_fail_closed(
    tmp_path: Path,
) -> None:
    tasks_digest = "b" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=tasks_digest,
        )
    )
    runtime = Runtime.open(":memory:", config=config)
    human_preview_sha256 = "a" * 64
    try:
        with pytest.raises(NotFound, match="continuation not found"):
            runtime.mcp.get_continuation("missing-continuation", actor="cli")
        with pytest.raises(NotFound, match="continuation not found"):
            runtime.mcp.respond_continuation(
                "missing-continuation",
                expected_revision=0,
                responses={},
                human_request_id="missing-human-request",
                human_expected_revision=0,
                human_preview_sha256=human_preview_sha256,
                actor="cli",
            )
        with pytest.raises(NotFound, match="continuation not found"):
            runtime.mcp.cancel_continuation(
                "missing-continuation",
                expected_revision=0,
                actor="cli",
            )
        with pytest.raises(NotFound, match="remote Task not found"):
            runtime.mcp.get_remote_task(
                "missing-task",
                expected_revision=None,
                actor="cli",
            )
        with pytest.raises(NotFound, match="remote Task not found"):
            runtime.mcp.update_remote_task(
                "missing-task",
                expected_revision=0,
                responses={},
                human_request_id="missing-human-request",
                human_expected_revision=0,
                human_preview_sha256=human_preview_sha256,
                actor="cli",
            )
        with pytest.raises(NotFound, match="remote Task not found"):
            runtime.mcp.cancel_remote_task(
                "missing-task",
                expected_revision=0,
                actor="cli",
            )
    finally:
        runtime.shutdown()

    config_path = tmp_path / "tasks-config.yaml"
    config_path.write_text(
        "mcp:\n"
        "  tasks_extension_enabled: true\n"
        f"  tasks_extension_spec_sha256: {tasks_digest}\n",
        encoding="utf-8",
    )
    database = tmp_path / "runtime.sqlite"
    failures = {
        "get_continuation": _run_cli_failure(
            database,
            "mcp",
            "continuations",
            "inspect",
            "missing-continuation",
            config=config_path,
        ),
        "respond_continuation": _run_cli_failure(
            database,
            "mcp",
            "continuations",
            "respond",
            "missing-continuation",
            "--expected-revision",
            "0",
            "--human-request-id",
            "missing-human-request",
            "--human-expected-revision",
            "0",
            "--human-preview-sha256",
            human_preview_sha256,
            "--responses-json",
            "{}",
            config=config_path,
        ),
        "cancel_continuation": _run_cli_failure(
            database,
            "mcp",
            "continuations",
            "cancel",
            "missing-continuation",
            "--expected-revision",
            "0",
            config=config_path,
        ),
        "get_remote_task": _run_cli_failure(
            database,
            "mcp",
            "remote-tasks",
            "get",
            "missing-task",
            config=config_path,
        ),
        "update_remote_task": _run_cli_failure(
            database,
            "mcp",
            "remote-tasks",
            "update",
            "missing-task",
            "--expected-revision",
            "0",
            "--human-request-id",
            "missing-human-request",
            "--human-expected-revision",
            "0",
            "--human-preview-sha256",
            human_preview_sha256,
            "--responses-json",
            "{}",
            config=config_path,
        ),
        "cancel_remote_task": _run_cli_failure(
            database,
            "mcp",
            "remote-tasks",
            "cancel",
            "missing-task",
            "--expected-revision",
            "0",
            config=config_path,
        ),
    }
    for method_name, failure in failures.items():
        assert failure["error"]["message"] == (
            f"MCP {method_name} failed; sensitive request details were omitted"
        )


def test_offline_mcp_dx_does_not_open_the_selected_database(tmp_path: Path) -> None:
    database = tmp_path / "must-not-open.sqlite"

    result = _run_cli(
        database,
        "mcp",
        "validate",
        str(EXAMPLE_ROOT / "stdio-v3.yaml"),
    )

    assert result["schema_version"] == 3
    assert result["server_id"] == "demo-stdio-v3"
    assert not database.exists()


def _run_cli(database: Path, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_libos.api.cli",
            "--db",
            str(database),
            *arguments,
        ],
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""
    value = json.loads(completed.stdout)
    assert isinstance(value, dict)
    return value


def _run_cli_failure(
    database: Path,
    *arguments: str,
    config: Path | None = None,
    forbidden_output: tuple[str, ...] = (),
) -> dict[str, Any]:
    command = [sys.executable, "-m", "agent_libos.api.cli"]
    if config is not None:
        command.extend(("--config", str(config)))
    command.extend(("--db", str(database), *arguments))
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stderr == ""
    for forbidden in forbidden_output:
        assert forbidden not in completed.stdout
        assert forbidden not in completed.stderr
    value = json.loads(completed.stdout)
    assert value == {
        "schema_version": 1,
        "error": {
            "type": "ValidationError",
            "message": value["error"]["message"],
        },
    }
    return value


def _run_in_process_cli(
    cli_module: Any,
    arguments: list[str],
) -> tuple[dict[str, Any], str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        patch.object(sys, "stdin", io.StringIO()),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        cli_module.cli(arguments)
    value = json.loads(stdout.getvalue())
    assert isinstance(value, dict)
    return value, stderr.getvalue()


class _OAuthCallbackInput:
    def __init__(
        self,
        terminal: io.StringIO,
        *,
        redirect_uri: str,
        issuer: str,
        code: str,
    ) -> None:
        self._terminal = terminal
        self._redirect_uri = redirect_uri
        self._issuer = issuer
        self._code = code

    def isatty(self) -> bool:
        return False

    def readline(self, _limit: int = -1) -> str:
        authorization_url = next(
            line
            for line in self._terminal.getvalue().splitlines()
            if line.startswith("https://auth.example.test/authorize?")
        )
        query = parse_qs(urlsplit(authorization_url).query)
        return (
            f"{self._redirect_uri}?"
            + urlencode(
                {
                    "code": self._code,
                    "state": query["state"][0],
                    "iss": self._issuer,
                }
            )
            + "\n"
        )


class _CliOAuthTransport:
    def __init__(
        self,
        responses: dict[tuple[str, str], list[McpOAuthHttpResponse]],
    ) -> None:
        self._responses = {
            key: list(selected) for key, selected in responses.items()
        }

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Any,
        body: bytes | None,
        deadline: float,
        max_response_bytes: int,
    ) -> McpOAuthHttpResponse:
        del headers, body, deadline
        selected = self._responses.get((method, url))
        if not selected:
            raise AssertionError(f"unexpected OAuth request: {method} {url}")
        response = selected.pop(0)
        assert len(response.body) <= max_response_bytes
        return response


def _oauth_json_response(value: dict[str, Any]) -> McpOAuthHttpResponse:
    return McpOAuthHttpResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps(value, separators=(",", ":")).encode(),
    )


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as selected:
        selected.bind(("127.0.0.1", 0))
        return int(selected.getsockname()[1])


def _wait_for_loopback(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("loopback MCP fixture exited before accepting connections")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as selected:
            selected.settimeout(0.1)
            if selected.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.02)
    raise RuntimeError("loopback MCP fixture did not start within 15 seconds")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
