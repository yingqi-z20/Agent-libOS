from __future__ import annotations

import importlib
import importlib.metadata
import http.client
import json
import os
import shutil
import socket
import ssl
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit


_INSTALLED_MCP_SERVER = r'''from __future__ import annotations

import asyncio
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.subscriptions import InMemorySubscriptionBus, ResourceUpdated
from mcp.types import Completion


RESOURCE_URI = "artifact://status"
RESOURCE_TEMPLATE_URI = "artifact://greeting/{name}"


class FixtureSubscriptionBus(InMemorySubscriptionBus):
    def subscribe(self, listener):
        unsubscribe = super().subscribe(listener)

        async def publish() -> None:
            await asyncio.sleep(0.1)
            await self.publish(ResourceUpdated(uri=RESOURCE_URI))

        asyncio.get_running_loop().create_task(publish())
        return unsubscribe


server = MCPServer(
    "agent-libos-installed-artifact-smoke",
    version="2.0.0",
    subscriptions=FixtureSubscriptionBus(),
    log_level="ERROR",
)


@server.resource(
    RESOURCE_URI,
    name="artifact-status",
    description="Deterministic installed-package Resource smoke.",
    mime_type="text/plain",
)
def status() -> str:
    return "installed MCP Resource is reachable"


@server.resource(
    RESOURCE_TEMPLATE_URI,
    name="artifact-greeting",
    description="Deterministic installed-package Resource Template smoke.",
    mime_type="text/plain",
)
def greeting(name: str) -> str:
    return f"installed MCP Resource Template says hello to {name}"


@server.prompt(
    name="artifact_review",
    description="Deterministic installed-package Prompt smoke.",
)
def review(topic: str) -> str:
    return f"Review installed artifact {topic}."


@server.completion()
async def complete(_reference, argument, _context):
    return Completion(
        values=[f"{argument.value}-installed-completion"],
        total=1,
        hasMore=False,
    )


@server.tool(
    name="artifact_echo",
    description="Deterministic installed-package Tool smoke.",
)
def echo(text: str) -> dict[str, str]:
    return {"echo": text, "source": "installed-mcp-sdk-2.0.0"}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--http-port":
        server.run(
            "streamable-http",
            host="127.0.0.1",
            port=int(sys.argv[2]),
            streamable_http_path="/mcp",
        )
    elif len(sys.argv) == 1:
        server.run("stdio")
    else:
        raise SystemExit("invalid installed MCP fixture arguments")
'''


_INSTALLED_OAUTH_TLS_SERVER = r'''from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode

import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route


ACCESS_TOKEN = "agent-libos-oauth-tls-access-token"
AUTHORIZATION_CODE = "agent-libos-oauth-tls-code"
RESOURCE_URI = "fixture://oauth/status"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


class OAuthTlsFixture:
    def __init__(self, *, port: int, evidence_path: Path) -> None:
        self.origin = f"https://localhost:{port}"
        self.resource = f"{self.origin}/mcp"
        self.redirect_uri = "http://127.0.0.1:8765/oauth/callback"
        self.client_id = "agent-libos-oauth-tls-client"
        self.evidence_path = evidence_path
        self._evidence: dict[str, bool] = {}
        self._authorization: dict[str, str] | None = None
        server = MCPServer(
            "agent-libos-installed-oauth-tls-smoke",
            version="2.0.0",
            log_level="ERROR",
        )

        @server.resource(
            RESOURCE_URI,
            name="oauth-status",
            description="Installed-package OAuth TLS smoke Resource.",
            mime_type="text/plain",
        )
        def oauth_status() -> str:
            return "agent-libos OAuth TLS Runtime path authorized"

        app = server.streamable_http_app(
            streamable_http_path="/mcp",
            json_response=True,
            stateless_http=True,
            host="localhost",
        )
        app.router.routes[0:0] = [
            Route(
                "/.well-known/oauth-protected-resource/mcp",
                self.protected_resource_metadata,
                methods=["GET"],
            ),
            Route(
                "/.well-known/oauth-authorization-server",
                self.authorization_server_metadata,
                methods=["GET"],
            ),
            Route("/authorize", self.authorize, methods=["GET"]),
            Route("/token", self.token, methods=["POST"]),
        ]
        self.app = BearerProtectedApp(app, fixture=self)

    def observe(self, **items: bool) -> None:
        self._evidence.update(items)
        temporary = self.evidence_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._evidence, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.evidence_path)

    async def protected_resource_metadata(self, _request: Request) -> Response:
        self.observe(protected_resource_metadata_served=True)
        return JSONResponse(
            {
                "resource": self.resource,
                "authorization_servers": [self.origin],
                "scopes_supported": ["mcp.read"],
            }
        )

    async def authorization_server_metadata(self, _request: Request) -> Response:
        self.observe(authorization_server_metadata_served=True)
        return JSONResponse(
            {
                "issuer": self.origin,
                "authorization_endpoint": f"{self.origin}/authorize",
                "token_endpoint": f"{self.origin}/token",
                "code_challenge_methods_supported": ["S256"],
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code"],
                "token_endpoint_auth_methods_supported": ["none"],
                "authorization_response_iss_parameter_supported": True,
                "scopes_supported": ["mcp.read"],
            }
        )

    async def authorize(self, request: Request) -> Response:
        query = request.query_params
        required = {
            "response_type",
            "client_id",
            "redirect_uri",
            "resource",
            "scope",
            "state",
            "code_challenge",
            "code_challenge_method",
        }
        if not required.issubset(query):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        valid = {
            "response_type": query["response_type"] == "code",
            "client_id": query["client_id"] == self.client_id,
            "redirect_uri": query["redirect_uri"] == self.redirect_uri,
            "resource": query["resource"] == self.resource,
            "scope": query["scope"] == "mcp.read",
            "state": bool(query["state"]),
            "challenge": len(query["code_challenge"]) == 43,
            "challenge_method": query["code_challenge_method"] == "S256",
        }
        if not all(valid.values()):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        self._authorization = {
            "state": query["state"],
            "code_challenge": query["code_challenge"],
        }
        self.observe(
            authorization_client_id_pinned=valid["client_id"],
            authorization_redirect_pinned=valid["redirect_uri"],
            authorization_resource_pinned=valid["resource"],
            authorization_scope_pinned=valid["scope"],
            authorization_state_present=valid["state"],
            authorization_pkce_s256=(
                valid["challenge"] and valid["challenge_method"]
            ),
        )
        location = f"{self.redirect_uri}?{urlencode({'code': AUTHORIZATION_CODE, 'state': query['state'], 'iss': self.origin})}"
        return RedirectResponse(location, status_code=302)

    async def token(self, request: Request) -> Response:
        authorization = self._authorization
        if authorization is None:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        try:
            form = parse_qs(
                (await request.body()).decode("ascii"),
                strict_parsing=True,
            )
            code_verifier = form["code_verifier"][0]
        except (KeyError, UnicodeDecodeError, ValueError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        derived_challenge = _b64url(
            hashlib.sha256(code_verifier.encode("ascii")).digest()
        )
        valid = {
            "grant_type": form.get("grant_type") == ["authorization_code"],
            "code": form.get("code") == [AUTHORIZATION_CODE],
            "client_id": form.get("client_id") == [self.client_id],
            "redirect_uri": form.get("redirect_uri") == [self.redirect_uri],
            "resource": form.get("resource") == [self.resource],
            "pkce": derived_challenge == authorization["code_challenge"],
        }
        if not all(valid.values()):
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        self.observe(
            token_code_bound=valid["code"],
            token_client_id_pinned=valid["client_id"],
            token_redirect_pinned=valid["redirect_uri"],
            token_resource_pinned=valid["resource"],
            token_pkce_verified=valid["pkce"],
        )
        return JSONResponse(
            {
                "access_token": ACCESS_TOKEN,
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "mcp.read",
            }
        )


class BearerProtectedApp:
    def __init__(self, app: Any, *, fixture: OAuthTlsFixture) -> None:
        self._app = app
        self._fixture = fixture

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/mcp":
            headers = {
                bytes(key).lower(): bytes(value)
                for key, value in scope.get("headers", ())
            }
            expected = f"Bearer {ACCESS_TOKEN}".encode("ascii")
            if headers.get(b"authorization") != expected:
                response = JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": (
                            'Bearer resource_metadata="'
                            f"{self._fixture.origin}/.well-known/"
                            'oauth-protected-resource/mcp", scope="mcp.read"'
                        )
                    },
                )
                await response(scope, receive, send)
                return
            self._fixture.observe(mcp_bearer_verified=True)
        await self._app(scope, receive, send)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cert", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    fixture = OAuthTlsFixture(port=args.port, evidence_path=args.evidence)
    uvicorn.run(
        fixture.app,
        host="127.0.0.1",
        port=args.port,
        ssl_certfile=str(args.cert),
        ssl_keyfile=str(args.key),
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
'''


REQUIRED_DISTRIBUTIONS: tuple[tuple[str, str, int], ...] = (
    ("anyio", "anyio", 4),
    ("httpcore2", "httpcore2", 2),
    ("httpx2", "httpx2", 2),
    ("keyring", "keyring", 25),
    ("mcp", "mcp", 2),
    ("opentelemetry-api", "opentelemetry", 1),
)

def _major_version(distribution: str) -> int:
    raw = importlib.metadata.version(distribution)
    major, separator, _remainder = raw.partition(".")
    if not separator or not major.isdigit():
        raise RuntimeError(
            f"{distribution} has an unrecognizable installed version: {raw!r}"
        )
    return int(major)


def _runtime_smoke() -> dict[str, object]:
    """Exercise only the installed wheel/sdist plus its frozen MCP extra."""

    from agent_libos import Runtime
    from agent_libos.mcp.types import (
        McpComplete,
        McpCompletionResult,
        McpSubscriptionStatus,
        McpTextContent,
    )
    from agent_libos.models import CapabilityRight
    import anyio

    with tempfile.TemporaryDirectory(prefix="agent-libos-installed-mcp-") as root:
        root_path = Path(root)
        server_path = root_path / "installed_mcp_server.py"
        server_path.write_text(_INSTALLED_MCP_SERVER, encoding="utf-8")
        arguments = [str(server_path)]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
            reserved.bind(("127.0.0.1", 0))
            http_port = int(reserved.getsockname()[1])
        http_process = subprocess.Popen(
            [sys.executable, str(server_path), "--http-port", str(http_port)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        http_deadline = time.monotonic() + 10
        while time.monotonic() < http_deadline:
            if http_process.poll() is not None:
                error = (http_process.stderr.read() if http_process.stderr else "")
                raise RuntimeError(f"installed MCP HTTP fixture exited: {error[-2000:]}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.1)
                if probe.connect_ex(("127.0.0.1", http_port)) == 0:
                    break
            time.sleep(0.02)
        else:
            raise RuntimeError("installed MCP HTTP fixture did not become ready")
        runtime = Runtime.open(":memory:")
        try:
            runtime.mcp.register_server(
                {
                    "schema_version": 3,
                    "server_id": "installed-artifact",
                    "transport": "stdio",
                    "protocol_mode": "2026-07-28",
                    "stdio": {"command": sys.executable, "args": arguments},
                    "tools": [
                        {
                            "tool_id": "echo",
                            "mcp_name": "artifact_echo",
                            "right": "read",
                            "rollback_class": "no_rollback_required",
                            "state_mutation": False,
                            "information_flow": True,
                            "input_schema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                                "additionalProperties": False,
                            },
                        }
                    ],
                    "resources": [
                        {
                            "resource_id": "status",
                            "remote_uri": "artifact://status",
                            "right": "read",
                            "information_flow": True,
                            "model_visible": False,
                            "mime_types": ["text/plain"],
                        }
                    ],
                    "resource_templates": [
                        {
                            "template_id": "greeting",
                            "remote_uri_template": "artifact://greeting/{name}",
                            "variables": ["name"],
                            "right": "read",
                            "information_flow": True,
                            "model_visible": False,
                            "mime_types": ["text/plain"],
                        }
                    ],
                    "prompts": [
                        {
                            "prompt_id": "review",
                            "mcp_name": "artifact_review",
                            "argument_names": ["topic"],
                        }
                    ],
                    "subscriptions": ["resourceSubscriptions"],
                    "timeout_s": 10,
                    "max_request_bytes": 65_536,
                    "max_response_bytes": 1_048_576,
                },
                actor="installed-artifact-smoke",
                require_capability=False,
            )
            runtime.mcp.register_server(
                {
                    "schema_version": 3,
                    "server_id": "installed-artifact-http",
                    "transport": "streamable_http",
                    "protocol_mode": "2026-07-28",
                    "http": {"url": f"http://127.0.0.1:{http_port}/mcp"},
                    "resources": [
                        {
                            "resource_id": "status",
                            "remote_uri": "artifact://status",
                            "right": "read",
                            "information_flow": True,
                            "model_visible": False,
                            "mime_types": ["text/plain"],
                        }
                    ],
                    "subscriptions": [],
                    "timeout_s": 10,
                    "max_request_bytes": 65_536,
                    "max_response_bytes": 1_048_576,
                },
                actor="installed-artifact-smoke",
                require_capability=False,
            )

            resources = runtime.mcp.list_resources(
                "installed-artifact",
                actor="installed-artifact-smoke",
            )
            if [item.resource_id for item in resources.items] != ["status"]:
                raise RuntimeError("installed MCP Resource catalog did not round-trip")
            templates = runtime.mcp.list_resource_templates(
                "installed-artifact",
                actor="installed-artifact-smoke",
            )
            if [item.template_id for item in templates.items] != ["greeting"]:
                raise RuntimeError(
                    "installed MCP Resource Template catalog did not round-trip"
                )
            resource = runtime.mcp.read_resource(
                "installed-artifact",
                "status",
                actor="installed-artifact-smoke",
            )
            if (
                not isinstance(resource, McpComplete)
                or resource.value is None
                or len(resource.value.contents) != 1
                or not isinstance(resource.value.contents[0], McpTextContent)
                or "installed MCP Resource is reachable"
                not in resource.value.contents[0].text
            ):
                raise RuntimeError("installed MCP Resource read failed")
            template_resource = runtime.mcp.read_resource(
                "installed-artifact",
                "greeting",
                variables={"name": "Ada Lovelace"},
                actor="installed-artifact-smoke",
            )
            if (
                not isinstance(template_resource, McpComplete)
                or template_resource.value is None
                or len(template_resource.value.contents) != 1
                or not isinstance(
                    template_resource.value.contents[0], McpTextContent
                )
                or "hello to Ada Lovelace"
                not in template_resource.value.contents[0].text
            ):
                raise RuntimeError("installed MCP Resource Template read failed")
            http_resource = runtime.mcp.read_resource(
                "installed-artifact-http",
                "status",
                actor="installed-artifact-smoke",
            )
            if (
                not isinstance(http_resource, McpComplete)
                or http_resource.value is None
                or "installed MCP Resource is reachable" not in repr(http_resource)
            ):
                raise RuntimeError("installed MCP loopback HTTP path failed")

            prompts = runtime.mcp.list_prompts(
                "installed-artifact",
                actor="installed-artifact-smoke",
            )
            if [item.prompt_id for item in prompts.items] != ["review"]:
                raise RuntimeError("installed MCP Prompt catalog did not round-trip")
            prompt = runtime.mcp.get_prompt(
                "installed-artifact",
                "review",
                arguments={"topic": "Runtime boundaries"},
                actor="installed-artifact-smoke",
            )
            if (
                not isinstance(prompt, McpComplete)
                or prompt.value is None
                or not prompt.value.messages
                or not isinstance(prompt.value.messages[0].content, McpTextContent)
                or "Runtime boundaries" not in prompt.value.messages[0].content.text
            ):
                raise RuntimeError("installed MCP Prompt get failed")

            completion = runtime.mcp.complete_prompt(
                "installed-artifact",
                "prompt",
                "review",
                {"name": "topic", "value": "runtime"},
                actor="installed-artifact-smoke",
            )
            if completion != McpComplete(
                value=McpCompletionResult(
                    values=("runtime-installed-completion",),
                    total=1,
                    has_more=False,
                )
            ):
                raise RuntimeError("installed MCP Completion failed")

            subscription = runtime.mcp.start_subscription(
                "installed-artifact",
                filters=("resourceSubscriptions",),
                actor="installed-artifact-smoke",
            )
            if subscription.status is not McpSubscriptionStatus.ACTIVE:
                raise RuntimeError("installed MCP subscription did not become active")
            events = ()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                events = runtime.mcp.subscription_events(
                    subscription.subscription_id,
                    after=0,
                    limit=10,
                    actor="installed-artifact-smoke",
                )
                if events:
                    break
                time.sleep(0.02)
            if not events or events[0].event_type != "resourceUpdated":
                raise RuntimeError("installed MCP subscription event was not delivered")
            stopped = runtime.mcp.stop_subscription(
                subscription.subscription_id,
                actor="installed-artifact-smoke",
            )
            if stopped.status is not McpSubscriptionStatus.CLOSED:
                raise RuntimeError("installed MCP subscription did not close cleanly")
            if anyio.run(runtime._mcp_connection_supervisor.snapshot):
                raise RuntimeError("installed MCP subscription leaked a connection")

            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="installed MCP v3 Tool smoke",
            )
            runtime.capability.grant(
                pid,
                "mcp:installed-artifact:echo",
                [CapabilityRight.READ],
                issued_by="installed-artifact-smoke",
            )
            runtime.capability.grant(
                pid,
                "mcp_server:installed-artifact",
                [CapabilityRight.EXECUTE],
                issued_by="installed-artifact-smoke",
            )
            runtime.capability.grant(
                pid,
                "process:spawn",
                [CapabilityRight.WRITE],
                issued_by="installed-artifact-smoke",
            )
            runtime.capability.grant(
                pid,
                runtime.mcp.stdio_resource_for_argv(sys.executable, arguments),
                [CapabilityRight.EXECUTE],
                issued_by="installed-artifact-smoke",
            )
            tool = runtime.mcp.call_tool(
                pid,
                "installed-artifact",
                "echo",
                {"text": "artifact-ok"},
            )
            if not isinstance(tool, McpComplete) or "artifact-ok" not in repr(tool):
                raise RuntimeError("installed MCP v3 Tool call failed")

            audit_actions = {record.action for record in runtime.audit.trace()}
            required_actions = {
                "primitive.mcp.resources.list",
                "primitive.mcp.resource_templates.list",
                "primitive.mcp.resources.read",
                "primitive.mcp.prompts.list",
                "primitive.mcp.prompts.get",
                "primitive.mcp.completion.complete",
                "primitive.mcp.subscriptions.start",
                "primitive.mcp.subscriptions.events",
                "primitive.mcp.subscriptions.stop",
                "primitive.mcp.call",
            }
            if not required_actions.issubset(audit_actions):
                raise RuntimeError("installed MCP Runtime protected evidence is incomplete")
            return {
                "protocol_revision": "2026-07-28",
                "resource": "status",
                "resource_template": "greeting/name",
                "http_resource": "status",
                "prompt": "review",
                "completion": "review/topic",
                "subscription": "resourceSubscriptions",
                "tool": "echo",
                "protected_actions": sorted(required_actions),
            }
        finally:
            runtime.close()
            http_process.terminate()
            try:
                http_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                http_process.kill()
                http_process.wait(timeout=5)
            if http_process.stderr is not None:
                error = http_process.stderr.read()
                if error.strip():
                    raise RuntimeError(
                        "installed MCP HTTP fixture emitted stderr: " + error[-2000:]
                    )


def _migration_smoke() -> dict[str, object]:
    """Exercise the installed offline v6-to-v7 planner, apply, and reopen."""

    from agent_libos.storage import SQLiteStore
    from agent_libos.storage.mcp_v7_migration import (
        apply_store_v7_migration,
        plan_store_v7_migration,
    )
    from agent_libos.storage.v7_schema_contract import V7_TABLES

    with tempfile.TemporaryDirectory(
        prefix="agent-libos-installed-mcp-v7-migration-"
    ) as root:
        source = Path(root) / "source.sqlite"
        backup = Path(root) / "source-v6.backup.sqlite"
        fresh = SQLiteStore(source)
        fresh.close()
        connection = sqlite3.connect(source)
        try:
            for table in sorted(V7_TABLES):
                connection.execute(f'DROP TABLE "{table}"')
            changed = connection.execute(
                "UPDATE runtime_schema SET schema_version = 6 "
                "WHERE singleton = 1 AND schema_version = 7"
            )
            if changed.rowcount != 1:
                raise RuntimeError("installed MCP migration fixture was not schema v7")
            connection.commit()
        finally:
            connection.close()
        os.chmod(source, 0o600)
        shutil.copyfile(source, backup)
        os.chmod(backup, 0o600)

        plan = plan_store_v7_migration(source, sqlite_backup=backup)
        if plan.from_schema_version != 6 or plan.to_schema_version != 7:
            raise RuntimeError("installed MCP migration plan version mismatch")
        result = apply_store_v7_migration(
            source,
            expected_plan_sha256=plan.plan_sha256,
            sqlite_backup=backup,
        )
        if not result.applied or result.already_applied:
            raise RuntimeError("installed MCP v6-to-v7 migration did not apply")
        reopened = SQLiteStore(source)
        try:
            marker = reopened.conn.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone()
            present = {
                str(row[0])
                for row in reopened.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if marker is None or int(marker[0]) != 7 or not V7_TABLES <= present:
                raise RuntimeError("installed MCP migrated Store did not reopen as v7")
        finally:
            reopened.close()
        return {
            "from": plan.from_schema_version,
            "to": plan.to_schema_version,
            "backend": plan.backend,
            "reopened": True,
        }


def _oauth_tls_smoke() -> dict[str, object]:
    """Run the installed Runtime through Host-pinned TLS, PKCE, and Bearer."""

    from dataclasses import replace

    import anyio

    from agent_libos import Runtime
    from agent_libos.config import DEFAULT_CONFIG
    from agent_libos.mcp import (
        InMemoryMcpCredentialBroker,
        McpOAuthProfile,
        McpOAuthRegistrationMode,
        McpOAuthStatusKind,
        McpOAuthTokenEndpointAuthMethod,
        PinnedMcpOAuthHttpTransport,
    )
    from agent_libos.mcp.types import McpComplete
    from agent_libos.substrate import LocalResourceProviderSubstrate

    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("installed MCP OAuth/TLS smoke requires openssl")

    with tempfile.TemporaryDirectory(prefix="agent-libos-installed-mcp-oauth-") as root:
        root_path = Path(root)
        cert = root_path / "localhost-ca-cert.pem"
        key = root_path / "localhost-ca-key.pem"
        evidence = root_path / "evidence.json"
        fixture = root_path / "installed_oauth_tls_server.py"
        fixture.write_text(_INSTALLED_OAUTH_TLS_SERVER, encoding="utf-8")
        generated = subprocess.run(
            [
                openssl,
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-sha256",
                "-days",
                "1",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=DNS:localhost,IP:127.0.0.1",
                "-addext",
                "basicConstraints=critical,CA:TRUE",
                "-addext",
                "keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if generated.returncode != 0:
            raise RuntimeError(
                "installed MCP OAuth/TLS certificate generation failed: "
                + generated.stderr[-2000:]
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = int(reserved.getsockname()[1])
        process = subprocess.Popen(
            [
                sys.executable,
                str(fixture),
                "--port",
                str(port),
                "--cert",
                str(cert),
                "--key",
                str(key),
                "--evidence",
                str(evidence),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                error = process.stderr.read() if process.stderr else ""
                raise RuntimeError(
                    "installed MCP OAuth/TLS fixture exited: " + error[-2000:]
                )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.1)
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.02)
        else:
            raise RuntimeError("installed MCP OAuth/TLS fixture did not become ready")

        previous_ca = os.environ.get("SSL_CERT_FILE")
        os.environ["SSL_CERT_FILE"] = str(cert)
        origin = f"https://localhost:{port}"
        resource_url = f"{origin}/mcp"
        context = ssl.create_default_context(cafile=str(cert))
        oauth_transport = PinnedMcpOAuthHttpTransport(
            resolver=lambda host, selected_port, _deadline: (
                ("127.0.0.1",)
                if host == "localhost" and selected_port == port
                else (_ for _ in ()).throw(
                    RuntimeError("OAuth fixture attempted an unpinned endpoint")
                )
            ),
            allow_loopback_http=True,
            allow_loopback_tls=True,
            ssl_context=context,
        )
        broker = InMemoryMcpCredentialBroker()
        substrate = LocalResourceProviderSubstrate(root_path)
        substrate.mcp_credential_broker = broker
        substrate.mcp_oauth_transport = oauth_transport
        runtime = Runtime.open(
            ":memory:",
            substrate=substrate,
            config=replace(
                DEFAULT_CONFIG,
                mcp=replace(DEFAULT_CONFIG.mcp, oauth_enabled=True),
            ),
        )
        try:
            profile = McpOAuthProfile(
                profile_id="installed-oauth-profile",
                server_id="installed-oauth-server",
                resource_uri=resource_url,
                expected_issuer=origin,
                redirect_uri="http://127.0.0.1:8765/oauth/callback",
                client_id="agent-libos-oauth-tls-client",
                registration_mode=McpOAuthRegistrationMode.PREREGISTERED,
                token_endpoint_auth_method=McpOAuthTokenEndpointAuthMethod.NONE,
                allowed_scopes=("mcp.read",),
                default_scopes=("mcp.read",),
                protected_resource_metadata_url=(
                    f"{origin}/.well-known/oauth-protected-resource/mcp"
                ),
                authorization_server_metadata_url=(
                    f"{origin}/.well-known/oauth-authorization-server"
                ),
                allowed_endpoint_origins=(origin,),
                allow_loopback_http=True,
            )
            provisional = runtime.mcp.add_oauth_profile(
                profile,
                actor="installed-artifact-smoke",
            )
            if provisional.status is not McpOAuthStatusKind.AUTHORIZATION_REQUIRED:
                raise RuntimeError("installed MCP OAuth profile was not provisional")
            runtime.mcp.register_server(
                {
                    "schema_version": 3,
                    "server_id": profile.server_id,
                    "transport": "streamable_http",
                    "protocol_mode": "2026-07-28",
                    "http": {"url": resource_url},
                    "resources": [
                        {
                            "resource_id": "status",
                            "remote_uri": "fixture://oauth/status",
                            "right": "read",
                            "information_flow": True,
                            "model_visible": False,
                            "mime_types": ["text/plain"],
                        }
                    ],
                    "auth_profile_id": profile.profile_id,
                    "subscriptions": [],
                    "timeout_s": 10,
                    "max_request_bytes": 65_536,
                    "max_response_bytes": 1_048_576,
                },
                actor="installed-artifact-smoke",
                require_capability=False,
            )
            challenge = runtime.mcp.auth_begin(
                profile.profile_id,
                scopes=("mcp.read",),
                actor="installed-artifact-smoke",
            )
            parsed = urlsplit(challenge.authorization_url)
            connection = http.client.HTTPSConnection(
                parsed.hostname,
                parsed.port,
                context=context,
                timeout=5,
            )
            try:
                connection.request(
                    "GET",
                    parsed.path + (f"?{parsed.query}" if parsed.query else ""),
                    headers={"Accept": "application/json"},
                )
                response = connection.getresponse()
                response.read()
                callback = response.getheader("Location")
            finally:
                connection.close()
            if response.status != 302 or not callback:
                raise RuntimeError("installed MCP OAuth authorization failed")
            status = runtime.mcp.auth_complete(
                challenge.challenge_id,
                callback,
                actor="installed-artifact-smoke",
            )
            if status.status is not McpOAuthStatusKind.AUTHORIZED:
                raise RuntimeError("installed MCP OAuth token exchange failed")
            resource = runtime.mcp.read_resource(
                profile.server_id,
                "status",
                actor="installed-artifact-smoke",
            )
            if not isinstance(resource, McpComplete) or "authorized" not in repr(resource):
                raise RuntimeError("installed MCP OAuth Bearer path failed")
            observed = json.loads(evidence.read_text(encoding="utf-8"))
            if not observed or not all(observed.values()):
                raise RuntimeError("installed MCP OAuth fixture evidence is incomplete")
            if anyio.run(runtime._mcp_connection_supervisor.snapshot):
                raise RuntimeError("installed MCP OAuth leaked a connection")
            return {
                "tls": True,
                "pkce": True,
                "bearer": True,
                "evidence_count": len(observed),
            }
        finally:
            runtime.close()
            broker.close()
            if previous_ca is None:
                os.environ.pop("SSL_CERT_FILE", None)
            else:
                os.environ["SSL_CERT_FILE"] = previous_ca
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stderr is not None:
                error = process.stderr.read()
                if error.strip():
                    raise RuntimeError(
                        "installed MCP OAuth/TLS fixture emitted stderr: "
                        + error[-2000:]
                    )


def main() -> int:
    import agent_libos

    installed_module = Path(agent_libos.__file__).resolve()
    installed_prefix = Path(sys.prefix).resolve()
    if not installed_module.is_relative_to(installed_prefix):
        raise RuntimeError(
            "release MCP smoke imported agent_libos outside the clean-install "
            f"environment: module={installed_module} prefix={installed_prefix}"
        )

    installed: dict[str, str] = {}
    for distribution, module, expected_major in REQUIRED_DISTRIBUTIONS:
        importlib.import_module(module)
        version = importlib.metadata.version(distribution)
        if _major_version(distribution) != expected_major:
            raise RuntimeError(
                f"{distribution} must use major version {expected_major}, found {version}"
            )
        installed[distribution] = version

    project_version = importlib.metadata.version("agent-libos")
    if agent_libos.__version__ != project_version:
        raise RuntimeError(
            "installed agent_libos.__version__ does not match distribution metadata"
        )
    if installed["mcp"] != "2.0.0":
        raise RuntimeError(
            "the reviewed MCP Python SDK must be exactly 2.0.0, found "
            f"{installed['mcp']}"
        )
    runtime_smoke = _runtime_smoke()
    migration_smoke = _migration_smoke()
    oauth_smoke = _oauth_tls_smoke()
    print(
        json.dumps(
            {
                "agent-libos": project_version,
                "mcp-extra": installed,
                "runtime-v3-stdio": runtime_smoke,
                "store-v6-to-v7": migration_smoke,
                "oauth-tls-pkce-bearer": oauth_smoke,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
