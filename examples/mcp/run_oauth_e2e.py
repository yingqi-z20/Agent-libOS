#!/usr/bin/env python3
"""Run the Host-preconfigured MCP OAuth lifecycle with a scripted broker."""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.mcp.manifest import McpResourceSpec, McpServerManifestV3
from agent_libos.mcp.oauth import (
    InMemoryMcpCredentialBroker,
    McpOAuthHttpResponse,
    McpOAuthProfile,
    McpOAuthRegistrationMode,
    McpOAuthTokenEndpointAuthMethod,
)
from agent_libos.models.mcp import McpHttpTransportSpec, McpProtocolMode
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.serde import to_jsonable


RESOURCE = "https://mcp.example.test/mcp"
ISSUER = "https://auth.example.test/tenant"
RESOURCE_METADATA = "https://mcp.example.test/.well-known/oauth-protected-resource/mcp"
OAUTH_METADATA = "https://auth.example.test/.well-known/oauth-authorization-server/tenant"
AUTHORIZATION_ENDPOINT = "https://auth.example.test/authorize"
TOKEN_ENDPOINT = "https://auth.example.test/token"
REVOCATION_ENDPOINT = "https://auth.example.test/revoke"
REDIRECT = "http://127.0.0.1:49152/oauth/callback"
ACCESS_TOKEN = "oauth-access-token-MUST-NOT-PROJECT"
REFRESH_TOKEN = "oauth-refresh-token-MUST-NOT-PROJECT"


class ScriptedOAuthTransport:
    def __init__(self, responses: Mapping[tuple[str, str], list[McpOAuthHttpResponse]]):
        self.responses = {key: list(value) for key, value in responses.items()}
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        body: bytes | None,
        deadline: float,
        max_response_bytes: int,
    ) -> McpOAuthHttpResponse:
        del deadline
        with self.lock:
            self.requests.append(
                {
                    "method": method,
                    "url": url,
                    "headers": dict(headers or {}),
                    "body": bytes(body or b""),
                }
            )
            pending = self.responses.get((method, url), [])
            if not pending:
                raise RuntimeError(f"unexpected scripted OAuth request: {method} {url}")
            response = pending.pop(0)
        if len(response.body) > max_response_bytes:
            raise RuntimeError("scripted OAuth response exceeds the Runtime bound")
        return response


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="agent-libos-mcp-oauth-") as directory:
        root = Path(directory)
        database = root / "oauth.sqlite"
        transport = ScriptedOAuthTransport(_responses())
        runtime = Runtime.open(
            database,
            config=replace(
                DEFAULT_CONFIG,
                mcp=replace(DEFAULT_CONFIG.mcp, oauth_enabled=True),
            ),
            substrate=_substrate(root, InMemoryMcpCredentialBroker(), transport),
        )
        authorized_projection: object
        challenge_keys: list[str]
        effects: list[str]
        evidence_text: str
        try:
            initial = runtime.mcp.add_oauth_profile(_profile(), actor="oauth-example")
            runtime.mcp.register_server(
                _manifest(),
                actor="oauth-example",
                require_capability=False,
            )
            challenge = runtime.mcp.auth_begin("work-account", actor="oauth-example")
            query = parse_qs(urlsplit(challenge.authorization_url).query)
            challenge_keys = sorted(query)
            authorized = runtime.mcp.auth_complete(
                challenge.challenge_id,
                _callback(query),
                actor="oauth-example",
            )
            authorized_projection = to_jsonable(authorized)
            effects = [
                effect.operation
                for effect in runtime.store.list_external_effects()
                if effect.provider == "mcp" and effect.operation.startswith("auth.")
            ]
            evidence_text = json.dumps(
                {
                    "initial": to_jsonable(initial),
                    "authorized": authorized_projection,
                    "audit": to_jsonable(runtime.audit.trace()),
                    "events": to_jsonable(runtime.events.list()),
                    "effects": to_jsonable(runtime.store.list_external_effects()),
                },
                sort_keys=True,
            )
            _assert_secrets_absent(evidence_text)
        finally:
            runtime.close()

        _assert_database_has_no_secrets(database)
        reopened = Runtime.open(
            database,
            config=replace(
                DEFAULT_CONFIG,
                mcp=replace(DEFAULT_CONFIG.mcp, oauth_enabled=True),
            ),
            substrate=_substrate(
                root,
                InMemoryMcpCredentialBroker(),
                ScriptedOAuthTransport({}),
            ),
        )
        try:
            restarted = reopened.mcp.auth_status("work-account", actor="oauth-example")
            reconfigured = reopened.mcp.add_oauth_profile(
                _profile(), actor="oauth-example"
            )
            logged_out = reopened.mcp.auth_logout(
                "work-account", actor="oauth-example"
            )
            output = {
                "schema_version": 1,
                "registration_mode": "preregistered",
                "dynamic_client_registration": "unsupported",
                "authorization_url": {
                    "origin": f"{urlsplit(AUTHORIZATION_ENDPOINT).scheme}://{urlsplit(AUTHORIZATION_ENDPOINT).netloc}",
                    "query_keys": challenge_keys,
                    "opened_automatically": False,
                },
                "authorized": authorized_projection,
                "provider_requests": [
                    {"method": request["method"], "url": request["url"]}
                    for request in transport.requests
                ],
                "protected_effect_operations": effects,
                "restart_without_broker_credentials": to_jsonable(restarted),
                "reconfigured": to_jsonable(reconfigured),
                "logged_out": to_jsonable(logged_out),
            }
            encoded = json.dumps(output, indent=2, sort_keys=True)
            _assert_secrets_absent(encoded)
            print(encoded)
        finally:
            reopened.close()
    return 0


def _profile() -> McpOAuthProfile:
    return McpOAuthProfile(
        profile_id="work-account",
        server_id="oauth-files",
        resource_uri=RESOURCE,
        expected_issuer=ISSUER,
        redirect_uri=REDIRECT,
        client_id="agent-libos-local-example",
        registration_mode=McpOAuthRegistrationMode.PREREGISTERED,
        token_endpoint_auth_method=McpOAuthTokenEndpointAuthMethod.NONE,
        allowed_scopes=("files:read",),
        default_scopes=("files:read",),
    )


def _manifest() -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id="oauth-files",
        transport="streamable_http",
        http=McpHttpTransportSpec(url=RESOURCE),
        timeout_s=5.0,
        max_request_bytes=64 * 1024,
        max_response_bytes=256 * 1024,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        resources=(McpResourceSpec(resource_id="status", remote_uri=f"{RESOURCE}/status"),),
        auth_profile_id="work-account",
    )


def _substrate(
    root: Path,
    broker: InMemoryMcpCredentialBroker,
    transport: ScriptedOAuthTransport,
) -> LocalResourceProviderSubstrate:
    selected = LocalResourceProviderSubstrate(root)
    selected.mcp_credential_broker = broker
    selected.mcp_oauth_transport = transport
    return selected


def _responses() -> dict[tuple[str, str], list[McpOAuthHttpResponse]]:
    return {
        ("GET", RESOURCE_METADATA): [
            _json_response(
                {
                    "resource": RESOURCE,
                    "authorization_servers": [ISSUER],
                    "scopes_supported": ["files:read"],
                }
            )
        ],
        ("GET", OAUTH_METADATA): [
            _json_response(
                {
                    "issuer": ISSUER,
                    "authorization_endpoint": AUTHORIZATION_ENDPOINT,
                    "token_endpoint": TOKEN_ENDPOINT,
                    "revocation_endpoint": REVOCATION_ENDPOINT,
                    "code_challenge_methods_supported": ["S256"],
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                    "token_endpoint_auth_methods_supported": ["none"],
                    "authorization_response_iss_parameter_supported": True,
                }
            )
        ],
        ("POST", TOKEN_ENDPOINT): [
            _json_response(
                {
                    "access_token": ACCESS_TOKEN,
                    "refresh_token": REFRESH_TOKEN,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "files:read",
                    "resource": RESOURCE,
                }
            )
        ],
    }


def _json_response(value: object) -> McpOAuthHttpResponse:
    return McpOAuthHttpResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps(value, separators=(",", ":")).encode("utf-8"),
    )


def _callback(query: Mapping[str, list[str]]) -> str:
    return f"{REDIRECT}?{urlencode({'code': 'local-code', 'state': query['state'][0], 'iss': ISSUER})}"


def _assert_secrets_absent(value: str) -> None:
    for secret in (ACCESS_TOKEN, REFRESH_TOKEN, "local-code"):
        if secret in value:
            raise RuntimeError("OAuth example projected a secret")


def _assert_database_has_no_secrets(database: Path) -> None:
    for candidate in (database, *database.parent.glob(f"{database.name}-*")):
        if not candidate.exists():
            continue
        raw = candidate.read_bytes()
        for secret in (ACCESS_TOKEN, REFRESH_TOKEN, "local-code"):
            if secret.encode("utf-8") in raw:
                raise RuntimeError("OAuth example persisted a secret")


if __name__ == "__main__":
    raise SystemExit(main())
