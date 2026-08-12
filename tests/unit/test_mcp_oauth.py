from __future__ import annotations

import base64
import importlib
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

import agent_libos.mcp.oauth as oauth_module
from agent_libos.mcp.oauth import (
    InMemoryMcpCredentialBroker,
    McpOAuthAuthorizationRequired,
    McpOAuthError,
    McpOAuthHttpResponse,
    McpOAuthManager,
    McpOAuthNeedsAttention,
    McpOAuthProfile,
    McpOAuthRegistrationMode,
    McpOAuthTokenEndpointAuthMethod,
    McpOAuthTransportError,
    PinnedMcpOAuthHttpTransport,
    SystemKeyringMcpCredentialBroker,
    mcp_oauth_profile_from_mapping,
    parse_mcp_oauth_www_authenticate,
)
from agent_libos.mcp.types import McpOAuthStatusKind


RESOURCE = "https://mcp.example.test/mcp"
ISSUER = "https://auth.example.test/tenant"
RESOURCE_METADATA = (
    "https://mcp.example.test/.well-known/oauth-protected-resource/mcp"
)
PINNED_RESOURCE_METADATA = "https://mcp.example.test/oauth/resource-metadata"
OAUTH_METADATA = (
    "https://auth.example.test/.well-known/oauth-authorization-server/tenant"
)
AUTHORIZATION_ENDPOINT = "https://auth.example.test/authorize"
TOKEN_ENDPOINT = "https://auth.example.test/token"
REVOCATION_ENDPOINT = "https://auth.example.test/revoke"
REDIRECT = "http://127.0.0.1:49152/oauth/callback"


class ScriptedTransport:
    def __init__(self, responses: Mapping[tuple[str, str], list[McpOAuthHttpResponse]]) -> None:
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
        with self.lock:
            self.requests.append(
                {
                    "method": method,
                    "url": url,
                    "headers": dict(headers or {}),
                    "body": bytes(body or b""),
                }
            )
            selected = self.responses.get((method, url), [])
            if not selected:
                raise AssertionError(f"unexpected OAuth request: {method}")
            response = selected.pop(0)
        if len(response.body) > max_response_bytes:
            raise AssertionError("test response exceeds manager limit")
        return response


class _FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        backend_type = importlib.import_module("keyring.backends.macOS").Keyring
        self.backend = object.__new__(backend_type)

    def get_keyring(self) -> object:
        return self.backend

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


def _json_response(value: object, *, status: int = 200) -> McpOAuthHttpResponse:
    return McpOAuthHttpResponse(
        status=status,
        headers={"content-type": "application/json"},
        body=json.dumps(value, separators=(",", ":")).encode(),
    )


def _metadata_responses(
    *,
    resource_overrides: Mapping[str, object] | None = None,
    server_overrides: Mapping[str, object] | None = None,
    token_responses: list[McpOAuthHttpResponse] | None = None,
) -> dict[tuple[str, str], list[McpOAuthHttpResponse]]:
    resource: dict[str, object] = {
        "resource": RESOURCE,
        "authorization_servers": [ISSUER],
        "scopes_supported": ["files:read", "files:write"],
    }
    resource.update(resource_overrides or {})
    server: dict[str, object] = {
        "issuer": ISSUER,
        "authorization_endpoint": AUTHORIZATION_ENDPOINT,
        "token_endpoint": TOKEN_ENDPOINT,
        "revocation_endpoint": REVOCATION_ENDPOINT,
        "code_challenge_methods_supported": ["S256"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_basic"],
        "authorization_response_iss_parameter_supported": True,
    }
    server.update(server_overrides or {})
    return {
        ("GET", RESOURCE_METADATA): [_json_response(resource)],
        ("GET", OAUTH_METADATA): [_json_response(server)],
        ("POST", TOKEN_ENDPOINT): list(token_responses or []),
    }


def _profile(**overrides: object) -> McpOAuthProfile:
    selected: dict[str, object] = {
        "profile_id": "work-account",
        "server_id": "files",
        "resource_uri": RESOURCE,
        "expected_issuer": ISSUER,
        "redirect_uri": REDIRECT,
        "client_id": "agent-libos-desktop",
        "registration_mode": McpOAuthRegistrationMode.PREREGISTERED,
        "token_endpoint_auth_method": McpOAuthTokenEndpointAuthMethod.NONE,
        "allowed_scopes": ("files:read", "files:write"),
        "default_scopes": ("files:read",),
    }
    selected.update(overrides)
    return McpOAuthProfile(**selected)  # type: ignore[arg-type]


def _begin(
    *,
    profile: McpOAuthProfile | None = None,
    transport: ScriptedTransport | None = None,
    client_secret: bytes | None = None,
) -> tuple[McpOAuthManager, ScriptedTransport, str, dict[str, list[str]]]:
    selected_transport = transport or ScriptedTransport(_metadata_responses())
    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(),
        transport=selected_transport,
    )
    manager.add_profile(profile or _profile(), client_secret=client_secret)
    challenge = manager.begin("work-account")
    query = parse_qs(urlsplit(challenge.authorization_url).query)
    return manager, selected_transport, challenge.challenge_id, query


def _callback(query: Mapping[str, list[str]], **values: str) -> str:
    params = {
        "code": "authorization-code",
        "state": query["state"][0],
        "iss": ISSUER,
    }
    params.update(values)
    from urllib.parse import urlencode

    return f"{REDIRECT}?{urlencode(params)}"


def test_begin_uses_pkce_s256_and_binds_resource_redirect_and_scopes() -> None:
    manager, _transport, _challenge_id, query = _begin()

    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["agent-libos-desktop"]
    assert query["redirect_uri"] == [REDIRECT]
    assert query["resource"] == [RESOURCE]
    assert query["scope"] == ["files:read"]
    assert query["code_challenge_method"] == ["S256"]
    assert len(query["state"][0]) >= 43
    assert len(query["code_challenge"][0]) == 43
    assert manager.status("work-account").status is McpOAuthStatusKind.AUTHORIZATION_REQUIRED
    assert manager.list_profiles() == (manager.status("work-account"),)


def test_custom_transport_late_response_is_never_accepted_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(),
        transport=ScriptedTransport(_metadata_responses()),
    )
    manager.add_profile(_profile())
    ticks = iter((0.0, 11.0))
    monkeypatch.setattr(
        oauth_module.time,
        "monotonic",
        lambda: next(ticks, 11.0),
    )

    with pytest.raises(McpOAuthTransportError) as raised:
        manager.begin("work-account", deadline=10.0)

    assert raised.value.dispatch_state == "started"
    assert (
        manager.status("work-account").status
        is McpOAuthStatusKind.AUTHORIZATION_REQUIRED
    )
    manager.close()


def test_initial_scope_falls_back_to_metadata_but_cannot_expand_host_allowlist() -> None:
    profile = replace(_profile(), default_scopes=(), allowed_scopes=("files:read",))
    manager, _transport, _challenge_id, query = _begin(profile=profile)
    assert query["scope"] == ["files:read"]


def test_explicit_same_origin_resource_metadata_pin_is_accepted_and_used() -> None:
    responses = _metadata_responses()
    responses[("GET", PINNED_RESOURCE_METADATA)] = responses.pop(
        ("GET", RESOURCE_METADATA)
    )
    transport = ScriptedTransport(responses)
    profile = replace(
        _profile(),
        protected_resource_metadata_url=PINNED_RESOURCE_METADATA,
    )

    _manager, transport, _challenge_id, _query = _begin(
        profile=profile,
        transport=transport,
    )

    assert transport.requests[0]["url"] == PINNED_RESOURCE_METADATA


def test_complete_validates_state_issuer_and_exchanges_code_once() -> None:
    access = "access-token-never-project"
    refresh = "refresh-token-never-project"
    transport = ScriptedTransport(
        _metadata_responses(
            token_responses=[
                _json_response(
                    {
                        "access_token": access,
                        "refresh_token": refresh,
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "files:read",
                        "resource": RESOURCE,
                    }
                )
            ]
        )
    )
    manager, transport, challenge_id, query = _begin(transport=transport)
    status = manager.complete(challenge_id, _callback(query))

    assert status.status is McpOAuthStatusKind.AUTHORIZED
    assert status.scopes == ("files:read",)
    token_request = transport.requests[-1]
    form = parse_qs(token_request["body"].decode())
    assert form["grant_type"] == ["authorization_code"]
    assert form["resource"] == [RESOURCE]
    assert form["redirect_uri"] == [REDIRECT]
    verifier = form["code_verifier"][0]
    expected_challenge = base64.urlsafe_b64encode(sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert expected_challenge == query["code_challenge"][0]
    outward = repr((status, manager.status("work-account"), manager.__dict__))
    assert access not in outward
    assert refresh not in outward
    assert manager.access_token("work-account") == access.encode()


@pytest.mark.parametrize(
    "callback",
    [
        f"{REDIRECT}?code=x&state=wrong&iss={ISSUER}",
        f"{REDIRECT}?code=x&state=wrong&iss=https://evil.example/",
        "http://127.0.0.1:49153/oauth/callback?code=x&state=wrong",
    ],
)
def test_callback_binding_failure_is_single_use_and_does_not_dispatch(callback: str) -> None:
    manager, transport, challenge_id, _query = _begin()
    with pytest.raises(McpOAuthError, match="authorization callback rejected"):
        manager.complete(challenge_id, callback)
    assert [request["method"] for request in transport.requests] == ["GET", "GET"]
    assert manager.status("work-account").status is (
        McpOAuthStatusKind.AUTHORIZATION_REQUIRED
    )
    with pytest.raises(McpOAuthError, match="authorization challenge is unavailable"):
        manager.complete(challenge_id, callback)


def test_issuer_comparison_is_exact_without_url_normalization() -> None:
    manager, transport, challenge_id, query = _begin()
    with pytest.raises(McpOAuthError, match="authorization callback rejected"):
        manager.complete(challenge_id, _callback(query, iss=f"{ISSUER}/"))
    assert [request["method"] for request in transport.requests] == ["GET", "GET"]


def test_advertised_issuer_parameter_is_mandatory_and_remote_error_is_inert() -> None:
    manager, transport, challenge_id, query = _begin()
    callback = f"{REDIRECT}?code=x&state={query['state'][0]}"
    with pytest.raises(McpOAuthError, match="authorization callback rejected"):
        manager.complete(challenge_id, callback)
    assert [request["method"] for request in transport.requests] == ["GET", "GET"]

    manager, transport, challenge_id, query = _begin()
    reflected = "remote-error-secret"
    callback = (
        f"{REDIRECT}?error=access_denied&error_description={reflected}"
        f"&state={query['state'][0]}&iss={ISSUER}"
    )
    with pytest.raises(McpOAuthError) as raised:
        manager.complete(challenge_id, callback)
    assert reflected not in str(raised.value)
    assert [request["method"] for request in transport.requests] == ["GET", "GET"]


def test_metadata_issuer_mismatch_and_cross_origin_endpoint_fail_before_browser_url() -> None:
    mismatch = ScriptedTransport(
        _metadata_responses(server_overrides={"issuer": f"{ISSUER}/"})
    )
    manager = McpOAuthManager(broker=InMemoryMcpCredentialBroker(), transport=mismatch)
    manager.add_profile(_profile())
    with pytest.raises(McpOAuthError, match="authorization metadata rejected"):
        manager.begin("work-account")


def test_metadata_redirect_is_not_followed_and_custom_transport_errors_are_sanitized() -> None:
    redirect = ScriptedTransport(
        {
            ("GET", RESOURCE_METADATA): [
                McpOAuthHttpResponse(
                    status=302,
                    headers={"location": "https://evil.example/metadata"},
                    body=b"",
                )
            ]
        }
    )
    manager = McpOAuthManager(broker=InMemoryMcpCredentialBroker(), transport=redirect)
    manager.add_profile(_profile())
    with pytest.raises(McpOAuthError, match="metadata is unavailable"):
        manager.begin("work-account")
    assert len(redirect.requests) == 1

    reflected = "client-secret-reflected-by-transport"

    class LeakyTransport:
        def request(self, *args: Any, **kwargs: Any) -> McpOAuthHttpResponse:
            raise RuntimeError(reflected)

    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(), transport=LeakyTransport()
    )
    manager.add_profile(_profile())
    with pytest.raises(McpOAuthTransportError) as raised:
        manager.begin("work-account")
    assert reflected not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None

    class ChainedLeakyTransport:
        def request(self, *args: Any, **kwargs: Any) -> McpOAuthHttpResponse:
            failure = McpOAuthTransportError("unknown")
            failure.__cause__ = RuntimeError(reflected)
            raise failure

    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(), transport=ChainedLeakyTransport()
    )
    manager.add_profile(_profile())
    with pytest.raises(McpOAuthTransportError) as raised:
        manager.begin("work-account")
    assert reflected not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None

    cross_origin = ScriptedTransport(
        _metadata_responses(
            server_overrides={"token_endpoint": "https://evil.example/token"}
        )
    )
    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(), transport=cross_origin
    )
    manager.add_profile(_profile())
    with pytest.raises(McpOAuthError, match="authorization metadata rejected"):
        manager.begin("work-account")


def test_rejected_authorization_url_never_persists_pkce_or_state() -> None:
    class RecordingBroker(InMemoryMcpCredentialBroker):
        def __init__(self) -> None:
            super().__init__()
            self.namespaces: list[str] = []

        def put_secret(
            self,
            namespace: str,
            value: bytes,
            *,
            expires_at: str | None,
        ) -> str:
            self.namespaces.append(namespace)
            return super().put_secret(namespace, value, expires_at=expires_at)

    broker = RecordingBroker()
    manager = McpOAuthManager(
        broker=broker,
        transport=ScriptedTransport(
            _metadata_responses(
                server_overrides={
                    "authorization_endpoint": f"{AUTHORIZATION_ENDPOINT}?state=remote"
                }
            )
        ),
    )
    manager.add_profile(_profile())

    with pytest.raises(McpOAuthError, match="metadata rejected"):
        manager.begin("work-account")

    assert broker.namespaces == []


def test_scope_escalation_is_rejected_before_discovery() -> None:
    transport = ScriptedTransport(_metadata_responses())
    manager = McpOAuthManager(broker=InMemoryMcpCredentialBroker(), transport=transport)
    manager.add_profile(_profile())
    with pytest.raises(McpOAuthError, match="scope request rejected"):
        manager.begin("work-account", scopes=("admin",))
    assert transport.requests == []


def test_www_authenticate_step_up_unions_scopes_without_replaying_operation() -> None:
    responses = _metadata_responses(
        token_responses=[
            _json_response(
                {
                    "access_token": "initial-access",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "files:read",
                }
            ),
            _json_response(
                {
                    "access_token": "step-up-access",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "files:read files:write",
                }
            ),
        ]
    )
    responses[("GET", RESOURCE_METADATA)] *= 2
    responses[("GET", OAUTH_METADATA)] *= 2
    transport = ScriptedTransport(responses)
    manager, _transport, challenge_id, query = _begin(transport=transport)
    manager.complete(challenge_id, _callback(query))

    challenge = manager.authorize_for_challenge(
        "work-account",
        (
            f'Bearer error="insufficient_scope", scope="files:write", '
            f'resource_metadata="{RESOURCE_METADATA}"'
        ),
    )
    step_up_query = parse_qs(urlsplit(challenge.authorization_url).query)
    assert step_up_query["scope"] == ["files:read files:write"]
    # authorize_for_challenge only created a browser flow; it did not replay
    # the protected MCP operation or contact the token endpoint yet.
    assert len([item for item in transport.requests if item["method"] == "POST"]) == 1
    manager.complete(challenge.challenge_id, _callback(step_up_query))
    assert manager.access_token("work-account") == b"step-up-access"


@pytest.mark.parametrize(
    "header",
    [
        'Bearer scope="files:read", scope="files:write"',
        'Bearer scope="files:read"\r\nX-Evil: yes',
        'Basic realm="x"',
        'Bearer scope="unterminated',
    ],
)
def test_www_authenticate_parser_rejects_ambiguous_or_injected_challenges(
    header: str,
) -> None:
    with pytest.raises(McpOAuthError, match="challenge rejected"):
        parse_mcp_oauth_www_authenticate(header)


def test_challenge_cannot_change_resource_metadata_origin() -> None:
    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(),
        transport=ScriptedTransport(_metadata_responses()),
    )
    manager.add_profile(_profile())
    with pytest.raises(McpOAuthError, match="challenge rejected"):
        manager.authorize_for_challenge(
            "work-account",
            'Bearer scope="files:read", resource_metadata="https://evil.example/meta"',
        )


def test_token_scope_escalation_is_generic_and_token_never_appears_in_error() -> None:
    secret = "server-reflected-secret-token"
    transport = ScriptedTransport(
        _metadata_responses(
            token_responses=[
                _json_response(
                    {
                        "access_token": secret,
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "files:read admin",
                    }
                )
            ]
        )
    )
    manager, _transport, challenge_id, query = _begin(transport=transport)
    with pytest.raises(McpOAuthError) as raised:
        manager.complete(challenge_id, _callback(query))
    assert secret not in str(raised.value)
    assert manager.status("work-account").status is McpOAuthStatusKind.NEEDS_ATTENTION


def test_cimd_is_supported_but_dynamic_registration_is_rejected() -> None:
    profile = replace(
        _profile(),
        registration_mode=McpOAuthRegistrationMode.CIMD,
        client_id="https://client.example.test/oauth/metadata.json",
    )
    manager, cimd_transport, _challenge_id, query = _begin(profile=profile)
    assert query["client_id"] == [profile.client_id]
    assert profile.client_id not in {
        request["url"] for request in cimd_transport.requests
    }

    invalid = replace(profile, registration_mode="dcr")  # type: ignore[arg-type]
    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(),
        transport=ScriptedTransport(_metadata_responses()),
    )
    with pytest.raises(McpOAuthError, match="dynamic client registration is unsupported"):
        manager.add_profile(invalid)


@pytest.mark.parametrize(
    "overrides",
    [
        {"protocol_revision": "2025-11-25"},
        {"transport": "stdio"},
    ],
)
def test_oauth_is_v3_modern_streamable_http_only(overrides: dict[str, str]) -> None:
    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(),
        transport=ScriptedTransport(_metadata_responses()),
    )
    with pytest.raises(McpOAuthError, match="Manifest v3 Streamable HTTP"):
        manager.add_profile(replace(_profile(), **overrides))  # type: ignore[arg-type]


def test_loopback_http_requires_explicit_host_profile_opt_in() -> None:
    loopback = replace(
        _profile(),
        resource_uri="http://127.0.0.1:8123/mcp",
        expected_issuer="http://127.0.0.1:8124",
        audience="http://127.0.0.1:8123/mcp",
    )
    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(), transport=ScriptedTransport({})
    )
    with pytest.raises(McpOAuthError):
        manager.add_profile(loopback)
    assert manager.add_profile(replace(loopback, allow_loopback_http=True)).status is (
        McpOAuthStatusKind.AUTHORIZATION_REQUIRED
    )


def test_confidential_client_secret_lives_only_in_broker_and_basic_header() -> None:
    secret = b"client/secret-never-project"
    profile = replace(
        _profile(),
        client_id="agent:desktop",
        token_endpoint_auth_method=McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_BASIC,
    )
    transport = ScriptedTransport(
        _metadata_responses(
            token_responses=[
                _json_response(
                    {
                        "access_token": "access",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "files:read",
                    }
                )
            ]
        )
    )
    manager, transport, challenge_id, query = _begin(
        profile=profile, transport=transport, client_secret=secret
    )
    assert secret.decode() not in repr(manager.__dict__)
    manager.complete(challenge_id, _callback(query))
    request = transport.requests[-1]
    assert request["headers"]["Authorization"].startswith("Basic ")
    encoded = request["headers"]["Authorization"].removeprefix("Basic ")
    assert base64.b64decode(encoded) == b"agent%3Adesktop:client%2Fsecret-never-project"
    assert secret not in request["body"]
    lease = manager.transport_access(profile.profile_id)
    assert secret.decode() in lease.redaction_values()
    assert request["headers"]["Authorization"] in lease.redaction_values()
    assert secret.decode() not in repr(lease)
    lease.close()


def test_local_logout_retains_host_registration_secret_for_explicit_reauthorization() -> None:
    secret = b"durable-host-registration-secret"
    profile = replace(
        _profile(),
        token_endpoint_auth_method=McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_BASIC,
    )
    token = _json_response(
        {
            "access_token": "access",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "files:read",
        }
    )
    responses = _metadata_responses(token_responses=[token, token])
    responses[("GET", RESOURCE_METADATA)] *= 2
    responses[("GET", OAUTH_METADATA)] *= 2
    broker = InMemoryMcpCredentialBroker()
    transport = ScriptedTransport(responses)
    manager = McpOAuthManager(broker=broker, transport=transport)
    manager.add_profile(profile, client_secret=secret)

    first = manager.begin(profile.profile_id)
    first_query = parse_qs(urlsplit(first.authorization_url).query)
    manager.complete(first.challenge_id, _callback(first_query))
    assert manager.logout(profile.profile_id).status is McpOAuthStatusKind.REVOKED
    assert len(broker._secrets) == 1  # noqa: SLF001 - exact secret lifecycle assertion

    second = manager.begin(profile.profile_id)
    second_query = parse_qs(urlsplit(second.authorization_url).query)
    assert manager.complete(second.challenge_id, _callback(second_query)).status is (
        McpOAuthStatusKind.AUTHORIZED
    )
    basic_headers = [
        request["headers"]["Authorization"]
        for request in transport.requests
        if "Authorization" in request["headers"]
    ]
    assert len(basic_headers) == 2
    assert all(secret.decode() not in header for header in basic_headers)


def test_credential_rotation_and_logout_invalidate_server_connections_best_effort() -> None:
    invalidated: list[str] = []

    def invalidate(server_id: str) -> None:
        invalidated.append(server_id)
        if len(invalidated) == 2:
            raise RuntimeError("supervisor already shutting down")

    transport = ScriptedTransport(
        _metadata_responses(
            token_responses=[
                _json_response(
                    {
                        "access_token": "short",
                        "refresh_token": "rotate-once",
                        "token_type": "Bearer",
                        "expires_in": 1,
                        "scope": "files:read",
                    }
                ),
                _json_response(
                    {
                        "access_token": "fresh",
                        "refresh_token": "rotated",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "files:read",
                    }
                ),
            ]
        )
    )
    manager = McpOAuthManager(
        broker=InMemoryMcpCredentialBroker(),
        transport=transport,
        connection_invalidator=invalidate,
    )
    manager.add_profile(_profile())
    challenge = manager.begin("work-account")
    query = parse_qs(urlsplit(challenge.authorization_url).query)
    manager.complete(challenge.challenge_id, _callback(query))
    assert manager.access_token("work-account", min_validity_s=30) == b"fresh"
    assert manager.logout("work-account").status is McpOAuthStatusKind.REVOKED
    assert invalidated == ["files", "files", "files"]


def test_refresh_is_singleflight_and_rotation_unknown_disables_retry() -> None:
    initial = _json_response(
        {
            "access_token": "expired-access",
            "refresh_token": "one-shot-refresh",
            "token_type": "Bearer",
            "expires_in": 1,
            "scope": "files:read",
        }
    )
    refreshed = _json_response(
        {
            "access_token": "new-access",
            "refresh_token": "rotated-refresh",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "files:read",
        }
    )
    transport = ScriptedTransport(
        _metadata_responses(token_responses=[initial, refreshed])
    )
    manager, transport, challenge_id, query = _begin(transport=transport)
    manager.complete(challenge_id, _callback(query))

    barrier = threading.Barrier(5)
    results: list[bytes] = []

    def obtain() -> None:
        barrier.wait()
        results.append(manager.access_token("work-account", min_validity_s=30))

    threads = [threading.Thread(target=obtain) for _ in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
    assert results == [b"new-access"] * 4
    assert len([r for r in transport.requests if r["method"] == "POST"]) == 2

    class UnknownTransport(ScriptedTransport):
        def request(self, *args: Any, **kwargs: Any) -> McpOAuthHttpResponse:
            if args[0] == "POST":
                raise McpOAuthTransportError("unknown")
            return super().request(*args, **kwargs)

    unknown = UnknownTransport(_metadata_responses(token_responses=[initial]))
    manager, _transport, challenge_id, query = _begin(transport=unknown)
    # Let the initial code exchange succeed before switching POST to unknown.
    unknown_post = ScriptedTransport.request
    unknown.request = unknown_post.__get__(unknown, UnknownTransport)  # type: ignore[method-assign]
    manager.complete(challenge_id, _callback(query))

    def fail_post(*args: Any, **kwargs: Any) -> McpOAuthHttpResponse:
        raise McpOAuthTransportError("unknown")

    unknown.request = fail_post  # type: ignore[method-assign]
    with pytest.raises(McpOAuthNeedsAttention):
        manager.access_token("work-account", min_validity_s=30)
    assert manager.status("work-account").status is McpOAuthStatusKind.NEEDS_ATTENTION


def test_refresh_token_remains_brokered_after_access_token_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _json_response(
        {
            "access_token": "short-access",
            "refresh_token": "long-refresh",
            "token_type": "Bearer",
            "expires_in": 1,
            "scope": "files:read",
        }
    )
    refreshed = _json_response(
        {
            "access_token": "after-expiry",
            "refresh_token": "rotated-after-expiry",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "files:read",
        }
    )
    transport = ScriptedTransport(
        _metadata_responses(token_responses=[initial, refreshed])
    )
    manager, _transport, challenge_id, query = _begin(transport=transport)
    manager.complete(challenge_id, _callback(query))
    real_now = time.time()
    monkeypatch.setattr("agent_libos.mcp.oauth.time.time", lambda: real_now + 2)
    assert manager.access_token("work-account", min_validity_s=0) == b"after-expiry"


def test_valid_token_reader_cannot_join_another_threads_refresh_claim() -> None:
    initial = _json_response(
        {
            "access_token": "old-access",
            "refresh_token": "one-shot-refresh",
            "token_type": "Bearer",
            "expires_in": 60,
            "scope": "files:read",
        }
    )
    refreshed = _json_response(
        {
            "access_token": "new-access",
            "refresh_token": "rotated-refresh",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "files:read",
        }
    )

    class BlockingRefreshTransport(ScriptedTransport):
        def __init__(self) -> None:
            super().__init__(_metadata_responses(token_responses=[initial, refreshed]))
            self.block_refresh = False
            self.refresh_started = threading.Event()
            self.release_refresh = threading.Event()

        def request(self, *args: Any, **kwargs: Any) -> McpOAuthHttpResponse:
            if self.block_refresh and args[0] == "POST" and args[1] == TOKEN_ENDPOINT:
                self.refresh_started.set()
                assert self.release_refresh.wait(timeout=5)
            return super().request(*args, **kwargs)

    transport = BlockingRefreshTransport()
    manager, _transport, challenge_id, query = _begin(transport=transport)
    manager.complete(challenge_id, _callback(query))
    transport.block_refresh = True
    results: list[bytes] = []
    refresh_thread = threading.Thread(
        target=lambda: results.append(
            manager.access_token("work-account", min_validity_s=120)
        )
    )
    refresh_thread.start()
    assert transport.refresh_started.wait(timeout=5)
    valid_reader_done = threading.Event()

    def valid_reader() -> None:
        results.append(manager.access_token("work-account", min_validity_s=0))
        valid_reader_done.set()

    reader_thread = threading.Thread(target=valid_reader)
    reader_thread.start()
    assert valid_reader_done.wait(timeout=0.05) is False
    transport.release_refresh.set()
    refresh_thread.join(timeout=5)
    reader_thread.join(timeout=5)
    assert results == [b"new-access", b"new-access"]
    assert len([item for item in transport.requests if item["method"] == "POST"]) == 2


def test_revoke_is_one_shot_and_logout_does_not_claim_remote_state() -> None:
    transport = ScriptedTransport(
        {
            **_metadata_responses(
                token_responses=[
                    _json_response(
                        {
                            "access_token": "access-to-revoke",
                            "refresh_token": "refresh-to-revoke",
                            "token_type": "Bearer",
                            "expires_in": 3600,
                            "scope": "files:read",
                        }
                    )
                ]
            ),
            ("POST", REVOCATION_ENDPOINT): [
                McpOAuthHttpResponse(status=200, headers={}, body=b"")
            ],
        }
    )
    manager, transport, challenge_id, query = _begin(transport=transport)
    manager.complete(challenge_id, _callback(query))
    status = manager.revoke("work-account")
    assert status.status is McpOAuthStatusKind.REVOKED
    revocation_requests = [
        request for request in transport.requests if request["url"] == REVOCATION_ENDPOINT
    ]
    assert len(revocation_requests) == 1
    assert b"refresh-to-revoke" in revocation_requests[0]["body"]
    with pytest.raises(McpOAuthAuthorizationRequired):
        manager.access_token("work-account")


def test_concurrent_revoke_has_one_generation_claim_and_one_remote_post() -> None:
    class BlockingRevokeTransport(ScriptedTransport):
        def __init__(self) -> None:
            super().__init__(
                {
                    **_metadata_responses(
                        token_responses=[
                            _json_response(
                                {
                                    "access_token": "access",
                                    "refresh_token": "refresh",
                                    "token_type": "Bearer",
                                    "expires_in": 3600,
                                    "scope": "files:read",
                                }
                            )
                        ]
                    ),
                    ("POST", REVOCATION_ENDPOINT): [
                        McpOAuthHttpResponse(status=200, headers={}, body=b"")
                    ],
                }
            )
            self.revoke_started = threading.Event()
            self.release_revoke = threading.Event()

        def request(self, *args: Any, **kwargs: Any) -> McpOAuthHttpResponse:
            if args[0] == "POST" and args[1] == REVOCATION_ENDPOINT:
                self.revoke_started.set()
                assert self.release_revoke.wait(timeout=5)
            return super().request(*args, **kwargs)

    transport = BlockingRevokeTransport()
    manager, _transport, challenge_id, query = _begin(transport=transport)
    manager.complete(challenge_id, _callback(query))
    outcomes: list[McpOAuthStatusKind] = []
    thread = threading.Thread(
        target=lambda: outcomes.append(manager.revoke("work-account").status)
    )
    thread.start()
    assert transport.revoke_started.wait(timeout=5)
    with pytest.raises(McpOAuthError, match="profile is busy"):
        manager.revoke("work-account")
    preflight = manager.status("work-account")
    assert preflight.status is McpOAuthStatusKind.AUTHORIZED
    transport.release_revoke.set()
    thread.join(timeout=5)
    assert outcomes == [McpOAuthStatusKind.REVOKED]
    assert len(
        [item for item in transport.requests if item["url"] == REVOCATION_ENDPOINT]
    ) == 1


def test_transport_access_pairs_exact_generation_and_zeroizes_lease() -> None:
    transport = ScriptedTransport(
        _metadata_responses(
            token_responses=[
                _json_response(
                    {
                        "access_token": "lease-token",
                        "refresh_token": "lease-refresh-token",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "files:read",
                    }
                )
            ]
        )
    )
    manager, _transport, challenge_id, query = _begin(transport=transport)
    manager.complete(challenge_id, _callback(query))
    preflight_fence = manager.credential_fence("work-account")
    lease = manager.transport_access("work-account")
    assert "lease-token" not in repr(lease)
    assert lease.authorization_header() == "Bearer lease-token"
    assert lease.redaction_values() == ("lease-refresh-token",)
    assert lease.fence == preflight_fence
    assert manager.validate_credential_fence(lease.fence) is True
    lease.close()
    with pytest.raises(McpOAuthError, match="lease is closed"):
        lease.bearer_token()
    with pytest.raises(McpOAuthError, match="lease is closed"):
        lease.redaction_values()
    manager.logout("work-account")
    assert manager.validate_credential_fence(lease.fence) is False


def test_expired_token_invalidates_existing_credential_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = ScriptedTransport(
        _metadata_responses(
            token_responses=[
                _json_response(
                    {
                        "access_token": "short",
                        "token_type": "Bearer",
                        "expires_in": 1,
                        "scope": "files:read",
                    }
                )
            ]
        )
    )
    manager, _transport, challenge_id, query = _begin(transport=transport)
    manager.complete(challenge_id, _callback(query))
    fence = manager.credential_fence("work-account")
    real_now = time.time()
    monkeypatch.setattr("agent_libos.mcp.oauth.time.time", lambda: real_now + 2)
    assert manager.validate_credential_fence(fence) is False
    with pytest.raises(McpOAuthAuthorizationRequired):
        manager.credential_fence("work-account")


def test_pinned_transport_rejects_private_or_rebound_dns_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected: list[tuple[object, ...]] = []

    def connect(*args: object, **kwargs: object) -> None:
        connected.append(args)
        raise AssertionError("must not connect")

    monkeypatch.setattr("socket.create_connection", connect)
    transport = PinnedMcpOAuthHttpTransport(
        resolver=lambda _host, _port, _deadline: ("93.184.216.34", "127.0.0.1")
    )
    with pytest.raises(McpOAuthTransportError) as raised:
        transport.request(
            "GET",
            "https://metadata.example.test/oauth",
            headers=None,
            body=None,
            deadline=time.monotonic() + 1,
            max_response_bytes=1024,
        )
    assert raised.value.dispatch_state == "not_started"
    assert connected == []


def test_pinned_transport_loopback_tls_requires_explicit_host_private_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected: list[tuple[object, ...]] = []

    def connect(*args: object, **kwargs: object) -> None:
        connected.append(args)
        raise OSError("fixture did not start a socket")

    monkeypatch.setattr("socket.create_connection", connect)
    default = PinnedMcpOAuthHttpTransport(
        resolver=lambda _host, _port, _deadline: ("127.0.0.1",)
    )
    with pytest.raises(McpOAuthTransportError):
        default.request(
            "GET",
            "https://localhost:9443/oauth",
            headers=None,
            body=None,
            deadline=time.monotonic() + 1,
            max_response_bytes=1024,
        )
    assert connected == []

    opted_in = PinnedMcpOAuthHttpTransport(
        resolver=lambda _host, _port, _deadline: ("127.0.0.1",),
        allow_loopback_tls=True,
    )
    with pytest.raises(McpOAuthTransportError):
        opted_in.request(
            "GET",
            "https://localhost:9443/oauth",
            headers=None,
            body=None,
            deadline=time.monotonic() + 1,
            max_response_bytes=1024,
        )
    assert connected == [(('127.0.0.1', 9443),)]


def test_system_keyring_fails_closed_for_null_backend() -> None:
    class NullBackend:
        priority = 0

    class FakeKeyring:
        @staticmethod
        def get_keyring() -> NullBackend:
            return NullBackend()

    broker = SystemKeyringMcpCredentialBroker(keyring_module=FakeKeyring())
    assert broker.available() is False
    with pytest.raises(McpOAuthError, match="secure credential backend unavailable"):
        broker.put_secret("oauth:test", b"secret", expires_at=None)


def test_system_keyring_fails_closed_for_innocent_positive_priority_plaintext_backend(
) -> None:
    class EnterpriseVaultBackend:
        priority = 100

        def set_password(self, _service: str, _account: str, value: str) -> None:
            self.plaintext = value

    class FakeKeyring:
        @staticmethod
        def get_keyring() -> EnterpriseVaultBackend:
            return EnterpriseVaultBackend()

    broker = SystemKeyringMcpCredentialBroker(keyring_module=FakeKeyring())
    assert broker.available() is False
    with pytest.raises(McpOAuthError, match="secure credential backend unavailable"):
        broker.put_secret("oauth:test", b"must-not-reach-plaintext", expires_at=None)


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("keyring.backends.macOS", "Keyring"),
        ("keyring.backends.Windows", "WinVaultKeyring"),
        ("keyring.backends.SecretService", "Keyring"),
        ("keyring.backends.libsecret", "Keyring"),
        ("keyring.backends.kwallet", "DBusKeyring"),
        ("keyring.backends.kwallet", "DBusKeyringKWallet4"),
    ],
)
def test_system_keyring_accepts_only_exact_reviewed_backend_objects_without_os_io(
    module_name: str,
    class_name: str,
) -> None:
    backend_type = getattr(importlib.import_module(module_name), class_name)
    backend = object.__new__(backend_type)

    class FakeKeyring:
        @staticmethod
        def get_keyring() -> object:
            return backend

    assert SystemKeyringMcpCredentialBroker(
        keyring_module=FakeKeyring()
    ).available()


def test_system_keyring_rejects_exact_identity_lookalike_subclass() -> None:
    official = importlib.import_module("keyring.backends.macOS").Keyring

    class Keyring(official):
        priority = 100

    Keyring.__module__ = official.__module__
    Keyring.__qualname__ = official.__qualname__
    backend = object.__new__(Keyring)

    class FakeKeyring:
        @staticmethod
        def get_keyring() -> object:
            return backend

    assert not SystemKeyringMcpCredentialBroker(
        keyring_module=FakeKeyring()
    ).available()


def test_system_keyring_rejects_chainer_even_when_it_contains_os_backends() -> None:
    chainer_type = importlib.import_module(
        "keyring.backends.chainer"
    ).ChainerBackend
    backend = object.__new__(chainer_type)

    class FakeKeyring:
        @staticmethod
        def get_keyring() -> object:
            return backend

    assert not SystemKeyringMcpCredentialBroker(
        keyring_module=FakeKeyring()
    ).available()


def test_system_keyring_rejects_unreviewed_distribution_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnreviewedDistribution:
        version = "25.7.1"

    monkeypatch.setattr(
        oauth_module.importlib_metadata,
        "distribution",
        lambda _name: UnreviewedDistribution(),
    )
    oauth_module._audited_system_keyring_backend_types.cache_clear()
    try:
        assert not SystemKeyringMcpCredentialBroker(
            keyring_module=_FakeKeyring()
        ).available()
    finally:
        # The monkeypatch is restored after this test; leave no cached result
        # derived from the temporary distribution view.
        oauth_module._audited_system_keyring_backend_types.cache_clear()


def test_system_keyring_rejects_unreviewed_backend_source_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, *remaining = oauth_module._AUDITED_KEYRING_BACKENDS
    monkeypatch.setattr(
        oauth_module,
        "_AUDITED_KEYRING_BACKENDS",
        ((first[0], first[1], first[2], "0" * 64), *remaining),
    )
    oauth_module._audited_system_keyring_backend_types.cache_clear()
    try:
        assert not SystemKeyringMcpCredentialBroker(
            keyring_module=_FakeKeyring()
        ).available()
    finally:
        oauth_module._audited_system_keyring_backend_types.cache_clear()


def test_system_keyring_rejects_corrupt_empty_secret_envelope() -> None:
    class FakeKeyring(_FakeKeyring):
        @staticmethod
        def get_password(_service: str, _secret_ref: str) -> str:
            return '{"expires_at":null,"value":""}'

    broker = SystemKeyringMcpCredentialBroker(keyring_module=FakeKeyring())
    with pytest.raises(McpOAuthError, match="MCP credential is unavailable"):
        broker.get_secret("keyring:" + ("A" * 32))


def test_brokers_reserve_exact_staged_slots_without_aliasing_replacements() -> None:
    for broker in (
        InMemoryMcpCredentialBroker(),
        SystemKeyringMcpCredentialBroker(keyring_module=_FakeKeyring()),
    ):
        first = broker.reserve_secret_ref("mcp.remote-task.state.local-ref")
        second = broker.reserve_secret_ref("mcp.remote-task.state.local-ref")
        assert first != second
        broker.put_secret_at(
            first,
            "mcp.remote-task.state.local-ref",
            b"first",
            expires_at=None,
        )
        broker.put_secret_at(
            first,
            "mcp.remote-task.state.local-ref",
            b"first",
            expires_at=None,
        )
        with pytest.raises(McpOAuthError, match="already contains"):
            broker.put_secret_at(
                first,
                "mcp.remote-task.state.local-ref",
                b"conflicting-replacement",
                expires_at=None,
            )
        broker.put_secret_at(
            second,
            "mcp.remote-task.state.local-ref",
            b"second",
            expires_at=None,
        )
        broker.delete_secret(first)
        assert broker.get_secret(second) == b"second"

        oauth_first = broker.reserve_secret_ref("oauth:profile:challenge")
        oauth_second = broker.reserve_secret_ref("oauth:profile:challenge")
        assert oauth_first == oauth_second
        with pytest.raises(McpOAuthError, match="slot does not match"):
            broker.put_secret_at(
                oauth_first,
                "oauth:another:challenge",
                b"mismatch",
                expires_at=None,
            )
        broker.delete_secret(second)


def test_keyring_crash_rebind_restores_token_but_never_browser_challenge() -> None:
    keyring = _FakeKeyring()
    profile = _profile(
        token_endpoint_auth_method=McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_BASIC
    )
    responses = _metadata_responses(
        token_responses=[
            _json_response(
                {
                    "access_token": "restart-access",
                    "refresh_token": "restart-refresh",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "files:read",
                }
            )
        ]
    )
    responses[("GET", RESOURCE_METADATA)] *= 2
    responses[("GET", OAUTH_METADATA)] *= 2
    transport = ScriptedTransport(responses)
    first_broker = SystemKeyringMcpCredentialBroker(keyring_module=keyring)
    first = McpOAuthManager(broker=first_broker, transport=transport)
    first.add_profile(profile, client_secret=b"restart-client-secret")
    challenge = first.begin(profile.profile_id)
    query = parse_qs(urlsplit(challenge.authorization_url).query)
    first.complete(challenge.challenge_id, _callback(query))
    pending = first.begin(profile.profile_id)
    pending_ref = first_broker.reserve_secret_ref(
        f"oauth:{profile.profile_id}:challenge"
    )
    assert first_broker.get_secret(pending_ref)

    # Simulate a process crash: no first.close().  A new Runtime receives the
    # same explicit Host profile but not the prior client secret or challenge.
    second_broker = SystemKeyringMcpCredentialBroker(keyring_module=keyring)
    second = McpOAuthManager(
        broker=second_broker,
        transport=ScriptedTransport({}),
    )
    restored = second.add_profile(profile)
    assert restored.status is McpOAuthStatusKind.AUTHORIZED
    assert second.access_token(profile.profile_id, min_validity_s=0) == b"restart-access"
    with pytest.raises(McpOAuthError, match="credential is unavailable"):
        second_broker.get_secret(pending_ref)
    with pytest.raises(McpOAuthError, match="challenge is unavailable"):
        second.challenge_profile_id(pending.challenge_id)

    second.close()
    third = McpOAuthManager(
        broker=SystemKeyringMcpCredentialBroker(keyring_module=keyring),
        transport=ScriptedTransport({}),
    )
    assert third.add_profile(profile).status is McpOAuthStatusKind.AUTHORIZED
    assert third.logout(profile.profile_id).status is McpOAuthStatusKind.REVOKED
    with pytest.raises(McpOAuthAuthorizationRequired):
        third.access_token(profile.profile_id)
    third.remove_profile(profile.profile_id)
    assert keyring.values == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {"client_id": "different-client"},
        {
            "token_endpoint_auth_method": (
                McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_POST
            )
        },
        {"redirect_uri": "http://127.0.0.1:49153/oauth/callback"},
        {"audience": RESOURCE},
        {"protected_resource_metadata_url": RESOURCE_METADATA},
        {"authorization_server_metadata_url": OAUTH_METADATA},
        {"authorization_server_metadata_sha256": "a" * 64},
        {"allowed_endpoint_origins": ("https://auth.example.test",)},
        {"allowed_scopes": ("files:read",)},
        {"default_scopes": ()},
        {"server_id": "changed-files-server"},
        {"allow_loopback_http": True},
    ],
    ids=(
        "client-id",
        "auth-method",
        "redirect",
        "audience",
        "resource-metadata-url",
        "authorization-metadata-url",
        "authorization-metadata-pin",
        "endpoint-origins",
        "allowed-scopes",
        "default-scopes",
        "server-id",
        "loopback-policy",
    ),
)
def test_keyring_rebind_rejects_every_exact_profile_authority_change(
    overrides: dict[str, object],
) -> None:
    keyring = _FakeKeyring()
    original = _profile(
        token_endpoint_auth_method=McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_BASIC
    )
    transport = ScriptedTransport(
        _metadata_responses(
            token_responses=[
                _json_response(
                    {
                        "access_token": "old-authority-access",
                        "refresh_token": "old-authority-refresh",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "files:read",
                    }
                )
            ]
        )
    )
    first = McpOAuthManager(
        broker=SystemKeyringMcpCredentialBroker(keyring_module=keyring),
        transport=transport,
    )
    first.add_profile(original, client_secret=b"old-client-secret")
    challenge = first.begin(original.profile_id)
    query = parse_qs(urlsplit(challenge.authorization_url).query)
    first.complete(challenge.challenge_id, _callback(query))

    changed = replace(original, **overrides)
    second = McpOAuthManager(
        broker=SystemKeyringMcpCredentialBroker(keyring_module=keyring),
        transport=ScriptedTransport({}),
    )
    status = second.add_profile(changed, client_secret=b"new-client-secret")
    assert status.status is McpOAuthStatusKind.AUTHORIZATION_REQUIRED
    with pytest.raises(McpOAuthAuthorizationRequired):
        second.access_token(changed.profile_id)
    decoded_slots = b"".join(
        base64.b64decode(json.loads(envelope)["value"], validate=True)
        for envelope in keyring.values.values()
    )
    assert b"old-authority-access" not in decoded_slots
    assert b"old-authority-refresh" not in decoded_slots
    assert b"old-client-secret" not in decoded_slots
    second.remove_profile(changed.profile_id)
    assert keyring.values == {}


def test_rehydrated_token_preserves_generation_and_refreshes_from_bound_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keyring = _FakeKeyring()
    responses = _metadata_responses(
        token_responses=[
            _json_response(
                {
                    "access_token": "expiring-access",
                    "refresh_token": "rotating-refresh",
                    "token_type": "Bearer",
                    "expires_in": 1,
                    "scope": "files:read",
                }
            ),
            _json_response(
                {
                    "access_token": "refreshed-after-restart",
                    "refresh_token": "rotated-after-restart",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "files:read",
                }
            ),
        ]
    )
    transport = ScriptedTransport(responses)
    first = McpOAuthManager(
        broker=SystemKeyringMcpCredentialBroker(keyring_module=keyring),
        transport=transport,
    )
    first.add_profile(_profile())
    challenge = first.begin("work-account")
    query = parse_qs(urlsplit(challenge.authorization_url).query)
    first.complete(challenge.challenge_id, _callback(query))
    assert first.credential_generation("work-account") == 1
    first.close()

    now = time.time()
    monkeypatch.setattr("agent_libos.mcp.oauth.time.time", lambda: now + 2)
    second = McpOAuthManager(
        broker=SystemKeyringMcpCredentialBroker(keyring_module=keyring),
        transport=transport,
    )
    assert second.add_profile(_profile()).status is McpOAuthStatusKind.EXPIRED
    assert second.credential_generation("work-account") == 1
    second.set_minimum_credential_generation("work-account", 1)
    assert second.access_token("work-account") == b"refreshed-after-restart"
    assert second.credential_generation("work-account") == 2
    second.remove_profile("work-account")
    assert keyring.values == {}


def test_durable_newer_generation_purges_stale_rehydrated_bundle() -> None:
    keyring = _FakeKeyring()
    transport = ScriptedTransport(
        _metadata_responses(
            token_responses=[
                _json_response(
                    {
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "files:read",
                    }
                )
            ]
        )
    )
    first = McpOAuthManager(
        broker=SystemKeyringMcpCredentialBroker(keyring_module=keyring),
        transport=transport,
    )
    first.add_profile(_profile())
    challenge = first.begin("work-account")
    query = parse_qs(urlsplit(challenge.authorization_url).query)
    first.complete(challenge.challenge_id, _callback(query))

    second = McpOAuthManager(
        broker=SystemKeyringMcpCredentialBroker(keyring_module=keyring),
        transport=ScriptedTransport({}),
    )
    assert second.add_profile(_profile()).status is McpOAuthStatusKind.AUTHORIZED
    second.set_minimum_credential_generation("work-account", 2)
    assert second.status("work-account").status is McpOAuthStatusKind.NEEDS_ATTENTION
    with pytest.raises(McpOAuthNeedsAttention):
        second.access_token("work-account")
    assert "stale-access" not in repr(keyring.values)
    assert "stale-refresh" not in repr(keyring.values)
    second.remove_profile("work-account")
    assert keyring.values == {}


def test_duplicate_token_generation_after_crash_fails_closed() -> None:
    keyring = _FakeKeyring()
    broker = SystemKeyringMcpCredentialBroker(keyring_module=keyring)
    transport = ScriptedTransport(
        _metadata_responses(
            token_responses=[
                _json_response(
                    {
                        "access_token": "ambiguous-access",
                        "refresh_token": "ambiguous-refresh",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "files:read",
                    }
                )
            ]
        )
    )
    first = McpOAuthManager(broker=broker, transport=transport)
    first.add_profile(_profile())
    challenge = first.begin("work-account")
    query = parse_qs(urlsplit(challenge.authorization_url).query)
    first.complete(challenge.challenge_id, _callback(query))

    slot_one = broker.reserve_secret_ref("oauth:work-account:tokens:1")
    slot_zero = broker.reserve_secret_ref("oauth:work-account:tokens:0")
    service = "agent-libos.mcp.oauth.v1"
    keyring.values[(service, slot_zero)] = keyring.values[(service, slot_one)]

    second = McpOAuthManager(
        broker=SystemKeyringMcpCredentialBroker(keyring_module=keyring),
        transport=ScriptedTransport({}),
    )
    assert second.add_profile(_profile()).status is McpOAuthStatusKind.NEEDS_ATTENTION
    with pytest.raises(McpOAuthNeedsAttention):
        second.access_token("work-account")
    assert keyring.values == {}


def test_keyring_close_and_expiry_purge_pkce_challenge_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keyring = _FakeKeyring()
    responses = _metadata_responses()
    responses[("GET", RESOURCE_METADATA)] *= 2
    responses[("GET", OAUTH_METADATA)] *= 2
    broker = SystemKeyringMcpCredentialBroker(keyring_module=keyring)
    first = McpOAuthManager(
        broker=broker,
        transport=ScriptedTransport(responses),
        challenge_ttl_s=30,
    )
    first.add_profile(_profile())
    first.begin("work-account")
    challenge_ref = broker.reserve_secret_ref("oauth:work-account:challenge")
    assert broker.get_secret(challenge_ref)
    first.close()
    with pytest.raises(McpOAuthError, match="credential is unavailable"):
        broker.get_secret(challenge_ref)

    second = McpOAuthManager(
        broker=broker,
        transport=first._transport,  # noqa: SLF001 - reuse remaining scripted metadata
        challenge_ttl_s=30,
    )
    second.add_profile(_profile())
    challenge = second.begin("work-account")
    now = time.time()
    monkeypatch.setattr("agent_libos.mcp.oauth.time.time", lambda: now + 31)
    with pytest.raises(McpOAuthError, match="challenge is unavailable"):
        second.challenge_profile_id(challenge.challenge_id)
    with pytest.raises(McpOAuthError, match="credential is unavailable"):
        broker.get_secret(challenge_ref)


@pytest.mark.parametrize("operation", ["logout", "revoke"])
def test_logout_and_local_revoke_purge_pending_pkce_state(operation: str) -> None:
    broker = InMemoryMcpCredentialBroker()
    manager = McpOAuthManager(
        broker=broker,
        transport=ScriptedTransport(_metadata_responses()),
    )
    manager.add_profile(_profile())
    challenge = manager.begin("work-account")
    challenge_ref = broker.reserve_secret_ref("oauth:work-account:challenge")
    assert broker.get_secret(challenge_ref)

    selected = getattr(manager, operation)("work-account")
    assert selected.status is McpOAuthStatusKind.REVOKED
    with pytest.raises(McpOAuthError, match="challenge is unavailable"):
        manager.challenge_profile_id(challenge.challenge_id)
    with pytest.raises(McpOAuthError, match="credential is unavailable"):
        broker.get_secret(challenge_ref)
    assert broker._secrets == {}  # noqa: SLF001 - exact lifecycle assertion


def test_profile_mapping_parser_is_exact_and_contains_no_secret_field() -> None:
    raw = {
        "profile_id": "work-account",
        "server_id": "files",
        "resource_uri": RESOURCE,
        "expected_issuer": ISSUER,
        "redirect_uri": REDIRECT,
        "client_id": "agent-libos-desktop",
        "registration_mode": "preregistered",
        "allowed_scopes": ["files:read"],
        "default_scopes": ["files:read"],
    }
    assert mcp_oauth_profile_from_mapping(raw).allowed_scopes == ("files:read",)
    with pytest.raises(McpOAuthError, match="fields are invalid"):
        mcp_oauth_profile_from_mapping({**raw, "client_secret": "PRIVATE"})
    with pytest.raises(McpOAuthError, match="field is invalid"):
        mcp_oauth_profile_from_mapping({**raw, "allow_loopback_http": 1})


def test_manager_structurally_checks_injected_broker_and_sanitizes_availability() -> None:
    with pytest.raises(McpOAuthError, match="invalid MCP credential broker"):
        McpOAuthManager(broker=object(), transport=ScriptedTransport({}))  # type: ignore[arg-type]

    reflected = "backend-secret-error"

    class ExplodingBroker(InMemoryMcpCredentialBroker):
        def available(self) -> bool:
            raise RuntimeError(reflected)

    manager = McpOAuthManager(
        broker=ExplodingBroker(),
        transport=ScriptedTransport(_metadata_responses()),
    )
    with pytest.raises(McpOAuthError) as raised:
        manager.add_profile(_profile())
    assert reflected not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_broker_read_exception_chain_is_not_propagated() -> None:
    reflected = "broker-read-secret-error"

    class ExplodingReadBroker(InMemoryMcpCredentialBroker):
        explode = False

        def get_secret(self, secret_ref: str) -> bytes:
            if self.explode:
                failure = RuntimeError("sanitized-wrapper")
                failure.__cause__ = RuntimeError(reflected)
                raise failure
            return super().get_secret(secret_ref)

    broker = ExplodingReadBroker()
    manager = McpOAuthManager(
        broker=broker,
        transport=ScriptedTransport(_metadata_responses()),
    )
    manager.add_profile(_profile())
    challenge = manager.begin("work-account")
    broker.explode = True

    with pytest.raises(McpOAuthError) as raised:
        manager.complete(challenge.challenge_id, f"{REDIRECT}?code=x&state=x")

    assert reflected not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_in_memory_broker_is_explicitly_test_only_and_deletes_secrets() -> None:
    broker = InMemoryMcpCredentialBroker()
    ref = broker.put_secret("oauth:test", b"secret", expires_at=None)
    assert broker.get_secret(ref) == b"secret"
    broker.delete_secret(ref)
    with pytest.raises(McpOAuthError, match="credential is unavailable"):
        broker.get_secret(ref)
