from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.mcp.manifest import (
    McpResourceSpec,
    McpServerManifestV3,
)
from agent_libos.mcp.oauth import (
    InMemoryMcpCredentialBroker,
    McpOAuthError,
    McpOAuthHttpResponse,
    McpOAuthNeedsAttention,
)
from agent_libos.mcp.types import McpOAuthStatusKind
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.mcp import (
    McpHttpTransportSpec,
    McpProtocolMode,
)
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.serde import to_jsonable

from tests.unit.test_mcp_oauth import (
    ISSUER,
    REVOCATION_ENDPOINT,
    RESOURCE,
    ScriptedTransport,
    _callback,
    _json_response,
    _metadata_responses,
    _profile,
)


def _runtime_config():
    return replace(
        DEFAULT_CONFIG,
        mcp=replace(DEFAULT_CONFIG.mcp, oauth_enabled=True),
    )


def _manifest() -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id="files",
        transport="streamable_http",
        http=McpHttpTransportSpec(url=RESOURCE),
        timeout_s=5.0,
        max_request_bytes=64 * 1024,
        max_response_bytes=256 * 1024,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        resources=(
            McpResourceSpec(
                resource_id="status",
                remote_uri=f"{RESOURCE}/status",
            ),
        ),
        auth_profile_id="work-account",
    )


def _substrate(root: Path, broker: InMemoryMcpCredentialBroker, transport: object):
    selected = LocalResourceProviderSubstrate(root)
    selected.mcp_credential_broker = broker
    selected.mcp_oauth_transport = transport
    return selected


def test_runtime_host_oauth_profile_binding_protected_flow_and_restart_secret_scan(
    tmp_path: Path,
) -> None:
    database = tmp_path / "oauth-runtime.sqlite"
    access = "runtime-access-token-SENTINEL"
    refresh = "runtime-refresh-token-SENTINEL"
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
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        database,
        config=_runtime_config(),
        substrate=_substrate(tmp_path, broker, transport),
    )
    state = ""
    verifier = ""
    try:
        manifest = _manifest()
        before_audit = runtime.audit.trace()
        with pytest.raises(ValidationError, match="not Host-configured"):
            runtime.mcp.register_server(
                manifest,
                actor="host",
                require_capability=False,
            )
        assert runtime.mcp.list_servers(require_capability=False) == []
        assert runtime.audit.trace() == before_audit

        provisional = runtime.mcp.add_oauth_profile(_profile(), actor="host")
        assert provisional.status is McpOAuthStatusKind.AUTHORIZATION_REQUIRED
        assert runtime.uow.mcp_auth.get("work-account") is None
        runtime.mcp.register_server(
            manifest,
            actor="host",
            require_capability=False,
        )
        initial = runtime.uow.mcp_auth.get("work-account")
        assert initial is not None
        assert initial.server_id == manifest.server_id
        assert initial.status == "authorization_required"
        assert initial.issuer_sha256 is not None
        assert ISSUER not in repr(initial)
        assert RESOURCE not in repr(initial)

        challenge = runtime.mcp.auth_begin("work-account", actor="host")
        query = parse_qs(urlsplit(challenge.authorization_url).query)
        state = query["state"][0]
        status = runtime.mcp.auth_complete(
            challenge.challenge_id,
            _callback(query),
            actor="host",
        )
        assert status.status is McpOAuthStatusKind.AUTHORIZED
        request_form = parse_qs(transport.requests[-1]["body"].decode("utf-8"))
        verifier = request_form["code_verifier"][0]

        persisted = runtime.uow.mcp_auth.get("work-account")
        assert persisted is not None
        assert persisted.status == "authorized"
        assert persisted.credential_generation > initial.credential_generation
        assert persisted.principal_sha256 is None

        evidence = json.dumps(
            {
                "audit": to_jsonable(runtime.audit.trace()),
                "events": to_jsonable(runtime.events.list()),
                "effects": to_jsonable(runtime.store.list_external_effects()),
                "auth": persisted.to_dict(),
            },
            sort_keys=True,
        )
        for secret in (access, refresh, state, verifier, "authorization-code"):
            assert secret not in evidence
        oauth_effects = [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.provider == "mcp" and effect.operation.startswith("auth.")
        ]
        assert [effect.operation for effect in oauth_effects] == [
            "auth.begin",
            "auth.complete",
        ]
        assert oauth_effects[-1].state_mutation is True
    finally:
        runtime.close()


    for candidate in (database, *database.parent.glob(f"{database.name}-*")):
        if not candidate.exists():
            continue
        raw = candidate.read_bytes()
        for secret in (access, refresh, state, verifier, "authorization-code"):
            assert secret.encode("utf-8") not in raw

    reopened = Runtime.open(
        database,
        config=_runtime_config(),
        substrate=_substrate(
            tmp_path,
            InMemoryMcpCredentialBroker(),
            ScriptedTransport({}),
        ),
    )
    try:
        restarted = reopened.mcp.auth_status("work-account", actor="host")
        assert restarted.status is McpOAuthStatusKind.NEEDS_ATTENTION
        durable = reopened.uow.mcp_auth.get("work-account")
        assert durable is not None
        assert durable.status == "needs_attention"
        old_generation = durable.credential_generation

        reconfigured = reopened.mcp.add_oauth_profile(_profile(), actor="host")
        assert reconfigured.status is McpOAuthStatusKind.AUTHORIZATION_REQUIRED
        rebound = reopened.uow.mcp_auth.get("work-account")
        assert rebound is not None
        assert rebound.credential_generation >= old_generation
        assert rebound.status == "authorization_required"
    finally:
        reopened.close()


def test_runtime_revoke_failure_is_pending_first_not_retried_and_persists_attention(
    tmp_path: Path,
) -> None:
    access = "revoke-access-SENTINEL"
    refresh = "revoke-refresh-SENTINEL"
    responses = _metadata_responses(
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
    responses[("POST", REVOCATION_ENDPOINT)] = [
        McpOAuthHttpResponse(
            status=503,
            headers={"content-type": "application/json"},
            body=(f'{{"error":"temporarily_unavailable","token":"{refresh}"}}').encode(),
        )
    ]
    transport = ScriptedTransport(responses)
    runtime = Runtime.open(
        tmp_path / "oauth-revoke.sqlite",
        config=_runtime_config(),
        substrate=_substrate(tmp_path, InMemoryMcpCredentialBroker(), transport),
    )
    try:
        runtime.mcp.add_oauth_profile(_profile(), actor="host")
        runtime.mcp.register_server(
            _manifest(),
            actor="host",
            require_capability=False,
        )
        challenge = runtime.mcp.auth_begin("work-account", actor="host")
        query = parse_qs(urlsplit(challenge.authorization_url).query)
        runtime.mcp.auth_complete(
            challenge.challenge_id,
            _callback(query),
            actor="host",
        )

        before = len(runtime.store.list_external_effects())
        with pytest.raises(McpOAuthNeedsAttention):
            runtime.mcp.auth_revoke("work-account", actor="host")

        revoke_requests = [
            request
            for request in transport.requests
            if request["url"] == REVOCATION_ENDPOINT
        ]
        assert len(revoke_requests) == 1
        durable = runtime.uow.mcp_auth.get("work-account")
        assert durable is not None
        assert durable.status == "needs_attention"
        assert durable.metadata == {"reason_code": "needs_attention"}
        effects = runtime.store.list_external_effects()[before:]
        assert len(effects) == 1
        assert effects[0].operation == "auth.revoke"
        assert effects[0].state_mutation is True
        assert effects[0].provider_metadata["automatic_retry_disabled"] is True
        evidence = json.dumps(
            {
                "effect": to_jsonable(effects[0]),
                "audit": to_jsonable(runtime.audit.trace()),
                "events": to_jsonable(runtime.events.list()),
                "auth": durable.to_dict(),
            },
            sort_keys=True,
        )
        assert access not in evidence
        assert refresh not in evidence
    finally:
        runtime.close()


def test_runtime_rejects_late_oauth_success_and_discards_unreturned_challenge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        tmp_path / "oauth-late-provider.sqlite",
        config=_runtime_config(),
        substrate=_substrate(
            tmp_path,
            broker,
            ScriptedTransport(_metadata_responses()),
        ),
    )
    issued: list[Any] = []
    expired_deadline: list[float] = []
    real_monotonic = time.monotonic
    try:
        runtime.mcp.add_oauth_profile(_profile(), actor="host")
        runtime.mcp.register_server(
            _manifest(),
            actor="host",
            require_capability=False,
        )
        manager = runtime._mcp_oauth_manager
        original_begin = manager.begin

        def late_begin(*args: object, **kwargs: object):
            challenge = original_begin(*args, **kwargs)
            issued.append(challenge)
            expired_deadline.append(float(kwargs["deadline"]))
            return challenge

        def controlled_monotonic() -> float:
            if expired_deadline:
                return expired_deadline[0] + 1.0
            return real_monotonic()

        monkeypatch.setattr(manager, "begin", late_begin)
        monkeypatch.setattr(
            "agent_libos.primitives.mcp.time.monotonic",
            controlled_monotonic,
        )

        with pytest.raises(
            TimeoutError,
            match="OAuth provider exceeded the absolute deadline",
        ):
            runtime.mcp.auth_begin("work-account", actor="host")

        assert len(issued) == 1
        with pytest.raises(McpOAuthError, match="challenge is unavailable"):
            manager.challenge_profile_id(issued[0].challenge_id)
        challenge_ref = broker.reserve_secret_ref("oauth:work-account:challenge")
        with pytest.raises(McpOAuthError, match="credential is unavailable"):
            broker.get_secret(challenge_ref)
        effects = [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.operation == "auth.begin"
        ]
        assert len(effects) == 1
        assert effects[0].provider_metadata["ok"] is False
    finally:
        runtime.close()


def test_runtime_discards_unreturned_challenge_when_protected_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        tmp_path / "oauth-begin-commit-failure.sqlite",
        config=_runtime_config(),
        substrate=_substrate(
            tmp_path,
            broker,
            ScriptedTransport(_metadata_responses()),
        ),
    )
    issued: list[Any] = []
    try:
        runtime.mcp.add_oauth_profile(_profile(), actor="host")
        runtime.mcp.register_server(
            _manifest(),
            actor="host",
            require_capability=False,
        )
        manager = runtime._mcp_oauth_manager
        original_begin = manager.begin
        original_record = runtime.audit.record

        def capture_begin(*args: object, **kwargs: object):
            challenge = original_begin(*args, **kwargs)
            issued.append(challenge)
            return challenge

        def fail_result_audit(*args: object, **kwargs: object):
            if kwargs.get("action") == "primitive.mcp.auth.begin":
                raise RuntimeError("injected OAuth result audit failure")
            return original_record(*args, **kwargs)

        monkeypatch.setattr(manager, "begin", capture_begin)
        monkeypatch.setattr(runtime.audit, "record", fail_result_audit)

        with pytest.raises(RuntimeError, match="OAuth result audit failure"):
            runtime.mcp.auth_begin("work-account", actor="host")

        assert len(issued) == 1
        with pytest.raises(McpOAuthError, match="challenge is unavailable"):
            manager.challenge_profile_id(issued[0].challenge_id)
        challenge_ref = broker.reserve_secret_ref("oauth:work-account:challenge")
        with pytest.raises(McpOAuthError, match="credential is unavailable"):
            broker.get_secret(challenge_ref)
    finally:
        runtime.close()
