from __future__ import annotations

import http.client
import ipaddress
import json
import os
import socket
import ssl
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

import anyio
import pytest

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
from agent_libos.mcp.types import McpComplete, McpTextContent
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.serde import to_jsonable


pytestmark = [pytest.mark.mcp, pytest.mark.mcp_transport]

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "mcp_sdk_v2" / "oauth_tls_server.py"
RESOURCE_URI = "fixture://oauth/status"
ACCESS_TOKEN_SENTINEL = "agent-libos-oauth-tls-access-token"
AUTHORIZATION_CODE_SENTINEL = "agent-libos-oauth-tls-code"


def test_runtime_oauth_pkce_tls_and_bearer_transport_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the Host-pinned OAuth path without trusting runner discovery."""

    ca_path, cert_path, key_path = _write_test_certificates(tmp_path)
    # The OAuth transport receives an explicit Host-owned TLS context below.  The
    # independent MCP SDK HTTP transport uses the process trust store, so pin the
    # same ephemeral CA before Runtime construction instead of disabling TLS
    # verification for the integration gate.
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_path))
    port = _available_loopback_port()
    evidence_path = tmp_path / "oauth-tls-evidence.json"
    with _oauth_tls_fixture(
        port=port,
        cert_path=cert_path,
        key_path=key_path,
        evidence_path=evidence_path,
        ca_path=ca_path,
    ):
        origin = f"https://localhost:{port}"
        resource_url = f"{origin}/mcp"
        ssl_context = ssl.create_default_context(cafile=str(ca_path))
        oauth_transport = PinnedMcpOAuthHttpTransport(
            resolver=lambda host, selected_port, _deadline: _fixture_addresses(
                host, selected_port, expected_port=port
            ),
            allow_loopback_http=True,
            allow_loopback_tls=True,
            ssl_context=ssl_context,
        )
        broker = InMemoryMcpCredentialBroker()
        substrate = LocalResourceProviderSubstrate(tmp_path)
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
                profile_id="oauth-tls-profile",
                server_id="oauth-tls-server",
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
                actor="oauth-tls-host",
            )
            assert provisional.status is McpOAuthStatusKind.AUTHORIZATION_REQUIRED
            runtime.mcp.register_server(
                _manifest(resource_url),
                actor="oauth-tls-host",
                require_capability=False,
            )

            challenge = runtime.mcp.auth_begin(
                profile.profile_id,
                scopes=("mcp.read",),
                actor="oauth-tls-host",
            )
            callback_url = _visit_authorization_endpoint(
                challenge.authorization_url,
                ssl_context=ssl_context,
            )
            status = runtime.mcp.auth_complete(
                challenge.challenge_id,
                callback_url,
                actor="oauth-tls-host",
            )
            assert status.status is McpOAuthStatusKind.AUTHORIZED
            assert status.scopes == ("mcp.read",)

            resources = runtime.mcp.list_resources(
                profile.server_id,
                actor="oauth-tls-host",
            )
            assert [item.resource_id for item in resources.items] == ["status"]
            result = runtime.mcp.read_resource(
                profile.server_id,
                "status",
                actor="oauth-tls-host",
            )
            assert isinstance(result, McpComplete)
            assert result.value is not None
            assert len(result.value.contents) == 1
            assert isinstance(result.value.contents[0], McpTextContent)
            assert "OAuth TLS Runtime path authorized" in result.value.contents[0].text

            evidence = _await_fixture_evidence(evidence_path)
            assert evidence == {
                "authorization_client_id_pinned": True,
                "authorization_pkce_s256": True,
                "authorization_redirect_pinned": True,
                "authorization_resource_pinned": True,
                "authorization_scope_pinned": True,
                "authorization_server_metadata_served": True,
                "authorization_state_present": True,
                "mcp_bearer_verified": True,
                "protected_resource_metadata_served": True,
                "token_client_id_pinned": True,
                "token_code_bound": True,
                "token_pkce_verified": True,
                "token_redirect_pinned": True,
                "token_resource_pinned": True,
            }
            durable = runtime.uow.mcp_auth.get(profile.profile_id)
            assert durable is not None
            public_evidence = json.dumps(
                {
                    "audit": to_jsonable(runtime.audit.trace()),
                    "events": to_jsonable(runtime.events.list()),
                    "effects": to_jsonable(runtime.store.list_external_effects()),
                    "auth": durable.to_dict(),
                },
                sort_keys=True,
            )
            assert ACCESS_TOKEN_SENTINEL not in public_evidence
            assert AUTHORIZATION_CODE_SENTINEL not in public_evidence
            assert challenge.challenge_id not in public_evidence
            assert runtime._mcp_oauth_transport is oauth_transport
            assert anyio.run(runtime._mcp_connection_supervisor.snapshot) == ()
        finally:
            runtime.close()
        assert anyio.run(runtime._mcp_connection_supervisor.snapshot) == ()


def _manifest(resource_url: str) -> dict[str, object]:
    return {
        "schema_version": 3,
        "server_id": "oauth-tls-server",
        "transport": "streamable_http",
        "protocol_mode": "2026-07-28",
        "http": {"url": resource_url},
        "resources": [
            {
                "resource_id": "status",
                "remote_uri": RESOURCE_URI,
                "right": "read",
                "information_flow": True,
                "model_visible": False,
                "mime_types": ["text/plain"],
            }
        ],
        "auth_profile_id": "oauth-tls-profile",
        "subscriptions": [],
        "timeout_s": 10,
        "max_request_bytes": 65_536,
        "max_response_bytes": 1_048_576,
    }


def _fixture_addresses(
    host: str,
    port: int,
    *,
    expected_port: int,
) -> tuple[str, ...]:
    if host != "localhost" or port != expected_port:
        raise AssertionError("OAuth fixture transport attempted an unpinned endpoint")
    return ("127.0.0.1",)


def _visit_authorization_endpoint(
    authorization_url: str,
    *,
    ssl_context: ssl.SSLContext,
) -> str:
    parsed = urlsplit(authorization_url)
    assert parsed.scheme == "https"
    assert parsed.hostname == "localhost"
    assert parsed.port is not None
    connection = http.client.HTTPSConnection(
        parsed.hostname,
        parsed.port,
        context=ssl_context,
        timeout=5,
    )
    try:
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        connection.request("GET", target, headers={"Accept": "application/json"})
        response = connection.getresponse()
        response.read()
        assert response.status == 302
        location = response.getheader("Location")
        assert location is not None
        return location
    finally:
        connection.close()


def _available_loopback_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _await_fixture_evidence(path: Path) -> dict[str, bool]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
            continue
        if isinstance(value, dict) and value.get("mcp_bearer_verified") is True:
            return {str(key): bool(item) for key, item in value.items()}
        time.sleep(0.01)
    raise AssertionError("OAuth TLS fixture did not emit complete evidence")


@contextmanager
def _oauth_tls_fixture(
    *,
    port: int,
    cert_path: Path,
    key_path: Path,
    evidence_path: Path,
    ca_path: Path,
) -> Iterator[None]:
    environment = {
        name: os.environ[name]
        for name in (
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "SystemRoot",
            "WINDIR",
        )
        if os.environ.get(name)
    }
    environment.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "SSL_CERT_FILE": str(ca_path),
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(FIXTURE),
            "--port",
            str(port),
            "--cert",
            str(cert_path),
            "--key",
            str(key_path),
            "--evidence",
            str(evidence_path),
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        context = ssl.create_default_context(cafile=str(ca_path))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    "OAuth TLS fixture exited before readiness: "
                    f"stdout={stdout[-2000:]!r} stderr={stderr[-2000:]!r}"
                )
            try:
                connection = http.client.HTTPSConnection(
                    "localhost",
                    port,
                    context=context,
                    timeout=0.5,
                )
                connection.request(
                    "GET",
                    "/.well-known/oauth-protected-resource/mcp",
                )
                response = connection.getresponse()
                response.read()
                connection.close()
                if response.status == 200:
                    break
            except (OSError, ssl.SSLError, TimeoutError):
                time.sleep(0.02)
        else:
            raise AssertionError("OAuth TLS fixture did not become ready")
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        stdout, stderr = process.communicate()
        assert process.returncode in {0, -15}, (
            f"OAuth TLS fixture failed: stdout={stdout[-2000:]!r} "
            f"stderr={stderr[-2000:]!r}"
        )


def _write_test_certificates(tmp_path: Path) -> tuple[Path, Path, Path]:
    # ``cryptography`` belongs to the optional MCP SDK environment.  Import it
    # only when this transport test actually runs so default/postgres pytest
    # collection can still discover the invariant node without that extra.
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65_537, key_size=2048)
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "Agent libOS MCP test CA")]
    )
    ca_certificate = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    server_key = rsa.generate_private_key(public_exponent=65_537, key_size=2048)
    server_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]
    )
    server_certificate = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.IPAddress(ipaddress.ip_address("::1")),
                ]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    ca_path = tmp_path / "oauth-test-ca.pem"
    cert_path = tmp_path / "oauth-test-server.pem"
    key_path = tmp_path / "oauth-test-server-key.pem"
    ca_path.write_bytes(ca_certificate.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(server_certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return ca_path, cert_path, key_path
