"""Fail-closed Host OAuth support for MCP Streamable HTTP clients.

The module deliberately does not implement dynamic client registration.  A
Host must configure either a pre-registered client or an HTTPS Client ID
Metadata Document (CIMD) client.  Client credentials, Runtime-held PKCE/state
material, and tokens live at rest only behind :class:`McpCredentialBroker`.
An authorization code necessarily crosses the Host callback and token-request
path in transient memory; it is never written to RuntimeStore, broker storage,
audit/event evidence, output, or logs, and application-owned references are
released after the single exchange attempt.

Network discovery and token operations use one absolute deadline, never follow
redirects, reject ambient proxies, resolve every hostname under the MCP SSRF
policy, and connect to the exact validated address.  Token requests are never
retried.  An indeterminate refresh/revocation response therefore becomes
``needs_attention`` instead of risking replay of a rotating credential.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import http.client
import importlib
import inspect
import ipaddress
import json
import math
import re
import secrets
import socket
import ssl
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from agent_libos.mcp.types import (
    McpAuthorizationChallenge,
    McpOAuthStatus,
    McpOAuthStatusKind,
)
from agent_libos.mcp.providers import McpCredentialBroker
from agent_libos.models.base import StrEnum
from agent_libos.models.exceptions import ValidationError
from agent_libos.substrate.base import _bounded_provider_getaddrinfo
from agent_libos.utils.serde import bounded_json_loads


_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]{0,127}$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_REF_RE = re.compile(r"^(?:mem|keyring):[A-Za-z0-9_-]{32,128}$")
_FORBIDDEN_HOSTS = frozenset({"metadata.google.internal"})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_OAUTH_CALLBACK_KEYS = frozenset(
    {"code", "state", "iss", "error", "error_description", "error_uri"}
)
_MAX_SECRET_BYTES = 256 * 1024
_MAX_METADATA_BYTES = 64 * 1024
_MAX_TOKEN_BYTES = 128 * 1024
_MAX_FORM_BYTES = 64 * 1024
_MAX_URL_CHARS = 8 * 1024
_MAX_SCOPES = 128
_MAX_SCOPE_CHARS = 256
_MAX_TOKEN_CHARS = 96 * 1024
_DEFAULT_DEADLINE_S = 30.0
_CHALLENGE_TTL_S = 10 * 60.0
_MAX_TOKEN_LIFETIME_S = 366 * 24 * 60 * 60
_KEYRING_SERVICE = "agent-libos.mcp.oauth.v1"
_TOKEN_SLOT_COUNT = 2
_AUDITED_KEYRING_VERSION = "25.7.0"
_AUDITED_KEYRING_BACKENDS = (
    (
        "keyring.backends.macOS",
        "Keyring",
        "keyring/backends/macOS/__init__.py",
        "f8220e36fc2b2456deba3eb4a296c2319c38c16b621b21ee182b21a1c77835d8",
    ),
    (
        "keyring.backends.Windows",
        "WinVaultKeyring",
        "keyring/backends/Windows.py",
        "da98b72d2576442c17acb61e3699485152603cb1f9b8c9f261c481828fba926c",
    ),
    (
        "keyring.backends.SecretService",
        "Keyring",
        "keyring/backends/SecretService.py",
        "aadf654296bc87aac69e3cd3384f063080c7d9ed89e34448855df27354d74ac7",
    ),
    (
        "keyring.backends.libsecret",
        "Keyring",
        "keyring/backends/libsecret.py",
        "816794bde138e30647d23eeddb0d8bfa579832924e1075743aa494882fac1d01",
    ),
    (
        "keyring.backends.kwallet",
        "DBusKeyring",
        "keyring/backends/kwallet.py",
        "2def9bc1f25537b74d52230b60b13ae9ed07ccce609896695e798b4240e5084a",
    ),
    (
        "keyring.backends.kwallet",
        "DBusKeyringKWallet4",
        "keyring/backends/kwallet.py",
        "2def9bc1f25537b74d52230b60b13ae9ed07ccce609896695e798b4240e5084a",
    ),
)


class McpOAuthError(ValidationError):
    """Sanitized OAuth failure safe for Host-facing projections."""


class McpOAuthNeedsAttention(McpOAuthError):
    """An operation may have consumed or rotated a credential."""


class McpOAuthAuthorizationRequired(McpOAuthError):
    """No usable authorization grant exists for the selected profile."""


class McpOAuthTransportError(McpOAuthError):
    """Sanitized transport failure with a bounded dispatch classification."""

    def __init__(self, dispatch_state: Literal["not_started", "started", "unknown"]):
        if dispatch_state not in {"not_started", "started", "unknown"}:
            raise TypeError("invalid OAuth transport dispatch state")
        self.dispatch_state = dispatch_state
        super().__init__("MCP OAuth transport failed")


class McpOAuthRegistrationMode(StrEnum):
    PREREGISTERED = "preregistered"
    CIMD = "cimd"


class McpOAuthTokenEndpointAuthMethod(StrEnum):
    NONE = "none"
    CLIENT_SECRET_BASIC = "client_secret_basic"
    CLIENT_SECRET_POST = "client_secret_post"


@dataclass(frozen=True)
class McpOAuthProfile:
    """Host-owned, non-secret OAuth authority and binding configuration.

    A CIMD ``client_id`` is transmitted as an HTTPS URL but is deliberately not
    fetched by the Runtime: the authorization server is the protocol actor
    that fetches and validates that document.  Host onboarding/doctor tooling
    may validate the Host's deployed document as an explicit external action.
    """

    profile_id: str
    server_id: str
    resource_uri: str
    expected_issuer: str
    redirect_uri: str
    client_id: str
    registration_mode: McpOAuthRegistrationMode
    token_endpoint_auth_method: McpOAuthTokenEndpointAuthMethod = (
        McpOAuthTokenEndpointAuthMethod.NONE
    )
    allowed_scopes: tuple[str, ...] = ()
    default_scopes: tuple[str, ...] = ()
    audience: str | None = None
    protected_resource_metadata_url: str | None = None
    authorization_server_metadata_url: str | None = None
    protected_resource_metadata_sha256: str | None = None
    authorization_server_metadata_sha256: str | None = None
    allowed_endpoint_origins: tuple[str, ...] = ()
    allow_loopback_http: bool = False
    protocol_revision: Literal["2026-07-28"] = "2026-07-28"
    transport: Literal["streamable_http"] = "streamable_http"


def mcp_oauth_profile_from_mapping(value: Mapping[str, Any]) -> McpOAuthProfile:
    """Parse one exact non-secret Host profile mapping.

    Secret material is intentionally absent from this shape and must be sent
    separately to :meth:`McpOAuthManager.add_profile` through the broker path.
    """

    if type(value) is not dict:
        raise McpOAuthError("MCP OAuth profile must be an object")
    required = {
        "profile_id",
        "server_id",
        "resource_uri",
        "expected_issuer",
        "redirect_uri",
        "client_id",
        "registration_mode",
    }
    optional = {
        "token_endpoint_auth_method",
        "allowed_scopes",
        "default_scopes",
        "audience",
        "protected_resource_metadata_url",
        "authorization_server_metadata_url",
        "protected_resource_metadata_sha256",
        "authorization_server_metadata_sha256",
        "allowed_endpoint_origins",
        "allow_loopback_http",
        "protocol_revision",
        "transport",
    }
    keys = set(value)
    if keys.difference(required | optional) or required.difference(keys):
        raise McpOAuthError("MCP OAuth profile fields are invalid")
    text = {
        key: _profile_mapping_text(value, key)
        for key in (
            "profile_id",
            "server_id",
            "resource_uri",
            "expected_issuer",
            "redirect_uri",
            "client_id",
        )
    }
    profile = McpOAuthProfile(
        **text,
        registration_mode=_profile_mapping_enum(
            value,
            "registration_mode",
            McpOAuthRegistrationMode,
        ),
        token_endpoint_auth_method=_profile_mapping_enum(
            value,
            "token_endpoint_auth_method",
            McpOAuthTokenEndpointAuthMethod,
            default=McpOAuthTokenEndpointAuthMethod.NONE,
        ),
        allowed_scopes=_profile_mapping_strings(value, "allowed_scopes"),
        default_scopes=_profile_mapping_strings(value, "default_scopes"),
        audience=_profile_mapping_optional_text(value, "audience"),
        protected_resource_metadata_url=_profile_mapping_optional_text(
            value, "protected_resource_metadata_url"
        ),
        authorization_server_metadata_url=_profile_mapping_optional_text(
            value, "authorization_server_metadata_url"
        ),
        protected_resource_metadata_sha256=_profile_mapping_optional_text(
            value, "protected_resource_metadata_sha256"
        ),
        authorization_server_metadata_sha256=_profile_mapping_optional_text(
            value, "authorization_server_metadata_sha256"
        ),
        allowed_endpoint_origins=_profile_mapping_strings(
            value, "allowed_endpoint_origins"
        ),
        allow_loopback_http=_profile_mapping_bool(
            value, "allow_loopback_http", default=False
        ),
        protocol_revision=_profile_mapping_literal(
            value, "protocol_revision", "2026-07-28"
        ),
        transport=_profile_mapping_literal(
            value, "transport", "streamable_http"
        ),
    )
    _validate_profile(profile)
    return profile


@dataclass(frozen=True)
class McpOAuthHttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class McpOAuthCredentialFence:
    """Non-secret identity fence paired with one exact bearer credential."""

    profile_id: str
    server_id: str
    issuer: str
    resource: str
    scopes: tuple[str, ...]
    principal_sha256: str | None
    credential_generation: int


@dataclass(frozen=True)
class McpOAuthChallengeHints:
    """Sanitized, authority-neutral projection of one Bearer challenge."""

    resource_metadata_url: str | None
    scopes: tuple[str, ...]
    error_code: str | None = None


class McpOAuthAccessLease:
    """Short-lived transport-only bearer lease with a redacted representation."""

    __slots__ = ("_closed", "_sensitive_values", "_token", "fence")

    def __init__(
        self,
        token: bytes,
        fence: McpOAuthCredentialFence,
        *,
        sensitive_values: tuple[str, ...] = (),
    ) -> None:
        self._token = bytearray(token)
        self._sensitive_values = tuple(
            bytearray(value.encode("utf-8"))
            for value in sensitive_values
            if value
        )
        self.fence = fence
        self._closed = False

    def bearer_token(self) -> bytes:
        if self._closed:
            raise McpOAuthError("MCP OAuth access lease is closed")
        return bytes(self._token)

    def authorization_header(self) -> str:
        try:
            token = self.bearer_token().decode("utf-8")
        except UnicodeDecodeError:
            raise McpOAuthNeedsAttention(
                "MCP OAuth credential requires Host attention"
            ) from None
        return f"Bearer {token}"

    def redaction_values(self) -> tuple[str, ...]:
        """Return exact non-bearer secrets only to the active transport guard."""

        if self._closed:
            raise McpOAuthError("MCP OAuth access lease is closed")
        try:
            return tuple(value.decode("utf-8") for value in self._sensitive_values)
        except UnicodeDecodeError:
            raise McpOAuthNeedsAttention(
                "MCP OAuth credential requires Host attention"
            ) from None

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        token = getattr(self, "_token", None)
        if isinstance(token, bytearray):
            for index in range(len(token)):
                token[index] = 0
        for value in getattr(self, "_sensitive_values", ()):
            for index in range(len(value)):
                value[index] = 0
        self._closed = True

    def __enter__(self) -> McpOAuthAccessLease:
        if self._closed:
            raise McpOAuthError("MCP OAuth access lease is closed")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            "McpOAuthAccessLease(token=<redacted>, "
            f"fence={self.fence!r}, closed={self._closed!r})"
        )


class McpOAuthHttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        body: bytes | None,
        deadline: float,
        max_response_bytes: int,
    ) -> McpOAuthHttpResponse: ...


@dataclass
class _MemorySecret:
    namespace: str
    value: bytearray
    expires_at: float | None


class InMemoryMcpCredentialBroker:
    """Explicit test/ephemeral broker; never selected as a production default."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._secrets: dict[str, _MemorySecret] = {}

    def available(self) -> bool:
        return True

    def reserve_secret_ref(self, namespace: str) -> str:
        _validate_secret_namespace(namespace)
        return _reserved_secret_ref(
            "mem",
            "agent-libos.mcp.memory.v1",
            namespace,
        )

    def put_secret_at(
        self,
        secret_ref: str,
        namespace: str,
        value: bytes,
        *,
        expires_at: str | None,
    ) -> None:
        _validate_secret_input(namespace, value)
        if not _secret_ref_matches_namespace(
            secret_ref,
            prefix="mem",
            service="agent-libos.mcp.memory.v1",
            namespace=namespace,
        ):
            raise McpOAuthError("MCP credential slot does not match namespace")
        selected_expiry = _parse_optional_timestamp(expires_at)
        replacement = _MemorySecret(
            namespace=namespace,
            value=bytearray(value),
            expires_at=selected_expiry,
        )
        with self._lock:
            previous = self._secrets.get(secret_ref)
            if previous is not None:
                same = (
                    previous.namespace == namespace
                    and previous.expires_at == selected_expiry
                    and hmac.compare_digest(previous.value, replacement.value)
                )
                _zero_bytearray(replacement.value)
                if same:
                    return
                raise McpOAuthError("MCP credential slot already contains a value")
            self._secrets[secret_ref] = replacement

    def put_secret(self, namespace: str, value: bytes, *, expires_at: str | None) -> str:
        secret_ref = self.reserve_secret_ref(namespace)
        self.put_secret_at(
            secret_ref,
            namespace,
            value,
            expires_at=expires_at,
        )
        return secret_ref

    def get_secret(self, secret_ref: str) -> bytes:
        _validate_secret_ref(secret_ref, prefix="mem")
        with self._lock:
            selected = self._secrets.get(secret_ref)
            if selected is None:
                raise McpOAuthError("MCP credential is unavailable")
            if selected.expires_at is not None and selected.expires_at <= time.time():
                self._erase_locked(secret_ref, selected)
                raise McpOAuthError("MCP credential is unavailable")
            return bytes(selected.value)

    def delete_secret(self, secret_ref: str) -> None:
        _validate_secret_ref(secret_ref, prefix="mem")
        with self._lock:
            selected = self._secrets.get(secret_ref)
            if selected is not None:
                self._erase_locked(secret_ref, selected)

    def close(self) -> None:
        with self._lock:
            for secret_ref, selected in tuple(self._secrets.items()):
                self._erase_locked(secret_ref, selected)

    def _erase_locked(self, secret_ref: str, selected: _MemorySecret) -> None:
        _zero_bytearray(selected.value)
        self._secrets.pop(secret_ref, None)


class SystemKeyringMcpCredentialBroker:
    """Broker for the exact audited OS backends in the locked keyring build.

    A positive keyring priority, a familiar class name, or a ``keyring.*``
    module prefix is not an attestation.  The default broker requires an exact
    official backend class from the exact reviewed distribution source.  A
    Host that intentionally uses another secure facility must inject its own
    :class:`McpCredentialBroker` instead.
    """

    def __init__(
        self,
        *,
        service_name: str = _KEYRING_SERVICE,
        keyring_module: Any | None = None,
    ) -> None:
        if type(service_name) is not str or not service_name or len(service_name) > 200:
            raise McpOAuthError("invalid MCP credential service name")
        self._service_name = service_name
        self._keyring_module = keyring_module
        self._lock = threading.RLock()

    def _module(self) -> Any:
        if self._keyring_module is not None:
            return self._keyring_module
        try:
            import keyring
        except (ImportError, ModuleNotFoundError):
            keyring = None
        if keyring is None:
            # Raise after leaving the import handler so a loader/backend
            # exception can never survive as an inspectable exception context.
            raise McpOAuthError("secure credential backend unavailable")
        return keyring

    def available(self) -> bool:
        try:
            module = self._module()
            backend = module.get_keyring()
            return any(
                type(backend) is audited
                for audited in _audited_system_keyring_backend_types()
            )
        except Exception:
            return False

    def reserve_secret_ref(self, namespace: str) -> str:
        _validate_secret_namespace(namespace)
        return _reserved_secret_ref(
            "keyring",
            self._service_name,
            namespace,
        )

    def put_secret_at(
        self,
        secret_ref: str,
        namespace: str,
        value: bytes,
        *,
        expires_at: str | None,
    ) -> None:
        _validate_secret_input(namespace, value)
        if not _secret_ref_matches_namespace(
            secret_ref,
            prefix="keyring",
            service=self._service_name,
            namespace=namespace,
        ):
            raise McpOAuthError("MCP credential slot does not match namespace")
        _parse_optional_timestamp(expires_at)
        if not self.available():
            raise McpOAuthError("secure credential backend unavailable")
        envelope = json.dumps(
            {
                "expires_at": expires_at,
                "namespace_sha256": hashlib.sha256(
                    namespace.encode("utf-8")
                ).hexdigest(),
                "value": base64.b64encode(value).decode("ascii"),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        write_failed = False
        conflict = False
        try:
            with self._lock:
                module = self._module()
                previous = module.get_password(self._service_name, secret_ref)
                if previous is not None:
                    if type(previous) is str and hmac.compare_digest(
                        previous,
                        envelope,
                    ):
                        return
                    conflict = True
                else:
                    module.set_password(self._service_name, secret_ref, envelope)
        except Exception:
            write_failed = True
        if conflict:
            raise McpOAuthError("MCP credential slot already contains a value")
        if write_failed:
            raise McpOAuthError("secure credential backend unavailable")

    def put_secret(self, namespace: str, value: bytes, *, expires_at: str | None) -> str:
        secret_ref = self.reserve_secret_ref(namespace)
        self.put_secret_at(
            secret_ref,
            namespace,
            value,
            expires_at=expires_at,
        )
        return secret_ref

    def get_secret(self, secret_ref: str) -> bytes:
        _validate_secret_ref(secret_ref, prefix="keyring")
        if not self.available():
            raise McpOAuthError("secure credential backend unavailable")
        read_failed = False
        decoded: bytes | None = None
        try:
            with self._lock:
                encoded = self._module().get_password(self._service_name, secret_ref)
            if encoded is None or len(encoded) > (_MAX_SECRET_BYTES * 2):
                raise ValueError("missing keyring value")
            value = bounded_json_loads(encoded, max_bytes=_MAX_SECRET_BYTES * 2)
            if not isinstance(value, dict):
                raise ValueError("invalid keyring value")
            namespace_sha256 = value.get("namespace_sha256")
            if (
                type(namespace_sha256) is not str
                or _SHA256_RE.fullmatch(namespace_sha256) is None
            ):
                raise ValueError("invalid keyring value")
            expiry = _parse_optional_timestamp(value.get("expires_at"))
            if expiry is not None and expiry <= time.time():
                self.delete_secret(secret_ref)
                raise ValueError("expired keyring value")
            raw = value.get("value")
            if type(raw) is not str:
                raise ValueError("invalid keyring value")
            decoded = base64.b64decode(raw, validate=True)
            if not (1 <= len(decoded) <= _MAX_SECRET_BYTES):
                raise ValueError("oversized keyring value")
        except Exception:
            read_failed = True
        if read_failed or decoded is None:
            raise McpOAuthError("MCP credential is unavailable")
        return decoded

    def delete_secret(self, secret_ref: str) -> None:
        _validate_secret_ref(secret_ref, prefix="keyring")
        if not self.available():
            raise McpOAuthError("secure credential backend unavailable")
        delete_failed = False
        try:
            with self._lock:
                module = self._module()
                if module.get_password(self._service_name, secret_ref) is None:
                    return
                module.delete_password(self._service_name, secret_ref)
        except Exception:
            # Missing entries and backend errors are intentionally indistinct.
            delete_failed = True
        if delete_failed:
            raise McpOAuthError("secure credential backend unavailable")


@functools.lru_cache(maxsize=1)
def _audited_system_keyring_backend_types() -> tuple[type[Any], ...]:
    """Load exact reviewed backend class objects or fail the whole set closed."""

    try:
        distribution = importlib_metadata.distribution("keyring")
        if distribution.version != _AUDITED_KEYRING_VERSION:
            return ()
        installed_files = {
            str(candidate).replace("\\", "/")
            for candidate in (distribution.files or ())
        }
        audited: list[type[Any]] = []
        for module_name, class_name, relative_source, expected_sha256 in (
            _AUDITED_KEYRING_BACKENDS
        ):
            if relative_source not in installed_files:
                return ()
            expected_source = Path(
                distribution.locate_file(relative_source)
            ).resolve(strict=True)
            if (
                hashlib.sha256(expected_source.read_bytes()).hexdigest()
                != expected_sha256
            ):
                return ()
            module = importlib.import_module(module_name)
            if Path(module.__file__).resolve(strict=True) != expected_source:
                return ()
            backend_type = getattr(module, class_name, None)
            if (
                not isinstance(backend_type, type)
                or backend_type.__module__ != module_name
                or backend_type.__qualname__ != class_name
            ):
                return ()
            source = inspect.getsourcefile(backend_type)
            if source is None or Path(source).resolve(strict=True) != expected_source:
                return ()
            audited.append(backend_type)
        return tuple(audited)
    except Exception:
        return ()


Resolver = Callable[[str, int, float], Sequence[str]]


class _McpOAuthAddressUnavailable(Exception):
    """Internal signal for an address that failed before request dispatch."""


@dataclass(frozen=True)
class _PreparedOAuthRequest:
    endpoint: _Endpoint
    body: bytes
    request_bytes: bytes
    addresses: tuple[str, ...]


class PinnedMcpOAuthHttpTransport:
    """Small HTTPS client with exact-address connection and no redirects."""

    def __init__(
        self,
        *,
        resolver: Resolver | None = None,
        allow_loopback_http: bool = False,
        allow_loopback_tls: bool = False,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._resolver = resolver or _resolve_addresses
        self._allow_loopback_http = bool(allow_loopback_http)
        self._allow_loopback_tls = bool(allow_loopback_tls)
        self._ssl_context = ssl_context or ssl.create_default_context()

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
        prepared = _prepare_oauth_http_request(
            method=method,
            url=url,
            headers=headers,
            body=body,
            deadline=deadline,
            max_response_bytes=max_response_bytes,
            resolver=self._resolver,
            allow_loopback_http=self._allow_loopback_http,
            allow_loopback_tls=self._allow_loopback_tls,
        )
        for address in prepared.addresses:
            try:
                return _request_oauth_address(
                    prepared,
                    address=address,
                    deadline=deadline,
                    max_response_bytes=max_response_bytes,
                    ssl_context=self._ssl_context,
                )
            except _McpOAuthAddressUnavailable:
                continue
        raise McpOAuthTransportError("not_started") from None


@dataclass(frozen=True)
class _AuthorizationMetadata:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    revocation_endpoint: str | None
    issuer_parameter_required: bool
    scopes_supported: tuple[str, ...]


@dataclass
class _ProfileState:
    client_secret_ref: str | None = None
    token_ref: str | None = None
    token_expires_at: float | None = None
    scopes: tuple[str, ...] = ()
    metadata: _AuthorizationMetadata | None = None
    status: McpOAuthStatusKind = McpOAuthStatusKind.UNCONFIGURED
    principal_sha256: str | None = None
    credential_generation: int = 0
    refreshing: bool = False
    authorizing: bool = False


@dataclass(frozen=True)
class _ChallengeRecord:
    profile_id: str
    secret_ref: str
    expires_at: float


@dataclass(frozen=True)
class _AccessClaim:
    profile: McpOAuthProfile
    state: _ProfileState
    token_ref: str
    metadata: _AuthorizationMetadata | None
    credential_generation: int
    owns_refresh: bool


@dataclass(frozen=True)
class _RevocationClaim:
    profile: McpOAuthProfile
    state: _ProfileState
    token_ref: str | None
    metadata: _AuthorizationMetadata | None
    credential_generation: int
    local_refs: tuple[str, ...] = ()
    local_projection: McpOAuthStatus | None = None


class McpOAuthManager:
    """Runtime-owned Host OAuth manager.

    Public methods expose only non-secret status/challenge projections.  The
    sole secret-returning method, :meth:`access_token`, is intentionally named
    as a low-level transport hook and returns bytes so callers do not
    accidentally serialize it as a normal result.
    """

    def __init__(
        self,
        *,
        broker: McpCredentialBroker | None = None,
        transport: McpOAuthHttpTransport | None = None,
        connection_invalidator: Callable[[str], None] | None = None,
        challenge_ttl_s: float = _CHALLENGE_TTL_S,
        default_timeout_s: float = _DEFAULT_DEADLINE_S,
    ) -> None:
        self._broker = broker or SystemKeyringMcpCredentialBroker()
        for method_name in (
            "reserve_secret_ref",
            "put_secret_at",
            "put_secret",
            "get_secret",
            "delete_secret",
            "available",
        ):
            if not callable(getattr(self._broker, method_name, None)):
                raise McpOAuthError("invalid MCP credential broker")
        # The profile remains the authority gate.  The built-in transport is
        # technically capable of Host-opted loopback HTTP so a profile with
        # allow_loopback_http=True works; all ordinary profiles still require
        # HTTPS during local URL validation before this transport is reached.
        self._transport = transport or PinnedMcpOAuthHttpTransport(
            allow_loopback_http=True
        )
        if connection_invalidator is not None and not callable(
            connection_invalidator
        ):
            raise McpOAuthError("invalid MCP OAuth connection invalidator")
        self._connection_invalidator = connection_invalidator
        if (
            type(challenge_ttl_s) not in {int, float}
            or not math.isfinite(float(challenge_ttl_s))
            or not (30 <= float(challenge_ttl_s) <= 3600)
        ):
            raise McpOAuthError("invalid MCP OAuth challenge lifetime")
        if (
            type(default_timeout_s) not in {int, float}
            or not math.isfinite(float(default_timeout_s))
            or not (0 < float(default_timeout_s) <= 300)
        ):
            raise McpOAuthError("invalid MCP OAuth timeout")
        self._challenge_ttl_s = float(challenge_ttl_s)
        self._default_timeout_s = float(default_timeout_s)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._profiles: dict[str, McpOAuthProfile] = {}
        self._states: dict[str, _ProfileState] = {}
        self._credential_generations: dict[str, int] = {}
        self._challenges: dict[str, _ChallengeRecord] = {}
        self._registering_profiles: set[str] = set()
        self._closed = False

    def add_profile(
        self,
        profile: McpOAuthProfile,
        *,
        client_secret: bytes | None = None,
    ) -> McpOAuthStatus:
        _validate_profile(profile)
        if not self._broker_available():
            raise McpOAuthError("secure credential backend unavailable")
        secret_ref: str | None = None
        client_written = False
        requires_secret = profile.token_endpoint_auth_method in {
            McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_BASIC,
            McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_POST,
        }
        if requires_secret:
            if client_secret is not None:
                _validate_secret_input(
                    _oauth_client_namespace(profile.profile_id),
                    client_secret,
                )
        elif client_secret is not None:
            raise McpOAuthError("public MCP OAuth client must not provide a secret")
        with self._condition:
            self._require_open_locked()
            if (
                profile.profile_id in self._profiles
                or profile.profile_id in self._registering_profiles
            ):
                raise McpOAuthError("MCP OAuth profile already exists")
            self._registering_profiles.add(profile.profile_id)
        try:
            # Browser state is never resumed after close/crash.  Durable
            # client/token slots, however, are re-opened only after the Host
            # supplies the same strict profile authority again.
            self._delete_reserved_namespace(
                _oauth_challenge_namespace(profile.profile_id)
            )
            if requires_secret:
                if client_secret is not None:
                    self._delete_reserved_namespace(
                        _oauth_client_namespace(profile.profile_id)
                    )
                    secret_ref = self._put_secret(
                        _oauth_client_namespace(profile.profile_id),
                        _encoded_bound_client_secret(profile, bytes(client_secret)),
                        expires_at=None,
                    )
                    client_written = True
                else:
                    secret_ref = self._existing_secret_ref(
                        _oauth_client_namespace(profile.profile_id)
                    )
                    try:
                        _validated_bound_client_secret(
                            self._get_secret(secret_ref),
                            profile,
                        )
                    except Exception:
                        _delete_secret_quietly(self._broker, secret_ref)
                        self._delete_profile_slots_quietly(
                            profile.profile_id,
                            include_client=False,
                        )
                        raise McpOAuthError(
                            "MCP OAuth client credential requires Host attention"
                        ) from None
            else:
                self._delete_reserved_namespace(
                    _oauth_client_namespace(profile.profile_id)
                )
            restored_state = self._restore_profile_state(
                profile,
                client_secret_ref=secret_ref,
            )
            with self._condition:
                self._require_open_locked()
                if profile.profile_id in self._profiles:
                    raise McpOAuthError("MCP OAuth profile already exists")
                self._profiles[profile.profile_id] = profile
                self._states[profile.profile_id] = restored_state
        except Exception:
            self._delete_reserved_namespace_quietly(
                _oauth_challenge_namespace(profile.profile_id)
            )
            if client_written:
                self._delete_reserved_namespace_quietly(
                    _oauth_client_namespace(profile.profile_id)
                )
            raise
        finally:
            with self._condition:
                self._registering_profiles.discard(profile.profile_id)
        return self.status(profile.profile_id)

    def replace_profile(
        self,
        profile: McpOAuthProfile,
        *,
        client_secret: bytes | None = None,
    ) -> McpOAuthStatus:
        """Replace authority only after deleting every old credential fence."""

        _validate_profile(profile)
        with self._condition:
            old_state = self._states.get(profile.profile_id)
            old_client_secret_ref = (
                old_state.client_secret_ref if old_state is not None else None
            )
        self.logout(profile.profile_id, missing_ok=True)
        with self._condition:
            self._profiles.pop(profile.profile_id, None)
            self._states.pop(profile.profile_id, None)
        if old_client_secret_ref is not None:
            _delete_secret_quietly(self._broker, old_client_secret_ref)
        return self.add_profile(profile, client_secret=client_secret)

    def status(self, profile_id: str) -> McpOAuthStatus:
        with self._condition:
            profile, state = self._selected_locked(profile_id)
            status = state.status
            if (
                status is McpOAuthStatusKind.AUTHORIZED
                and state.token_expires_at is not None
                and state.token_expires_at <= time.time()
            ):
                status = McpOAuthStatusKind.EXPIRED
            expires_at = _format_timestamp(state.token_expires_at)
            return McpOAuthStatus(
                profile_id=profile.profile_id,
                status=status,
                issuer=profile.expected_issuer,
                resource=profile.resource_uri,
                scopes=state.scopes,
                principal_sha256=state.principal_sha256,
                expires_at=expires_at,
            )

    def list_profiles(self) -> tuple[McpOAuthStatus, ...]:
        """Return deterministic non-secret Host profile status projections."""

        with self._condition:
            profile_ids = tuple(sorted(self._profiles))
        return tuple(self.status(profile_id) for profile_id in profile_ids)

    def profile_snapshot(self, profile_id: str) -> McpOAuthProfile:
        """Return the immutable non-secret Host authority configuration."""

        with self._condition:
            profile, _state = self._selected_locked(profile_id)
            return profile

    def has_profile(self, profile_id: str) -> bool:
        """Return exact Host profile membership without exposing authority data."""

        if type(profile_id) is not str:
            return False
        with self._condition:
            return not self._closed and profile_id in self._profiles

    def credential_generation(self, profile_id: str) -> int:
        """Return the monotonic non-secret credential fence generation."""

        with self._condition:
            _profile, state = self._selected_locked(profile_id)
            return state.credential_generation

    def set_minimum_credential_generation(
        self,
        profile_id: str,
        generation: int,
    ) -> None:
        """Restore only a non-secret monotonic fence after Runtime restart."""

        if type(generation) is not int or generation < 0:
            raise McpOAuthError("invalid MCP OAuth credential generation")
        stale_ref: str | None = None
        server_id: str | None = None
        with self._condition:
            _profile, state = self._selected_locked(profile_id)
            if state.refreshing or state.authorizing:
                raise McpOAuthError("MCP OAuth credential generation is already active")
            if state.token_ref is not None and state.credential_generation < generation:
                # Durable non-secret evidence is newer than this keychain
                # bundle.  The bundle may have survived a crash between a
                # logout/revocation fence and local deletion; it must never be
                # resurrected or refreshed.
                stale_ref = state.token_ref
                server_id = _profile.server_id
                state.token_ref = None
                state.token_expires_at = None
                state.scopes = ()
                state.metadata = None
                state.principal_sha256 = None
                state.status = McpOAuthStatusKind.NEEDS_ATTENTION
            selected = max(state.credential_generation, generation)
            state.credential_generation = selected
            self._credential_generations[profile_id] = max(
                self._credential_generations.get(profile_id, 0),
                selected,
            )
        if stale_ref is not None:
            _delete_secret_quietly(self._broker, stale_ref)
        if server_id is not None:
            self._invalidate_connections(server_id)

    def remove_profile(self, profile_id: str, *, missing_ok: bool = False) -> None:
        """Delete one Host profile and every broker handle owned by it."""

        with self._condition:
            profile = self._profiles.get(profile_id)
            state = self._states.get(profile_id)
            if profile is None or state is None:
                if missing_ok:
                    return
                raise McpOAuthError("MCP OAuth profile is unavailable")
            refs = list(self._clear_authorization_locked(profile_id, state))
            if state.client_secret_ref is not None:
                refs.append(state.client_secret_ref)
            self._profiles.pop(profile_id, None)
            self._states.pop(profile_id, None)
        for secret_ref in refs:
            _delete_secret_quietly(self._broker, secret_ref)
        self._delete_profile_slots_quietly(profile_id, include_client=True)
        self._invalidate_connections(profile.server_id)

    def invalidate_server(self, server_id: str) -> None:
        """Fence credentials after registry replacement or unregistration."""

        if type(server_id) is not str or not server_id:
            raise McpOAuthError("invalid MCP OAuth server_id")
        with self._condition:
            selected = tuple(
                profile_id
                for profile_id, profile in self._profiles.items()
                if profile.server_id == server_id
            )
            refs: list[str] = []
            for profile_id in selected:
                state = self._states[profile_id]
                refs.extend(self._clear_authorization_locked(profile_id, state))
        for secret_ref in refs:
            _delete_secret_quietly(self._broker, secret_ref)
        for profile_id in selected:
            self._delete_profile_slots_quietly(profile_id, include_client=False)
        if selected:
            self._invalidate_connections(server_id)

    def begin(
        self,
        profile_id: str,
        *,
        scopes: tuple[str, ...] = (),
        resource_metadata_url: str | None = None,
        deadline: float | None = None,
    ) -> McpAuthorizationChallenge:
        selected_deadline = self._deadline(deadline)
        profile, state, profile_generation = self._claim_authorization_begin(
            profile_id
        )
        challenge_namespace = _oauth_challenge_namespace(profile_id)
        installed = False
        try:
            self._delete_reserved_namespace(challenge_namespace)
            challenge = self._begin_claimed_authorization(
                profile,
                state,
                profile_generation=profile_generation,
                scopes=scopes,
                resource_metadata_url=resource_metadata_url,
                deadline=selected_deadline,
            )
            installed = True
            return challenge
        finally:
            if not installed:
                self._delete_reserved_namespace_quietly(challenge_namespace)
            with self._condition:
                if self._states.get(profile_id) is state:
                    state.authorizing = False
                    self._condition.notify_all()

    def _claim_authorization_begin(
        self,
        profile_id: str,
    ) -> tuple[McpOAuthProfile, _ProfileState, int]:
        with self._condition:
            profile, state = self._selected_locked(profile_id)
            if state.refreshing or state.authorizing:
                raise McpOAuthError("MCP OAuth profile is busy")
            state.authorizing = True
            profile_generation = state.credential_generation
            for challenge_id, record in tuple(self._challenges.items()):
                if record.profile_id == profile_id:
                    self._challenges.pop(challenge_id, None)
        return profile, state, profile_generation

    def _begin_claimed_authorization(
        self,
        profile: McpOAuthProfile,
        state: _ProfileState,
        *,
        profile_generation: int,
        scopes: tuple[str, ...],
        resource_metadata_url: str | None,
        deadline: float,
    ) -> McpAuthorizationChallenge:
        requested_scopes = _selected_scopes(profile, scopes)
        metadata = self._discover(
            profile,
            resource_metadata_url=resource_metadata_url,
            deadline=deadline,
        )
        if not requested_scopes and metadata.scopes_supported:
            # The MCP authorization spec selects Protected Resource Metadata
            # scopes when a 401 supplied none.  Remote metadata still cannot
            # expand the Host's profile allowlist.
            requested_scopes = tuple(
                scope
                for scope in metadata.scopes_supported
                if scope in profile.allowed_scopes
            )

        state_secret = _random_b64url(32)
        verifier = _random_b64url(64)
        code_challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        expires_at = time.time() + self._challenge_ttl_s
        challenge_id = f"oauth_challenge_{_random_b64url(24)}"
        challenge_payload = {
            "client_id": profile.client_id,
            "expected_issuer": metadata.issuer,
            "issuer_parameter_required": metadata.issuer_parameter_required,
            "profile_id": profile.profile_id,
            "redirect_uri": profile.redirect_uri,
            "requested_scopes": list(requested_scopes),
            "resource": profile.resource_uri,
            "state": state_secret,
            "token_endpoint": metadata.token_endpoint,
            "verifier": verifier,
        }
        authorization_url = _authorization_url(
            profile,
            metadata,
            requested_scopes=requested_scopes,
            state=state_secret,
            code_challenge=code_challenge,
        )
        encoded_challenge = json.dumps(
            challenge_payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        secret_ref = self._put_secret(
            _oauth_challenge_namespace(profile.profile_id),
            encoded_challenge,
            expires_at=_format_timestamp(expires_at),
        )
        with self._condition:
            self._require_open_locked()
            current_profile = self._profiles.get(profile.profile_id)
            current_state = self._states.get(profile.profile_id)
            if (
                current_profile != profile
                or current_state is not state
                or state.credential_generation != profile_generation
                or state.refreshing
                or not state.authorizing
            ):
                _delete_secret_quietly(self._broker, secret_ref)
                raise McpOAuthError("MCP OAuth profile changed during authorization")
            self._prune_challenges_locked()
            self._challenges[challenge_id] = _ChallengeRecord(
                profile_id=profile.profile_id,
                secret_ref=secret_ref,
                expires_at=expires_at,
            )
            state.metadata = metadata
            state.status = McpOAuthStatusKind.AUTHORIZATION_REQUIRED
        return McpAuthorizationChallenge(
            challenge_id=challenge_id,
            authorization_url=authorization_url,
            expires_at=_format_timestamp(expires_at) or "",
        )

    def authorize_for_challenge(
        self,
        profile_id: str,
        www_authenticate: str,
        *,
        deadline: float | None = None,
    ) -> McpAuthorizationChallenge:
        """Start explicit initial/step-up authorization from one 401/403.

        The remote challenge can select required scopes and a protected
        resource metadata document, but both remain bounded by the Host
        profile.  This method never retries the original MCP operation.
        """

        hints = parse_mcp_oauth_www_authenticate(www_authenticate)
        with self._condition:
            profile, state = self._selected_locked(profile_id)
            previous_scopes = state.scopes
        metadata_url = hints.resource_metadata_url
        if metadata_url is not None:
            if _origin(metadata_url) != _origin(profile.resource_uri):
                raise McpOAuthError("MCP OAuth challenge rejected")
            if (
                profile.protected_resource_metadata_url is not None
                and metadata_url != profile.protected_resource_metadata_url
            ):
                raise McpOAuthError("MCP OAuth challenge rejected")
        challenged = hints.scopes
        selected_scopes = tuple(sorted(set(previous_scopes).union(challenged)))
        if not selected_scopes:
            selected_scopes = profile.default_scopes
        if not set(selected_scopes).issubset(profile.allowed_scopes):
            raise McpOAuthError("MCP OAuth scope request rejected")
        return self.begin(
            profile_id,
            scopes=selected_scopes,
            resource_metadata_url=metadata_url,
            deadline=deadline,
        )

    def challenge_profile_id(self, challenge_id: str) -> str:
        """Resolve a pending opaque challenge to its non-secret Host profile."""

        if type(challenge_id) is not str or not challenge_id.startswith(
            "oauth_challenge_"
        ):
            raise McpOAuthError("MCP authorization challenge is unavailable")
        with self._condition:
            self._require_open_locked()
            self._prune_challenges_locked()
            record = self._challenges.get(challenge_id)
            if record is None:
                raise McpOAuthError("MCP authorization challenge is unavailable")
            return record.profile_id

    def discard_challenge(self, challenge_id: str) -> None:
        """Delete one unreturned Host challenge without exchanging its code."""

        record = self._consume_challenge(challenge_id)
        try:
            self._broker.delete_secret(record.secret_ref)
        except Exception:
            # Preserve the in-memory owner when deletion is ambiguous so close,
            # expiry, or the next exact-profile begin can retry deterministic
            # slot cleanup.  The challenge remains unavailable to the caller
            # that did not receive its public projection.
            with self._condition:
                if (
                    not self._closed
                    and record.expires_at > time.time()
                    and challenge_id not in self._challenges
                ):
                    self._challenges[challenge_id] = record
            raise McpOAuthError("secure credential backend unavailable") from None

    def complete(
        self,
        challenge_id: str,
        callback_url: str,
        *,
        deadline: float | None = None,
    ) -> McpOAuthStatus:
        selected_deadline = self._deadline(deadline)
        record = self._consume_challenge(challenge_id)
        challenge_read_failed = False
        encoded: bytes | None = None
        try:
            encoded = self._get_secret(record.secret_ref)
        except McpOAuthError:
            challenge_read_failed = True
        finally:
            _delete_secret_quietly(self._broker, record.secret_ref)
        if challenge_read_failed or encoded is None:
            raise McpOAuthError("MCP authorization challenge is unavailable")
        claimed_state: _ProfileState | None = None
        exchange_attempted = False
        try:
            challenge = _decode_secret_object(encoded)
            with self._condition:
                profile, state = self._selected_locked(record.profile_id)
            callback = _validated_callback(profile, challenge, callback_url)
            if "error" in callback:
                raise McpOAuthError("MCP authorization was not granted")
            code = callback.get("code")
            if type(code) is not str or not (1 <= len(code) <= _MAX_TOKEN_CHARS):
                raise McpOAuthError("MCP authorization callback rejected")
            with self._condition:
                current_profile, current_state = self._selected_locked(
                    record.profile_id
                )
                if (
                    current_profile != profile
                    or current_state is not state
                    or state.refreshing
                ):
                    raise McpOAuthError("MCP OAuth profile is busy")
                metadata = state.metadata
                if (
                    metadata is None
                    or metadata.token_endpoint != challenge.get("token_endpoint")
                ):
                    raise McpOAuthError("MCP authorization callback rejected")
                claim_generation = state.credential_generation
                state.refreshing = True
                claimed_state = state
            exchange_attempted = True
            token_value = self._exchange_code(
                profile,
                state,
                metadata,
                code=code,
                verifier=_required_secret_string(challenge, "verifier"),
                requested_scopes=_secret_scopes(challenge.get("requested_scopes")),
                deadline=selected_deadline,
            )
            self._install_token(
                profile,
                state,
                metadata,
                token_value,
                expected_generation=claim_generation,
            )
            return self.status(profile.profile_id)
        except McpOAuthError:
            if exchange_attempted:
                with self._condition:
                    selected = self._states.get(record.profile_id)
                    if selected is claimed_state:
                        selected.status = McpOAuthStatusKind.NEEDS_ATTENTION
            raise
        except Exception:
            if exchange_attempted:
                with self._condition:
                    selected = self._states.get(record.profile_id)
                    if selected is claimed_state:
                        selected.status = McpOAuthStatusKind.NEEDS_ATTENTION
            raise McpOAuthError("MCP authorization callback rejected") from None
        finally:
            if claimed_state is not None:
                with self._condition:
                    if self._states.get(record.profile_id) is claimed_state:
                        claimed_state.refreshing = False
                        self._condition.notify_all()

    def access_token(
        self,
        profile_id: str,
        *,
        min_validity_s: float = 30.0,
        deadline: float | None = None,
    ) -> bytes:
        selected_validity = _validated_min_token_validity(min_validity_s)
        selected_deadline = self._deadline(deadline)
        claim = self._claim_access_token(
            profile_id,
            min_validity_s=selected_validity,
            deadline=selected_deadline,
        )
        token_ref = claim.token_ref
        if claim.owns_refresh:
            token_ref = self._refresh_access_claim(claim, deadline=selected_deadline)
        return self._access_bytes(claim.profile, claim.state, token_ref)

    def _claim_access_token(
        self,
        profile_id: str,
        *,
        min_validity_s: float,
        deadline: float,
    ) -> _AccessClaim:
        while True:
            with self._condition:
                profile, state = self._selected_locked(profile_id)
                if state.status is McpOAuthStatusKind.NEEDS_ATTENTION:
                    raise McpOAuthNeedsAttention(
                        "MCP OAuth credential requires Host attention"
                    )
                if state.token_ref is None:
                    raise McpOAuthAuthorizationRequired(
                        "MCP OAuth authorization is required"
                    )
                if state.refreshing:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._condition.wait(remaining):
                        raise McpOAuthError("MCP OAuth operation timed out")
                    continue
                owns_refresh = not (
                    state.token_expires_at is not None
                    and state.token_expires_at > time.time() + min_validity_s
                )
                if owns_refresh:
                    state.refreshing = True
                return _AccessClaim(
                    profile=profile,
                    state=state,
                    token_ref=state.token_ref,
                    metadata=state.metadata,
                    credential_generation=state.credential_generation,
                    owns_refresh=owns_refresh,
                )

    def _refresh_access_claim(
        self,
        claim: _AccessClaim,
        *,
        deadline: float,
    ) -> str:
        state = claim.state
        try:
            if claim.metadata is None:
                raise McpOAuthNeedsAttention(
                    "MCP OAuth credential requires Host attention"
                )
            current = _decode_token_secret(self._get_secret(claim.token_ref))
            _validate_stored_token_fence(
                current,
                claim.profile,
                expected_generation=claim.credential_generation,
            )
            if not _has_refresh_token(current):
                with self._condition:
                    state.status = McpOAuthStatusKind.EXPIRED
                raise McpOAuthAuthorizationRequired(
                    "MCP OAuth authorization is required"
                )
            refreshed = self._refresh(
                claim.profile,
                state,
                claim.metadata,
                current,
                deadline=deadline,
            )
            self._install_token(
                claim.profile,
                state,
                claim.metadata,
                refreshed,
                expected_generation=claim.credential_generation,
            )
        except McpOAuthAuthorizationRequired:
            raise
        except Exception as exc:
            with self._condition:
                state.status = McpOAuthStatusKind.NEEDS_ATTENTION
            if isinstance(exc, McpOAuthNeedsAttention):
                raise
            raise McpOAuthNeedsAttention(
                "MCP OAuth credential requires Host attention"
            ) from None
        finally:
            with self._condition:
                state.refreshing = False
                self._condition.notify_all()
        with self._condition:
            token_ref = state.token_ref
            if token_ref is None:
                raise McpOAuthNeedsAttention(
                    "MCP OAuth credential requires Host attention"
                )
            return token_ref

    def _access_bytes(
        self,
        profile: McpOAuthProfile,
        state: _ProfileState,
        token_ref: str,
    ) -> bytes:
        value = _decode_token_secret(self._get_secret(token_ref))
        scopes, _expires_at, _principal, generation, _metadata = (
            _validated_stored_token(value, profile)
        )
        if scopes != state.scopes or generation != state.credential_generation:
            raise McpOAuthNeedsAttention(
                "MCP OAuth credential requires Host attention"
            )
        access = value.get("access_token")
        if type(access) is not str or not access:
            raise McpOAuthNeedsAttention("MCP OAuth credential requires Host attention")
        return access.encode("utf-8")

    def transport_access(
        self,
        profile_id: str,
        *,
        min_validity_s: float = 30.0,
        deadline: float | None = None,
    ) -> McpOAuthAccessLease:
        """Return one exact token+identity snapshot for an in-memory dispatch."""

        selected_deadline = self._deadline(deadline)
        for _attempt in range(3):
            token = self.access_token(
                profile_id,
                min_validity_s=min_validity_s,
                deadline=selected_deadline,
            )
            with self._condition:
                profile, state = self._selected_locked(profile_id)
                token_ref = state.token_ref
                client_secret_ref = state.client_secret_ref
                generation = state.credential_generation
                if token_ref is None or state.status is not McpOAuthStatusKind.AUTHORIZED:
                    continue
                fence = McpOAuthCredentialFence(
                    profile_id=profile.profile_id,
                    server_id=profile.server_id,
                    issuer=profile.expected_issuer,
                    resource=profile.resource_uri,
                    scopes=state.scopes,
                    principal_sha256=state.principal_sha256,
                    credential_generation=generation,
                )
            current = _decode_token_secret(self._get_secret(token_ref))
            _validate_stored_token_fence(
                current,
                profile,
                expected_generation=generation,
            )
            current_access = current.get("access_token")
            if type(current_access) is not str or not hmac.compare_digest(
                current_access.encode("utf-8"), token
            ):
                continue
            redaction_values = self._lease_redaction_values(
                profile,
                current,
                client_secret_ref=client_secret_ref,
            )
            with self._condition:
                _profile, selected_state = self._selected_locked(profile_id)
                if (
                    selected_state.token_ref != token_ref
                    or selected_state.client_secret_ref != client_secret_ref
                    or selected_state.credential_generation != generation
                    or selected_state.status is not McpOAuthStatusKind.AUTHORIZED
                ):
                    continue
            return McpOAuthAccessLease(
                token,
                fence,
                sensitive_values=redaction_values,
            )
        raise McpOAuthNeedsAttention(
            "MCP OAuth credential changed during transport dispatch"
        )

    def _lease_redaction_values(
        self,
        profile: McpOAuthProfile,
        token_value: Mapping[str, Any],
        *,
        client_secret_ref: str | None,
    ) -> tuple[str, ...]:
        selected: list[str] = []
        refresh_token = token_value.get("refresh_token")
        if type(refresh_token) is str and refresh_token:
            selected.append(refresh_token)
        if client_secret_ref is None:
            return tuple(selected)
        try:
            client_secret = _validated_bound_client_secret(
                self._get_secret(client_secret_ref),
                profile,
            ).decode("utf-8")
        except (McpOAuthError, UnicodeDecodeError):
            raise McpOAuthNeedsAttention(
                "MCP OAuth credential requires Host attention"
            ) from None
        selected.append(client_secret)
        if (
            profile.token_endpoint_auth_method
            is McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_BASIC
        ):
            selected.append(_basic_client_auth_header(profile.client_id, client_secret))
        return tuple(selected)

    def credential_fence(self, profile_id: str) -> McpOAuthCredentialFence:
        """Read a non-secret preflight fence without refreshing or loading a token."""

        with self._condition:
            profile, state = self._selected_locked(profile_id)
            if (
                state.status is not McpOAuthStatusKind.AUTHORIZED
                or state.refreshing
                or state.token_ref is None
                or state.token_expires_at is None
                or state.token_expires_at <= time.time()
            ):
                raise McpOAuthAuthorizationRequired(
                    "MCP OAuth authorization is required"
                )
            return McpOAuthCredentialFence(
                profile_id=profile.profile_id,
                server_id=profile.server_id,
                issuer=profile.expected_issuer,
                resource=profile.resource_uri,
                scopes=state.scopes,
                principal_sha256=state.principal_sha256,
                credential_generation=state.credential_generation,
            )

    def validate_credential_fence(self, fence: McpOAuthCredentialFence) -> bool:
        """Check a lease fence immediately before provider dispatch."""

        if not isinstance(fence, McpOAuthCredentialFence):
            return False
        with self._condition:
            profile = self._profiles.get(fence.profile_id)
            state = self._states.get(fence.profile_id)
            return bool(
                not self._closed
                and profile is not None
                and state is not None
                and profile.server_id == fence.server_id
                and profile.expected_issuer == fence.issuer
                and profile.resource_uri == fence.resource
                and state.scopes == fence.scopes
                and state.principal_sha256 == fence.principal_sha256
                and state.credential_generation == fence.credential_generation
                and state.status is McpOAuthStatusKind.AUTHORIZED
                and not state.refreshing
                and state.token_ref is not None
                and state.token_expires_at is not None
                and state.token_expires_at > time.time()
            )

    def revoke(
        self,
        profile_id: str,
        *,
        deadline: float | None = None,
    ) -> McpOAuthStatus:
        """Revoke one grant once; an unknown response is never replayed."""

        selected_deadline = self._deadline(deadline)
        claim = self._claim_revocation(profile_id)
        if claim.token_ref is None:
            return self._finish_local_revocation(claim)
        return self._revoke_claim(claim, deadline=selected_deadline)

    def _claim_revocation(self, profile_id: str) -> _RevocationClaim:
        with self._condition:
            profile, state = self._selected_locked(profile_id)
            if state.refreshing:
                raise McpOAuthError("MCP OAuth profile is busy")
            if state.token_ref is None:
                refs = self._clear_authorization_locked(profile_id, state)
                projection = self._revoked_projection(profile)
                return _RevocationClaim(
                    profile=profile,
                    state=state,
                    token_ref=None,
                    metadata=None,
                    credential_generation=state.credential_generation,
                    local_refs=refs,
                    local_projection=projection,
                )
            metadata = state.metadata
            if metadata is None or metadata.revocation_endpoint is None:
                raise McpOAuthError("MCP OAuth revocation endpoint is unavailable")
            state.refreshing = True
            return _RevocationClaim(
                profile=profile,
                state=state,
                token_ref=state.token_ref,
                metadata=metadata,
                credential_generation=state.credential_generation,
            )

    def _finish_local_revocation(self, claim: _RevocationClaim) -> McpOAuthStatus:
        for secret_ref in claim.local_refs:
            _delete_secret_quietly(self._broker, secret_ref)
        self._delete_profile_slots_quietly(
            claim.profile.profile_id,
            include_client=False,
        )
        self._invalidate_connections(claim.profile.server_id)
        if claim.local_projection is None:
            raise McpOAuthNeedsAttention(
                "MCP OAuth revocation requires Host attention"
            )
        return claim.local_projection

    def _revoke_claim(
        self,
        claim: _RevocationClaim,
        *,
        deadline: float,
    ) -> McpOAuthStatus:
        finalized = False
        try:
            self._dispatch_revocation(claim, deadline=deadline)
            refs, projection = self._finalize_revocation(claim)
            finalized = True
            for secret_ref in refs:
                _delete_secret_quietly(self._broker, secret_ref)
            self._delete_profile_slots_quietly(
                claim.profile.profile_id,
                include_client=False,
            )
            self._invalidate_connections(claim.profile.server_id)
            return projection
        except Exception:
            with self._condition:
                if self._states.get(claim.profile.profile_id) is claim.state:
                    claim.state.status = McpOAuthStatusKind.NEEDS_ATTENTION
            raise McpOAuthNeedsAttention(
                "MCP OAuth revocation requires Host attention"
            ) from None
        finally:
            with self._condition:
                current = self._states.get(claim.profile.profile_id)
                if not finalized and current is claim.state:
                    claim.state.refreshing = False
                    self._condition.notify_all()

    def _dispatch_revocation(
        self,
        claim: _RevocationClaim,
        *,
        deadline: float,
    ) -> None:
        if claim.token_ref is None or claim.metadata is None:
            raise McpOAuthNeedsAttention(
                "MCP OAuth revocation requires Host attention"
            )
        token_value = _decode_token_secret(self._get_secret(claim.token_ref))
        _validate_stored_token_fence(
            token_value,
            claim.profile,
            expected_generation=claim.credential_generation,
        )
        token, token_type_hint = _selected_revocation_token(token_value)
        form: list[tuple[str, str]] = [
            ("token", token),
            ("token_type_hint", token_type_hint),
            ("client_id", claim.profile.client_id),
        ]
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        self._apply_client_auth(claim.profile, claim.state, form, headers)
        response = self._transport_request(
            "POST",
            claim.metadata.revocation_endpoint,
            headers=headers,
            body=urlencode(form).encode("ascii"),
            deadline=deadline,
            max_response_bytes=_MAX_METADATA_BYTES,
        )
        if not (200 <= response.status < 300):
            raise McpOAuthNeedsAttention(
                "MCP OAuth revocation requires Host attention"
            )

    def _finalize_revocation(
        self,
        claim: _RevocationClaim,
    ) -> tuple[tuple[str, ...], McpOAuthStatus]:
        with self._condition:
            current_profile = self._profiles.get(claim.profile.profile_id)
            current_state = self._states.get(claim.profile.profile_id)
            if (
                current_profile != claim.profile
                or current_state is not claim.state
                or claim.state.token_ref != claim.token_ref
                or claim.state.credential_generation != claim.credential_generation
                or not claim.state.refreshing
            ):
                raise McpOAuthNeedsAttention(
                    "MCP OAuth revocation requires Host attention"
                )
            refs = self._clear_authorization_locked(
                claim.profile.profile_id,
                claim.state,
            )
            return refs, self._revoked_projection(claim.profile)

    def logout(
        self,
        profile_id: str,
        *,
        missing_ok: bool = False,
    ) -> McpOAuthStatus:
        """Delete local credentials without claiming remote revocation."""

        with self._condition:
            if profile_id not in self._profiles:
                if missing_ok:
                    return McpOAuthStatus(
                        profile_id=profile_id,
                        status=McpOAuthStatusKind.UNCONFIGURED,
                    )
                raise McpOAuthError("MCP OAuth profile is unavailable")
            profile, state = self._selected_locked(profile_id)
            if state.refreshing:
                raise McpOAuthError("MCP OAuth profile is busy")
            refs = self._clear_authorization_locked(profile_id, state)
            projection = self._revoked_projection(profile)
        for secret_ref in refs:
            _delete_secret_quietly(self._broker, secret_ref)
        self._delete_profile_slots_quietly(profile_id, include_client=False)
        self._invalidate_connections(profile.server_id)
        return projection

    def _clear_authorization_locked(
        self,
        profile_id: str,
        state: _ProfileState,
    ) -> tuple[str, ...]:
        refs: list[str] = []
        if state.token_ref is not None:
            refs.append(state.token_ref)
        for challenge_id, record in tuple(self._challenges.items()):
            if record.profile_id != profile_id:
                continue
            self._challenges.pop(challenge_id, None)
            refs.append(record.secret_ref)
        state.token_ref = None
        state.token_expires_at = None
        state.scopes = ()
        state.metadata = None
        state.principal_sha256 = None
        state.credential_generation += 1
        self._credential_generations[profile_id] = state.credential_generation
        state.status = McpOAuthStatusKind.REVOKED
        state.refreshing = False
        self._condition.notify_all()
        return tuple(refs)

    @staticmethod
    def _revoked_projection(profile: McpOAuthProfile) -> McpOAuthStatus:
        return McpOAuthStatus(
            profile_id=profile.profile_id,
            status=McpOAuthStatusKind.REVOKED,
            issuer=profile.expected_issuer,
            resource=profile.resource_uri,
        )

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            profile_ids = tuple(self._profiles)
            refs: list[str] = []
            refs.extend(record.secret_ref for record in self._challenges.values())
            self._challenges.clear()
            for state in self._states.values():
                state.authorizing = False
                state.refreshing = False
            self._condition.notify_all()
        for secret_ref in refs:
            _delete_secret_quietly(self._broker, secret_ref)
        # PKCE/state is never resumable.  Client credentials and token bundles
        # intentionally remain in their deterministic secure slots so a later
        # Runtime can rebind them only after receiving the same strict Host
        # profile configuration.  logout/revoke/remove perform explicit purge.
        for profile_id in profile_ids:
            self._delete_reserved_namespace_quietly(
                _oauth_challenge_namespace(profile_id)
            )

    def _discover(
        self,
        profile: McpOAuthProfile,
        *,
        resource_metadata_url: str | None = None,
        deadline: float,
    ) -> _AuthorizationMetadata:
        advertised_scopes = self._discover_resource_scopes(
            profile,
            resource_metadata_url=resource_metadata_url,
            deadline=deadline,
        )
        server_metadata = self._fetch_authorization_metadata_document(
            profile,
            deadline=deadline,
        )
        return _authorization_metadata_from_document(
            profile,
            server_metadata,
            advertised_scopes=advertised_scopes,
        )

    def _discover_resource_scopes(
        self,
        profile: McpOAuthProfile,
        *,
        resource_metadata_url: str | None,
        deadline: float,
    ) -> tuple[str, ...]:
        selected_resource_metadata_url = (
            resource_metadata_url
            or profile.protected_resource_metadata_url
            or _protected_resource_metadata_url(profile.resource_uri)
        )
        if (
            profile.protected_resource_metadata_url is not None
            and resource_metadata_url is not None
            and resource_metadata_url != profile.protected_resource_metadata_url
        ):
            raise McpOAuthError("MCP authorization metadata rejected")
        if _origin(selected_resource_metadata_url) != _origin(profile.resource_uri):
            raise McpOAuthError("MCP authorization metadata rejected")
        resource_response = self._request_json(
            "GET",
            selected_resource_metadata_url,
            deadline=deadline,
            max_bytes=_MAX_METADATA_BYTES,
        )
        _require_digest(
            resource_response.body,
            profile.protected_resource_metadata_sha256,
        )
        metadata = _decode_remote_object(resource_response.body)
        advertised_resource = metadata.get("resource")
        if advertised_resource is not None and advertised_resource != profile.resource_uri:
            raise McpOAuthError("MCP authorization metadata rejected")
        servers = metadata.get("authorization_servers")
        if (
            not isinstance(servers, list)
            or not servers
            or any(type(value) is not str for value in servers)
            or profile.expected_issuer not in servers
        ):
            raise McpOAuthError("MCP authorization metadata rejected")
        return _optional_remote_scopes(metadata.get("scopes_supported"))

    def _fetch_authorization_metadata_document(
        self,
        profile: McpOAuthProfile,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        candidates = _authorization_metadata_candidates(profile)
        response = self._request_authorization_metadata_candidate(
            profile,
            candidates,
            deadline=deadline,
        )
        _require_digest(
            response.body,
            profile.authorization_server_metadata_sha256,
        )
        return _decode_remote_object(response.body)

    def _request_authorization_metadata_candidate(
        self,
        profile: McpOAuthProfile,
        candidates: tuple[str, ...],
        *,
        deadline: float,
    ) -> McpOAuthHttpResponse:
        for index, candidate in enumerate(candidates):
            if _origin(candidate) != _origin(profile.expected_issuer):
                raise McpOAuthError("MCP authorization metadata rejected")
            response = self._transport_request(
                "GET",
                candidate,
                headers=None,
                body=None,
                deadline=deadline,
                max_response_bytes=_MAX_METADATA_BYTES,
            )
            if response.status == 404 and index + 1 < len(candidates):
                continue
            if response.status != 200:
                raise McpOAuthError("MCP authorization metadata is unavailable")
            return response
        raise McpOAuthError("MCP authorization metadata is unavailable")

    def _exchange_code(
        self,
        profile: McpOAuthProfile,
        state: _ProfileState,
        metadata: _AuthorizationMetadata,
        *,
        code: str,
        verifier: str,
        requested_scopes: tuple[str, ...],
        deadline: float,
    ) -> dict[str, Any]:
        form: list[tuple[str, str]] = [
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", profile.redirect_uri),
            ("client_id", profile.client_id),
            ("code_verifier", verifier),
            ("resource", profile.resource_uri),
        ]
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        self._apply_client_auth(profile, state, form, headers)
        response = self._transport_request(
            "POST",
            metadata.token_endpoint,
            headers=headers,
            body=urlencode(form).encode("utf-8"),
            deadline=deadline,
            max_response_bytes=_MAX_TOKEN_BYTES,
        )
        return _validated_token_response(
            response,
            profile=profile,
            requested_scopes=requested_scopes,
            previous_refresh_token=None,
        )

    def _refresh(
        self,
        profile: McpOAuthProfile,
        state: _ProfileState,
        metadata: _AuthorizationMetadata,
        current: Mapping[str, Any],
        *,
        deadline: float,
    ) -> dict[str, Any]:
        refresh_token = current.get("refresh_token")
        if type(refresh_token) is not str or not refresh_token:
            raise McpOAuthAuthorizationRequired("MCP OAuth authorization is required")
        form: list[tuple[str, str]] = [
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", profile.client_id),
            ("resource", profile.resource_uri),
        ]
        if state.scopes:
            form.append(("scope", " ".join(state.scopes)))
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        self._apply_client_auth(profile, state, form, headers)
        # Exactly one call.  A timeout/connection loss after dispatch may have
        # consumed a rotating refresh token and is never retried here.
        response = self._transport_request(
            "POST",
            metadata.token_endpoint,
            headers=headers,
            body=urlencode(form).encode("utf-8"),
            deadline=deadline,
            max_response_bytes=_MAX_TOKEN_BYTES,
        )
        return _validated_token_response(
            response,
            profile=profile,
            requested_scopes=state.scopes,
            previous_refresh_token=refresh_token,
        )

    def _apply_client_auth(
        self,
        profile: McpOAuthProfile,
        state: _ProfileState,
        form: list[tuple[str, str]],
        headers: dict[str, str],
    ) -> None:
        method = profile.token_endpoint_auth_method
        if method is McpOAuthTokenEndpointAuthMethod.NONE:
            return
        if state.client_secret_ref is None:
            raise McpOAuthNeedsAttention("MCP OAuth credential requires Host attention")
        secret = _validated_bound_client_secret(
            self._get_secret(state.client_secret_ref),
            profile,
        )
        try:
            secret_text = secret.decode("utf-8")
        except UnicodeDecodeError:
            raise McpOAuthNeedsAttention(
                "MCP OAuth credential requires Host attention"
            ) from None
        if method is McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_POST:
            form.append(("client_secret", secret_text))
            return
        if method is McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_BASIC:
            headers["Authorization"] = _basic_client_auth_header(
                profile.client_id,
                secret_text,
            )
            return
        raise McpOAuthError("unsupported MCP OAuth client authentication method")

    def _install_token(
        self,
        profile: McpOAuthProfile,
        state: _ProfileState,
        metadata: _AuthorizationMetadata,
        token_value: Mapping[str, Any],
        *,
        expected_generation: int,
    ) -> None:
        expires_at = float(token_value["expires_at"])
        with self._condition:
            current_profile = self._profiles.get(profile.profile_id)
            current_state = self._states.get(profile.profile_id)
            if (
                self._closed
                or current_profile != profile
                or current_state is not state
                or not state.refreshing
                or state.credential_generation != expected_generation
            ):
                raise McpOAuthNeedsAttention(
                    "MCP OAuth profile changed while installing credentials"
                )
            next_generation = max(
                state.credential_generation,
                self._credential_generations.get(profile.profile_id, 0),
            ) + 1
        stored_value = dict(token_value)
        stored_value.update(
            {
                "authorization_metadata": _authorization_metadata_projection(
                    metadata
                ),
                "credential_generation": next_generation,
                "profile_authority_sha256": _profile_authority_sha256(profile),
            }
        )
        encoded = json.dumps(
            stored_value,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        token_namespace = _oauth_token_namespace(
            profile.profile_id,
            next_generation % _TOKEN_SLOT_COUNT,
        )
        # Alternating deterministic slots preserve the current generation if
        # the process crashes between old-generation pruning and this write.
        # Exact-slot brokers reject implicit replacement by contract.
        self._delete_reserved_namespace(token_namespace)
        new_ref = self._put_secret(
            token_namespace,
            encoded,
            # Access-token expiry is the refresh trigger, not the lifetime of
            # a separately issued refresh credential.  Keep the encrypted
            # bundle until explicit rotation/logout when a refresh token is
            # present; otherwise the broker can expire it with the access token.
            expires_at=(
                None
                if type(token_value.get("refresh_token")) is str
                else _format_timestamp(expires_at)
            ),
        )
        scopes = tuple(token_value["scopes"])
        principal = token_value.get("principal_sha256")
        with self._condition:
            current_profile = self._profiles.get(profile.profile_id)
            current_state = self._states.get(profile.profile_id)
            if (
                self._closed
                or current_profile != profile
                or current_state is not state
                or not state.refreshing
                or state.credential_generation != expected_generation
                or next_generation
                != max(
                    state.credential_generation,
                    self._credential_generations.get(profile.profile_id, 0),
                )
                + 1
            ):
                _delete_secret_quietly(self._broker, new_ref)
                raise McpOAuthNeedsAttention(
                    "MCP OAuth profile changed while installing credentials"
                )
            old_ref = state.token_ref
            state.token_ref = new_ref
            state.token_expires_at = expires_at
            state.scopes = scopes
            state.metadata = metadata
            state.principal_sha256 = principal if type(principal) is str else None
            state.credential_generation = next_generation
            self._credential_generations[profile.profile_id] = next_generation
            state.status = McpOAuthStatusKind.AUTHORIZED
        if old_ref is not None and old_ref != new_ref:
            _delete_secret_quietly(self._broker, old_ref)
        self._invalidate_connections(profile.server_id)

    def _invalidate_connections(self, server_id: str) -> None:
        callback = self._connection_invalidator
        if callback is None:
            return
        try:
            callback(server_id)
        except BaseException:
            # Credential CAS has already committed.  Connection invalidation
            # is best-effort and may never turn success into an apparent
            # rollback that tempts callers to replay a rotating mutation.
            pass

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        deadline: float,
        max_bytes: int,
    ) -> McpOAuthHttpResponse:
        response = self._transport_request(
            method,
            url,
            headers=None,
            body=None,
            deadline=deadline,
            max_response_bytes=max_bytes,
        )
        if response.status != 200:
            raise McpOAuthError("MCP authorization metadata is unavailable")
        return response

    def _transport_request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        body: bytes | None,
        deadline: float,
        max_response_bytes: int,
    ) -> McpOAuthHttpResponse:
        transport_failure: McpOAuthTransportError | None = None
        try:
            response = self._transport.request(
                method,
                url,
                headers=headers,
                body=body,
                deadline=deadline,
                max_response_bytes=max_response_bytes,
            )
        except McpOAuthTransportError as exc:
            # Preserve only the bounded dispatch classification.  A custom
            # transport may attach response bodies or credentials to its own
            # exception chain, so never propagate that exception object.
            transport_failure = McpOAuthTransportError(exc.dispatch_state)
        except Exception:
            # A custom Host transport may include headers/bodies in its own
            # exception.  Never chain or project that exception.
            transport_failure = McpOAuthTransportError("unknown")
        if transport_failure is not None:
            raise transport_failure
        if not isinstance(response, McpOAuthHttpResponse):
            raise McpOAuthTransportError("unknown")
        if time.monotonic() >= deadline:
            # A synchronous Host SPI is trusted to cooperate with the passed
            # deadline and cannot be forcibly interrupted without leaving an
            # unowned credential mutation.  A late return is nevertheless
            # never accepted as success; the request was already dispatched.
            raise McpOAuthTransportError("started")
        return response

    def _put_secret(
        self,
        namespace: str,
        value: bytes,
        *,
        expires_at: str | None,
    ) -> str:
        write_failed = False
        secret_ref: Any = None
        try:
            secret_ref = self._broker.reserve_secret_ref(namespace)
            self._broker.put_secret_at(
                secret_ref,
                namespace,
                value,
                expires_at=expires_at,
            )
        except Exception:
            write_failed = True
        if write_failed or type(secret_ref) is not str or not secret_ref:
            raise McpOAuthError("secure credential backend unavailable")
        return secret_ref

    def _existing_secret_ref(self, namespace: str) -> str:
        try:
            secret_ref = self._broker.reserve_secret_ref(namespace)
        except Exception:
            raise McpOAuthError("secure credential backend unavailable") from None
        if type(secret_ref) is not str or not secret_ref:
            raise McpOAuthError("secure credential backend unavailable")
        return secret_ref

    def _restore_profile_state(
        self,
        profile: McpOAuthProfile,
        *,
        client_secret_ref: str | None,
    ) -> _ProfileState:
        generation = self._credential_generations.get(profile.profile_id, 0)
        base = _ProfileState(
            client_secret_ref=client_secret_ref,
            status=McpOAuthStatusKind.AUTHORIZATION_REQUIRED,
            credential_generation=generation,
        )
        candidates: list[
            tuple[
                int,
                str,
                tuple[str, ...],
                float,
                str | None,
                _AuthorizationMetadata,
            ]
        ] = []
        namespaces = (
            _oauth_token_bundle_namespace(profile.profile_id),
            *(
                _oauth_token_namespace(profile.profile_id, slot)
                for slot in range(_TOKEN_SLOT_COUNT)
            ),
        )
        for namespace in namespaces:
            token_ref = self._existing_secret_ref(namespace)
            try:
                token = _decode_token_secret(self._get_secret(token_ref))
                (
                    scopes,
                    expires_at,
                    principal,
                    token_generation,
                    metadata,
                ) = _validated_stored_token(token, profile)
                if token_generation < generation:
                    raise McpOAuthNeedsAttention(
                        "MCP OAuth credential requires Host attention"
                    )
                candidates.append(
                    (
                        token_generation,
                        token_ref,
                        scopes,
                        expires_at,
                        principal,
                        metadata,
                    )
                )
            except Exception:
                _delete_secret_quietly(self._broker, token_ref)
        if not candidates:
            return base
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            for _generation, ambiguous_ref, *_rest in candidates:
                _delete_secret_quietly(self._broker, ambiguous_ref)
            base.status = McpOAuthStatusKind.NEEDS_ATTENTION
            base.credential_generation = max(generation, candidates[0][0])
            self._credential_generations[profile.profile_id] = (
                base.credential_generation
            )
            return base
        (
            token_generation,
            token_ref,
            scopes,
            expires_at,
            principal,
            metadata,
        ) = candidates[0]
        for _generation, stale_ref, *_rest in candidates[1:]:
            _delete_secret_quietly(self._broker, stale_ref)
        base.token_ref = token_ref
        base.token_expires_at = expires_at
        base.scopes = scopes
        base.metadata = metadata
        base.principal_sha256 = principal
        base.credential_generation = max(generation, token_generation)
        self._credential_generations[profile.profile_id] = base.credential_generation
        base.status = (
            McpOAuthStatusKind.AUTHORIZED
            if expires_at > time.time()
            else McpOAuthStatusKind.EXPIRED
        )
        return base

    def _delete_profile_slots(
        self,
        profile_id: str,
        *,
        include_client: bool,
    ) -> None:
        failed = False
        for namespace in _oauth_profile_namespaces(
            profile_id,
            include_client=include_client,
        ):
            try:
                secret_ref = self._broker.reserve_secret_ref(namespace)
                self._broker.delete_secret(secret_ref)
            except Exception:
                failed = True
        if failed:
            raise McpOAuthError("secure credential backend unavailable")

    def _delete_reserved_namespace(self, namespace: str) -> None:
        try:
            secret_ref = self._broker.reserve_secret_ref(namespace)
            self._broker.delete_secret(secret_ref)
        except Exception:
            raise McpOAuthError("secure credential backend unavailable") from None

    def _delete_reserved_namespace_quietly(self, namespace: str) -> None:
        try:
            self._delete_reserved_namespace(namespace)
        except Exception:
            pass

    def _delete_profile_slots_quietly(
        self,
        profile_id: str,
        *,
        include_client: bool,
    ) -> None:
        try:
            self._delete_profile_slots(
                profile_id,
                include_client=include_client,
            )
        except Exception:
            pass

    def _broker_available(self) -> bool:
        try:
            return self._broker.available() is True
        except Exception:
            return False

    def _get_secret(self, secret_ref: str) -> bytes:
        read_failed = False
        value: Any = None
        try:
            value = self._broker.get_secret(secret_ref)
        except Exception:
            read_failed = True
        if (
            read_failed
            or not isinstance(value, bytes)
            or not (1 <= len(value) <= _MAX_SECRET_BYTES)
        ):
            raise McpOAuthError("MCP credential is unavailable")
        return value

    def _consume_challenge(self, challenge_id: str) -> _ChallengeRecord:
        if type(challenge_id) is not str or not challenge_id.startswith("oauth_challenge_"):
            raise McpOAuthError("MCP authorization challenge is unavailable")
        with self._condition:
            self._require_open_locked()
            record = self._challenges.pop(challenge_id, None)
        if record is None or record.expires_at <= time.time():
            if record is not None:
                _delete_secret_quietly(self._broker, record.secret_ref)
            raise McpOAuthError("MCP authorization challenge is unavailable")
        return record

    def _prune_challenges_locked(self) -> None:
        now = time.time()
        expired = tuple(
            (challenge_id, record.secret_ref)
            for challenge_id, record in self._challenges.items()
            if record.expires_at <= now
        )
        for challenge_id, secret_ref in expired:
            self._challenges.pop(challenge_id, None)
            _delete_secret_quietly(self._broker, secret_ref)

    def _selected_locked(self, profile_id: str) -> tuple[McpOAuthProfile, _ProfileState]:
        self._require_open_locked()
        profile = self._profiles.get(profile_id)
        state = self._states.get(profile_id)
        if profile is None or state is None:
            raise McpOAuthError("MCP OAuth profile is unavailable")
        return profile, state

    def _require_open_locked(self) -> None:
        if self._closed:
            raise McpOAuthError("MCP OAuth manager is closed")

    def _deadline(self, deadline: float | None) -> float:
        if deadline is None:
            return time.monotonic() + self._default_timeout_s
        if type(deadline) not in {int, float} or not math.isfinite(float(deadline)):
            raise McpOAuthError("invalid MCP OAuth deadline")
        selected = float(deadline)
        if selected <= time.monotonic():
            raise McpOAuthError("MCP OAuth operation timed out")
        return selected


@dataclass(frozen=True)
class _Endpoint:
    scheme: str
    hostname: str
    port: int
    path: str
    query: str


def _profile_mapping_text(value: Mapping[str, Any], key: str) -> str:
    selected = value.get(key)
    if type(selected) is not str or not selected:
        raise McpOAuthError("MCP OAuth profile field is invalid")
    return selected


def _profile_mapping_optional_text(
    value: Mapping[str, Any],
    key: str,
) -> str | None:
    selected = value.get(key)
    if selected is None:
        return None
    if type(selected) is not str or not selected:
        raise McpOAuthError("MCP OAuth profile field is invalid")
    return selected


def _profile_mapping_strings(
    value: Mapping[str, Any],
    key: str,
) -> tuple[str, ...]:
    selected = value.get(key, [])
    if type(selected) is not list or any(type(item) is not str for item in selected):
        raise McpOAuthError("MCP OAuth profile field is invalid")
    return tuple(selected)


def _profile_mapping_bool(
    value: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    selected = value.get(key, default)
    if type(selected) is not bool:
        raise McpOAuthError("MCP OAuth profile field is invalid")
    return selected


def _profile_mapping_literal(
    value: Mapping[str, Any],
    key: str,
    expected: str,
) -> Any:
    selected = value.get(key, expected)
    if type(selected) is not str or selected != expected:
        raise McpOAuthError("MCP OAuth profile field is invalid")
    return selected


def _profile_mapping_enum(
    value: Mapping[str, Any],
    key: str,
    enum_type: type[StrEnum],
    *,
    default: StrEnum | None = None,
) -> Any:
    selected = value.get(key, default)
    if isinstance(selected, enum_type):
        return selected
    if type(selected) is not str:
        raise McpOAuthError("MCP OAuth profile field is invalid")
    try:
        return enum_type(selected)
    except ValueError:
        raise McpOAuthError("MCP OAuth profile field is invalid") from None


def _validate_profile(profile: McpOAuthProfile) -> None:
    _validate_profile_identity_and_mode(profile)
    resource, issuer = _validate_profile_authorities(profile)
    _validate_profile_client(profile)
    _validate_redirect_uri(profile.redirect_uri)
    _validate_profile_scopes_and_audience(profile)
    _validate_profile_metadata_pins(profile, resource=resource, issuer=issuer)
    _profile_endpoint_origins(profile)


def _validate_profile_identity_and_mode(profile: McpOAuthProfile) -> None:
    if not isinstance(profile, McpOAuthProfile):
        raise McpOAuthError("invalid MCP OAuth profile")
    for label, value in (
        ("profile_id", profile.profile_id),
        ("server_id", profile.server_id),
    ):
        if type(value) is not str or not _PROFILE_ID_RE.fullmatch(value):
            raise McpOAuthError(f"invalid MCP OAuth {label}")
    if (
        profile.protocol_revision != "2026-07-28"
        or profile.transport != "streamable_http"
        or type(profile.allow_loopback_http) is not bool
    ):
        raise McpOAuthError(
            "MCP OAuth requires Manifest v3 Streamable HTTP protocol 2026-07-28"
        )
    if profile.registration_mode == "dcr":
        raise McpOAuthError("MCP OAuth dynamic client registration is unsupported")
    if type(profile.registration_mode) is not McpOAuthRegistrationMode or (
        profile.registration_mode not in {
            McpOAuthRegistrationMode.PREREGISTERED,
            McpOAuthRegistrationMode.CIMD,
        }
    ):
        raise McpOAuthError("MCP OAuth dynamic client registration is unsupported")
    if (
        type(profile.token_endpoint_auth_method)
        is not McpOAuthTokenEndpointAuthMethod
        or profile.token_endpoint_auth_method
        not in {
            McpOAuthTokenEndpointAuthMethod.NONE,
            McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_BASIC,
            McpOAuthTokenEndpointAuthMethod.CLIENT_SECRET_POST,
        }
    ):
        raise McpOAuthError("unsupported MCP OAuth client authentication method")


def _validate_profile_authorities(
    profile: McpOAuthProfile,
) -> tuple[_Endpoint, _Endpoint]:
    resource = _validated_endpoint(
        profile.resource_uri,
        allow_loopback_http=profile.allow_loopback_http,
        endpoint_label="resource URI",
    )
    if urlsplit(profile.resource_uri).fragment:
        raise McpOAuthError("invalid MCP OAuth resource URI")
    issuer = _validated_endpoint(
        profile.expected_issuer,
        allow_loopback_http=profile.allow_loopback_http,
        endpoint_label="issuer",
    )
    if issuer.query or urlsplit(profile.expected_issuer).fragment:
        raise McpOAuthError("invalid MCP OAuth issuer")
    return resource, issuer


def _validate_profile_client(profile: McpOAuthProfile) -> None:
    if type(profile.client_id) is not str or not (1 <= len(profile.client_id) <= 2048):
        raise McpOAuthError("invalid MCP OAuth client_id")
    if any(ord(character) < 0x20 for character in profile.client_id):
        raise McpOAuthError("invalid MCP OAuth client_id")
    if profile.registration_mode is McpOAuthRegistrationMode.CIMD:
        cimd = _validated_endpoint(
            profile.client_id,
            allow_loopback_http=False,
            endpoint_label="CIMD client_id",
        )
        split_client = urlsplit(profile.client_id)
        if (
            cimd.scheme != "https"
            or cimd.path in {"", "/"}
            or cimd.query
            or split_client.fragment
            or profile.token_endpoint_auth_method
            is not McpOAuthTokenEndpointAuthMethod.NONE
        ):
            raise McpOAuthError("invalid MCP OAuth CIMD client_id")


def _validate_profile_scopes_and_audience(profile: McpOAuthProfile) -> None:
    if (
        type(profile.allowed_scopes) is not tuple
        or type(profile.default_scopes) is not tuple
    ):
        raise McpOAuthError("invalid MCP OAuth scope configuration")
    allowed_scopes = _validate_scopes(profile.allowed_scopes, label="allowed scopes")
    default_scopes = _validate_scopes(profile.default_scopes, label="default scopes")
    if not set(default_scopes).issubset(allowed_scopes):
        raise McpOAuthError("MCP OAuth default scopes exceed the Host allowlist")
    audience = profile.audience or profile.resource_uri
    if audience != profile.resource_uri:
        raise McpOAuthError("MCP OAuth audience must equal the pinned resource")


def _validate_profile_metadata_pins(
    profile: McpOAuthProfile,
    *,
    resource: _Endpoint,
    issuer: _Endpoint,
) -> None:
    if profile.protected_resource_metadata_url is not None:
        endpoint = _validated_endpoint(
            profile.protected_resource_metadata_url,
            allow_loopback_http=profile.allow_loopback_http,
            endpoint_label="protected resource metadata URL",
        )
        if (endpoint.scheme, endpoint.hostname, endpoint.port) != (
            resource.scheme,
            resource.hostname,
            resource.port,
        ):
            raise McpOAuthError("MCP authorization metadata rejected")
    if profile.authorization_server_metadata_url is not None:
        endpoint = _validated_endpoint(
            profile.authorization_server_metadata_url,
            allow_loopback_http=profile.allow_loopback_http,
            endpoint_label="authorization server metadata URL",
        )
        if (endpoint.scheme, endpoint.hostname, endpoint.port) != (
            issuer.scheme,
            issuer.hostname,
            issuer.port,
        ):
            raise McpOAuthError("MCP authorization metadata rejected")
    for digest in (
        profile.protected_resource_metadata_sha256,
        profile.authorization_server_metadata_sha256,
    ):
        if digest is not None and (
            type(digest) is not str or not _SHA256_RE.fullmatch(digest)
        ):
            raise McpOAuthError("invalid MCP OAuth metadata digest pin")


def _validate_redirect_uri(value: str) -> None:
    if type(value) is not str or not (1 <= len(value) <= _MAX_URL_CHARS):
        raise McpOAuthError("invalid MCP OAuth redirect URI")
    endpoint = _validated_endpoint(
        value,
        allow_loopback_http=True,
        endpoint_label="OAuth redirect URI",
    )
    parsed = urlsplit(value)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
        or not parsed.path
    ):
        raise McpOAuthError("invalid MCP OAuth redirect URI")
    if endpoint.scheme == "https":
        return
    if endpoint.scheme == "http" and endpoint.hostname in _LOOPBACK_HOSTS:
        return
    raise McpOAuthError("invalid MCP OAuth redirect URI")


def _profile_endpoint_origins(profile: McpOAuthProfile) -> frozenset[str]:
    if type(profile.allowed_endpoint_origins) is not tuple:
        raise McpOAuthError("invalid MCP OAuth endpoint origin allowlist")
    issuer_origin = _origin(profile.expected_issuer)
    origins = {issuer_origin}
    for value in profile.allowed_endpoint_origins:
        if type(value) is not str:
            raise McpOAuthError("invalid MCP OAuth endpoint origin allowlist")
        parsed = urlsplit(value)
        _validated_endpoint(
            value,
            allow_loopback_http=profile.allow_loopback_http,
            endpoint_label="endpoint origin",
        )
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise McpOAuthError("invalid MCP OAuth endpoint origin allowlist")
        origins.add(_origin(value))
    return frozenset(origins)


def _selected_scopes(
    profile: McpOAuthProfile,
    scopes: tuple[str, ...],
) -> tuple[str, ...]:
    selected = profile.default_scopes if not scopes else scopes
    selected = _validate_scopes(selected, label="requested scopes")
    if not set(selected).issubset(profile.allowed_scopes):
        raise McpOAuthError("MCP OAuth scope request rejected")
    return selected


def _validate_scopes(values: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or len(values) > _MAX_SCOPES:
        raise McpOAuthError(f"invalid MCP OAuth {label}")
    seen: set[str] = set()
    selected: list[str] = []
    for value in values:
        if (
            type(value) is not str
            or not (1 <= len(value) <= _MAX_SCOPE_CHARS)
            or any(
                not (
                    ord(character) == 0x21
                    or 0x23 <= ord(character) <= 0x5B
                    or 0x5D <= ord(character) <= 0x7E
                )
                for character in value
            )
            or value in seen
        ):
            raise McpOAuthError(f"invalid MCP OAuth {label}")
        seen.add(value)
        selected.append(value)
    return tuple(sorted(selected))


def _validated_endpoint(
    value: str,
    *,
    allow_loopback_http: bool,
    endpoint_label: str,
) -> _Endpoint:
    if (
        type(value) is not str
        or not (1 <= len(value) <= _MAX_URL_CHARS)
        or any(ord(character) < 0x20 for character in value)
    ):
        raise McpOAuthError(f"invalid MCP {endpoint_label}")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"https", "http"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not parsed.hostname
    ):
        raise McpOAuthError(f"invalid MCP {endpoint_label}")
    try:
        host = parsed.hostname.encode("idna").decode("ascii").casefold().strip("[]")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except (UnicodeError, ValueError) as exc:
        raise McpOAuthError(f"invalid MCP {endpoint_label}") from exc
    if host in _FORBIDDEN_HOSTS:
        raise McpOAuthError(f"invalid MCP {endpoint_label}")
    if parsed.scheme == "http":
        if not allow_loopback_http or host not in _LOOPBACK_HOSTS:
            raise McpOAuthError(f"invalid MCP {endpoint_label}")
    return _Endpoint(
        scheme=parsed.scheme,
        hostname=host,
        port=port,
        path=parsed.path or "/",
        query=parsed.query,
    )


def _origin(value: str) -> str:
    endpoint = _validated_endpoint(
        value,
        allow_loopback_http=True,
        endpoint_label="OAuth URL",
    )
    default_port = 443 if endpoint.scheme == "https" else 80
    suffix = "" if endpoint.port == default_port else f":{endpoint.port}"
    host = (
        f"[{endpoint.hostname}]" if ":" in endpoint.hostname else endpoint.hostname
    )
    return f"{endpoint.scheme}://{host}{suffix}"


def _protected_resource_metadata_url(resource_uri: str) -> str:
    parsed = urlsplit(resource_uri)
    path = parsed.path.rstrip("/")
    metadata_path = "/.well-known/oauth-protected-resource"
    if path:
        metadata_path += path
    return urlunsplit((parsed.scheme, parsed.netloc, metadata_path, "", ""))


def _authorization_metadata_urls(issuer: str) -> tuple[str, str]:
    parsed = urlsplit(issuer)
    issuer_path = parsed.path.rstrip("/")
    oauth_path = "/.well-known/oauth-authorization-server"
    if issuer_path:
        oauth_path += issuer_path
    oauth = urlunsplit((parsed.scheme, parsed.netloc, oauth_path, "", ""))
    oidc_path = (issuer_path or "") + "/.well-known/openid-configuration"
    oidc = urlunsplit((parsed.scheme, parsed.netloc, oidc_path, "", ""))
    return oauth, oidc


def _authorization_metadata_candidates(profile: McpOAuthProfile) -> tuple[str, ...]:
    if profile.authorization_server_metadata_url is not None:
        return (profile.authorization_server_metadata_url,)
    return _authorization_metadata_urls(profile.expected_issuer)


def _authorization_metadata_from_document(
    profile: McpOAuthProfile,
    metadata: Mapping[str, Any],
    *,
    advertised_scopes: tuple[str, ...],
) -> _AuthorizationMetadata:
    if metadata.get("issuer") != profile.expected_issuer:
        raise McpOAuthError("MCP authorization metadata rejected")
    authorization_endpoint, token_endpoint, revocation_endpoint = (
        _validated_authorization_endpoints(profile, metadata)
    )
    _validate_authorization_server_features(profile, metadata)
    issuer_required = metadata.get(
        "authorization_response_iss_parameter_supported",
        False,
    )
    if type(issuer_required) is not bool:
        raise McpOAuthError("MCP authorization metadata rejected")
    server_scopes = _optional_remote_scopes(metadata.get("scopes_supported"))
    return _AuthorizationMetadata(
        issuer=profile.expected_issuer,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        revocation_endpoint=revocation_endpoint,
        issuer_parameter_required=issuer_required,
        scopes_supported=server_scopes or advertised_scopes,
    )


def _validated_authorization_endpoints(
    profile: McpOAuthProfile,
    metadata: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    endpoints = (
        _required_remote_string(metadata, "authorization_endpoint"),
        _required_remote_string(metadata, "token_endpoint"),
        _optional_remote_string(metadata, "revocation_endpoint"),
    )
    allowed_origins = _profile_endpoint_origins(profile)
    for endpoint in endpoints:
        if endpoint is None:
            continue
        _validated_endpoint(
            endpoint,
            allow_loopback_http=profile.allow_loopback_http,
            endpoint_label="authorization metadata endpoint",
        )
        if _origin(endpoint) not in allowed_origins:
            raise McpOAuthError("MCP authorization metadata rejected")
    return endpoints


def _validate_authorization_server_features(
    profile: McpOAuthProfile,
    metadata: Mapping[str, Any],
) -> None:
    requirements = (
        (
            "token_endpoint_auth_methods_supported",
            profile.token_endpoint_auth_method.value,
        ),
        ("code_challenge_methods_supported", "S256"),
        ("response_types_supported", "code"),
        ("grant_types_supported", "authorization_code"),
    )
    for field_name, required in requirements:
        advertised = _optional_remote_strings(metadata.get(field_name))
        if advertised and required not in advertised:
            raise McpOAuthError("MCP authorization metadata rejected")


def _authorization_url(
    profile: McpOAuthProfile,
    metadata: _AuthorizationMetadata,
    *,
    requested_scopes: tuple[str, ...],
    state: str,
    code_challenge: str,
) -> str:
    parsed = urlsplit(metadata.authorization_endpoint)
    existing = parse_qsl(parsed.query, keep_blank_values=True)
    reserved = {
        "response_type",
        "client_id",
        "redirect_uri",
        "scope",
        "state",
        "code_challenge",
        "code_challenge_method",
        "resource",
    }
    if any(key in reserved for key, _value in existing):
        raise McpOAuthError("MCP authorization metadata rejected")
    values = list(existing)
    values.extend(
        [
            ("response_type", "code"),
            ("client_id", profile.client_id),
            ("redirect_uri", profile.redirect_uri),
            ("state", state),
            ("code_challenge", code_challenge),
            ("code_challenge_method", "S256"),
            ("resource", profile.resource_uri),
        ]
    )
    if requested_scopes:
        values.append(("scope", " ".join(requested_scopes)))
    selected = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(values), "")
    )
    if len(selected) > _MAX_URL_CHARS:
        raise McpOAuthError("MCP authorization request is too large")
    return selected


def parse_mcp_oauth_www_authenticate(value: str) -> McpOAuthChallengeHints:
    """Parse one bounded RFC 6750 Bearer challenge without trusting authority.

    Multiple auth schemes are rejected instead of heuristically choosing one;
    the Runtime may call this function on the already-selected Bearer header
    value when an HTTP stack provides challenges separately.
    """

    if (
        type(value) is not str
        or not (1 <= len(value) <= 16 * 1024)
        or "\r" in value
        or "\n" in value
    ):
        raise McpOAuthError("MCP OAuth challenge rejected")
    scheme, separator, remainder = value.strip().partition(" ")
    if separator != " " or scheme.casefold() != "bearer":
        raise McpOAuthError("MCP OAuth challenge rejected")
    parameters = _parse_auth_parameters(remainder)
    metadata_url = parameters.get("resource_metadata")
    if metadata_url is not None:
        _validated_endpoint(
            metadata_url,
            allow_loopback_http=True,
            endpoint_label="OAuth resource metadata challenge",
        )
    raw_scope = parameters.get("scope")
    scopes = ()
    if raw_scope is not None:
        scopes = _validate_scopes(tuple(raw_scope.split()), label="challenge scopes")
    error_code = parameters.get("error")
    if error_code is not None and (
        not error_code
        or len(error_code) > 128
        or not _HEADER_NAME_RE.fullmatch(error_code)
    ):
        raise McpOAuthError("MCP OAuth challenge rejected")
    return McpOAuthChallengeHints(
        resource_metadata_url=metadata_url,
        scopes=scopes,
        error_code=error_code,
    )


def _parse_auth_parameters(value: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    index = 0
    while index < len(value):
        index = _skip_auth_delimiters(value, index)
        if index >= len(value):
            break
        key, index = _parse_auth_parameter_key(value, index)
        parsed_value, index = _parse_auth_parameter_value(value, index)
        index = _consume_auth_parameter_separator(value, index)
        if key in selected or len(selected) >= 32:
            raise McpOAuthError("MCP OAuth challenge rejected")
        selected[key] = parsed_value
    if not selected:
        raise McpOAuthError("MCP OAuth challenge rejected")
    return selected


def _skip_auth_delimiters(value: str, index: int) -> int:
    while index < len(value) and value[index] in " \t,":
        index += 1
    return index


def _skip_auth_whitespace(value: str, index: int) -> int:
    while index < len(value) and value[index] in " \t":
        index += 1
    return index


def _parse_auth_parameter_key(value: str, index: int) -> tuple[str, int]:
    token_characters = (
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "!#$%&'*+-.^_`|~"
    )
    start = index
    while index < len(value) and value[index] in token_characters:
        index += 1
    key = value[start:index].casefold()
    index = _skip_auth_whitespace(value, index)
    if not key or index >= len(value) or value[index] != "=":
        raise McpOAuthError("MCP OAuth challenge rejected")
    return key, _skip_auth_whitespace(value, index + 1)


def _parse_auth_parameter_value(value: str, index: int) -> tuple[str, int]:
    if index < len(value) and value[index] == '"':
        return _parse_quoted_auth_value(value, index + 1)
    return _parse_unquoted_auth_value(value, index)


def _parse_quoted_auth_value(value: str, index: int) -> tuple[str, int]:
    characters: list[str] = []
    while index < len(value):
        character = value[index]
        index += 1
        if character == '"':
            return "".join(characters), index
        if character == "\\":
            character, index = _parse_quoted_auth_escape(value, index)
        if ord(character) < 0x20 and character != "\t":
            raise McpOAuthError("MCP OAuth challenge rejected")
        characters.append(character)
        if len(characters) > _MAX_URL_CHARS:
            raise McpOAuthError("MCP OAuth challenge rejected")
    raise McpOAuthError("MCP OAuth challenge rejected")


def _parse_quoted_auth_escape(value: str, index: int) -> tuple[str, int]:
    if index >= len(value) or value[index] not in {'"', "\\"}:
        raise McpOAuthError("MCP OAuth challenge rejected")
    return value[index], index + 1


def _parse_unquoted_auth_value(value: str, index: int) -> tuple[str, int]:
    start = index
    while index < len(value) and value[index] not in " \t,":
        if ord(value[index]) < 0x21:
            raise McpOAuthError("MCP OAuth challenge rejected")
        index += 1
    selected = value[start:index]
    if not selected or len(selected) > _MAX_URL_CHARS:
        raise McpOAuthError("MCP OAuth challenge rejected")
    return selected, index


def _consume_auth_parameter_separator(value: str, index: int) -> int:
    index = _skip_auth_whitespace(value, index)
    if index >= len(value):
        return index
    if value[index] != ",":
        raise McpOAuthError("MCP OAuth challenge rejected")
    return index + 1


def _validated_callback(
    profile: McpOAuthProfile,
    challenge: Mapping[str, Any],
    callback_url: str,
) -> dict[str, str]:
    if type(callback_url) is not str or not (1 <= len(callback_url) <= _MAX_URL_CHARS):
        raise McpOAuthError("MCP authorization callback rejected")
    parsed = urlsplit(callback_url)
    callback_base = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    if callback_base != profile.redirect_uri or parsed.fragment:
        raise McpOAuthError("MCP authorization callback rejected")
    pairs = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=16)
    if any(key not in _OAUTH_CALLBACK_KEYS for key, _value in pairs):
        raise McpOAuthError("MCP authorization callback rejected")
    values: dict[str, str] = {}
    for key, value in pairs:
        if key in values or len(value) > _MAX_TOKEN_CHARS:
            raise McpOAuthError("MCP authorization callback rejected")
        values[key] = value
    received_state = values.get("state")
    expected_state = challenge.get("state")
    if (
        type(received_state) is not str
        or type(expected_state) is not str
        or not hmac.compare_digest(received_state, expected_state)
    ):
        raise McpOAuthError("MCP authorization callback rejected")
    expected_issuer = challenge.get("expected_issuer")
    received_issuer = values.get("iss")
    issuer_required = challenge.get("issuer_parameter_required") is True
    if received_issuer is not None:
        if type(expected_issuer) is not str or not hmac.compare_digest(
            received_issuer, expected_issuer
        ):
            raise McpOAuthError("MCP authorization callback rejected")
    elif issuer_required:
        raise McpOAuthError("MCP authorization callback rejected")
    return values


def _validated_token_response(
    response: McpOAuthHttpResponse,
    *,
    profile: McpOAuthProfile,
    requested_scopes: tuple[str, ...],
    previous_refresh_token: str | None,
) -> dict[str, Any]:
    if response.status != 200:
        # Token endpoint bodies regularly contain echoed fields and are never
        # promoted into an exception or status projection.
        raise McpOAuthError("MCP OAuth token request was rejected")
    value = _decode_remote_object(response.body, max_bytes=_MAX_TOKEN_BYTES)
    access_token, expires_in, refresh_token = _validated_token_core(
        value,
        previous_refresh_token=previous_refresh_token,
    )
    scopes = _validated_token_scopes(
        value,
        profile=profile,
        requested_scopes=requested_scopes,
    )
    _validate_token_resource_and_audience(value, profile)
    return _token_secret_projection(
        profile,
        access_token=access_token,
        expires_in=expires_in,
        refresh_token=refresh_token,
        scopes=scopes,
        principal_sha256=_token_principal_sha256(value, profile),
    )


def _validated_token_core(
    value: Mapping[str, Any],
    *,
    previous_refresh_token: str | None,
) -> tuple[str, int, str | None]:
    access_token = value.get("access_token")
    token_type = value.get("token_type")
    expires_in = value.get("expires_in")
    if (
        type(access_token) is not str
        or not (1 <= len(access_token) <= _MAX_TOKEN_CHARS)
        or type(token_type) is not str
        or token_type.casefold() != "bearer"
        or type(expires_in) is not int
        or isinstance(expires_in, bool)
        or not (1 <= expires_in <= _MAX_TOKEN_LIFETIME_S)
    ):
        raise McpOAuthError("MCP OAuth token response rejected")
    refresh_token = value.get("refresh_token", previous_refresh_token)
    if refresh_token is not None and (
        type(refresh_token) is not str
        or not (1 <= len(refresh_token) <= _MAX_TOKEN_CHARS)
    ):
        raise McpOAuthError("MCP OAuth token response rejected")
    return access_token, expires_in, refresh_token


def _validated_token_scopes(
    value: Mapping[str, Any],
    *,
    profile: McpOAuthProfile,
    requested_scopes: tuple[str, ...],
) -> tuple[str, ...]:
    raw_scope = value.get("scope")
    if raw_scope is None:
        scopes = requested_scopes
    elif type(raw_scope) is str:
        pieces = tuple(raw_scope.split())
        scopes = _validate_scopes(pieces, label="token scopes")
    else:
        raise McpOAuthError("MCP OAuth token response rejected")
    if not set(requested_scopes).issubset(scopes) or not set(scopes).issubset(
        profile.allowed_scopes
    ):
        raise McpOAuthError("MCP OAuth token response rejected")
    return scopes


def _validate_token_resource_and_audience(
    value: Mapping[str, Any],
    profile: McpOAuthProfile,
) -> None:
    returned_resource = value.get("resource")
    if returned_resource is not None and returned_resource != profile.resource_uri:
        raise McpOAuthError("MCP OAuth token response rejected")
    returned_audience = value.get("audience")
    if returned_audience is not None:
        if type(returned_audience) is str:
            audiences = (returned_audience,)
        elif (
            isinstance(returned_audience, list)
            and returned_audience
            and all(type(item) is str for item in returned_audience)
        ):
            audiences = tuple(returned_audience)
        else:
            raise McpOAuthError("MCP OAuth token response rejected")
        expected_audience = profile.audience or profile.resource_uri
        if expected_audience not in audiences:
            raise McpOAuthError("MCP OAuth token response rejected")


def _token_principal_sha256(
    value: Mapping[str, Any],
    profile: McpOAuthProfile,
) -> str | None:
    subject = value.get("subject", value.get("sub"))
    if type(subject) is str and 0 < len(subject) <= 1024:
        return hashlib.sha256(
            f"{profile.expected_issuer}\x00{subject}".encode("utf-8")
        ).hexdigest()
    return None


def _token_secret_projection(
    profile: McpOAuthProfile,
    *,
    access_token: str,
    expires_in: int,
    refresh_token: str | None,
    scopes: tuple[str, ...],
    principal_sha256: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "access_token": access_token,
        "expires_at": time.time() + expires_in,
        "issuer": profile.expected_issuer,
        "resource": profile.resource_uri,
        "scopes": list(scopes),
        "token_type": "Bearer",
    }
    if refresh_token is not None:
        result["refresh_token"] = refresh_token
    if principal_sha256 is not None:
        result["principal_sha256"] = principal_sha256
    return result


def _profile_authority_projection(profile: McpOAuthProfile) -> dict[str, Any]:
    """Return the complete non-secret Host authority used to fence secrets."""

    return {
        "allowed_endpoint_origins": list(profile.allowed_endpoint_origins),
        "allowed_scopes": list(profile.allowed_scopes),
        "audience": profile.audience,
        "authorization_server_metadata_sha256": (
            profile.authorization_server_metadata_sha256
        ),
        "authorization_server_metadata_url": (
            profile.authorization_server_metadata_url
        ),
        "client_id": profile.client_id,
        "default_scopes": list(profile.default_scopes),
        "expected_issuer": profile.expected_issuer,
        "profile_id": profile.profile_id,
        "protected_resource_metadata_sha256": (
            profile.protected_resource_metadata_sha256
        ),
        "protected_resource_metadata_url": (
            profile.protected_resource_metadata_url
        ),
        "protocol_revision": profile.protocol_revision,
        "redirect_uri": profile.redirect_uri,
        "registration_mode": profile.registration_mode.value,
        "resource_uri": profile.resource_uri,
        "server_id": profile.server_id,
        "token_endpoint_auth_method": profile.token_endpoint_auth_method.value,
        "transport": profile.transport,
        "allow_loopback_http": profile.allow_loopback_http,
    }


def _profile_authority_sha256(profile: McpOAuthProfile) -> str:
    encoded = json.dumps(
        _profile_authority_projection(profile),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _encoded_bound_client_secret(
    profile: McpOAuthProfile,
    client_secret: bytes,
) -> bytes:
    encoded = json.dumps(
        {
            "client_secret": base64.b64encode(client_secret).decode("ascii"),
            "profile_authority_sha256": _profile_authority_sha256(profile),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > _MAX_SECRET_BYTES:
        raise McpOAuthError("invalid MCP credential value")
    return encoded


def _validated_bound_client_secret(
    encoded: bytes,
    profile: McpOAuthProfile,
) -> bytes:
    try:
        value = bounded_json_loads(encoded, max_bytes=_MAX_SECRET_BYTES)
        if not isinstance(value, dict) or set(value) != {
            "client_secret",
            "profile_authority_sha256",
        }:
            raise ValueError("invalid client secret envelope")
        profile_digest = value.get("profile_authority_sha256")
        if type(profile_digest) is not str or not hmac.compare_digest(
            profile_digest,
            _profile_authority_sha256(profile),
        ):
            raise ValueError("client secret authority changed")
        raw = value.get("client_secret")
        if type(raw) is not str:
            raise ValueError("invalid client secret envelope")
        decoded = base64.b64decode(raw, validate=True)
        if not (1 <= len(decoded) <= _MAX_SECRET_BYTES):
            raise ValueError("invalid client secret envelope")
        return decoded
    except Exception:
        raise McpOAuthNeedsAttention(
            "MCP OAuth credential requires Host attention"
        ) from None


def _authorization_metadata_projection(
    metadata: _AuthorizationMetadata,
) -> dict[str, Any]:
    return {
        "authorization_endpoint": metadata.authorization_endpoint,
        "issuer": metadata.issuer,
        "issuer_parameter_required": metadata.issuer_parameter_required,
        "revocation_endpoint": metadata.revocation_endpoint,
        "scopes_supported": list(metadata.scopes_supported),
        "token_endpoint": metadata.token_endpoint,
    }


def _validated_stored_authorization_metadata(
    value: Any,
    profile: McpOAuthProfile,
) -> _AuthorizationMetadata:
    if not isinstance(value, dict) or set(value) != {
        "authorization_endpoint",
        "issuer",
        "issuer_parameter_required",
        "revocation_endpoint",
        "scopes_supported",
        "token_endpoint",
    }:
        raise McpOAuthNeedsAttention(
            "MCP OAuth credential requires Host attention"
        )
    if value.get("issuer") != profile.expected_issuer:
        raise McpOAuthNeedsAttention(
            "MCP OAuth credential requires Host attention"
        )
    issuer_parameter_required = value.get("issuer_parameter_required")
    raw_scopes = value.get("scopes_supported")
    if type(issuer_parameter_required) is not bool or type(raw_scopes) is not list:
        raise McpOAuthNeedsAttention(
            "MCP OAuth credential requires Host attention"
        )
    try:
        authorization_endpoint, token_endpoint, revocation_endpoint = (
            _validated_authorization_endpoints(profile, value)
        )
        scopes = _validate_scopes(
            tuple(raw_scopes),
            label="stored authorization scopes",
        )
    except Exception:
        raise McpOAuthNeedsAttention(
            "MCP OAuth credential requires Host attention"
        ) from None
    return _AuthorizationMetadata(
        issuer=profile.expected_issuer,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        revocation_endpoint=revocation_endpoint,
        issuer_parameter_required=issuer_parameter_required,
        scopes_supported=scopes,
    )


def _validate_token_binding(
    value: Mapping[str, Any],
    profile: McpOAuthProfile,
    scopes: tuple[str, ...],
) -> None:
    if (
        value.get("issuer") != profile.expected_issuer
        or value.get("resource") != profile.resource_uri
        or value.get("scopes") != list(scopes)
        or value.get("token_type") != "Bearer"
    ):
        raise McpOAuthNeedsAttention("MCP OAuth credential requires Host attention")


def _validated_stored_token(
    value: Mapping[str, Any],
    profile: McpOAuthProfile,
) -> tuple[
    tuple[str, ...],
    float,
    str | None,
    int,
    _AuthorizationMetadata,
]:
    allowed = {
        "access_token",
        "authorization_metadata",
        "credential_generation",
        "expires_at",
        "issuer",
        "principal_sha256",
        "profile_authority_sha256",
        "refresh_token",
        "resource",
        "scopes",
        "token_type",
    }
    if set(value).difference(allowed):
        raise McpOAuthNeedsAttention("MCP OAuth credential requires Host attention")
    (
        refresh_token,
        selected_expiry,
        raw_scopes,
        principal,
        credential_generation,
        profile_digest,
    ) = _stored_token_core(value)
    scopes = _validate_scopes(tuple(raw_scopes), label="stored token scopes")
    if not set(scopes).issubset(profile.allowed_scopes):
        raise McpOAuthNeedsAttention("MCP OAuth credential requires Host attention")
    if not hmac.compare_digest(profile_digest, _profile_authority_sha256(profile)):
        raise McpOAuthNeedsAttention("MCP OAuth credential requires Host attention")
    _validate_token_binding(value, profile, scopes)
    metadata = _validated_stored_authorization_metadata(
        value.get("authorization_metadata"),
        profile,
    )
    if selected_expiry <= time.time() and not _has_refresh_token(value):
        raise McpOAuthAuthorizationRequired("MCP OAuth authorization is required")
    return (
        scopes,
        selected_expiry,
        principal,
        credential_generation,
        metadata,
    )


def _stored_token_core(
    value: Mapping[str, Any],
) -> tuple[str | None, float, list[Any], str | None, int, str]:
    _stored_bounded_text(value.get("access_token"), optional=False)
    refresh_token = _stored_bounded_text(
        value.get("refresh_token"),
        optional=True,
    )
    selected_expiry = _stored_token_expiry(value.get("expires_at"))
    raw_scopes = value.get("scopes")
    if type(raw_scopes) is not list:
        _invalid_stored_credential()
    principal = _stored_sha256(value.get("principal_sha256"), optional=True)
    credential_generation = _stored_credential_generation(
        value.get("credential_generation")
    )
    profile_digest = _stored_sha256(
        value.get("profile_authority_sha256"),
        optional=False,
    )
    return (
        refresh_token,
        selected_expiry,
        raw_scopes,
        principal,
        credential_generation,
        profile_digest,
    )


def _stored_bounded_text(value: Any, *, optional: bool) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or not (1 <= len(value) <= _MAX_TOKEN_CHARS):
        _invalid_stored_credential()
    return value


def _stored_token_expiry(value: Any) -> float:
    if (
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        _invalid_stored_credential()
    return float(value)


def _stored_credential_generation(value: Any) -> int:
    if (
        type(value) is not int
        or isinstance(value, bool)
        or not (1 <= value <= (2**63 - 1))
    ):
        _invalid_stored_credential()
    return value


def _stored_sha256(value: Any, *, optional: bool) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _invalid_stored_credential()
    return value


def _invalid_stored_credential() -> None:
    raise McpOAuthNeedsAttention("MCP OAuth credential requires Host attention")


def _validate_stored_token_fence(
    value: Mapping[str, Any],
    profile: McpOAuthProfile,
    *,
    expected_generation: int,
) -> None:
    _scopes, _expires_at, _principal, generation, _metadata = (
        _validated_stored_token(value, profile)
    )
    if generation != expected_generation:
        raise McpOAuthNeedsAttention(
            "MCP OAuth credential requires Host attention"
        )


def _validated_min_token_validity(value: float) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or not (0 <= float(value) <= 3600)
    ):
        raise McpOAuthError("invalid MCP OAuth token validity window")
    return float(value)


def _has_refresh_token(value: Mapping[str, Any]) -> bool:
    refresh_token = value.get("refresh_token")
    return type(refresh_token) is str and bool(refresh_token)


def _selected_revocation_token(value: Mapping[str, Any]) -> tuple[str, str]:
    refresh_token = value.get("refresh_token")
    token = refresh_token or value.get("access_token")
    if type(token) is not str or not token:
        raise McpOAuthNeedsAttention(
            "MCP OAuth credential requires Host attention"
        )
    token_type = "refresh_token" if refresh_token else "access_token"
    return token, token_type


def _basic_client_auth_header(client_id: str, client_secret: str) -> str:
    raw = (
        f"{quote(client_id, safe='')}:{quote(client_secret, safe='')}"
    ).encode("ascii")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _decode_token_secret(value: bytes) -> dict[str, Any]:
    decode_failed = False
    selected: Any = None
    try:
        selected = bounded_json_loads(value, max_bytes=_MAX_SECRET_BYTES)
    except Exception:
        decode_failed = True
    if decode_failed or not isinstance(selected, dict):
        raise McpOAuthNeedsAttention("MCP OAuth credential requires Host attention")
    return selected


def _decode_secret_object(value: bytes) -> dict[str, Any]:
    decode_failed = False
    selected: Any = None
    try:
        selected = bounded_json_loads(value, max_bytes=_MAX_SECRET_BYTES)
    except Exception:
        decode_failed = True
    if decode_failed or not isinstance(selected, dict):
        raise McpOAuthError("MCP authorization challenge is unavailable")
    return selected


def _decode_remote_object(
    value: bytes,
    *,
    max_bytes: int = _MAX_METADATA_BYTES,
) -> dict[str, Any]:
    decode_failed = False
    selected: Any = None
    try:
        selected = bounded_json_loads(value, max_bytes=max_bytes)
    except Exception:
        decode_failed = True
    if decode_failed or not isinstance(selected, dict):
        raise McpOAuthError("MCP authorization metadata rejected")
    return selected


def _required_secret_string(value: Mapping[str, Any], key: str) -> str:
    selected = value.get(key)
    if type(selected) is not str or not selected or len(selected) > _MAX_TOKEN_CHARS:
        raise McpOAuthError("MCP authorization challenge is unavailable")
    return selected


def _secret_scopes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise McpOAuthError("MCP authorization challenge is unavailable")
    return _validate_scopes(tuple(value), label="challenge scopes")


def _required_remote_string(value: Mapping[str, Any], key: str) -> str:
    selected = value.get(key)
    if type(selected) is not str or not (1 <= len(selected) <= _MAX_URL_CHARS):
        raise McpOAuthError("MCP authorization metadata rejected")
    return selected


def _optional_remote_string(value: Mapping[str, Any], key: str) -> str | None:
    selected = value.get(key)
    if selected is None:
        return None
    if type(selected) is not str or not (1 <= len(selected) <= _MAX_URL_CHARS):
        raise McpOAuthError("MCP authorization metadata rejected")
    return selected


def _optional_remote_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if (
        not isinstance(value, list)
        or len(value) > _MAX_SCOPES
        or any(type(item) is not str or len(item) > _MAX_SCOPE_CHARS for item in value)
    ):
        raise McpOAuthError("MCP authorization metadata rejected")
    return tuple(value)


def _optional_remote_scopes(value: Any) -> tuple[str, ...]:
    raw = _optional_remote_strings(value)
    if not raw:
        return ()
    try:
        return _validate_scopes(tuple(raw), label="metadata scopes")
    except McpOAuthError as exc:
        raise McpOAuthError("MCP authorization metadata rejected") from exc


def _require_digest(value: bytes, expected: str | None) -> None:
    if expected is None:
        return
    actual = hashlib.sha256(value).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise McpOAuthError("MCP authorization metadata rejected")


def _validate_secret_namespace(namespace: str) -> None:
    if (
        type(namespace) is not str
        or not (1 <= len(namespace) <= 256)
        or any(ord(character) < 0x20 for character in namespace)
    ):
        raise McpOAuthError("invalid MCP credential namespace")


def _oauth_client_namespace(profile_id: str) -> str:
    return f"oauth:{profile_id}:client"


def _oauth_challenge_namespace(profile_id: str) -> str:
    return f"oauth:{profile_id}:challenge"


def _oauth_token_bundle_namespace(profile_id: str) -> str:
    return f"oauth:{profile_id}:tokens"


def _oauth_token_namespace(profile_id: str, slot: int) -> str:
    if type(slot) is not int or not (0 <= slot < _TOKEN_SLOT_COUNT):
        raise McpOAuthError("invalid MCP OAuth token slot")
    return f"oauth:{profile_id}:tokens:{slot}"


def _oauth_profile_namespaces(
    profile_id: str,
    *,
    include_client: bool,
) -> tuple[str, ...]:
    if type(profile_id) is not str or _PROFILE_ID_RE.fullmatch(profile_id) is None:
        raise McpOAuthError("invalid MCP OAuth profile id")
    selected = [
        _oauth_challenge_namespace(profile_id),
        # Clean the pre-exact-slot development namespace as well.  Released
        # slot generations use the two bounded entries below.
        _oauth_token_bundle_namespace(profile_id),
        *(
            _oauth_token_namespace(profile_id, slot)
            for slot in range(_TOKEN_SLOT_COUNT)
        ),
    ]
    if include_client:
        selected.append(_oauth_client_namespace(profile_id))
    return tuple(selected)


def _validate_secret_input(namespace: str, value: bytes) -> None:
    _validate_secret_namespace(namespace)
    if not isinstance(value, bytes) or not (1 <= len(value) <= _MAX_SECRET_BYTES):
        raise McpOAuthError("invalid MCP credential value")


def _reserved_secret_ref(prefix: str, service: str, namespace: str) -> str:
    _validate_secret_namespace(namespace)
    digest = hashlib.sha256(f"{service}\0{namespace}".encode("utf-8")).digest()
    # OAuth owns a closed set of profile-scoped slots and intentionally needs
    # to derive the same reference after a crash.  Other MCP secret users get
    # a fresh exact slot so staged replacement cannot alias and then delete
    # the prior committed value.
    material = digest if namespace.startswith("oauth:") else digest[:16] + secrets.token_bytes(16)
    return f"{prefix}:{_b64url(material)}"


def _secret_ref_matches_namespace(
    secret_ref: str,
    *,
    prefix: str,
    service: str,
    namespace: str,
) -> bool:
    try:
        _validate_secret_ref(secret_ref, prefix=prefix)
        encoded = secret_ref.split(":", 1)[1]
        padded = encoded + ("=" * (-len(encoded) % 4))
        material = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception:
        return False
    if len(material) != 32:
        return False
    digest = hashlib.sha256(f"{service}\0{namespace}".encode("utf-8")).digest()
    expected = digest if namespace.startswith("oauth:") else digest[:16]
    observed = material if namespace.startswith("oauth:") else material[:16]
    return hmac.compare_digest(observed, expected)


def _zero_bytearray(value: bytearray) -> None:
    for index in range(len(value)):
        value[index] = 0


def _validate_secret_ref(secret_ref: str, *, prefix: str) -> None:
    if (
        type(secret_ref) is not str
        or not secret_ref.startswith(f"{prefix}:")
        or not _SECRET_REF_RE.fullmatch(secret_ref)
    ):
        raise McpOAuthError("MCP credential is unavailable")


def _delete_secret_quietly(broker: McpCredentialBroker, secret_ref: str) -> None:
    try:
        broker.delete_secret(secret_ref)
    except Exception:
        # Cleanup must not replace the already-sanitized primary outcome.  A
        # failed keychain delete is handled by Host lifecycle diagnostics.
        pass


def _parse_optional_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) is not str or len(value) > 64:
        raise McpOAuthError("invalid MCP credential expiration")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise McpOAuthError("invalid MCP credential expiration") from exc
    if parsed.tzinfo is None:
        raise McpOAuthError("invalid MCP credential expiration")
    return parsed.timestamp()


def _format_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _random_b64url(byte_length: int) -> str:
    return _b64url(secrets.token_bytes(byte_length))


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise McpOAuthTransportError("not_started")
    return remaining


def _prepare_oauth_http_request(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str] | None,
    body: bytes | None,
    deadline: float,
    max_response_bytes: int,
    resolver: Resolver,
    allow_loopback_http: bool,
    allow_loopback_tls: bool,
) -> _PreparedOAuthRequest:
    if method not in {"GET", "POST"}:
        raise McpOAuthTransportError("not_started")
    if type(max_response_bytes) is not int or not (
        1 <= max_response_bytes <= _MAX_TOKEN_BYTES
    ):
        raise McpOAuthTransportError("not_started")
    selected_body = b"" if body is None else bytes(body)
    if len(selected_body) > _MAX_FORM_BYTES:
        raise McpOAuthTransportError("not_started")
    endpoint = _validated_endpoint(
        url,
        allow_loopback_http=allow_loopback_http,
        endpoint_label="OAuth endpoint",
    )
    addresses = _resolved_oauth_addresses(
        resolver,
        endpoint,
        deadline=deadline,
        allow_loopback_http=allow_loopback_http,
        allow_loopback_tls=allow_loopback_tls,
    )
    request_target = endpoint.path or "/"
    if endpoint.query:
        request_target = f"{request_target}?{endpoint.query}"
    request_headers = _oauth_request_headers(
        endpoint,
        headers=headers,
        body_length=len(selected_body),
    )
    return _PreparedOAuthRequest(
        endpoint=endpoint,
        body=selected_body,
        request_bytes=_serialize_request(method, request_target, request_headers),
        addresses=addresses,
    )


def _resolved_oauth_addresses(
    resolver: Resolver,
    endpoint: _Endpoint,
    *,
    deadline: float,
    allow_loopback_http: bool,
    allow_loopback_tls: bool,
) -> tuple[str, ...]:
    try:
        addresses = tuple(resolver(endpoint.hostname, endpoint.port, deadline))
        allow_loopback = (
            endpoint.scheme == "http" and allow_loopback_http
        ) or (endpoint.scheme == "https" and allow_loopback_tls)
        _validate_resolved_addresses(
            endpoint.hostname,
            addresses,
            allow_loopback_http=allow_loopback,
        )
    except Exception:
        raise McpOAuthTransportError("not_started") from None
    return addresses


def _oauth_request_headers(
    endpoint: _Endpoint,
    *,
    headers: Mapping[str, str] | None,
    body_length: int,
) -> dict[str, str]:
    return {
        "Host": _host_header(endpoint.hostname, endpoint.port, endpoint.scheme),
        "Accept": "application/json",
        "User-Agent": "agent-libos-mcp-oauth/2",
        "Content-Length": str(body_length),
        "Connection": "close",
        **_validated_headers(headers or {}),
    }


def _request_oauth_address(
    prepared: _PreparedOAuthRequest,
    *,
    address: str,
    deadline: float,
    max_response_bytes: int,
    ssl_context: ssl.SSLContext,
) -> McpOAuthHttpResponse:
    sock: socket.socket | ssl.SSLSocket | None = None
    deadline_timer: threading.Timer | None = None
    dispatched = False
    try:
        endpoint = prepared.endpoint
        sock = socket.create_connection(
            (address, endpoint.port),
            timeout=_remaining(deadline),
        )
        deadline_timer = _socket_deadline_timer(sock, deadline)
        sock.settimeout(_remaining(deadline))
        if endpoint.scheme == "https":
            sock = ssl_context.wrap_socket(sock, server_hostname=endpoint.hostname)
            sock.settimeout(_remaining(deadline))
        dispatched = True
        sock.sendall(prepared.request_bytes + prepared.body)
        return _read_oauth_http_response(
            sock,
            deadline=deadline,
            max_response_bytes=max_response_bytes,
        )
    except McpOAuthTransportError:
        raise
    except Exception:
        if dispatched:
            raise McpOAuthTransportError("unknown") from None
        raise _McpOAuthAddressUnavailable() from None
    finally:
        if deadline_timer is not None:
            deadline_timer.cancel()
            deadline_timer.join()
        _close_socket_quietly(sock)


def _socket_deadline_timer(
    sock: socket.socket | ssl.SSLSocket,
    deadline: float,
) -> threading.Timer:
    timer = threading.Timer(
        _remaining(deadline),
        lambda: _close_socket_quietly(sock),
    )
    timer.name = "agent-libos-mcp-oauth-deadline"
    timer.daemon = True
    timer.start()
    return timer


def _read_oauth_http_response(
    sock: socket.socket | ssl.SSLSocket,
    *,
    deadline: float,
    max_response_bytes: int,
) -> McpOAuthHttpResponse:
    response = http.client.HTTPResponse(sock)
    response.begin()
    raw_headers = response.getheaders()
    _validate_response_headers(raw_headers)
    body = _read_bounded_response(
        response,
        sock,
        max_response_bytes=max_response_bytes,
        deadline=deadline,
    )
    return McpOAuthHttpResponse(
        status=int(response.status),
        headers={str(key).casefold(): str(value) for key, value in raw_headers},
        body=body,
    )


def _resolve_addresses(host: str, port: int, deadline: float) -> Sequence[str]:
    try:
        infos = _bounded_provider_getaddrinfo(
            host,
            port,
            deadline=deadline,
            operation="MCP OAuth",
        )
    except Exception as exc:
        raise McpOAuthTransportError("not_started") from exc
    return tuple(sorted({str(info[4][0]) for info in infos}))


def _validate_resolved_addresses(
    host: str,
    addresses: Sequence[str],
    *,
    allow_loopback_http: bool,
) -> None:
    if not addresses:
        raise McpOAuthTransportError("not_started")
    host_is_loopback_name = host.casefold() in _LOOPBACK_HOSTS
    for address in addresses:
        try:
            selected = ipaddress.ip_address(address.strip("[]"))
        except (AttributeError, ValueError) as exc:
            raise McpOAuthTransportError("not_started") from exc
        if allow_loopback_http and host_is_loopback_name:
            if selected.is_loopback:
                continue
            raise McpOAuthTransportError("not_started")
        if (
            not selected.is_global
            or selected.is_private
            or selected.is_loopback
            or selected.is_link_local
            or selected.is_reserved
            or selected.is_multicast
            or selected.is_unspecified
        ):
            raise McpOAuthTransportError("not_started")


def _validated_headers(headers: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(headers, Mapping) or len(headers) > 32:
        raise McpOAuthTransportError("not_started")
    selected: dict[str, str] = {}
    forbidden = {"host", "content-length", "connection", "transfer-encoding"}
    for key, value in headers.items():
        if (
            type(key) is not str
            or type(value) is not str
            or not _HEADER_NAME_RE.fullmatch(key)
            or key.casefold() in forbidden
            or "\r" in value
            or "\n" in value
            or len(value) > _MAX_FORM_BYTES
        ):
            raise McpOAuthTransportError("not_started")
        selected[key] = value
    return selected


def _validate_response_headers(headers: Sequence[tuple[str, str]]) -> None:
    total = 0
    if len(headers) > 128:
        raise McpOAuthTransportError("started")
    for key, value in headers:
        total += len(str(key)) + len(str(value)) + 4
        if total > 64 * 1024:
            raise McpOAuthTransportError("started")


def _read_bounded_response(
    response: http.client.HTTPResponse,
    sock: socket.socket | ssl.SSLSocket,
    *,
    max_response_bytes: int,
    deadline: float,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        sock.settimeout(_remaining(deadline))
        remaining_capacity = max_response_bytes + 1 - total
        if remaining_capacity <= 0:
            raise McpOAuthTransportError("started")
        chunk = response.read1(min(64 * 1024, remaining_capacity))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_response_bytes:
            raise McpOAuthTransportError("started")
    return b"".join(chunks)


def _close_socket_quietly(sock: socket.socket | ssl.SSLSocket | None) -> None:
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


def _serialize_request(
    method: str,
    target: str,
    headers: Mapping[str, str],
) -> bytes:
    if "\r" in target or "\n" in target or not target.startswith("/"):
        raise McpOAuthTransportError("not_started")
    lines = [f"{method} {target} HTTP/1.1"]
    lines.extend(f"{key}: {value}" for key, value in headers.items())
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def _host_header(host: str, port: int, scheme: str) -> str:
    selected_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    if port == default_port:
        return selected_host
    return f"{selected_host}:{port}"


__all__ = [
    "InMemoryMcpCredentialBroker",
    "McpCredentialBroker",
    "McpOAuthAccessLease",
    "McpOAuthAuthorizationRequired",
    "McpOAuthChallengeHints",
    "McpOAuthCredentialFence",
    "McpOAuthError",
    "McpOAuthHttpResponse",
    "McpOAuthHttpTransport",
    "McpOAuthManager",
    "McpOAuthNeedsAttention",
    "McpOAuthProfile",
    "McpOAuthRegistrationMode",
    "McpOAuthTokenEndpointAuthMethod",
    "McpOAuthTransportError",
    "PinnedMcpOAuthHttpTransport",
    "SystemKeyringMcpCredentialBroker",
    "parse_mcp_oauth_www_authenticate",
]
