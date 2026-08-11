"""Runtime composition bridge for the modern MCP client.

This module contains no authority admission.  ``McpPrimitive`` owns the
ProtectedOperation/DataFlow/Capability facade and supplies the governed SDK
session context.  The bridge resolves exact v3 registry/auth fences, selects
optional caller-owned SPI implementations, and ensures every SDK session is
owned by the Runtime connection supervisor.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from collections.abc import AsyncIterator, Callable, Mapping
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, AsyncContextManager, Protocol

from agent_libos.mcp.client import (
    McpClientBinding,
    current_mcp_client_binding,
    mcp_transport_spec_from_v3,
)
from agent_libos.mcp.environment import McpTransportEnvironmentSnapshot
from agent_libos.mcp.manifest import McpServerManifestV3
from agent_libos.mcp.oauth import (
    McpOAuthAccessLease,
    McpOAuthCredentialFence,
    McpOAuthManager,
)
from agent_libos.mcp.supervisor import (
    McpConnectionFence,
    McpConnectionSupervisor,
)
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.models.mcp import McpHeaderSpec, McpServerSpec
from agent_libos.substrate.base import ProviderEffectNotStarted
from agent_libos.utils.serde import dumps


_SHA256_CHARS = frozenset("0123456789abcdef")
_INTERNAL_OAUTH_ENV = "AGENT_LIBOS_INTERNAL_MCP_OAUTH_TOKEN"


class McpGovernedSessionContextFactory(Protocol):
    """Primitive callback entered inside the protected provider phase."""

    def __call__(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        binding: McpClientBinding,
        task_notification_ingress: Callable[[Mapping[str, Any] | None], None]
        | None = None,
    ) -> AsyncContextManager[Any]: ...


class McpRuntimeBindingResolver:
    """Resolve exact v3 registry and non-secret OAuth authority state."""

    def __init__(
        self,
        extensions: Any,
        primitive: Any,
        *,
        oauth_manager: McpOAuthManager | None = None,
    ) -> None:
        self.extensions = extensions
        self.primitive = primitive
        self.oauth_manager = oauth_manager

    def __call__(self, server_id: str) -> McpClientBinding:
        return self.resolve(server_id, owner_id=None)

    def resolve(
        self,
        server_id: str,
        *,
        owner_id: str | None,
    ) -> McpClientBinding:
        if type(server_id) is not str or not server_id:
            raise ValidationError("MCP server_id is invalid")
        if owner_id is not None and (type(owner_id) is not str or not owner_id):
            raise ValidationError("MCP operation owner is invalid")
        found = self.extensions.get_mcp_v3_server(server_id)
        if found is None:
            raise NotFound(f"MCP Manifest v3 server not found: {server_id}")
        manifest, _metadata = found
        if not isinstance(manifest, McpServerManifestV3):
            raise ValidationError("MCP registry returned a non-v3 server")
        registry = self.extensions.get_mcp_registry_binding(server_id)
        generation, spec_sha256 = _registry_fence(registry)
        # McpClientBinding uses the canonical manifest serializer itself; the
        # explicit comparison here catches a mismatched Store implementation
        # before environment or credential material is read.
        provisional = McpClientBinding(
            manifest=manifest,
            registry_generation=generation,
            owner_id=owner_id,
        )
        expected_sha256 = provisional.manifest_sha256
        if spec_sha256 != expected_sha256:
            raise ValidationError("MCP registry spec digest changed during binding")

        server = mcp_transport_spec_from_v3(manifest)
        # Capture the ambient inputs once, then resolve and validate them from
        # that immutable snapshot.  The governed provider-phase context uses
        # the same input snapshot, so a later os.environ mutation cannot
        # change headers or stdio child variables after authority admission.
        snapshot = self.primitive.snapshot_modern_transport_environment(server)
        if not isinstance(snapshot, McpTransportEnvironmentSnapshot):
            raise ValidationError("MCP primitive returned an invalid environment snapshot")
        selected_environment = dict(snapshot.runtime_environment)
        static_secrets = snapshot.sensitive_values

        auth_generation = 0
        auth_principal_sha256: str | None = None
        auth_scope_sha256: str | None = None
        if manifest.auth_profile_id is not None:
            if manifest.transport != "streamable_http":  # manifest invariant, defensive
                raise ValidationError("MCP OAuth is supported only for Streamable HTTP")
            if self.oauth_manager is None:
                raise ValidationError("MCP OAuth manager is unavailable")
            auth_fence = self.oauth_manager.credential_fence(manifest.auth_profile_id)
            _validate_auth_fence(auth_fence, manifest)
            auth_generation = auth_fence.credential_generation
            auth_principal_sha256 = auth_fence.principal_sha256
            auth_scope_sha256 = _scope_sha256(auth_fence.scopes)

        return McpClientBinding(
            manifest=manifest,
            registry_generation=generation,
            auth_generation=auth_generation,
            auth_principal_sha256=auth_principal_sha256,
            auth_scope_sha256=auth_scope_sha256,
            owner_id=owner_id,
            sensitive_values=static_secrets,
            runtime_environment=selected_environment,
        )


class McpGovernedSdkSessionFactory:
    """Enter an OAuth-aware raw governed context in the calling task.

    This layer deliberately does not acquire a connection supervisor.  It is
    used directly by the subscription adapter's dedicated owner task and is
    wrapped by :class:`McpSupervisedSdkSessionFactory` for stateless reads.
    """

    def __init__(
        self,
        governed_context_factory: McpGovernedSessionContextFactory,
        *,
        oauth_manager: McpOAuthManager | None = None,
    ) -> None:
        self.governed_context_factory = governed_context_factory
        self.oauth_manager = oauth_manager
        self._operation_secrets: ContextVar[tuple[str, ...]] = ContextVar(
            f"agent_libos_mcp_transport_secrets_{id(self)}", default=()
        )

    def sensitive_values(self, server_id: str) -> tuple[str, ...]:
        binding = current_mcp_client_binding()
        if binding.manifest.server_id != server_id:
            raise ValidationError("MCP transport secret snapshot belongs to another server")
        return tuple(
            dict.fromkeys((*binding.sensitive_values, *self._operation_secrets.get()))
        )

    @contextlib.asynccontextmanager
    async def __call__(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        binding: McpClientBinding | None = None,
        task_notification_ingress: Callable[[Mapping[str, Any] | None], None]
        | None = None,
    ) -> AsyncIterator[Any]:
        selected_binding = binding or current_mcp_client_binding()
        _validate_session_server(server, selected_binding)
        if selected_binding.owner_id is None:
            raise ValidationError("MCP governed SDK session requires an operation owner")

        lease: McpOAuthAccessLease | None = None
        environment = dict(selected_binding.runtime_environment or {})
        operation_secrets: list[str] = list(selected_binding.sensitive_values)
        selected_server = server
        if selected_binding.manifest.auth_profile_id is not None:
            if self.oauth_manager is None:
                raise ValidationError("MCP OAuth manager is unavailable")
            # transport_access may refresh once under the OAuth manager's own
            # no-replay policy.  A rotated fence invalidates this already
            # admitted MCP operation instead of silently widening authority.
            lease = self.oauth_manager.transport_access(
                selected_binding.manifest.auth_profile_id,
                deadline=deadline,
            )
            if not _lease_matches_binding(lease, selected_binding):
                lease.close()
                raise ValidationError("MCP OAuth credential fence changed before dispatch")
            if not self.oauth_manager.validate_credential_fence(lease.fence):
                lease.close()
                raise ValidationError("MCP OAuth credential fence is no longer active")
            authorization = lease.authorization_header()
            # This synthetic environment name is never accepted from a
            # manifest and exists only in this provider-phase snapshot.  The
            # primitive resolves it through the synthetic Authorization
            # header spec immediately before entering the governed SDK
            # transport.
            environment[_INTERNAL_OAUTH_ENV] = authorization
            operation_secrets.extend(
                (
                    authorization,
                    authorization.removeprefix("Bearer "),
                    *lease.redaction_values(),
                )
            )
            selected_server = _server_with_synthetic_oauth_header(server)

        secret_token = self._operation_secrets.set(tuple(operation_secrets))
        try:
            # SDK transports and their AnyIO/OTel scopes are task-affine.
            # Enter, use, and exit the operation-local context in this exact
            # provider task; passing an entered context through the generic
            # supervisor would make its opener/closer tasks violate that
            # invariant.  Stateless reads are never reusable and therefore
            # leave no connection for the supervisor to retain.  The
            # supervisor remains the owner of long-lived subscription handles.
            async with self.governed_context_factory(
                replace(selected_server),
                deadline=deadline,
                binding=replace(selected_binding, runtime_environment=environment),
                task_notification_ingress=task_notification_ingress,
            ) as selected:
                yield selected
        except Exception as exc:
            if isinstance(exc, ProviderEffectNotStarted):
                raise
            # Provider exception messages can contain undeclared credentials,
            # URLs, local executable paths, or raw protocol bytes.  Known
            # exact-secret replacement is insufficient for unknown values, so
            # this bridge exposes only stable Host-owned classifications.
            if isinstance(exc, TimeoutError):
                raise TimeoutError(
                    "MCP governed session exceeded the absolute deadline"
                ) from None
            raise ValidationError("MCP governed session failed") from None
        finally:
            self._operation_secrets.reset(secret_token)
            if lease is not None:
                lease.close()


class _McpConnectionPermit:
    """No-I/O marker closed by the supervisor on a fence transition."""

    async def aclose(self) -> None:
        return None


class McpSupervisedSdkSessionFactory:
    """Fence one stateless SDK operation without owning its task-affine context."""

    def __init__(
        self,
        supervisor: McpConnectionSupervisor,
        governed_context_factory: (
            McpGovernedSessionContextFactory | McpGovernedSdkSessionFactory
        ),
        *,
        oauth_manager: McpOAuthManager | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.governed = (
            governed_context_factory
            if isinstance(governed_context_factory, McpGovernedSdkSessionFactory)
            else McpGovernedSdkSessionFactory(
                governed_context_factory,
                oauth_manager=oauth_manager,
            )
        )

    def sensitive_values(self, server_id: str) -> tuple[str, ...]:
        return self.governed.sensitive_values(server_id)

    @contextlib.asynccontextmanager
    async def __call__(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
    ) -> AsyncIterator[Any]:
        binding = current_mcp_client_binding()
        _validate_session_server(server, binding)
        if binding.owner_id is None:
            raise ValidationError("MCP governed SDK session requires an operation owner")
        owner_task = asyncio.current_task()
        if owner_task is None:  # pragma: no cover - requires a broken event loop
            raise RuntimeError("MCP SDK operation has no owner task")

        def cancel_operation(_connection_id: str, _reason: str) -> None:
            # McpModernClient._invoke creates one dedicated task per Provider
            # operation.  Cancelling that captured task unwinds the SDK/AnyIO
            # context in its owner without cancelling the outer Runtime task.
            if not owner_task.done():
                owner_task.cancel()

        async def open_permit() -> _McpConnectionPermit:
            return _McpConnectionPermit()

        managed = await self.supervisor.acquire(
            _connection_fence(binding),
            "read",
            open_permit,
            reusable=False,
            deadline=deadline,
            on_lost=cancel_operation,
            # The permit has no transport/context state and may be closed by
            # an invalidation task.  The SDK context below remains task-affine.
            task_affine=False,
        )
        try:
            async with self.governed(
                server,
                deadline=deadline,
                binding=binding,
            ) as selected:
                yield selected
        finally:
            await self.supervisor.release(managed)

    async def close(self) -> None:
        await self.supervisor.close()


def _registry_fence(value: Any) -> tuple[int, str]:
    if not isinstance(value, Mapping):
        raise ValidationError("MCP registry binding is invalid")
    generation = value.get("registry_generation")
    spec_sha256 = value.get("registry_spec_sha256")
    if type(generation) is not int or generation < 0:
        raise ValidationError("MCP registry generation is invalid")
    _validate_sha256(spec_sha256, "registry spec")
    return generation, spec_sha256


def _validate_auth_fence(
    fence: McpOAuthCredentialFence,
    manifest: McpServerManifestV3,
) -> None:
    if not isinstance(fence, McpOAuthCredentialFence):
        raise ValidationError("MCP OAuth manager returned an invalid credential fence")
    if fence.profile_id != manifest.auth_profile_id or fence.server_id != manifest.server_id:
        raise ValidationError("MCP OAuth profile belongs to another server")
    if type(fence.credential_generation) is not int or fence.credential_generation < 0:
        raise ValidationError("MCP OAuth credential generation is invalid")
    if fence.principal_sha256 is not None:
        _validate_sha256(fence.principal_sha256, "OAuth principal")


def _scope_sha256(scopes: tuple[str, ...]) -> str:
    if type(scopes) is not tuple or any(type(item) is not str or not item for item in scopes):
        raise ValidationError("MCP OAuth scope fence is invalid")
    return hashlib.sha256(dumps(list(scopes)).encode("utf-8")).hexdigest()


def _lease_matches_binding(
    lease: McpOAuthAccessLease,
    binding: McpClientBinding,
) -> bool:
    if not isinstance(lease, McpOAuthAccessLease):
        return False
    fence = lease.fence
    return bool(
        fence.profile_id == binding.manifest.auth_profile_id
        and fence.server_id == binding.manifest.server_id
        and fence.credential_generation == binding.auth_generation
        and fence.principal_sha256 == binding.auth_principal_sha256
        and _scope_sha256(fence.scopes) == binding.auth_scope_sha256
    )


def _server_with_synthetic_oauth_header(server: McpServerSpec) -> McpServerSpec:
    if server.transport != "streamable_http" or server.http is None:
        raise ValidationError("MCP OAuth requires Streamable HTTP")
    if any(name.casefold() == "authorization" for name in server.http.headers):
        raise ValidationError("MCP OAuth conflicts with a static Authorization header")
    headers = dict(server.http.headers)
    headers["Authorization"] = McpHeaderSpec(env=_INTERNAL_OAUTH_ENV)
    return replace(server, http=replace(server.http, headers=headers))


def _validate_session_server(server: McpServerSpec, binding: McpClientBinding) -> None:
    expected = mcp_transport_spec_from_v3(binding.manifest)
    if server != expected:
        raise ValidationError("MCP SDK provider requested a session for another manifest")


def _connection_fence(binding: McpClientBinding) -> McpConnectionFence:
    values: dict[str, Any] = {
        "server_id": binding.manifest.server_id,
        "server_spec_sha256": binding.manifest_sha256,
        "registry_generation": binding.registry_generation,
        "owner": binding.owner_id or "",
        "auth_principal_sha256": binding.auth_principal_sha256,
        "auth_generation": binding.auth_generation,
    }
    # Older custom supervisor implementations may predate the scope field;
    # the released built-in supervisor includes it.  Fail closed if a scoped
    # OAuth binding cannot be represented.
    fields = getattr(McpConnectionFence, "__dataclass_fields__", {})
    if "auth_scope_sha256" in fields:
        values["auth_scope_sha256"] = binding.auth_scope_sha256
    elif binding.auth_scope_sha256 is not None:
        raise ValidationError("MCP connection supervisor cannot fence OAuth scope")
    return McpConnectionFence(**values)


def mcp_connection_fence(binding: McpClientBinding) -> McpConnectionFence:
    """Project one validated client binding into the supervisor fence type."""

    if not isinstance(binding, McpClientBinding):
        raise ValidationError("MCP client binding is invalid")
    return _connection_fence(binding)


def _validate_sha256(value: Any, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValidationError(f"MCP {label} digest is invalid")


__all__ = [
    "McpGovernedSdkSessionFactory",
    "McpGovernedSessionContextFactory",
    "McpRuntimeBindingResolver",
    "McpSupervisedSdkSessionFactory",
    "mcp_connection_fence",
]
