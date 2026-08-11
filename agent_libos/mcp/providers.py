"""Public Host/provider protocols for the MCP 2026-07-28 client."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from agent_libos.mcp.manifest import McpServerManifestV3
from agent_libos.mcp.types import (
    JsonValue,
    McpCompletionResult,
    McpOperationResult,
    McpPage,
    McpPrompt,
    McpPromptResult,
    McpResource,
    McpResourceContents,
    McpResourceTemplate,
    McpSubscriptionEvent,
)
from agent_libos.models.mcp import (
    McpProviderCallResult,
    McpServerSpec,
    McpToolListResult,
    McpToolSpec,
)
from agent_libos.substrate.base import ExecutableSnapshot


@runtime_checkable
class McpToolProvider(Protocol):
    """Frozen legacy v1/v2 Tool SPI; atomic validate+call is optional."""

    def list_tools(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpToolListResult: ...

    def call_tool(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, JsonValue],
        *,
        deadline: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpProviderCallResult: ...


@runtime_checkable
class McpAtomicToolProvider(Protocol):
    """Frozen legacy v1/v2 atomic Tool SPI."""

    def validate_and_call(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, JsonValue],
        *,
        deadline: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpProviderCallResult: ...


@runtime_checkable
class McpModernProviderIdentity(Protocol):
    """Explicit identity shared by every exact Manifest-v3 Provider SPI.

    Method-name overlap is insufficient to opt a legacy Provider into the
    2026-07-28 client. Runtime composition additionally validates these exact
    marker values, coroutine methods, and their closed signatures.

    A custom implementation is trusted Host code and must be cooperative: its
    async methods must not block the event-loop thread, run unbounded CPU work,
    or suppress cancellation, and must stop at the supplied absolute deadline.
    Runtime performs pre/post deadline checks and treats any entered-provider
    overrun as UNKNOWN/no-replay, but Python cannot safely preempt arbitrary
    in-process code.  Built-in SDK transports separately enforce bounded I/O.
    """

    mcp_manifest_schema_version: ClassVar[Literal[3]]
    mcp_protocol_revision: ClassVar[Literal["2026-07-28"]]


@runtime_checkable
class McpModernToolProvider(McpModernProviderIdentity, Protocol):
    """Exact-v3 asynchronous initial Tool-call SPI.

    Non-complete results are Host-captured into durable local continuation or
    Task references before they may cross the Runtime facade.
    """

    async def call_tool(
        self,
        manifest: McpServerManifestV3,
        tool_id: str,
        arguments: dict[str, JsonValue],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpOperationResult[dict[str, JsonValue]]: ...


@runtime_checkable
class McpModernContinuationProvider(McpModernProviderIdentity, Protocol):
    """Exact-v3 continuation SPI for every currently supported surface."""

    async def continue_tool(
        self,
        server: McpServerSpec,
        mcp_name: str,
        arguments: dict[str, JsonValue],
        input_responses: dict[str, JsonValue],
        request_state: str | None,
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]: ...

    async def continue_resource(
        self,
        server: McpServerSpec,
        resource_name: str,
        logical_id: str,
        input_responses: dict[str, JsonValue],
        request_state: str | None,
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]: ...

    async def continue_prompt(
        self,
        server: McpServerSpec,
        prompt_name: str,
        logical_id: str,
        arguments: dict[str, str],
        input_responses: dict[str, JsonValue],
        request_state: str | None,
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]: ...


@runtime_checkable
class McpResourceProvider(McpModernProviderIdentity, Protocol):
    """Exact-v3 asynchronous Resource SPI with Host-owned durable capture."""

    async def list_resources(
        self, server: McpServerSpec, cursor: str | None, *, deadline: float
    ) -> McpPage[McpResource]: ...

    async def list_resource_templates(
        self, server: McpServerSpec, cursor: str | None, *, deadline: float
    ) -> McpPage[McpResourceTemplate]: ...

    async def read_resource(
        self,
        server: McpServerSpec,
        resource_name: str,
        variables: Mapping[str, str] | None,
        *,
        deadline: float,
    ) -> McpOperationResult[McpResourceContents]: ...


@runtime_checkable
class McpPromptProvider(McpModernProviderIdentity, Protocol):
    """Exact-v3 Prompt SPI; Completion remains complete-only."""

    async def list_prompts(
        self, server: McpServerSpec, cursor: str | None, *, deadline: float
    ) -> McpPage[McpPrompt]: ...

    async def get_prompt(
        self,
        server: McpServerSpec,
        prompt_name: str,
        arguments: Mapping[str, str],
        *,
        deadline: float,
    ) -> McpOperationResult[McpPromptResult]: ...

    async def complete(
        self,
        server: McpServerSpec,
        reference: Mapping[str, JsonValue],
        argument: Mapping[str, str],
        context: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
    ) -> McpOperationResult[McpCompletionResult]: ...


@runtime_checkable
class McpSubscriptionProvider(McpModernProviderIdentity, Protocol):
    async def listen(
        self,
        server: McpServerSpec,
        filters: tuple[str, ...],
        *,
        deadline: float,
    ) -> "McpSubscriptionSession": ...

    async def receive(self, handle: Any, *, deadline: float) -> McpSubscriptionEvent: ...

    async def close(self, handle: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class McpSubscriptionSession:
    """Exact Host-owned projection of an opaque Provider listen handle.

    The manager never invokes arbitrary properties or iterates a Provider
    object to discover acknowledgement state.
    """

    handle: Any
    owner_task: asyncio.Task[Any]
    acknowledged_filters: tuple[str, ...] = ()


@runtime_checkable
class McpTasksExtensionProvider(McpModernProviderIdentity, Protocol):
    async def get_remote_task(
        self, server: McpServerSpec, remote_task_id: str, *, deadline: float
    ) -> Mapping[str, JsonValue]: ...

    async def update_remote_task(
        self,
        server: McpServerSpec,
        remote_task_id: str,
        response: Mapping[str, JsonValue],
        *,
        deadline: float,
    ) -> Mapping[str, JsonValue]: ...

    async def cancel_remote_task(
        self, server: McpServerSpec, remote_task_id: str, *, deadline: float
    ) -> Mapping[str, JsonValue]: ...


class McpCredentialBroker(Protocol):
    """Host secret SPI.  Implementations must not persist plaintext in Store."""

    def reserve_secret_ref(self, namespace: str) -> str:
        """Reserve an opaque exact slot locally, without a persistent write.

        General staged values use fresh refs.  A broker may use a deterministic
        ref only for a closed Host-owned namespace that needs crash cleanup.
        """

    def put_secret_at(
        self,
        secret_ref: str,
        namespace: str,
        value: bytes,
        *,
        expires_at: str | None,
    ) -> None:
        """Write that exact slot, idempotently for equal bytes or fail on conflict."""

    def put_secret(self, namespace: str, value: bytes, *, expires_at: str | None) -> str: ...

    def get_secret(self, secret_ref: str) -> bytes: ...

    def delete_secret(self, secret_ref: str) -> None: ...

    def available(self) -> bool: ...
