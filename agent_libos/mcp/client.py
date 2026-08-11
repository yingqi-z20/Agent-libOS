"""Governed Host client for Manifest-v3 MCP read surfaces.

The client deliberately owns no ambient registry, credential, or transport
state.  A Host supplies a fenced binding resolver and surface-specific
providers.  Every operation snapshots that fence, carries one absolute
deadline through provider and projection work, and rejects a result if the
registry or authenticated principal changed meanwhile.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, AsyncContextManager, Iterator, Literal, Protocol, TypeVar, cast
from urllib.parse import quote

from agent_libos.mcp._input import sdk_json_mapping
from agent_libos.mcp.manifest import (
    MCP_V3_PROTOCOL_REVISION,
    McpPromptSpec,
    McpResourceSpec,
    McpResourceTemplateSpec,
    McpServerManifestV3,
    canonical_mcp_v3_manifest_json,
    validate_mcp_v3_manifest,
)
from agent_libos.mcp.prompts import (
    sanitize_completion_result,
    sanitize_prompt_result,
    sdk_completion_result,
    sdk_prompt,
    sdk_prompt_result,
)
from agent_libos.mcp.providers import McpPromptProvider, McpResourceProvider
from agent_libos.mcp.resources import (
    McpArtifactWriter,
    bounded_public_size,
    cache_hint_from_sdk,
    is_mcp_app_mime,
    redact_public_dataclass,
    reject_mcp_app_selector,
    sanitize_provider_json,
    sanitize_resource_contents,
    sdk_resource,
    sdk_resource_contents,
    sdk_resource_template,
)
from agent_libos.mcp.types import (
    JsonValue,
    McpCacheHint,
    McpComplete,
    McpCompletionResult,
    McpInputRequest,
    McpInputRequestKind,
    McpInputRequired,
    McpOperationResult,
    McpPage,
    McpPrompt,
    McpPromptResult,
    McpRemoteTask,
    McpResource,
    McpResourceContents,
    McpResourceTemplate,
)
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.models.mcp import McpServerSpec
from agent_libos.substrate.base import ProviderEffectNotStarted
from agent_libos.utils.redaction import redact_sensitive_text
from agent_libos.utils.serde import dumps, to_jsonable


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TEMPLATE_FIELD_RE = re.compile(r"\{([A-Za-z0-9][A-Za-z0-9_.@+-]*)\}")
_T = TypeVar("_T")
_ACTIVE_CLIENT_BINDING: ContextVar[McpClientBinding | None]  # defined after class
_PROVIDER_CANCELLATION_DRAIN_S = 1.0
_SYNC_LOOP_CANCELLATION_DRAIN_S = 0.05
_SYNC_LOOP_FORCE_CLOSE_PASSES = 3
_INPUT_REQUIRED_CONTINUATION_METHODS = frozenset(
    {"tools/call", "resources/read", "prompts/get"}
)


class McpContinuationSurfaceUnsupported(ValidationError):
    """The official exact-modern SDK cannot continue this operation surface."""


@dataclass(frozen=True)
class McpClientBinding:
    """One ephemeral Host snapshot used to fence an MCP read operation."""

    manifest: McpServerManifestV3
    registry_generation: int
    auth_generation: int = 0
    auth_principal_sha256: str | None = None
    auth_scope_sha256: str | None = None
    owner_id: str | None = None
    sensitive_values: tuple[str, ...] = field(default=(), repr=False, compare=False)
    runtime_environment: Mapping[str, str] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(
            canonical_mcp_v3_manifest_json(self.manifest).encode("utf-8")
        ).hexdigest()

    @property
    def fence(
        self,
    ) -> tuple[str, int, int, str | None, str | None, str | None]:
        return (
            self.manifest_sha256,
            self.registry_generation,
            self.auth_generation,
            self.auth_principal_sha256,
            self.auth_scope_sha256,
            self.owner_id,
        )


class McpClientBindingResolver(Protocol):
    def __call__(self, server_id: str) -> McpClientBinding: ...


_ACTIVE_CLIENT_BINDING = ContextVar(
    "agent_libos_mcp_active_client_binding", default=None
)
_ACTIVE_CLIENT_LOGICAL_ID: ContextVar[str | None] = ContextVar(
    "agent_libos_mcp_active_client_logical_id", default=None
)


def current_mcp_client_binding() -> McpClientBinding:
    """Return the binding only while a provider dispatch is active."""

    selected = _ACTIVE_CLIENT_BINDING.get()
    if selected is None:
        raise ValidationError("MCP client binding is unavailable outside provider dispatch")
    return selected


def _active_client_logical_id(fallback: str) -> str:
    """Select the Host logical id without placing it on the MCP wire."""

    selected = _ACTIVE_CLIENT_LOGICAL_ID.get()
    if selected is None:
        return fallback
    _validate_id(selected, "operation logical id")
    return selected


def _active_operation_sensitive_values(
    explicit: tuple[str, ...],
) -> tuple[str, ...]:
    """Merge caller hints with the unforgeable active binding snapshot."""

    current = _ACTIVE_CLIENT_BINDING.get()
    sources = (explicit, current.sensitive_values if current is not None else ())
    selected: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if type(source) is not tuple:
            raise ValidationError("MCP operation sensitive values are invalid")
        for value in source:
            if type(value) is not str or not value:
                raise ValidationError("MCP operation sensitive value is invalid")
            if value not in seen:
                seen.add(value)
                selected.append(value)
    return tuple(selected)


@contextmanager
def bind_mcp_client_binding(binding: McpClientBinding) -> Iterator[None]:
    """Bind an exact Host-resolved fence during protected non-client dispatch.

    Subscription start uses this scope after its own ProtectedOperation
    admission.  Provider-owned owner tasks inherit the ContextVar snapshot;
    untrusted GUI/model arguments can never construct or select the binding.
    """

    if not isinstance(binding, McpClientBinding):
        raise ValidationError("MCP client binding is invalid")
    validate_mcp_v3_manifest(binding.manifest)
    _validate_binding(binding, binding.owner_id)
    current = _ACTIVE_CLIENT_BINDING.get()
    if current is not None and current.fence != binding.fence:
        raise ValidationError("MCP client binding scope conflicts with active operation")
    token = _ACTIVE_CLIENT_BINDING.set(binding)
    try:
        yield
    finally:
        _ACTIVE_CLIENT_BINDING.reset(token)


def mcp_prompt_preview_sha256(
    *,
    binding: McpClientBinding,
    prompt_id: str,
    arguments: Mapping[str, str],
    prompt: McpPromptResult,
) -> str:
    """Bind one sanitized Prompt preview to its exact authority snapshot."""

    if not isinstance(binding, McpClientBinding):
        raise ValidationError("MCP Prompt preview binding is invalid")
    _validate_binding(binding, binding.owner_id)
    _validate_id(prompt_id, "prompt_id")
    if not isinstance(prompt, McpPromptResult) or prompt.prompt_id != prompt_id:
        raise ValidationError("MCP Prompt preview projection is invalid")
    if prompt.user_confirmation_required is not True:
        raise ValidationError("MCP Prompt preview must require user confirmation")
    normalized_arguments: dict[str, str] = {}
    if not isinstance(arguments, Mapping):
        raise ValidationError("MCP Prompt preview arguments must be an object")
    for key, value in arguments.items():
        if type(key) is not str or type(value) is not str:
            raise ValidationError("MCP Prompt preview arguments must be strings")
        normalized_arguments[key] = value
    for message in prompt.messages:
        if message.role not in {"user", "assistant"}:
            raise ValidationError("MCP Prompt preview role is invalid")
        if message.provenance != "untrusted_mcp_prompt":
            raise ValidationError("MCP Prompt preview provenance is invalid")
    payload = {
        "schema_version": 1,
        "server_id": binding.manifest.server_id,
        "server_spec_sha256": binding.manifest_sha256,
        "registry_generation": binding.registry_generation,
        "auth_principal_sha256": binding.auth_principal_sha256,
        "auth_scope_sha256": binding.auth_scope_sha256,
        "auth_generation": binding.auth_generation,
        "owner_id": binding.owner_id,
        "prompt_id": prompt_id,
        "arguments": normalized_arguments,
        "public_prompt": to_jsonable(prompt),
    }
    return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class McpModernClientLimits:
    # ``max_page_items`` is a compatibility override for early Host callers.
    # Runtime composition leaves it unset and applies the purpose-specific
    # catalog bounds below so one surface cannot silently widen another.
    max_page_items: int | None = None
    max_resource_items: int = 100
    max_resource_template_items: int = 100
    max_prompt_items: int = 100
    max_content_blocks: int = 256
    max_prompt_messages: int = 100
    max_completion_values: int = 100
    max_cursor_bytes: int = 4096
    max_cursor_handles: int = 256
    cursor_ttl_s: float = 300.0
    max_cache_ttl_ms: int = 300_000
    max_argument_name_bytes: int = 256
    max_argument_value_bytes: int = 64 * 1024

    def validate(self) -> None:
        integer_fields = (
            "max_resource_items",
            "max_resource_template_items",
            "max_prompt_items",
            "max_content_blocks",
            "max_prompt_messages",
            "max_completion_values",
            "max_cursor_bytes",
            "max_cursor_handles",
            "max_cache_ttl_ms",
            "max_argument_name_bytes",
            "max_argument_value_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValidationError(f"MCP client {name} must be a positive integer")
        if self.max_page_items is not None and (
            type(self.max_page_items) is not int or self.max_page_items <= 0
        ):
            raise ValidationError(
                "MCP client max_page_items must be null or a positive integer"
            )
        if type(self.cursor_ttl_s) not in {int, float} or self.cursor_ttl_s <= 0:
            raise ValidationError("MCP client cursor_ttl_s must be positive")


def sanitize_mcp_operation_result(
    raw: Any,
    *,
    binding: McpClientBinding,
    logical_id: str,
    value_type: type[Any],
    surface: str,
    limits: McpModernClientLimits,
) -> McpOperationResult[Any]:
    """Project one provider-controlled modern result before any public sink.

    Custom Tool providers do not necessarily pass through the SDK result
    adapter.  Keeping this projection public and shared ensures their results
    receive the same exact-operation-secret redaction and structural bounds as
    Resources, Prompts, and Completions before Runtime evidence is completed.
    """

    if not isinstance(binding, McpClientBinding):
        raise ValidationError("MCP operation result binding is invalid")
    if not isinstance(limits, McpModernClientLimits):
        raise ValidationError("MCP operation result limits are invalid")
    limits.validate()
    bounded_public_size(
        raw,
        maximum=binding.manifest.max_response_bytes,
        label=f"MCP {surface} provider result",
    )
    if isinstance(raw, McpComplete):
        if not isinstance(raw.value, value_type):
            raise ValidationError(f"MCP {surface} provider returned an invalid Complete value")
        if value_type is McpResourceContents:
            value = sanitize_resource_contents(
                raw.value,
                server_id=binding.manifest.server_id,
                logical_id=logical_id,
                sensitive_values=binding.sensitive_values,
                maximum_content_blocks=limits.max_content_blocks,
            )
        elif value_type is McpPromptResult:
            value = sanitize_prompt_result(
                raw.value,
                server_id=binding.manifest.server_id,
                logical_id=logical_id,
                sensitive_values=binding.sensitive_values,
                maximum_messages=limits.max_prompt_messages,
                maximum_content_blocks=limits.max_content_blocks,
            )
        elif value_type is McpCompletionResult:
            value = sanitize_completion_result(
                raw.value,
                sensitive_values=binding.sensitive_values,
                maximum_values=limits.max_completion_values,
            )
        else:
            value = redact_public_dataclass(
                raw.value, sensitive_values=binding.sensitive_values
            )
        result: McpOperationResult[Any] = McpComplete(value=value)
    elif isinstance(raw, McpInputRequired):
        result = _sanitize_input_required(raw, binding.sensitive_values)
    elif isinstance(raw, McpRemoteTask):
        result = _sanitize_remote_task(raw, binding.sensitive_values)
    else:
        raise ValidationError(f"MCP {surface} provider returned an invalid operation result")
    bounded_public_size(
        result,
        maximum=binding.manifest.max_response_bytes,
        label=f"MCP {surface} public result",
    )
    return result


@dataclass(frozen=True)
class McpCatalogCollectionLimits:
    """Independent bounds for one governed, Host-only full catalog probe."""

    max_pages_per_catalog: int = 16
    max_tools: int = 100
    max_resources: int = 200
    max_resource_templates: int = 200
    max_prompts: int = 200
    max_cursor_bytes: int = 4096
    max_identifier_bytes: int = 8192
    max_cache_ttl_ms: int = 3_600_000
    max_public_bytes: int = 8 * 1024 * 1024

    def validate(self) -> None:
        for name in (
            "max_pages_per_catalog",
            "max_tools",
            "max_resources",
            "max_resource_templates",
            "max_prompts",
            "max_cursor_bytes",
            "max_identifier_bytes",
            "max_cache_ttl_ms",
            "max_public_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValidationError(f"MCP catalog {name} must be a positive integer")


@dataclass(frozen=True)
class McpCatalogTool:
    """Detached public Tool description from an unregistered live catalog."""

    name: str
    title: str | None
    description: str | None
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None
    annotations: dict[str, JsonValue]
    metadata: dict[str, JsonValue]
    execution: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class McpCollectedCatalog:
    """Complete, bounded catalogs collected over an already governed session."""

    tools: tuple[McpCatalogTool, ...]
    resources: tuple[McpResource, ...]
    resource_templates: tuple[McpResourceTemplate, ...]
    prompts: tuple[McpPrompt, ...]
    tool_pages: int
    resource_pages: int
    resource_template_pages: int
    prompt_pages: int


async def collect_catalog(
    session: Any,
    limits: McpCatalogCollectionLimits,
    deadline: float,
    *,
    sensitive_values: Iterable[str] = (),
    monotonic: Callable[[], float] = time.monotonic,
) -> McpCollectedCatalog:
    """Collect all four modern discovery catalogs without opening transport.

    ``session`` must have been created by the Runtime's protected, governed
    provider phase.  This helper neither creates a connection nor mutates the
    registry.  Raw cursors remain local to this call and are never projected.
    """

    if not isinstance(limits, McpCatalogCollectionLimits):
        raise ValidationError("MCP catalog limits are invalid")
    limits.validate()
    if type(deadline) not in {int, float}:
        raise ValidationError("MCP catalog deadline is invalid")
    selected_secrets = tuple(sensitive_values)
    if any(type(value) is not str or not value for value in selected_secrets):
        raise ValidationError("MCP catalog sensitive values are invalid")
    selected = _exact_modern_session(session)

    tools, tool_pages = await _collect_catalog_pages(
        selected,
        method_name="list_tools",
        item_fields=("tools",),
        surface="tools/list",
        item_limit=limits.max_tools,
        projector=lambda item: _sdk_catalog_tool(item, selected_secrets, limits),
        identity=lambda item: item.name,
        limits=limits,
        deadline=float(deadline),
        monotonic=monotonic,
    )
    resources, resource_pages = await _collect_catalog_pages(
        selected,
        method_name="list_resources",
        item_fields=("resources",),
        surface="resources/list",
        item_limit=limits.max_resources,
        projector=lambda item: _catalog_resource(item, selected_secrets, limits),
        identity=lambda item: item.resource_id,
        limits=limits,
        deadline=float(deadline),
        monotonic=monotonic,
    )
    templates, template_pages = await _collect_catalog_pages(
        selected,
        method_name="list_resource_templates",
        item_fields=("resource_templates", "resourceTemplates"),
        surface="resources/templates/list",
        item_limit=limits.max_resource_templates,
        projector=lambda item: _catalog_resource_template(
            item, selected_secrets, limits
        ),
        identity=lambda item: item.template_id,
        limits=limits,
        deadline=float(deadline),
        monotonic=monotonic,
    )
    prompts, prompt_pages = await _collect_catalog_pages(
        selected,
        method_name="list_prompts",
        item_fields=("prompts",),
        surface="prompts/list",
        item_limit=limits.max_prompts,
        projector=lambda item: _catalog_prompt(item, selected_secrets, limits),
        identity=lambda item: item.prompt_id,
        limits=limits,
        deadline=float(deadline),
        monotonic=monotonic,
    )
    result = McpCollectedCatalog(
        tools=cast(tuple[McpCatalogTool, ...], tools),
        resources=cast(tuple[McpResource, ...], resources),
        resource_templates=cast(tuple[McpResourceTemplate, ...], templates),
        prompts=cast(tuple[McpPrompt, ...], prompts),
        tool_pages=tool_pages,
        resource_pages=resource_pages,
        resource_template_pages=template_pages,
        prompt_pages=prompt_pages,
    )
    _check_catalog_deadline(float(deadline), monotonic, "public projection")
    bounded_public_size(
        result,
        maximum=limits.max_public_bytes,
        label="MCP full catalog",
    )
    _check_catalog_deadline(float(deadline), monotonic, "result release")
    return result


@dataclass(frozen=True)
class _CursorState:
    server_id: str
    surface: str
    cursor: str
    fence: tuple[str, int, int, str | None, str | None, str | None]
    seen_sha256: tuple[str, ...]
    expires_at: float


class _CursorVault:
    """Bounded in-memory indirection for provider cursors.

    Cursors can be bearer-like and provider-controlled.  Raw values are never
    returned to a caller or stored durably; restart intentionally invalidates
    them and requires a full refresh.
    """

    def __init__(self, limits: McpModernClientLimits) -> None:
        self._limits = limits
        self._states: OrderedDict[str, _CursorState] = OrderedDict()
        self._lock = threading.Lock()

    def put(
        self,
        *,
        server_id: str,
        surface: str,
        cursor: str,
        fence: tuple[str, int, int, str | None, str | None, str | None],
        seen_sha256: tuple[str, ...],
    ) -> str:
        _validate_raw_cursor(cursor, self._limits)
        digest = hashlib.sha256(cursor.encode("utf-8")).hexdigest()
        if digest in seen_sha256:
            raise ValidationError("MCP provider repeated a pagination cursor")
        token = f"mcpcur_{secrets.token_urlsafe(24)}"
        state = _CursorState(
            server_id=server_id,
            surface=surface,
            cursor=cursor,
            fence=fence,
            seen_sha256=(*seen_sha256, digest),
            expires_at=time.monotonic() + self._limits.cursor_ttl_s,
        )
        with self._lock:
            self._prune_locked()
            self._states[token] = state
            while len(self._states) > self._limits.max_cursor_handles:
                self._states.popitem(last=False)
        return token

    def take(
        self,
        token: str | None,
        *,
        server_id: str,
        surface: str,
        fence: tuple[str, int, int, str | None, str | None, str | None],
    ) -> tuple[str | None, tuple[str, ...]]:
        if token is None:
            return None, ()
        if type(token) is not str or not token.startswith("mcpcur_"):
            raise ValidationError("MCP pagination requires an opaque client cursor")
        with self._lock:
            self._prune_locked()
            state = self._states.pop(token, None)
        if state is None:
            raise ValidationError("MCP pagination cursor is expired or unknown")
        if (
            state.server_id != server_id
            or state.surface != surface
            or state.fence != fence
        ):
            raise ValidationError("MCP pagination cursor is bound to another operation")
        return state.cursor, state.seen_sha256

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = [key for key, state in self._states.items() if state.expires_at <= now]
        for key in expired:
            self._states.pop(key, None)

    def invalidate_server(self, server_id: str) -> None:
        with self._lock:
            selected = [
                key
                for key, state in self._states.items()
                if state.server_id == server_id
            ]
            for key in selected:
                self._states.pop(key, None)


class McpSdkInputRequiredHandler(Protocol):
    def capture_input_required(
        self,
        *,
        server_id: str,
        operation: str,
        logical_id: str,
        request_state: str | None,
        input_requests: Mapping[str, Any],
        deadline: float,
        sensitive_values: tuple[str, ...],
    ) -> McpInputRequired: ...


class McpSdkRemoteTaskHandler(Protocol):
    def capture_remote_task(
        self,
        *,
        server_id: str,
        operation: str,
        logical_id: str,
        result: Any,
        deadline: float,
        sensitive_values: tuple[str, ...],
    ) -> McpRemoteTask: ...


class McpSdkV2ResultAdapter:
    """Convert real Python SDK v2 models to Agent libOS public contracts."""

    def __init__(
        self,
        *,
        artifact_writer: McpArtifactWriter | None = None,
        input_required_handler: McpSdkInputRequiredHandler | None = None,
        remote_task_handler: McpSdkRemoteTaskHandler | None = None,
        max_cache_ttl_ms: int = 300_000,
        max_content_blocks: int = 256,
        max_prompt_messages: int = 128,
        max_completion_values: int = 100,
    ) -> None:
        self.artifact_writer = artifact_writer
        self.input_required_handler = input_required_handler
        self.remote_task_handler = remote_task_handler
        if type(max_cache_ttl_ms) is not int or max_cache_ttl_ms <= 0:
            raise ValidationError(
                "MCP adapter max_cache_ttl_ms must be a positive integer"
            )
        self.max_cache_ttl_ms = max_cache_ttl_ms
        for name, value in (
            ("max_content_blocks", max_content_blocks),
            ("max_prompt_messages", max_prompt_messages),
            ("max_completion_values", max_completion_values),
        ):
            if type(value) is not int or value <= 0:
                raise ValidationError(f"MCP adapter {name} must be a positive integer")
        self.max_content_blocks = max_content_blocks
        self.max_prompt_messages = max_prompt_messages
        self.max_completion_values = max_completion_values

    def resource_page(
        self, result: Any, *, sensitive_values: tuple[str, ...] = ()
    ) -> McpPage[McpResource]:
        items = getattr(result, "resources", None)
        if type(items) is not list:
            raise ValidationError("MCP resources/list result is malformed")
        return McpPage(
            items=tuple(
                sdk_resource(item, sensitive_values=sensitive_values) for item in items
            ),
            next_cursor=_sdk_next_cursor(result),
            cache_hint=cache_hint_from_sdk(
                result, maximum_ttl_ms=self.max_cache_ttl_ms
            ),
        )

    def resource_template_page(
        self, result: Any, *, sensitive_values: tuple[str, ...] = ()
    ) -> McpPage[McpResourceTemplate]:
        items = getattr(
            result,
            "resource_templates",
            getattr(result, "resourceTemplates", None),
        )
        if type(items) is not list:
            raise ValidationError("MCP resources/templates/list result is malformed")
        return McpPage(
            items=tuple(
                sdk_resource_template(item, sensitive_values=sensitive_values)
                for item in items
            ),
            next_cursor=_sdk_next_cursor(result),
            cache_hint=cache_hint_from_sdk(
                result, maximum_ttl_ms=self.max_cache_ttl_ms
            ),
        )

    def prompt_page(
        self, result: Any, *, sensitive_values: tuple[str, ...] = ()
    ) -> McpPage[McpPrompt]:
        items = getattr(result, "prompts", None)
        if type(items) is not list:
            raise ValidationError("MCP prompts/list result is malformed")
        return McpPage(
            items=tuple(sdk_prompt(item, sensitive_values=sensitive_values) for item in items),
            next_cursor=_sdk_next_cursor(result),
            cache_hint=cache_hint_from_sdk(
                result, maximum_ttl_ms=self.max_cache_ttl_ms
            ),
        )

    def read_resource_result(
        self,
        result: Any,
        *,
        server_id: str,
        logical_id: str,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpOperationResult[McpResourceContents]:
        special = self._special_result(
            result,
            server_id=server_id,
            operation="resources/read",
            logical_id=logical_id,
            deadline=deadline,
            sensitive_values=sensitive_values,
        )
        if special is not None:
            return special
        return McpComplete(
            value=sdk_resource_contents(
                result,
                server_id=server_id,
                logical_id=logical_id,
                artifact_writer=self.artifact_writer,
                sensitive_values=sensitive_values,
                maximum_content_blocks=self.max_content_blocks,
            )
        )

    def prompt_result(
        self,
        result: Any,
        *,
        server_id: str,
        logical_id: str,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpOperationResult[McpPromptResult]:
        special = self._special_result(
            result,
            server_id=server_id,
            operation="prompts/get",
            logical_id=logical_id,
            deadline=deadline,
            sensitive_values=sensitive_values,
        )
        if special is not None:
            return special
        return McpComplete(
            value=sdk_prompt_result(
                result,
                server_id=server_id,
                logical_id=logical_id,
                artifact_writer=self.artifact_writer,
                sensitive_values=sensitive_values,
                maximum_messages=self.max_prompt_messages,
                maximum_content_blocks=self.max_content_blocks,
            )
        )

    def completion_result(
        self,
        result: Any,
        *,
        server_id: str,
        logical_id: str,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpOperationResult[McpCompletionResult]:
        special = self._special_result(
            result,
            server_id=server_id,
            operation="completion/complete",
            logical_id=logical_id,
            deadline=deadline,
            sensitive_values=sensitive_values,
        )
        if special is not None:
            return special
        return McpComplete(
            value=sdk_completion_result(
                result,
                sensitive_values=sensitive_values,
                maximum_values=self.max_completion_values,
            )
        )

    def tool_result(
        self,
        result: Any,
        *,
        server_id: str,
        logical_id: str,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpOperationResult[dict[str, JsonValue]]:
        """Adapt one exact-modern ``tools/call`` polymorphic result.

        MRTR and Tasks are captured by their Host handlers before any opaque
        state or remote identifier can reach the public result.  Ordinary
        results remain untrusted JSON, recursively secret-redacted and with
        MCP Apps selectors/metadata excluded by the shared sanitizer.
        """

        special = self._special_result(
            result,
            server_id=server_id,
            operation="tools/call",
            logical_id=logical_id,
            deadline=deadline,
            sensitive_values=sensitive_values,
        )
        if special is not None:
            return special
        raw = sdk_json_mapping(result, label="MCP SDK tools/call result")
        if raw.get("resultType", "complete") != "complete":
            raise ValidationError("MCP tools/call resultType is unsupported")
        allowed = {"resultType", "content", "structuredContent", "isError", "_meta"}
        if set(raw) - allowed:
            raise ValidationError("MCP tools/call result contains unsupported fields")
        content = raw.get("content", [])
        if type(content) is not list:
            raise ValidationError("MCP tools/call content must be a list")
        if len(content) > self.max_content_blocks:
            raise ValidationError("MCP tools/call exceeded maximum content block count")
        selected = sanitize_provider_json(
            {key: value for key, value in raw.items() if key != "resultType"},
            sensitive_values=sensitive_values,
        )
        if type(selected) is not dict:  # pragma: no cover - object projection invariant
            raise ValidationError("MCP tools/call result is invalid")
        return McpComplete(value=selected)

    def _special_result(
        self,
        result: Any,
        *,
        server_id: str,
        operation: str,
        logical_id: str,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpInputRequired | McpRemoteTask | None:
        if isinstance(result, Mapping):
            result_type = result.get("resultType", "complete")
        else:
            result_type = getattr(
                result,
                "result_type",
                getattr(result, "resultType", "complete"),
            )
        if type(result_type) is not str:
            raise ValidationError("MCP resultType is invalid")
        if result_type == "complete":
            return None
        sensitive_values = _active_operation_sensitive_values(sensitive_values)
        if result_type == "input_required":
            if operation not in _INPUT_REQUIRED_CONTINUATION_METHODS:
                raise McpContinuationSurfaceUnsupported(
                    f"MCP {operation} does not support durable input-required continuation"
                )
            if self.input_required_handler is None:
                raise ValidationError(
                    "MCP input-required result needs a Host continuation handler"
                )
            requests = (
                result.get("inputRequests")
                if isinstance(result, Mapping)
                else getattr(
                    result, "input_requests", getattr(result, "inputRequests", None)
                )
            ) or {}
            if not isinstance(requests, Mapping):
                raise ValidationError("MCP inputRequests must be an object")
            return self.input_required_handler.capture_input_required(
                server_id=server_id,
                operation=operation,
                logical_id=logical_id,
                request_state=(
                    result.get("requestState")
                    if isinstance(result, Mapping)
                    else getattr(
                        result, "request_state", getattr(result, "requestState", None)
                    )
                ),
                input_requests=requests,
                deadline=deadline,
                sensitive_values=sensitive_values,
            )
        if result_type == "task":
            if self.remote_task_handler is None:
                raise ValidationError("MCP remote task needs a Host Tasks handler")
            return self.remote_task_handler.capture_remote_task(
                server_id=server_id,
                operation=operation,
                logical_id=logical_id,
                result=result,
                deadline=deadline,
                sensitive_values=sensitive_values,
            )
        raise ValidationError("unsupported MCP resultType")


class McpSdkV2SessionFactory(Protocol):
    """Runtime-owned supervisor that yields an already governed SDK session."""

    def __call__(
        self, server: McpServerSpec, *, deadline: float
    ) -> AsyncContextManager[Any]: ...


class McpSdkV2SessionProvider(McpResourceProvider, McpPromptProvider):
    """Resources/Prompts provider over a supervisor-owned Python SDK v2 session.

    The factory, rather than this adapter, owns DNS, credentials, subprocess
    snapshots, lifecycle fences, and transport limits.  This prevents a second
    less-governed transport implementation from bypassing the existing MCP
    primitive boundary.
    """

    mcp_manifest_schema_version: Literal[3] = 3
    mcp_protocol_revision: Literal["2026-07-28"] = MCP_V3_PROTOCOL_REVISION

    def __init__(
        self,
        session_factory: McpSdkV2SessionFactory,
        *,
        result_adapter: McpSdkV2ResultAdapter | None = None,
        sensitive_values_resolver: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.result_adapter = result_adapter or McpSdkV2ResultAdapter()
        self.sensitive_values_resolver = sensitive_values_resolver or (lambda _id: ())

    async def list_resources(
        self, server: McpServerSpec, cursor: str | None, *, deadline: float
    ) -> McpPage[McpResource]:
        async with self.session_factory(server, deadline=deadline) as selected:
            session = _exact_modern_session(selected)
            result = await session.list_resources(params=_sdk_page_params(cursor))
            return self.result_adapter.resource_page(
                result,
                sensitive_values=self.sensitive_values_resolver(server.server_id),
            )

    async def list_resource_templates(
        self, server: McpServerSpec, cursor: str | None, *, deadline: float
    ) -> McpPage[McpResourceTemplate]:
        async with self.session_factory(server, deadline=deadline) as selected:
            session = _exact_modern_session(selected)
            result = await session.list_resource_templates(params=_sdk_page_params(cursor))
            return self.result_adapter.resource_template_page(
                result,
                sensitive_values=self.sensitive_values_resolver(server.server_id),
            )

    async def read_resource(
        self,
        server: McpServerSpec,
        resource_name: str,
        variables: Mapping[str, str] | None,
        *,
        deadline: float,
    ) -> McpOperationResult[McpResourceContents]:
        if variables:
            raise ValidationError("MCP SDK provider requires an expanded resource selector")
        async with self.session_factory(server, deadline=deadline) as selected:
            session = _exact_modern_session(selected)
            result = await session.read_resource(
                resource_name, allow_input_required=True
            )
            return self.result_adapter.read_resource_result(
                result,
                server_id=server.server_id,
                logical_id=_active_client_logical_id(resource_name),
                deadline=deadline,
                sensitive_values=self.sensitive_values_resolver(server.server_id),
            )

    async def list_prompts(
        self, server: McpServerSpec, cursor: str | None, *, deadline: float
    ) -> McpPage[McpPrompt]:
        async with self.session_factory(server, deadline=deadline) as selected:
            session = _exact_modern_session(selected)
            result = await session.list_prompts(params=_sdk_page_params(cursor))
            return self.result_adapter.prompt_page(
                result,
                sensitive_values=self.sensitive_values_resolver(server.server_id),
            )

    async def get_prompt(
        self,
        server: McpServerSpec,
        prompt_name: str,
        arguments: Mapping[str, str],
        *,
        deadline: float,
    ) -> McpOperationResult[McpPromptResult]:
        async with self.session_factory(server, deadline=deadline) as selected:
            session = _exact_modern_session(selected)
            result = await session.get_prompt(
                prompt_name,
                dict(arguments),
                allow_input_required=True,
            )
            return self.result_adapter.prompt_result(
                result,
                server_id=server.server_id,
                logical_id=_active_client_logical_id(prompt_name),
                deadline=deadline,
                sensitive_values=self.sensitive_values_resolver(server.server_id),
            )

    async def complete(
        self,
        server: McpServerSpec,
        reference: Mapping[str, JsonValue],
        argument: Mapping[str, str],
        context: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
    ) -> McpOperationResult[McpCompletionResult]:
        try:
            import mcp.types as mcp_types
        except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
            raise ValidationError("MCP Python SDK v2 is unavailable") from exc
        ref_type = reference.get("type")
        if ref_type == "ref/prompt":
            ref = mcp_types.PromptReference(name=str(reference.get("name", "")))
        elif ref_type == "ref/resource":
            ref = mcp_types.ResourceTemplateReference(
                uri=str(reference.get("uri", ""))
            )
        else:
            raise ValidationError("MCP completion reference is invalid")
        context_arguments = None
        if context is not None:
            if any(type(key) is not str or type(value) is not str for key, value in context.items()):
                raise ValidationError("MCP completion context must contain string values")
            context_arguments = cast(dict[str, str], dict(context))
        async with self.session_factory(server, deadline=deadline) as selected:
            session = _exact_modern_session(selected)
            result = await session.complete(
                ref,
                dict(argument),
                context_arguments=context_arguments,
            )
            logical_id = str(reference.get("name", reference.get("uri", "")))
            return self.result_adapter.completion_result(
                result,
                server_id=server.server_id,
                logical_id=_active_client_logical_id(logical_id),
                deadline=deadline,
                sensitive_values=self.sensitive_values_resolver(server.server_id),
            )


class McpModernClient:
    """Host-facing Manifest-v3 Resources/Prompts/Completion manager."""

    def __init__(
        self,
        binding_resolver: McpClientBindingResolver,
        *,
        resource_provider: McpResourceProvider | None = None,
        prompt_provider: McpPromptProvider | None = None,
        limits: McpModernClientLimits | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.binding_resolver = binding_resolver
        self.resource_provider = resource_provider
        self.prompt_provider = prompt_provider
        self.limits = limits or McpModernClientLimits()
        self.limits.validate()
        self.monotonic = monotonic
        self._cursors = _CursorVault(self.limits)

    def invalidate_server(self, server_id: str) -> None:
        """Synchronously revoke all opaque pagination handles for a server."""

        _validate_id(server_id, "server_id")
        self._cursors.invalidate_server(server_id)

    # Sync facade for Runtime/CLI/GUI Host callers.
    def list_resources(self, server_id: str, cursor: str | None = None, **kwargs: Any) -> McpPage[McpResource]:
        return _run_sync(self.alist_resources(server_id, cursor=cursor, **kwargs))

    def list_resource_templates(self, server_id: str, cursor: str | None = None, **kwargs: Any) -> McpPage[McpResourceTemplate]:
        return _run_sync(self.alist_resource_templates(server_id, cursor=cursor, **kwargs))

    def read_resource(self, server_id: str, resource_id: str, variables: Mapping[str, str] | None = None, **kwargs: Any) -> McpOperationResult[McpResourceContents]:
        return _run_sync(self.aread_resource(server_id, resource_id, variables=variables, **kwargs))

    def list_prompts(self, server_id: str, cursor: str | None = None, **kwargs: Any) -> McpPage[McpPrompt]:
        return _run_sync(self.alist_prompts(server_id, cursor=cursor, **kwargs))

    def get_prompt(self, server_id: str, prompt_id: str, arguments: Mapping[str, str] | None = None, **kwargs: Any) -> McpOperationResult[McpPromptResult]:
        return _run_sync(self.aget_prompt(server_id, prompt_id, arguments=arguments, **kwargs))

    def complete_prompt(
        self,
        server_id: str,
        reference_type: Literal["prompt", "resource_template"],
        reference_id: str,
        argument: Mapping[str, str],
        context: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> McpOperationResult[McpCompletionResult]:
        return _run_sync(
            self.acomplete_prompt(
                server_id,
                reference_type,
                reference_id,
                argument,
                context=context,
                **kwargs,
            )
        )

    # Compatibility spelling for direct provider-style callers.  Product
    # surfaces use complete_prompt to keep the Runtime API unambiguous.
    def complete(
        self,
        server_id: str,
        reference_type: Literal["prompt", "resource_template"],
        reference_id: str,
        argument: Mapping[str, str],
        context: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> McpOperationResult[McpCompletionResult]:
        return self.complete_prompt(
            server_id,
            reference_type,
            reference_id,
            argument,
            context=context,
            **kwargs,
        )

    async def alist_resources(
        self,
        server_id: str,
        cursor: str | None = None,
        *,
        deadline: float | None = None,
        owner_id: str | None = None,
        model_visible_only: bool = False,
    ) -> McpPage[McpResource]:
        binding, selected_deadline = self._begin(server_id, deadline, owner_id)
        provider = self._require_resource_provider()
        raw_cursor, seen = self._cursors.take(
            cursor,
            server_id=server_id,
            surface="resources/list",
            fence=binding.fence,
        )
        raw = await self._invoke(
            provider.list_resources,
            mcp_transport_spec_from_v3(binding.manifest),
            raw_cursor,
            deadline=selected_deadline,
            sensitive_values=binding.sensitive_values,
            binding=binding,
        )
        if not isinstance(raw, McpPage):
            raise ValidationError("MCP Resource provider returned an invalid page")
        self._bound_raw(raw, binding, "MCP resources/list result")
        allowed = {
            item.remote_uri: item
            for item in binding.manifest.resources
            if not model_visible_only or item.model_visible
        }
        items: list[McpResource] = []
        emitted: set[str] = set()
        for item in raw.items:
            self._check_deadline(selected_deadline, "resources/list projection")
            if not isinstance(item, McpResource):
                raise ValidationError("MCP Resource provider returned an invalid item")
            spec = allowed.get(item.resource_id)
            if spec is None or spec.resource_id in emitted:
                continue
            if is_mcp_app_mime(item.mime_type):
                raise ValidationError("MCP Apps HTML resources are unsupported")
            emitted.add(spec.resource_id)
            items.append(
                replace(
                    cast(McpResource, redact_public_dataclass(
                        item, sensitive_values=binding.sensitive_values
                    )),
                    resource_id=spec.resource_id,
                    icons=(),
                )
            )
        page = self._finish_page(
            items,
            raw,
            binding=binding,
            surface="resources/list",
            seen=seen,
        )
        self._finish(binding, selected_deadline)
        return page

    async def alist_resource_templates(
        self,
        server_id: str,
        cursor: str | None = None,
        *,
        deadline: float | None = None,
        owner_id: str | None = None,
        model_visible_only: bool = False,
    ) -> McpPage[McpResourceTemplate]:
        binding, selected_deadline = self._begin(server_id, deadline, owner_id)
        provider = self._require_resource_provider()
        raw_cursor, seen = self._cursors.take(
            cursor,
            server_id=server_id,
            surface="resources/templates/list",
            fence=binding.fence,
        )
        raw = await self._invoke(
            provider.list_resource_templates,
            mcp_transport_spec_from_v3(binding.manifest),
            raw_cursor,
            deadline=selected_deadline,
            sensitive_values=binding.sensitive_values,
            binding=binding,
        )
        if not isinstance(raw, McpPage):
            raise ValidationError("MCP Resource provider returned an invalid template page")
        self._bound_raw(raw, binding, "MCP resources/templates/list result")
        allowed = {
            item.remote_uri_template: item
            for item in binding.manifest.resource_templates
            if not model_visible_only or item.model_visible
        }
        items: list[McpResourceTemplate] = []
        emitted: set[str] = set()
        for item in raw.items:
            self._check_deadline(selected_deadline, "resource template projection")
            if not isinstance(item, McpResourceTemplate):
                raise ValidationError("MCP Resource provider returned an invalid template")
            spec = allowed.get(item.template_id)
            if spec is None or spec.template_id in emitted:
                continue
            if is_mcp_app_mime(item.mime_type):
                raise ValidationError("MCP Apps HTML resource templates are unsupported")
            emitted.add(spec.template_id)
            items.append(
                replace(
                    cast(McpResourceTemplate, redact_public_dataclass(
                        item, sensitive_values=binding.sensitive_values
                    )),
                    template_id=spec.template_id,
                    icons=(),
                )
            )
        page = self._finish_page(
            items,
            raw,
            binding=binding,
            surface="resources/templates/list",
            seen=seen,
        )
        self._finish(binding, selected_deadline)
        return page

    async def aread_resource(
        self,
        server_id: str,
        resource_id: str,
        variables: Mapping[str, str] | None = None,
        *,
        deadline: float | None = None,
        owner_id: str | None = None,
        for_model: bool = False,
    ) -> McpOperationResult[McpResourceContents]:
        binding, selected_deadline = self._begin(server_id, deadline, owner_id)
        spec, selector = self._resource_selector(
            binding.manifest, resource_id, variables, for_model=for_model
        )
        provider = self._require_resource_provider()
        raw = await self._invoke(
            provider.read_resource,
            mcp_transport_spec_from_v3(binding.manifest),
            selector,
            None,
            deadline=selected_deadline,
            sensitive_values=binding.sensitive_values,
            binding=binding,
            logical_id=resource_id,
        )
        self._bound_raw(raw, binding, "MCP resources/read result")
        result = self._sanitize_operation_result(
            raw,
            binding=binding,
            logical_id=spec.resource_id if isinstance(spec, McpResourceSpec) else spec.template_id,
            value_type=McpResourceContents,
            surface="resource",
        )
        self._finish(binding, selected_deadline)
        return cast(McpOperationResult[McpResourceContents], result)

    async def alist_prompts(
        self,
        server_id: str,
        cursor: str | None = None,
        *,
        deadline: float | None = None,
        owner_id: str | None = None,
    ) -> McpPage[McpPrompt]:
        binding, selected_deadline = self._begin(server_id, deadline, owner_id)
        provider = self._require_prompt_provider()
        raw_cursor, seen = self._cursors.take(
            cursor,
            server_id=server_id,
            surface="prompts/list",
            fence=binding.fence,
        )
        raw = await self._invoke(
            provider.list_prompts,
            mcp_transport_spec_from_v3(binding.manifest),
            raw_cursor,
            deadline=selected_deadline,
            sensitive_values=binding.sensitive_values,
            binding=binding,
        )
        if not isinstance(raw, McpPage):
            raise ValidationError("MCP Prompt provider returned an invalid page")
        self._bound_raw(raw, binding, "MCP prompts/list result")
        allowed = {item.mcp_name: item for item in binding.manifest.prompts}
        items: list[McpPrompt] = []
        emitted: set[str] = set()
        for item in raw.items:
            self._check_deadline(selected_deadline, "prompts/list projection")
            if not isinstance(item, McpPrompt):
                raise ValidationError("MCP Prompt provider returned an invalid item")
            spec = allowed.get(item.prompt_id)
            if spec is None or spec.prompt_id in emitted:
                continue
            emitted.add(spec.prompt_id)
            sanitized = cast(
                McpPrompt,
                redact_public_dataclass(item, sensitive_values=binding.sensitive_values),
            )
            allowed_arguments = set(spec.argument_names)
            projected_arguments = tuple(
                projected
                for raw_argument, projected in zip(item.arguments, sanitized.arguments)
                if raw_argument.name in allowed_arguments
            )
            items.append(
                replace(
                    sanitized,
                    prompt_id=spec.prompt_id,
                    arguments=projected_arguments,
                    icons=(),
                )
            )
        page = self._finish_page(
            items,
            raw,
            binding=binding,
            surface="prompts/list",
            seen=seen,
        )
        self._finish(binding, selected_deadline)
        return page

    async def aget_prompt(
        self,
        server_id: str,
        prompt_id: str,
        arguments: Mapping[str, str] | None = None,
        *,
        deadline: float | None = None,
        owner_id: str | None = None,
    ) -> McpOperationResult[McpPromptResult]:
        binding, selected_deadline = self._begin(server_id, deadline, owner_id)
        spec = _prompt_by_id(binding.manifest, prompt_id)
        selected_arguments = self._arguments(
            arguments or {}, allowed=frozenset(spec.argument_names), label="prompt"
        )
        provider = self._require_prompt_provider()
        raw = await self._invoke(
            provider.get_prompt,
            mcp_transport_spec_from_v3(binding.manifest),
            spec.mcp_name,
            selected_arguments,
            deadline=selected_deadline,
            sensitive_values=binding.sensitive_values,
            binding=binding,
            logical_id=prompt_id,
        )
        self._bound_raw(raw, binding, "MCP prompts/get result")
        result = self._sanitize_operation_result(
            raw,
            binding=binding,
            logical_id=spec.prompt_id,
            value_type=McpPromptResult,
            surface="prompt",
        )
        if isinstance(result, McpComplete):
            if not isinstance(result.value, McpPromptResult):  # defensive
                raise ValidationError("MCP Prompt Complete result is invalid")
            result = replace(
                result,
                preview_sha256=mcp_prompt_preview_sha256(
                    binding=binding,
                    prompt_id=spec.prompt_id,
                    arguments=selected_arguments,
                    prompt=result.value,
                ),
            )
        self._finish(binding, selected_deadline)
        return cast(McpOperationResult[McpPromptResult], result)

    async def acomplete_prompt(
        self,
        server_id: str,
        reference_type: Literal["prompt", "resource_template"],
        reference_id: str,
        argument: Mapping[str, str],
        context: Mapping[str, str] | None = None,
        *,
        deadline: float | None = None,
        owner_id: str | None = None,
    ) -> McpOperationResult[McpCompletionResult]:
        binding, selected_deadline = self._begin(server_id, deadline, owner_id)
        if reference_type == "prompt":
            selected = _prompt_by_id(binding.manifest, reference_id)
            reference: dict[str, JsonValue] = {
                "type": "ref/prompt",
                "name": selected.mcp_name,
            }
            allowed = frozenset(selected.argument_names)
        elif reference_type == "resource_template":
            selected_template = _template_by_id(binding.manifest, reference_id)
            reference = {
                "type": "ref/resource",
                "uri": selected_template.remote_uri_template,
            }
            allowed = frozenset(selected_template.variables)
        else:
            raise ValidationError("MCP completion reference_type is invalid")
        selected_argument = self._completion_argument(argument, allowed=allowed)
        selected_context = (
            None
            if context is None
            else self._arguments(
                context,
                allowed=allowed,
                label="completion context",
            )
        )
        provider = self._require_prompt_provider()
        raw = await self._invoke(
            provider.complete,
            mcp_transport_spec_from_v3(binding.manifest),
            reference,
            selected_argument,
            selected_context,
            deadline=selected_deadline,
            sensitive_values=binding.sensitive_values,
            binding=binding,
            logical_id=reference_id,
        )
        self._bound_raw(raw, binding, "MCP completion result")
        result = self._sanitize_operation_result(
            raw,
            binding=binding,
            logical_id=reference_id,
            value_type=McpCompletionResult,
            surface="completion",
        )
        if isinstance(result, McpComplete) and result.value is not None:
            if len(result.value.values) > self.limits.max_completion_values:
                raise ValidationError("MCP completion exceeded maximum value count")
        self._finish(binding, selected_deadline)
        return cast(McpOperationResult[McpCompletionResult], result)

    async def acomplete(
        self,
        server_id: str,
        reference_type: Literal["prompt", "resource_template"],
        reference_id: str,
        argument: Mapping[str, str],
        context: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> McpOperationResult[McpCompletionResult]:
        return await self.acomplete_prompt(
            server_id,
            reference_type,
            reference_id,
            argument,
            context=context,
            **kwargs,
        )

    def _begin(
        self, server_id: str, deadline: float | None, owner_id: str | None
    ) -> tuple[McpClientBinding, float]:
        _validate_id(server_id, "server_id")
        binding = self._resolve_binding(server_id, owner_id=owner_id)
        if not isinstance(binding, McpClientBinding):
            raise ValidationError("MCP binding resolver returned an invalid binding")
        validate_mcp_v3_manifest(binding.manifest)
        if binding.manifest.server_id != server_id:
            raise ValidationError("MCP binding returned another server")
        _validate_binding(binding, owner_id)
        selected_deadline = (
            self.monotonic() + binding.manifest.timeout_s
            if deadline is None
            else deadline
        )
        self._check_deadline(selected_deadline, "binding resolution")
        _validate_manifest_remote_maps(binding.manifest)
        return binding, selected_deadline

    def _finish(self, binding: McpClientBinding, deadline: float) -> None:
        self._check_deadline(deadline, "post-provider fence")
        current = self._resolve_binding(
            binding.manifest.server_id, owner_id=binding.owner_id
        )
        if not isinstance(current, McpClientBinding) or current.fence != binding.fence:
            raise ValidationError("MCP registry or authentication fence changed during operation")
        self._check_deadline(deadline, "result release")

    def _resolve_binding(
        self,
        server_id: str,
        *,
        owner_id: str | None,
    ) -> McpClientBinding:
        resolver = getattr(self.binding_resolver, "resolve", None)
        if callable(resolver):
            return resolver(server_id, owner_id=owner_id)
        return self.binding_resolver(server_id)

    async def _invoke(
        self,
        method: Any,
        *args: Any,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
        binding: McpClientBinding | None = None,
        logical_id: str | None = None,
    ) -> Any:
        self._check_deadline(deadline, "provider dispatch")
        binding_token = _ACTIVE_CLIENT_BINDING.set(binding)
        logical_id_token = _ACTIVE_CLIENT_LOGICAL_ID.set(logical_id)
        try:
            try:
                result = method(*args, deadline=deadline)
            except McpContinuationSurfaceUnsupported:
                raise
            except ProviderEffectNotStarted:
                raise
            except Exception as exc:
                raise safe_mcp_provider_error(exc, sensitive_values) from None
            if not inspect.isawaitable(result):
                raise ValidationError("MCP modern provider method must be asynchronous")
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                if inspect.iscoroutine(result):
                    result.close()
                raise TimeoutError("MCP absolute deadline exhausted before provider")
            task = asyncio.ensure_future(result)
            try:
                done, _pending = await asyncio.wait({task}, timeout=remaining)
            except asyncio.CancelledError:
                await _cancel_and_drain_provider_task(task)
                raise
            if not done:
                await _cancel_and_drain_provider_task(task)
                raise TimeoutError("MCP provider exceeded the absolute deadline")
            try:
                selected = task.result()
            except asyncio.CancelledError:
                raise
            except McpContinuationSurfaceUnsupported:
                raise
            except ProviderEffectNotStarted:
                raise
            except Exception as exc:
                raise safe_mcp_provider_error(exc, sensitive_values) from None
            self._check_deadline(deadline, "provider completion")
            return selected
        finally:
            _ACTIVE_CLIENT_LOGICAL_ID.reset(logical_id_token)
            _ACTIVE_CLIENT_BINDING.reset(binding_token)

    def _finish_page(
        self,
        items: list[_T],
        raw: McpPage[Any],
        *,
        binding: McpClientBinding,
        surface: str,
        seen: tuple[str, ...],
    ) -> McpPage[_T]:
        page_limit = self._page_limit(surface)
        if len(raw.items) > page_limit:
            raise ValidationError("MCP provider page exceeded maximum item count")
        next_cursor = None
        if raw.next_cursor is not None:
            next_cursor = self._cursors.put(
                server_id=binding.manifest.server_id,
                surface=surface,
                cursor=raw.next_cursor,
                fence=binding.fence,
                seen_sha256=seen,
            )
        hint = _sanitize_cache_hint(raw.cache_hint, self.limits)
        page = McpPage(items=tuple(items), next_cursor=next_cursor, cache_hint=hint)
        bounded_public_size(
            page,
            maximum=binding.manifest.max_response_bytes,
            label=f"MCP {surface} public result",
        )
        return page

    def _page_limit(self, surface: str) -> int:
        if self.limits.max_page_items is not None:
            return self.limits.max_page_items
        limits = {
            "resources/list": self.limits.max_resource_items,
            "resources/templates/list": self.limits.max_resource_template_items,
            "prompts/list": self.limits.max_prompt_items,
        }
        try:
            return limits[surface]
        except KeyError as exc:  # pragma: no cover - internal call-site invariant
            raise ValidationError("MCP page surface is invalid") from exc

    def _sanitize_operation_result(
        self,
        raw: Any,
        *,
        binding: McpClientBinding,
        logical_id: str,
        value_type: type[Any],
        surface: str,
    ) -> McpOperationResult[Any]:
        return sanitize_mcp_operation_result(
            raw,
            binding=binding,
            logical_id=logical_id,
            value_type=value_type,
            surface=surface,
            limits=self.limits,
        )

    def _bound_raw(self, raw: Any, binding: McpClientBinding, label: str) -> None:
        bounded_public_size(raw, maximum=binding.manifest.max_response_bytes, label=label)

    def _resource_selector(
        self,
        manifest: McpServerManifestV3,
        logical_id: str,
        variables: Mapping[str, str] | None,
        *,
        for_model: bool,
    ) -> tuple[McpResourceSpec | McpResourceTemplateSpec, str]:
        _validate_id(logical_id, "resource_id")
        resource = next(
            (item for item in manifest.resources if item.resource_id == logical_id), None
        )
        template = next(
            (item for item in manifest.resource_templates if item.template_id == logical_id),
            None,
        )
        if resource is not None and template is not None:
            raise ValidationError("MCP resource logical id is ambiguous")
        if resource is not None:
            if variables:
                raise ValidationError("MCP concrete Resource does not accept variables")
            if for_model and not resource.model_visible:
                raise NotFound(f"MCP model-visible resource not found: {logical_id}")
            reject_mcp_app_selector(resource.remote_uri)
            return resource, resource.remote_uri
        if template is not None:
            if for_model and not template.model_visible:
                raise NotFound(f"MCP model-visible resource template not found: {logical_id}")
            return template, _expand_template(template, variables, self.limits)
        raise NotFound(f"MCP resource not found: {logical_id}")

    def resolve_resource_selector(
        self,
        manifest: McpServerManifestV3,
        logical_id: str,
        variables: Mapping[str, str] | None,
        *,
        for_model: bool,
    ) -> str:
        """Resolve one registered logical Resource without provider I/O.

        Durable continuation dispatch uses this public composition seam to
        reproduce the exact selector validation and URI-template expansion
        performed by the initial operation.  The current registry fence is
        still checked by the primitive before this pure resolution step.
        """

        _spec, selector = self._resource_selector(
            manifest,
            logical_id,
            variables,
            for_model=for_model,
        )
        return selector

    def _arguments(
        self,
        value: Mapping[str, str],
        *,
        allowed: frozenset[str],
        label: str,
    ) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValidationError(f"MCP {label} arguments must be an object")
        selected: dict[str, str] = {}
        for key, item in value.items():
            if type(key) is not str or type(item) is not str:
                raise ValidationError(f"MCP {label} arguments must contain string values")
            if key not in allowed:
                raise ValidationError(f"MCP {label} argument is not manifest-authorized: {key}")
            if len(key.encode("utf-8")) > self.limits.max_argument_name_bytes:
                raise ValidationError(f"MCP {label} argument name is too large")
            if len(item.encode("utf-8")) > self.limits.max_argument_value_bytes:
                raise ValidationError(f"MCP {label} argument value is too large")
            selected[key] = item
        return selected

    def _completion_argument(
        self,
        value: Mapping[str, str],
        *,
        allowed: frozenset[str],
    ) -> dict[str, str]:
        if not isinstance(value, Mapping) or set(value) != {"name", "value"}:
            raise ValidationError(
                "MCP completion argument requires exactly name and value"
            )
        name = value.get("name")
        selected_value = value.get("value")
        if type(name) is not str or type(selected_value) is not str:
            raise ValidationError("MCP completion name and value must be strings")
        if name not in allowed:
            raise ValidationError(
                f"MCP completion argument is not manifest-authorized: {name}"
            )
        if len(name.encode("utf-8")) > self.limits.max_argument_name_bytes:
            raise ValidationError("MCP completion argument name is too large")
        if len(selected_value.encode("utf-8")) > self.limits.max_argument_value_bytes:
            raise ValidationError("MCP completion argument value is too large")
        return {"name": name, "value": selected_value}

    def _string_mapping(
        self,
        value: Mapping[str, str],
        *,
        label: str,
    ) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValidationError(f"MCP {label} must be an object")
        selected: dict[str, str] = {}
        for key, item in value.items():
            if type(key) is not str or type(item) is not str:
                raise ValidationError(f"MCP {label} must contain string values")
            if len(key.encode("utf-8")) > self.limits.max_argument_name_bytes:
                raise ValidationError(f"MCP {label} name is too large")
            if len(item.encode("utf-8")) > self.limits.max_argument_value_bytes:
                raise ValidationError(f"MCP {label} value is too large")
            selected[key] = item
        return selected

    def _check_deadline(self, deadline: float, stage: str) -> None:
        if type(deadline) not in {int, float} or self.monotonic() >= deadline:
            raise TimeoutError(f"MCP absolute deadline exhausted during {stage}")

    def _require_resource_provider(self) -> McpResourceProvider:
        if self.resource_provider is None:
            raise ValidationError("MCP Resource provider is not configured")
        return self.resource_provider

    def _require_prompt_provider(self) -> McpPromptProvider:
        if self.prompt_provider is None:
            raise ValidationError("MCP Prompt provider is not configured")
        return self.prompt_provider


def mcp_transport_spec_from_v3(manifest: McpServerManifestV3) -> McpServerSpec:
    """Project v3 transport into the released provider transport contract."""

    return McpServerSpec(
        schema_version=2,
        server_id=manifest.server_id,
        transport=manifest.transport,
        tools=list(manifest.tools),
        timeout_s=manifest.timeout_s,
        max_request_bytes=manifest.max_request_bytes,
        max_response_bytes=manifest.max_response_bytes,
        stdio=manifest.stdio,
        http=manifest.http,
        metadata={},
        protocol_mode=manifest.protocol_mode,
    )


def _validate_binding(binding: McpClientBinding, owner_id: str | None) -> None:
    _validate_binding_generations(binding)
    _validate_binding_digests(binding)
    _validate_binding_owner(binding, owner_id)
    _validate_binding_secrets(binding)
    _validate_binding_environment(binding)


def _validate_binding_generations(binding: McpClientBinding) -> None:
    if type(binding.registry_generation) is not int or binding.registry_generation < 0:
        raise ValidationError("MCP registry generation is invalid")
    if type(binding.auth_generation) is not int or binding.auth_generation < 0:
        raise ValidationError("MCP auth generation is invalid")


def _validate_binding_digests(binding: McpClientBinding) -> None:
    if binding.auth_principal_sha256 is not None and not _SHA256_RE.fullmatch(
        binding.auth_principal_sha256
    ):
        raise ValidationError("MCP auth principal digest is invalid")
    if binding.auth_scope_sha256 is not None and not _SHA256_RE.fullmatch(
        binding.auth_scope_sha256
    ):
        raise ValidationError("MCP auth scope digest is invalid")


def _validate_binding_owner(
    binding: McpClientBinding,
    owner_id: str | None,
) -> None:
    if owner_id is not None and binding.owner_id != owner_id:
        raise ValidationError("MCP binding belongs to another owner")


def _validate_binding_secrets(binding: McpClientBinding) -> None:
    if any(type(value) is not str or not value for value in binding.sensitive_values):
        raise ValidationError("MCP sensitive value snapshot is invalid")


def _validate_binding_environment(binding: McpClientBinding) -> None:
    environment = binding.runtime_environment
    if environment is None:
        return
    if not isinstance(environment, Mapping):
        raise ValidationError("MCP runtime environment snapshot is invalid")
    if any(not _valid_environment_item(key, value) for key, value in environment.items()):
        raise ValidationError("MCP runtime environment snapshot is invalid")


def _valid_environment_item(key: Any, value: Any) -> bool:
    return bool(
        type(key) is str
        and key
        and type(value) is str
        and "\x00" not in value
    )


def _validate_manifest_remote_maps(manifest: McpServerManifestV3) -> None:
    _unique_remote((item.remote_uri for item in manifest.resources), "Resource selector")
    _unique_remote(
        (item.remote_uri_template for item in manifest.resource_templates),
        "Resource Template selector",
    )
    resource_ids = {item.resource_id for item in manifest.resources}
    overlap = resource_ids.intersection(item.template_id for item in manifest.resource_templates)
    if overlap:
        raise ValidationError("MCP Resource and Resource Template logical ids must be distinct")


def _unique_remote(values: Any, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValidationError(f"MCP {label} values must be unique")
        seen.add(value)


def _prompt_by_id(manifest: McpServerManifestV3, prompt_id: str) -> McpPromptSpec:
    _validate_id(prompt_id, "prompt_id")
    selected = next((item for item in manifest.prompts if item.prompt_id == prompt_id), None)
    if selected is None:
        raise NotFound(f"MCP prompt not found: {prompt_id}")
    return selected


def _template_by_id(
    manifest: McpServerManifestV3, template_id: str
) -> McpResourceTemplateSpec:
    _validate_id(template_id, "template_id")
    selected = next(
        (item for item in manifest.resource_templates if item.template_id == template_id),
        None,
    )
    if selected is None:
        raise NotFound(f"MCP resource template not found: {template_id}")
    return selected


def _expand_template(
    spec: McpResourceTemplateSpec,
    variables: Mapping[str, str] | None,
    limits: McpModernClientLimits,
) -> str:
    selected = {} if variables is None else variables
    if not isinstance(selected, Mapping):
        raise ValidationError("MCP resource template variables must be an object")
    if set(selected) != set(spec.variables):
        raise ValidationError("MCP resource template requires exactly its declared variables")
    placeholders = tuple(_TEMPLATE_FIELD_RE.findall(spec.remote_uri_template))
    if placeholders != spec.variables or _TEMPLATE_FIELD_RE.sub("", spec.remote_uri_template).find("{") >= 0 or _TEMPLATE_FIELD_RE.sub("", spec.remote_uri_template).find("}") >= 0:
        raise ValidationError("MCP resource template uses an unsupported URI template expression")
    rendered = spec.remote_uri_template
    for name in spec.variables:
        value = selected.get(name)
        if type(value) is not str:
            raise ValidationError("MCP resource template values must be strings")
        if len(value.encode("utf-8")) > limits.max_argument_value_bytes:
            raise ValidationError("MCP resource template value is too large")
        rendered = rendered.replace(f"{{{name}}}", quote(value, safe=""))
    reject_mcp_app_selector(rendered)
    return rendered


def _sanitize_cache_hint(
    hint: McpCacheHint | None, limits: McpModernClientLimits
) -> McpCacheHint | None:
    if hint is None:
        return None
    if not isinstance(hint, McpCacheHint) or type(hint.ttl_ms) is not int or hint.ttl_ms <= 0:
        raise ValidationError("MCP provider cache hint is invalid")
    return replace(hint, ttl_ms=min(hint.ttl_ms, limits.max_cache_ttl_ms))


def _sanitize_input_required(
    value: McpInputRequired, sensitive_values: tuple[str, ...]
) -> McpInputRequired:
    if not value.respondable:
        if (
            value.continuation_id
            or value.expires_at is not None
            or value.revision != 0
            or value.human_request_id is not None
            or value.human_revision is not None
            or value.human_preview_sha256 is not None
            or not value.input_requests
            or any(
                request.kind
                not in {
                    McpInputRequestKind.SAMPLING_UNSUPPORTED,
                    McpInputRequestKind.ROOTS_UNSUPPORTED,
                }
                for request in value.input_requests
            )
        ):
            raise ValidationError("MCP nonrespondable input result is invalid")
        return replace(
            value,
            input_requests=tuple(
                _sanitize_input_request(request, sensitive_values)
                for request in value.input_requests
            ),
        )
    continuation_id = redact_sensitive_text(
        value.continuation_id, sensitive_values=sensitive_values
    )
    if not continuation_id or continuation_id != value.continuation_id:
        # A continuation id is a Host-generated opaque reference.  Mutating a
        # reflected credential into a different reference would be unsafe.
        raise ValidationError("MCP continuation id reflected an operation secret")
    requests: list[McpInputRequest] = []
    for request in value.input_requests:
        requests.append(_sanitize_input_request(request, sensitive_values))
    return replace(
        value,
        continuation_id=continuation_id,
        input_requests=tuple(requests),
        expires_at=(
            None
            if value.expires_at is None
            else redact_sensitive_text(value.expires_at, sensitive_values=sensitive_values)
        ),
    )


def _sanitize_remote_task(
    value: McpRemoteTask, sensitive_values: tuple[str, ...]
) -> McpRemoteTask:
    task_ref = redact_sensitive_text(value.task_ref, sensitive_values=sensitive_values)
    if not task_ref or task_ref != value.task_ref:
        raise ValidationError("MCP task ref reflected an operation secret")
    result = (
        None
        if value.result is None
        else sanitize_provider_json(value.result, sensitive_values=sensitive_values)
    )
    return replace(
        value,
        task_ref=task_ref,
        status_message=(
            None
            if value.status_message is None
            else redact_sensitive_text(
                value.status_message, sensitive_values=sensitive_values
            )
        ),
        result=result,
        input_requests=tuple(
            _sanitize_input_request(request, sensitive_values)
            for request in value.input_requests
        ),
        created_at=(
            None
            if value.created_at is None
            else redact_sensitive_text(value.created_at, sensitive_values=sensitive_values)
        ),
        updated_at=(
            None
            if value.updated_at is None
            else redact_sensitive_text(value.updated_at, sensitive_values=sensitive_values)
        ),
    )


def _sanitize_input_request(
    request: McpInputRequest,
    sensitive_values: tuple[str, ...],
) -> McpInputRequest:
    if not isinstance(request, McpInputRequest):
        raise ValidationError("MCP input request is invalid")
    schema = sanitize_provider_json(request.schema, sensitive_values=sensitive_values)
    if not isinstance(schema, dict):
        raise ValidationError("MCP input request schema must be an object")
    return replace(
        request,
        request_id=redact_sensitive_text(
            request.request_id, sensitive_values=sensitive_values
        ),
        prompt=(
            None
            if request.prompt is None
            else redact_sensitive_text(request.prompt, sensitive_values=sensitive_values)
        ),
        schema=schema,
        inert_url=(
            None
            if request.inert_url is None
            else redact_sensitive_text(
                request.inert_url, sensitive_values=sensitive_values
            )
        ),
    )


def _sdk_next_cursor(result: Any) -> str | None:
    value = getattr(result, "next_cursor", getattr(result, "nextCursor", None))
    if value is not None and type(value) is not str:
        raise ValidationError("MCP provider cursor must be a string")
    return value


async def _collect_catalog_pages(
    session: Any,
    *,
    method_name: str,
    item_fields: tuple[str, ...],
    surface: str,
    item_limit: int,
    projector: Callable[[Any], Any],
    identity: Callable[[Any], str],
    limits: McpCatalogCollectionLimits,
    deadline: float,
    monotonic: Callable[[], float],
) -> tuple[tuple[Any, ...], int]:
    method = getattr(session, method_name, None)
    if not callable(method):
        raise ValidationError(f"MCP catalog session does not support {surface}")
    cursor: str | None = None
    seen_cursors: set[str] = set()
    seen_items: set[str] = set()
    selected: list[Any] = []
    page_count = 0
    while True:
        _check_catalog_deadline(deadline, monotonic, surface)
        params = _sdk_page_params(cursor)
        try:
            candidate = method(params=params)
        except ProviderEffectNotStarted:
            raise
        except Exception:
            raise ValidationError(f"MCP catalog {surface} provider call failed") from None
        if not inspect.isawaitable(candidate):
            raise ValidationError(f"MCP catalog {surface} method must be asynchronous")
        remaining = deadline - monotonic()
        if remaining <= 0:
            if inspect.iscoroutine(candidate):
                candidate.close()
            raise TimeoutError("MCP catalog absolute deadline exhausted")
        try:
            async with asyncio.timeout(remaining):
                page = await candidate
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise TimeoutError("MCP catalog absolute deadline exhausted") from None
        except ProviderEffectNotStarted:
            raise
        except Exception:
            raise ValidationError(f"MCP catalog {surface} provider call failed") from None
        _check_catalog_deadline(deadline, monotonic, surface)
        _require_complete_catalog_page(page, surface)
        raw_items = _catalog_page_items(page, item_fields, surface)
        if len(selected) + len(raw_items) > item_limit:
            raise ValidationError(f"MCP catalog {surface} exceeded its item limit")
        for raw_item in raw_items:
            public_item = projector(raw_item)
            public_identity = identity(public_item)
            if public_identity in seen_items:
                raise ValidationError(f"MCP catalog {surface} contains duplicate entries")
            seen_items.add(public_identity)
            selected.append(public_item)
        cache_hint_from_sdk(page, maximum_ttl_ms=limits.max_cache_ttl_ms)
        page_count += 1
        next_cursor = _sdk_next_cursor(page)
        if next_cursor is None:
            return tuple(selected), page_count
        _validate_catalog_cursor(next_cursor, limits)
        if next_cursor in seen_cursors:
            raise ValidationError(f"MCP catalog {surface} cursor cycle detected")
        seen_cursors.add(next_cursor)
        if page_count >= limits.max_pages_per_catalog:
            raise ValidationError(f"MCP catalog {surface} exceeded its page limit")
        cursor = next_cursor


def _require_complete_catalog_page(page: Any, surface: str) -> None:
    result_type = getattr(page, "result_type", getattr(page, "resultType", "complete"))
    if result_type != "complete":
        raise ValidationError(f"MCP catalog {surface} returned a non-complete result")


def _catalog_page_items(
    page: Any,
    fields: tuple[str, ...],
    surface: str,
) -> list[Any]:
    for field_name in fields:
        value = getattr(page, field_name, None)
        if value is not None:
            if type(value) is not list:
                break
            return value
    raise ValidationError(f"MCP catalog {surface} result is malformed")


def _validate_catalog_cursor(
    cursor: str,
    limits: McpCatalogCollectionLimits,
) -> None:
    if type(cursor) is not str or not cursor:
        raise ValidationError("MCP catalog provider cursor must be a non-empty string")
    if len(cursor.encode("utf-8")) > limits.max_cursor_bytes:
        raise ValidationError("MCP catalog provider cursor is too large")


def _sdk_catalog_tool(
    item: Any,
    sensitive_values: tuple[str, ...],
    limits: McpCatalogCollectionLimits,
) -> McpCatalogTool:
    name = _catalog_identifier(
        getattr(item, "name", None),
        sensitive_values,
        limits,
        label="Tool name",
    )
    input_schema = _catalog_json_object(
        getattr(item, "input_schema", getattr(item, "inputSchema", None)),
        sensitive_values,
        label="Tool input schema",
        required=True,
    )
    output_schema_value = getattr(
        item, "output_schema", getattr(item, "outputSchema", None)
    )
    output_schema = (
        None
        if output_schema_value is None
        else _catalog_json_object(
            output_schema_value,
            sensitive_values,
            label="Tool output schema",
            required=True,
        )
    )
    return McpCatalogTool(
        name=name,
        title=_catalog_optional_text(
            getattr(item, "title", None), sensitive_values, label="Tool title"
        ),
        description=_catalog_optional_text(
            getattr(item, "description", None),
            sensitive_values,
            label="Tool description",
        ),
        input_schema=input_schema,
        output_schema=output_schema,
        annotations=_catalog_json_object(
            getattr(item, "annotations", None),
            sensitive_values,
            label="Tool annotations",
            required=False,
        ),
        metadata=_catalog_json_object(
            getattr(item, "meta", None),
            sensitive_values,
            label="Tool metadata",
            required=False,
        ),
        execution=_catalog_json_object(
            getattr(item, "execution", None),
            sensitive_values,
            label="Tool execution metadata",
            required=False,
        ),
    )


def _catalog_resource(
    item: Any,
    sensitive_values: tuple[str, ...],
    limits: McpCatalogCollectionLimits,
) -> McpResource:
    selector = _catalog_identifier(
        getattr(item, "uri", None),
        sensitive_values,
        limits,
        label="Resource selector",
    )
    public = sdk_resource(item, sensitive_values=sensitive_values)
    if public.resource_id != selector:
        raise ValidationError("MCP catalog Resource selector projection is invalid")
    return public


def _catalog_resource_template(
    item: Any,
    sensitive_values: tuple[str, ...],
    limits: McpCatalogCollectionLimits,
) -> McpResourceTemplate:
    selector = _catalog_identifier(
        getattr(item, "uri_template", getattr(item, "uriTemplate", None)),
        sensitive_values,
        limits,
        label="Resource Template selector",
    )
    public = sdk_resource_template(item, sensitive_values=sensitive_values)
    if public.template_id != selector:
        raise ValidationError("MCP catalog Resource Template selector projection is invalid")
    return public


def _catalog_prompt(
    item: Any,
    sensitive_values: tuple[str, ...],
    limits: McpCatalogCollectionLimits,
) -> McpPrompt:
    name = _catalog_identifier(
        getattr(item, "name", None),
        sensitive_values,
        limits,
        label="Prompt name",
    )
    public = sdk_prompt(item, sensitive_values=sensitive_values)
    if public.prompt_id != name:
        raise ValidationError("MCP catalog Prompt name projection is invalid")
    return public


def _catalog_identifier(
    value: Any,
    sensitive_values: tuple[str, ...],
    limits: McpCatalogCollectionLimits,
    *,
    label: str,
) -> str:
    if type(value) is not str or not value:
        raise ValidationError(f"MCP catalog {label} must be non-empty text")
    if len(value.encode("utf-8")) > limits.max_identifier_bytes:
        raise ValidationError(f"MCP catalog {label} is too large")
    public = redact_sensitive_text(value, sensitive_values=sensitive_values)
    if public != value:
        raise ValidationError(f"MCP catalog {label} reflected an operation secret")
    return value


def _catalog_optional_text(
    value: Any,
    sensitive_values: tuple[str, ...],
    *,
    label: str,
) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValidationError(f"MCP catalog {label} must be text or null")
    return redact_sensitive_text(value, sensitive_values=sensitive_values)


def _catalog_json_object(
    value: Any,
    sensitive_values: tuple[str, ...],
    *,
    label: str,
    required: bool,
) -> dict[str, JsonValue]:
    if value is None:
        if required:
            raise ValidationError(f"MCP catalog {label} must be an object")
        return {}
    if not isinstance(value, Mapping):
        dump = getattr(value, "model_dump", None)
        if not callable(dump):
            raise ValidationError(f"MCP catalog {label} must be an object")
        try:
            value = dump(mode="json", by_alias=True, exclude_none=True)
        except Exception:
            raise ValidationError(f"MCP catalog {label} must be an object") from None
    selected = sanitize_provider_json(value, sensitive_values=sensitive_values)
    if type(selected) is not dict:
        raise ValidationError(f"MCP catalog {label} must be an object")
    return selected


def _check_catalog_deadline(
    deadline: float,
    monotonic: Callable[[], float],
    stage: str,
) -> None:
    if monotonic() >= deadline:
        raise TimeoutError(f"MCP catalog absolute deadline exhausted during {stage}")


def safe_mcp_provider_error(
    error: Exception, sensitive_values: tuple[str, ...]
) -> ValidationError:
    """Return one stable public error after exact operation-secret redaction."""

    message = redact_sensitive_text(str(error), sensitive_values=sensitive_values)
    if not message:
        message = type(error).__name__
    return ValidationError(f"MCP provider operation failed: {message}")


async def _cancel_and_drain_provider_task(task: asyncio.Future[Any]) -> None:
    """Bound cancellation while allowing task-affine SDK scopes to exit.

    The official SDK opens AnyIO cancel scopes and OTel contexts in the
    provider task.  Merely calling ``cancel()`` and returning abandons those
    scopes and leaks late exceptions into the event loop.  Give the same task
    a small, bounded cleanup window; a cancellation-resistant custom Host SPI
    is detached with its eventual result consumed and is never replayed.
    """

    if not task.done():
        task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.shield(task),
            timeout=_PROVIDER_CANCELLATION_DRAIN_S,
        )
    except BaseException:
        pass
    if task.done():
        _consume_provider_task(task)
        return
    task.cancel()
    task.add_done_callback(_consume_provider_task)


def _consume_provider_task(task: asyncio.Future[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass


def _sdk_page_params(cursor: str | None) -> Any:
    if cursor is None:
        return None
    try:
        import mcp.types as mcp_types
    except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
        raise ValidationError("MCP Python SDK v2 is unavailable") from exc
    return mcp_types.PaginatedRequestParams(cursor=cursor)


def _exact_modern_session(selected: Any) -> Any:
    session = getattr(selected, "session", selected)
    if str(getattr(session, "protocol_version", "")) != "2026-07-28":
        raise ValidationError("MCP modern surfaces require exact protocol 2026-07-28")
    return session


def _validate_raw_cursor(cursor: str, limits: McpModernClientLimits) -> None:
    if type(cursor) is not str or not cursor:
        raise ValidationError("MCP provider cursor must be a non-empty string")
    if len(cursor.encode("utf-8")) > limits.max_cursor_bytes:
        raise ValidationError("MCP provider cursor is too large")


def _validate_id(value: Any, label: str) -> None:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 256:
        raise ValidationError(f"MCP {label} is invalid")


def _run_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(awaitable)
        finally:
            try:
                _dispose_sync_operation_loop(loop)
            finally:
                asyncio.set_event_loop(None)
                loop.close()
    if inspect.iscoroutine(awaitable):
        awaitable.close()
    raise RuntimeError(
        "MCP synchronous client cannot run inside an active event loop; use the async method"
    )


def _dispose_sync_operation_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Bound shutdown of one sync operation without Runner's unbounded gather.

    Modern custom Providers are trusted Host SPIs and are expected to cooperate
    with cancellation.  The official SDK gets the ordinary task-affine drain
    in ``_invoke``.  If a custom coroutine yields but repeatedly swallows
    ``CancelledError``, close only this operation-local loop and settle its
    suspended tasks instead of letting ``asyncio.run`` wait forever.  Code that
    blocks before yielding remains outside the async Provider contract.
    """

    pending = _sync_loop_pending_tasks(loop)
    tracked = set(pending)
    for task in pending:
        task.cancel()
    if pending:
        try:
            loop.run_until_complete(_bounded_sync_loop_drain(pending))
        except BaseException:
            pass
    for _attempt in range(_SYNC_LOOP_FORCE_CLOSE_PASSES):
        pending = _sync_loop_pending_tasks(loop)
        if not pending:
            for task in tracked:
                if task.done():
                    _consume_provider_task(task)
            return
        tracked.update(pending)
        for task in pending:
            _force_close_suspended_task(task)
        # Deliver waiter cancellation/task wakeups without creating a new
        # runner Task which would itself become part of the disposal set.
        loop.call_soon(loop.stop)
        loop.run_forever()
        for task in tracked:
            if task.done():
                _consume_provider_task(task)
    remaining = _sync_loop_pending_tasks(loop)
    tracked.update(remaining)
    for task in remaining:
        _force_close_suspended_task(task)
    if remaining:
        loop.call_soon(loop.stop)
        loop.run_forever()
    for task in tracked:
        if task.done():
            _consume_provider_task(task)
    for task in _sync_loop_pending_tasks(loop):
        # Suppress only the destructor diagnostic for a task attached to this
        # now-closed private loop; it can no longer execute or escape.
        if hasattr(task, "_log_destroy_pending"):
            task._log_destroy_pending = False  # type: ignore[attr-defined]  # noqa: SLF001


async def _bounded_sync_loop_drain(tasks: set[asyncio.Task[Any]]) -> None:
    await asyncio.wait(tasks, timeout=_SYNC_LOOP_CANCELLATION_DRAIN_S)


def _sync_loop_pending_tasks(
    loop: asyncio.AbstractEventLoop,
) -> set[asyncio.Task[Any]]:
    return {task for task in asyncio.all_tasks(loop) if not task.done()}


def _force_close_suspended_task(task: asyncio.Task[Any]) -> None:
    if task.done():
        _consume_provider_task(task)
        return
    task.cancel()
    coroutine = task.get_coro()
    close = getattr(coroutine, "close", None)
    if callable(close):
        try:
            close()
        except BaseException:
            pass
    waiter = getattr(task, "_fut_waiter", None)
    if waiter is not None and not waiter.done():
        waiter.cancel()


__all__ = [
    "McpClientBinding",
    "McpClientBindingResolver",
    "McpModernClient",
    "McpModernClientLimits",
    "McpSdkInputRequiredHandler",
    "McpSdkRemoteTaskHandler",
    "McpSdkV2ResultAdapter",
    "McpSdkV2SessionFactory",
    "McpSdkV2SessionProvider",
    "bind_mcp_client_binding",
    "current_mcp_client_binding",
    "mcp_prompt_preview_sha256",
    "mcp_transport_spec_from_v3",
    "sanitize_mcp_operation_result",
]
