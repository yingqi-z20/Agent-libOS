"""Public, versioned MCP 2026-07-28 client data contracts.

The types in this module deliberately contain no transport credentials or raw
provider continuation/task identifiers.  Secret-bearing values are represented
by opaque Host references and are resolved only by the credential broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from agent_libos.models.base import StrEnum


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class McpCacheScope(StrEnum):
    PRIVATE = "private"
    PUBLIC = "public"


@dataclass(frozen=True)
class McpCacheHint:
    ttl_ms: int
    scope: McpCacheScope = McpCacheScope.PRIVATE


T = TypeVar("T")


@dataclass(frozen=True)
class McpPage(Generic[T]):
    items: tuple[T, ...]
    next_cursor: str | None = None
    cache_hint: McpCacheHint | None = None

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None


@dataclass(frozen=True)
class McpIcon:
    src: str
    mime_type: str | None = None
    sizes: tuple[str, ...] = ()


@dataclass(frozen=True)
class McpAnnotations:
    audience: tuple[Literal["user", "assistant"], ...] = ()
    priority: float | None = None
    last_modified: str | None = None


@dataclass(frozen=True)
class McpTextContent:
    kind: Literal["text"] = "text"
    text: str = ""
    annotations: McpAnnotations | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class McpArtifactReceipt:
    """Host artifact reference used instead of model-facing base64 content."""

    artifact_id: str
    byte_length: int
    sha256: str
    mime_type: str | None = None


@dataclass(frozen=True)
class McpBlobContent:
    kind: Literal["blob"] = "blob"
    artifact: McpArtifactReceipt | None = None
    annotations: McpAnnotations | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class McpResourceLinkContent:
    """An inert selector.  Agent libOS never dereferences it automatically."""

    kind: Literal["resource_link"] = "resource_link"
    resource_handle: str = ""
    name: str = ""
    title: str | None = None
    description: str | None = None
    mime_type: str | None = None
    annotations: McpAnnotations | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


McpContentBlock: TypeAlias = McpTextContent | McpBlobContent | McpResourceLinkContent


@dataclass(frozen=True)
class McpResource:
    resource_id: str
    name: str
    title: str | None = None
    description: str | None = None
    mime_type: str | None = None
    size: int | None = None
    icons: tuple[McpIcon, ...] = ()
    annotations: McpAnnotations | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class McpResourceTemplate:
    template_id: str
    name: str
    title: str | None = None
    description: str | None = None
    mime_type: str | None = None
    icons: tuple[McpIcon, ...] = ()
    annotations: McpAnnotations | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class McpResourceContents:
    resource_id: str
    contents: tuple[McpContentBlock, ...]
    provenance: Literal["untrusted_mcp_resource"] = "untrusted_mcp_resource"


@dataclass(frozen=True)
class McpPromptArgument:
    name: str
    title: str | None = None
    description: str | None = None
    required: bool = False


@dataclass(frozen=True)
class McpPrompt:
    prompt_id: str
    name: str
    title: str | None = None
    description: str | None = None
    arguments: tuple[McpPromptArgument, ...] = ()
    icons: tuple[McpIcon, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class McpPromptMessage:
    role: Literal["user", "assistant"]
    content: McpContentBlock
    provenance: Literal["untrusted_mcp_prompt"] = "untrusted_mcp_prompt"


@dataclass(frozen=True)
class McpPromptResult:
    prompt_id: str
    messages: tuple[McpPromptMessage, ...]
    description: str | None = None
    user_confirmation_required: bool = True


@dataclass(frozen=True)
class McpCompletionResult:
    values: tuple[str, ...]
    total: int | None = None
    has_more: bool = False


class McpInputRequestKind(StrEnum):
    ELICITATION = "elicitation"
    SAMPLING_UNSUPPORTED = "sampling_unsupported"
    ROOTS_UNSUPPORTED = "roots_unsupported"


@dataclass(frozen=True)
class McpInputRequest:
    request_id: str
    kind: McpInputRequestKind
    mode: Literal["form", "url"] | None = None
    prompt: str | None = None
    schema: dict[str, JsonValue] = field(default_factory=dict)
    # Display-only.  Runtimes and clients must never fetch or open this URL
    # automatically; it remains untrusted MCP Provider content.
    inert_url: str | None = None


@dataclass(frozen=True)
class McpComplete(Generic[T]):
    kind: Literal["complete"] = "complete"
    value: T | None = None
    # Present only for a Host-generated, fence-bound Prompt preview.  Other
    # Complete results retain ``None`` and callers must not invent a digest.
    preview_sha256: str | None = None


@dataclass(frozen=True)
class McpInputRequired:
    kind: Literal["input_required"] = "input_required"
    continuation_id: str = ""
    input_requests: tuple[McpInputRequest, ...] = ()
    expires_at: str | None = None
    revision: int = 0
    respondable: bool = True
    human_request_id: str | None = None
    human_revision: int | None = None
    human_preview_sha256: str | None = None


class McpRemoteTaskStatus(StrEnum):
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCEL_REQUESTED = "cancel_requested"
    NEEDS_ATTENTION = "needs_attention"


@dataclass(frozen=True)
class McpRemoteTask:
    """Safe public projection; the bearer-like remote id is never included."""

    kind: Literal["remote_task"] = "remote_task"
    task_ref: str = ""
    status: McpRemoteTaskStatus = McpRemoteTaskStatus.WORKING
    status_message: str | None = None
    result: JsonValue | None = None
    input_requests: tuple[McpInputRequest, ...] = ()
    created_at: str | None = None
    updated_at: str | None = None
    ttl_ms: int | None = None
    poll_interval_ms: int | None = None
    revision: int = 0
    human_request_id: str | None = None
    human_revision: int | None = None
    human_preview_sha256: str | None = None


McpOperationResult: TypeAlias = McpComplete[T] | McpInputRequired | McpRemoteTask


class McpOAuthStatusKind(StrEnum):
    UNCONFIGURED = "unconfigured"
    AUTHORIZATION_REQUIRED = "authorization_required"
    AUTHORIZED = "authorized"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NEEDS_ATTENTION = "needs_attention"


@dataclass(frozen=True)
class McpOAuthStatus:
    profile_id: str
    status: McpOAuthStatusKind
    issuer: str | None = None
    resource: str | None = None
    scopes: tuple[str, ...] = ()
    principal_sha256: str | None = None
    expires_at: str | None = None


@dataclass(frozen=True)
class McpAuthorizationChallenge:
    challenge_id: str
    authorization_url: str
    expires_at: str


class McpSubscriptionStatus(StrEnum):
    OPENING = "opening"
    ACTIVE = "active"
    LOST = "lost"
    CLOSED = "closed"


@dataclass(frozen=True)
class McpSubscription:
    subscription_id: str
    server_id: str
    status: McpSubscriptionStatus
    requested_filters: tuple[str, ...]
    acknowledged_filters: tuple[str, ...] = ()
    opened_at: str | None = None
    closed_at: str | None = None
    lost_reason: str | None = None


@dataclass(frozen=True)
class McpSubscriptionEvent:
    sequence: int
    event_type: str
    payload: JsonValue
    received_at: str
    provenance: Literal["untrusted_mcp_notification"] = (
        "untrusted_mcp_notification"
    )
