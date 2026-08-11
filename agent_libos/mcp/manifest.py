"""Strict Manifest v3 contract for the MCP 2026-07-28 client.

Manifest v3 is intentionally a new type instead of an extension interpreted as
v1/v2.  That keeps the released v1/v2 canonical registry identity byte-stable
and makes accidental legacy downgrade impossible.
"""

from __future__ import annotations

import ipaddress
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping
from urllib.parse import urlsplit

import regex as bounded_regex
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.validators import (
    extend as extend_jsonschema_validator,
    validator_for as jsonschema_validator_for,
)

from agent_libos.mcp.app_policy import (
    is_mcp_app_metadata_key,
    is_mcp_app_mime,
    reject_mcp_app_selector,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.external_effect import (
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.models.mcp import (
    McpHeaderSpec,
    McpHttpTransportSpec,
    McpProtocolMode,
    McpStdioTransportSpec,
    McpToolSpec,
)
from agent_libos.utils.serde import dumps, to_jsonable
from agent_libos.utils.yaml_loader import load_yaml_mapping

MCP_MANIFEST_V3_SCHEMA_VERSION = 3
MCP_V3_PROTOCOL_REVISION = "2026-07-28"
MCP_TASKS_EXTENSION_ID = "io.modelcontextprotocol/tasks"
MCP_V3_SUBSCRIPTION_FILTERS = frozenset(
    {
        "toolsListChanged",
        "promptsListChanged",
        "resourcesListChanged",
        "resourceSubscriptions",
        "taskIds",
    }
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_CALL_RIGHTS = frozenset({"read", "write", "execute"})
_ALLOWED_HEADER_PREFIXES = frozenset({"", "Bearer ", "Token ", "Basic "})
_MODERN_FORBIDDEN_HEADERS = frozenset(
    {
        "accept",
        "accept-charset",
        "accept-encoding",
        "accept-language",
        "baggage",
        "connection",
        "content-encoding",
        "content-language",
        "content-length",
        "content-type",
        "host",
        "last-event-id",
        "mcp-method",
        "mcp-name",
        "mcp-protocol-version",
        "mcp-session-id",
        "traceparent",
        "tracestate",
        "transfer-encoding",
        "upgrade",
    }
)
_LOCAL_HTTP_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_FORBIDDEN_METADATA_HOSTS = frozenset(
    {
        "instance-data",
        "metadata",
        "metadata.google.internal",
    }
)
_FORBIDDEN_PLATFORM_IPS = frozenset(
    {
        # Link-local metadata is also rejected by the non-global literal rule;
        # retaining the exact address here makes that intent auditable.
        "169.254.169.254",
        # Alibaba Cloud and Azure platform endpoints are publicly-routed
        # literals according to generic IP classifiers, but are still ambient
        # Host control-plane destinations and never valid manifest targets.
        "100.100.100.200",
        "168.63.129.16",
    }
)
_DYNAMIC_REFERENCE_KEYS = frozenset(
    {"$dynamicAnchor", "$dynamicRef", "$recursiveAnchor", "$recursiveRef"}
)

# Release ceilings are part of the v3 structural/security contract.  A Host
# policy can only attenuate these values; canonical registry identity therefore
# never depends on one deployment's selected policy.
_RELEASE_SERVER_ID_MAX_CHARS = 96
_RELEASE_TOOL_ID_MAX_CHARS = 96
_RELEASE_MCP_NAME_MAX_CHARS = 256
_RELEASE_HEADER_NAME_MAX_CHARS = 128
_RELEASE_TIMEOUT_MAX_S = 60.0
_RELEASE_REQUEST_MAX_BYTES = 1_048_576
_RELEASE_RESPONSE_MAX_BYTES = 8_388_608
_RELEASE_TOOL_CATALOG_MAX = 100
_RELEASE_SURFACE_CATALOG_MAX = 1_000
_RELEASE_SCHEMA_MAX_DEPTH = 64
_RELEASE_SCHEMA_MAX_NODES = 10_000
_RELEASE_SCHEMA_MAX_REF_HOPS = 128
_RELEASE_SCHEMA_MAX_COMPOSITION_EXPANSIONS = 1_024
_RELEASE_SCHEMA_REGEX_PATTERN_MAX_BYTES = 1_024
_RELEASE_SCHEMA_REGEX_MAX_EVALUATIONS = 4_096
_RELEASE_SCHEMA_REGEX_TIMEOUT_S = 0.05


@dataclass(frozen=True)
class McpManifestV3HostPolicy:
    """Host-selected attenuation applied at registration/deployment time.

    Parsing and canonicalization always enforce the release structural and
    security contract.  Callers that admit a manifest into a concrete Host
    must additionally request policy enforcement.  Environment allowlists
    contain exact names or a non-empty prefix followed by one terminal ``*``;
    a bare wildcard is intentionally invalid.
    """

    server_id_max_chars: int = _RELEASE_SERVER_ID_MAX_CHARS
    tool_id_max_chars: int = _RELEASE_TOOL_ID_MAX_CHARS
    mcp_name_max_chars: int = _RELEASE_MCP_NAME_MAX_CHARS
    header_name_max_chars: int = _RELEASE_HEADER_NAME_MAX_CHARS
    timeout_hard_limit_s: float = _RELEASE_TIMEOUT_MAX_S
    max_request_hard_limit_bytes: int = _RELEASE_REQUEST_MAX_BYTES
    max_response_hard_limit_bytes: int = _RELEASE_RESPONSE_MAX_BYTES
    tool_catalog_limit: int = _RELEASE_TOOL_CATALOG_MAX
    resource_catalog_limit: int = 200
    resource_template_limit: int = 200
    prompt_catalog_limit: int = 200
    schema_max_depth: int = _RELEASE_SCHEMA_MAX_DEPTH
    schema_max_nodes: int = _RELEASE_SCHEMA_MAX_NODES
    schema_max_ref_hops: int = _RELEASE_SCHEMA_MAX_REF_HOPS
    schema_max_composition_expansions: int = (
        _RELEASE_SCHEMA_MAX_COMPOSITION_EXPANSIONS
    )
    schema_regex_pattern_max_bytes: int = _RELEASE_SCHEMA_REGEX_PATTERN_MAX_BYTES
    schema_regex_max_evaluations: int = _RELEASE_SCHEMA_REGEX_MAX_EVALUATIONS
    schema_regex_match_timeout_s: float = _RELEASE_SCHEMA_REGEX_TIMEOUT_S
    header_env_allowlist: tuple[str, ...] = ("AGENT_LIBOS_MCP_*",)
    stdio_env_allowlist: tuple[str, ...] = ("AGENT_LIBOS_MCP_*",)
    oauth_enabled: bool = False
    tasks_extension_enabled: bool = False
    tasks_extension_spec_sha256: str | None = None


DEFAULT_MCP_MANIFEST_V3_HOST_POLICY = McpManifestV3HostPolicy()


@dataclass(frozen=True)
class McpResourceSpec:
    resource_id: str
    remote_uri: str
    right: str = "read"
    information_flow: bool = True
    model_visible: bool = False
    mime_types: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpResourceTemplateSpec:
    template_id: str
    remote_uri_template: str
    variables: tuple[str, ...] = ()
    right: str = "read"
    information_flow: bool = True
    model_visible: bool = False
    mime_types: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpPromptSpec:
    prompt_id: str
    mcp_name: str
    argument_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpTasksExtensionSpec:
    extension_id: str
    spec_sha256: str


@dataclass(frozen=True)
class McpServerManifestV3:
    schema_version: int
    server_id: str
    transport: str
    timeout_s: float
    max_request_bytes: int
    max_response_bytes: int
    protocol_mode: McpProtocolMode
    tools: tuple[McpToolSpec, ...] = ()
    resources: tuple[McpResourceSpec, ...] = ()
    resource_templates: tuple[McpResourceTemplateSpec, ...] = ()
    prompts: tuple[McpPromptSpec, ...] = ()
    stdio: McpStdioTransportSpec | None = None
    http: McpHttpTransportSpec | None = None
    auth_profile_id: str | None = None
    subscriptions: tuple[str, ...] = ()
    tasks_extension: McpTasksExtensionSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def tool_by_id(self, tool_id: str) -> McpToolSpec | None:
        return next((tool for tool in self.tools if tool.tool_id == tool_id), None)


def canonical_mcp_v3_manifest_json(manifest: McpServerManifestV3) -> str:
    """Return the stable v3 registry/evidence identity."""

    # Deployment policy (notably environment allowlists and optional OAuth /
    # Tasks enablement) must not change persisted identity or make a manifest
    # unreadable after a Host policy change.
    validate_mcp_v3_manifest(manifest, enforce_host_policy=False)
    value = to_jsonable(manifest)
    if not isinstance(value, dict):  # pragma: no cover - dataclass invariant
        raise TypeError("MCP Manifest v3 must encode as an object")
    value["protocol_mode"] = manifest.protocol_mode.value
    # JSON has a single number grammar but Python preserves ``1`` versus
    # ``1.0``.  Normalize the public float field so a parse/reopen round trip
    # cannot make a freshly persisted manifest fail its canonicality check.
    value["timeout_s"] = float(manifest.timeout_s)
    return dumps(value)


def validate_mcp_v3_manifest(
    manifest: McpServerManifestV3,
    *,
    tasks_extension_sha256: str | None = None,
    host_policy: McpManifestV3HostPolicy | None = None,
    enforce_host_policy: bool = False,
) -> None:
    """Validate one non-downgradable Manifest v3 authority declaration.

    Release structural/security validation is unconditional and offline.  A
    registration surface must pass ``enforce_host_policy=True``; omitting
    ``host_policy`` then selects :data:`DEFAULT_MCP_MANIFEST_V3_HOST_POLICY`.
    Supplying a policy without requesting enforcement is rejected instead of
    silently ignoring an allowlist or feature gate.
    """

    selected_policy = _select_host_policy(
        host_policy,
        enforce_host_policy=enforce_host_policy,
    )
    bounds = selected_policy or DEFAULT_MCP_MANIFEST_V3_HOST_POLICY
    _validate_optional_tasks_pin(tasks_extension_sha256)
    _validate_manifest_identity(manifest, selected_policy, bounds)
    _validate_manifest_transport(manifest, selected_policy)
    _validate_manifest_limits(manifest, selected_policy, bounds)
    _validate_manifest_catalog_shapes(manifest, selected_policy, bounds)
    _validate_tool_catalog(manifest.tools, selected_policy, bounds)
    _validate_resource_catalog(manifest.resources)
    _validate_resource_template_catalog(manifest.resource_templates)
    _validate_prompt_catalog(manifest.prompts)
    _validate_manifest_extensions(
        manifest,
        selected_policy=selected_policy,
        tasks_extension_sha256=tasks_extension_sha256,
    )
    _validate_json_object(manifest.metadata, "server metadata")
    _reject_apps_metadata(manifest.metadata, "server metadata")


def _validate_optional_tasks_pin(value: str | None) -> None:
    if value is not None and (
        type(value) is not str or _HEX_64_RE.fullmatch(value) is None
    ):
        raise ValidationError("MCP Tasks Host pin must be lowercase SHA-256")


def _validate_manifest_identity(
    manifest: Any,
    policy: McpManifestV3HostPolicy | None,
    bounds: McpManifestV3HostPolicy,
) -> None:
    if not isinstance(manifest, McpServerManifestV3):
        raise ValidationError("MCP Manifest v3 must use McpServerManifestV3")
    if manifest.schema_version != MCP_MANIFEST_V3_SCHEMA_VERSION:
        raise ValidationError("MCP Manifest v3 schema_version must be 3")
    if manifest.protocol_mode is not McpProtocolMode.REVISION_2026_07_28:
        raise ValidationError(
            "MCP Manifest v3 requires exact protocol_mode 2026-07-28"
        )
    _validate_id(
        manifest.server_id,
        "server_id",
        max_chars=(
            bounds.server_id_max_chars
            if policy is not None
            else _RELEASE_SERVER_ID_MAX_CHARS
        ),
    )


def _validate_manifest_transport(
    manifest: McpServerManifestV3,
    policy: McpManifestV3HostPolicy | None,
) -> None:
    if manifest.transport not in {"stdio", "streamable_http"}:
        raise ValidationError(
            "MCP Manifest v3 transport must be stdio or streamable_http"
        )
    if manifest.transport == "stdio":
        if manifest.stdio is None or manifest.http is not None:
            raise ValidationError(
                "MCP Manifest v3 stdio transport requires only stdio"
            )
        if manifest.auth_profile_id is not None:
            raise ValidationError(
                "MCP OAuth auth_profile_id requires streamable_http"
            )
        _validate_stdio_transport(manifest.stdio, policy)
        return
    if manifest.http is None or manifest.stdio is not None:
        raise ValidationError(
            "MCP Manifest v3 streamable_http transport requires only http"
        )
    if manifest.auth_profile_id is not None and _has_authorization_header(
        manifest.http
    ):
        raise ValidationError(
            "MCP OAuth auth_profile_id cannot be combined with a static "
            "Authorization header"
        )
    _validate_http_transport(manifest.http, policy)


def _has_authorization_header(http: McpHttpTransportSpec) -> bool:
    return any(name.casefold() == "authorization" for name in http.headers)


def _validate_manifest_limits(
    manifest: McpServerManifestV3,
    policy: McpManifestV3HostPolicy | None,
    bounds: McpManifestV3HostPolicy,
) -> None:
    if not _is_finite_positive_number(manifest.timeout_s):
        raise ValidationError(
            "MCP Manifest v3 timeout_s must be finite and positive"
        )
    timeout_limit = (
        bounds.timeout_hard_limit_s if policy is not None else _RELEASE_TIMEOUT_MAX_S
    )
    if manifest.timeout_s > timeout_limit:
        raise ValidationError("MCP Manifest v3 timeout_s exceeds hard limit")
    _validate_positive_manifest_integer(
        "max_request_bytes", manifest.max_request_bytes
    )
    _validate_positive_manifest_integer(
        "max_response_bytes", manifest.max_response_bytes
    )
    request_limit = (
        bounds.max_request_hard_limit_bytes
        if policy is not None
        else _RELEASE_REQUEST_MAX_BYTES
    )
    response_limit = (
        bounds.max_response_hard_limit_bytes
        if policy is not None
        else _RELEASE_RESPONSE_MAX_BYTES
    )
    if manifest.max_request_bytes > request_limit:
        raise ValidationError(
            "MCP Manifest v3 max_request_bytes exceeds hard limit"
        )
    if manifest.max_response_bytes > response_limit:
        raise ValidationError(
            "MCP Manifest v3 max_response_bytes exceeds hard limit"
        )


def _is_finite_positive_number(value: Any) -> bool:
    return (
        type(value) in {int, float}
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def _validate_positive_manifest_integer(name: str, value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise ValidationError(
            f"MCP Manifest v3 {name} must be a positive integer"
        )


def _validate_manifest_catalog_shapes(
    manifest: McpServerManifestV3,
    policy: McpManifestV3HostPolicy | None,
    bounds: McpManifestV3HostPolicy,
) -> None:
    catalogs = (
        ("tool", manifest.tools),
        ("resource", manifest.resources),
        ("resource template", manifest.resource_templates),
        ("prompt", manifest.prompts),
    )
    if not any(catalog for _label, catalog in catalogs):
        raise ValidationError(
            "MCP Manifest v3 must declare at least one Tool, Resource, "
            "Resource Template, or Prompt"
        )
    limits = _manifest_catalog_limits(policy, bounds)
    for (label, catalog), limit in zip(catalogs, limits, strict=True):
        if type(catalog) is not tuple:
            raise ValidationError(
                f"MCP Manifest v3 {label} catalog must be a tuple"
            )
        if len(catalog) > limit:
            raise ValidationError(
                f"MCP Manifest v3 {label} catalog exceeds limit={limit}"
            )


def _manifest_catalog_limits(
    policy: McpManifestV3HostPolicy | None,
    bounds: McpManifestV3HostPolicy,
) -> tuple[int, int, int, int]:
    if policy is None:
        return (
            _RELEASE_TOOL_CATALOG_MAX,
            _RELEASE_SURFACE_CATALOG_MAX,
            _RELEASE_SURFACE_CATALOG_MAX,
            _RELEASE_SURFACE_CATALOG_MAX,
        )
    return (
        bounds.tool_catalog_limit,
        bounds.resource_catalog_limit,
        bounds.resource_template_limit,
        bounds.prompt_catalog_limit,
    )


def _validate_tool_catalog(
    tools: tuple[McpToolSpec, ...],
    policy: McpManifestV3HostPolicy | None,
    bounds: McpManifestV3HostPolicy,
) -> None:
    if any(not isinstance(item, McpToolSpec) for item in tools):
        raise ValidationError("MCP Manifest v3 tools must use McpToolSpec")
    _validate_unique((item.tool_id for item in tools), "tool_id")
    _validate_unique((item.mcp_name for item in tools), "tool mcp_name")
    for tool in tools:
        _validate_tool(tool, policy, bounds)


def _validate_tool(
    tool: McpToolSpec,
    policy: McpManifestV3HostPolicy | None,
    bounds: McpManifestV3HostPolicy,
) -> None:
    _validate_id(
        tool.tool_id,
        "tool_id",
        max_chars=(
            bounds.tool_id_max_chars
            if policy is not None
            else _RELEASE_TOOL_ID_MAX_CHARS
        ),
    )
    _validate_mcp_name(
        tool.mcp_name,
        max_chars=(
            bounds.mcp_name_max_chars
            if policy is not None
            else _RELEASE_MCP_NAME_MAX_CHARS
        ),
    )
    if tool.right not in _CALL_RIGHTS:
        raise ValidationError("MCP tool right must be read, write, or execute")
    rollback_class = _validate_tool_effect(tool)
    if (
        rollback_class is ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        and tool.state_mutation
    ):
        raise ValidationError(
            "MCP tool with state_mutation=true cannot use no_rollback_required"
        )
    _validate_v3_json_schema(tool.input_schema, "input_schema", bounds)
    _validate_json_object(tool.metadata, "tool metadata")
    _reject_apps_metadata(tool.metadata, "tool metadata")


def _validate_tool_effect(tool: McpToolSpec) -> ExternalEffectRollbackClass:
    try:
        rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
        if tool.rollback_status is not None:
            ExternalEffectRollbackStatus(tool.rollback_status)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "MCP tool rollback_class or rollback_status is invalid"
        ) from exc
    if type(tool.state_mutation) is not bool:
        raise ValidationError("MCP tool state_mutation must be a boolean")
    if tool.information_flow is not True:
        raise ValidationError(
            "MCP Manifest v3 tool information_flow must be true"
        )
    return rollback_class


def _validate_resource_catalog(resources: tuple[McpResourceSpec, ...]) -> None:
    if any(not isinstance(item, McpResourceSpec) for item in resources):
        raise ValidationError("MCP resources must use McpResourceSpec")
    _validate_unique((item.resource_id for item in resources), "resource_id")
    for resource in resources:
        _validate_id(
            resource.resource_id,
            "resource_id",
            max_chars=_RELEASE_TOOL_ID_MAX_CHARS,
        )
        _validate_remote_selector(resource.remote_uri, "remote_uri")
        _validate_read_surface(
            resource.right, resource.information_flow, "resource"
        )
        if type(resource.model_visible) is not bool:
            raise ValidationError("MCP resource model_visible must be a boolean")
        _validate_mime_types(resource.mime_types)
        _validate_json_object(resource.metadata, "resource metadata")
        _reject_apps_metadata(resource.metadata, "resource metadata")


def _validate_resource_template_catalog(
    templates: tuple[McpResourceTemplateSpec, ...],
) -> None:
    if any(not isinstance(item, McpResourceTemplateSpec) for item in templates):
        raise ValidationError(
            "MCP resource templates must use McpResourceTemplateSpec"
        )
    _validate_unique((item.template_id for item in templates), "template_id")
    for template in templates:
        _validate_resource_template(template)


def _validate_resource_template(template: McpResourceTemplateSpec) -> None:
    _validate_id(
        template.template_id,
        "template_id",
        max_chars=_RELEASE_TOOL_ID_MAX_CHARS,
    )
    _validate_remote_selector(
        template.remote_uri_template, "remote_uri_template"
    )
    _validate_unique(template.variables, "resource template variable")
    for variable in template.variables:
        _validate_id(
            variable,
            "resource template variable",
            max_chars=_RELEASE_TOOL_ID_MAX_CHARS,
        )
    _validate_read_surface(
        template.right,
        template.information_flow,
        "resource template",
    )
    if type(template.model_visible) is not bool:
        raise ValidationError(
            "MCP resource template model_visible must be a boolean"
        )
    _validate_mime_types(template.mime_types)
    _validate_json_object(template.metadata, "resource template metadata")
    _reject_apps_metadata(template.metadata, "resource template metadata")


def _validate_prompt_catalog(prompts: tuple[McpPromptSpec, ...]) -> None:
    if any(not isinstance(item, McpPromptSpec) for item in prompts):
        raise ValidationError("MCP prompts must use McpPromptSpec")
    _validate_unique((item.prompt_id for item in prompts), "prompt_id")
    _validate_unique((item.mcp_name for item in prompts), "prompt mcp_name")
    for prompt in prompts:
        _validate_id(
            prompt.prompt_id,
            "prompt_id",
            max_chars=_RELEASE_TOOL_ID_MAX_CHARS,
        )
        _validate_mcp_name(prompt.mcp_name, max_chars=_RELEASE_MCP_NAME_MAX_CHARS)
        _validate_unique(prompt.argument_names, "prompt argument")
        for name in prompt.argument_names:
            _validate_id(
                name,
                "prompt argument",
                max_chars=_RELEASE_TOOL_ID_MAX_CHARS,
            )
        _validate_json_object(prompt.metadata, "prompt metadata")
        _reject_apps_metadata(prompt.metadata, "prompt metadata")


def _validate_manifest_extensions(
    manifest: McpServerManifestV3,
    *,
    selected_policy: McpManifestV3HostPolicy | None,
    tasks_extension_sha256: str | None,
) -> None:
    _validate_manifest_auth(manifest, selected_policy)
    if type(manifest.subscriptions) is not tuple:
        raise ValidationError("MCP subscriptions must be a tuple")
    _validate_unique(manifest.subscriptions, "subscription filter")
    unknown_filters = sorted(
        set(manifest.subscriptions) - MCP_V3_SUBSCRIPTION_FILTERS
    )
    if unknown_filters:
        raise ValidationError(
            f"unsupported MCP subscription filters: {unknown_filters}"
        )
    if "taskIds" in manifest.subscriptions and manifest.tasks_extension is None:
        raise ValidationError(
            "taskIds subscription requires the pinned Tasks extension"
        )
    if manifest.tasks_extension is not None:
        _validate_tasks_extension(
            manifest.tasks_extension,
            selected_policy=selected_policy,
            tasks_extension_sha256=tasks_extension_sha256,
        )


def _validate_manifest_auth(
    manifest: McpServerManifestV3,
    policy: McpManifestV3HostPolicy | None,
) -> None:
    if manifest.auth_profile_id is None:
        return
    _validate_id(
        manifest.auth_profile_id,
        "auth_profile_id",
        max_chars=_RELEASE_SERVER_ID_MAX_CHARS,
    )
    if policy is not None and not policy.oauth_enabled:
        raise ValidationError("MCP OAuth is disabled by Host policy")


def _validate_tasks_extension(
    selected: McpTasksExtensionSpec,
    *,
    selected_policy: McpManifestV3HostPolicy | None,
    tasks_extension_sha256: str | None,
) -> None:
    if not isinstance(selected, McpTasksExtensionSpec):
        raise ValidationError("MCP tasks_extension must use McpTasksExtensionSpec")
    if selected.extension_id != MCP_TASKS_EXTENSION_ID:
        raise ValidationError("unsupported MCP Tasks extension identifier")
    if not _HEX_64_RE.fullmatch(selected.spec_sha256):
        raise ValidationError(
            "MCP Tasks extension spec_sha256 must be lowercase SHA-256"
        )
    if (
        tasks_extension_sha256 is not None
        and selected.spec_sha256 != tasks_extension_sha256
    ):
        raise ValidationError(
            "MCP Tasks extension spec digest does not match Host pin"
        )
    if selected_policy is None:
        return
    if not selected_policy.tasks_extension_enabled:
        raise ValidationError("MCP Tasks extension is disabled by Host policy")
    if selected.spec_sha256 != selected_policy.tasks_extension_spec_sha256:
        raise ValidationError(
            "MCP Tasks extension spec digest does not match Host policy pin"
        )


def _select_host_policy(
    host_policy: McpManifestV3HostPolicy | None,
    *,
    enforce_host_policy: bool,
) -> McpManifestV3HostPolicy | None:
    if type(enforce_host_policy) is not bool:
        raise ValidationError("MCP enforce_host_policy must be a boolean")
    if not enforce_host_policy:
        if host_policy is not None:
            raise ValidationError(
                "MCP host_policy requires enforce_host_policy=true"
            )
        return None
    selected = host_policy or DEFAULT_MCP_MANIFEST_V3_HOST_POLICY
    if not isinstance(selected, McpManifestV3HostPolicy):
        raise ValidationError("MCP Manifest v3 Host policy type is invalid")
    _validate_host_policy(selected)
    return selected


def _validate_host_policy(policy: McpManifestV3HostPolicy) -> None:
    _validate_host_policy_integer_limits(policy)
    _validate_host_policy_float_limits(policy)
    _validate_host_policy_feature_flags(policy)
    _validate_host_policy_env_allowlists(policy)
    _validate_host_policy_tasks_pin(policy)


def _validate_host_policy_integer_limits(policy: McpManifestV3HostPolicy) -> None:
    limits = (
        ("server_id_max_chars", policy.server_id_max_chars, _RELEASE_SERVER_ID_MAX_CHARS),
        ("tool_id_max_chars", policy.tool_id_max_chars, _RELEASE_TOOL_ID_MAX_CHARS),
        ("mcp_name_max_chars", policy.mcp_name_max_chars, _RELEASE_MCP_NAME_MAX_CHARS),
        ("header_name_max_chars", policy.header_name_max_chars, _RELEASE_HEADER_NAME_MAX_CHARS),
        (
            "max_request_hard_limit_bytes",
            policy.max_request_hard_limit_bytes,
            _RELEASE_REQUEST_MAX_BYTES,
        ),
        (
            "max_response_hard_limit_bytes",
            policy.max_response_hard_limit_bytes,
            _RELEASE_RESPONSE_MAX_BYTES,
        ),
        ("tool_catalog_limit", policy.tool_catalog_limit, _RELEASE_TOOL_CATALOG_MAX),
        (
            "resource_catalog_limit",
            policy.resource_catalog_limit,
            _RELEASE_SURFACE_CATALOG_MAX,
        ),
        (
            "resource_template_limit",
            policy.resource_template_limit,
            _RELEASE_SURFACE_CATALOG_MAX,
        ),
        ("prompt_catalog_limit", policy.prompt_catalog_limit, _RELEASE_SURFACE_CATALOG_MAX),
        ("schema_max_depth", policy.schema_max_depth, _RELEASE_SCHEMA_MAX_DEPTH),
        ("schema_max_nodes", policy.schema_max_nodes, _RELEASE_SCHEMA_MAX_NODES),
        ("schema_max_ref_hops", policy.schema_max_ref_hops, _RELEASE_SCHEMA_MAX_REF_HOPS),
        (
            "schema_max_composition_expansions",
            policy.schema_max_composition_expansions,
            _RELEASE_SCHEMA_MAX_COMPOSITION_EXPANSIONS,
        ),
        (
            "schema_regex_pattern_max_bytes",
            policy.schema_regex_pattern_max_bytes,
            _RELEASE_SCHEMA_REGEX_PATTERN_MAX_BYTES,
        ),
        (
            "schema_regex_max_evaluations",
            policy.schema_regex_max_evaluations,
            _RELEASE_SCHEMA_REGEX_MAX_EVALUATIONS,
        ),
    )
    for name, value, release_maximum in limits:
        if type(value) is not int or value <= 0 or value > release_maximum:
            raise ValidationError(
                f"MCP Host policy {name} must be a positive integer no greater "
                f"than release maximum={release_maximum}"
            )


def _validate_host_policy_float_limits(policy: McpManifestV3HostPolicy) -> None:
    for name, value, release_maximum in (
        ("timeout_hard_limit_s", policy.timeout_hard_limit_s, _RELEASE_TIMEOUT_MAX_S),
        (
            "schema_regex_match_timeout_s",
            policy.schema_regex_match_timeout_s,
            _RELEASE_SCHEMA_REGEX_TIMEOUT_S,
        ),
    ):
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
            or value > release_maximum
        ):
            raise ValidationError(
                f"MCP Host policy {name} must be finite, positive, and no "
                f"greater than release maximum={release_maximum}"
            )


def _validate_host_policy_feature_flags(policy: McpManifestV3HostPolicy) -> None:
    for name, value in (
        ("oauth_enabled", policy.oauth_enabled),
        ("tasks_extension_enabled", policy.tasks_extension_enabled),
    ):
        if type(value) is not bool:
            raise ValidationError(f"MCP Host policy {name} must be a boolean")


def _validate_host_policy_env_allowlists(policy: McpManifestV3HostPolicy) -> None:
    for label, patterns in (
        ("header_env_allowlist", policy.header_env_allowlist),
        ("stdio_env_allowlist", policy.stdio_env_allowlist),
    ):
        if type(patterns) is not tuple:
            raise ValidationError(f"MCP Host policy {label} must be a tuple")
        for pattern in patterns:
            _validate_env_allowlist_pattern(pattern, label)


def _validate_host_policy_tasks_pin(policy: McpManifestV3HostPolicy) -> None:
    digest = policy.tasks_extension_spec_sha256
    if digest is not None and (
        type(digest) is not str or _HEX_64_RE.fullmatch(digest) is None
    ):
        raise ValidationError(
            "MCP Host policy Tasks extension pin must be lowercase SHA-256"
        )
    if policy.tasks_extension_enabled and digest is None:
        raise ValidationError(
            "MCP Host policy Tasks enablement requires an exact spec digest pin"
        )


def _validate_env_allowlist_pattern(value: Any, label: str) -> None:
    if type(value) is not str or not value:
        raise ValidationError(f"MCP Host policy {label} entries must be strings")
    if value.endswith("*"):
        prefix = value[:-1]
        if not prefix or "*" in prefix or not _ENV_RE.fullmatch(prefix):
            raise ValidationError(f"MCP Host policy {label} pattern is invalid")
        return
    if "*" in value or not _ENV_RE.fullmatch(value):
        raise ValidationError(f"MCP Host policy {label} pattern is invalid")


def _env_allowed(name: str, patterns: tuple[str, ...]) -> bool:
    return any(
        name == pattern
        or (pattern.endswith("*") and name.startswith(pattern[:-1]))
        for pattern in patterns
    )


def _validate_env_name(value: Any, label: str) -> None:
    if type(value) is not str or _ENV_RE.fullmatch(value) is None:
        raise ValidationError(f"MCP {label} is not a valid environment name")


def _validate_stdio_transport(
    stdio: McpStdioTransportSpec,
    policy: McpManifestV3HostPolicy | None,
) -> None:
    if not isinstance(stdio, McpStdioTransportSpec):
        raise ValidationError("MCP Manifest v3 stdio configuration type is invalid")
    _validate_stdio_command(stdio.command)
    _validate_stdio_args(stdio.args)
    _validate_stdio_environment(stdio.env, policy)
    _validate_stdio_cwd(stdio.cwd)


def _validate_stdio_command(command: Any) -> None:
    if (
        type(command) is not str
        or not command
        or command != command.strip()
        or any(char.isspace() for char in command)
        or any(char in command for char in "\x00\r\n;&|<>")
    ):
        raise ValidationError(
            "MCP stdio command must be a single argv token, not a shell string"
        )
    if command.startswith("~"):
        raise ValidationError(
            "MCP stdio command must not use Host home-directory expansion"
        )


def _validate_stdio_args(args: Any) -> None:
    if type(args) is not list:
        raise ValidationError("MCP stdio args must be a list")
    for arg in args:
        if type(arg) is not str or "\x00" in arg:
            raise ValidationError("MCP stdio args must be strings without NUL bytes")


def _validate_stdio_environment(
    environment: Any,
    policy: McpManifestV3HostPolicy | None,
) -> None:
    if type(environment) is not dict:
        raise ValidationError("MCP stdio env must be an object")
    for child_name, host_name in environment.items():
        _validate_env_name(child_name, "stdio env name")
        _validate_env_name(host_name, "stdio env source")
        if policy is not None and not _env_allowed(
            host_name, policy.stdio_env_allowlist
        ):
            raise ValidationError(
                f"MCP stdio env source is not allowlisted: {host_name}"
            )


def _validate_stdio_cwd(cwd: Any) -> None:
    if cwd is None:
        return
    if type(cwd) is not str or not cwd or cwd != cwd.strip() or "\x00" in cwd:
        raise ValidationError("MCP stdio cwd must be a non-empty relative path")
    normalized = cwd.replace("\\", "/")
    windows_path = PureWindowsPath(cwd)
    if any(
        (
            PurePosixPath(normalized).is_absolute(),
            windows_path.is_absolute(),
            bool(windows_path.drive),
            cwd.startswith("~"),
        )
    ):
        raise ValidationError("MCP stdio cwd must be a non-empty relative path")
    _validate_relative_cwd_parts(normalized.split("/"))


def _validate_relative_cwd_parts(parts: list[str]) -> None:
    retained: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not retained:
                raise ValidationError("MCP stdio cwd escapes workspace root")
            retained.pop()
            continue
        retained.append(part)


def _validate_http_transport(
    http: McpHttpTransportSpec,
    policy: McpManifestV3HostPolicy | None,
) -> None:
    if not isinstance(http, McpHttpTransportSpec):
        raise ValidationError("MCP Manifest v3 http configuration type is invalid")
    _validate_http_url(http.url)
    if type(http.headers) is not dict:
        raise ValidationError("MCP HTTP headers must be an object")
    if len(http.headers) > 128:
        raise ValidationError("MCP HTTP header catalog exceeds release limit=128")
    header_limit = (
        policy.header_name_max_chars
        if policy is not None
        else _RELEASE_HEADER_NAME_MAX_CHARS
    )
    for name, header in http.headers.items():
        _validate_http_header(name, header, policy, header_limit=header_limit)


def _validate_http_header(
    name: Any,
    header: Any,
    policy: McpManifestV3HostPolicy | None,
    *,
    header_limit: int,
) -> None:
    if (
        type(name) is not str
        or len(name) > header_limit
        or _HEADER_RE.fullmatch(name) is None
    ):
        raise ValidationError(f"invalid MCP header name: {name!r}")
    lowered = name.casefold()
    if lowered in _MODERN_FORBIDDEN_HEADERS or lowered.startswith("mcp-param-"):
        raise ValidationError(f"MCP header is forbidden: {name}")
    if not isinstance(header, McpHeaderSpec):
        raise ValidationError(f"MCP header {name} configuration type is invalid")
    _validate_env_name(header.env, f"header {name} env")
    if policy is not None and not _env_allowed(
        header.env, policy.header_env_allowlist
    ):
        raise ValidationError(f"MCP header env is not allowlisted: {header.env}")
    if header.prefix not in _ALLOWED_HEADER_PREFIXES:
        raise ValidationError(f"MCP header {name} prefix is not allowed")
    if header.suffix != "":
        raise ValidationError(f"MCP header {name} suffix is not allowed")


def _validate_http_url(value: Any) -> None:
    _validate_http_url_text(value)
    parsed = _parse_http_url(value)
    normalized_host, literal = _normalize_http_host(parsed.hostname)
    _validate_http_host(
        normalized_host,
        literal,
        scheme=parsed.scheme,
    )


def _validate_http_url_text(value: Any) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValidationError("MCP HTTP URL must be a non-empty string")
    if any(_invalid_http_url_character(char) for char in value):
        raise ValidationError("MCP HTTP URL contains control characters")


def _invalid_http_url_character(character: str) -> bool:
    return (
        ord(character) < 32
        or ord(character) == 127
        or character.isspace()
        or character == "\\"
    )


def _parse_http_url(value: str) -> Any:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValidationError("MCP HTTP URL or port is invalid") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValidationError("MCP HTTP URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValidationError("MCP HTTP URL must not include userinfo")
    if parsed.fragment:
        raise ValidationError("MCP HTTP URL must not include a fragment")
    if port is not None and port == 0:
        raise ValidationError("MCP HTTP URL has invalid port")
    if not parsed.hostname:
        raise ValidationError("MCP HTTP URL must include a host")
    return parsed


def _normalize_http_host(
    host: str | None,
) -> tuple[str, ipaddress.IPv4Address | ipaddress.IPv6Address | None]:
    raw_host = (host or "").rstrip(".").casefold()
    if not raw_host:
        raise ValidationError("MCP HTTP URL must include a host")
    try:
        return raw_host, ipaddress.ip_address(raw_host.strip("[]"))
    except ValueError:
        try:
            # Socket resolvers apply IDNA mappings, including alternate Unicode
            # dot characters.  Apply the same normalization before comparing
            # local and metadata hostnames so it cannot bypass offline policy.
            normalized_host = raw_host.encode("idna").decode("ascii").casefold()
        except UnicodeError as exc:
            raise ValidationError("MCP HTTP hostname is invalid") from exc
        return normalized_host, None


def _validate_http_host(
    normalized_host: str,
    literal: ipaddress.IPv4Address | ipaddress.IPv6Address | None,
    *,
    scheme: str,
) -> None:
    if normalized_host in _FORBIDDEN_METADATA_HOSTS:
        raise ValidationError("MCP HTTP metadata host is not allowed")
    if scheme == "http" and normalized_host not in _LOCAL_HTTP_HOSTS:
        raise ValidationError(
            "MCP plain HTTP is allowed only for local development hosts"
        )
    if literal is None:
        _validate_dns_host_spelling(normalized_host)
        return
    if str(literal) in _FORBIDDEN_PLATFORM_IPS:
        raise ValidationError("MCP HTTP platform metadata IP is not allowed")
    if normalized_host in _LOCAL_HTTP_HOSTS:
        return
    if _is_forbidden_http_literal(literal):
        raise ValidationError("MCP HTTP IP address is not allowed")


def _validate_dns_host_spelling(host: str) -> None:
    # Numeric and hexadecimal single-label spellings can be interpreted as
    # alternate IP literals by network stacks. Reject them as DNS names.
    if any(
        (
            host.startswith("0x"),
            all(char in "0123456789." for char in host),
            "%" in host,
        )
    ):
        raise ValidationError("MCP HTTP host uses an ambiguous IP literal")


def _is_forbidden_http_literal(
    literal: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    return any(
        (
            not literal.is_global,
            literal.is_private,
            literal.is_loopback,
            literal.is_link_local,
            literal.is_reserved,
            literal.is_multicast,
            literal.is_unspecified,
        )
    )


def _validate_id(value: Any, label: str, *, max_chars: int) -> None:
    if (
        type(value) is not str
        or len(value) > max_chars
        or not _ID_RE.fullmatch(value)
    ):
        raise ValidationError(f"MCP {label} is invalid")


def _validate_mcp_name(value: Any, *, max_chars: int) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > max_chars
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValidationError(
            "MCP mcp_name must be non-empty and within configured length"
        )


def _validate_unique(values: Any, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if type(value) is not str:
            raise ValidationError(f"MCP {label} values must be strings")
        if value in seen:
            raise ValidationError(f"duplicate MCP {label}: {value}")
        seen.add(value)


def _validate_remote_selector(value: Any, label: str) -> None:
    reject_mcp_app_selector(value, label=label)


def _validate_read_surface(right: Any, information_flow: Any, label: str) -> None:
    if right != "read" or information_flow is not True:
        raise ValidationError(
            f"MCP {label} must use right=read and information_flow=true"
        )


def _validate_mime_types(values: tuple[str, ...]) -> None:
    _validate_unique(values, "mime type")
    for value in values:
        if is_mcp_app_mime(value):
            raise ValidationError("MCP Apps HTML resources are unsupported")


def _reject_apps_metadata(value: Mapping[str, Any], label: str) -> None:
    for key, child in value.items():
        if type(key) is not str:
            raise ValidationError(f"MCP {label} keys must be strings")
        if is_mcp_app_metadata_key(key):
            raise ValidationError("MCP Apps metadata is unsupported")
        if isinstance(child, Mapping):
            _reject_apps_metadata(child, label)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, Mapping):
                    _reject_apps_metadata(item, label)


def _validate_json_tree(value: Any, label: str, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValidationError(f"MCP {label} exceeds maximum JSON depth")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValidationError(f"MCP {label} contains a non-finite number")
    if type(value) is list:
        for item in value:
            _validate_json_tree(item, label, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValidationError(f"MCP {label} keys must be strings")
            _validate_json_tree(item, label, depth=depth + 1)
        return
    raise ValidationError(f"MCP {label} must be strict JSON")


def _validate_json_object(value: Any, label: str) -> None:
    if type(value) is not dict:
        raise ValidationError(f"MCP {label} must be a strict JSON object")
    _validate_json_tree(value, label)


def _validate_v3_json_schema(
    schema: Any,
    field: str,
    bounds: McpManifestV3HostPolicy,
) -> None:
    if type(schema) is not dict:
        raise ValidationError(f"MCP {field} must be a strict JSON object")
    if not schema:
        return
    _validate_json_tree(schema, field)
    if schema.get("type") != "object":
        raise ValidationError(f"MCP {field} Manifest v3 root type must be object")

    node_count = 0
    composition_expansions = 1
    references: list[tuple[dict[str, Any], str]] = []
    regex_evaluations = 0
    regex_deadline = time.monotonic() + float(bounds.schema_regex_match_timeout_s)
    active_containers: set[int] = set()

    def validate_pattern(pattern: Any) -> None:
        nonlocal regex_evaluations
        if type(pattern) is not str:
            raise ValidationError(f"MCP {field} regex patterns must be strings")
        if len(pattern.encode("utf-8")) > bounds.schema_regex_pattern_max_bytes:
            raise ValidationError(
                f"MCP {field} regex pattern exceeds "
                f"{bounds.schema_regex_pattern_max_bytes} UTF-8 bytes"
            )
        regex_evaluations += 1
        if regex_evaluations > bounds.schema_regex_max_evaluations:
            raise ValidationError(f"MCP {field} regex evaluation budget exhausted")
        remaining = regex_deadline - time.monotonic()
        if remaining <= 0:
            raise ValidationError(f"MCP {field} regex validation timed out")
        try:
            bounded_regex.search(pattern, "", timeout=remaining)
        except TimeoutError as exc:
            raise ValidationError(f"MCP {field} regex validation timed out") from exc
        except bounded_regex.error as exc:
            raise ValidationError(f"MCP {field} contains an invalid regex") from exc
        if time.monotonic() > regex_deadline:
            raise ValidationError(f"MCP {field} regex validation timed out")

    def multiply_composition(multiplier: int) -> None:
        nonlocal composition_expansions
        composition_expansions *= multiplier
        if composition_expansions > bounds.schema_max_composition_expansions:
            raise ValidationError(
                f"MCP {field} exceeds combinator expansion="
                f"{bounds.schema_max_composition_expansions}"
            )

    def record_schema_keyword(
        container: dict[str, Any],
        key: str,
        item: Any,
    ) -> None:
        if key in _DYNAMIC_REFERENCE_KEYS:
            raise ValidationError(
                f"MCP {field} does not allow dynamic or recursive references"
            )
        if key == "$ref":
            if type(item) is not str or not item.startswith("#"):
                raise ValidationError(f"MCP {field} external $ref is not allowed")
            references.append((container, item))
        if key in {"allOf", "anyOf", "oneOf"}:
            if type(item) is not list:
                raise ValidationError(f"MCP {field} {key} must be an array")
            multiply_composition(max(1, len(item)))

    def walk_mapping(value: dict[str, Any], *, depth: int) -> None:
        if "if" in value and ("then" in value or "else" in value):
            multiply_composition(2)
        pattern = value.get("pattern")
        if pattern is not None:
            validate_pattern(pattern)
        pattern_properties = value.get("patternProperties")
        if pattern_properties is not None:
            if type(pattern_properties) is not dict:
                raise ValidationError(
                    f"MCP {field} patternProperties must be an object"
                )
            for pattern_key in pattern_properties:
                validate_pattern(pattern_key)
        for key, item in value.items():
            record_schema_keyword(value, key, item)
            walk(item, depth=depth + 1)

    def walk(value: Any, *, depth: int) -> None:
        nonlocal node_count
        if depth > bounds.schema_max_depth:
            raise ValidationError(
                f"MCP {field} exceeds schema depth={bounds.schema_max_depth}"
            )
        node_count += 1
        if node_count > bounds.schema_max_nodes:
            raise ValidationError(
                f"MCP {field} exceeds schema nodes={bounds.schema_max_nodes}"
            )
        if type(value) not in {dict, list}:
            return
        identity = id(value)
        if identity in active_containers:
            raise ValidationError(f"MCP {field} contains a cyclic Python value")
        active_containers.add(identity)
        try:
            if type(value) is dict:
                walk_mapping(value, depth=depth)
            else:
                for item in value:
                    walk(item, depth=depth + 1)
        finally:
            active_containers.remove(identity)

    walk(schema, depth=0)
    _validate_schema_reference_graph(
        schema,
        references,
        field=field,
        max_ref_hops=bounds.schema_max_ref_hops,
    )
    try:
        jsonschema_validator_for(schema).check_schema(schema)
    except JsonSchemaSchemaError as exc:
        raise ValidationError(f"MCP {field} is not a valid JSON Schema") from exc


class _McpV3SchemaRegexBudget:
    """Bound every provider-controlled regex evaluation for one argument tree."""

    def __init__(
        self,
        bounds: McpManifestV3HostPolicy,
        *,
        deadline: float | None,
    ) -> None:
        self._bounds = bounds
        local_deadline = time.monotonic() + float(
            bounds.schema_regex_match_timeout_s
        )
        self._deadline = (
            local_deadline if deadline is None else min(local_deadline, deadline)
        )
        self._evaluations = 0
        self._compiled: dict[str, Any] = {}

    def search(self, pattern: Any, value: str) -> bool:
        if type(pattern) is not str:
            raise ValidationError("MCP input_schema regex pattern is invalid")
        if len(pattern.encode("utf-8")) > self._bounds.schema_regex_pattern_max_bytes:
            raise ValidationError("MCP input_schema regex pattern exceeds the byte limit")
        if self._evaluations >= self._bounds.schema_regex_max_evaluations:
            raise ValidationError("MCP schema regex evaluation budget exhausted")
        self._evaluations += 1
        compiled = self._compiled.get(pattern)
        if compiled is None:
            self.remaining()
            try:
                compiled = bounded_regex.compile(pattern)
            except bounded_regex.error as exc:
                raise ValidationError("MCP input_schema contains an invalid regex") from exc
            self._compiled[pattern] = compiled
        try:
            return compiled.search(value, timeout=self.remaining()) is not None
        except TimeoutError as exc:
            raise ValidationError("MCP schema regex validation timed out") from exc

    def remaining(self) -> float:
        selected = self._deadline - time.monotonic()
        if selected <= 0:
            raise ValidationError("MCP schema regex validation timed out")
        return selected


class _McpV3ArgumentValidatorKeywords:
    """Bounded replacements for JSON Schema regex-bearing keywords."""

    def __init__(self, budget: _McpV3SchemaRegexBudget) -> None:
        self.budget = budget

    def validate_pattern(
        self,
        validator: Any,
        pattern: Any,
        instance: Any,
        _schema: Any,
    ) -> Any:
        if validator.is_type(instance, "string") and not self.budget.search(
            pattern, instance
        ):
            yield JsonSchemaValidationError(
                "string does not match schema pattern"
            )

    def validate_pattern_properties(
        self,
        validator: Any,
        pattern_properties: Any,
        instance: Any,
        _schema: Any,
    ) -> Any:
        if not validator.is_type(instance, "object"):
            return
        for pattern, subschema in pattern_properties.items():
            for key, value in instance.items():
                if self.budget.search(pattern, key):
                    yield from validator.descend(
                        value,
                        subschema,
                        path=key,
                        schema_path=pattern,
                    )

    def validate_additional_properties(
        self,
        validator: Any,
        additional: Any,
        instance: Any,
        current_schema: Any,
    ) -> Any:
        if not validator.is_type(instance, "object"):
            return
        extras = self._additional_property_keys(instance, current_schema)
        if validator.is_type(additional, "object"):
            for key in extras:
                yield from validator.descend(
                    instance[key],
                    additional,
                    path=key,
                )
        elif additional is False and extras:
            yield JsonSchemaValidationError(
                "additional properties are not allowed"
            )

    def _additional_property_keys(
        self,
        instance: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> list[str]:
        properties = current_schema.get("properties", {})
        patterns = current_schema.get("patternProperties", {})
        extras: list[str] = []
        for key in instance:
            if key in properties:
                continue
            if any(self.budget.search(pattern, key) for pattern in patterns):
                continue
            extras.append(key)
        return extras

    @staticmethod
    def _descend_is_valid(
        validator: Any,
        instance: Any,
        subschema: Any,
    ) -> bool:
        return next(validator.descend(instance, subschema), None) is None

    def evaluated_property_keys(
        self,
        validator: Any,
        instance: dict[str, Any],
        current_schema: Any,
    ) -> list[str]:
        if validator.is_type(current_schema, "boolean"):
            return []
        evaluated = self._reference_property_keys(
            validator, instance, current_schema
        )
        evaluated.extend(self._direct_property_keys(validator, instance, current_schema))
        evaluated.extend(
            self._dependent_property_keys(validator, instance, current_schema)
        )
        evaluated.extend(
            self._combinator_property_keys(validator, instance, current_schema)
        )
        evaluated.extend(
            self._conditional_property_keys(validator, instance, current_schema)
        )
        return evaluated

    def _reference_property_keys(
        self,
        validator: Any,
        instance: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> list[str]:
        reference = current_schema.get("$ref")
        if reference is None:
            return []
        resolved = validator._resolver.lookup(reference)
        return self.evaluated_property_keys(
            validator.evolve(
                schema=resolved.contents,
                _resolver=resolved.resolver,
            ),
            instance,
            resolved.contents,
        )

    def _direct_property_keys(
        self,
        validator: Any,
        instance: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> list[str]:
        evaluated: list[str] = []
        properties = current_schema.get("properties")
        if validator.is_type(properties, "object"):
            evaluated.extend(properties.keys() & instance.keys())
        for keyword in ("additionalProperties", "unevaluatedProperties"):
            subschema = current_schema.get(keyword)
            if subschema is not None:
                evaluated.extend(
                    key
                    for key, value in instance.items()
                    if self._descend_is_valid(validator, value, subschema)
                )
        for key in instance:
            for pattern in current_schema.get("patternProperties", {}):
                if self.budget.search(pattern, key):
                    evaluated.append(key)
        return evaluated

    def _dependent_property_keys(
        self,
        validator: Any,
        instance: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> list[str]:
        evaluated: list[str] = []
        for key, subschema in current_schema.get("dependentSchemas", {}).items():
            if key in instance:
                evaluated.extend(
                    self.evaluated_property_keys(validator, instance, subschema)
                )
        return evaluated

    def _combinator_property_keys(
        self,
        validator: Any,
        instance: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> list[str]:
        evaluated: list[str] = []
        for keyword in ("allOf", "oneOf", "anyOf"):
            for subschema in current_schema.get(keyword, []):
                if self._descend_is_valid(validator, instance, subschema):
                    evaluated.extend(
                        self.evaluated_property_keys(validator, instance, subschema)
                    )
        return evaluated

    def _conditional_property_keys(
        self,
        validator: Any,
        instance: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> list[str]:
        conditional = current_schema.get("if")
        if conditional is None:
            return []
        if self._descend_is_valid(validator, instance, conditional):
            evaluated = self.evaluated_property_keys(
                validator, instance, conditional
            )
            consequent = current_schema.get("then")
            if consequent is not None:
                evaluated.extend(
                    self.evaluated_property_keys(validator, instance, consequent)
                )
            return evaluated
        alternative = current_schema.get("else")
        if alternative is None:
            return []
        return self.evaluated_property_keys(validator, instance, alternative)

    def validate_unevaluated_properties(
        self,
        validator: Any,
        unevaluated: Any,
        instance: Any,
        current_schema: Any,
    ) -> Any:
        if not validator.is_type(instance, "object"):
            return
        evaluated = set(
            self.evaluated_property_keys(validator, instance, current_schema)
        )
        invalid = any(
            key not in evaluated
            and not self._descend_is_valid(validator, value, unevaluated)
            for key, value in instance.items()
        )
        if invalid:
            yield JsonSchemaValidationError(
                "unevaluated properties are not allowed"
                if unevaluated is False
                else "unevaluated properties are invalid"
            )

    def mapping(self, base_validator: Any) -> dict[str, Any]:
        validators = {
            "additionalProperties": self.validate_additional_properties,
            "pattern": self.validate_pattern,
            "patternProperties": self.validate_pattern_properties,
        }
        if "unevaluatedProperties" in base_validator.VALIDATORS:
            validators["unevaluatedProperties"] = (
                self.validate_unevaluated_properties
            )
        return validators


def validate_mcp_v3_tool_arguments(
    schema: dict[str, Any],
    arguments: dict[str, Any],
    *,
    host_policy: McpManifestV3HostPolicy | None = None,
    deadline: float | None = None,
) -> None:
    """Validate a strict Tool argument object with bounded regex execution."""

    bounds = host_policy or DEFAULT_MCP_MANIFEST_V3_HOST_POLICY
    _validate_host_policy(bounds)
    _validate_json_object(arguments, "tool arguments")
    _validate_v3_json_schema(schema, "input_schema", bounds)
    _validate_tool_argument_deadline(deadline)
    if not schema:
        return
    budget = _McpV3SchemaRegexBudget(bounds, deadline=deadline)
    base_validator = jsonschema_validator_for(schema)
    keywords = _McpV3ArgumentValidatorKeywords(budget)
    bounded_validator = extend_jsonschema_validator(
        base_validator,
        validators=keywords.mapping(base_validator),
    )(schema)
    try:
        bounded_validator.validate(arguments)
        budget.remaining()
    except JsonSchemaValidationError as exc:
        raise ValidationError(
            f"MCP tool arguments failed schema validation: {exc.message}"
        ) from exc


def _validate_tool_argument_deadline(deadline: float | None) -> None:
    if deadline is None:
        return
    if (
        type(deadline) not in {int, float}
        or isinstance(deadline, bool)
        or not math.isfinite(float(deadline))
        or time.monotonic() >= float(deadline)
    ):
        raise ValidationError("MCP tool argument validation deadline expired")


def _resolve_local_schema_ref(
    schema: dict[str, Any],
    reference: str,
    *,
    field: str,
) -> Any:
    if reference == "#":
        return schema
    if not reference.startswith("#/"):
        raise ValidationError(
            f"MCP {field} local $ref must use a JSON Pointer fragment"
        )
    selected: Any = schema
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(selected, dict) and part in selected:
            selected = selected[part]
            continue
        if isinstance(selected, list) and part.isdigit():
            index = int(part)
            if index < len(selected):
                selected = selected[index]
                continue
        raise ValidationError(f"MCP {field} contains an unresolved local $ref")
    return selected


def _validate_schema_reference_graph(
    schema: dict[str, Any],
    references: list[tuple[dict[str, Any], str]],
    *,
    field: str,
    max_ref_hops: int,
) -> None:
    nodes: dict[int, dict[str, Any] | list[Any]] = {}
    edges: dict[int, list[tuple[int, int]]] = {}
    pending: list[dict[str, Any] | list[Any]] = [schema]
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in nodes:
            continue
        nodes[identity] = value
        edges[identity] = []
        children = value.values() if isinstance(value, dict) else value
        for child in children:
            if not isinstance(child, (dict, list)):
                continue
            child_identity = id(child)
            edges[identity].append((child_identity, 0))
            pending.append(child)
    for source, reference in references:
        target = _resolve_local_schema_ref(schema, reference, field=field)
        if not isinstance(target, (dict, list)):
            continue
        source_identity = id(source)
        target_identity = id(target)
        nodes.setdefault(source_identity, source)
        nodes.setdefault(target_identity, target)
        edges.setdefault(source_identity, []).append((target_identity, 1))
        edges.setdefault(target_identity, [])

    indegree = {identity: 0 for identity in nodes}
    for outgoing in edges.values():
        for target_identity, _is_reference in outgoing:
            indegree[target_identity] += 1
    ready = [identity for identity, degree in indegree.items() if degree == 0]
    ref_hops = {identity: 0 for identity in nodes}
    processed = 0
    while ready:
        identity = ready.pop()
        processed += 1
        for target_identity, is_reference in edges[identity]:
            ref_hops[target_identity] = max(
                ref_hops[target_identity],
                ref_hops[identity] + is_reference,
            )
            if ref_hops[target_identity] > max_ref_hops:
                raise ValidationError(
                    f"MCP {field} exceeds local $ref hops={max_ref_hops}"
                )
            indegree[target_identity] -= 1
            if indegree[target_identity] == 0:
                ready.append(target_identity)
    if processed != len(nodes):
        raise ValidationError(f"MCP {field} contains a cyclic local $ref")


_V3_SERVER_FIELDS = frozenset(
    {
        "schema_version",
        "server_id",
        "transport",
        "timeout_s",
        "max_request_bytes",
        "max_response_bytes",
        "protocol_mode",
        "tools",
        "resources",
        "resource_templates",
        "prompts",
        "stdio",
        "http",
        "auth_profile_id",
        "subscriptions",
        "tasks_extension",
        "metadata",
    }
)
_V3_STDIO_FIELDS = frozenset({"command", "args", "env", "cwd"})
_V3_HTTP_FIELDS = frozenset({"url", "headers"})
_V3_HEADER_FIELDS = frozenset({"env", "prefix", "suffix"})
_V3_TOOL_FIELDS = frozenset(
    {
        "tool_id",
        "mcp_name",
        "right",
        "rollback_class",
        "state_mutation",
        "information_flow",
        "rollback_status",
        "input_schema",
        "metadata",
    }
)
_V3_RESOURCE_FIELDS = frozenset(
    {
        "resource_id",
        "remote_uri",
        "right",
        "information_flow",
        "model_visible",
        "mime_types",
        "metadata",
    }
)
_V3_RESOURCE_TEMPLATE_FIELDS = frozenset(
    {
        "template_id",
        "remote_uri_template",
        "variables",
        "right",
        "information_flow",
        "model_visible",
        "mime_types",
        "metadata",
    }
)
_V3_PROMPT_FIELDS = frozenset(
    {"prompt_id", "mcp_name", "argument_names", "metadata"}
)
_V3_TASKS_EXTENSION_FIELDS = frozenset({"extension_id", "spec_sha256"})


def parse_mcp_v3_manifest_mapping(
    value: Mapping[str, Any],
    *,
    tasks_extension_sha256: str | None = None,
    host_policy: McpManifestV3HostPolicy | None = None,
    enforce_host_policy: bool = False,
) -> McpServerManifestV3:
    """Decode and validate one strict MCP Manifest v3 mapping.

    Closed manifest objects reject unknown fields.  ``metadata`` and
    ``input_schema`` are deliberately open JSON objects, but still reject
    non-JSON values and non-string keys.  Sequence inputs may be lists (as
    produced by YAML/JSON) or tuples (for direct Python callers).
    """

    source = _v3_object(value, "MCP Manifest v3", _V3_SERVER_FIELDS)
    protocol_mode_value = _v3_required_string(
        source, "protocol_mode", "MCP Manifest v3"
    )
    try:
        protocol_mode = McpProtocolMode(protocol_mode_value)
    except ValueError as exc:
        raise ValidationError(
            "MCP Manifest v3 protocol_mode must be 2026-07-28"
        ) from exc

    stdio_value = source.get("stdio")
    http_value = source.get("http")
    tasks_value = source.get("tasks_extension")
    manifest = McpServerManifestV3(
        schema_version=_v3_required_int(
            source, "schema_version", "MCP Manifest v3"
        ),
        server_id=_v3_required_string(source, "server_id", "MCP Manifest v3"),
        transport=_v3_required_string(source, "transport", "MCP Manifest v3"),
        timeout_s=_v3_required_number(source, "timeout_s", "MCP Manifest v3"),
        max_request_bytes=_v3_required_int(
            source, "max_request_bytes", "MCP Manifest v3"
        ),
        max_response_bytes=_v3_required_int(
            source, "max_response_bytes", "MCP Manifest v3"
        ),
        protocol_mode=protocol_mode,
        tools=tuple(
            _parse_v3_tool(item, index)
            for index, item in enumerate(
                _v3_sequence(source.get("tools", ()), "MCP Manifest v3 tools")
            )
        ),
        resources=tuple(
            _parse_v3_resource(item, index)
            for index, item in enumerate(
                _v3_sequence(
                    source.get("resources", ()), "MCP Manifest v3 resources"
                )
            )
        ),
        resource_templates=tuple(
            _parse_v3_resource_template(item, index)
            for index, item in enumerate(
                _v3_sequence(
                    source.get("resource_templates", ()),
                    "MCP Manifest v3 resource_templates",
                )
            )
        ),
        prompts=tuple(
            _parse_v3_prompt(item, index)
            for index, item in enumerate(
                _v3_sequence(source.get("prompts", ()), "MCP Manifest v3 prompts")
            )
        ),
        stdio=_parse_v3_stdio(stdio_value) if stdio_value is not None else None,
        http=_parse_v3_http(http_value) if http_value is not None else None,
        auth_profile_id=_v3_optional_string(
            source.get("auth_profile_id"), "MCP Manifest v3 auth_profile_id"
        ),
        subscriptions=_v3_string_tuple(
            source.get("subscriptions", ()), "MCP Manifest v3 subscriptions"
        ),
        tasks_extension=(
            _parse_v3_tasks_extension(tasks_value)
            if tasks_value is not None
            else None
        ),
        metadata=_v3_json_object(
            source.get("metadata", {}), "MCP Manifest v3 metadata"
        ),
    )
    validate_mcp_v3_manifest(
        manifest,
        tasks_extension_sha256=tasks_extension_sha256,
        host_policy=host_policy,
        enforce_host_policy=enforce_host_policy,
    )
    return manifest


def parse_mcp_v3_manifest_yaml_text(
    text: str,
    *,
    tasks_extension_sha256: str | None = None,
    host_policy: McpManifestV3HostPolicy | None = None,
    enforce_host_policy: bool = False,
) -> McpServerManifestV3:
    """Load one bounded YAML document and decode it as Manifest v3."""

    return parse_mcp_v3_manifest_mapping(
        load_yaml_mapping(text),
        tasks_extension_sha256=tasks_extension_sha256,
        host_policy=host_policy,
        enforce_host_policy=enforce_host_policy,
    )


def _parse_v3_stdio(value: Any) -> McpStdioTransportSpec:
    source = _v3_object(value, "MCP Manifest v3 stdio", _V3_STDIO_FIELDS)
    return McpStdioTransportSpec(
        command=_v3_required_string(source, "command", "MCP Manifest v3 stdio"),
        args=list(
            _v3_string_tuple(source.get("args", ()), "MCP Manifest v3 stdio args")
        ),
        env=_v3_string_mapping(source.get("env", {}), "MCP Manifest v3 stdio env"),
        cwd=_v3_optional_string(source.get("cwd"), "MCP Manifest v3 stdio cwd"),
    )


def _parse_v3_http(value: Any) -> McpHttpTransportSpec:
    source = _v3_object(value, "MCP Manifest v3 http", _V3_HTTP_FIELDS)
    raw_headers = _v3_mapping(source.get("headers", {}), "MCP Manifest v3 headers")
    headers: dict[str, McpHeaderSpec] = {}
    for name, raw_header in raw_headers.items():
        if type(name) is not str:
            raise ValidationError("MCP Manifest v3 header names must be strings")
        header = _v3_object(
            raw_header,
            f"MCP Manifest v3 header {name!r}",
            _V3_HEADER_FIELDS,
        )
        headers[name] = McpHeaderSpec(
            env=_v3_required_string(
                header, "env", f"MCP Manifest v3 header {name!r}"
            ),
            prefix=_v3_optional_default_string(
                header, "prefix", f"MCP Manifest v3 header {name!r}", ""
            ),
            suffix=_v3_optional_default_string(
                header, "suffix", f"MCP Manifest v3 header {name!r}", ""
            ),
        )
    return McpHttpTransportSpec(
        url=_v3_required_string(source, "url", "MCP Manifest v3 http"),
        headers=headers,
    )


def _parse_v3_tool(value: Any, index: int) -> McpToolSpec:
    label = f"MCP Manifest v3 tools[{index}]"
    source = _v3_object(value, label, _V3_TOOL_FIELDS)
    return McpToolSpec(
        tool_id=_v3_required_string(source, "tool_id", label),
        mcp_name=_v3_required_string(source, "mcp_name", label),
        right=_v3_required_string(source, "right", label),
        rollback_class=_v3_required_string(source, "rollback_class", label),
        state_mutation=_v3_required_bool(source, "state_mutation", label),
        information_flow=_v3_required_bool(source, "information_flow", label),
        rollback_status=_v3_optional_string(
            source.get("rollback_status"), f"{label} rollback_status"
        ),
        input_schema=_v3_json_object(
            source.get("input_schema", {}), f"{label} input_schema"
        ),
        metadata=_v3_json_object(source.get("metadata", {}), f"{label} metadata"),
    )


def _parse_v3_resource(value: Any, index: int) -> McpResourceSpec:
    label = f"MCP Manifest v3 resources[{index}]"
    source = _v3_object(value, label, _V3_RESOURCE_FIELDS)
    return McpResourceSpec(
        resource_id=_v3_required_string(source, "resource_id", label),
        remote_uri=_v3_required_string(source, "remote_uri", label),
        right=_v3_optional_default_string(source, "right", label, "read"),
        information_flow=_v3_optional_default_bool(
            source, "information_flow", label, True
        ),
        model_visible=_v3_optional_default_bool(
            source, "model_visible", label, False
        ),
        mime_types=_v3_string_tuple(
            source.get("mime_types", ()), f"{label} mime_types"
        ),
        metadata=_v3_json_object(source.get("metadata", {}), f"{label} metadata"),
    )


def _parse_v3_resource_template(
    value: Any, index: int
) -> McpResourceTemplateSpec:
    label = f"MCP Manifest v3 resource_templates[{index}]"
    source = _v3_object(value, label, _V3_RESOURCE_TEMPLATE_FIELDS)
    return McpResourceTemplateSpec(
        template_id=_v3_required_string(source, "template_id", label),
        remote_uri_template=_v3_required_string(
            source, "remote_uri_template", label
        ),
        variables=_v3_string_tuple(
            source.get("variables", ()), f"{label} variables"
        ),
        right=_v3_optional_default_string(source, "right", label, "read"),
        information_flow=_v3_optional_default_bool(
            source, "information_flow", label, True
        ),
        model_visible=_v3_optional_default_bool(
            source, "model_visible", label, False
        ),
        mime_types=_v3_string_tuple(
            source.get("mime_types", ()), f"{label} mime_types"
        ),
        metadata=_v3_json_object(source.get("metadata", {}), f"{label} metadata"),
    )


def _parse_v3_prompt(value: Any, index: int) -> McpPromptSpec:
    label = f"MCP Manifest v3 prompts[{index}]"
    source = _v3_object(value, label, _V3_PROMPT_FIELDS)
    return McpPromptSpec(
        prompt_id=_v3_required_string(source, "prompt_id", label),
        mcp_name=_v3_required_string(source, "mcp_name", label),
        argument_names=_v3_string_tuple(
            source.get("argument_names", ()), f"{label} argument_names"
        ),
        metadata=_v3_json_object(source.get("metadata", {}), f"{label} metadata"),
    )


def _parse_v3_tasks_extension(value: Any) -> McpTasksExtensionSpec:
    label = "MCP Manifest v3 tasks_extension"
    source = _v3_object(value, label, _V3_TASKS_EXTENSION_FIELDS)
    return McpTasksExtensionSpec(
        extension_id=_v3_required_string(source, "extension_id", label),
        spec_sha256=_v3_required_string(source, "spec_sha256", label),
    )


def _v3_object(
    value: Any,
    label: str,
    allowed_fields: frozenset[str],
) -> dict[str, Any]:
    source = _v3_mapping(value, label)
    for key in source:
        if type(key) is not str:
            raise ValidationError(f"{label} field names must be strings")
    unknown = sorted(set(source) - allowed_fields)
    if unknown:
        raise ValidationError(f"{label} has unknown fields: {unknown}")
    return dict(source)


def _v3_mapping(value: Any, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    return value


def _v3_required(source: Mapping[str, Any], field_name: str, label: str) -> Any:
    if field_name not in source:
        raise ValidationError(f"{label} requires {field_name}")
    return source[field_name]


def _v3_required_string(
    source: Mapping[str, Any], field_name: str, label: str
) -> str:
    value = _v3_required(source, field_name, label)
    if type(value) is not str:
        raise ValidationError(f"{label} {field_name} must be a string")
    return value


def _v3_required_int(
    source: Mapping[str, Any], field_name: str, label: str
) -> int:
    value = _v3_required(source, field_name, label)
    if type(value) is not int:
        raise ValidationError(f"{label} {field_name} must be an integer")
    return value


def _v3_required_number(
    source: Mapping[str, Any], field_name: str, label: str
) -> float:
    value = _v3_required(source, field_name, label)
    if type(value) not in {int, float}:
        raise ValidationError(f"{label} {field_name} must be a number")
    try:
        return float(value)
    except OverflowError as exc:
        raise ValidationError(f"{label} {field_name} is outside numeric bounds") from exc


def _v3_required_bool(
    source: Mapping[str, Any], field_name: str, label: str
) -> bool:
    value = _v3_required(source, field_name, label)
    if type(value) is not bool:
        raise ValidationError(f"{label} {field_name} must be a boolean")
    return value


def _v3_optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValidationError(f"{label} must be a string or null")
    return value


def _v3_optional_default_string(
    source: Mapping[str, Any],
    field_name: str,
    label: str,
    default: str,
) -> str:
    if field_name not in source:
        return default
    value = source[field_name]
    if type(value) is not str:
        raise ValidationError(f"{label} {field_name} must be a string")
    return value


def _v3_optional_default_bool(
    source: Mapping[str, Any],
    field_name: str,
    label: str,
    default: bool,
) -> bool:
    if field_name not in source:
        return default
    value = source[field_name]
    if type(value) is not bool:
        raise ValidationError(f"{label} {field_name} must be a boolean")
    return value


def _v3_sequence(value: Any, label: str) -> list[Any] | tuple[Any, ...]:
    if type(value) not in {list, tuple}:
        raise ValidationError(f"{label} must be a list or tuple")
    return value


def _v3_string_tuple(value: Any, label: str) -> tuple[str, ...]:
    sequence = _v3_sequence(value, label)
    if any(type(item) is not str for item in sequence):
        raise ValidationError(f"{label} entries must be strings")
    return tuple(sequence)


def _v3_string_mapping(value: Any, label: str) -> dict[str, str]:
    source = _v3_mapping(value, label)
    if any(type(key) is not str or type(item) is not str for key, item in source.items()):
        raise ValidationError(f"{label} must map strings to strings")
    return dict(source)


def _v3_json_object(value: Any, label: str) -> dict[str, Any]:
    source = _v3_mapping(value, label)
    selected = dict(source)
    _validate_json_tree(selected, label.removeprefix("MCP "))
    return selected
