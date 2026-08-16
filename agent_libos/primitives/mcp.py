from __future__ import annotations

import asyncio
import hashlib
import inspect
import ipaddress
import json
import math
import os
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from contextvars import Context, ContextVar
from dataclasses import replace as dataclass_replace
from functools import partial
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

import regex as bounded_regex
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.validators import (
    extend as extend_jsonschema_validator,
    validator_for as jsonschema_validator_for,
)

from agent_libos.capability.manager import CapabilityManager
from agent_libos.capability.rules import AUTHORITY_RULES_KEY
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.human.manager import HumanObjectManager
from agent_libos.mcp._input import json_sha256
from agent_libos.mcp.manifest import (
    MCP_TASKS_EXTENSION_ID,
    MCP_V3_SUBSCRIPTION_FILTERS,
    McpManifestV3HostPolicy,
    McpServerManifestV3,
    canonical_mcp_v3_manifest_json,
    parse_mcp_v3_manifest_mapping,
    validate_mcp_v3_tool_arguments,
    validate_mcp_v3_manifest,
)
from agent_libos.mcp.subscriptions import (
    McpSubscriptionStartSettlement,
    McpTasksSubscriptionFence,
)
from agent_libos.mcp.environment import McpTransportEnvironmentSnapshot
from agent_libos.mcp.client import (
    McpCatalogCollectionLimits,
    McpClientBinding,
    McpCollectedCatalog,
    McpContinuationSurfaceUnsupported,
    bind_mcp_client_binding,
    collect_catalog,
    current_mcp_client_binding,
    mcp_transport_spec_from_v3,
    safe_mcp_provider_error,
    sanitize_mcp_operation_result,
)
from agent_libos.mcp.continuations import (
    McpContinuationBinding,
    McpContinuationDispatchNotStarted,
)
from agent_libos.mcp.oauth import McpOAuthProfile
from agent_libos.mcp.resources import bounded_public_size, sanitize_provider_json
from agent_libos.mcp.tasks import (
    McpRemoteTaskBinding,
    McpRemoteTaskDispatchNotStarted,
)
from agent_libos.mcp.runtime_bridge import mcp_connection_fence
from agent_libos.mcp.types import (
    McpAuthorizationChallenge,
    McpComplete,
    McpInputRequestKind,
    McpInputRequired,
    McpOAuthStatus,
    McpOAuthStatusKind,
    McpPromptResult,
    McpRemoteTask,
    McpSubscription,
    McpSubscriptionEvent,
)
from agent_libos.models import (
    canonical_mcp_server_spec_json,
    CapabilityEffect,
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    DataSink,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    McpCallResult,
    McpCallStatus,
    McpConnectionInfo,
    McpDispatchState,
    McpDiscoveryResult,
    McpExchangePhase,
    McpExchangeReceipt,
    McpHeaderSpec,
    McpHttpTransportSpec,
    McpProtocolEra,
    McpProtocolMode,
    McpRetryClass,
    McpProviderCallResult,
    McpProviderDiscoveryResult,
    McpProviderTool,
    McpServerSpec,
    McpStdioTransportSpec,
    McpToolListResult,
    McpToolSpec,
    ProcessStatus,
    mcp_server_spec_to_jsonable,
    ResourceUsage,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProviderHostError,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.models.external_effect import default_external_effect_rollback_status
from agent_libos.models.mcp import mcp_runtime_secret_values
from agent_libos.ports import AuditPort, EventPort
from agent_libos.storage import McpAuthMetadataRecord, UnitOfWork
from agent_libos.substrate import (
    ExecutableSnapshot,
    executable_content_sha256,
    McpAbsoluteDeadlineProvider,
    McpModernProtocolProvider,
    McpProvider,
    McpSubprocessLimitsProvider,
    ProviderEffectNotStarted,
    SubprocessLimits,
    snapshot_executable,
)
from agent_libos.substrate.local import (
    _allowed_mcp_connect_addresses,
    _bounded_mcp_content,
    SdkMcpProvider,
)
from agent_libos.sdk import (
    ProviderEffectNotStartedResult,
    ProviderRegistryBinding,
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProtectedOperationSDK,
    ProviderPhase,
    ResourceSettlement,
)
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.public_errors import provider_error_envelope_from_mapping
from agent_libos.utils.redaction import redact_sensitive_text
from agent_libos.utils.serde import bounded_json_loads, dumps, to_jsonable
from agent_libos.utils.yaml_loader import load_yaml_mapping

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*$")
_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_LEGACY_FORBIDDEN_HEADERS = {
    "baggage",
    "connection",
    "content-length",
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
_MODERN_FORBIDDEN_HEADERS = _LEGACY_FORBIDDEN_HEADERS | {
    "accept",
    "accept-charset",
    "accept-encoding",
    "accept-language",
    "content-encoding",
    "content-language",
    "content-type",
}
_FORBIDDEN_MCP_HOSTS = {"metadata.google.internal"}
_LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}
_CALL_RIGHTS = {CapabilityRight.READ.value, CapabilityRight.WRITE.value, CapabilityRight.EXECUTE.value}
_ALLOWED_HEADER_PREFIXES = {"", "Bearer ", "Token ", "Basic "}
_ALLOWED_HEADER_SUFFIXES = {""}
_TRANSPORTS = {"stdio", "streamable_http"}
_MCP_WINDOWS = os.name == "nt"
_MCP_WINDOWS_EXECUTABLE_SUFFIXES = {".com", ".exe"}
_MCP_PLATFORM_ENV_KEYS = ("SYSTEMROOT", "WINDIR") if os.name == "nt" else ()
_STDIO_EXECUTABLE_IDENTITY_UNSET = object()
_MCP_IMPORT_CAS_UNSET = object()
_PROVIDER_RESULT_RETURNED_ATTR = "_agent_libos_provider_result_returned"
_INVALID_MCP_TEXT_JSON = object()
_MCP_PROVIDER_JSON_MAX_DEPTH = 128
_MCP_PROVIDER_JSON_MAX_NODES = 100_000
_MCP_STDIO_PROTOCOL_OUTPUT_MULTIPLIER = 4
_MCP_V2_DYNAMIC_REFERENCE_KEYS = {
    "$dynamicAnchor",
    "$dynamicRef",
    "$recursiveAnchor",
    "$recursiveRef",
}
_MCP_RELEASE_PROTOCOL_REVISIONS = {
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
    "2026-07-28",
}
_SERVER_FIELDS = {
    "schema_version",
    "server_id",
    "transport",
    "stdio",
    "http",
    "tools",
    "timeout_s",
    "max_request_bytes",
    "max_response_bytes",
    "metadata",
    "protocol_mode",
}
_STDIO_FIELDS = {"command", "args", "env", "cwd"}
_HTTP_FIELDS = {"url", "headers"}
_TOOL_FIELDS = {
    "tool_id",
    "mcp_name",
    "right",
    "rollback_class",
    "rollback_status",
    "state_mutation",
    "information_flow",
    "input_schema",
    "metadata",
}
_HEADER_FIELDS = {"env", "prefix", "suffix"}

McpRegisteredServer = McpServerSpec | McpServerManifestV3


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    *,
    context: str,
) -> None:
    unknown = sorted(str(field) for field in value if field not in allowed)
    if unknown:
        raise ValidationError(f"unknown {context} fields: {unknown}")


def _mark_provider_result_returned(error: ProviderHostError) -> ProviderHostError:
    """Mark a public provider error as occurring after a result was returned."""

    object.__setattr__(error, _PROVIDER_RESULT_RETURNED_ATTR, True)
    return error


def _provider_result_was_returned(error: BaseException) -> bool:
    try:
        attributes = object.__getattribute__(error, "__dict__")
    except Exception:
        return False
    return attributes.get(_PROVIDER_RESULT_RETURNED_ATTR) is True


def _wire_failure_evidence(
    error: BaseException,
) -> tuple[bool, tuple[McpExchangeReceipt, ...], McpConnectionInfo | None]:
    """Read inert, operation-local wire evidence from an exception chain."""

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    certified = False
    receipts: tuple[McpExchangeReceipt, ...] = ()
    connection: McpConnectionInfo | None = None
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        try:
            attributes = object.__getattribute__(current, "__dict__")
        except Exception:
            attributes = {}
        if attributes.get("_agent_libos_mcp_wire_evidence") is True:
            certified = True
            candidate = attributes.get("_agent_libos_mcp_receipts", ())
            if (
                type(candidate) is tuple
                and all(isinstance(item, McpExchangeReceipt) for item in candidate)
                and len(candidate) >= len(receipts)
            ):
                receipts = candidate
            candidate_connection = attributes.get("_agent_libos_mcp_connection")
            if isinstance(candidate_connection, McpConnectionInfo):
                connection = candidate_connection
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return certified, receipts, connection


class _McpLiveToolValidationError(ValidationError):
    def __init__(self, message: str, result: McpToolListResult) -> None:
        super().__init__(message)
        self.result = result


class _ProviderJsonBudget:
    """Bound normalization work for one provider-owned response tree."""

    __slots__ = (
        "max_depth",
        "max_nodes",
        "max_string_bytes",
        "nodes",
        "string_bytes",
    )

    def __init__(self, max_response_bytes: int) -> None:
        self.max_depth = _MCP_PROVIDER_JSON_MAX_DEPTH
        self.max_nodes = min(_MCP_PROVIDER_JSON_MAX_NODES, max(1, max_response_bytes))
        self.max_string_bytes = max_response_bytes
        self.nodes = 0
        self.string_bytes = 0

    def consume_node(self, *, path: str, depth: int) -> None:
        if depth > self.max_depth:
            raise TypeError(
                f"provider JSON exceeds maximum depth={self.max_depth} at {path}"
            )
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise TypeError(
                f"provider JSON exceeds maximum nodes={self.max_nodes} at {path}"
            )

    def consume_string(self, value: str, *, path: str) -> None:
        try:
            encoded_bytes = len(value.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise TypeError(f"provider JSON contains invalid UTF-8 at {path}") from error
        self.string_bytes += encoded_bytes
        if self.string_bytes > self.max_string_bytes:
            raise TypeError(
                "provider JSON exceeds maximum aggregate string bytes="
                f"{self.max_string_bytes} at {path}"
            )


def _strict_provider_json_value(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
    budget: _ProviderJsonBudget,
    depth: int = 0,
) -> Any:
    """Detach an exact JSON tree returned by a Host provider.

    Provider result objects are outside the runtime trust boundary.  In
    particular, attribute access, container iteration, and nested values may
    execute provider-owned code, so normalize the whole tree before any later
    evidence, accounting, or model-visible projection reads it.
    """

    budget.consume_node(path=path, depth=depth)
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        if value_type is str:
            budget.consume_string(value, path=path)
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise TypeError(f"provider JSON contains a non-finite number at {path}")
        return value
    if value_type is dict:
        identity = id(value)
        if identity in active_containers:
            raise TypeError(f"provider JSON contains a cycle at {path}")
        active_containers.add(identity)
        try:
            selected: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(
                        f"provider JSON contains a non-string key at {path}"
                    )
                budget.consume_string(key, path=f"{path}.<key>")
                selected[key] = _strict_provider_json_value(
                    item,
                    path=f"{path}[{key!r}]",
                    active_containers=active_containers,
                    budget=budget,
                    depth=depth + 1,
                )
            return selected
        finally:
            active_containers.remove(identity)
    if value_type is list:
        identity = id(value)
        if identity in active_containers:
            raise TypeError(f"provider JSON contains a cycle at {path}")
        active_containers.add(identity)
        try:
            return [
                _strict_provider_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                    budget=budget,
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)
    raise TypeError(f"provider JSON contains a non-JSON value at {path}")


def _canonical_mcp_arguments(
    arguments: Any,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    """Return one detached, exact-JSON argument snapshot for all boundaries.

    ``None`` remains a compatibility shorthand for the empty object on the
    trusted Python API. Agent-facing adapters reject explicit JSON null before
    reaching this helper. Exact container types prevent mapping subclasses and
    iterable coercions from running code or changing after authorization.
    """

    if arguments is None:
        return {}
    if type(arguments) is not dict:
        raise ValidationError("MCP tool arguments must be a strict JSON object")
    try:
        selected = _strict_provider_json_value(
            arguments,
            path="$.arguments",
            active_containers=set(),
            budget=_ProviderJsonBudget(max_bytes),
        )
        encoded = dumps(selected).encode("utf-8")
        if len(encoded) > max_bytes:
            raise ValueError("MCP arguments exceed the canonical byte budget")
        canonical = bounded_json_loads(encoded, max_bytes=max_bytes)
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError) as error:
        raise ValidationError(
            "MCP tool arguments must be a strict JSON object"
        ) from error
    if type(canonical) is not dict:  # pragma: no cover - encoded root is exact dict
        raise ValidationError("MCP tool arguments must be a strict JSON object")
    return canonical


def _redact_mcp_provider_json(
    value: Any,
    *,
    sensitive_values: tuple[str, ...],
) -> Any:
    """Recursively project an already-detached provider JSON tree safely."""

    value_type = type(value)
    if value_type is str:
        return redact_sensitive_text(value, sensitive_values=sensitive_values)
    if value_type is list:
        return [
            _redact_mcp_provider_json(
                item,
                sensitive_values=sensitive_values,
            )
            for item in value
        ]
    if value_type is dict:
        selected: dict[str, Any] = {}
        for raw_key, item in value.items():
            public_key = redact_sensitive_text(
                raw_key,
                sensitive_values=sensitive_values,
            )
            candidate = public_key
            suffix = 2
            while candidate in selected:
                candidate = f"{public_key}#{suffix}"
                suffix += 1
            selected[candidate] = _redact_mcp_provider_json(
                item,
                sensitive_values=sensitive_values,
            )
        return selected
    return value


def _redact_mcp_provider_tools(
    tools: list[McpProviderTool],
    *,
    sensitive_values: tuple[str, ...],
) -> list[McpProviderTool]:
    selected: list[McpProviderTool] = []
    public_names: set[str] = set()
    for tool in tools:
        public_name = redact_sensitive_text(
            tool.name,
            sensitive_values=sensitive_values,
        )
        candidate = public_name
        suffix = 2
        while candidate in public_names:
            candidate = f"{public_name}#{suffix}"
            suffix += 1
        public_names.add(candidate)
        input_schema = _redact_mcp_provider_json(
            tool.input_schema,
            sensitive_values=sensitive_values,
        )
        metadata = _redact_mcp_provider_json(
            tool.metadata,
            sensitive_values=sensitive_values,
        )
        selected.append(
            McpProviderTool(
                name=candidate,
                description=(
                    redact_sensitive_text(
                        tool.description,
                        sensitive_values=sensitive_values,
                    )
                    if tool.description is not None
                    else None
                ),
                input_schema=input_schema,
                metadata=metadata,
            )
        )
    return selected


class _McpSchemaRegexBudget:
    """Bound all regex work performed by one JSON Schema validation."""

    __slots__ = (
        "_compiled",
        "_deadline",
        "_evaluations",
        "_max_evaluations",
        "_operation_deadline",
        "_pattern_max_bytes",
    )

    def __init__(
        self,
        *,
        pattern_max_bytes: int,
        max_evaluations: int,
        timeout_s: float,
        operation_deadline: float | None = None,
    ) -> None:
        self._pattern_max_bytes = pattern_max_bytes
        self._max_evaluations = max_evaluations
        self._evaluations = 0
        self._deadline = time.monotonic() + timeout_s
        self._operation_deadline = operation_deadline
        self._compiled: dict[str, Any] = {}

    def compile(self, pattern: Any, *, field: str) -> Any:
        if type(pattern) is not str:
            raise ValidationError(f"MCP {field} regex pattern is invalid")
        try:
            pattern_bytes = len(pattern.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValidationError(
                f"MCP {field} regex pattern is invalid"
            ) from error
        if pattern_bytes > self._pattern_max_bytes:
            raise ValidationError(
                f"MCP {field} regex pattern exceeds maximum bytes="
                f"{self._pattern_max_bytes}"
            )
        compiled = self._compiled.get(pattern)
        if compiled is not None:
            return compiled
        self._remaining_timeout()
        try:
            compiled = bounded_regex.compile(pattern)
        except bounded_regex.error as error:
            raise ValidationError(
                f"MCP {field} regex pattern is invalid"
            ) from error
        self._remaining_timeout()
        self._compiled[pattern] = compiled
        return compiled

    def search(self, pattern: Any, value: str, *, field: str) -> bool:
        if self._evaluations >= self._max_evaluations:
            raise ValidationError("MCP schema regex evaluation budget exhausted")
        self._evaluations += 1
        compiled = self.compile(pattern, field=field)
        remaining = self._remaining_timeout()
        try:
            return compiled.search(value, timeout=remaining) is not None
        except TimeoutError as error:
            self._require_operation_time_remaining()
            raise ValidationError(
                "MCP schema regex validation timed out"
            ) from error

    def _remaining_timeout(self) -> float:
        now = time.monotonic()
        if self._operation_deadline is not None and now >= self._operation_deadline:
            raise ProviderEffectNotStarted(
                "MCP deadline exhausted during schema preflight"
            )
        remaining = self._deadline - now
        if self._operation_deadline is not None:
            remaining = min(remaining, self._operation_deadline - now)
        if remaining <= 0:
            raise ValidationError("MCP schema regex validation timed out")
        return remaining

    def checkpoint(self) -> None:
        """Reject an exhausted operation budget even when no regex is evaluated."""

        self._remaining_timeout()

    def _require_operation_time_remaining(self) -> None:
        if (
            self._operation_deadline is not None
            and time.monotonic() >= self._operation_deadline
        ):
            raise ProviderEffectNotStarted(
                "MCP deadline exhausted during schema preflight"
            )


class _McpBoundedSchemaCallbacks:
    """jsonschema keyword callbacks sharing one regex time/evaluation budget."""

    def __init__(self, budget: _McpSchemaRegexBudget, *, field: str) -> None:
        self._budget = budget
        self._field = field

    def validate_pattern(
        self,
        validator: Any,
        pattern: Any,
        instance: Any,
        _schema: Any,
    ) -> Any:
        if validator.is_type(instance, "string") and not self._budget.search(
            pattern,
            instance,
            field=self._field,
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
                if self._budget.search(
                    pattern,
                    key,
                    field=self._field,
                ):
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
        return [
            key
            for key in instance
            if key not in properties
            and not self._matches_any_pattern(key, patterns)
        ]

    def _matches_any_pattern(self, key: str, patterns: Any) -> bool:
        return any(
            self._budget.search(pattern, key, field=self._field)
            for pattern in patterns
        )

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
        return [
            *self._evaluated_reference_keys(
                validator,
                instance,
                current_schema,
            ),
            *self._evaluated_direct_keys(
                validator,
                instance,
                current_schema,
            ),
            *self._evaluated_dependent_keys(
                validator,
                instance,
                current_schema,
            ),
            *self._evaluated_combinator_keys(
                validator,
                instance,
                current_schema,
            ),
            *self._evaluated_conditional_keys(
                validator,
                instance,
                current_schema,
            ),
        ]

    def _evaluated_reference_keys(
        self,
        validator: Any,
        instance: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> list[str]:
        evaluated: list[str] = []
        for keyword in ("$ref", "$dynamicRef"):
            reference = current_schema.get(keyword)
            if reference is None:
                continue
            resolved = validator._resolver.lookup(reference)
            evaluated.extend(
                self.evaluated_property_keys(
                    validator.evolve(
                        schema=resolved.contents,
                        _resolver=resolved.resolver,
                    ),
                    instance,
                    resolved.contents,
                )
            )
        return evaluated

    def _evaluated_direct_keys(
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
            if subschema is None:
                continue
            evaluated.extend(
                key
                for key, value in instance.items()
                if self._descend_is_valid(validator, value, subschema)
            )
        patterns = current_schema.get("patternProperties", {})
        evaluated.extend(
            key
            for key in instance
            if self._matches_any_pattern(key, patterns)
        )
        return evaluated

    def _evaluated_dependent_keys(
        self,
        validator: Any,
        instance: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> list[str]:
        evaluated: list[str] = []
        for key, subschema in current_schema.get(
            "dependentSchemas",
            {},
        ).items():
            if key in instance:
                evaluated.extend(
                    self.evaluated_property_keys(
                        validator,
                        instance,
                        subschema,
                    )
                )
        return evaluated

    def _evaluated_combinator_keys(
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
                        self.evaluated_property_keys(
                            validator,
                            instance,
                            subschema,
                        )
                    )
        return evaluated

    def _evaluated_conditional_keys(
        self,
        validator: Any,
        instance: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> list[str]:
        conditional = current_schema.get("if")
        if conditional is None:
            return []
        if not validator.evolve(schema=conditional).is_valid(instance):
            return self._evaluated_optional_subschema(
                validator,
                instance,
                current_schema.get("else"),
            )
        return [
            *self.evaluated_property_keys(
                validator,
                instance,
                conditional,
            ),
            *self._evaluated_optional_subschema(
                validator,
                instance,
                current_schema.get("then"),
            ),
        ]

    def _evaluated_optional_subschema(
        self,
        validator: Any,
        instance: dict[str, Any],
        subschema: Any,
    ) -> list[str]:
        if subschema is None:
            return []
        return self.evaluated_property_keys(
            validator,
            instance,
            subschema,
        )

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
            self.evaluated_property_keys(
                validator,
                instance,
                current_schema,
            )
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

    def overrides(self, base_validator: Any) -> dict[str, Any]:
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


def _model_facing_mcp_call_payload(
    content: Any,
    structured_content: Any,
) -> dict[str, Any]:
    """Build a compact MCP result view without discarding distinct content.

    MCP servers commonly return the same object twice: once in
    ``structuredContent`` and once as JSON in a text content block.  Only a
    text-only block whose decoded JSON is exactly equivalent is removed.  All
    non-equivalent prose, annotations, and structured values remain visible.
    Binary content is projected to bounded receipts before equivalence checks,
    so base64 never survives merely because it was nested or JSON-encoded.
    """

    projected_structured = _project_mcp_model_value(structured_content)
    projected_content = _project_mcp_model_value(content)
    if structured_content is not None:
        projected_content = _drop_equivalent_mcp_content(
            projected_content,
            projected_structured,
        )
    return {
        "content": projected_content,
        "structured_content": projected_structured,
    }


def _project_mcp_model_value(value: Any) -> Any:
    projected = _bounded_mcp_content(value)
    if isinstance(projected, list):
        return [_project_mcp_text_json(item) for item in projected]
    return _project_mcp_text_json(projected)


def _project_mcp_text_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_project_mcp_text_json(item) for item in value]
    if not isinstance(value, dict):
        return value

    projected = {
        key: _project_mcp_text_json(item)
        for key, item in value.items()
    }
    if value.get("type") != "text" or not isinstance(value.get("text"), str):
        return projected
    decoded = _decode_mcp_text_json(value["text"])
    if decoded is _INVALID_MCP_TEXT_JSON:
        return projected
    decoded_projection = _project_mcp_model_value(decoded)
    if not _json_values_equivalent(decoded, decoded_projection):
        projected["text"] = dumps(decoded_projection)
    return projected


def _drop_equivalent_mcp_content(content: Any, structured_content: Any) -> Any:
    if _json_values_equivalent(content, structured_content):
        return None
    if isinstance(content, list):
        retained = [
            item
            for item in content
            if not _mcp_text_block_duplicates(item, structured_content)
        ]
        return retained
    if _mcp_text_block_duplicates(content, structured_content):
        return None
    return content


def _mcp_text_block_duplicates(value: Any, structured_content: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"type", "text"}:
        return False
    if value.get("type") != "text" or not isinstance(value.get("text"), str):
        return False
    decoded = _decode_mcp_text_json(value["text"])
    if decoded is _INVALID_MCP_TEXT_JSON:
        return False
    return _json_values_equivalent(
        _project_mcp_model_value(decoded),
        structured_content,
    )


def _decode_mcp_text_json(value: str) -> Any:
    try:
        return bounded_json_loads(value)
    except (TypeError, ValueError, RecursionError):
        return _INVALID_MCP_TEXT_JSON


def _json_values_equivalent(left: Any, right: Any) -> bool:
    try:
        return dumps(left).encode("utf-8") == dumps(right).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError):
        return False


_MCP_SUBSCRIPTION_CANCEL_DRAIN_S = 1.0
_MCP_PROVIDER_CANCEL_DRAIN_S = 0.01
_MCP_SUBSCRIPTION_SETTLEMENT_DRAIN_S = 30.0


class _McpPreparedSubscriptionResult:
    __slots__ = ("effect_settlement", "public", "settlement")

    def __init__(
        self,
        public: McpSubscription,
        settlement: McpSubscriptionStartSettlement,
        effect_settlement: "_McpSubscriptionEffectSettlement",
    ) -> None:
        self.public = public
        self.settlement = settlement
        self.effect_settlement = effect_settlement


class _McpSubscriptionEffectSettlement:
    """Synchronous protected-effect adapter for an owner-loop settlement."""

    fail_closed_on_finalize_error = True
    abort_on_base_exception = True

    def __init__(
        self,
        *,
        runner: "_McpSubscriptionLoopRunner",
        settlement: McpSubscriptionStartSettlement,
        binding: McpClientBinding,
        dispatch_context_var: ContextVar[dict[str, Any] | None],
        dispatch_context: dict[str, Any],
        public: McpSubscription,
    ) -> None:
        self._runner = runner
        self._settlement = settlement
        self._binding = binding
        self._dispatch_context_var = dispatch_context_var
        self._dispatch_context = dispatch_context
        self._public = public

    def commit_deferred(self) -> None:
        self._settlement.commit_deferred()

    def finalize(self) -> None:
        selected = self._run_owned(self._settlement.finalize)
        if selected != self._public:
            raise ValidationError("MCP subscription start publication changed")

    def abort(self, *, reason: str = "subscription_publication_failed") -> None:
        self._run_owned(lambda: self._settlement.abort(reason=reason))

    def _run_owned(self, operation: Any) -> Any:
        token = self._dispatch_context_var.set(self._dispatch_context)
        try:
            return self._runner.run(
                operation,
                deadline=time.monotonic() + _MCP_SUBSCRIPTION_SETTLEMENT_DRAIN_S,
                binding=self._binding,
            )
        finally:
            self._dispatch_context_var.reset(token)


class _McpSubscriptionLoopRunner:
    """Keep long-lived subscription tasks on one Runtime-owned event loop."""

    def __init__(
        self,
        manager: Any,
        *,
        dispatch_context_var: ContextVar[dict[str, Any] | None],
    ) -> None:
        self._manager = manager
        self._dispatch_context_var = dispatch_context_var
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="agent-libos-mcp-subscriptions",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("MCP subscription event loop did not start")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    @property
    def manager(self) -> Any:
        return self._manager

    def run(
        self,
        operation: Any,
        *,
        deadline: float,
        binding: McpClientBinding,
    ) -> Any:
        if not callable(operation):
            raise TypeError("MCP subscription operation must be callable")
        if not isinstance(binding, McpClientBinding):
            raise TypeError("MCP subscription operation binding is invalid")
        dispatch_context = self._dispatch_context_var.get()
        if not isinstance(dispatch_context, dict):
            raise RuntimeError("MCP subscription dispatch context is unavailable")

        settled = threading.Event()

        async def invoke() -> Any:
            # ``run_coroutine_threadsafe`` otherwise copies the caller's full
            # Context, including a ProtectedOperation/lifecycle admission lease
            # which becomes inactive as soon as the synchronous facade returns.
            # Start from an empty Context and restore only the two exact MCP
            # provider-phase bindings needed by the governed SDK session.
            token = self._dispatch_context_var.set(dispatch_context)
            try:
                with bind_mcp_client_binding(binding):
                    return await operation()
            finally:
                self._dispatch_context_var.reset(token)
                settled.set()

        if self._closed or self._loop.is_closed():
            raise RuntimeError("MCP subscription event loop is closed")
        if threading.current_thread() is self._thread:
            raise RuntimeError("MCP subscription facade cannot block its owner loop")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("MCP subscription deadline exceeded")
        # Calling the submission helper inside a fresh Context controls the
        # Context copied by ``loop.call_soon_threadsafe`` into the new Task.
        submitted = invoke()
        try:
            future = Context().run(
                asyncio.run_coroutine_threadsafe,
                submitted,
                self._loop,
            )
        except BaseException:
            submitted.close()
            raise
        try:
            return future.result(timeout=remaining)
        except TimeoutError as exc:
            future.cancel()
            # Cancellation is part of the deadline failure path. Give the
            # owner-loop coroutine one small, fixed cleanup window and consume
            # the concurrent Future state before returning the stable public
            # error. A hostile coroutine can still exceed this window, but it
            # cannot block the synchronous facade indefinitely.
            settled.wait(timeout=_MCP_SUBSCRIPTION_CANCEL_DRAIN_S)
            try:
                future.exception(timeout=0)
            except BaseException:
                pass
            raise TimeoutError("MCP subscription deadline exceeded") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        if not self._loop.is_closed():
            # Manager shutdown must run on the same Runtime-owned loop as its
            # listener/receive tasks and must not inherit the shutdown caller's
            # stale admission or tracing Context.
            shutdown = self._manager.close()
            try:
                close = Context().run(
                    asyncio.run_coroutine_threadsafe,
                    shutdown,
                    self._loop,
                )
            except BaseException:
                shutdown.close()
                raise
            try:
                close.result(timeout=30.0)
            except BaseException as exc:
                close.cancel()
                failures.append(exc)
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=35.0)
        if self._thread.is_alive():
            failures.append(RuntimeError("MCP subscription event loop did not stop"))
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "MCP subscription runner cleanup failed",
                failures,
            )


class McpPrimitiveContinuationBoundary:
    """Async manager adapter whose dispatch remains inside McpPrimitive."""

    def __init__(self, primitive: "McpPrimitive") -> None:
        self.primitive = primitive

    async def continue_request(self, **kwargs: Any) -> Any:
        try:
            return await self.primitive.dispatch_continuation_boundary(
                "respond", **kwargs
            )
        except ProviderEffectNotStarted as error:
            raise McpContinuationDispatchNotStarted(str(error)) from (
                _mcp_pre_provider_cause(error)
            )

    async def cancel_continuation(self, **kwargs: Any) -> None:
        try:
            await self.primitive.dispatch_continuation_boundary("cancel", **kwargs)
        except ProviderEffectNotStarted as error:
            raise McpContinuationDispatchNotStarted(str(error)) from (
                _mcp_pre_provider_cause(error)
            )


class McpPrimitiveRemoteTaskBoundary:
    """Async Tasks manager adapter backed by protected primitive kernels."""

    def __init__(self, primitive: "McpPrimitive") -> None:
        self.primitive = primitive

    async def get_remote_task(self, **kwargs: Any) -> Mapping[str, Any]:
        return await self._dispatch("get", **kwargs)

    async def update_remote_task(self, **kwargs: Any) -> Mapping[str, Any]:
        return await self._dispatch("update", **kwargs)

    async def cancel_remote_task(self, **kwargs: Any) -> Mapping[str, Any]:
        return await self._dispatch("cancel", **kwargs)

    async def _dispatch(self, operation: str, **kwargs: Any) -> Mapping[str, Any]:
        try:
            result = await self.primitive.dispatch_remote_task_boundary(
                operation,
                **kwargs,
            )
        except ProviderEffectNotStarted as error:
            raise McpRemoteTaskDispatchNotStarted(str(error)) from (
                _mcp_pre_provider_cause(error)
            )
        if operation != "get":
            return result
        remote_task_id = kwargs.get("remote_task_id")
        if type(remote_task_id) is not str or not remote_task_id:
            raise ValidationError("MCP remote Task identity is unavailable")
        # The primitive validates the provider's exact identity and redacts it
        # before protected settlement. Reintroduce it only across this private
        # manager boundary so the durable manager can bind its broker secret;
        # no public/evidence observer sees the bearer-like id.
        return {**dict(result), "taskId": remote_task_id}


def _mcp_pre_provider_cause(error: ProviderEffectNotStarted) -> Exception:
    """Recover the exact local denial/preflight cause from its certificate."""

    cause = error.__cause__
    return cause if isinstance(cause, Exception) else error


class _McpProviderEntry:
    """Mutable phase witness shared across a protected synchronous dispatch."""

    __slots__ = ("entered",)

    def __init__(self) -> None:
        self.entered = False


@contextmanager
def _certify_mcp_pre_provider(
    entry: _McpProviderEntry,
    *,
    operation: str,
) -> Any:
    """Certify only failures raised before the Provider callable is entered."""

    try:
        yield
    except ProviderEffectNotStarted:
        # This branch is reserved for primitive-owned pre-provider checks. A
        # Provider-raised instance is normalized inside
        # ``_await_modern_provider_result`` before it can reach this boundary.
        raise
    except Exception as error:
        if entry.entered:
            raise
        raise ProviderEffectNotStarted(
            f"MCP {operation} failed before provider dispatch"
        ) from error


async def _cancel_and_drain_mcp_provider_task(
    task: asyncio.Future[Any],
) -> None:
    """Bound cleanup of one entered exact-v3 custom Provider awaitable."""

    if not task.done():
        task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.shield(task),
            timeout=_MCP_PROVIDER_CANCEL_DRAIN_S,
        )
    except BaseException:
        pass
    if task.done():
        _consume_mcp_provider_task(task)
        return
    task.cancel()
    task.add_done_callback(_consume_mcp_provider_task)


def _consume_mcp_provider_task(task: asyncio.Future[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass


def _run_mcp_provider_awaitable(awaitable: Any) -> Any:
    """Run one cooperative custom-provider exchange on an operation-local loop.

    This loop contains a yielding Provider that mishandles cancellation, but it
    cannot preempt Python code that blocks the event-loop thread.  Custom Host
    SPIs are therefore an explicitly trusted, cooperative composition surface;
    the built-in SDK/transport path owns the hard I/O deadline.
    """

    if not inspect.isawaitable(awaitable):
        raise ValidationError("MCP modern Provider awaitable is invalid")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(awaitable)
    finally:
        # `_await_modern_provider_result` already offers a small bounded
        # cooperative cancellation/drain window. A contract-violating SPI may
        # still swallow CancelledError while yielding; asyncio.run() would then
        # gather it forever during shutdown. This operation-local loop is
        # instead disposed after the bounded attempt. Suppress only asyncio's
        # pending-task destructor diagnostic; provider failure is already
        # recorded as UNKNOWN by ProtectedOperation.
        for task in asyncio.all_tasks(loop):
            if task.done():
                _consume_mcp_provider_task(task)
                continue
            task.cancel()
            setattr(task, "_log_destroy_pending", False)
            coroutine = task.get_coro()
            close = getattr(coroutine, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    pass
        asyncio.set_event_loop(None)
        loop.close()


class McpPrimitive:
    """Capability-controlled MCP client primitive for registered external servers."""

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        capabilities: CapabilityManager,
        audit: AuditPort,
        events: EventPort,
        *,
        protected_operations: ProtectedOperationSDK,
        human: HumanObjectManager | None,
        provider: McpProvider,
        config: AgentLibOSConfig | None = None,
        resources: Any | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.unit_of_work = unit_of_work
        self.extensions = unit_of_work.extensions
        self.authority = unit_of_work.authority
        self.processes = unit_of_work.processes
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.protected_operations = protected_operations
        self.human = human
        self.provider = provider
        self.resources = resources
        self._registry_phase_lock = threading.RLock()
        self._oauth_phase_lock = threading.RLock()
        self._modern_client: Any | None = None
        self._modern_continuations: Any | None = None
        self._modern_remote_tasks: Any | None = None
        self._modern_subscriptions: Any | None = None
        self._modern_subscription_provider: Any | None = None
        self._modern_subscription_runner: _McpSubscriptionLoopRunner | None = None
        self._modern_subscription_lock = threading.RLock()
        self._modern_oauth: Any | None = None
        self._modern_invalidator: Any | None = None
        self._modern_tool_provider: Any | None = None
        self._modern_continuation_provider: Any | None = None
        self._modern_tasks_provider: Any | None = None
        self._modern_request_id_scope = new_id("mcp_request_scope")
        self._modern_request_id_lock = threading.Lock()
        self._modern_request_id_next = 0
        self._modern_dispatch_context: ContextVar[dict[str, Any] | None] = (
            ContextVar(f"agent_libos_mcp_modern_dispatch_{id(self)}", default=None)
        )

    def _bind_modern_client(self, client: Any) -> None:
        """Bind the Host-composed v3 client exactly once before Runtime OPEN."""

        if client is None:
            raise ValidationError("MCP modern client binding cannot be null")
        if self._modern_client is not None and self._modern_client is not client:
            raise ValidationError("MCP modern client is already bound")
        self._modern_client = client

    def _bind_modern_managers(
        self,
        *,
        continuations: Any | None = None,
        remote_tasks: Any | None = None,
        subscriptions: Any | None = None,
        subscription_provider: Any | None = None,
        oauth: Any | None = None,
        invalidator: Any | None = None,
    ) -> None:
        """Attach optional v3 state managers without replacing live bindings."""

        for attribute, value in (
            ("_modern_continuations", continuations),
            ("_modern_remote_tasks", remote_tasks),
            ("_modern_subscriptions", subscriptions),
            ("_modern_subscription_provider", subscription_provider),
            ("_modern_oauth", oauth),
            ("_modern_invalidator", invalidator),
        ):
            if value is None:
                continue
            current = getattr(self, attribute)
            if current is not None and current is not value:
                raise ValidationError(
                    f"MCP modern manager is already bound: {attribute.removeprefix('_modern_')}"
                )
            setattr(self, attribute, value)

    def _bind_modern_wire_providers(
        self,
        *,
        tool_provider: Any | None = None,
        continuation_provider: Any | None = None,
        tasks_provider: Any | None = None,
    ) -> None:
        """Bind exact-2026-07-28 wire adapters without exposing raw SPI calls."""

        for attribute, value in (
            ("_modern_tool_provider", tool_provider),
            ("_modern_continuation_provider", continuation_provider),
            ("_modern_tasks_provider", tasks_provider),
        ):
            if value is None:
                continue
            current = getattr(self, attribute)
            if current is not None and current is not value:
                raise ValidationError(
                    f"MCP modern wire provider is already bound: "
                    f"{attribute.removeprefix('_modern_')}"
                )
            setattr(self, attribute, value)

    def _invalidate_modern_server(self, server_id: str) -> None:
        """Synchronously invalidate all exact-server ephemeral modern state."""

        if not isinstance(server_id, str) or not _ID_PATTERN.fullmatch(server_id):
            raise ValidationError("MCP server_id is invalid")
        delegates = (
            self._modern_client,
            self._modern_subscriptions,
            self._modern_oauth,
            self._modern_invalidator,
        )
        for delegate in delegates:
            if delegate is None:
                continue
            method = getattr(delegate, "invalidate_server_nowait", None)
            if not callable(method):
                method = getattr(delegate, "invalidate_server", None)
            if not callable(method):
                method = getattr(delegate, "close_server", None)
            if not callable(method):
                continue
            try:
                result = method(server_id)
                if hasattr(result, "__await__"):
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
            except BaseException:
                # Registry replacement/unregister is already committed.  A
                # local cleanup failure must not be surfaced as a false
                # rollback; each released manager keeps its own revocation
                # latch and performs Provider cleanup best effort.
                continue

    # ------------------------------------------------------------------
    # Manifest-v3 Runtime facade
    # ------------------------------------------------------------------

    def probe_candidate_manifest(
        self,
        server: McpServerManifestV3 | Mapping[str, Any],
        *,
        expected_manifest_sha256: str,
        confirmed: bool,
        reviewer: str,
        reason: str,
    ) -> McpCollectedCatalog:
        """Collect one unregistered v3 server's complete bounded catalogs.

        This is a trusted Host onboarding operation, never a model or process
        tool.  It validates the exact candidate against active Host policy,
        creates pending-first external-read evidence before DNS/session I/O,
        and uses one transport snapshot and absolute deadline for all four
        catalogs.  It deliberately does not read or mutate the MCP registry.
        """

        manifest = self.validate_server_manifest(server)
        if not isinstance(manifest, McpServerManifestV3):
            raise ValidationError("MCP candidate probe requires Manifest v3")
        selected_reviewer, selected_reason = self._candidate_probe_confirmation(
            confirmed=confirmed,
            reviewer=reviewer,
            reason=reason,
        )
        manifest_sha256 = self._server_spec_sha256(manifest)
        if (
            type(expected_manifest_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None
            or expected_manifest_sha256 != manifest_sha256
        ):
            raise ValidationError("MCP candidate probe manifest digest does not match")
        if manifest.auth_profile_id is not None:
            raise ValidationError(
                "MCP unregistered candidate probe cannot use an OAuth profile; "
                "review static transport discovery first"
            )
        return self._probe_candidate_catalogs(
            manifest,
            manifest_sha256=manifest_sha256,
            reviewer=selected_reviewer,
            reason=selected_reason,
        )

    async def aprobe_candidate_manifest(
        self,
        server: McpServerManifestV3 | Mapping[str, Any],
        *,
        expected_manifest_sha256: str,
        confirmed: bool,
        reviewer: str,
        reason: str,
    ) -> McpCollectedCatalog:
        return await self._data_flow().run_sync_in_worker(
            self.probe_candidate_manifest,
            server,
            expected_manifest_sha256=expected_manifest_sha256,
            confirmed=confirmed,
            reviewer=reviewer,
            reason=reason,
        )

    @staticmethod
    def _candidate_probe_confirmation(
        *,
        confirmed: bool,
        reviewer: str,
        reason: str,
    ) -> tuple[str, str]:
        if confirmed is not True:
            raise ValidationError("MCP candidate probe requires explicit confirmation")
        if type(reviewer) is not str or not reviewer.strip() or len(reviewer) > 128:
            raise ValidationError("MCP candidate probe reviewer is invalid")
        if type(reason) is not str or not reason.strip() or len(reason) > 512:
            raise ValidationError("MCP candidate probe reason is invalid")
        if any(ord(character) < 32 for character in reviewer + reason):
            raise ValidationError("MCP candidate probe review text contains controls")
        return reviewer.strip(), reason.strip()

    def _probe_candidate_catalogs(
        self,
        manifest: McpServerManifestV3,
        *,
        manifest_sha256: str,
        reviewer: str,
        reason: str,
    ) -> McpCollectedCatalog:
        server = mcp_transport_spec_from_v3(manifest)
        environment = self.snapshot_modern_transport_environment(server)
        binding = McpClientBinding(
            manifest=manifest,
            registry_generation=0,
            owner_id="mcp-dx-probe",
            sensitive_values=environment.sensitive_values,
            runtime_environment=environment.runtime_environment,
        )
        deadline = time.monotonic() + manifest.timeout_s
        request_payload = {
            "method": "catalogs/probe",
            "server_id": manifest.server_id,
            "manifest_sha256": manifest_sha256,
            "catalogs": ["tools", "resources", "resource_templates", "prompts"],
            "reviewer": reviewer,
            "reason": reason,
        }
        request_bytes = len(dumps(request_payload).encode("utf-8"))
        if request_bytes > manifest.max_request_bytes:
            raise ValidationError(
                "MCP candidate probe request exceeds "
                f"max_request_bytes={manifest.max_request_bytes}"
            )
        operation_context = {
            "pid": "mcp-dx-probe",
            "primitive": "runtime.mcp.probe_candidate_manifest",
            "operation": "mcp.probe_candidate",
            "authority_operation": "mcp.probe_candidate",
            "server_id": manifest.server_id,
            "logical_id": "full_catalog",
            "tool_id": "full_catalog",
            "right": CapabilityRight.READ.value,
            "registry_spec_sha256": manifest_sha256,
            "registry_generation": 0,
            "request_sha256": hashlib.sha256(
                dumps(request_payload).encode("utf-8")
            ).hexdigest(),
            "arguments_sha256": manifest_sha256,
            "request_bytes": request_bytes,
            "transport": manifest.transport,
            "auth_generation": 0,
            "auth_principal_sha256": None,
            "auth_scope_sha256": None,
            "confirmed": True,
            "reviewer": reviewer,
            "reason": reason,
        }
        plan: dict[str, Any] = {
            "client": None,
            "usage_pid": None,
            "binding": binding,
            "manifest": manifest,
            "deadline": deadline,
            "registry_binding": {
                "registry_spec_sha256": manifest_sha256,
                "registry_generation": 0,
            },
            "request_bytes": request_bytes,
            "operation_context": operation_context,
            "decisions": [],
            "server": server,
        }
        self._prepare_modern_transport(
            plan,
            operation="probe_candidate",
            logical_id="full_catalog",
            actor="mcp-dx-probe",
        )
        target = f"mcp_candidate:{manifest_sha256}"
        plan["invocation"] = self._candidate_probe_invocation(
            plan,
            target=target,
            payload=request_payload,
        )
        return self._execute_modern_operation(
            plan,
            operation="probe_candidate",
            server_id=manifest.server_id,
            logical_id="full_catalog",
            actor="mcp-dx-probe",
            target=target,
            payload=request_payload,
            invoke=self._invoke_candidate_catalog_probe,
            state_mutation=False,
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            contract_name="primitive.mcp.probe_candidate",
            result_settler=None,
        )

    def _candidate_probe_invocation(
        self,
        plan: Mapping[str, Any],
        *,
        target: str,
        payload: Mapping[str, Any],
    ) -> ProtectedOperationInvocation:
        manifest = plan["manifest"]
        context = plan["operation_context"]
        effect_context = {
            **context,
            "rollback_class": ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED.value,
            "rollback_status": ExternalEffectRollbackStatus.NOT_REQUIRED.value,
            "state_mutation": False,
            "information_flow": True,
        }
        return ProtectedOperationInvocation(
            pid="mcp-dx-probe",
            actor="mcp-dx-probe",
            target=target,
            decisions=(),
            canonical_args=context,
            observation=effect_context,
            resource_source="primitive.mcp.probe_candidate",
            resource_context={
                "server_id": manifest.server_id,
                "logical_id": "full_catalog",
                "request_bytes": plan["request_bytes"],
            },
            failure_evidence=lambda error, phase: self._modern_failure_evidence(
                actor="mcp-dx-probe",
                target=target,
                operation="probe_candidate",
                context=context,
                error=error,
                phase=phase,
            ),
            data_sink=plan["sink"],
            data_sink_revalidator=lambda: self._modern_data_sink(
                manifest,
                operation="probe_candidate",
                logical_id="full_catalog",
                stdio_identity=self._stdio_executable_identity(
                    plan["server"],
                    runtime_environment=plan["runtime_environment"],
                    deadline=plan["deadline"],
                    fail_closed=True,
                ),
            ),
            data_flow_context=plan["flow_context"],
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                plan["flow_context"],
                origin="external:mcp",
            ),
            data_flow_payload=dict(payload),
            data_flow_operation="mcp.probe_candidate",
        )

    def _invoke_candidate_catalog_probe(
        self,
        _client: Any,
        deadline: float,
    ) -> McpCollectedCatalog:
        context = self._modern_dispatch_context.get()
        binding = context.get("binding") if isinstance(context, dict) else None
        server = context.get("server") if isinstance(context, dict) else None
        if not isinstance(binding, McpClientBinding) or not isinstance(
            server, McpServerSpec
        ):
            raise ProviderEffectNotStarted(
                "MCP candidate probe is outside a protected provider phase"
            )
        limits = McpCatalogCollectionLimits(
            max_pages_per_catalog=self.config.mcp.list_max_pages,
            max_tools=self.config.mcp.tool_catalog_limit,
            max_resources=self.config.mcp.resource_catalog_limit,
            max_resource_templates=self.config.mcp.resource_template_limit,
            max_prompts=self.config.mcp.prompt_catalog_limit,
            max_cursor_bytes=min(4096, binding.manifest.max_response_bytes),
            max_identifier_bytes=min(8192, binding.manifest.max_response_bytes),
            max_cache_ttl_ms=self.config.mcp.cache_hint_ttl_cap_ms,
            max_public_bytes=binding.manifest.max_response_bytes,
        )

        async def collect_once() -> McpCollectedCatalog:
            with bind_mcp_client_binding(binding):
                async with self._modern_session_factory_context(
                    server,
                    deadline=deadline,
                    binding=binding,
                ) as session:
                    return await collect_catalog(
                        session,
                        limits,
                        deadline,
                        sensitive_values=binding.sensitive_values,
                    )

        result = _run_mcp_provider_awaitable(collect_once())
        if not isinstance(result, McpCollectedCatalog):
            raise ValidationError("MCP candidate probe returned an invalid catalog")
        return result

    def list_resources(
        self,
        server_id: str,
        *,
        cursor: str | None = None,
        actor: str = "runtime",
        model_visible_only: bool = False,
    ) -> Any:
        self._validate_modern_cursor(cursor)
        if type(model_visible_only) is not bool:
            raise ValidationError("MCP model_visible_only must be a boolean")
        payload = {
            "method": "resources/list",
            "server_id": server_id,
            "cursor_present": cursor is not None,
            "model_visible_only": model_visible_only,
        }
        return self._run_modern_read(
            operation="resources.list",
            server_id=server_id,
            logical_id="catalog",
            actor=actor,
            target=self.server_resource(server_id),
            right=CapabilityRight.READ,
            payload=payload,
            invoke=lambda client, deadline: client.list_resources(
                server_id,
                cursor=cursor,
                deadline=deadline,
                owner_id=actor,
                model_visible_only=model_visible_only,
            ),
        )

    async def alist_resources(
        self,
        server_id: str,
        *,
        cursor: str | None = None,
        actor: str = "runtime",
        model_visible_only: bool = False,
    ) -> Any:
        return await self._data_flow().run_sync_in_worker(
            self.list_resources,
            server_id,
            cursor=cursor,
            actor=actor,
            model_visible_only=model_visible_only,
        )

    def list_resource_templates(
        self,
        server_id: str,
        *,
        cursor: str | None = None,
        actor: str = "runtime",
        model_visible_only: bool = False,
    ) -> Any:
        self._validate_modern_cursor(cursor)
        if type(model_visible_only) is not bool:
            raise ValidationError("MCP model_visible_only must be a boolean")
        payload = {
            "method": "resources/templates/list",
            "server_id": server_id,
            "cursor_present": cursor is not None,
            "model_visible_only": model_visible_only,
        }
        return self._run_modern_read(
            operation="resource_templates.list",
            server_id=server_id,
            logical_id="catalog",
            actor=actor,
            target=self.server_resource(server_id),
            right=CapabilityRight.READ,
            payload=payload,
            invoke=lambda client, deadline: client.list_resource_templates(
                server_id,
                cursor=cursor,
                deadline=deadline,
                owner_id=actor,
                model_visible_only=model_visible_only,
            ),
        )

    async def alist_resource_templates(
        self,
        server_id: str,
        *,
        cursor: str | None = None,
        actor: str = "runtime",
        model_visible_only: bool = False,
    ) -> Any:
        return await self._data_flow().run_sync_in_worker(
            self.list_resource_templates,
            server_id,
            cursor=cursor,
            actor=actor,
            model_visible_only=model_visible_only,
        )

    def read_resource(
        self,
        server_id: str,
        resource_id: str,
        *,
        variables: Mapping[str, str] | None = None,
        actor: str = "runtime",
        for_model: bool = False,
    ) -> Any:
        selected_variables = self._modern_string_mapping(
            variables,
            label="Resource variables",
        )
        if type(for_model) is not bool:
            raise ValidationError("MCP for_model must be a boolean")
        self._validate_identifier(
            resource_id,
            "resource_id",
            self.config.mcp.tool_id_max_chars,
        )
        payload = {
            "method": "resources/read",
            "server_id": server_id,
            "resource_id": resource_id,
            "variables": selected_variables,
            "for_model": for_model,
        }
        return self._run_modern_read(
            operation="resources.read",
            server_id=server_id,
            logical_id=resource_id,
            actor=actor,
            target=f"mcp:{server_id}:resource:{resource_id}",
            right=CapabilityRight.READ,
            payload=payload,
            invoke=lambda client, deadline: client.read_resource(
                server_id,
                resource_id,
                variables=selected_variables,
                deadline=deadline,
                owner_id=actor,
                for_model=for_model,
            ),
        )

    async def aread_resource(
        self,
        server_id: str,
        resource_id: str,
        *,
        variables: Mapping[str, str] | None = None,
        actor: str = "runtime",
        for_model: bool = False,
    ) -> Any:
        return await self._data_flow().run_sync_in_worker(
            self.read_resource,
            server_id,
            resource_id,
            variables=variables,
            actor=actor,
            for_model=for_model,
        )

    def list_prompts(
        self,
        server_id: str,
        *,
        cursor: str | None = None,
        actor: str = "runtime",
    ) -> Any:
        self._validate_modern_cursor(cursor)
        payload = {
            "method": "prompts/list",
            "server_id": server_id,
            "cursor_present": cursor is not None,
        }
        return self._run_modern_read(
            operation="prompts.list",
            server_id=server_id,
            logical_id="catalog",
            actor=actor,
            target=self.server_resource(server_id),
            right=CapabilityRight.READ,
            payload=payload,
            invoke=lambda client, deadline: client.list_prompts(
                server_id,
                cursor=cursor,
                deadline=deadline,
                owner_id=actor,
            ),
        )

    async def alist_prompts(
        self,
        server_id: str,
        *,
        cursor: str | None = None,
        actor: str = "runtime",
    ) -> Any:
        return await self._data_flow().run_sync_in_worker(
            self.list_prompts,
            server_id,
            cursor=cursor,
            actor=actor,
        )

    def get_prompt(
        self,
        server_id: str,
        prompt_id: str,
        *,
        arguments: Mapping[str, str] | None = None,
        confirmed: bool = False,
        expected_preview_sha256: str | None = None,
        actor: str = "runtime",
    ) -> Any:
        selected_arguments = self._modern_string_mapping(
            arguments,
            label="Prompt arguments",
        )
        self._validate_identifier(
            prompt_id,
            "prompt_id",
            self.config.mcp.tool_id_max_chars,
        )
        if type(confirmed) is not bool:
            raise ValidationError("MCP Prompt confirmed must be a boolean")
        self._validate_prompt_preview_request(
            confirmed=confirmed,
            expected_preview_sha256=expected_preview_sha256,
        )
        payload = {
            "method": "prompts/get",
            "server_id": server_id,
            "prompt_id": prompt_id,
            "arguments": selected_arguments,
            "confirmed": confirmed,
        }

        def invoke(client: Any, deadline: float) -> Any:
            result = client.get_prompt(
                server_id,
                prompt_id,
                arguments=selected_arguments,
                deadline=deadline,
                owner_id=actor,
            )
            if not isinstance(result, McpComplete) or not isinstance(
                result.value,
                McpPromptResult,
            ):
                if confirmed:
                    raise ValidationError(
                        "confirmed MCP Prompt did not return a Complete preview"
                    )
                return result
            digest = result.preview_sha256
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValidationError("MCP Prompt preview digest is unavailable")
            if confirmed and digest != expected_preview_sha256:
                raise CapabilityDenied(
                    "MCP Prompt preview changed before confirmation"
                )
            return result

        return self._run_modern_read(
            operation="prompts.get",
            server_id=server_id,
            logical_id=prompt_id,
            actor=actor,
            target=f"mcp:{server_id}:prompt:{prompt_id}",
            right=CapabilityRight.READ,
            payload=payload,
            invoke=invoke,
        )

    async def aget_prompt(
        self,
        server_id: str,
        prompt_id: str,
        *,
        arguments: Mapping[str, str] | None = None,
        confirmed: bool = False,
        expected_preview_sha256: str | None = None,
        actor: str = "runtime",
    ) -> Any:
        return await self._data_flow().run_sync_in_worker(
            self.get_prompt,
            server_id,
            prompt_id,
            arguments=arguments,
            confirmed=confirmed,
            expected_preview_sha256=expected_preview_sha256,
            actor=actor,
        )

    def complete_prompt(
        self,
        server_id: str,
        reference_type: str,
        reference_id: str,
        argument: Mapping[str, str],
        *,
        context: Mapping[str, str] | None = None,
        actor: str = "runtime",
    ) -> Any:
        if reference_type not in {"prompt", "resource_template"}:
            raise ValidationError("MCP completion reference_type is invalid")
        self._validate_identifier(
            reference_id,
            "reference_id",
            self.config.mcp.tool_id_max_chars,
        )
        selected_argument = self._modern_string_mapping(
            argument,
            label="Completion argument",
            required=True,
        )
        selected_context = (
            None
            if context is None
            else self._modern_string_mapping(
                context,
                label="Completion context",
                required=True,
            )
        )
        payload = {
            "method": "completion/complete",
            "server_id": server_id,
            "reference_type": reference_type,
            "reference_id": reference_id,
            "argument": selected_argument,
            "context": selected_context,
        }
        return self._run_modern_read(
            operation="completion.complete",
            server_id=server_id,
            logical_id=f"{reference_type}:{reference_id}",
            actor=actor,
            target=f"mcp:{server_id}:completion:{reference_type}:{reference_id}",
            right=CapabilityRight.READ,
            payload=payload,
            invoke=lambda client, deadline: client.complete_prompt(
                server_id,
                reference_type,
                reference_id,
                selected_argument,
                context=selected_context,
                deadline=deadline,
                owner_id=actor,
            ),
        )

    async def acomplete_prompt(
        self,
        server_id: str,
        reference_type: str,
        reference_id: str,
        argument: Mapping[str, str],
        *,
        context: Mapping[str, str] | None = None,
        actor: str = "runtime",
    ) -> Any:
        return await self._data_flow().run_sync_in_worker(
            self.complete_prompt,
            server_id,
            reference_type,
            reference_id,
            argument,
            context=context,
            actor=actor,
        )

    def start_subscription(
        self,
        server_id: str,
        *,
        filters: tuple[str, ...],
        actor: str = "runtime",
    ) -> McpSubscription:
        selected_filters = self._validate_subscription_filters(filters)
        target = f"mcp:{server_id}:subscription:catalog"
        payload = {
            "method": "subscriptions/listen",
            "server_id": server_id,
            "filters": list(selected_filters),
        }

        def invoke(_client: Any, deadline: float) -> _McpPreparedSubscriptionResult:
            return self._prepare_subscription_start_result(
                selected_filters,
                deadline=deadline,
            )

        def settle_result(
            result: Any,
            effect_id: str,
        ) -> tuple[McpSubscription, _McpSubscriptionEffectSettlement]:
            if type(result) is not _McpPreparedSubscriptionResult:
                raise ValidationError("MCP subscription prepared result changed")
            try:
                opening = result.settlement.opening
                context = self._modern_dispatch_context.get()
                binding = context.get("binding") if isinstance(context, dict) else None
                if (
                    opening.origin_effect_id != effect_id
                    or not isinstance(context, dict)
                    or context.get("effect_id") != effect_id
                    or not isinstance(binding, McpClientBinding)
                    or binding is not result.effect_settlement._binding
                ):
                    raise ValidationError("MCP subscription settlement binding changed")
                return result.public, result.effect_settlement
            except BaseException as error:
                try:
                    result.effect_settlement.abort(
                        reason="protected_result_settlement_failed"
                    )
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "MCP subscription result settlement cleanup failed",
                        [error, cleanup_error],
                    )
                raise

        return self._run_modern_read(
            operation="subscriptions.start",
            server_id=server_id,
            logical_id="catalog",
            actor=actor,
            target=target,
            right=CapabilityRight.WRITE,
            payload=payload,
            invoke=invoke,
            state_mutation=True,
            rollback_class=ExternalEffectRollbackClass.ROLLBACKABLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_APPLIED,
            result_settler=settle_result,
        )

    async def astart_subscription(
        self,
        server_id: str,
        *,
        filters: tuple[str, ...],
        actor: str = "runtime",
    ) -> McpSubscription:
        return await self._data_flow().run_sync_in_worker(
            self.start_subscription,
            server_id,
            filters=filters,
            actor=actor,
        )

    def subscription_status(
        self,
        subscription_id: str,
        *,
        actor: str = "runtime",
    ) -> McpSubscription:
        record, target = self._subscription_record_for_operation(
            subscription_id,
            actor=actor,
            right=CapabilityRight.READ,
            operation="subscriptions.status",
        )
        payload = {
            "method": "subscriptions/status",
            "subscription_id": subscription_id,
        }

        def invoke(_client: Any, deadline: float) -> McpSubscription:
            binding, _server = self._subscription_dispatch_binding()
            self._require_subscription_record_binding(record, binding, actor=actor)
            manager = self._subscription_manager()
            result = self._subscription_runner(manager).run(
                lambda: manager.status(subscription_id),
                deadline=deadline,
                binding=binding,
            )
            if (
                not isinstance(result, McpSubscription)
                or result.subscription_id != subscription_id
                or result.server_id != record.server_id
            ):
                raise ValidationError(
                    "MCP subscription manager returned an invalid status"
                )
            return result

        return self._run_modern_read(
            operation="subscriptions.status",
            server_id=record.server_id,
            logical_id=subscription_id,
            actor=actor,
            target=target,
            right=CapabilityRight.READ,
            payload=payload,
            invoke=invoke,
        )

    async def asubscription_status(
        self,
        subscription_id: str,
        *,
        actor: str = "runtime",
    ) -> McpSubscription:
        return await self._data_flow().run_sync_in_worker(
            self.subscription_status,
            subscription_id,
            actor=actor,
        )

    def subscription_events(
        self,
        subscription_id: str,
        *,
        after: int = 0,
        limit: int = 100,
        actor: str = "runtime",
    ) -> tuple[McpSubscriptionEvent, ...]:
        """Consume a batch; ``after`` must equal this owner's last sequence."""

        if (
            type(after) is not int
            or after < 0
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise ValidationError("invalid MCP subscription event window")
        record, target = self._subscription_record_for_operation(
            subscription_id,
            actor=actor,
            right=CapabilityRight.READ,
            operation="subscriptions.events",
        )
        payload = {
            "method": "subscriptions/events",
            "subscription_id": subscription_id,
            "after": after,
            "limit": limit,
        }

        def invoke(_client: Any, deadline: float) -> tuple[McpSubscriptionEvent, ...]:
            binding, _server = self._subscription_dispatch_binding()
            self._require_subscription_record_binding(record, binding, actor=actor)
            manager = self._subscription_manager()
            try:
                result = self._subscription_runner(manager).run(
                    lambda: manager.events(
                        subscription_id,
                        after=after,
                        limit=limit,
                    ),
                    deadline=deadline,
                    binding=binding,
                )
            except KeyError as error:
                if error.args != (subscription_id,):
                    raise
                # Event payloads are intentionally memory-only.  A durable
                # LOST record after restart proves the local handle existed,
                # but it must not be projected as an empty event history and
                # the manager's raw mapping key must not escape the facade.
                raise NotFound(
                    f"MCP subscription events unavailable: {subscription_id}"
                ) from None
            if type(result) is not tuple or any(
                not isinstance(item, McpSubscriptionEvent) for item in result
            ):
                raise ValidationError(
                    "MCP subscription manager returned invalid events"
                )
            return result

        return self._run_modern_read(
            operation="subscriptions.events",
            server_id=record.server_id,
            logical_id=subscription_id,
            actor=actor,
            target=target,
            right=CapabilityRight.READ,
            payload=payload,
            invoke=invoke,
        )

    async def asubscription_events(
        self,
        subscription_id: str,
        *,
        after: int = 0,
        limit: int = 100,
        actor: str = "runtime",
    ) -> tuple[McpSubscriptionEvent, ...]:
        return await self._data_flow().run_sync_in_worker(
            self.subscription_events,
            subscription_id,
            after=after,
            limit=limit,
            actor=actor,
        )

    def stop_subscription(
        self,
        subscription_id: str,
        *,
        actor: str = "runtime",
    ) -> McpSubscription:
        record, target = self._subscription_record_for_operation(
            subscription_id,
            actor=actor,
            right=CapabilityRight.WRITE,
            operation="subscriptions.stop",
        )
        payload = {
            "method": "subscriptions/stop",
            "subscription_id": subscription_id,
        }

        def invoke(_client: Any, deadline: float) -> McpSubscription:
            binding, _server = self._subscription_dispatch_binding()
            self._require_subscription_record_binding(record, binding, actor=actor)
            manager = self._subscription_manager()
            result = self._subscription_runner(manager).run(
                lambda: manager.stop(subscription_id),
                deadline=deadline,
                binding=binding,
            )
            if (
                not isinstance(result, McpSubscription)
                or result.subscription_id != subscription_id
                or result.server_id != record.server_id
            ):
                raise ValidationError(
                    "MCP subscription manager returned an invalid stop result"
                )
            return result

        return self._run_modern_read(
            operation="subscriptions.stop",
            server_id=record.server_id,
            logical_id=subscription_id,
            actor=actor,
            target=target,
            right=CapabilityRight.WRITE,
            payload=payload,
            invoke=invoke,
            state_mutation=True,
            rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        )

    async def astop_subscription(
        self,
        subscription_id: str,
        *,
        actor: str = "runtime",
    ) -> McpSubscription:
        return await self._data_flow().run_sync_in_worker(
            self.stop_subscription,
            subscription_id,
            actor=actor,
        )

    def get_continuation(
        self,
        continuation_id: str,
        *,
        actor: str = "runtime",
    ) -> McpInputRequired:
        """Inspect a durable Elicitation round without provider dispatch."""

        self._validate_durable_host_actor(actor)
        manager = self._continuation_manager()
        binding = manager.binding_material(continuation_id)
        result = manager.get(continuation_id, binding=binding)
        if not isinstance(result, McpInputRequired):
            raise ValidationError("MCP continuation manager returned an invalid view")
        return result

    async def aget_continuation(
        self,
        continuation_id: str,
        *,
        actor: str = "runtime",
    ) -> McpInputRequired:
        return await self._data_flow().run_sync_in_worker(
            self.get_continuation,
            continuation_id,
            actor=actor,
        )

    def recover_durable_result(
        self,
        effect_id: str,
        *,
        actor: str = "runtime",
    ) -> McpInputRequired | McpRemoteTask:
        """Recover one exact local ref from a committed MCP effect receipt.

        This Host-only lookup is deliberately not a Tasks/continuations list.
        It accepts the pending-first local effect id and returns only the safe
        durable projection atomically published with that effect.
        """

        self._validate_durable_host_actor(actor)
        if type(effect_id) is not str or not effect_id or len(effect_id) > 512:
            raise ValidationError("MCP durable result effect id is invalid")
        effect = self.unit_of_work.evidence.get_external_effect(effect_id)
        if (
            effect is None
            or effect.provider != "mcp"
            or effect.effect_state != "finalized"
            or effect.transaction_state != "committed"
        ):
            raise NotFound("MCP durable result receipt was not found")
        selected = effect.provider_receipt.get("mcp_durable_result")
        if not isinstance(selected, Mapping):
            raise NotFound("MCP durable result receipt was not found")
        kind = selected.get("kind")
        if kind == "input_required" and set(selected) == {
            "kind",
            "continuation_id",
        }:
            continuation_id = selected.get("continuation_id")
            if type(continuation_id) is not str:
                raise ValidationError("MCP continuation receipt is invalid")
            manager = self._continuation_manager()
            if not manager.accepts_recovery_effect(continuation_id, effect_id):
                raise CapabilityDenied("MCP continuation effect receipt changed")
            recovered = manager.recover_local_result(continuation_id)
            if isinstance(recovered, McpInputRequired):
                return recovered
            task_ref, response_effect_id = manager.completed_remote_task_handoff(
                continuation_id
            )
            if recovered != task_ref:
                raise ValidationError("MCP continuation Task receipt changed")
            task_manager = self._remote_task_manager()
            task_binding = self._remote_task_binding(task_manager, task_ref)
            if task_binding.origin_effect_id != response_effect_id:
                raise CapabilityDenied("MCP continuation Task effect receipt changed")
            self._require_remote_task_effect_receipt(response_effect_id, task_ref)
            return task_manager.inspect(task_ref, binding=task_binding)
        if kind == "remote_task" and set(selected) == {"kind", "task_ref"}:
            task_ref = selected.get("task_ref")
            if type(task_ref) is not str:
                raise ValidationError("MCP remote Task receipt is invalid")
            manager = self._remote_task_manager()
            binding = self._remote_task_binding(manager, task_ref)
            if binding.origin_effect_id != effect_id:
                raise CapabilityDenied("MCP remote Task effect receipt changed")
            self._require_remote_task_effect_receipt(effect_id, task_ref)
            return manager.inspect(task_ref, binding=binding)
        raise ValidationError("MCP durable result receipt is invalid")

    def _require_remote_task_effect_receipt(
        self,
        effect_id: str,
        task_ref: str,
    ) -> None:
        effect = self.unit_of_work.evidence.get_external_effect(effect_id)
        selected = (
            effect.provider_receipt.get("mcp_durable_result")
            if effect is not None
            else None
        )
        if (
            effect is None
            or effect.provider != "mcp"
            or effect.effect_state != "finalized"
            or effect.transaction_state != "committed"
            or not isinstance(selected, Mapping)
            or dict(selected) != {"kind": "remote_task", "task_ref": task_ref}
        ):
            raise CapabilityDenied("MCP remote Task effect receipt changed")

    async def arecover_durable_result(
        self,
        effect_id: str,
        *,
        actor: str = "runtime",
    ) -> McpInputRequired | McpRemoteTask:
        return await self._data_flow().run_sync_in_worker(
            self.recover_durable_result,
            effect_id,
            actor=actor,
        )

    def respond_continuation(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        responses: dict[str, Any],
        human_request_id: str,
        human_expected_revision: int,
        human_preview_sha256: str,
        actor: str = "runtime",
    ) -> McpComplete[Any] | McpInputRequired | McpRemoteTask:
        """Settle one real Human answer, then dispatch the dedicated MRTR path."""

        self._validate_durable_host_actor(actor)
        selected_responses = _canonical_mcp_arguments(
            responses,
            max_bytes=self.config.mcp.max_request_hard_limit_bytes,
        )
        deadline = self._durable_mcp_deadline()
        manager = self._continuation_manager()
        binding = manager.binding_material(continuation_id)
        pending = manager.get(continuation_id, binding=binding)
        self._require_durable_human_fence(
            pending,
            expected_revision=expected_revision,
            human_request_id=human_request_id,
            human_expected_revision=human_expected_revision,
            human_preview_sha256=human_preview_sha256,
        )
        manager.prevalidate_response(
            continuation_id,
            expected_revision=expected_revision,
            binding=binding,
            human_request_id=human_request_id,
            human_expected_revision=human_expected_revision,
            human_preview_sha256=human_preview_sha256,
            responses=selected_responses,
        )
        manager.human_requests.settle_answer(
            human_request_id,
            selected_responses,
            expected_revision=human_expected_revision,
            preview_sha256=human_preview_sha256,
            responder=actor,
        )
        return asyncio.run(
            manager.respond(
                continuation_id,
                expected_revision=expected_revision,
                binding=binding,
                human_request_id=human_request_id,
                human_expected_revision=human_expected_revision,
                human_preview_sha256=human_preview_sha256,
                deadline=deadline,
            )
        )

    async def arespond_continuation(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        responses: dict[str, Any],
        human_request_id: str,
        human_expected_revision: int,
        human_preview_sha256: str,
        actor: str = "runtime",
    ) -> McpComplete[Any] | McpInputRequired | McpRemoteTask:
        return await self._data_flow().run_sync_in_worker(
            self.respond_continuation,
            continuation_id,
            expected_revision=expected_revision,
            responses=responses,
            human_request_id=human_request_id,
            human_expected_revision=human_expected_revision,
            human_preview_sha256=human_preview_sha256,
            actor=actor,
        )

    def cancel_continuation(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        actor: str = "runtime",
    ) -> McpComplete[None]:
        self._validate_durable_host_actor(actor)
        manager = self._continuation_manager()
        binding = manager.binding_material(continuation_id)
        result = asyncio.run(
            manager.cancel(
                continuation_id,
                expected_revision=expected_revision,
                binding=binding,
                deadline=self._durable_mcp_deadline(),
            )
        )
        if not isinstance(result, McpComplete) or result.value is not None:
            raise ValidationError("MCP continuation cancel result is invalid")
        return result

    async def acancel_continuation(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        actor: str = "runtime",
    ) -> McpComplete[None]:
        return await self._data_flow().run_sync_in_worker(
            self.cancel_continuation,
            continuation_id,
            expected_revision=expected_revision,
            actor=actor,
        )

    def get_remote_task(
        self,
        task_ref: str,
        *,
        expected_revision: int | None = None,
        actor: str = "runtime",
    ) -> McpRemoteTask:
        """Explicitly re-observe one local Task ref; never list or auto-poll."""

        self._validate_durable_host_actor(actor)
        manager = self._remote_task_manager()
        binding = self._remote_task_binding(manager, task_ref)
        local = manager.inspect(task_ref, binding=binding)
        selected_revision = local.revision if expected_revision is None else expected_revision
        return asyncio.run(
            manager.get(
                task_ref,
                expected_revision=selected_revision,
                binding=binding,
                deadline=self._durable_mcp_deadline(),
            )
        )

    async def aget_remote_task(
        self,
        task_ref: str,
        *,
        expected_revision: int | None = None,
        actor: str = "runtime",
    ) -> McpRemoteTask:
        return await self._data_flow().run_sync_in_worker(
            self.get_remote_task,
            task_ref,
            expected_revision=expected_revision,
            actor=actor,
        )

    def update_remote_task(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        responses: dict[str, Any],
        human_request_id: str,
        human_expected_revision: int,
        human_preview_sha256: str,
        actor: str = "runtime",
    ) -> McpRemoteTask:
        self._validate_durable_host_actor(actor)
        selected_responses = _canonical_mcp_arguments(
            responses,
            max_bytes=self.config.mcp.max_request_hard_limit_bytes,
        )
        deadline = self._durable_mcp_deadline()
        manager = self._remote_task_manager()
        binding = self._remote_task_binding(manager, task_ref)
        pending = manager.inspect(task_ref, binding=binding)
        self._require_durable_human_fence(
            pending,
            expected_revision=expected_revision,
            human_request_id=human_request_id,
            human_expected_revision=human_expected_revision,
            human_preview_sha256=human_preview_sha256,
        )
        manager.prevalidate_update(
            task_ref,
            expected_revision=expected_revision,
            binding=binding,
            human_request_id=human_request_id,
            human_expected_revision=human_expected_revision,
            human_preview_sha256=human_preview_sha256,
            responses=selected_responses,
        )
        manager.human_requests.settle_answer(
            human_request_id,
            selected_responses,
            expected_revision=human_expected_revision,
            preview_sha256=human_preview_sha256,
            responder=actor,
        )
        return asyncio.run(
            manager.update(
                task_ref,
                expected_revision=expected_revision,
                binding=binding,
                human_request_id=human_request_id,
                human_expected_revision=human_expected_revision,
                human_preview_sha256=human_preview_sha256,
                deadline=deadline,
            )
        )

    async def aupdate_remote_task(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        responses: dict[str, Any],
        human_request_id: str,
        human_expected_revision: int,
        human_preview_sha256: str,
        actor: str = "runtime",
    ) -> McpRemoteTask:
        return await self._data_flow().run_sync_in_worker(
            self.update_remote_task,
            task_ref,
            expected_revision=expected_revision,
            responses=responses,
            human_request_id=human_request_id,
            human_expected_revision=human_expected_revision,
            human_preview_sha256=human_preview_sha256,
            actor=actor,
        )

    def cancel_remote_task(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        actor: str = "runtime",
    ) -> McpRemoteTask:
        self._validate_durable_host_actor(actor)
        manager = self._remote_task_manager()
        binding = self._remote_task_binding(manager, task_ref)
        return asyncio.run(
            manager.cancel(
                task_ref,
                expected_revision=expected_revision,
                binding=binding,
                deadline=self._durable_mcp_deadline(),
            )
        )

    async def acancel_remote_task(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        actor: str = "runtime",
    ) -> McpRemoteTask:
        return await self._data_flow().run_sync_in_worker(
            self.cancel_remote_task,
            task_ref,
            expected_revision=expected_revision,
            actor=actor,
        )

    def add_oauth_profile(
        self,
        profile: McpOAuthProfile,
        *,
        client_secret: bytes | None = None,
        actor: str = "runtime",
    ) -> McpOAuthStatus:
        """Add Host OAuth configuration without projecting its secret input."""

        self._validate_oauth_actor(actor)
        self._require_oauth_enabled()
        if not isinstance(profile, McpOAuthProfile):
            raise ValidationError("MCP OAuth profile must be a typed McpOAuthProfile")
        manager = self._oauth_manager()
        with self._oauth_phase_lock:
            status = manager.add_profile(profile, client_secret=client_secret)
            try:
                found = self.extensions.get_mcp_v3_server(profile.server_id)
                if found is not None:
                    manifest, _metadata = found
                    binding = self._validate_oauth_profile_manifest(profile, manifest)
                    self._restore_oauth_generation(manager, profile, binding)
                    status = manager.status(profile.profile_id)
                    with self.unit_of_work.transaction():
                        self._sync_oauth_metadata(
                            manager,
                            profile,
                            manifest,
                            binding,
                            status,
                        )
                        self._record_oauth_profile_change(
                            "add", profile, status, actor=actor, bound=True
                        )
                else:
                    # A provisional profile lets a Host atomically admit a
                    # manifest that references it.  It cannot begin OAuth or
                    # produce durable authority until that exact v3 server is
                    # registered and bound below in _register_server.
                    with self.unit_of_work.transaction():
                        self._record_oauth_profile_change(
                            "add", profile, status, actor=actor, bound=False
                        )
            except BaseException:
                manager.remove_profile(profile.profile_id, missing_ok=True)
                raise
            return status

    def replace_oauth_profile(
        self,
        profile: McpOAuthProfile,
        *,
        client_secret: bytes | None = None,
        actor: str = "runtime",
    ) -> McpOAuthStatus:
        """Replace one exact Host profile and revoke its prior local fence."""

        self._validate_oauth_actor(actor)
        self._require_oauth_enabled()
        if not isinstance(profile, McpOAuthProfile):
            raise ValidationError("MCP OAuth profile must be a typed McpOAuthProfile")
        manager = self._oauth_manager()
        with self._oauth_phase_lock:
            if not manager.has_profile(profile.profile_id):
                raise ValidationError("MCP OAuth profile is unavailable")
            found = self.extensions.get_mcp_v3_server(profile.server_id)
            manifest: McpServerManifestV3 | None = None
            binding: dict[str, Any] | None = None
            if found is not None:
                selected, _metadata = found
                binding = self._validate_oauth_profile_manifest(profile, selected)
                manifest = selected
                self._require_oauth_record_identity(profile, binding)
            status = manager.replace_profile(profile, client_secret=client_secret)
            if manifest is not None and binding is not None:
                with self.unit_of_work.transaction():
                    status = manager.status(profile.profile_id)
                    self._sync_oauth_metadata(
                        manager,
                        profile,
                        manifest,
                        binding,
                        status,
                    )
                    self._record_oauth_profile_change(
                        "replace", profile, status, actor=actor, bound=True
                    )
            else:
                with self.unit_of_work.transaction():
                    self._record_oauth_profile_change(
                        "replace", profile, status, actor=actor, bound=False
                    )
            return status

    def remove_oauth_profile(
        self,
        profile_id: str,
        *,
        actor: str = "runtime",
    ) -> McpOAuthStatus:
        """Remove broker handles; retain only a revoked non-secret Store row."""

        self._validate_oauth_actor(actor)
        manager = self._oauth_manager()
        with self._oauth_phase_lock:
            profile = manager.profile_snapshot(profile_id)
            found = self.extensions.get_mcp_v3_server(profile.server_id)
            manifest: McpServerManifestV3 | None = None
            binding: dict[str, Any] | None = None
            if found is not None:
                selected, _metadata = found
                binding = self._validate_oauth_profile_manifest(profile, selected)
                manifest = selected
            manager.remove_profile(profile_id)
            status = McpOAuthStatus(
                profile_id=profile_id,
                status=McpOAuthStatusKind.REVOKED,
                issuer=profile.expected_issuer,
                resource=profile.resource_uri,
            )
            with self.unit_of_work.transaction():
                if manifest is not None and binding is not None:
                    self._sync_oauth_metadata(
                        manager=None,
                        profile=profile,
                        manifest=manifest,
                        binding=binding,
                        status=status,
                    )
                self._record_oauth_profile_change(
                    "remove", profile, status, actor=actor, bound=manifest is not None
                )
            return status

    def list_oauth_profiles(self, *, actor: str = "runtime") -> tuple[McpOAuthStatus, ...]:
        self._validate_oauth_actor(actor)
        manager = self._oauth_manager()
        return manager.list_profiles()

    def auth_status(
        self,
        profile_id: str,
        *,
        actor: str = "runtime",
    ) -> McpOAuthStatus:
        """Return non-secret status without implicitly reconfiguring a profile.

        After restart the Host must explicitly add the same exact profile; only
        that registration may rebind a deterministic secure-broker token slot.
        A status read by itself never loads or revives credential material.
        """

        self._validate_oauth_actor(actor)
        manager = self._oauth_manager()
        with self._oauth_phase_lock:
            if not manager.has_profile(profile_id):
                return self._restarted_oauth_status(profile_id)
            profile, manifest, binding = self._bound_oauth_context(profile_id)
            status = manager.status(profile_id)
            with self.unit_of_work.transaction():
                self._sync_oauth_metadata(
                    manager, profile, manifest, binding, status
                )
            return status

    def auth_begin(
        self,
        profile_id: str,
        *,
        scopes: tuple[str, ...] = (),
        actor: str = "runtime",
    ) -> McpAuthorizationChallenge:
        self._validate_oauth_actor(actor)
        if type(scopes) is not tuple or any(type(item) is not str for item in scopes):
            raise ValidationError("MCP OAuth scopes must be a tuple of strings")
        with self._oauth_phase_lock:
            profile, manifest, binding = self._bound_oauth_context(profile_id)
            manager = self._oauth_manager()
            issued_challenge: McpAuthorizationChallenge | None = None

            def invoke(deadline: float) -> McpAuthorizationChallenge:
                nonlocal issued_challenge
                challenge = manager.begin(profile_id, scopes=scopes, deadline=deadline)
                issued_challenge = challenge
                self._sync_oauth_metadata(
                    manager,
                    profile,
                    manifest,
                    binding,
                    manager.status(profile_id),
                )
                return challenge

            try:
                return self._run_oauth_provider_operation(
                    operation="auth.begin",
                    profile=profile,
                    manifest=manifest,
                    binding=binding,
                    actor=actor,
                    payload={
                        "profile_id": profile_id,
                        "scopes": list(sorted(scopes)),
                    },
                    mutation=False,
                    invoke=invoke,
                )
            except BaseException:
                self._discard_oauth_challenge_quietly(manager, issued_challenge)
                raise

    def auth_authorize_for_challenge(
        self,
        profile_id: str,
        www_authenticate: str,
        *,
        actor: str = "runtime",
    ) -> McpAuthorizationChallenge:
        self._validate_oauth_actor(actor)
        if type(www_authenticate) is not str or not www_authenticate:
            raise ValidationError("MCP OAuth challenge header is invalid")
        with self._oauth_phase_lock:
            profile, manifest, binding = self._bound_oauth_context(profile_id)
            manager = self._oauth_manager()
            header_sha256 = hashlib.sha256(www_authenticate.encode("utf-8")).hexdigest()
            issued_challenge: McpAuthorizationChallenge | None = None

            def invoke(deadline: float) -> McpAuthorizationChallenge:
                nonlocal issued_challenge
                challenge = manager.authorize_for_challenge(
                    profile_id,
                    www_authenticate,
                    deadline=deadline,
                )
                issued_challenge = challenge
                self._sync_oauth_metadata(
                    manager,
                    profile,
                    manifest,
                    binding,
                    manager.status(profile_id),
                )
                return challenge

            try:
                return self._run_oauth_provider_operation(
                    operation="auth.challenge",
                    profile=profile,
                    manifest=manifest,
                    binding=binding,
                    actor=actor,
                    payload={
                        "profile_id": profile_id,
                        "www_authenticate_sha256": header_sha256,
                    },
                    mutation=False,
                    invoke=invoke,
                )
            except BaseException:
                self._discard_oauth_challenge_quietly(manager, issued_challenge)
                raise

    def auth_complete(
        self,
        challenge_id: str,
        callback_url: str,
        *,
        actor: str = "runtime",
    ) -> McpOAuthStatus:
        self._validate_oauth_actor(actor)
        manager = self._oauth_manager()
        with self._oauth_phase_lock:
            profile_id = manager.challenge_profile_id(challenge_id)
            profile, manifest, binding = self._bound_oauth_context(profile_id)

            def invoke(deadline: float) -> McpOAuthStatus:
                try:
                    status = manager.complete(
                        challenge_id,
                        callback_url,
                        deadline=deadline,
                    )
                except Exception:
                    self._sync_oauth_failure_status(
                        manager, profile, manifest, binding
                    )
                    raise
                self._sync_oauth_metadata(
                    manager, profile, manifest, binding, status
                )
                return status

            return self._run_oauth_provider_operation(
                operation="auth.complete",
                profile=profile,
                manifest=manifest,
                binding=binding,
                actor=actor,
                payload={
                    "profile_id": profile_id,
                    "challenge_sha256": hashlib.sha256(
                        challenge_id.encode("utf-8")
                    ).hexdigest(),
                },
                mutation=True,
                invoke=invoke,
            )

    def auth_revoke(
        self,
        profile_id: str,
        *,
        actor: str = "runtime",
    ) -> McpOAuthStatus:
        self._validate_oauth_actor(actor)
        manager = self._oauth_manager()
        with self._oauth_phase_lock:
            profile, manifest, binding = self._bound_oauth_context(profile_id)

            def invoke(deadline: float) -> McpOAuthStatus:
                try:
                    status = manager.revoke(profile_id, deadline=deadline)
                except Exception:
                    self._sync_oauth_failure_status(
                        manager, profile, manifest, binding
                    )
                    raise
                self._sync_oauth_metadata(
                    manager, profile, manifest, binding, status
                )
                return status

            return self._run_oauth_provider_operation(
                operation="auth.revoke",
                profile=profile,
                manifest=manifest,
                binding=binding,
                actor=actor,
                payload={"profile_id": profile_id},
                mutation=True,
                invoke=invoke,
            )

    def auth_logout(
        self,
        profile_id: str,
        *,
        actor: str = "runtime",
    ) -> McpOAuthStatus:
        self._validate_oauth_actor(actor)
        manager = self._oauth_manager()
        with self._oauth_phase_lock:
            profile, manifest, binding = self._bound_oauth_context(profile_id)
            status = manager.logout(profile_id)
            with self.unit_of_work.transaction():
                self._sync_oauth_metadata(
                    manager, profile, manifest, binding, status
                )
                self._record_oauth_profile_change(
                    "logout", profile, status, actor=actor, bound=True
                )
            return status

    def _oauth_manager(self) -> Any:
        manager = self._modern_oauth
        required = (
            "add_profile",
            "replace_profile",
            "remove_profile",
            "profile_snapshot",
            "status",
            "begin",
            "complete",
            "revoke",
            "logout",
            "credential_generation",
        )
        if manager is None or any(
            not callable(getattr(manager, name, None)) for name in required
        ):
            raise ValidationError("MCP OAuth manager is unavailable")
        return manager

    def _require_oauth_enabled(self) -> None:
        if self.config.mcp.oauth_enabled is not True:
            raise ValidationError("MCP OAuth is disabled by Host policy")

    @staticmethod
    def _validate_oauth_actor(actor: str) -> None:
        if type(actor) is not str or not actor or len(actor) > 512 or "\x00" in actor:
            raise ValidationError("MCP OAuth actor is invalid")

    def _validate_oauth_profile_manifest(
        self,
        profile: McpOAuthProfile,
        manifest: Any,
    ) -> dict[str, Any]:
        self._validate_oauth_profile_manifest_fields(profile, manifest)
        return self._registry_binding_for_server_spec(manifest)

    def _validate_oauth_profile_manifest_fields(
        self,
        profile: McpOAuthProfile,
        manifest: Any,
    ) -> None:
        self._require_oauth_enabled()
        if not isinstance(manifest, McpServerManifestV3):
            raise ValidationError("MCP OAuth requires a Manifest v3 server")
        if (
            manifest.transport != "streamable_http"
            or manifest.http is None
            or manifest.auth_profile_id != profile.profile_id
            or manifest.server_id != profile.server_id
            or manifest.http.url != profile.resource_uri
        ):
            raise ValidationError(
                "MCP OAuth profile does not match the exact v3 server binding"
            )

    def _bound_oauth_context(
        self,
        profile_id: str,
    ) -> tuple[McpOAuthProfile, McpServerManifestV3, dict[str, Any]]:
        if type(profile_id) is not str or not profile_id or "\x00" in profile_id:
            raise ValidationError("MCP OAuth profile_id is invalid")
        manager = self._oauth_manager()
        profile = manager.profile_snapshot(profile_id)
        found = self.extensions.get_mcp_v3_server(profile.server_id)
        if found is None:
            raise ValidationError("MCP OAuth profile is not bound to a registered server")
        manifest, _metadata = found
        binding = self._validate_oauth_profile_manifest(profile, manifest)
        self._require_oauth_record_identity(profile, binding)
        return profile, manifest, binding

    def _restore_oauth_generation(
        self,
        manager: Any,
        profile: McpOAuthProfile,
        binding: Mapping[str, Any],
    ) -> None:
        existing = self.unit_of_work.mcp_auth.get(profile.profile_id)
        if existing is None:
            return
        self._require_oauth_record_identity(profile, binding, record=existing)
        manager.set_minimum_credential_generation(
            profile.profile_id,
            existing.credential_generation,
        )

    def _require_oauth_record_identity(
        self,
        profile: McpOAuthProfile,
        binding: Mapping[str, Any],
        *,
        record: McpAuthMetadataRecord | None = None,
    ) -> None:
        selected = record or self.unit_of_work.mcp_auth.get(profile.profile_id)
        if selected is None:
            return
        if (
            selected.server_id != profile.server_id
            or selected.server_spec_sha256 != binding.get("registry_spec_sha256")
        ):
            raise ValidationError(
                "MCP OAuth profile binding changed; configure a new profile_id"
            )

    @staticmethod
    def _oauth_digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _oauth_scope_digest(scopes: tuple[str, ...]) -> str:
        return hashlib.sha256(
            dumps(list(sorted(scopes))).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _oauth_reason_code(status: McpOAuthStatusKind) -> str | None:
        return {
            McpOAuthStatusKind.UNCONFIGURED: "credential_missing",
            McpOAuthStatusKind.AUTHORIZATION_REQUIRED: "authorization_required",
            McpOAuthStatusKind.AUTHORIZED: None,
            McpOAuthStatusKind.EXPIRED: "credential_expired",
            McpOAuthStatusKind.REVOKED: "credential_revoked",
            McpOAuthStatusKind.NEEDS_ATTENTION: "needs_attention",
        }[status]

    def _sync_oauth_metadata(
        self,
        manager: Any | None,
        profile: McpOAuthProfile,
        manifest: McpServerManifestV3,
        binding: Mapping[str, Any],
        status: McpOAuthStatus,
    ) -> McpAuthMetadataRecord:
        if status.profile_id != profile.profile_id:
            raise ValidationError("MCP OAuth status belongs to another profile")
        self._validate_oauth_profile_manifest(profile, manifest)
        repository = self.unit_of_work.mcp_auth
        existing = repository.get(profile.profile_id)
        self._require_oauth_record_identity(
            profile,
            binding,
            record=existing,
        )
        current_generation = (
            manager.credential_generation(profile.profile_id)
            if manager is not None
            else ((existing.credential_generation + 1) if existing is not None else 0)
        )
        if (
            existing is not None
            and current_generation < existing.credential_generation
        ):
            raise ValidationError("MCP OAuth credential generation regressed")
        now = utc_now()
        reason = self._oauth_reason_code(status.status)
        metadata = {} if reason is None else {"reason_code": reason}
        record = McpAuthMetadataRecord(
            profile_id=profile.profile_id,
            server_id=profile.server_id,
            server_spec_sha256=(
                existing.server_spec_sha256
                if existing is not None
                else str(binding["registry_spec_sha256"])
            ),
            server_generation=(
                existing.server_generation
                if existing is not None
                else int(binding["registry_generation"])
            ),
            status=status.status.value,
            issuer_sha256=self._oauth_digest(profile.expected_issuer),
            resource_sha256=self._oauth_digest(profile.resource_uri),
            audience_sha256=self._oauth_digest(
                profile.audience or profile.resource_uri
            ),
            scopes_sha256=self._oauth_scope_digest(status.scopes),
            principal_sha256=status.principal_sha256,
            expires_at=status.expires_at,
            credential_generation=current_generation,
            revision=0 if existing is None else existing.revision + 1,
            metadata=metadata,
            created_at=now if existing is None else existing.created_at,
            updated_at=now,
        )
        if existing is None:
            return repository.insert(record)
        if not repository.compare_and_swap(
            profile.profile_id,
            expected_revision=existing.revision,
            replacement=record,
        ):
            raise ValidationError("MCP OAuth metadata changed concurrently")
        return record

    def _sync_oauth_failure_status(
        self,
        manager: Any,
        profile: McpOAuthProfile,
        manifest: McpServerManifestV3,
        binding: Mapping[str, Any],
    ) -> None:
        """Persist only a sanitized status after a one-shot OAuth failure."""

        try:
            status = manager.status(profile.profile_id)
            with self.unit_of_work.transaction():
                self._sync_oauth_metadata(
                    manager,
                    profile,
                    manifest,
                    binding,
                    status,
                )
        except BaseException:
            # The original remote mutation is already unknown and must remain
            # the surfaced failure. Store diagnostics are not authority and a
            # projection failure must never tempt a caller to replay it.
            pass

    def _restarted_oauth_status(self, profile_id: str) -> McpOAuthStatus:
        if type(profile_id) is not str or not profile_id or "\x00" in profile_id:
            raise ValidationError("MCP OAuth profile_id is invalid")
        repository = self.unit_of_work.mcp_auth
        existing = repository.get(profile_id)
        if existing is None:
            raise ValidationError("MCP OAuth profile is unavailable")
        selected_status = (
            McpOAuthStatusKind.REVOKED
            if existing.status == McpOAuthStatusKind.REVOKED.value
            else McpOAuthStatusKind.NEEDS_ATTENTION
        )
        if existing.status != selected_status.value:
            now = utc_now()
            replacement = dataclass_replace(
                existing,
                status=selected_status.value,
                revision=existing.revision + 1,
                metadata={
                    "reason_code": (
                        "credential_revoked"
                        if selected_status is McpOAuthStatusKind.REVOKED
                        else "credential_missing"
                    )
                },
                updated_at=now,
            )
            with self.unit_of_work.transaction():
                if not repository.compare_and_swap(
                    profile_id,
                    expected_revision=existing.revision,
                    replacement=replacement,
                ):
                    raise ValidationError("MCP OAuth metadata changed concurrently")
        return McpOAuthStatus(profile_id=profile_id, status=selected_status)

    def _record_oauth_profile_change(
        self,
        action: str,
        profile: McpOAuthProfile,
        status: McpOAuthStatus,
        *,
        actor: str,
        bound: bool,
    ) -> None:
        payload = {
            "adapter": "mcp",
            "operation": f"oauth_profile_{action}",
            "profile_id": profile.profile_id,
            "server_id": profile.server_id,
            "status": status.status.value,
            "bound": bound,
        }
        self.events.emit(
            EventType.EXTERNAL_WRITE,
            source=actor,
            target=f"mcp_oauth_profile:{profile.profile_id}",
            payload=payload,
        )
        self.audit.record(
            actor=actor,
            action=f"mcp.oauth.profile.{action}",
            target=f"mcp_oauth_profile:{profile.profile_id}",
            decision=payload,
        )

    def _run_oauth_provider_operation(
        self,
        *,
        operation: str,
        profile: McpOAuthProfile,
        manifest: McpServerManifestV3,
        binding: dict[str, Any],
        actor: str,
        payload: Mapping[str, Any],
        mutation: bool,
        invoke: Any,
    ) -> Any:
        request_json = dumps(dict(payload))
        request_bytes = len(request_json.encode("utf-8"))
        request_sha256 = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        target = f"mcp:{manifest.server_id}:oauth:{profile.profile_id}"
        context = {
            "server_id": manifest.server_id,
            "logical_id": profile.profile_id,
            "operation": operation,
            "request_sha256": request_sha256,
            "request_bytes": request_bytes,
            **binding,
        }
        rollback_class = (
            ExternalEffectRollbackClass.IRREVERSIBLE
            if mutation
            else ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        )
        rollback_status = (
            ExternalEffectRollbackStatus.NOT_SUPPORTED
            if mutation
            else ExternalEffectRollbackStatus.NOT_REQUIRED
        )
        observation = {
            **context,
            "state_mutation": mutation,
            "information_flow": True,
            "rollback_class": rollback_class.value,
            "rollback_status": rollback_status.value,
            "automatic_retry_disabled": mutation,
        }
        flow_context = DataFlowContext(
            labels=DataLabels(
                sensitivity="public",
                trust_level="verified",
                integrity="verified",
                origin=f"runtime:mcp-oauth:{actor}",
            )
        )
        sink = self._modern_data_sink(
            manifest,
            operation=operation,
            logical_id=profile.profile_id,
            stdio_identity=None,
        )
        invocation = ProtectedOperationInvocation(
            pid=actor,
            actor=actor,
            target=target,
            decisions=(),
            canonical_args=context,
            observation=observation,
            **self._protected_registry_guard(binding, manifest.server_id),
            data_sink=sink,
            data_sink_revalidator=lambda: self._modern_data_sink(
                manifest,
                operation=operation,
                logical_id=profile.profile_id,
                stdio_identity=None,
            ),
            data_flow_context=flow_context,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                flow_context,
                origin="external:mcp-oauth",
            ),
            data_flow_payload=dict(payload),
            data_flow_operation=f"mcp.{operation}",
            failure_evidence=lambda error, phase: self._oauth_operation_evidence(
                actor=actor,
                target=target,
                context=context,
                operation=operation,
                ok=False,
                mutation=mutation,
                phase=phase,
                result_kind=type(error).__name__,
                response_bytes=0,
            ),
        )
        deadline = time.monotonic() + manifest.timeout_s
        contract = f"primitive.mcp.{operation}.internal"
        with self._protected().start(
            contract,
            invocation,
            provider=self.provider,
        ) as protected:
            result = protected.call(
                ProviderPhase(
                    "oauth_provider_operation",
                    state_mutation=mutation,
                    information_flow=True,
                ),
                invoke,
                deadline,
            )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "MCP OAuth provider exceeded the absolute deadline"
                )
            response_bytes = len(dumps(to_jsonable(result)).encode("utf-8"))
            classification = ExternalEffectClassification(
                rollback_class=rollback_class,
                rollback_status=rollback_status,
                state_mutation=mutation,
                information_flow=True,
                metadata={
                    "outcome": "succeeded",
                    "operation": operation,
                    "automatic_retry_disabled": mutation,
                },
            )
            return protected.complete(
                result,
                self._oauth_operation_evidence(
                    actor=actor,
                    target=target,
                    context=context,
                    operation=operation,
                    ok=True,
                    mutation=mutation,
                    phase="complete",
                    result_kind=type(result).__name__,
                    response_bytes=response_bytes,
                ),
                classification_override=classification,
            )

    @staticmethod
    def _discard_oauth_challenge_quietly(
        manager: Any,
        challenge: McpAuthorizationChallenge | None,
    ) -> None:
        if challenge is None:
            return
        try:
            manager.discard_challenge(challenge.challenge_id)
        except BaseException:
            # Preserve the original protected-operation failure. The manager
            # retains an ambiguous deletion for close/expiry/next-begin retry.
            pass

    @staticmethod
    def _oauth_operation_evidence(
        *,
        actor: str,
        target: str,
        context: Mapping[str, Any],
        operation: str,
        ok: bool,
        mutation: bool,
        phase: str,
        result_kind: str,
        response_bytes: int,
    ) -> ProtectedOperationEvidence:
        payload = {
            "adapter": "mcp",
            "operation": operation,
            "server_id": context["server_id"],
            "profile_id": context["logical_id"],
            "ok": ok,
            "phase": phase,
            "result_kind": result_kind,
            "request_bytes": context["request_bytes"],
            "response_bytes": response_bytes,
            "automatic_retry_disabled": mutation,
        }
        return ProtectedOperationEvidence(
            event_type=(EventType.EXTERNAL_WRITE if mutation else EventType.EXTERNAL_READ),
            event_source=actor,
            event_target=target,
            event_payload=payload,
            audit_action=f"primitive.mcp.{operation}",
            audit_actor=actor,
            audit_target=target,
            audit_decision={
                **payload,
                "request_sha256": context["request_sha256"],
                "registry_spec_sha256": context["registry_spec_sha256"],
                "registry_generation": context["registry_generation"],
            },
            effect_metadata=payload,
            provider_receipt={
                "request_bytes": context["request_bytes"],
                "response_bytes": response_bytes,
            },
        )

    @asynccontextmanager
    async def _modern_session_factory_context(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        binding: McpClientBinding,
        task_notification_ingress: Callable[[Mapping[str, Any] | None], None]
        | None = None,
    ):
        """Enter the existing strict SDK transport inside one protected phase."""

        context, base_server = self._validate_modern_session_binding(
            server,
            binding=binding,
            deadline=deadline,
        )
        runtime_environment, snapshot, limits, session = (
            self._prepare_modern_session_dispatch(
                server,
                binding=binding,
                context=context,
                base_server=base_server,
                deadline=deadline,
            )
        )
        try:
            tasks_extension_sha256 = self._modern_session_tasks_pin(binding)
            async with session(
                server,
                deadline=deadline,
                max_response_bytes=server.max_response_bytes,
                executable_snapshot=snapshot,
                runtime_environment=runtime_environment,
                limits=limits,
                allow_server_notifications=(
                    context.get("operation") == "subscriptions.start"
                ),
                enable_modern_mrtr=True,
                tasks_extension_sha256=tasks_extension_sha256,
                request_id_allocator=self._next_modern_request_id,
                task_notification_ingress=task_notification_ingress,
            ) as selected:
                yield selected
        finally:
            if snapshot is not None:
                snapshot.close()

    def _next_modern_request_id(self) -> str:
        """Mint a Runtime-scoped JSON-RPC id for an exact-v3 exchange.

        Modern operations deliberately use independent task-affine SDK
        sessions.  A session-local integer counter would therefore reuse the
        same Tool request id when a durable MRTR continuation is resumed.
        The random Runtime scope also avoids reuse after a Runtime reopen;
        the locked suffix makes concurrent operations deterministic within
        that scope.  Manifest-v1/v2 never receive this allocator and retain
        their released numeric wire identity.
        """

        with self._modern_request_id_lock:
            self._modern_request_id_next += 1
            sequence = self._modern_request_id_next
        return f"{self._modern_request_id_scope}:{sequence}"

    def _modern_session_tasks_pin(
        self,
        binding: McpClientBinding,
    ) -> str | None:
        """Return the current local Tasks review pin for one exact v3 binding."""

        extension = binding.manifest.tasks_extension
        if extension is None:
            return None
        host_pin = self.config.mcp.tasks_extension_spec_sha256
        if (
            not self.config.mcp.tasks_extension_enabled
            or extension.extension_id != MCP_TASKS_EXTENSION_ID
            or type(host_pin) is not str
            or extension.spec_sha256 != host_pin
        ):
            raise ProviderEffectNotStarted(
                "MCP Tasks extension binding changed before provider dispatch"
            )
        return host_pin

    def _validate_modern_session_binding(
        self,
        server: McpServerSpec,
        *,
        binding: McpClientBinding,
        deadline: float,
    ) -> tuple[dict[str, Any], McpServerSpec]:
        context = self._modern_dispatch_context.get()
        if not isinstance(context, dict):
            raise ProviderEffectNotStarted(
                "MCP modern session is outside a protected provider phase"
            )
        if not isinstance(binding, McpClientBinding):
            raise ProviderEffectNotStarted("MCP modern session binding is invalid")
        active_binding = current_mcp_client_binding()
        expected_binding = context.get("binding")
        if (
            not isinstance(expected_binding, McpClientBinding)
            or binding.fence != expected_binding.fence
            or active_binding.fence != binding.fence
            or binding.manifest.server_id != server.server_id
        ):
            raise ProviderEffectNotStarted(
                "MCP modern session binding changed before provider dispatch"
            )
        expected_deadline = context.get("deadline")
        if (
            type(deadline) not in {int, float}
            or type(expected_deadline) not in {int, float}
            or deadline > float(expected_deadline)
        ):
            raise ProviderEffectNotStarted("MCP modern deadline binding is invalid")
        self._remaining_timeout(deadline)
        base_server = context.get("server")
        if not isinstance(base_server, McpServerSpec):
            raise ProviderEffectNotStarted("MCP modern transport binding is invalid")
        if self._modern_transport_identity(base_server) != self._modern_transport_identity(
            server
        ):
            raise ProviderEffectNotStarted(
                "MCP modern transport changed before provider dispatch"
            )
        return context, base_server

    @staticmethod
    def _modern_transport_identity(server: McpServerSpec) -> tuple[Any, ...]:
        return (
            server.server_id,
            server.transport,
            server.max_request_bytes,
            server.max_response_bytes,
            server.protocol_mode,
        )

    def _prepare_modern_session_dispatch(
        self,
        server: McpServerSpec,
        *,
        binding: McpClientBinding,
        context: Mapping[str, Any],
        base_server: McpServerSpec,
        deadline: float,
    ) -> tuple[Mapping[str, str], ExecutableSnapshot | None, SubprocessLimits | None, Any]:
        if base_server.transport == "streamable_http":
            self._validate_runtime_resolution(base_server, deadline=deadline)
        host_environment = binding.runtime_environment
        if not isinstance(host_environment, Mapping):
            raise ProviderEffectNotStarted(
                "MCP modern runtime environment snapshot is unavailable"
            )
        runtime_environment = self._require_runtime_environment(
            server,
            host_environment=host_environment,
        )
        snapshot = self._stdio_snapshot_for_dispatch(
            pid=str(context["actor"]),
            spec=base_server,
            expected_identity=context.get("stdio_identity"),
            sink=context["sink"],
            context=context["flow_context"],
            payload=context["payload"],
            runtime_environment=runtime_environment,
            deadline=deadline,
        )
        usage_pid = context.get("usage_pid")
        limits = self._subprocess_limits(usage_pid) if usage_pid is not None else None
        session = getattr(self.provider, "modern_session", None)
        if not callable(session):
            if snapshot is not None:
                snapshot.close()
            raise ProviderEffectNotStarted(
                "MCP Runtime provider cannot create a governed SDK session"
            )
        return runtime_environment, snapshot, limits, session

    def _run_modern_read(
        self,
        *,
        operation: str,
        server_id: str,
        logical_id: str,
        actor: str,
        target: str,
        right: CapabilityRight,
        payload: Mapping[str, Any],
        invoke: Any,
        manifest_preflight: Any | None = None,
        binding_preflight: Any | None = None,
        expected_registry_binding: Mapping[str, Any] | None = None,
        absolute_deadline: float | None = None,
        state_mutation: bool = False,
        rollback_class: ExternalEffectRollbackClass = (
            ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        ),
        rollback_status: ExternalEffectRollbackStatus = (
            ExternalEffectRollbackStatus.NOT_REQUIRED
        ),
        contract_name: str | None = None,
        result_settler: Any | None = None,
    ) -> Any:
        plan = self._prepare_modern_operation(
            operation=operation,
            server_id=server_id,
            logical_id=logical_id,
            actor=actor,
            target=target,
            right=right,
            payload=payload,
            manifest_preflight=manifest_preflight,
            binding_preflight=binding_preflight,
            expected_registry_binding=expected_registry_binding,
            absolute_deadline=absolute_deadline,
        )
        self._prepare_modern_transport(
            plan,
            operation=operation,
            logical_id=logical_id,
            actor=actor,
        )
        plan["invocation"] = self._modern_protected_invocation(
            plan,
            operation=operation,
            server_id=server_id,
            logical_id=logical_id,
            actor=actor,
            target=target,
            payload=payload,
            state_mutation=state_mutation,
            rollback_class=rollback_class,
            rollback_status=rollback_status,
        )
        return self._execute_modern_operation(
            plan,
            operation=operation,
            server_id=server_id,
            logical_id=logical_id,
            actor=actor,
            target=target,
            payload=payload,
            invoke=invoke,
            state_mutation=state_mutation,
            rollback_class=rollback_class,
            rollback_status=rollback_status,
            contract_name=contract_name,
            result_settler=result_settler,
        )

    def _prepare_modern_operation(
        self,
        *,
        operation: str,
        server_id: str,
        logical_id: str,
        actor: str,
        target: str,
        right: CapabilityRight,
        payload: Mapping[str, Any],
        manifest_preflight: Any | None,
        binding_preflight: Any | None,
        expected_registry_binding: Mapping[str, Any] | None,
        absolute_deadline: float | None,
    ) -> dict[str, Any]:
        self._validate_identifier(
            server_id,
            "server_id",
            self.config.mcp.server_id_max_chars,
        )
        if type(actor) is not str or not actor:
            raise ValidationError("MCP modern operation actor is invalid")
        client = self._modern_client
        if client is None:
            raise ValidationError("MCP modern client is unavailable")
        usage_pid = self._resource_usage_pid(actor)
        visibility = {
            "pid": actor,
            "primitive": f"runtime.mcp.{operation}",
            "operation": f"mcp.{operation}",
            "authority_operation": f"mcp.{operation}",
            "server_id": server_id,
            "logical_id": logical_id,
            # Reuse the released exact-request condition vocabulary.  These
            # aliases bind a modern logical selector and its full canonical
            # request without widening the authority-rule language.
            "tool_id": logical_id,
            "right": right.value,
        }
        if usage_pid is not None:
            self._precheck_modern_authority(actor, target, right, visibility)
            self._precheck_modern_authority(
                actor,
                self.server_resource(server_id),
                CapabilityRight.EXECUTE,
                visibility,
            )

        binding = self._resolve_modern_binding(server_id, owner_id=actor)
        if binding_preflight is not None:
            binding_preflight(binding)
        manifest = binding.manifest
        if not isinstance(manifest, McpServerManifestV3):
            raise ValidationError("MCP modern operation requires Manifest v3")
        if absolute_deadline is not None and (
            type(absolute_deadline) not in {int, float}
            or not math.isfinite(float(absolute_deadline))
        ):
            raise ValidationError("MCP modern absolute deadline is invalid")
        manifest_deadline = time.monotonic() + manifest.timeout_s
        deadline = (
            manifest_deadline
            if absolute_deadline is None
            else min(float(absolute_deadline), manifest_deadline)
        )
        self._remaining_timeout(deadline)
        if manifest_preflight is not None:
            manifest_preflight(manifest, deadline)
        registry_binding = self._registry_binding_for_server_spec(manifest)
        if expected_registry_binding is not None and any(
            registry_binding.get(key) != expected_registry_binding.get(key)
            for key in ("registry_spec_sha256", "registry_generation")
        ):
            raise CapabilityDenied(
                "MCP registry changed before modern protected operation"
            )
        if (
            binding.registry_generation != registry_binding["registry_generation"]
            or binding.manifest_sha256 != registry_binding["registry_spec_sha256"]
        ):
            raise CapabilityDenied("MCP registry changed before modern authorization")
        try:
            request_json = dumps(dict(payload))
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
            raise ValidationError("MCP modern request is not canonical JSON") from error
        request_bytes = len(request_json.encode("utf-8"))
        if request_bytes > manifest.max_request_bytes:
            raise ValidationError(
                "MCP modern request exceeds "
                f"max_request_bytes={manifest.max_request_bytes}"
            )
        request_sha256 = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        operation_context = {
            **visibility,
            **registry_binding,
            "authority_mode": self._modern_authority_mode(usage_pid),
            "request_sha256": request_sha256,
            "arguments_sha256": request_sha256,
            "request_bytes": request_bytes,
            "transport": manifest.transport,
            "auth_generation": binding.auth_generation,
            "auth_principal_sha256": binding.auth_principal_sha256,
            "auth_scope_sha256": binding.auth_scope_sha256,
        }
        decisions: list[Any] = []
        if usage_pid is not None:
            decisions.append(
                self._authorize_modern_operation(
                    actor,
                    target,
                    right,
                    operation_context,
                )
            )
            decisions.append(
                self._authorize_modern_operation(
                    actor,
                    self.server_resource(server_id),
                    CapabilityRight.EXECUTE,
                    operation_context,
                )
            )
        server = mcp_transport_spec_from_v3(manifest)
        return {
            "client": client,
            "usage_pid": usage_pid,
            "binding": binding,
            "manifest": manifest,
            "deadline": deadline,
            "registry_binding": registry_binding,
            "request_bytes": request_bytes,
            "operation_context": operation_context,
            "decisions": decisions,
            "server": server,
        }

    def _prepare_modern_transport(
        self,
        plan: dict[str, Any],
        *,
        operation: str,
        logical_id: str,
        actor: str,
    ) -> None:
        deadline = plan["deadline"]
        self._remaining_timeout(deadline)
        usage_pid = plan["usage_pid"]
        server = plan["server"]
        binding = plan["binding"]
        manifest = plan["manifest"]
        decisions = plan["decisions"]
        if usage_pid is not None:
            decisions.extend(
                self._require_stdio_process_spawn(actor, server, consume=False)
            )
        host_environment = binding.runtime_environment
        if not isinstance(host_environment, Mapping):
            raise ValidationError("MCP modern runtime environment is unavailable")
        runtime_environment = self._require_runtime_environment(
            server,
            host_environment=host_environment,
        )
        self._remaining_timeout(deadline)
        stdio_identity = self._stdio_executable_identity(
            server,
            runtime_environment=runtime_environment,
            deadline=deadline,
            fail_closed=True,
        )
        self._remaining_timeout(deadline)
        sink = self._modern_data_sink(
            manifest,
            operation=operation,
            logical_id=logical_id,
            stdio_identity=stdio_identity,
        )
        flow_context = (
            self._data_flow().current_context()
            if usage_pid is not None
            else DataFlowContext(
                labels=DataLabels(
                    sensitivity="public",
                    trust_level="verified",
                    integrity="verified",
                    origin=f"runtime:mcp-modern:{actor}",
                )
            )
        )
        plan.update(
            runtime_environment=runtime_environment,
            stdio_identity=stdio_identity,
            sink=sink,
            flow_context=flow_context,
        )

    def _modern_protected_invocation(
        self,
        plan: Mapping[str, Any],
        *,
        operation: str,
        server_id: str,
        logical_id: str,
        actor: str,
        target: str,
        payload: Mapping[str, Any],
        state_mutation: bool,
        rollback_class: ExternalEffectRollbackClass,
        rollback_status: ExternalEffectRollbackStatus,
    ) -> ProtectedOperationInvocation:
        manifest = plan["manifest"]
        operation_context = plan["operation_context"]
        request_bytes = plan["request_bytes"]
        usage_pid = plan["usage_pid"]
        registry_binding = plan["registry_binding"]
        sink = plan["sink"]
        flow_context = plan["flow_context"]
        server = plan["server"]
        runtime_environment = plan["runtime_environment"]
        canonical_args = self._modern_protected_canonical_args(plan)
        effect_context = {
            **operation_context,
            "rollback_class": rollback_class.value,
            "rollback_status": rollback_status.value,
            "state_mutation": state_mutation,
            "information_flow": True,
        }
        reservation = (
            ResourceUsage(
                mcp_request_bytes=manifest.max_request_bytes,
                mcp_response_bytes=manifest.max_response_bytes,
            )
            if usage_pid is not None
            else None
        )
        return ProtectedOperationInvocation(
            pid=actor,
            actor=actor,
            target=target,
            decisions=tuple(plan["decisions"]),
            canonical_args=canonical_args,
            observation=effect_context,
            reservation_usage=reservation,
            resource_source=f"primitive.mcp.{operation}",
            resource_context={
                "server_id": server_id,
                "logical_id": logical_id,
                "request_bytes": request_bytes,
            },
            **self._protected_registry_guard(registry_binding, server_id),
            failure_evidence=lambda error, phase: self._modern_failure_evidence(
                actor=actor,
                target=target,
                operation=operation,
                context=operation_context,
                error=error,
                phase=phase,
                state_mutation=state_mutation,
            ),
            data_sink=sink,
            data_sink_revalidator=lambda: self._modern_data_sink(
                manifest,
                operation=operation,
                logical_id=logical_id,
                stdio_identity=self._stdio_executable_identity(
                    server,
                    runtime_environment=runtime_environment,
                    deadline=plan["deadline"],
                    fail_closed=True,
                ),
            ),
            data_flow_context=flow_context,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                flow_context,
                origin="external:mcp",
            ),
            data_flow_payload=dict(payload),
            data_flow_operation=f"mcp.{operation}",
        )

    @staticmethod
    def _modern_protected_canonical_args(
        plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Use the exact Human-bound args when one approved grant is active."""

        decisions = plan.get("decisions")
        if not isinstance(decisions, list):
            raise ValidationError("MCP modern authority decisions are invalid")
        approval_key = CapabilityManager.APPROVAL_BINDING_KEY
        approved_contexts = []
        for decision in decisions:
            results = getattr(decision, "constraint_results", None)
            binding = results.get(approval_key) if isinstance(results, dict) else None
            if (
                bool(getattr(decision, "allowed", False))
                and getattr(decision, "consume_capability_id", None) is not None
                and isinstance(binding, dict)
                and binding.get("ok") is True
                and isinstance(getattr(decision, "context", None), dict)
            ):
                approved_contexts.append(dict(decision.context))
        if len(approved_contexts) > 1:
            raise CapabilityDenied(
                "MCP modern operation has conflicting Human approval bindings"
            )
        if approved_contexts:
            return approved_contexts[0]
        context = plan.get("operation_context")
        if not isinstance(context, dict):
            raise ValidationError("MCP modern operation context is invalid")
        return dict(context)

    def _execute_modern_operation(
        self,
        plan: Mapping[str, Any],
        *,
        operation: str,
        server_id: str,
        logical_id: str,
        actor: str,
        target: str,
        payload: Mapping[str, Any],
        invoke: Any,
        state_mutation: bool,
        rollback_class: ExternalEffectRollbackClass,
        rollback_status: ExternalEffectRollbackStatus,
        contract_name: str | None,
        result_settler: Any | None,
    ) -> Any:
        usage_pid = plan["usage_pid"]
        if contract_name is not None:
            contract = (
                contract_name
                if usage_pid is not None
                else f"{contract_name}.internal"
            )
        else:
            contract = (
                f"primitive.mcp.{operation}"
                if usage_pid is not None
                else f"primitive.mcp.{operation}.internal"
            )
        invocation = plan["invocation"]
        with self._protected().start(
            contract,
            invocation,
            provider=self.provider,
        ) as protected:
            capture_settlement: Any | None = None
            dispatch_context = {
                "actor": actor,
                "usage_pid": usage_pid,
                "binding": plan["binding"],
                "server": plan["server"],
                "deadline": plan["deadline"],
                "effect_id": protected.effect_id,
                "operation": operation,
                "logical_id": logical_id,
                "operation_context": plan["operation_context"],
                "decisions": tuple(plan["decisions"]),
                "sink": plan["sink"],
                "flow_context": plan["flow_context"],
                "payload": dict(payload),
                "stdio_identity": plan["stdio_identity"],
            }

            def dispatch() -> Any:
                nonlocal capture_settlement
                self._remaining_timeout(plan["deadline"])
                token = self._modern_dispatch_context.set(dispatch_context)
                try:
                    result = invoke(plan["client"], plan["deadline"])
                    result, capture_settlement = self._settle_modern_dispatch_result(
                        result,
                        result_settler=result_settler,
                        server_id=server_id,
                        logical_id=logical_id,
                        payload=payload,
                        effect_id=protected.effect_id,
                    )
                    return result
                finally:
                    self._modern_dispatch_context.reset(token)

            try:
                result = protected.call(
                    ProviderPhase(
                        "modern_provider_operation",
                        state_mutation=state_mutation,
                        information_flow=True,
                    ),
                    dispatch,
                )
            except BaseException as error:
                if capture_settlement is None:
                    if isinstance(error, Exception):
                        self._abort_modern_prepared_capture(protected.effect_id)
                else:
                    self._abort_modern_capture_after_error(
                        capture_settlement,
                        error,
                        reason="protected_dispatch_failed",
                        group_label="MCP modern dispatch cleanup failed",
                    )
                raise
            try:
                response_bytes, classification = self._modern_response_classification(
                    plan,
                    result,
                    operation=operation,
                    state_mutation=state_mutation,
                    rollback_class=rollback_class,
                    rollback_status=rollback_status,
                )
            except BaseException as error:
                self._abort_modern_capture_after_error(
                    capture_settlement,
                    error,
                    reason="protected_projection_failed",
                    group_label="MCP modern projection cleanup failed",
                )
                raise
            try:
                completed = protected.complete(
                    result,
                    self._modern_success_evidence(
                        actor=actor,
                        target=target,
                        operation=operation,
                        context=plan["operation_context"],
                        response_bytes=response_bytes,
                        result=result,
                        decisions=plan["decisions"],
                        state_mutation=state_mutation,
                    ),
                    classification_override=classification,
                    settle_success=(
                        capture_settlement.commit_deferred
                        if capture_settlement is not None
                        else None
                    ),
                    resource=self._modern_resource_settlement(
                        plan,
                        operation=operation,
                        server_id=server_id,
                        logical_id=logical_id,
                        response_bytes=response_bytes,
                        enabled=usage_pid is not None,
                    ),
                )
            except BaseException as error:
                self._abort_modern_capture_after_error(
                    capture_settlement,
                    error,
                    reason="protected_settlement_failed",
                    group_label="MCP modern settlement cleanup failed",
                )
                raise
            self._finalize_modern_capture(capture_settlement)
            return completed

    def _settle_modern_dispatch_result(
        self,
        result: Any,
        *,
        result_settler: Any | None,
        server_id: str,
        logical_id: str,
        payload: Mapping[str, Any],
        effect_id: str,
    ) -> tuple[Any, Any | None]:
        if result_settler is None:
            settlement = self._require_modern_durable_result_provenance(
                result,
                server_id=server_id,
                logical_id=logical_id,
                payload=payload,
                effect_id=effect_id,
            )
            return result, settlement
        settled = result_settler(result, effect_id)
        if type(settled) is not tuple or len(settled) != 2:
            raise ValidationError("MCP modern result settlement is invalid")
        public, settlement = settled
        if settlement is None:
            raise ValidationError("MCP modern result lacks durable settlement")
        return public, settlement

    def _modern_response_classification(
        self,
        plan: Mapping[str, Any],
        result: Any,
        *,
        operation: str,
        state_mutation: bool,
        rollback_class: ExternalEffectRollbackClass,
        rollback_status: ExternalEffectRollbackStatus,
    ) -> tuple[int, ExternalEffectClassification]:
        self._remaining_timeout(plan["deadline"])
        response_bytes = len(dumps(to_jsonable(result)).encode("utf-8"))
        self._remaining_timeout(plan["deadline"])
        if response_bytes > plan["manifest"].max_response_bytes:
            raise ValidationError(
                "MCP modern public result exceeds "
                f"max_response_bytes={plan['manifest'].max_response_bytes}"
            )
        return response_bytes, ExternalEffectClassification(
            rollback_class=rollback_class,
            rollback_status=rollback_status,
            state_mutation=state_mutation,
            information_flow=True,
            metadata={
                "outcome": "succeeded",
                "operation": operation,
                "request_bytes": plan["request_bytes"],
                "response_bytes": response_bytes,
            },
        )

    @staticmethod
    def _abort_modern_capture_after_error(
        settlement: Any | None,
        error: BaseException,
        *,
        reason: str,
        group_label: str,
    ) -> None:
        should_abort = settlement is not None and (
            isinstance(error, Exception)
            or getattr(settlement, "abort_on_base_exception", False) is True
        )
        if not should_abort:
            return
        try:
            settlement.abort(reason=reason)
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(group_label, [error, cleanup_error])

    @staticmethod
    def _modern_resource_settlement(
        plan: Mapping[str, Any],
        *,
        operation: str,
        server_id: str,
        logical_id: str,
        response_bytes: int,
        enabled: bool,
    ) -> ResourceSettlement | None:
        if not enabled:
            return None
        return ResourceSettlement(
            usage=ResourceUsage(
                mcp_request_bytes=plan["request_bytes"],
                mcp_response_bytes=response_bytes,
            ),
            source=f"primitive.mcp.{operation}",
            context={
                "server_id": server_id,
                "logical_id": logical_id,
                "request_bytes": plan["request_bytes"],
                "response_bytes": response_bytes,
            },
        )

    @staticmethod
    def _finalize_modern_capture(settlement: Any | None) -> None:
        if settlement is None:
            return
        try:
            settlement.finalize()
        except BaseException as error:
            if getattr(settlement, "fail_closed_on_finalize_error", False) is True:
                try:
                    settlement.abort(reason="protected_finalize_failed")
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "MCP modern finalization cleanup failed",
                        [error, cleanup_error],
                    )
                raise
            if not isinstance(error, Exception):
                raise
            # RuntimeStore is authoritative; restart reconciles its durable
            # cleaning receipt without turning a committed ref ambiguous.

    def _require_modern_durable_result_provenance(
        self,
        result: Any,
        *,
        server_id: str,
        logical_id: str,
        payload: Mapping[str, Any],
        effect_id: str,
    ) -> Any | None:
        """Reject public non-Complete refs without exact durable backing.

        A custom modern Provider can construct the public dataclasses directly.
        Treating those values as proof of Host capture would expose a
        continuation/task reference that cannot be resumed, or let Completion
        bypass its non-respondable protocol boundary.  This postcondition runs
        inside the same protected Provider phase used by the capture adapters,
        before result bytes, evidence, GUI, or model projection.
        """

        if isinstance(result, McpComplete):
            self._require_no_prepared_modern_capture(effect_id)
            return
        method = payload.get("method")
        if type(method) is not str or not method:
            raise ValidationError("MCP modern result method binding is invalid")
        if method == "completion/complete":
            raise McpContinuationSurfaceUnsupported(
                "MCP completion/complete cannot return a non-Complete result"
            )
        if isinstance(result, McpInputRequired):
            if not result.respondable:
                self._require_typed_unsupported_input_result(result, effect_id)
                return None
            return self._require_modern_continuation_provenance(
                result,
                server_id=server_id,
                operation=method,
                logical_id=logical_id,
            )
        elif isinstance(result, McpRemoteTask):
            return self._require_modern_task_provenance(
                result,
                server_id=server_id,
                operation=method,
                logical_id=logical_id,
            )

    def _require_typed_unsupported_input_result(
        self,
        result: McpInputRequired,
        effect_id: str,
    ) -> None:
        if (
            result.continuation_id
            or result.expires_at is not None
            or result.revision != 0
            or result.human_request_id is not None
            or result.human_revision is not None
            or result.human_preview_sha256 is not None
            or not result.input_requests
            or any(
                request.kind
                not in {
                    McpInputRequestKind.SAMPLING_UNSUPPORTED,
                    McpInputRequestKind.ROOTS_UNSUPPORTED,
                }
                for request in result.input_requests
            )
        ):
            raise ValidationError("MCP nonrespondable input result is invalid")
        self._require_no_prepared_modern_capture(effect_id)

    def _require_modern_continuation_provenance(
        self,
        result: McpInputRequired,
        *,
        server_id: str,
        operation: str,
        logical_id: str,
    ) -> Any:
        if operation not in {"tools/call", "resources/read", "prompts/get"}:
            raise McpContinuationSurfaceUnsupported(
                "MCP operation does not support durable input-required continuation"
            )
        expected = self._capture_continuation_binding(
            server_id,
            operation,
            logical_id,
        )
        try:
            manager = self._continuation_manager()
            return manager.claim_initial_capture(result, binding=expected)
        except Exception:
            raise ValidationError(
                "MCP continuation lacks exact durable provenance"
            ) from None

    def _require_modern_task_provenance(
        self,
        result: McpRemoteTask,
        *,
        server_id: str,
        operation: str,
        logical_id: str,
    ) -> Any:
        try:
            manager = self._remote_task_manager()
            expected = self._capture_remote_task_binding(
                server_id,
                operation,
                logical_id,
            )
            return manager.claim_initial_capture(result, binding=expected)
        except Exception:
            raise ValidationError(
                "MCP remote Task lacks exact durable provenance"
            ) from None

    def _require_no_prepared_modern_capture(self, effect_id: str) -> None:
        for manager in (self._modern_continuations, self._modern_remote_tasks):
            checker = getattr(manager, "has_prepared_effect", None)
            if callable(checker) and checker(effect_id):
                raise ValidationError("MCP Provider changed its prepared result")

    def _abort_modern_prepared_capture(self, effect_id: str) -> None:
        for manager in (self._modern_continuations, self._modern_remote_tasks):
            abort = getattr(manager, "abort_prepared_effect", None)
            if callable(abort):
                abort(effect_id)

    def _resolve_modern_binding(
        self,
        server_id: str,
        *,
        owner_id: str,
    ) -> McpClientBinding:
        client = self._modern_client
        resolver = getattr(client, "binding_resolver", None) if client is not None else None
        resolve = getattr(resolver, "resolve", None)
        if callable(resolve):
            binding = resolve(server_id, owner_id=owner_id)
        elif callable(resolver):
            binding = resolver(server_id)
        else:
            raise ValidationError("MCP modern binding resolver is unavailable")
        if not isinstance(binding, McpClientBinding):
            raise ValidationError("MCP modern binding resolver returned an invalid binding")
        if binding.owner_id != owner_id:
            raise CapabilityDenied("MCP modern binding belongs to another owner")
        return binding

    def _capture_continuation_binding(
        self,
        server_id: str,
        operation: str,
        logical_id: str,
    ) -> McpContinuationBinding:
        """Capture real initial-call authority/effect state for durable MRTR."""

        context, binding = self._active_modern_capture_context(
            server_id,
            operation,
            logical_id,
        )
        return McpContinuationBinding(
            server_id=server_id,
            server_spec_sha256=binding.manifest_sha256,
            server_generation=binding.registry_generation,
            owner_id=str(binding.owner_id),
            auth_principal_sha256=self._modern_optional_fence_sha256(
                binding.auth_principal_sha256,
                empty_value=None,
            ),
            auth_scope_sha256=self._modern_optional_fence_sha256(
                binding.auth_scope_sha256,
                empty_value=[],
            ),
            canonical_request=dict(context["payload"]),
            effect_id=str(context["effect_id"]),
            capability_sha256=self._modern_capability_binding_sha256(context),
            data_flow_sha256=self._modern_flow_binding_sha256(context),
        )

    def capture_continuation_binding(
        self,
        server_id: str,
        operation: str,
        logical_id: str,
    ) -> McpContinuationBinding:
        """Composition port for durable capture inside an active provider phase."""

        return self._capture_continuation_binding(server_id, operation, logical_id)

    def authorize_mcp_host_question(
        self,
        *,
        owner_id: str,
        server_id: str,
        operation: str,
        local_ref: str,
        preview: Mapping[str, Any],
        preview_sha256: str,
    ) -> None:
        """Authorize the narrow Host Human-question composition seam.

        This port is callable only while an exact Runtime/GUI MCP provider
        result is being captured inside its ProtectedOperation.  It does not
        mint or emulate a process Capability; it proves the Host-internal
        admission marker and binds the sanitized Human preview to that active
        server/method/effect context.
        """

        if owner_id not in {"runtime", "gui", "cli"}:
            raise CapabilityDenied("MCP Host Human question actor is invalid")
        self._require_active_mcp_host_question_context(
            owner_id=owner_id,
            server_id=server_id,
            operation=operation,
        )
        self._require_mcp_host_question_preview(
            server_id=server_id,
            operation=operation,
            local_ref=local_ref,
            preview=preview,
            preview_sha256=preview_sha256,
        )

    def _require_active_mcp_host_question_context(
        self,
        *,
        owner_id: str,
        server_id: str,
        operation: str,
    ) -> None:
        context = self._modern_dispatch_context.get()
        binding = context.get("binding") if isinstance(context, dict) else None
        operation_context = (
            context.get("operation_context") if isinstance(context, dict) else None
        )
        if (
            not isinstance(context, dict)
            or context.get("actor") != owner_id
            or context.get("usage_pid") is not None
            or type(context.get("effect_id")) is not str
            or not context["effect_id"]
            or not isinstance(binding, McpClientBinding)
            or binding.owner_id != owner_id
            or binding.manifest.server_id != server_id
            or not isinstance(operation_context, Mapping)
            or operation_context.get("authority_mode")
            != "host_protected_operation"
            or type(context.get("decisions")) is not tuple
            or context["decisions"]
            or not isinstance(context.get("payload"), Mapping)
            or context["payload"].get("method") != operation
        ):
            raise CapabilityDenied(
                "MCP Host Human question is outside protected capture"
            )

    @staticmethod
    def _require_mcp_host_question_preview(
        *,
        server_id: str,
        operation: str,
        local_ref: str,
        preview: Mapping[str, Any],
        preview_sha256: str,
    ) -> None:
        if (
            type(local_ref) is not str
            or not local_ref
            or not isinstance(preview, Mapping)
            or preview.get("contract") != "agent-libos.mcp.elicitation.v1"
            or preview.get("serverId") != server_id
            or preview.get("operation") != operation
            or preview.get("localRef") != local_ref
            or type(preview_sha256) is not str
            or json_sha256(
                dict(preview),
                label="MCP Host Human question preview",
            )
            != preview_sha256
        ):
            raise CapabilityDenied("MCP Host Human question preview changed")

    def _capture_remote_task_binding(
        self,
        server_id: str,
        operation: str,
        logical_id: str,
    ) -> McpRemoteTaskBinding:
        """Capture the exact Host Tasks pin and initial protected effect."""

        context, binding = self._active_modern_capture_context(
            server_id,
            operation,
            logical_id,
        )
        extension, host_pin = self._require_modern_tasks_manifest_pin(
            binding.manifest
        )
        request_sha256 = self._modern_dispatch_request_sha256(context)
        return McpRemoteTaskBinding(
            server_id=server_id,
            server_spec_sha256=binding.manifest_sha256,
            server_generation=binding.registry_generation,
            owner_id=str(binding.owner_id),
            auth_principal_sha256=self._modern_optional_fence_sha256(
                binding.auth_principal_sha256,
                empty_value=None,
            ),
            auth_scope_sha256=self._modern_optional_fence_sha256(
                binding.auth_scope_sha256,
                empty_value=[],
            ),
            origin_request_sha256=request_sha256,
            origin_effect_id=str(context["effect_id"]),
            extension_id=extension.extension_id,
            tasks_extension_sha256=extension.spec_sha256,
            host_tasks_extension_sha256=host_pin,
        )

    def capture_remote_task_binding(
        self,
        server_id: str,
        operation: str,
        logical_id: str,
    ) -> McpRemoteTaskBinding:
        """Composition port for Task capture inside an active provider phase."""

        return self._capture_remote_task_binding(server_id, operation, logical_id)

    def resolve_continuation_task_binding(
        self,
        binding: McpContinuationBinding,
        *,
        origin_effect_id: str,
    ) -> McpRemoteTaskBinding:
        """Bind a continuation-created Task to its current protected effect."""

        if not isinstance(binding, McpContinuationBinding):
            raise TypeError("MCP continuation binding is required")
        context = self._modern_dispatch_context.get()
        current_binding = context.get("binding") if isinstance(context, dict) else None
        payload = context.get("payload") if isinstance(context, dict) else None
        if (
            type(origin_effect_id) is not str
            or not origin_effect_id
            or not isinstance(context, dict)
            or context.get("effect_id") != origin_effect_id
            or context.get("operation") != "continuation.respond"
            or not isinstance(current_binding, McpClientBinding)
            or not isinstance(payload, dict)
        ):
            raise CapabilityDenied(
                "MCP continuation Task protected-effect binding changed"
            )
        self._require_stored_modern_binding(current_binding, binding)
        manifest, _metadata = self._load_server(binding.server_id)
        if not isinstance(manifest, McpServerManifestV3):
            raise ValidationError("MCP continuation Task requires Manifest v3")
        registry = self._registry_binding_for_server_spec(manifest)
        if (
            registry["registry_spec_sha256"] != binding.server_spec_sha256
            or registry["registry_generation"] != binding.server_generation
        ):
            raise CapabilityDenied(
                "MCP continuation Task registry binding changed"
            )
        extension, host_pin = self._require_modern_tasks_manifest_pin(manifest)
        return McpRemoteTaskBinding(
            server_id=binding.server_id,
            server_spec_sha256=binding.server_spec_sha256,
            server_generation=binding.server_generation,
            owner_id=binding.owner_id,
            auth_principal_sha256=binding.auth_principal_sha256,
            auth_scope_sha256=binding.auth_scope_sha256,
            origin_request_sha256=self._modern_dispatch_request_sha256(context),
            origin_effect_id=origin_effect_id,
            extension_id=extension.extension_id,
            tasks_extension_sha256=extension.spec_sha256,
            host_tasks_extension_sha256=host_pin,
        )

    def _require_modern_tasks_manifest_pin(
        self,
        manifest: McpServerManifestV3,
    ) -> tuple[Any, str]:
        extension = manifest.tasks_extension
        host_pin = self.config.mcp.tasks_extension_spec_sha256
        if (
            extension is None
            or extension.extension_id != MCP_TASKS_EXTENSION_ID
            or not self.config.mcp.tasks_extension_enabled
            or type(host_pin) is not str
            or extension.spec_sha256 != host_pin
        ):
            raise ValidationError(
                "MCP Tasks extension is not enabled with the exact Host pin"
            )
        return extension, host_pin

    async def dispatch_continuation_boundary(
        self,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        """Composition port used only by the durable continuation manager."""

        if operation == "respond":
            function = self._continue_v3_sync
        elif operation == "cancel":
            function = self._cancel_continuation_v3_sync
        else:
            raise ValidationError("MCP continuation boundary operation is invalid")
        return await self._data_flow().run_sync_in_worker(function, **kwargs)

    async def dispatch_remote_task_boundary(
        self,
        operation: str,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Composition port used only by the durable remote-Task manager."""

        return await self._data_flow().run_sync_in_worker(
            self._remote_task_v3_sync,
            operation,
            **kwargs,
        )

    def _continue_v3_sync(
        self,
        *,
        record: Any,
        binding: McpContinuationBinding,
        original_request: dict[str, Any],
        input_responses: dict[str, Any],
        request_state: str | None,
        deadline: float,
        result_settler: Any,
    ) -> Any:
        provider_entry = _McpProviderEntry()
        with _certify_mcp_pre_provider(
            provider_entry,
            operation="continuation respond",
        ):
            return self._continue_v3_sync_inner(
                record=record,
                binding=binding,
                original_request=original_request,
                input_responses=input_responses,
                request_state=request_state,
                deadline=deadline,
                result_settler=result_settler,
                provider_entry=provider_entry,
            )

    def _continue_v3_sync_inner(
        self,
        *,
        record: Any,
        binding: McpContinuationBinding,
        original_request: dict[str, Any],
        input_responses: dict[str, Any],
        request_state: str | None,
        deadline: float,
        result_settler: Any,
        provider_entry: _McpProviderEntry,
    ) -> Any:
        request = self._continuation_request_parts(
            record,
            binding,
            original_request,
        )
        actor = request["actor"]
        server_id = request["server_id"]
        provider = self._modern_continuation_provider
        if provider is None:
            raise ValidationError("MCP continuation provider is unavailable")
        selected_responses = _canonical_mcp_arguments(
            input_responses,
            max_bytes=self.config.mcp.max_request_hard_limit_bytes,
        )
        manifest, _metadata = self._load_server(server_id)
        if not isinstance(manifest, McpServerManifestV3):
            raise ValidationError("MCP continuation requires Manifest v3")
        surface = self._continuation_surface_plan(manifest, request)
        provider_method = getattr(provider, surface["provider_method"], None)
        if not callable(provider_method):
            raise ValidationError("MCP continuation provider surface is unavailable")
        payload = {
            **dict(original_request),
            "input_responses": selected_responses,
            "request_state": request_state,
        }

        def invoke(_client: Any, selected_deadline: float) -> Mapping[str, Any]:
            with _certify_mcp_pre_provider(
                provider_entry,
                operation="continuation respond",
            ):
                context = self._require_modern_dispatch_binding(binding)

                async def call_once() -> Mapping[str, Any]:
                    with bind_mcp_client_binding(context):
                        return await self._continue_v3_provider_call(
                            provider_method,
                            context,
                            request,
                            surface,
                            selected_responses,
                            request_state,
                            selected_deadline,
                            provider_entry=provider_entry,
                        )

                result = _run_mcp_provider_awaitable(call_once())
                if not isinstance(result, Mapping):
                    raise ValidationError("MCP continuation provider result is invalid")
                return dict(result)

        return self._run_modern_read(
            operation="continuation.respond",
            server_id=server_id,
            logical_id=surface["logical_id"],
            actor=actor,
            target=surface["target"],
            right=surface["right"],
            payload=payload,
            invoke=invoke,
            binding_preflight=lambda current: self._require_stored_modern_binding(
                current,
                binding,
            ),
            expected_registry_binding=self._stored_registry_binding(binding),
            absolute_deadline=deadline,
            state_mutation=surface["state_mutation"],
            rollback_class=surface["rollback_class"],
            rollback_status=surface["rollback_status"],
            contract_name="primitive.mcp.continuation.respond",
            result_settler=result_settler,
        )

    async def _continue_v3_provider_call(
        self,
        provider_method: Any,
        binding: McpClientBinding,
        request: Mapping[str, Any],
        surface: Mapping[str, Any],
        input_responses: dict[str, Any],
        request_state: str | None,
        deadline: float,
        *,
        provider_entry: _McpProviderEntry | None = None,
    ) -> Mapping[str, Any]:
        server = mcp_transport_spec_from_v3(binding.manifest)
        method = request["method"]
        sensitive_values = self._modern_provider_sensitive_values(
            binding,
            request_state,
        )
        if method == "tools/call":
            provider_call = lambda: provider_method(
                server,
                surface["remote_name"],
                request["arguments"],
                input_responses,
                request_state,
                deadline=deadline,
            )
        elif method == "resources/read":
            provider_call = lambda: provider_method(
                server,
                surface["remote_name"],
                surface["logical_id"],
                input_responses,
                request_state,
                deadline=deadline,
            )
        elif method == "prompts/get":
            provider_call = lambda: provider_method(
                server,
                surface["remote_name"],
                surface["logical_id"],
                request["arguments"],
                input_responses,
                request_state,
                deadline=deadline,
            )
        else:
            raise ValidationError("MCP continuation surface is unsupported")
        raw = await self._await_modern_provider_result(
            provider_call,
            deadline=deadline,
            sensitive_values=sensitive_values,
            provider_entry=provider_entry,
        )
        if not isinstance(raw, Mapping):
            raise ValidationError("MCP continuation Provider result is invalid")
        bounded_public_size(
            raw,
            maximum=server.max_response_bytes,
            label="MCP continuation Provider result",
        )
        sanitized = sanitize_provider_json(
            dict(raw),
            sensitive_values=sensitive_values,
        )
        if type(sanitized) is not dict:
            raise ValidationError("MCP continuation Provider result is invalid")
        if sanitized.get("resultType") == "task":
            # A continuation binding by itself does not carry Tasks extension
            # authority. Recheck the current exact Manifest and Host digest
            # before any durable Task capture/sidecar can observe this result.
            self._require_modern_tasks_manifest_pin(binding.manifest)
        return sanitized

    @staticmethod
    def _modern_provider_sensitive_values(
        binding: McpClientBinding,
        *dynamic_values: str | None,
    ) -> tuple[str, ...]:
        if not isinstance(binding, McpClientBinding):
            raise ValidationError("MCP modern Provider binding is invalid")
        selected = list(binding.sensitive_values)
        for value in dynamic_values:
            if value is None:
                continue
            if type(value) is not str:
                raise ValidationError("MCP modern Provider dynamic secret is invalid")
            if value:
                selected.append(value)
        return tuple(dict.fromkeys(selected))

    @staticmethod
    async def _await_modern_provider_result(
        provider_call: Any,
        *,
        deadline: float,
        sensitive_values: tuple[str, ...],
        provider_entry: _McpProviderEntry | None = None,
    ) -> Any:
        if (
            type(deadline) not in {int, float}
            or not math.isfinite(float(deadline))
        ):
            raise ValidationError("MCP provider absolute deadline is invalid")
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("MCP absolute deadline exhausted before provider")
        if provider_entry is not None:
            provider_entry.entered = True
        try:
            pending = provider_call()
        except McpContinuationSurfaceUnsupported:
            raise
        except Exception as error:
            # Once the Provider callable is entered, even a same-named
            # ProviderEffectNotStarted is untrusted. A custom Provider must
            # not be able to manufacture a replay certificate.
            raise safe_mcp_provider_error(error, sensitive_values) from None
        if not inspect.isawaitable(pending):
            raise safe_mcp_provider_error(
                ValidationError("MCP modern Provider method must be asynchronous"),
                sensitive_values,
            ) from None
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            if inspect.iscoroutine(pending):
                pending.close()
            raise safe_mcp_provider_error(
                TimeoutError("MCP provider exceeded the absolute deadline"),
                sensitive_values,
            ) from None
        task = asyncio.ensure_future(pending)
        try:
            done, _pending = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            await _cancel_and_drain_mcp_provider_task(task)
            raise
        if not done:
            await _cancel_and_drain_mcp_provider_task(task)
            raise safe_mcp_provider_error(
                TimeoutError("MCP provider exceeded the absolute deadline"),
                sensitive_values,
            ) from None
        try:
            result = task.result()
        except McpContinuationSurfaceUnsupported:
            raise
        except asyncio.CancelledError as error:
            raise safe_mcp_provider_error(error, sensitive_values) from None
        except Exception as error:
            raise safe_mcp_provider_error(error, sensitive_values) from None
        if time.monotonic() >= float(deadline):
            raise safe_mcp_provider_error(
                TimeoutError("MCP provider exceeded the absolute deadline"),
                sensitive_values,
            ) from None
        return result

    def _cancel_continuation_v3_sync(
        self,
        *,
        record: Any,
        binding: McpContinuationBinding,
        deadline: float,
    ) -> None:
        provider_entry = _McpProviderEntry()
        with _certify_mcp_pre_provider(
            provider_entry,
            operation="continuation cancel",
        ):
            self._cancel_continuation_v3_sync_inner(
                record=record,
                binding=binding,
                deadline=deadline,
            )

    def _cancel_continuation_v3_sync_inner(
        self,
        *,
        record: Any,
        binding: McpContinuationBinding,
        deadline: float,
    ) -> None:
        request = binding.detached_request()
        request_parts = self._continuation_request_parts(
            record,
            binding,
            request,
        )
        actor = request_parts["actor"]
        server_id = request_parts["server_id"]
        manifest, _metadata = self._load_server(server_id)
        if not isinstance(manifest, McpServerManifestV3):
            raise ValidationError("MCP continuation requires Manifest v3")
        surface = self._continuation_surface_plan(manifest, request_parts)
        self._run_modern_read(
            operation="continuation.cancel",
            server_id=server_id,
            logical_id=surface["logical_id"],
            actor=actor,
            target=surface["target"],
            right=surface["right"],
            payload={
                "method": "continuation/cancel",
                "server_id": server_id,
                "initial_method": request_parts["method"],
                "logical_id": surface["logical_id"],
                "request_sha256": binding.request_sha256,
            },
            invoke=lambda _client, _deadline: None,
            binding_preflight=lambda current: self._require_stored_modern_binding(
                current,
                binding,
            ),
            expected_registry_binding=self._stored_registry_binding(binding),
            absolute_deadline=deadline,
            state_mutation=False,
            rollback_class=surface["rollback_class"],
            rollback_status=surface["rollback_status"],
            contract_name="primitive.mcp.continuation.cancel",
        )

    def _remote_task_v3_sync(
        self,
        operation: str,
        *,
        record: Any,
        binding: McpRemoteTaskBinding,
        remote_task_id: str,
        deadline: float,
        input_responses: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        provider_entry = _McpProviderEntry()
        with _certify_mcp_pre_provider(
            provider_entry,
            operation=f"remote Task {operation}",
        ):
            return self._remote_task_v3_sync_inner(
                operation,
                record=record,
                binding=binding,
                remote_task_id=remote_task_id,
                deadline=deadline,
                input_responses=input_responses,
                provider_entry=provider_entry,
            )

    def _remote_task_v3_sync_inner(
        self,
        operation: str,
        *,
        record: Any,
        binding: McpRemoteTaskBinding,
        remote_task_id: str,
        deadline: float,
        input_responses: Mapping[str, Any] | None,
        provider_entry: _McpProviderEntry,
    ) -> Mapping[str, Any]:
        if operation not in {"get", "update", "cancel"}:
            raise ValidationError("MCP remote Task operation is invalid")
        actor = binding.owner_id
        if (
            getattr(record, "owner_id", None) != actor
            or getattr(record, "server_id", None) != binding.server_id
        ):
            raise CapabilityDenied("MCP remote Task owner binding changed")
        provider = self._modern_tasks_provider
        if provider is None:
            raise ValidationError("MCP Tasks provider is unavailable")
        selected_responses = (
            None
            if input_responses is None
            else _canonical_mcp_arguments(
                input_responses,
                max_bytes=self.config.mcp.max_request_hard_limit_bytes,
            )
        )
        right = CapabilityRight.READ if operation == "get" else CapabilityRight.WRITE
        mutation = operation != "get"
        rollback_class = (
            ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
            if not mutation
            else ExternalEffectRollbackClass.IRREVERSIBLE
        )
        rollback_status = (
            ExternalEffectRollbackStatus.NOT_REQUIRED
            if not mutation
            else ExternalEffectRollbackStatus.NOT_SUPPORTED
        )

        def invoke(_client: Any, selected_deadline: float) -> Mapping[str, Any]:
            with _certify_mcp_pre_provider(
                provider_entry,
                operation=f"remote Task {operation}",
            ):
                client_binding = self._require_modern_dispatch_binding(binding)
                server = mcp_transport_spec_from_v3(client_binding.manifest)
                sensitive_values = self._modern_provider_sensitive_values(
                    client_binding,
                    remote_task_id,
                )

                async def call_once() -> Mapping[str, Any]:
                    with bind_mcp_client_binding(client_binding):
                        if operation == "get":
                            provider_call = lambda: provider.get_remote_task(
                                server,
                                remote_task_id,
                                deadline=selected_deadline,
                            )
                        elif operation == "update":
                            provider_call = lambda: provider.update_remote_task(
                                server,
                                remote_task_id,
                                selected_responses,
                                deadline=selected_deadline,
                            )
                        else:
                            provider_call = lambda: provider.cancel_remote_task(
                                server,
                                remote_task_id,
                                deadline=selected_deadline,
                            )
                        return await self._await_modern_provider_result(
                            provider_call,
                            deadline=selected_deadline,
                            sensitive_values=sensitive_values,
                            provider_entry=provider_entry,
                        )

                result = _run_mcp_provider_awaitable(call_once())
                if not isinstance(result, Mapping):
                    raise ValidationError("MCP Tasks provider result is invalid")
                if operation == "get" and result.get("taskId") != remote_task_id:
                    raise ValidationError(
                        "MCP Tasks provider returned another task identity"
                    )
                provider_payload = dict(result)
                provider_task_id = provider_payload.pop("taskId", None)
                sanitized = sanitize_provider_json(
                    provider_payload,
                    sensitive_values=sensitive_values,
                )
                if type(sanitized) is not dict:
                    raise ValidationError("MCP Tasks provider result is invalid")
                if operation == "get":
                    # The manager needs the exact broker-only identity to verify
                    # the durable local ref.  It is never projected publicly; all
                    # provider-controlled siblings were sanitized with that
                    # bearer in the exact-sensitive set above.
                    sanitized["taskId"] = provider_task_id
                return sanitized

        return self._run_modern_read(
            operation=f"tasks.{operation}",
            server_id=binding.server_id,
            logical_id=str(getattr(record, "task_ref", "")),
            actor=actor,
            target=f"mcp_task:{getattr(record, 'task_ref', '')}",
            right=right,
            payload={
                "method": f"tasks/{operation}",
                "server_id": binding.server_id,
                "task_ref": getattr(record, "task_ref", ""),
                **(
                    {"input_responses": selected_responses}
                    if selected_responses is not None
                    else {}
                ),
            },
            invoke=invoke,
            binding_preflight=lambda current: self._require_stored_modern_binding(
                current,
                binding,
            ),
            expected_registry_binding=self._stored_registry_binding(binding),
            absolute_deadline=deadline,
            state_mutation=mutation,
            rollback_class=rollback_class,
            rollback_status=rollback_status,
            contract_name=f"primitive.mcp.tasks.{operation}",
        )

    def _continuation_request_parts(
        self,
        record: Any,
        binding: McpContinuationBinding,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(binding, McpContinuationBinding):
            raise TypeError("MCP continuation binding is required")
        if not isinstance(request, Mapping):
            raise ValidationError("MCP continuation original request is invalid")
        method = request.get("method")
        server_id = request.get("server_id")
        if (
            type(method) is not str
            or server_id != binding.server_id
            or getattr(record, "server_id", None) != server_id
            or getattr(record, "owner_id", None) != binding.owner_id
            or getattr(record, "request_sha256", None) != binding.request_sha256
            or getattr(record, "effect_id", None) != binding.effect_id
            or json_sha256(
                dict(request),
                label="MCP continuation original request",
            )
            != binding.request_sha256
        ):
            raise CapabilityDenied("MCP continuation durable binding changed")
        self._require_continuation_authority_binding(binding, method)
        if method == "tools/call":
            if set(request) != {"method", "server_id", "tool_id", "arguments"}:
                raise ValidationError("MCP Tool continuation request is invalid")
            logical_id = request.get("tool_id")
            arguments = _canonical_mcp_arguments(
                request.get("arguments"),
                max_bytes=self.config.mcp.max_request_hard_limit_bytes,
            )
            selected: dict[str, Any] = {"arguments": arguments}
        elif method == "resources/read":
            if set(request) != {
                "method",
                "server_id",
                "resource_id",
                "variables",
                "for_model",
            }:
                raise ValidationError("MCP Resource continuation request is invalid")
            logical_id = request.get("resource_id")
            selected = {
                "variables": self._modern_string_mapping(
                    request.get("variables"),
                    label="Resource continuation variables",
                ),
                "for_model": request.get("for_model"),
            }
            if type(selected["for_model"]) is not bool:
                raise ValidationError("MCP Resource continuation visibility is invalid")
        elif method == "prompts/get":
            if set(request) != {
                "method",
                "server_id",
                "prompt_id",
                "arguments",
                "confirmed",
            }:
                raise ValidationError("MCP Prompt continuation request is invalid")
            logical_id = request.get("prompt_id")
            selected = {
                "arguments": self._modern_string_mapping(
                    request.get("arguments"),
                    label="Prompt continuation arguments",
                )
            }
            if request.get("confirmed") is not False:
                raise ValidationError(
                    "confirmed MCP Prompt cannot become a continuation"
                )
        else:
            raise ValidationError("MCP continuation surface is unsupported")
        if type(logical_id) is not str or not logical_id:
            raise ValidationError("MCP continuation logical id is invalid")
        return {
            "actor": binding.owner_id,
            "server_id": binding.server_id,
            "method": method,
            "logical_id": logical_id,
            **selected,
        }

    def _require_continuation_authority_binding(
        self,
        binding: McpContinuationBinding,
        initial_method: str,
    ) -> None:
        host_digest = self._modern_host_authority_binding_sha256(
            binding.owner_id,
            initial_method,
        )
        current_process = self._resource_usage_pid(binding.owner_id)
        if binding.capability_sha256 == host_digest:
            if current_process is not None:
                raise CapabilityDenied("MCP Host continuation owner changed")
            return
        if current_process is None:
            raise CapabilityDenied("MCP process continuation owner is unavailable")

    def _continuation_surface_plan(
        self,
        manifest: McpServerManifestV3,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        method = request["method"]
        server_id = request["server_id"]
        logical_id = request["logical_id"]
        if method == "tools/call":
            tool = manifest.tool_by_id(logical_id)
            if tool is None:
                raise NotFound(f"MCP tool not found: {server_id}/{logical_id}")
            visibility = self._visibility_operation_context(
                request["actor"],
                server_id,
                logical_id,
                request["arguments"],
            )
            self._authorize_call_visibility(
                request["actor"],
                self.tool_resource(server_id, logical_id),
                visibility,
            )
            right, rollback_class, rollback_status = self._modern_tool_effect(tool)
            return {
                "logical_id": logical_id,
                "target": self.tool_resource(server_id, logical_id),
                "right": right,
                "state_mutation": tool.state_mutation,
                "rollback_class": rollback_class,
                "rollback_status": rollback_status,
                "provider_method": "continue_tool",
                "remote_name": tool.mcp_name,
            }
        if method == "resources/read":
            resolver = getattr(self._modern_client, "resolve_resource_selector", None)
            if not callable(resolver):
                raise ValidationError("MCP Resource selector resolver is unavailable")
            remote_name = resolver(
                manifest,
                logical_id,
                request["variables"],
                for_model=request["for_model"],
            )
            provider_method = "continue_resource"
            target = f"mcp:{server_id}:resource:{logical_id}"
        elif method == "prompts/get":
            prompt = next(
                (item for item in manifest.prompts if item.prompt_id == logical_id),
                None,
            )
            if prompt is None:
                raise NotFound(f"MCP prompt not found: {logical_id}")
            if not set(request["arguments"]).issubset(prompt.argument_names):
                raise ValidationError(
                    "MCP Prompt continuation argument is not manifest-authorized"
                )
            remote_name = prompt.mcp_name
            provider_method = "continue_prompt"
            target = f"mcp:{server_id}:prompt:{logical_id}"
        else:  # pragma: no cover - parsed before this helper
            raise ValidationError("MCP continuation surface is unsupported")
        return {
            "logical_id": logical_id,
            "target": target,
            "right": CapabilityRight.READ,
            "state_mutation": False,
            "rollback_class": ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            "rollback_status": ExternalEffectRollbackStatus.NOT_REQUIRED,
            "provider_method": provider_method,
            "remote_name": remote_name,
        }

    @staticmethod
    def _modern_tool_effect(
        tool: McpToolSpec,
    ) -> tuple[
        CapabilityRight,
        ExternalEffectRollbackClass,
        ExternalEffectRollbackStatus,
    ]:
        try:
            right = CapabilityRight(tool.right)
            rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
            rollback_status = (
                ExternalEffectRollbackStatus(tool.rollback_status)
                if tool.rollback_status is not None
                else default_external_effect_rollback_status(rollback_class)
            )
        except (TypeError, ValueError) as error:
            raise ValidationError("MCP Tool effect declaration is invalid") from error
        return right, rollback_class, rollback_status

    def _require_modern_dispatch_binding(
        self,
        stored: McpContinuationBinding | McpRemoteTaskBinding,
    ) -> McpClientBinding:
        context = self._modern_dispatch_context.get()
        selected = context.get("binding") if isinstance(context, dict) else None
        if not isinstance(selected, McpClientBinding):
            raise ProviderEffectNotStarted("MCP modern dispatch binding is unavailable")
        self._require_stored_modern_binding(selected, stored)
        return selected

    def _require_stored_modern_binding(
        self,
        current: McpClientBinding,
        stored: McpContinuationBinding | McpRemoteTaskBinding,
    ) -> None:
        if (
            current.manifest.server_id != stored.server_id
            or current.manifest_sha256 != stored.server_spec_sha256
            or current.registry_generation != stored.server_generation
            or current.owner_id != stored.owner_id
            or self._modern_optional_fence_sha256(
                current.auth_principal_sha256,
                empty_value=None,
            )
            != stored.auth_principal_sha256
            or self._modern_optional_fence_sha256(
                current.auth_scope_sha256,
                empty_value=[],
            )
            != stored.auth_scope_sha256
        ):
            raise CapabilityDenied("MCP durable operation fence changed")
        if isinstance(stored, McpRemoteTaskBinding):
            extension = current.manifest.tasks_extension
            if (
                extension is None
                or extension.extension_id != stored.extension_id
                or extension.spec_sha256 != stored.tasks_extension_sha256
                or stored.host_tasks_extension_sha256
                != self.config.mcp.tasks_extension_spec_sha256
            ):
                raise CapabilityDenied("MCP Tasks extension fence changed")

    @staticmethod
    def _stored_registry_binding(
        stored: McpContinuationBinding | McpRemoteTaskBinding,
    ) -> dict[str, Any]:
        return {
            "registry_spec_sha256": stored.server_spec_sha256,
            "registry_generation": stored.server_generation,
        }

    def _active_modern_capture_context(
        self,
        server_id: str,
        operation: str,
        logical_id: str,
    ) -> tuple[dict[str, Any], McpClientBinding]:
        context = self._modern_dispatch_context.get()
        if not isinstance(context, dict):
            raise ValidationError(
                "MCP durable result capture is outside the protected provider phase"
            )
        binding = context.get("binding")
        payload = context.get("payload")
        effect_id = context.get("effect_id")
        if (
            not isinstance(binding, McpClientBinding)
            or binding.owner_id is None
            or binding.manifest.server_id != server_id
            or context.get("logical_id") != logical_id
            or not isinstance(payload, dict)
            or payload.get("method") != operation
            or type(effect_id) is not str
            or not effect_id
        ):
            raise CapabilityDenied("MCP durable result capture binding changed")
        return context, binding

    @staticmethod
    def _modern_dispatch_request_sha256(context: Mapping[str, Any]) -> str:
        """Reuse the exact hash authorized by the active protected operation."""

        operation_context = context.get("operation_context")
        request_sha256 = (
            operation_context.get("request_sha256")
            if isinstance(operation_context, Mapping)
            else None
        )
        if (
            type(request_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None
        ):
            raise CapabilityDenied("MCP protected request digest is unavailable")
        return request_sha256

    @staticmethod
    def _modern_optional_fence_sha256(
        value: str | None,
        *,
        empty_value: Any,
    ) -> str:
        if value is not None:
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValidationError("MCP auth fence digest is invalid")
            return value
        return hashlib.sha256(dumps(empty_value).encode("utf-8")).hexdigest()

    @staticmethod
    def _modern_capability_binding_sha256(context: Mapping[str, Any]) -> str:
        decisions = context.get("decisions")
        if type(decisions) is not tuple:
            raise CapabilityDenied("MCP durable capture lacks Capability evidence")
        if not decisions:
            if context.get("usage_pid") is not None:
                raise CapabilityDenied(
                    "MCP durable capture lacks process Capability evidence"
                )
            operation_context = context.get("operation_context")
            payload = context.get("payload")
            actor = context.get("actor")
            if (
                not isinstance(operation_context, Mapping)
                or operation_context.get("authority_mode")
                != "host_protected_operation"
                or operation_context.get("pid") != actor
                or not isinstance(payload, Mapping)
                or type(payload.get("method")) is not str
            ):
                raise CapabilityDenied(
                    "MCP durable capture lacks Host authority evidence"
                )
            return McpPrimitive._modern_host_authority_binding_sha256(
                actor,
                payload["method"],
            )
        payload = [
            {
                "allowed": bool(getattr(decision, "allowed", False)),
                "matched_capability_ids": list(
                    getattr(decision, "matched_capability_ids", ())
                ),
                "selected_capability_id": getattr(
                    decision,
                    "selected_capability_id",
                    None,
                ),
            }
            for decision in decisions
        ]
        if any(item["allowed"] is not True for item in payload):
            raise CapabilityDenied("MCP durable capture authority is not allowed")
        return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _modern_authority_mode(usage_pid: str | None) -> str:
        return (
            "process_capability"
            if usage_pid is not None
            else "host_protected_operation"
        )

    @staticmethod
    def _modern_host_authority_binding_sha256(actor: Any, method: Any) -> str:
        if type(actor) is not str or not actor or type(method) is not str or not method:
            raise CapabilityDenied("MCP durable Host authority binding is invalid")
        marker = {
            "authority_mode": "host_protected_operation",
            "actor": actor,
            "initial_method": method,
            "process_capability_ids": [],
        }
        return hashlib.sha256(dumps(marker).encode("utf-8")).hexdigest()

    @staticmethod
    def _modern_flow_binding_sha256(context: Mapping[str, Any]) -> str:
        payload = {
            "context": to_jsonable(context.get("flow_context")),
            "sink": to_jsonable(context.get("sink")),
        }
        return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()

    def _precheck_modern_authority(
        self,
        actor: str,
        resource: str,
        right: CapabilityRight,
        context: Mapping[str, Any],
    ) -> None:
        decision = self.capabilities.authorize(
            actor,
            resource,
            right,
            dict(context),
        )
        if decision.allowed or decision.policy == CapabilityManager.ASK_EACH_TIME:
            return
        raise CapabilityDenied(decision.reason)

    def _authorize_modern_operation(
        self,
        actor: str,
        resource: str,
        right: CapabilityRight,
        context: Mapping[str, Any],
    ) -> Any:
        selected_context = {
            **dict(context),
            "resource": resource,
            "right": right.value,
        }
        decision = self.capabilities.authorize(
            actor,
            resource,
            right,
            selected_context,
            audit=True,
        )
        if decision.allowed:
            return decision
        if decision.policy != CapabilityManager.ASK_EACH_TIME:
            raise CapabilityDenied(decision.reason)
        if self.human is None:
            raise CapabilityDenied(
                f"{actor} requires human approval for MCP operation on {resource}"
            )
        operation = str(selected_context["operation"])
        logical_id = str(selected_context["logical_id"])
        profile = self.capabilities.profiles.mcp(
            resource=resource,
            effect=CapabilityEffect.ASK,
            server_id=str(selected_context["server_id"]),
            tool_id=logical_id,
        )
        constraints = {
            AUTHORITY_RULES_KEY: [
                {
                    "rule_id": (
                        f"mcp.modern.approval.{selected_context['server_id']}."
                        f"{hashlib.sha256(resource.encode('utf-8')).hexdigest()[:16]}"
                    ),
                    "operation": operation,
                    "effect": CapabilityEffect.ALLOW.value,
                    "risk": "high",
                    "conditions": {
                        "server_id": selected_context["server_id"],
                        "tool_id": logical_id,
                        "registry_spec_sha256": selected_context[
                            "registry_spec_sha256"
                        ],
                        "registry_generation": selected_context[
                            "registry_generation"
                        ],
                        "arguments_sha256": selected_context[
                            "arguments_sha256"
                        ],
                    },
                    "description": "one-shot approval for exact MCP modern request",
                }
            ]
        }
        request_id = self.human.query_authority_request(
            pid=actor,
            human=self.config.runtime.default_human,
            request={
                "type": "external_operation_approval",
                "question": f"Allow this process to perform {operation} on {resource}?",
                "requested_once_capability": {
                    "subject": actor,
                    "resource": resource,
                    "rights": [right.value],
                    "constraints": constraints,
                },
                "context": {
                    **selected_context,
                    "sandbox_profile": self._profile_json(profile),
                },
            },
            blocking=True,
            authority_origin="external_operation",
        )
        raise HumanApprovalRequired(
            request_id=request_id,
            message=f"{actor} is waiting for approval to perform {operation}",
        )

    @staticmethod
    def _modern_string_mapping(
        value: Mapping[str, str] | None,
        *,
        label: str,
        required: bool = False,
    ) -> dict[str, str]:
        if value is None:
            if required:
                raise ValidationError(f"MCP {label} must be an object")
            return {}
        if not isinstance(value, Mapping):
            raise ValidationError(f"MCP {label} must be an object")
        selected: dict[str, str] = {}
        for key, item in value.items():
            if type(key) is not str or type(item) is not str:
                raise ValidationError(f"MCP {label} must contain string values")
            selected[key] = item
        return selected

    @staticmethod
    def _validate_modern_cursor(cursor: str | None) -> None:
        if cursor is not None and type(cursor) is not str:
            raise ValidationError("MCP cursor must be a string or null")

    @staticmethod
    def _validate_prompt_preview_request(
        *,
        confirmed: bool,
        expected_preview_sha256: str | None,
    ) -> None:
        valid_digest = bool(
            isinstance(expected_preview_sha256, str)
            and len(expected_preview_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in expected_preview_sha256
            )
        )
        if confirmed and not valid_digest:
            raise ValidationError(
                "confirmed MCP Prompt requires expected_preview_sha256"
            )
        if not confirmed and expected_preview_sha256 is not None:
            raise ValidationError(
                "MCP Prompt preview must not supply expected_preview_sha256"
            )

    @staticmethod
    def _validate_subscription_filters(filters: Any) -> tuple[str, ...]:
        if type(filters) is not tuple or not filters or any(
            type(item) is not str for item in filters
        ):
            raise ValidationError(
                "MCP subscription filters must be a non-empty string tuple"
            )
        if len(filters) > len(MCP_V3_SUBSCRIPTION_FILTERS):
            raise ValidationError("MCP subscription filters exceed the maximum count")
        if len(set(filters)) != len(filters):
            raise ValidationError("MCP subscription filters must be unique")
        unknown = sorted(set(filters) - MCP_V3_SUBSCRIPTION_FILTERS)
        if unknown:
            raise ValidationError(
                f"unsupported MCP subscription filters: {unknown}"
            )
        return filters

    def _tasks_subscription_fence(
        self,
        manifest: McpServerManifestV3,
        filters: tuple[str, ...],
    ) -> McpTasksSubscriptionFence | None:
        if "taskIds" not in filters:
            return None
        extension = manifest.tasks_extension
        host_pin = self.config.mcp.tasks_extension_spec_sha256
        if (
            manifest.schema_version != 3
            or manifest.protocol_mode is not McpProtocolMode.REVISION_2026_07_28
            or self.config.mcp.tasks_extension_enabled is not True
            or extension is None
            or extension.extension_id != MCP_TASKS_EXTENSION_ID
            or type(host_pin) is not str
            or extension.spec_sha256 != host_pin
        ):
            raise ValidationError(
                "MCP taskIds subscription requires the exact Host-pinned Tasks extension"
            )
        return McpTasksSubscriptionFence(
            extension_id=extension.extension_id,
            manifest_spec_sha256=extension.spec_sha256,
            host_spec_sha256=host_pin,
        )

    def _prepare_subscription_start_result(
        self,
        filters: tuple[str, ...],
        *,
        deadline: float,
    ) -> _McpPreparedSubscriptionResult:
        binding, server = self._subscription_dispatch_binding()
        if not set(filters).issubset(binding.manifest.subscriptions):
            raise ValidationError(
                "MCP subscription filters are not declared by the manifest"
            )
        context = self._modern_dispatch_context.get()
        effect_id = context.get("effect_id") if isinstance(context, dict) else None
        if type(effect_id) is not str or not effect_id:
            raise ValidationError("MCP subscription protected effect is unavailable")
        manager = self._subscription_manager()
        runner = self._subscription_runner(manager)
        prepared = runner.run(
            lambda: manager.prepare_start(
                server,
                mcp_connection_fence(binding),
                self._subscription_provider(),
                filters,
                sensitive_values=binding.sensitive_values,
                tasks_extension_fence=self._tasks_subscription_fence(
                    binding.manifest,
                    filters,
                ),
                deadline=deadline,
                origin_effect_id=effect_id,
            ),
            deadline=deadline,
            binding=binding,
        )
        return self._coerce_prepared_subscription_start(
            prepared,
            manager=manager,
            runner=runner,
            binding=binding,
            dispatch_context=context,
        )

    def _coerce_prepared_subscription_start(
        self,
        prepared: Any,
        *,
        manager: Any,
        runner: _McpSubscriptionLoopRunner,
        binding: McpClientBinding,
        dispatch_context: Any,
    ) -> _McpPreparedSubscriptionResult:
        selected = self._subscription_start_settlement(
            prepared,
            manager=manager,
            runner=runner,
            binding=binding,
            dispatch_context=dispatch_context,
        )
        public = prepared[0] if type(prepared) is tuple and prepared else None
        valid = (
            isinstance(public, McpSubscription)
            and selected is not None
            and selected[0].subscription_id == public.subscription_id
        )
        if not valid:
            self._reject_prepared_subscription_start(
                selected[1] if selected is not None else None
            )
        assert isinstance(public, McpSubscription) and selected is not None
        return _McpPreparedSubscriptionResult(public, selected[0], selected[1])

    def _subscription_start_settlement(
        self,
        prepared: Any,
        *,
        manager: Any,
        runner: _McpSubscriptionLoopRunner,
        binding: McpClientBinding,
        dispatch_context: Any,
    ) -> tuple[
        McpSubscriptionStartSettlement,
        _McpSubscriptionEffectSettlement,
    ] | None:
        if type(prepared) is not tuple or len(prepared) != 2:
            return None
        settlement = prepared[1]
        if (
            type(settlement) is not McpSubscriptionStartSettlement
            or settlement.manager is not manager
        ):
            return None
        live = settlement.opening.prepared
        public = prepared[0] if isinstance(prepared[0], McpSubscription) else None
        cleanup_public = public if public is not None else live.public if live else None
        if cleanup_public is None or not isinstance(dispatch_context, dict):
            return None
        effect = _McpSubscriptionEffectSettlement(
            runner=runner,
            settlement=settlement,
            binding=binding,
            dispatch_context_var=self._modern_dispatch_context,
            dispatch_context=dict(dispatch_context),
            public=cleanup_public,
        )
        return settlement, effect

    @staticmethod
    def _reject_prepared_subscription_start(
        settlement: _McpSubscriptionEffectSettlement | None,
    ) -> None:
        error = ValidationError(
            "MCP subscription manager returned an invalid prepared start"
        )
        if settlement is None:
            raise error
        try:
            settlement.abort(reason="protected_result_validation_failed")
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "MCP subscription result validation cleanup failed",
                [error, cleanup_error],
            )
        raise error

    def _subscription_manager(self) -> Any:
        manager = self._modern_subscriptions
        required = ("prepare_start", "start", "status", "events", "stop", "close")
        if manager is None or any(
            not callable(getattr(manager, name, None)) for name in required
        ):
            raise ValidationError("MCP subscription manager is unavailable")
        return manager

    def _subscription_provider(self) -> Any:
        provider = self._modern_subscription_provider
        required = ("listen", "receive", "close")
        if provider is None or any(
            not callable(getattr(provider, name, None)) for name in required
        ):
            raise ValidationError("MCP subscription provider is unavailable")
        return provider

    def _subscription_runner(self, manager: Any) -> _McpSubscriptionLoopRunner:
        with self._modern_subscription_lock:
            runner = self._modern_subscription_runner
            if runner is None:
                runner = _McpSubscriptionLoopRunner(
                    manager,
                    dispatch_context_var=self._modern_dispatch_context,
                )
                self._modern_subscription_runner = runner
            elif runner.manager is not manager:
                raise ValidationError("MCP subscription manager binding changed")
            return runner

    def _continuation_manager(self) -> Any:
        manager = self._modern_continuations
        required = (
            "binding_material",
            "get",
            "respond",
            "cancel",
        )
        if manager is None or any(
            not callable(getattr(manager, name, None)) for name in required
        ):
            raise ValidationError("MCP continuation manager is unavailable")
        return manager

    def _remote_task_manager(self) -> Any:
        manager = self._modern_remote_tasks
        required = (
            "binding_material",
            "inspect",
            "get",
            "update",
            "cancel",
        )
        if manager is None or any(
            not callable(getattr(manager, name, None)) for name in required
        ):
            raise ValidationError("MCP remote Task manager is unavailable")
        return manager

    def _remote_task_binding(self, manager: Any, task_ref: str) -> McpRemoteTaskBinding:
        pin = self.config.mcp.tasks_extension_spec_sha256
        if self.config.mcp.tasks_extension_enabled is not True or type(pin) is not str:
            raise ValidationError("MCP Tasks extension is disabled or unpinned")
        return manager.binding_material(
            task_ref,
            tasks_extension_sha256=pin,
            host_tasks_extension_sha256=pin,
        )

    def _durable_mcp_deadline(self) -> float:
        timeout_s = self.config.mcp.timeout_s
        if (
            type(timeout_s) not in {int, float}
            or isinstance(timeout_s, bool)
            or not math.isfinite(float(timeout_s))
            or timeout_s <= 0
        ):
            raise ValidationError("MCP durable operation timeout is invalid")
        return time.monotonic() + float(timeout_s)

    @staticmethod
    def _validate_durable_host_actor(actor: str) -> None:
        if (
            type(actor) is not str
            or not actor
            or len(actor) > 512
            or actor != actor.strip()
            or "\x00" in actor
        ):
            raise ValidationError("MCP Host responder actor is invalid")

    @staticmethod
    def _require_durable_human_fence(
        pending: McpInputRequired | McpRemoteTask,
        *,
        expected_revision: int,
        human_request_id: str,
        human_expected_revision: int,
        human_preview_sha256: str,
    ) -> None:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValidationError("MCP durable expected_revision is invalid")
        if (
            pending.revision != expected_revision
            or pending.human_request_id != human_request_id
            or pending.human_revision != human_expected_revision
            or pending.human_preview_sha256 != human_preview_sha256
        ):
            raise CapabilityDenied("MCP Human response binding changed")

    def _subscription_dispatch_binding(
        self,
    ) -> tuple[McpClientBinding, McpServerSpec]:
        context = self._modern_dispatch_context.get()
        if not isinstance(context, dict):
            raise ValidationError("MCP subscription dispatch context is unavailable")
        binding = context.get("binding")
        server = context.get("server")
        if not isinstance(binding, McpClientBinding) or not isinstance(
            server, McpServerSpec
        ):
            raise ValidationError("MCP subscription dispatch binding is invalid")
        if binding.manifest.server_id != server.server_id:
            raise ValidationError("MCP subscription dispatch server changed")
        return binding, server

    def _subscription_record_for_operation(
        self,
        subscription_id: str,
        *,
        actor: str,
        right: CapabilityRight,
        operation: str,
    ) -> tuple[Any, str]:
        self._validate_identifier(subscription_id, "subscription_id", 512)
        if type(actor) is not str or not actor:
            raise ValidationError("MCP subscription actor is invalid")
        target = f"mcp_subscription:{subscription_id}"
        if self._resource_usage_pid(actor) is not None:
            self._precheck_modern_authority(
                actor,
                target,
                right,
                {
                    "pid": actor,
                    "primitive": f"runtime.mcp.{operation}",
                    "operation": f"mcp.{operation}",
                    "authority_operation": f"mcp.{operation}",
                    "logical_id": subscription_id,
                    "tool_id": subscription_id,
                    "right": right.value,
                },
            )
        record = self.unit_of_work.mcp_subscriptions.get(subscription_id)
        if record is None:
            raise NotFound(f"MCP subscription not found: {subscription_id}")
        if record.owner_id != actor:
            raise CapabilityDenied("MCP subscription belongs to another owner")
        return record, target

    @staticmethod
    def _require_subscription_record_binding(
        record: Any,
        binding: McpClientBinding,
        *,
        actor: str,
    ) -> None:
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        if (
            record.server_id != binding.manifest.server_id
            or record.server_spec_sha256 != binding.manifest_sha256
            or record.server_generation != binding.registry_generation
            or record.owner_id != actor
            or binding.owner_id != actor
            or record.auth_principal_sha256
            != (binding.auth_principal_sha256 or empty_sha256)
            or record.auth_scope_sha256
            != (binding.auth_scope_sha256 or empty_sha256)
        ):
            raise CapabilityDenied("MCP subscription binding changed")

    def _modern_data_sink(
        self,
        manifest: McpServerManifestV3,
        *,
        operation: str,
        logical_id: str,
        stdio_identity: dict[str, str] | None,
    ) -> DataSink:
        identity = f"mcp:{manifest.server_id}:{operation}:{logical_id}"
        if manifest.transport == "stdio" and stdio_identity is None:
            return DataSink(identity)
        digest = hashlib.sha256(
            dumps(
                {
                    "schema_version": 3,
                    "server": json.loads(canonical_mcp_v3_manifest_json(manifest)),
                    "operation": operation,
                    "logical_id": logical_id,
                    "stdio_executable": stdio_identity,
                }
            ).encode("utf-8")
        ).hexdigest()
        return DataSink(identity, digest)

    def _modern_success_evidence(
        self,
        *,
        actor: str,
        target: str,
        operation: str,
        context: Mapping[str, Any],
        response_bytes: int,
        result: Any,
        decisions: list[Any],
        state_mutation: bool,
    ) -> ProtectedOperationEvidence:
        kind = getattr(result, "kind", None)
        if kind is None and hasattr(result, "items"):
            kind = "page"
        review = self._modern_review_evidence(context)
        payload = {
            "adapter": "mcp",
            "operation": operation,
            "server_id": context["server_id"],
            "logical_id": context["logical_id"],
            "ok": True,
            "result_kind": str(kind or type(result).__name__),
            "request_bytes": context["request_bytes"],
            "response_bytes": response_bytes,
            **review,
        }
        durable_receipt: dict[str, Any] = {}
        if isinstance(result, McpInputRequired) and result.respondable:
            durable_receipt = {
                "kind": "input_required",
                "continuation_id": result.continuation_id,
            }
        elif isinstance(result, McpRemoteTask):
            durable_receipt = {
                "kind": "remote_task",
                "task_ref": result.task_ref,
            }
        return ProtectedOperationEvidence(
            event_type=(
                EventType.EXTERNAL_WRITE
                if state_mutation
                else EventType.EXTERNAL_READ
            ),
            event_source=actor,
            event_target=target,
            event_payload=payload,
            audit_action=f"primitive.mcp.{operation}",
            audit_actor=actor,
            audit_target=target,
            audit_decision={
                **payload,
                "request_sha256": context["request_sha256"],
                "registry_spec_sha256": context["registry_spec_sha256"],
                "registry_generation": context["registry_generation"],
            },
            capability_refs=tuple(
                decision.selected_capability_id
                for decision in decisions
                if getattr(decision, "selected_capability_id", None)
            ),
            effect_metadata=payload,
            provider_receipt={
                "request_bytes": context["request_bytes"],
                "response_bytes": response_bytes,
                **(
                    {"mcp_durable_result": durable_receipt}
                    if durable_receipt
                    else {}
                ),
            },
        )

    def _modern_failure_evidence(
        self,
        *,
        actor: str,
        target: str,
        operation: str,
        context: Mapping[str, Any],
        error: BaseException,
        phase: str,
        state_mutation: bool = False,
    ) -> ProtectedOperationEvidence:
        review = self._modern_review_evidence(context)
        payload = {
            "adapter": "mcp",
            "operation": operation,
            "server_id": context["server_id"],
            "logical_id": context["logical_id"],
            "ok": False,
            "error_type": type(error).__name__,
            "phase": phase,
            "request_bytes": context["request_bytes"],
            "response_bytes": 0,
            **review,
        }
        return ProtectedOperationEvidence(
            event_type=(
                EventType.EXTERNAL_WRITE
                if state_mutation
                else EventType.EXTERNAL_READ
            ),
            event_source=actor,
            event_target=target,
            event_payload=payload,
            audit_action=f"primitive.mcp.{operation}",
            audit_actor=actor,
            audit_target=target,
            audit_decision={
                **payload,
                "request_sha256": context["request_sha256"],
                "registry_spec_sha256": context["registry_spec_sha256"],
                "registry_generation": context["registry_generation"],
            },
            effect_metadata=payload,
            provider_receipt={"request_bytes": 0, "response_bytes": 0},
        )

    @staticmethod
    def _modern_review_evidence(context: Mapping[str, Any]) -> dict[str, Any]:
        if context.get("confirmed") is not True:
            return {}
        reviewer = context.get("reviewer")
        reason = context.get("reason")
        if type(reviewer) is not str or type(reason) is not str:
            raise ValidationError("MCP review evidence is invalid")
        return {"confirmed": True, "reviewer": reviewer, "reason": reason}

    def server_resource(self, server_id: str) -> str:
        return f"mcp_server:{server_id}"

    def tool_resource(self, server_id: str, tool_id: str) -> str:
        return f"mcp:{server_id}:{tool_id}"

    @staticmethod
    def stdio_resource_for_argv(
        command: str,
        args: list[str] | tuple[str, ...],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        payload = dumps(
            {
                "command": command,
                "args": list(args),
                "env": sorted((env or {}).items()),
                "cwd": McpPrimitive._canonical_stdio_cwd(cwd),
            }
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"mcp_stdio:{digest}"

    @classmethod
    def stdio_resource_for_server(cls, server: McpServerSpec) -> str | None:
        if server.transport != "stdio" or server.stdio is None:
            return None
        return cls.stdio_resource_for_argv(
            server.stdio.command,
            list(server.stdio.args),
            env=dict(server.stdio.env),
            cwd=server.stdio.cwd,
        )

    @staticmethod
    def _canonical_stdio_cwd(cwd: str | None) -> str | None:
        if cwd is None:
            return None
        raw = cwd.replace("\\", "/").strip()
        parts: list[str] = []
        for part in raw.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        if not parts:
            return None
        return "/".join(parts)

    def validate_server_manifest(
        self,
        server: McpRegisteredServer | dict[str, Any],
    ) -> McpRegisteredServer:
        """Return the strict typed manifest accepted by this Host.

        This public Host/DX bridge performs no registry or provider operation.
        Manifest v3 validation includes the active Host attenuation policy.
        """

        return self._coerce_server(server)

    def get_server_manifest(self, server_id: str) -> McpRegisteredServer:
        """Return one validated registered manifest for trusted Host tooling."""

        manifest, _metadata = self._load_server(server_id)
        return manifest

    def import_server_manifest(
        self,
        server: McpRegisteredServer | dict[str, Any],
        *,
        expected_current_sha256: str | None,
        actor: str = "runtime",
        replace: bool = False,
        require_capability: bool = True,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Apply one reviewed import against the caller's registry observation.

        Manifest v3 uses the Store's cross-Runtime atomic CAS. The legacy
        compatibility schemas retain their Runtime-local registry phase fence.
        """

        self._validate_import_expected_digest(expected_current_sha256)
        spec = self.validate_server_manifest(server)
        if isinstance(spec, McpServerManifestV3):
            return self.import_v3_manifest(
                spec,
                expected_current_sha256=expected_current_sha256,
                actor=actor,
                replace=replace,
                require_capability=require_capability,
                source=source,
            )
        with self._registry_phase_lock:
            try:
                current = self.get_server_manifest(spec.server_id)
            except NotFound:
                current = None
            current_sha256 = (
                self._server_spec_sha256(current) if current is not None else None
            )
            if current_sha256 != expected_current_sha256:
                raise ValidationError(
                    "MCP registry changed after import planning; create a new plan"
                )
            return self.register_server(
                spec,
                actor=actor,
                replace=replace,
                require_capability=require_capability,
                source=source,
            )

    @staticmethod
    def _validate_import_expected_digest(value: str | None) -> None:
        if value is not None and (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValidationError(
                "MCP import expected_current_sha256 must be lowercase SHA-256 or null"
            )

    def register_server(
        self,
        server: McpRegisteredServer | dict[str, Any],
        *,
        actor: str = "runtime",
        replace: bool = False,
        require_capability: bool = True,
        source: str | None = None,
    ) -> dict[str, Any]:
        return self._register_server(
            server,
            actor=actor,
            replace=replace,
            require_capability=require_capability,
            source=source,
            expected_current_sha256=_MCP_IMPORT_CAS_UNSET,
        )

    def import_v3_manifest(
        self,
        server: McpServerManifestV3 | dict[str, Any],
        *,
        actor: str = "runtime",
        replace: bool = False,
        require_capability: bool = True,
        expected_current_sha256: str | None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """CAS-bound Manifest-v3 import used by reviewed DX bundles.

        ``None`` means that the caller observed no current registry row.  A
        digest means that the caller observed that exact canonical manifest.
        The comparison and write share the registry phase lock, so an import
        plan can never silently overwrite a newer Host registration.
        """

        self._validate_import_expected_digest(expected_current_sha256)
        spec = self._coerce_server(server)
        if not isinstance(spec, McpServerManifestV3):
            raise ValidationError("MCP import_v3_manifest requires Manifest v3")
        return self._register_server(
            spec,
            actor=actor,
            replace=replace,
            require_capability=require_capability,
            source=source,
            expected_current_sha256=expected_current_sha256,
        )

    def _register_server(
        self,
        server: McpRegisteredServer | dict[str, Any],
        *,
        actor: str,
        replace: bool,
        require_capability: bool,
        source: str | None,
        expected_current_sha256: str | None | object,
    ) -> dict[str, Any]:
        spec = self._coerce_server(server)
        oauth_profile: McpOAuthProfile | None = None
        if isinstance(spec, McpServerManifestV3) and spec.auth_profile_id is not None:
            manager = self._oauth_manager()
            if not manager.has_profile(spec.auth_profile_id):
                raise ValidationError(
                    "MCP Manifest v3 auth_profile_id is not Host-configured"
                )
            oauth_profile = manager.profile_snapshot(spec.auth_profile_id)
            self._validate_oauth_profile_manifest_fields(oauth_profile, spec)
            self._require_oauth_record_identity(
                oauth_profile,
                {
                    "registry_spec_sha256": self._server_spec_sha256(spec),
                    "registry_generation": 0,
                },
            )
        authority_decisions: list[Any] = []
        if require_capability:
            required_right = CapabilityRight.ADMIN if replace else CapabilityRight.WRITE
            authority_decisions.append(
                self.capabilities.require(
                    actor,
                    self.server_resource(spec.server_id),
                    required_right,
                    consume=False,
                )
            )
            authority_decisions.extend(self._require_stdio_process_spawn(actor, spec, consume=False))
        now = utc_now()
        # Registry mutation, composite one-shot authority settlement, and
        # evidence are one store transaction. Local validation or a sink
        # failure rolls every reservation back to its pre-call state.
        with self._registry_phase_lock, self.capabilities.authority_transaction(
            authority_decisions,
            actor=actor,
            operation="MCP server register",
        ):
            existing = self.extensions.get_mcp_server_manifest(spec.server_id)
            if existing is not None and not replace:
                raise ValidationError(f"MCP server already exists: {spec.server_id}")
            if expected_current_sha256 is not _MCP_IMPORT_CAS_UNSET:
                if not isinstance(spec, McpServerManifestV3):  # pragma: no cover
                    raise ValidationError("MCP registry CAS import requires Manifest v3")
                applied = self.extensions.compare_and_swap_mcp_v3_server(
                    spec,
                    expected_current_sha256=expected_current_sha256,
                    registered_by=actor,
                    created_at=now,
                )
                if not applied:
                    raise ValidationError(
                        "MCP registry changed after import planning; create a new plan"
                    )
            elif isinstance(spec, McpServerManifestV3):
                self.extensions.upsert_mcp_v3_server(
                    spec,
                    registered_by=actor,
                    created_at=now,
                )
            else:
                self.extensions.upsert_mcp_server(
                    spec,
                    registered_by=actor,
                    created_at=now,
                )
            if oauth_profile is not None:
                oauth_binding = self._registry_binding_for_server_spec(spec)
                manager = self._oauth_manager()
                self._restore_oauth_generation(
                    manager,
                    oauth_profile,
                    oauth_binding,
                )
                self._sync_oauth_metadata(
                    manager,
                    oauth_profile,
                    spec,
                    oauth_binding,
                    manager.status(oauth_profile.profile_id),
                )
            if existing is not None:
                self._disable_replaced_server_tool_capabilities(spec.server_id, actor=actor)
            self.events.emit(
                EventType.EXTERNAL_WRITE,
                source=actor,
                target=self.server_resource(spec.server_id),
                payload={"adapter": "mcp", "operation": "server_register", "server_id": spec.server_id},
            )
            self.audit.record(
                actor=actor,
                action="mcp.server.register" if existing is None else "mcp.server.replace",
                target=self.server_resource(spec.server_id),
                decision={
                    "server_id": spec.server_id,
                    "transport": spec.transport,
                    "tools": [tool.tool_id for tool in spec.tools],
                    "auth_profile_id": (
                        spec.auth_profile_id
                        if isinstance(spec, McpServerManifestV3)
                        else None
                    ),
                    "replaced": existing is not None,
                    "source": source,
                },
            )
        if existing is not None:
            self._invalidate_modern_server(spec.server_id)
            if oauth_profile is not None:
                manager = self._oauth_manager()
                with self.unit_of_work.transaction():
                    self._sync_oauth_metadata(
                        manager,
                        oauth_profile,
                        spec,
                        self._registry_binding_for_server_spec(spec),
                        manager.status(oauth_profile.profile_id),
                    )
        return self.inspect_server(spec.server_id, actor=actor, require_capability=False)

    def register_server_from_yaml_text(
        self,
        text: str,
        *,
        actor: str,
        replace: bool = False,
        require_capability: bool = True,
        source: str | None = None,
    ) -> dict[str, Any]:
        if len(text.encode("utf-8")) > self.config.mcp.manifest_max_bytes:
            raise ValidationError(f"MCP manifest exceeds manifest_max_bytes={self.config.mcp.manifest_max_bytes}")
        data = load_yaml_mapping(text)
        if set(data) == {"mcp_server"} and isinstance(data["mcp_server"], dict):
            data = data["mcp_server"]
        if set(data) == {"server"} and isinstance(data["server"], dict):
            data = data["server"]
        return self.register_server(
            data,
            actor=actor,
            replace=replace,
            require_capability=require_capability,
            source=source,
        )

    def list_servers(
        self,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        text: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        servers, _has_more = self.list_servers_window(
            actor=actor,
            require_capability=require_capability,
            text=text,
            limit=limit,
        )
        return servers

    def list_servers_window(
        self,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        text: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return one bounded page plus an exact signal that another row exists."""

        if require_capability and actor is not None:
            self.capabilities.require(actor, self.config.mcp.registry_resource, CapabilityRight.READ)
        selected_limit = self._bounded_list_limit(limit)
        servers: list[dict[str, Any]] = []
        rows = self.extensions.list_mcp_server_manifests(
            text=text,
            limit=selected_limit + 1,
        )
        for spec, metadata in rows[:selected_limit]:
            self._validate_server(spec)
            servers.append(self._server_to_json(spec, metadata, include_sensitive_fields=False))
        return servers, len(rows) > selected_limit

    def inspect_server(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        include_sensitive_fields: bool = False,
    ) -> dict[str, Any]:
        if require_capability and actor is not None:
            self.capabilities.require(actor, self.server_resource(server_id), CapabilityRight.READ)
        spec, metadata = self._load_server(server_id)
        return self._server_to_json(spec, metadata, include_sensitive_fields=include_sensitive_fields)

    def discover(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> McpDiscoveryResult:
        """Perform one governed, operation-local MCP modern discovery."""

        try:
            return self._discover(
                server_id,
                actor=actor,
                require_capability=require_capability,
            )
        except ProviderEffectNotStarted as error:
            raise self._safe_not_started_error(error) from None

    def _discover(
        self,
        server_id: str,
        *,
        actor: str | None,
        require_capability: bool,
    ) -> McpDiscoveryResult:
        authority_decisions: list[Any] = []
        if require_capability and actor is not None:
            authority_decisions.append(
                self.capabilities.require(
                    actor,
                    self.server_resource(server_id),
                    CapabilityRight.READ,
                    consume=False,
                )
            )
        spec, _metadata = self._load_server(server_id)
        mode = self._effective_protocol_mode(spec)
        if spec.schema_version != 2 or mode is McpProtocolMode.LEGACY:
            raise ValidationError(
                "MCP discover requires Manifest v2 protocol_mode auto or 2026-07-28"
            )
        self._require_modern_protocol_provider(spec)
        deadline = time.monotonic() + spec.timeout_s
        if require_capability and actor is not None:
            authority_decisions.append(
                self.capabilities.require(
                    actor,
                    self.server_resource(server_id),
                    CapabilityRight.EXECUTE,
                    consume=False,
                )
            )
            authority_decisions.extend(
                self._require_stdio_process_spawn(actor, spec, consume=False)
            )
        effect_actor = actor or "runtime"
        usage_pid = self._resource_usage_pid(actor)
        request_bytes = len(
            dumps(
                {"method": "server/discover", "server_id": spec.server_id}
            ).encode("utf-8")
        )
        if request_bytes > spec.max_request_bytes:
            raise ValidationError(
                "MCP discover request exceeds "
                f"max_request_bytes={spec.max_request_bytes}"
            )
        effect_context = self._discover_effect_context(
            spec,
            request_bytes=request_bytes,
        )
        request_flow = (
            self._data_flow().current_context()
            if actor is not None
            else DataFlowContext(
                labels=DataLabels(
                    sensitivity="public",
                    trust_level="verified",
                    integrity="verified",
                    origin="runtime:mcp-discover-metadata",
                )
            )
        )
        runtime_environment = self._require_runtime_environment(
            spec,
            deadline=deadline,
        )
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=runtime_environment,
            deadline=deadline,
        )
        sink = DataSink(
            f"mcp:{server_id}:discover",
            self._discover_identity_sha256(
                spec,
                stdio_executable=stdio_identity,
                deadline=deadline,
            ),
        )
        self._remaining_timeout(deadline)
        contract_name = (
            "primitive.mcp.discover"
            if authority_decisions
            else "primitive.mcp.discover.internal"
        )
        registry_binding = self._registry_binding_for_server_spec(spec)
        invocation = self._discover_invocation(
            server_id=server_id,
            spec=spec,
            effect_actor=effect_actor,
            authority_decisions=authority_decisions,
            usage_pid=usage_pid,
            request_bytes=request_bytes,
            effect_context=effect_context,
            request_flow=request_flow,
            runtime_environment=runtime_environment,
            sink=sink,
            registry_binding=registry_binding,
            deadline=deadline,
        )
        with self._protected().start(
            contract_name,
            invocation,
            provider=self.provider,
        ) as protected:
            self._remaining_timeout(deadline)
            if spec.transport == "streamable_http":
                observes_host = self._runtime_resolution_observes_host(spec)
                protected.call(
                    ProviderPhase(
                        "dns_resolution",
                        information_flow=observes_host,
                        commits_authority=observes_host,
                    ),
                    self._validate_runtime_resolution,
                    spec,
                    deadline=deadline,
                )
            executable_snapshot = self._stdio_snapshot_for_dispatch(
                pid=effect_actor,
                spec=spec,
                expected_identity=stdio_identity,
                sink=sink,
                context=request_flow,
                payload={"method": "server/discover", "server_id": server_id},
                runtime_environment=runtime_environment,
                deadline=deadline,
            )
            try:
                provider_result = protected.call(
                    ProviderPhase(
                        "provider_not_started_after_dns",
                        information_flow=True,
                    ),
                    self._invoke_discover_provider,
                    spec,
                    deadline=deadline,
                    pid=effect_actor,
                    executable_snapshot=executable_snapshot,
                    runtime_environment=runtime_environment,
                )
            finally:
                if executable_snapshot is not None:
                    executable_snapshot.close()
            if self._registry_binding_context(server_id) != registry_binding:
                raise CapabilityDenied(
                    "MCP server registry changed during protocol discovery"
                )
            result = McpDiscoveryResult(
                server_id=spec.server_id,
                connection=provider_result.connection,
                request_bytes=provider_result.request_bytes,
                response_bytes=provider_result.response_bytes,
                duration_s=provider_result.duration_s,
                receipts=provider_result.receipts,
            )
            result_payload = {
                "ok": True,
                "status": "ok",
                "request_bytes": result.request_bytes,
                "response_bytes": result.response_bytes,
                "duration_s": result.duration_s,
                "connection": to_jsonable(result.connection),
                "receipts": to_jsonable(result.receipts),
            }
            return protected.complete(
                result,
                self._protected_discover_evidence(
                    effect_actor,
                    spec,
                    effect_context,
                    result_payload,
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                resource=(
                    ResourceSettlement(
                        usage=ResourceUsage(
                            mcp_request_bytes=result.request_bytes,
                            mcp_response_bytes=result.response_bytes,
                        ),
                        source="primitive.mcp.discover",
                        context={
                            "server_id": server_id,
                            "request_bytes": result.request_bytes,
                            "response_bytes": result.response_bytes,
                            "protocol_revision": result.connection.protocol_revision,
                        },
                    )
                    if usage_pid is not None
                    else None
                ),
            )

    def _discover_invocation(
        self,
        *,
        server_id: str,
        spec: McpServerSpec,
        effect_actor: str,
        authority_decisions: list[Any],
        usage_pid: str | None,
        request_bytes: int,
        effect_context: dict[str, Any],
        request_flow: DataFlowContext,
        runtime_environment: Mapping[str, str],
        sink: DataSink,
        registry_binding: ProviderRegistryBinding,
        deadline: float,
    ) -> ProtectedOperationInvocation:
        return ProtectedOperationInvocation(
            pid=effect_actor,
            actor=effect_actor,
            target=self.server_resource(server_id),
            decisions=tuple(authority_decisions),
            canonical_args=effect_context,
            observation=effect_context,
            reservation_usage=(
                ResourceUsage(
                    mcp_request_bytes=spec.max_request_bytes,
                    mcp_response_bytes=spec.max_response_bytes,
                )
                if usage_pid is not None
                else None
            ),
            resource_source="primitive.mcp.discover",
            resource_context={
                "server_id": server_id,
                "request_bytes": request_bytes,
            },
            **self._protected_registry_guard(
                registry_binding,
                server_id,
            ),
            data_sink=sink,
            data_sink_revalidator=lambda: self._discover_data_sink(
                server_id,
                spec,
                runtime_environment,
                deadline=deadline,
            ),
            data_flow_context=request_flow,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                request_flow,
                origin="external:mcp",
            ),
            data_flow_payload={"method": "server/discover", "server_id": server_id},
            data_flow_operation="mcp.discover",
            failure_evidence=lambda error, phase: self._protected_discover_failure_evidence(
                effect_actor,
                spec,
                effect_context,
                error,
                phase,
            ),
        )

    async def adiscover(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> McpDiscoveryResult:
        return await self._data_flow().run_sync_in_worker(
            self.discover,
            server_id,
            actor=actor,
            require_capability=require_capability,
        )

    def list_tools(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        refresh: bool = False,
    ) -> dict[str, Any]:
        try:
            return self._list_tools(
                server_id,
                actor=actor,
                require_capability=require_capability,
                refresh=refresh,
            )
        except ProviderEffectNotStarted as error:
            raise self._safe_not_started_error(error) from None

    def _list_tools(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        refresh: bool = False,
    ) -> dict[str, Any]:
        authority_decisions: list[Any] = []
        if require_capability and actor is not None:
            read_decision = self.capabilities.require(
                actor,
                self.server_resource(server_id),
                CapabilityRight.READ,
                consume=not refresh,
            )
            if refresh:
                authority_decisions.append(read_decision)
        spec, _metadata = self._load_server(server_id)
        if refresh:
            # Manifest v2 is an explicit provider-contract boundary, including
            # its legacy mode. Fence an unmarked provider before the refresh
            # path can resolve DNS, snapshot/spawn stdio, or dispatch I/O.
            self._require_modern_protocol_provider(spec)
            deadline = time.monotonic() + spec.timeout_s
        live_by_name: dict[str, McpProviderTool] = {}
        live_response_bytes = 0
        live_connection: McpConnectionInfo | None = None
        live_receipts: tuple[McpExchangeReceipt, ...] = ()
        if refresh:
            result = self._refresh_list_tools(
                server_id,
                actor=actor,
                require_capability=require_capability,
                spec=spec,
                authority_decisions=authority_decisions,
                deadline=deadline,
            )
            live_response_bytes = result.response_bytes
            live_connection = result.connection
            live_receipts = result.receipts
            live_by_name = {tool.name: tool for tool in result.tools}
        return {
            "server_id": spec.server_id,
            "schema_version": spec.schema_version,
            "transport": spec.transport,
            "protocol_mode": self._effective_protocol_mode(spec).value,
            "tools": [
                self._tool_to_json(spec.server_id, tool, live=live_by_name.get(tool.mcp_name) if refresh else None)
                for tool in spec.tools
            ],
            "refreshed": refresh,
            "response_bytes": live_response_bytes,
            **(
                {
                    "connection": to_jsonable(live_connection),
                    "receipts": to_jsonable(live_receipts),
                }
                if live_connection is not None
                else {}
            ),
        }

    def _refresh_list_tools(
        self,
        server_id: str,
        *,
        actor: str | None,
        require_capability: bool,
        spec: McpServerSpec,
        authority_decisions: list[Any],
        deadline: float,
    ) -> McpToolListResult:
        started = deadline - spec.timeout_s
        effect_actor = actor or "runtime"
        usage_pid = self._resource_usage_pid(actor)
        if require_capability and actor is not None:
            authority_decisions.append(
                self.capabilities.require(
                    actor,
                    self.server_resource(server_id),
                    CapabilityRight.EXECUTE,
                    consume=False,
                )
            )
            authority_decisions.extend(
                self._require_stdio_process_spawn(actor, spec, consume=False)
            )
        runtime_environment = self._require_runtime_environment(
            spec,
            deadline=deadline,
        )
        request_payload = {"method": "tools/list", "server_id": spec.server_id}
        request_bytes = len(dumps(request_payload).encode("utf-8"))
        if request_bytes > spec.max_request_bytes:
            raise ValidationError(
                f"MCP list_tools request exceeds max_request_bytes={spec.max_request_bytes}"
            )
        effect_context = self._list_tools_effect_context(
            spec,
            request_bytes=request_bytes,
        )
        resource_context = {"server_id": server_id, "request_bytes": request_bytes}
        request_flow = self._list_tools_request_flow(actor)
        stdio_executable_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=runtime_environment,
            deadline=deadline,
        )
        list_sink = DataSink(
            f"mcp:{server_id}:list_tools",
            self._list_tools_identity_sha256(
                spec,
                stdio_executable=stdio_executable_identity,
                deadline=deadline,
            ),
        )
        self._remaining_timeout(deadline)
        invocation = self._list_tools_invocation(
            server_id=server_id,
            effect_actor=effect_actor,
            spec=spec,
            authority_decisions=authority_decisions,
            effect_context=effect_context,
            request_bytes=request_bytes,
            resource_context=resource_context,
            usage_pid=usage_pid,
            runtime_environment=runtime_environment,
            request_flow=request_flow,
            list_sink=list_sink,
            request_payload=request_payload,
            deadline=deadline,
        )
        contract_name = (
            "primitive.mcp.list_tools"
            if authority_decisions
            else "primitive.mcp.list_tools.internal"
        )
        with self._protected().start(
            contract_name,
            invocation,
            provider=self.provider,
        ) as protected:
            self._remaining_timeout(deadline)
            result, provider_error = self._dispatch_list_tools(
                protected,
                spec,
                deadline=deadline,
                pid=effect_actor,
                expected_identity=stdio_executable_identity,
                sink=list_sink,
                context=request_flow,
                payload=request_payload,
                runtime_environment=runtime_environment,
            )
            if provider_error is not None:
                self._complete_list_tools_failure(
                    protected,
                    provider_error,
                    effect_actor=effect_actor,
                    spec=spec,
                    effect_context=effect_context,
                    resource_context=resource_context,
                    request_bytes=request_bytes,
                    usage_pid=usage_pid,
                    started=started,
                )
                raise provider_error
            self._complete_list_tools_success(
                protected,
                result,
                effect_actor=effect_actor,
                spec=spec,
                effect_context=effect_context,
                resource_context=resource_context,
                request_bytes=request_bytes,
                usage_pid=usage_pid,
            )
            return result

    def _list_tools_request_flow(self, actor: str | None) -> DataFlowContext:
        if actor is not None:
            return self._data_flow().current_context()
        return DataFlowContext(
            labels=DataLabels(
                sensitivity="public",
                trust_level="verified",
                integrity="verified",
                origin="runtime:mcp-list-tools-metadata",
            )
        )

    def _list_tools_invocation(
        self,
        *,
        server_id: str,
        effect_actor: str,
        spec: McpServerSpec,
        authority_decisions: list[Any],
        effect_context: dict[str, Any],
        request_bytes: int,
        resource_context: dict[str, Any],
        usage_pid: str | None,
        runtime_environment: Mapping[str, str],
        request_flow: DataFlowContext,
        list_sink: DataSink,
        request_payload: dict[str, str],
        deadline: float,
    ) -> ProtectedOperationInvocation:
        reservation_usage = None
        if usage_pid is not None:
            reservation_usage = ResourceUsage(
                mcp_request_bytes=(
                    spec.max_request_bytes
                    if spec.schema_version == 2
                    else request_bytes
                ),
                mcp_response_bytes=spec.max_response_bytes,
            )
        return ProtectedOperationInvocation(
            pid=effect_actor,
            actor=effect_actor,
            target=self.server_resource(spec.server_id),
            decisions=tuple(authority_decisions),
            canonical_args=effect_context,
            observation=effect_context,
            reservation_usage=reservation_usage,
            resource_source="primitive.mcp.list_tools",
            resource_context=resource_context,
            **self._protected_registry_guard(
                self._registry_binding_for_server_spec(spec),
                server_id,
            ),
            data_sink=list_sink,
            data_sink_revalidator=lambda: self._list_tools_data_sink(
                server_id,
                spec,
                runtime_environment,
                deadline=deadline,
            ),
            data_flow_context=request_flow,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                request_flow,
                origin="external:mcp",
            ),
            data_flow_payload=request_payload,
            data_flow_operation="mcp.list_tools",
            failure_evidence=lambda error, phase: self._protected_list_failure_evidence(
                effect_actor,
                spec,
                effect_context,
                error,
                phase,
            ),
        )

    def _complete_list_tools_failure(
        self,
        protected: Any,
        provider_error: Exception,
        *,
        effect_actor: str,
        spec: McpServerSpec,
        effect_context: dict[str, Any],
        resource_context: dict[str, Any],
        request_bytes: int,
        usage_pid: str | None,
        started: float,
    ) -> None:
        result_payload = self._list_tools_failure_payload(
            provider_error,
            duration_s=time.monotonic() - started,
        )
        settlement = None
        if usage_pid is not None:
            settlement = ResourceSettlement(
                usage=ResourceUsage(mcp_request_bytes=request_bytes),
                source="primitive.mcp.list_tools",
                context={
                    **resource_context,
                    "response_bytes": 0,
                    "status": result_payload["status"],
                },
                charge_reserved_maximum=True,
            )
        protected.complete(
            result_payload,
            self._protected_list_evidence(
                effect_actor,
                spec,
                effect_context,
                result_payload,
            ),
            classification_context=effect_context,
            classification_result=result_payload,
            resource=settlement,
        )

    def _complete_list_tools_success(
        self,
        protected: Any,
        result: McpToolListResult,
        *,
        effect_actor: str,
        spec: McpServerSpec,
        effect_context: dict[str, Any],
        resource_context: dict[str, Any],
        request_bytes: int,
        usage_pid: str | None,
    ) -> None:
        result_payload = self._list_tools_success_payload(result)
        settlement = None
        if usage_pid is not None:
            settled_request_bytes = request_bytes
            settled_response_bytes = result.response_bytes
            if spec.schema_version == 2 and result.receipts:
                settled_request_bytes = sum(item.request_bytes for item in result.receipts)
                settled_response_bytes = sum(item.response_bytes for item in result.receipts)
            settlement = ResourceSettlement(
                usage=ResourceUsage(
                    mcp_request_bytes=settled_request_bytes,
                    mcp_response_bytes=settled_response_bytes,
                ),
                source="primitive.mcp.list_tools",
                context={
                    **resource_context,
                    "response_bytes": result.response_bytes,
                },
            )
        protected.complete(
            result,
            self._protected_list_evidence(
                effect_actor,
                spec,
                effect_context,
                result_payload,
            ),
            classification_context=effect_context,
            classification_result=result_payload,
            resource=settlement,
        )

    async def alist_tools(
        self,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Async facade that keeps synchronous providers off the active loop."""

        return await self._data_flow().run_sync_in_worker(
            self.list_tools,
            server_id,
            actor=actor,
            require_capability=require_capability,
            refresh=refresh,
        )

    def unregister_server(
        self,
        server_id: str,
        *,
        actor: str = "runtime",
        require_capability: bool = True,
    ) -> dict[str, Any]:
        authority_decisions: list[Any] = []
        if require_capability:
            authority_decisions.append(
                self.capabilities.require(
                    actor,
                    self.server_resource(server_id),
                    CapabilityRight.ADMIN,
                    consume=False,
                )
            )
        with self._registry_phase_lock, self.capabilities.authority_transaction(
            authority_decisions,
            actor=actor,
            operation="MCP server unregister",
        ):
            self._load_server(server_id)
            self._disable_replaced_server_tool_capabilities(server_id, actor=actor)
            self.extensions.delete_mcp_server(server_id)
            self.events.emit(
                EventType.EXTERNAL_WRITE,
                source=actor,
                target=self.server_resource(server_id),
                payload={"adapter": "mcp", "operation": "server_unregister", "server_id": server_id},
            )
            self.audit.record(
                actor=actor,
                action="mcp.server.unregister",
                target=self.server_resource(server_id),
                decision={"server_id": server_id},
            )
        self._invalidate_modern_server(server_id)
        return {"server_id": server_id, "deleted": True}

    def call_tool(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        arguments: Any = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> McpCallResult | McpComplete[Any] | McpInputRequired | McpRemoteTask:
        try:
            if self._resource_usage_pid(pid) is None:
                raise NotFound(f"process not found: {pid}")
            selected_args = _canonical_mcp_arguments(
                arguments,
                max_bytes=self.config.mcp.max_request_hard_limit_bytes,
            )
            resource = self.tool_resource(server_id, tool_id)
            self._authorize_call_visibility(
                pid,
                resource,
                self._visibility_operation_context(
                    pid,
                    server_id,
                    tool_id,
                    selected_args,
                ),
                source_oids=source_oids,
            )
            registered, _metadata = self._load_server(server_id)
            if isinstance(registered, McpServerManifestV3):
                return self._call_tool_modern(
                    pid,
                    registered,
                    tool_id,
                    selected_args,
                )
            started = time.monotonic()
            deadline = started + registered.timeout_s
            return self._call_tool(
                pid,
                server_id,
                tool_id,
                selected_args,
                source_oids=source_oids,
                started=started,
                deadline=deadline,
            )
        except ProviderEffectNotStarted as error:
            raise self._safe_not_started_error(error) from None

    def _call_tool_modern(
        self,
        pid: str,
        manifest: McpServerManifestV3,
        tool_id: str,
        selected_args: dict[str, Any],
    ) -> McpComplete[Any] | McpInputRequired | McpRemoteTask:
        """Dispatch one Manifest-v3 Tool exactly once through the modern wire.

        Visibility admission has already happened before the manifest was read.
        The protected operation below repeats exact authority admission against
        the manifest-bound right and rejects a registry replacement between the
        two phases.  The wire adapter is reachable only from its provider phase.
        """

        if self._resource_usage_pid(pid) is None:
            raise NotFound(f"process not found: {pid}")
        provider = self._modern_tool_provider
        if provider is None or not callable(getattr(provider, "call_tool", None)):
            raise ValidationError("MCP v3 Tool provider is unavailable")
        tool = manifest.tool_by_id(tool_id)
        if tool is None:
            raise NotFound(f"MCP tool not found: {manifest.server_id}/{tool_id}")
        try:
            right = CapabilityRight(tool.right)
            rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
            rollback_status = (
                ExternalEffectRollbackStatus(tool.rollback_status)
                if tool.rollback_status is not None
                else default_external_effect_rollback_status(rollback_class)
            )
        except (TypeError, ValueError) as error:  # persisted rows fail closed
            raise ValidationError("MCP v3 Tool effect declaration is invalid") from error
        expected_registry_binding = self._registry_binding_for_server_spec(manifest)
        payload = {
            "method": "tools/call",
            "server_id": manifest.server_id,
            "tool_id": tool_id,
            "arguments": selected_args,
        }

        def preflight(current: McpServerManifestV3, deadline: float) -> None:
            current_tool = current.tool_by_id(tool_id)
            if current_tool is None or current_tool != tool:
                raise CapabilityDenied(
                    "MCP Tool declaration changed before provider dispatch"
                )
            validate_mcp_v3_tool_arguments(
                current_tool.input_schema,
                selected_args,
                host_policy=self._v3_host_policy(),
                deadline=deadline,
            )

        def invoke(_client: Any, deadline: float) -> Any:
            context = self._modern_dispatch_context.get()
            if not isinstance(context, dict):  # pragma: no cover - boundary invariant
                raise ProviderEffectNotStarted(
                    "MCP v3 Tool dispatch is outside a protected provider phase"
                )
            binding = context.get("binding")
            if not isinstance(binding, McpClientBinding):
                raise ProviderEffectNotStarted("MCP v3 Tool binding is unavailable")
            sensitive_values = self._modern_provider_sensitive_values(binding)

            async def call_once() -> Any:
                with bind_mcp_client_binding(binding):
                    return await self._await_modern_provider_result(
                        lambda: provider.call_tool(
                            binding.manifest,
                            tool_id,
                            selected_args,
                            deadline=deadline,
                            sensitive_values=sensitive_values,
                        ),
                        deadline=deadline,
                        sensitive_values=sensitive_values,
                    )

            result = _run_mcp_provider_awaitable(call_once())
            if not isinstance(result, (McpComplete, McpInputRequired, McpRemoteTask)):
                raise ValidationError("MCP v3 Tool provider returned an invalid result")
            client = self._modern_client
            limits = getattr(client, "limits", None)
            return sanitize_mcp_operation_result(
                result,
                binding=binding,
                logical_id=tool_id,
                value_type=dict,
                surface="tool",
                limits=limits,
            )

        return self._run_modern_read(
            operation="call",
            server_id=manifest.server_id,
            logical_id=tool_id,
            actor=pid,
            target=self.tool_resource(manifest.server_id, tool_id),
            right=right,
            payload=payload,
            invoke=invoke,
            manifest_preflight=preflight,
            expected_registry_binding=expected_registry_binding,
            state_mutation=tool.state_mutation,
            rollback_class=rollback_class,
            rollback_status=rollback_status,
            contract_name="primitive.mcp.call",
        )

    def _call_tool(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        arguments: Any = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
        started: float, deadline: float,
    ) -> McpCallResult:
        (
            resource,
            selected_args,
            spec,
            tool,
            registry_binding,
            operation_context,
            flow_context,
            sink,
            decision,
            auxiliary_decisions,
            stdio_identity,
            stdio_target_environment,
        ) = self._prepare_tool_call(
            pid,
            server_id=server_id,
            tool_id=tool_id,
            arguments=arguments,
            source_oids=source_oids,
            deadline=deadline,
        )
        (
            request_bytes,
            effect_context,
            list_request_bytes,
            resource_context,
            resource_progress,
        ) = self._legacy_call_accounting(
            spec,
            tool,
            selected_args,
            operation_context,
        )
        runtime_environment: Mapping[str, str] | None = None

        invocation = self._legacy_call_invocation(
            pid=pid,
            resource=resource,
            spec=spec,
            tool=tool,
            registry_binding=registry_binding,
            operation_context=operation_context,
            effect_context=effect_context,
            decision=decision,
            auxiliary_decisions=auxiliary_decisions,
            request_bytes=request_bytes,
            list_request_bytes=list_request_bytes,
            resource_context=resource_context,
            resource_progress=resource_progress,
            sink=sink,
            flow_context=flow_context,
            selected_args=selected_args,
            runtime_environment_getter=lambda: runtime_environment,
            deadline=deadline,
        )
        with self._protected().start("primitive.mcp.call", invocation, provider=self.provider) as protected:
            self._remaining_timeout(deadline)
            runtime_environment = self._require_runtime_environment(
                spec,
                pinned_stdio_environment=stdio_target_environment,
                deadline=deadline,
            )
            self._dispatch_legacy_runtime_resolution(protected, spec, deadline)

            validate_and_call = getattr(self.provider, "validate_and_call", None)
            if callable(validate_and_call):
                executable_snapshot = self._stdio_snapshot_for_dispatch(
                    pid=pid,
                    spec=spec,
                    expected_identity=stdio_identity,
                    sink=sink,
                    context=flow_context,
                    payload=selected_args,
                    runtime_environment=runtime_environment,
                    deadline=deadline,
                )

                invoke_validated_tool = partial(
                    self._invoke_validated_provider_tool,
                    validate_and_call,
                    spec,
                    tool,
                    selected_args,
                    deadline=deadline,
                    pid=pid,
                    runtime_environment=runtime_environment,
                    executable_snapshot=executable_snapshot,
                    started=started,
                )
                provider_outcome, wire_bound_combined_provider = self._dispatch_validated_provider_call(
                    protected,
                    invoke_validated_tool,
                    server=spec,
                    tool=tool,
                    executable_snapshot=executable_snapshot,
                )
                if isinstance(provider_outcome, ProviderEffectNotStartedResult):
                    return self._call_result_from_provider(
                        spec,
                        tool,
                        provider_outcome.result,
                    )
                provider_result = provider_outcome
                result = self._call_result_from_provider(
                    spec,
                    tool,
                    provider_result,
                    validated=True,
                )
                classification_override = self._pre_call_failure_classification_override(
                    spec,
                    tool,
                    provider_result,
                    wire_bound_combined_provider=wire_bound_combined_provider,
                )
                post_call_override = self._post_call_failure_classification_override(
                    tool,
                    provider_result,
                )
                if post_call_override is not None:
                    classification_override = post_call_override
                input_required_override = self._input_required_classification_override(
                    tool,
                    provider_result,
                )
                if input_required_override is not None:
                    classification_override = input_required_override
                return protected.complete(
                    result,
                    self._protected_call_evidence(pid, resource, result, tool, operation_context),
                    classification_context=effect_context,
                    classification_result=self._call_effect_result(result),
                    classification_override=classification_override,
                    resource=self._mcp_exchange_settlement(
                        provider_result,
                        fallback_list_request_bytes=list_request_bytes,
                        fallback_call_request_bytes=request_bytes,
                        server_id=server_id,
                        tool_id=tool_id,
                        status=result.status.value,
                    ),
                )

            live_list_result, validation_error, list_response_bytes = (
                self._dispatch_live_tool_validation(
                    protected,
                    spec,
                    tool,
                    deadline=deadline,
                    pid=pid,
                    expected_identity=stdio_identity,
                    sink=sink,
                    context=flow_context,
                    payload=selected_args,
                    runtime_environment=runtime_environment,
                )
            )
            resource_progress["list_response_bytes"] = list_response_bytes
            if validation_error is not None:
                result = self._live_validation_failure_result(
                    spec,
                    tool,
                    validation_error,
                    duration_s=time.monotonic() - started,
                )
                protected.complete(
                    result,
                    self._protected_call_evidence(pid, resource, result, tool, operation_context),
                    classification_context=effect_context,
                    classification_result=self._call_effect_result(result),
                    resource=self._mcp_live_validation_failure_settlement(
                        spec,
                        request_bytes,
                        result,
                        server_id=server_id,
                        tool_id=tool_id,
                        list_request_bytes=list_request_bytes,
                        live_list_result=live_list_result,
                    ),
                )
                raise validation_error

            deadline_result = self._complete_expired_legacy_exchange(
                protected=protected,
                spec=spec,
                tool=tool,
                operation_context=operation_context,
                effect_context=effect_context,
                live_list_result=live_list_result,
                deadline=deadline,
                started=started,
                list_request_bytes=list_request_bytes,
                request_bytes=request_bytes,
            )
            if deadline_result is not None:
                return deadline_result

            def invoke_tool() -> (
                tuple[McpProviderCallResult, ExternalEffectClassification | None]
                | ProviderEffectNotStartedResult
            ):
                provider_kwargs = self._provider_dispatch_kwargs(
                    spec,
                    deadline=deadline,
                    pid=pid,
                    runtime_environment=runtime_environment,
                    executable_snapshot=executable_snapshot,
                )
                self._remaining_timeout(deadline)
                try:
                    raw_result = self.provider.call_tool(
                        spec,
                        tool,
                        selected_args,
                        **provider_kwargs,
                    )
                    return (
                        self._validated_provider_call_result(
                            spec,
                            raw_result,
                            runtime_environment=runtime_environment,
                        ),
                        None,
                    )
                except ProviderEffectNotStarted as error:
                    return ProviderEffectNotStartedResult(
                        error=error,
                        outcome="call_tool_not_started_after_live_validation",
                        result=McpProviderCallResult(
                            error="provider call did not start",
                            error_type=type(error).__name__,
                            correlation_id=new_id("corr"),
                            duration_s=time.monotonic() - started,
                        ),
                    )
                except ProviderHostError:
                    raise
                except Exception as error:
                    return (
                        McpProviderCallResult(
                            error="provider call failed",
                            error_type=type(error).__name__,
                            correlation_id=new_id("corr"),
                            duration_s=time.monotonic() - started,
                        ),
                        ExternalEffectClassification(
                            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                            state_mutation=tool.state_mutation,
                            information_flow=True,
                            metadata={
                                "outcome": "unknown_provider_exception",
                                "phase": "provider_call",
                                "error_type": type(error).__name__,
                            },
                        ),
                    )

            executable_snapshot = self._stdio_snapshot_for_dispatch(
                pid=pid,
                spec=spec,
                expected_identity=stdio_identity,
                sink=sink,
                context=flow_context,
                payload=selected_args,
                runtime_environment=runtime_environment,
                deadline=deadline,
            )
            try:
                provider_outcome = protected.call(
                    ProviderPhase(
                        "provider_call",
                        state_mutation=tool.state_mutation,
                        information_flow=True,
                    ),
                    invoke_tool,
                )
            finally:
                if executable_snapshot is not None:
                    executable_snapshot.close()
            if isinstance(provider_outcome, ProviderEffectNotStartedResult):
                result = self._call_result_from_provider(spec, tool, provider_outcome.result)
                return result
            provider_result, classification_override = provider_outcome
            result = self._call_result_from_provider(
                spec,
                tool,
                provider_result,
                validated=True,
            )
            input_required_override = self._input_required_classification_override(
                tool,
                provider_result,
            )
            if input_required_override is not None:
                classification_override = input_required_override
            completed = protected.complete(
                result,
                self._protected_call_evidence(pid, resource, result, tool, operation_context),
                classification_context=effect_context,
                classification_result=self._call_effect_result(result),
                classification_override=classification_override,
                resource=self._mcp_legacy_call_completion_settlement(
                    spec,
                    request_bytes,
                    result,
                    provider_result,
                    classification_override,
                    server_id=server_id,
                    tool_id=tool_id,
                    list_request_bytes=list_request_bytes,
                    live_list_result=live_list_result,
                ),
            )
            return completed

    def _legacy_call_accounting(
        self,
        spec: McpServerSpec,
        tool: McpToolSpec,
        selected_args: Mapping[str, Any],
        operation_context: Mapping[str, Any],
    ) -> tuple[int, dict[str, Any], int, dict[str, Any], dict[str, int]]:
        request_bytes = len(
            dumps({"name": tool.mcp_name, "arguments": selected_args}).encode("utf-8")
        )
        if request_bytes > spec.max_request_bytes:
            raise ValidationError(
                f"MCP request exceeds max_request_bytes={spec.max_request_bytes}"
            )
        effect_context = self._effect_context(
            spec,
            tool,
            dict(operation_context),
            request_bytes=request_bytes,
        )
        list_request_bytes = len(
            dumps({"method": "tools/list", "server_id": spec.server_id}).encode("utf-8")
        )
        return (
            request_bytes,
            effect_context,
            list_request_bytes,
            {
                "server_id": spec.server_id,
                "tool_id": tool.tool_id,
                "request_bytes": request_bytes,
                "list_request_bytes": list_request_bytes,
            },
            {"list_response_bytes": 0},
        )

    def _legacy_call_invocation(
        self,
        *,
        pid: str,
        resource: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
        registry_binding: ProviderRegistryBinding,
        operation_context: dict[str, Any],
        effect_context: dict[str, Any],
        decision: Any,
        auxiliary_decisions: list[Any],
        request_bytes: int,
        list_request_bytes: int,
        resource_context: dict[str, Any],
        resource_progress: dict[str, int],
        sink: DataSink,
        flow_context: DataFlowContext,
        selected_args: dict[str, Any],
        runtime_environment_getter: Any,
        deadline: float,
    ) -> ProtectedOperationInvocation:
        return ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=tuple([decision, *auxiliary_decisions]),
            canonical_args=operation_context,
            observation=effect_context,
            reservation_usage=ResourceUsage(
                mcp_request_bytes=(
                    spec.max_request_bytes
                    if spec.schema_version == 2
                    else list_request_bytes + request_bytes
                ),
                mcp_response_bytes=(
                    spec.max_response_bytes
                    if spec.schema_version == 2
                    else spec.max_response_bytes * 2
                ),
            ),
            resource_source="primitive.mcp.call",
            resource_context=resource_context,
            **self._protected_registry_guard(registry_binding, spec.server_id),
            failure_resource=partial(
                self._mcp_call_failure_resource,
                spec=spec,
                request_bytes=request_bytes,
                list_request_bytes=list_request_bytes,
                resource_context=resource_context,
                resource_progress=resource_progress,
            ),
            failure_evidence=lambda error, phase: self._protected_call_failure_evidence(
                pid,
                resource,
                tool,
                operation_context,
                error,
                phase,
            ),
            data_sink=sink,
            data_sink_revalidator=lambda: self._tool_data_sink_after_runtime_resolution(
                spec.server_id,
                spec,
                tool,
                runtime_environment_getter(),
                expected=sink,
                deadline=deadline,
            ),
            data_flow_context=flow_context,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                flow_context,
                origin="external:mcp",
            ),
            data_flow_payload=selected_args,
            data_flow_operation="mcp.call_tool",
        )

    def _dispatch_legacy_runtime_resolution(
        self,
        protected: Any,
        spec: McpServerSpec,
        deadline: float,
    ) -> None:
        if spec.transport != "streamable_http":
            return
        observes_host = self._runtime_resolution_observes_host(spec)
        protected.call(
            ProviderPhase(
                "dns_resolution",
                information_flow=observes_host,
                commits_authority=observes_host,
            ),
            self._validate_runtime_resolution,
            spec,
            deadline=deadline,
        )

    def _prepare_tool_call(
        self,
        pid: str,
        *,
        server_id: str,
        tool_id: str,
        arguments: Any,
        source_oids: list[str] | tuple[str, ...] | None,
        deadline: float,
    ) -> tuple[
        str,
        dict[str, Any],
        McpServerSpec,
        McpToolSpec,
        dict[str, Any],
        dict[str, Any],
        DataFlowContext,
        DataSink,
        Any,
        list[Any],
        dict[str, str] | None,
        Mapping[str, str],
    ]:
        resource, selected_args, spec, tool, registry_binding = (
            self._resolve_tool_call_target(
                pid,
                server_id,
                tool_id,
                arguments,
                source_oids=source_oids,
            )
        )
        self._remaining_timeout(deadline)
        operation_context = self._operation_context(
            pid,
            spec,
            tool,
            selected_args,
            registry_binding=registry_binding,
        )
        flow_context = self._data_flow().context_from_source_oids(pid, source_oids)
        (
            sink,
            decision,
            auxiliary_decisions,
            stdio_identity,
            stdio_target_environment,
        ) = self._authorize_and_resolve_call_sink(
            pid=pid,
            server_id=server_id,
            resource=resource,
            spec=spec,
            tool=tool,
            arguments=selected_args,
            operation_context=operation_context,
            flow_context=flow_context,
            source_oids=source_oids,
            deadline=deadline,
        )
        profile = self.capabilities.profiles.mcp(
            resource=resource,
            effect=decision.effect or CapabilityEffect.DENY,
            server_id=server_id,
            tool_id=tool_id,
        )
        self._attach_call_authority_context(
            operation_context,
            decision=decision,
            auxiliary_decisions=auxiliary_decisions,
            profile=profile,
        )
        self._remaining_timeout(deadline)
        return (
            resource,
            selected_args,
            spec,
            tool,
            registry_binding,
            operation_context,
            flow_context,
            sink,
            decision,
            auxiliary_decisions,
            stdio_identity,
            stdio_target_environment,
        )

    def _attach_call_authority_context(
        self,
        operation_context: dict[str, Any],
        *,
        decision: Any,
        auxiliary_decisions: list[Any],
        profile: Any,
    ) -> None:
        capability_ids = [
            *decision.matched_capability_ids,
            *[
                cap_id
                for auxiliary in auxiliary_decisions
                for cap_id in auxiliary.matched_capability_ids
            ],
        ]
        operation_context.update(
            {
                "capability_ids": list(dict.fromkeys(capability_ids)),
                "selected_capability_id": decision.selected_capability_id,
                "sandbox_profile": self._profile_json(profile),
            }
        )

    def _resolve_tool_call_target(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        arguments: Any,
        *,
        source_oids: list[str] | tuple[str, ...] | None,
    ) -> tuple[
        str,
        dict[str, Any],
        McpServerSpec,
        McpToolSpec,
        dict[str, Any],
    ]:
        resource = self.tool_resource(server_id, tool_id)
        selected_args = _canonical_mcp_arguments(
            arguments,
            max_bytes=self.config.mcp.max_request_hard_limit_bytes,
        )
        visibility_context = self._visibility_operation_context(
            pid,
            server_id,
            tool_id,
            selected_args,
        )
        self._authorize_call_visibility(
            pid,
            resource,
            visibility_context,
            source_oids=source_oids,
        )
        spec, _metadata = self._load_server(server_id)
        # All Manifest v2 modes require the modern provider SPI, even when the
        # configured wire mode is ``legacy``. Keep this check ahead of sink
        # resolution and every DNS, stdio, or provider dispatch path.
        self._require_modern_protocol_provider(spec)
        tool = spec.tool_by_id(tool_id)
        if tool is None:
            raise NotFound(f"MCP tool not found: {server_id}/{tool_id}")
        return (
            resource,
            selected_args,
            spec,
            tool,
            self._registry_binding_for_server_spec(spec),
        )

    async def acall_tool(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        arguments: Any = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> McpCallResult | McpComplete[Any] | McpInputRequired | McpRemoteTask:
        return await self._data_flow().run_sync_in_worker(
            self.call_tool,
            pid,
            server_id,
            tool_id,
            arguments,
            source_oids=source_oids,
        )

    def grant_tool(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        *,
        right: str | CapabilityRight,
        issued_by: str = "mcp",
        delegable: bool = True,
    ) -> Any:
        return self.capabilities.grant(
            subject=pid,
            resource=self.tool_resource(server_id, tool_id),
            rights=[CapabilityRight(str(right))],
            issued_by=issued_by,
            delegable=delegable,
        )

    def _authorize_call(
        self,
        pid: str,
        resource: str,
        right: str,
        context: dict[str, Any],
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        decision = self.capabilities.authorize(pid, resource, right, context, audit=True)
        if decision.allowed:
            return decision
        if decision.policy == CapabilityManager.ASK_EACH_TIME:
            if self.human is None:
                raise CapabilityDenied(f"{pid} requires human approval for MCP call on {resource}")
            profile = self.capabilities.profiles.mcp(
                resource=resource,
                effect=CapabilityEffect.ASK,
                server_id=str(context["server_id"]),
                tool_id=str(context["tool_id"]),
            )
            approval_context = {**context, "sandbox_profile": self._profile_json(profile)}
            request_id = self.human.query_authority_request(
                pid=pid,
                human=self.config.runtime.default_human,
                request={
                    "type": "external_operation_approval",
                    "question": f"Allow this process to call MCP tool {resource}?",
                    "requested_once_capability": {
                        "subject": pid,
                        "resource": resource,
                        "rights": [right],
                        "constraints": self._approval_constraints(context),
                    },
                    "context": approval_context,
                },
                blocking=True,
                authority_origin="external_operation",
                source_oids=source_oids,
            )
            raise HumanApprovalRequired(
                request_id=request_id,
                message=f"{pid} is waiting for per-use human approval to call {resource}",
            )
        raise CapabilityDenied(decision.reason)

    def _authorize_and_resolve_call_sink(
        self,
        *,
        pid: str,
        server_id: str,
        resource: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        operation_context: dict[str, Any],
        flow_context: DataFlowContext,
        source_oids: list[str] | tuple[str, ...] | None,
        deadline: float,
    ) -> tuple[DataSink, Any, list[Any], dict[str, str] | None, Mapping[str, str]]:
        precheck_sink = self._tool_data_sink_for_clearance_precheck(
            server_id,
            spec,
            tool,
        )
        self._data_flow().precheck_egress_clearance(
            pid=pid,
            sink=precheck_sink,
            context=flow_context,
            payload=arguments,
        )
        # A precheck result is never authority: exact ordinary authority and
        # executable-bound authorize_egress below always run independently.
        decision = self._authorize_call(
            pid,
            resource,
            tool.right,
            operation_context,
            source_oids=source_oids,
        )
        auxiliary_decisions = self._require_stdio_process_spawn(
            pid,
            spec,
            consume=False,
        )
        self._validate_arguments_against_schema(
            spec,
            tool,
            arguments,
            deadline=deadline,
        )
        stdio_environment = self._stdio_executable_resolution_environment(
            spec,
            deadline=deadline,
        )
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=stdio_environment,
            deadline=deadline,
        )
        sink = self._tool_data_sink_from_stdio_identity(
            server_id,
            spec,
            tool,
            stdio_identity,
            deadline=deadline,
        )
        self._remaining_timeout(deadline)
        self._data_flow().authorize_egress(
            pid=pid,
            sink=sink,
            context=flow_context,
            payload=arguments,
            operation="mcp.call_tool",
        )
        self._remaining_timeout(deadline)
        return (
            sink,
            decision,
            auxiliary_decisions,
            stdio_identity,
            stdio_environment,
        )

    def _authorize_call_visibility(
        self,
        pid: str,
        resource: str,
        context: dict[str, Any],
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        bound_context: dict[str, Any] | None = None
        for right in (CapabilityRight.READ, CapabilityRight.WRITE, CapabilityRight.EXECUTE):
            selected_context = {**context, "right": str(right)}
            decision = self.capabilities.authorize(pid, resource, right, selected_context)
            if decision.allowed:
                return
            # Keep unprivileged callers ahead of all registry reads. An ASK
            # policy or an existing constrained grant may resolve only the
            # metadata-free digest/generation binding needed for exact use.
            if not (
                decision.policy == CapabilityManager.ASK_EACH_TIME
                or decision.matched_capability_ids
            ):
                continue
            if bound_context is None:
                bound_context = {
                    **context,
                    **self._registry_binding_context(str(context["server_id"])),
                }
            rebound = self.capabilities.authorize(
                pid,
                resource,
                right,
                {**bound_context, "right": str(right)},
            )
            if rebound.allowed:
                return
            if rebound.policy == CapabilityManager.ASK_EACH_TIME:
                self._request_visibility_approval(
                    pid,
                    resource,
                    str(right),
                    bound_context,
                    source_oids=source_oids,
                )
        raise CapabilityDenied(f"{pid} lacks MCP call authority on {resource}")

    def _request_visibility_approval(
        self,
        pid: str,
        resource: str,
        right: str,
        context: dict[str, Any],
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if self.human is None:
            raise CapabilityDenied(f"{pid} requires human approval for MCP call on {resource}")
        profile = self.capabilities.profiles.mcp(
            resource=resource,
            effect=CapabilityEffect.ASK,
            server_id=str(context["server_id"]),
            tool_id=str(context["tool_id"]),
        )
        approval_context = {**context, "right": right, "sandbox_profile": self._profile_json(profile)}
        request_id = self.human.query_authority_request(
            pid=pid,
            human=self.config.runtime.default_human,
            request={
                "type": "external_operation_approval",
                "question": f"Allow this process to call MCP tool {resource}?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [right],
                    "constraints": self._approval_constraints(context),
                },
                "context": approval_context,
            },
            blocking=True,
            authority_origin="external_operation",
            source_oids=source_oids,
        )
        raise HumanApprovalRequired(
            request_id=request_id,
            message=f"{pid} is waiting for per-use human approval to call {resource}",
        )

    def _validate_live_tool(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        *,
        deadline: float,
        pid: str | None = None,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpToolListResult:
        provider_kwargs = self._provider_dispatch_kwargs(
            server,
            deadline=deadline,
            pid=pid,
            runtime_environment=runtime_environment,
            executable_snapshot=executable_snapshot,
        )
        self._remaining_timeout(deadline)
        try:
            raw_result = self.provider.list_tools(
                server,
                **provider_kwargs,
            )
            result = self._validated_tool_list_result(
                server,
                raw_result,
                runtime_environment=runtime_environment,
            )
            live = next(
                (item for item in result.tools if item.name == tool.mcp_name),
                None,
            )
            if live is None:
                raise _McpLiveToolValidationError(
                    f"MCP server {server.server_id} no longer exposes tool {tool.mcp_name}",
                    result,
                )
            if tool.input_schema and not _json_values_equivalent(
                live.input_schema,
                tool.input_schema,
            ):
                raise _McpLiveToolValidationError(
                    f"MCP tool schema changed for {server.server_id}/{tool.tool_id}",
                    result,
                )
        except ProviderEffectNotStarted:
            raise
        except _McpLiveToolValidationError:
            raise
        except ProviderHostError:
            raise
        except Exception as error:
            raise ProviderHostError(
                code="mcp_provider_error",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            ) from None
        return result

    def _invoke_discover_provider(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        pid: str,
        executable_snapshot: ExecutableSnapshot | None,
        runtime_environment: Mapping[str, str],
    ) -> McpProviderDiscoveryResult:
        self._require_modern_protocol_provider(server)
        provider_kwargs = self._provider_dispatch_kwargs(
            server,
            deadline=deadline,
            pid=pid,
            runtime_environment=runtime_environment,
            executable_snapshot=executable_snapshot,
        )
        self._remaining_timeout(deadline)
        try:
            raw_result = self.provider.discover(server, **provider_kwargs)  # type: ignore[attr-defined]
            return self._validated_discovery_result(
                server,
                raw_result,
                runtime_environment=runtime_environment,
            )
        except ProviderEffectNotStarted:
            raise
        except ProviderHostError:
            raise
        except Exception as error:
            raise ProviderHostError(
                code="mcp_provider_error",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            ) from None

    def _validated_discovery_result(
        self,
        server: McpServerSpec,
        result: Any,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpProviderDiscoveryResult:
        try:
            if not isinstance(result, McpProviderDiscoveryResult):
                raise TypeError("MCP provider returned an invalid discovery result")
            self._validate_discovery_result_header(server, result)
            connection = self._validated_connection_info(
                server,
                result.connection,
                runtime_environment=runtime_environment,
            )
            if connection is None:  # pragma: no cover - schema v2 invariant
                raise TypeError("MCP discovery connection metadata is missing")
            receipts = self._validated_exchange_receipts(server, result.receipts)
            self._validate_discovery_receipts(
                server,
                result,
                connection,
                receipts,
            )
            return McpProviderDiscoveryResult(
                connection=connection,
                request_bytes=result.request_bytes,
                response_bytes=result.response_bytes,
                duration_s=float(result.duration_s),
                receipts=receipts,
            )
        except ProviderHostError as error:
            _mark_provider_result_returned(error)
            raise
        except Exception as error:
            raise _mark_provider_result_returned(
                ProviderHostError(
                    code="mcp_provider_error",
                    error_type=type(error).__name__,
                    correlation_id=new_id("corr"),
                )
            ) from None

    @staticmethod
    def _validate_discovery_result_header(
        server: McpServerSpec,
        result: McpProviderDiscoveryResult,
    ) -> None:
        if type(result.request_bytes) is not int or result.request_bytes < 0:
            raise TypeError("MCP discovery request_bytes is invalid")
        if type(result.response_bytes) is not int or result.response_bytes < 0:
            raise TypeError("MCP discovery response_bytes is invalid")
        if result.request_bytes > server.max_request_bytes:
            raise TypeError("MCP discovery exceeds cumulative request budget")
        if result.response_bytes > server.max_response_bytes:
            raise TypeError("MCP discovery exceeds cumulative response budget")
        if (
            type(result.duration_s) not in {int, float}
            or not math.isfinite(result.duration_s)
            or result.duration_s < 0
        ):
            raise TypeError("MCP discovery duration_s is invalid")

    def _validate_discovery_receipts(
        self,
        server: McpServerSpec,
        result: McpProviderDiscoveryResult,
        connection: McpConnectionInfo,
        receipts: tuple[McpExchangeReceipt, ...],
    ) -> None:
        negotiation_end = self._validated_v2_negotiation_prefix(
            server,
            connection,
            receipts,
        )
        if negotiation_end != len(receipts):
            raise TypeError("MCP discovery returned a non-negotiation phase receipt")
        request_total = sum(item.request_bytes for item in receipts)
        response_total = sum(item.response_bytes for item in receipts)
        if result.request_bytes != request_total:
            raise TypeError("MCP discovery request bytes do not match phase receipts")
        if result.response_bytes != response_total:
            raise TypeError("MCP discovery response bytes do not match phase receipts")

    def _invoke_list_tools_provider(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        pid: str,
        executable_snapshot: ExecutableSnapshot | None,
        runtime_environment: Mapping[str, str],
    ) -> tuple[McpToolListResult | None, ProviderHostError | None]:
        # Local deadline/resource/provider-compatibility checks are part of the
        # primitive boundary, not untrusted provider execution. Keep them out of
        # the provider exception mapper so their stable local error types survive.
        provider_kwargs = self._provider_dispatch_kwargs(
            server,
            deadline=deadline,
            pid=pid,
            runtime_environment=runtime_environment,
            executable_snapshot=executable_snapshot,
        )
        self._remaining_timeout(deadline)
        try:
            raw_result = self.provider.list_tools(server, **provider_kwargs)
            return (
                self._validated_tool_list_result(
                    server,
                    raw_result,
                    runtime_environment=runtime_environment,
                ),
                None,
            )
        except ProviderEffectNotStarted:
            raise
        except ProviderHostError as error:
            return None, error
        except Exception as error:
            return None, ProviderHostError(
                code="mcp_provider_error",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            )

    def _dispatch_list_tools(
        self,
        protected: Any,
        server: McpServerSpec,
        *,
        deadline: float,
        pid: str,
        expected_identity: dict[str, str] | None,
        sink: DataSink,
        context: DataFlowContext,
        payload: Any,
        runtime_environment: Mapping[str, str],
    ) -> tuple[McpToolListResult | None, ProviderHostError | None]:
        if server.transport == "streamable_http":
            observes_host = self._runtime_resolution_observes_host(server)
            protected.call(
                ProviderPhase(
                    "dns_resolution",
                    information_flow=observes_host,
                    commits_authority=observes_host,
                ),
                self._validate_runtime_resolution,
                server,
                deadline=deadline,
            )
        executable_snapshot = self._stdio_snapshot_for_dispatch(
            pid=pid,
            spec=server,
            expected_identity=expected_identity,
            sink=sink,
            context=context,
            payload=payload,
            runtime_environment=runtime_environment,
            deadline=deadline,
        )
        try:
            return protected.call(
                ProviderPhase(
                    "provider_not_started_after_dns",
                    information_flow=True,
                ),
                self._invoke_list_tools_provider,
                server,
                deadline=deadline,
                pid=pid,
                executable_snapshot=executable_snapshot,
                runtime_environment=runtime_environment,
            )
        finally:
            if executable_snapshot is not None:
                executable_snapshot.close()

    def _validated_tool_list_result(
        self,
        server: McpServerSpec,
        result: Any,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpToolListResult:
        """Decode every provider-owned tools/list field before it escapes."""

        try:
            if not isinstance(result, McpToolListResult):
                raise TypeError("MCP provider returned an invalid tools/list result")
            server_id = result.server_id
            tools = result.tools
            response_bytes = result.response_bytes
            duration_s = result.duration_s
            self._validate_tool_list_header(
                server,
                server_id=server_id,
                tools=tools,
                response_bytes=response_bytes,
                duration_s=duration_s,
            )
            budget = _ProviderJsonBudget(server.max_response_bytes)
            selected_tools = self._validated_provider_tools(
                tools,
                budget=budget,
                max_tools=self.config.mcp.list_limit,
            )
            canonical_response_bytes = self._canonical_provider_json_bytes(
                [to_jsonable(tool) for tool in selected_tools],
                context="MCP tools/list response",
            )
            if canonical_response_bytes > server.max_response_bytes:
                raise TypeError("MCP tools/list canonical response exceeds max_response_bytes")
            if (
                server.schema_version == 1
                and response_bytes < canonical_response_bytes
            ):
                raise TypeError("MCP tools/list response_bytes underreports canonical response")
            connection = self._validated_connection_info(
                server,
                result.connection,
                runtime_environment=runtime_environment,
            )
            receipts = self._validated_exchange_receipts(
                server,
                result.receipts,
            )
            if server.schema_version == 2:
                if connection is None:  # pragma: no cover - required above
                    raise TypeError("MCP Manifest v2 tools/list connection is missing")
                self._validate_v2_list_receipts(
                    server,
                    connection,
                    receipts,
                    response_bytes=response_bytes,
                )
            else:
                if any(
                    receipt.phase
                    not in {
                        McpExchangePhase.SERVER_DISCOVER,
                        McpExchangePhase.INITIALIZE,
                        McpExchangePhase.TOOLS_LIST,
                    }
                    for receipt in receipts
                ):
                    raise TypeError("MCP tools/list returned an unrelated phase receipt")
                if sum(
                    receipt.phase is McpExchangePhase.TOOLS_LIST
                    for receipt in receipts
                ) > getattr(self, "config", DEFAULT_CONFIG).mcp.list_max_pages:
                    raise TypeError(
                        "MCP tools/list exceeds maximum page receipts="
                        f"{getattr(self, 'config', DEFAULT_CONFIG).mcp.list_max_pages}"
                    )
            sensitive_values = mcp_runtime_secret_values(
                server,
                runtime_environment,
            )
            public_tools = _redact_mcp_provider_tools(
                selected_tools,
                sensitive_values=sensitive_values,
            )
            return McpToolListResult(
                server_id=server_id,
                tools=public_tools,
                response_bytes=response_bytes,
                duration_s=float(duration_s),
                connection=connection,
                receipts=receipts,
            )
        except ProviderHostError as error:
            _mark_provider_result_returned(error)
            raise
        except Exception as error:
            raise _mark_provider_result_returned(
                ProviderHostError(
                    code="mcp_provider_error",
                    error_type=type(error).__name__,
                    correlation_id=new_id("corr"),
                )
            ) from None

    @staticmethod
    def _validate_tool_list_header(
        server: McpServerSpec,
        *,
        server_id: Any,
        tools: Any,
        response_bytes: Any,
        duration_s: Any,
    ) -> None:
        if type(server_id) is not str or server_id != server.server_id:
            raise TypeError("MCP tools/list server identity is invalid")
        if type(tools) is not list:
            raise TypeError("MCP tools/list tools must be a list")
        if (
            type(response_bytes) is not int
            or response_bytes < 0
            or response_bytes > server.max_response_bytes
        ):
            raise TypeError("MCP tools/list response_bytes is invalid")
        if (
            type(duration_s) not in {int, float}
            or not math.isfinite(duration_s)
            or duration_s < 0
        ):
            raise TypeError("MCP tools/list duration_s is invalid")

    def _validated_connection_info(
        self,
        server: McpServerSpec,
        connection: Any,
        *,
        required: bool | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpConnectionInfo | None:
        if connection is None:
            if required is None:
                required = server.schema_version == 2
            if required:
                raise TypeError(
                    "MCP Manifest v2 provider result is missing connection metadata"
                )
            return None
        if not isinstance(connection, McpConnectionInfo):
            raise TypeError("MCP provider connection metadata is invalid")
        mode, era, revision = self._validated_connection_protocol(server, connection)
        self._validate_connection_server_identity(connection)
        sensitive_values = mcp_runtime_secret_values(server, runtime_environment)
        server_name = (
            redact_sensitive_text(
                connection.server_name,
                sensitive_values=sensitive_values,
            )
            if connection.server_name is not None
            else None
        )
        server_version = (
            redact_sensitive_text(
                connection.server_version,
                sensitive_values=sensitive_values,
            )
            if connection.server_version is not None
            else None
        )
        capabilities = self._validated_connection_names(
            connection.capabilities,
            field="capabilities",
            sensitive_values=sensitive_values,
        )
        unsupported = self._validated_connection_names(
            connection.unsupported_capabilities,
            field="unsupported_capabilities",
            sensitive_values=sensitive_values,
        )
        return McpConnectionInfo(
            protocol_mode=mode,
            protocol_era=era,
            protocol_revision=revision,
            sessionless=connection.sessionless,
            fallback_used=connection.fallback_used,
            server_name=server_name,
            server_version=server_version,
            capabilities=capabilities,
            unsupported_capabilities=unsupported,
        )

    def _validated_connection_protocol(
        self,
        server: McpServerSpec,
        connection: McpConnectionInfo,
    ) -> tuple[McpProtocolMode, McpProtocolEra, str]:
        mode = connection.protocol_mode
        era = connection.protocol_era
        revision = connection.protocol_revision
        if type(mode) is not McpProtocolMode or mode != self._effective_protocol_mode(server):
            raise TypeError("MCP provider protocol mode is invalid")
        if type(era) is not McpProtocolEra:
            raise TypeError("MCP provider protocol era is invalid")
        if type(revision) is not str or revision not in _MCP_RELEASE_PROTOCOL_REVISIONS:
            raise TypeError("MCP provider protocol revision is invalid")
        if (
            type(connection.sessionless) is not bool
            or type(connection.fallback_used) is not bool
        ):
            raise TypeError("MCP provider connection flags are invalid")
        self._validate_connection_protocol_semantics(
            mode,
            era,
            revision,
            sessionless=connection.sessionless,
            fallback_used=connection.fallback_used,
        )
        return mode, era, revision

    def _validate_connection_protocol_semantics(
        self,
        mode: McpProtocolMode,
        era: McpProtocolEra,
        revision: str,
        *,
        sessionless: bool,
        fallback_used: bool,
    ) -> None:
        if era is McpProtocolEra.MODERN:
            self._validate_modern_connection_flags(
                revision,
                sessionless=sessionless,
                fallback_used=fallback_used,
            )
        else:
            self._validate_legacy_connection_flags(
                mode,
                revision,
                sessionless=sessionless,
                fallback_used=fallback_used,
            )
        if mode is McpProtocolMode.LEGACY and era is not McpProtocolEra.LEGACY:
            raise TypeError("MCP forced legacy mode negotiated modern")

    @staticmethod
    def _validate_modern_connection_flags(
        revision: str,
        *,
        sessionless: bool,
        fallback_used: bool,
    ) -> None:
        if revision != McpProtocolMode.REVISION_2026_07_28.value:
            raise TypeError("MCP modern connection revision is invalid")
        if not sessionless or fallback_used:
            raise TypeError("MCP modern connection flags are invalid")

    @staticmethod
    def _validate_legacy_connection_flags(
        mode: McpProtocolMode,
        revision: str,
        *,
        sessionless: bool,
        fallback_used: bool,
    ) -> None:
        if revision == McpProtocolMode.REVISION_2026_07_28.value:
            raise TypeError("MCP legacy connection revision is invalid")
        if sessionless:
            raise TypeError("MCP legacy connection cannot be sessionless")
        if mode is McpProtocolMode.REVISION_2026_07_28:
            raise TypeError("MCP pinned modern mode cannot negotiate legacy")
        if mode is McpProtocolMode.AUTO and not fallback_used:
            raise TypeError("MCP auto legacy connection must report fallback")
        if mode is McpProtocolMode.LEGACY and fallback_used:
            raise TypeError("MCP forced legacy connection cannot report fallback")

    def _validate_connection_server_identity(
        self,
        connection: McpConnectionInfo,
    ) -> None:
        mcp_config = getattr(self, "config", DEFAULT_CONFIG).mcp
        for field_name, value in (
            ("server_name", connection.server_name),
            ("server_version", connection.server_version),
        ):
            if value is not None and (
                type(value) is not str
                or len(value) > mcp_config.header_value_max_chars
            ):
                raise TypeError(f"MCP provider {field_name} is invalid")

    def _validated_connection_names(
        self,
        names: Any,
        *,
        field: str,
        sensitive_values: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        mcp_config = getattr(self, "config", DEFAULT_CONFIG).mcp
        if (
            type(names) is not tuple
            or len(names) > mcp_config.provider_capability_limit
        ):
            raise TypeError(f"MCP provider {field} is invalid")
        selected: list[str] = []
        raw_names: list[str] = []
        for name in names:
            if (
                type(name) is not str
                or not name
                or len(name) > mcp_config.mcp_name_max_chars
            ):
                raise TypeError(f"MCP provider {field} is invalid")
            raw_names.append(name)
            sanitized = redact_sensitive_text(
                name,
                sensitive_values=sensitive_values,
            )
            if sanitized not in selected:
                selected.append(sanitized)
        if len(set(raw_names)) != len(raw_names):
            raise TypeError(f"MCP provider {field} contains duplicates")
        return tuple(selected)

    def _validated_exchange_receipts(
        self,
        server: McpServerSpec,
        receipts: Any,
    ) -> tuple[McpExchangeReceipt, ...]:
        if type(receipts) is not tuple:
            raise TypeError("MCP provider receipts must be a tuple")
        mcp_config = getattr(self, "config", DEFAULT_CONFIG).mcp
        if len(receipts) > mcp_config.list_max_pages + 3:
            raise TypeError("MCP provider returned too many exchange receipts")
        selected: list[McpExchangeReceipt] = []
        request_total = 0
        response_total = 0
        for receipt in receipts:
            if not isinstance(receipt, McpExchangeReceipt):
                raise TypeError("MCP provider exchange receipt is invalid")
            if type(receipt.phase) is not McpExchangePhase:
                raise TypeError("MCP provider receipt phase is invalid")
            if type(receipt.request_bytes) is not int or receipt.request_bytes < 0:
                raise TypeError("MCP provider receipt request_bytes is invalid")
            if type(receipt.response_bytes) is not int or receipt.response_bytes < 0:
                raise TypeError("MCP provider receipt response_bytes is invalid")
            if (
                type(receipt.duration_s) not in {int, float}
                or not math.isfinite(receipt.duration_s)
                or receipt.duration_s < 0
            ):
                raise TypeError("MCP provider receipt duration_s is invalid")
            if type(receipt.call_started) is not bool:
                raise TypeError("MCP provider receipt call_started is invalid")
            selected.append(
                McpExchangeReceipt(
                    phase=receipt.phase,
                    request_bytes=receipt.request_bytes,
                    response_bytes=receipt.response_bytes,
                    duration_s=float(receipt.duration_s),
                    call_started=receipt.call_started,
                )
            )
            request_total += receipt.request_bytes
            response_total += receipt.response_bytes
        if server.schema_version == 2:
            if request_total > server.max_request_bytes:
                raise TypeError("MCP provider receipts exceed cumulative request budget")
            if response_total > server.max_response_bytes:
                raise TypeError("MCP provider receipts exceed cumulative response budget")
        return tuple(selected)

    def _validated_v2_negotiation_prefix(
        self,
        server: McpServerSpec,
        connection: McpConnectionInfo,
        receipts: tuple[McpExchangeReceipt, ...],
    ) -> int:
        """Return the first post-negotiation receipt index for Manifest v2.

        Negotiation evidence is deliberately a small grammar rather than an
        unordered bag.  This prevents a provider from hiding automatic
        retries, reporting fallback without an initialize exchange, or
        attributing a later tools exchange to negotiation.
        """

        if server.schema_version != 2:  # pragma: no cover - caller invariant
            raise TypeError("MCP v2 negotiation receipts require Manifest v2")
        if not receipts:
            raise TypeError("MCP Manifest v2 provider result is missing receipts")

        mode = self._effective_protocol_mode(server)
        if mode is McpProtocolMode.LEGACY:
            receipt = receipts[0]
            if (
                receipt.phase is not McpExchangePhase.INITIALIZE
                or not receipt.call_started
            ):
                raise TypeError("MCP legacy negotiation must begin with initialize")
            return 1

        discover_count = 0
        while (
            discover_count < len(receipts)
            and receipts[discover_count].phase is McpExchangePhase.SERVER_DISCOVER
        ):
            if not receipts[discover_count].call_started:
                raise TypeError("MCP server/discover receipt must prove dispatch")
            discover_count += 1
        if discover_count == 0:
            raise TypeError("MCP modern negotiation must begin with server/discover")
        if discover_count > 2:
            raise TypeError("MCP provider exceeded the bounded server/discover retry")

        if connection.fallback_used:
            # A version-negotiation retry can only converge on the pinned
            # modern revision.  Legacy fallback therefore follows exactly one
            # probe and one initialize exchange.
            if discover_count != 1:
                raise TypeError("MCP legacy fallback cannot follow a discover retry")
            if (
                discover_count >= len(receipts)
                or receipts[discover_count].phase is not McpExchangePhase.INITIALIZE
                or not receipts[discover_count].call_started
            ):
                raise TypeError("MCP legacy fallback must include initialize")
            return discover_count + 1

        if (
            discover_count < len(receipts)
            and receipts[discover_count].phase is McpExchangePhase.INITIALIZE
        ):
            raise TypeError("MCP modern negotiation cannot include initialize")
        return discover_count

    def _validate_v2_list_receipts(
        self,
        server: McpServerSpec,
        connection: McpConnectionInfo,
        receipts: tuple[McpExchangeReceipt, ...],
        *,
        response_bytes: int,
    ) -> None:
        negotiation_end = self._validated_v2_negotiation_prefix(
            server,
            connection,
            receipts,
        )
        list_receipts = receipts[negotiation_end:]
        max_pages = getattr(self, "config", DEFAULT_CONFIG).mcp.list_max_pages
        if not 1 <= len(list_receipts) <= max_pages:
            raise TypeError(
                f"MCP tools/list requires 1..{max_pages} page receipts"
            )
        if any(
            receipt.phase is not McpExchangePhase.TOOLS_LIST
            or not receipt.call_started
            for receipt in list_receipts
        ):
            raise TypeError("MCP tools/list page receipts must be contiguous and dispatched")
        if sum(item.response_bytes for item in list_receipts) != response_bytes:
            raise TypeError("MCP tools/list response bytes do not match page receipts")

    def _validate_v2_call_receipts(
        self,
        server: McpServerSpec,
        result: McpProviderCallResult,
        connection: McpConnectionInfo,
        receipts: tuple[McpExchangeReceipt, ...],
    ) -> None:
        negotiation_end = self._validated_v2_negotiation_prefix(
            server,
            connection,
            receipts,
        )
        remaining = receipts[negotiation_end:]
        list_count = 0
        while (
            list_count < len(remaining)
            and remaining[list_count].phase is McpExchangePhase.TOOLS_LIST
        ):
            if not remaining[list_count].call_started:
                raise TypeError("MCP tools/list receipt must prove dispatch")
            list_count += 1
        max_pages = getattr(self, "config", DEFAULT_CONFIG).mcp.list_max_pages
        if not 1 <= list_count <= max_pages:
            raise TypeError(
                f"MCP tool call requires 1..{max_pages} live tools/list page receipts"
            )

        call_receipts = remaining[list_count:]
        if len(call_receipts) > 1 or any(
            receipt.phase is not McpExchangePhase.TOOLS_CALL
            for receipt in call_receipts
        ):
            raise TypeError("MCP tools/call receipt must be unique and terminal")
        if result.call_started:
            if len(call_receipts) != 1 or not call_receipts[0].call_started:
                raise TypeError("MCP dispatched tools/call is missing its terminal receipt")
        elif any(receipt.call_started for receipt in call_receipts):
            raise TypeError("MCP provider claims tools/call did not start after dispatch")

        list_receipts = remaining[:list_count]
        list_request_bytes = sum(item.request_bytes for item in list_receipts)
        list_response_bytes = sum(item.response_bytes for item in list_receipts)
        call_request_bytes = sum(item.request_bytes for item in call_receipts)
        call_response_bytes = sum(item.response_bytes for item in call_receipts)
        expected_fields = (
            ("list_request_bytes", result.list_request_bytes, list_request_bytes),
            ("list_response_bytes", result.list_response_bytes, list_response_bytes),
            ("call_request_bytes", result.call_request_bytes, call_request_bytes),
            ("call_response_bytes", result.call_response_bytes, call_response_bytes),
            ("response_bytes", result.response_bytes, call_response_bytes),
        )
        for field_name, actual, expected in expected_fields:
            if actual != expected:
                raise TypeError(
                    f"MCP provider {field_name} does not match phase receipts"
                )

    @staticmethod
    def _validated_provider_tools(
        tools: list[Any],
        *,
        budget: _ProviderJsonBudget,
        max_tools: int,
    ) -> list[McpProviderTool]:
        if len(tools) > max_tools:
            raise TypeError(f"MCP tools/list exceeds maximum tool count={max_tools}")
        selected_tools: list[McpProviderTool] = []
        names: set[str] = set()
        for index, item in enumerate(tools):
            budget.consume_node(path=f"$.tools[{index}]", depth=1)
            if not isinstance(item, McpProviderTool):
                raise TypeError("MCP tools/list contains an invalid tool")
            name = item.name
            description = item.description
            input_schema = item.input_schema
            metadata = item.metadata
            if type(name) is not str or not name:
                raise TypeError("MCP tools/list tool name is invalid")
            budget.consume_string(name, path=f"$.tools[{index}].name")
            if name in names:
                raise TypeError("MCP tools/list contains duplicate tool names")
            if description is not None and type(description) is not str:
                raise TypeError("MCP tools/list tool description is invalid")
            if description is not None:
                budget.consume_string(
                    description,
                    path=f"$.tools[{index}].description",
                )
            if type(input_schema) is not dict or type(metadata) is not dict:
                raise TypeError("MCP tools/list tool metadata is invalid")
            selected_tools.append(
                McpProviderTool(
                    name=name,
                    description=description,
                    input_schema=_strict_provider_json_value(
                        input_schema,
                        path=f"$.tools[{index}].input_schema",
                        active_containers=set(),
                        budget=budget,
                        depth=2,
                    ),
                    metadata=_strict_provider_json_value(
                        metadata,
                        path=f"$.tools[{index}].metadata",
                        active_containers=set(),
                        budget=budget,
                        depth=2,
                    ),
                )
            )
            names.add(name)
        return selected_tools

    @staticmethod
    def _canonical_provider_json_bytes(value: Any, *, context: str) -> int:
        try:
            return len(dumps(value).encode("utf-8"))
        except (TypeError, ValueError, RecursionError, UnicodeEncodeError) as error:
            raise TypeError(f"{context} is not canonical JSON") from error

    @staticmethod
    def _validate_provider_call_header(
        server: McpServerSpec,
        result: McpProviderCallResult,
    ) -> None:
        if type(result.is_error) is not bool or type(result.too_large) is not bool:
            raise TypeError("MCP provider call flags are invalid")
        if type(result.call_started) is not bool:
            raise TypeError("MCP provider call_started is invalid")
        byte_fields = (
            ("response_bytes", result.response_bytes),
            ("list_request_bytes", result.list_request_bytes),
            ("list_response_bytes", result.list_response_bytes),
            ("call_request_bytes", result.call_request_bytes),
            ("call_response_bytes", result.call_response_bytes),
        )
        for field_name, selected in byte_fields:
            if type(selected) is not int or selected < 0:
                raise TypeError(f"MCP provider {field_name} is invalid")
        for field_name, selected in (
            ("response_bytes", result.response_bytes),
            ("list_response_bytes", result.list_response_bytes),
            ("call_response_bytes", result.call_response_bytes),
        ):
            if selected > server.max_response_bytes:
                raise TypeError(
                    f"MCP provider {field_name} exceeds max_response_bytes"
                )
        if (
            type(result.duration_s) not in {int, float}
            or not math.isfinite(result.duration_s)
            or result.duration_s < 0
        ):
            raise TypeError("MCP provider duration_s is invalid")
        for field_name, selected in (
            ("error", result.error),
            ("error_type", result.error_type),
            ("correlation_id", result.correlation_id),
        ):
            if selected is not None and type(selected) is not str:
                raise TypeError(f"MCP provider {field_name} is invalid")
        if (
            result.error_type == "mcp_input_required_unsupported"
            and not result.call_started
        ):
            raise TypeError("MCP input_required must follow a dispatched tools/call")

    @staticmethod
    def _validated_provider_call_payloads(
        server: McpServerSpec,
        result: McpProviderCallResult,
    ) -> tuple[Any, Any]:
        budget = _ProviderJsonBudget(server.max_response_bytes)
        selected_content = _strict_provider_json_value(
            result.content,
            path="$.content",
            active_containers=set(),
            budget=budget,
        )
        selected_structured_content = _strict_provider_json_value(
            result.structured_content,
            path="$.structured_content",
            active_containers=set(),
            budget=budget,
        )
        for field_name, selected in (
            ("error", result.error),
            ("error_type", result.error_type),
            ("correlation_id", result.correlation_id),
        ):
            if selected is not None:
                budget.consume_string(selected, path=f"$.{field_name}")
        return selected_content, selected_structured_content

    @staticmethod
    def _validate_provider_call_byte_contract(
        server: McpServerSpec,
        result: McpProviderCallResult,
        *,
        selected_content: Any,
        selected_structured_content: Any,
    ) -> None:
        has_response_payload = (
            result.error is None
            or result.content is not None
            or result.structured_content is not None
        )
        canonical_response_bytes = 0
        if has_response_payload:
            canonical_response_bytes = McpPrimitive._canonical_provider_json_bytes(
                {
                    "content": selected_content,
                    "structured_content": selected_structured_content,
                },
                context="MCP tool response",
            )
            if canonical_response_bytes > server.max_response_bytes:
                raise TypeError("MCP tool canonical response exceeds max_response_bytes")
        # Manifest v1 byte fields predate phase receipts and retain their
        # released canonical-projection lower-bound contract. Manifest v2
        # fields are exact raw-wire measurements and are checked against the
        # phase receipts below; a decoded/canonical JSON projection can be
        # larger than its wire representation without implying undercounting.
        if server.schema_version == 1:
            if result.too_large:
                McpPrimitive._validate_oversized_provider_call_bytes(server, result)
            elif has_response_payload:
                McpPrimitive._validate_canonical_provider_call_bytes(
                    result,
                    canonical_response_bytes=canonical_response_bytes,
                )

    @staticmethod
    def _validate_oversized_provider_call_bytes(
        server: McpServerSpec,
        result: McpProviderCallResult,
    ) -> None:
        if result.response_bytes < server.max_response_bytes:
            raise TypeError(
                "MCP provider response_bytes underreports an oversized response"
            )
        if (
            result.call_response_bytes
            and result.call_response_bytes < server.max_response_bytes
        ):
            raise TypeError(
                "MCP provider call_response_bytes underreports an oversized response"
            )

    @staticmethod
    def _validate_canonical_provider_call_bytes(
        result: McpProviderCallResult,
        *,
        canonical_response_bytes: int,
    ) -> None:
        if result.response_bytes < canonical_response_bytes:
            raise TypeError(
                "MCP provider response_bytes underreports canonical response"
            )
        if (
            result.call_response_bytes
            and result.call_response_bytes < canonical_response_bytes
        ):
            raise TypeError(
                "MCP provider call_response_bytes underreports canonical response"
            )

    def _validated_provider_call_result(
        self,
        server: McpServerSpec,
        result: Any,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> McpProviderCallResult:
        """Decode every provider-owned call field into an inert value object."""

        try:
            if not isinstance(result, McpProviderCallResult):
                raise TypeError("MCP provider returned an invalid call result")
            McpPrimitive._validate_provider_call_header(server, result)
            selected_content, selected_structured_content = (
                McpPrimitive._validated_provider_call_payloads(server, result)
            )
            McpPrimitive._validate_provider_call_byte_contract(
                server,
                result,
                selected_content=selected_content,
                selected_structured_content=selected_structured_content,
            )
            receipts = self._validated_exchange_receipts(server, result.receipts)
            incomplete_negotiation_failure = (
                server.schema_version == 2
                and result.connection is None
                and self._validate_v2_incomplete_negotiation_failure(
                    server,
                    result,
                    receipts,
                )
            )
            connection = self._validated_connection_info(
                server,
                result.connection,
                required=(
                    server.schema_version == 2
                    and not incomplete_negotiation_failure
                ),
                runtime_environment=runtime_environment,
            )
            if server.schema_version == 2:
                if connection is not None:
                    negotiated_pre_list_failure = (
                        self._validate_v2_negotiated_pre_list_failure(
                            server,
                            result,
                            connection,
                            receipts,
                        )
                    )
                    if not negotiated_pre_list_failure:
                        self._validate_v2_call_receipts(
                            server,
                            result,
                            connection,
                            receipts,
                        )
            else:
                if any(
                    receipt.phase
                    not in {
                        McpExchangePhase.SERVER_DISCOVER,
                        McpExchangePhase.INITIALIZE,
                        McpExchangePhase.TOOLS_LIST,
                        McpExchangePhase.TOOLS_CALL,
                    }
                    for receipt in receipts
                ):
                    raise TypeError("MCP tool call returned an unrelated phase receipt")
                if sum(
                    receipt.phase is McpExchangePhase.TOOLS_LIST
                    for receipt in receipts
                ) > getattr(self, "config", DEFAULT_CONFIG).mcp.list_max_pages:
                    raise TypeError("MCP tool call exceeds maximum tools/list pages")
                if sum(
                    receipt.phase is McpExchangePhase.TOOLS_CALL
                    for receipt in receipts
                ) > 1:
                    raise TypeError("MCP provider automatically retried tools/call")
            sensitive_values = mcp_runtime_secret_values(
                server,
                runtime_environment,
            )
            public_content = _redact_mcp_provider_json(
                selected_content,
                sensitive_values=sensitive_values,
            )
            public_structured_content = _redact_mcp_provider_json(
                selected_structured_content,
                sensitive_values=sensitive_values,
            )
            return McpProviderCallResult(
                content=public_content,
                structured_content=public_structured_content,
                is_error=result.is_error,
                error=(
                    redact_sensitive_text(
                        result.error,
                        sensitive_values=sensitive_values,
                    )
                    if result.error is not None
                    else None
                ),
                response_bytes=result.response_bytes,
                duration_s=float(result.duration_s),
                too_large=result.too_large,
                error_type=(
                    redact_sensitive_text(
                        result.error_type,
                        sensitive_values=sensitive_values,
                    )
                    if result.error_type is not None
                    else None
                ),
                correlation_id=(
                    redact_sensitive_text(
                        result.correlation_id,
                        sensitive_values=sensitive_values,
                    )
                    if result.correlation_id is not None
                    else None
                ),
                list_request_bytes=result.list_request_bytes,
                list_response_bytes=result.list_response_bytes,
                call_request_bytes=result.call_request_bytes,
                call_response_bytes=result.call_response_bytes,
                call_started=result.call_started,
                connection=connection,
                receipts=receipts,
            )
        except ProviderHostError as error:
            _mark_provider_result_returned(error)
            raise
        except Exception as error:
            raise _mark_provider_result_returned(
                ProviderHostError(
                    code="mcp_provider_error",
                    error_type=type(error).__name__,
                    correlation_id=new_id("corr"),
                )
            ) from None

    def _validate_v2_incomplete_negotiation_failure(
        self,
        server: McpServerSpec,
        result: McpProviderCallResult,
        receipts: tuple[McpExchangeReceipt, ...],
    ) -> bool:
        """Accept only built-in wire proof that negotiation failed pre-call."""

        if type(self.provider) is not SdkMcpProvider:
            raise TypeError("MCP Manifest v2 tool-call connection is missing")
        if (
            not result.error
            or result.content is not None
            or result.structured_content is not None
            or result.is_error
            or result.too_large
            or result.call_started
            or any(
                value != 0
                for value in (
                    result.response_bytes,
                    result.list_request_bytes,
                    result.list_response_bytes,
                    result.call_request_bytes,
                    result.call_response_bytes,
                )
            )
            or any(not receipt.call_started for receipt in receipts)
        ):
            raise TypeError("MCP incomplete negotiation failure is invalid")

        phases = tuple(receipt.phase for receipt in receipts)
        mode = self._effective_protocol_mode(server)
        if mode is McpProtocolMode.LEGACY:
            valid_phases = phases in {(), (McpExchangePhase.INITIALIZE,)}
        elif mode is McpProtocolMode.REVISION_2026_07_28:
            valid_phases = phases in {
                (),
                (McpExchangePhase.SERVER_DISCOVER,),
                (
                    McpExchangePhase.SERVER_DISCOVER,
                    McpExchangePhase.SERVER_DISCOVER,
                ),
            }
        else:
            valid_phases = phases in {
                (),
                (McpExchangePhase.SERVER_DISCOVER,),
                (
                    McpExchangePhase.SERVER_DISCOVER,
                    McpExchangePhase.SERVER_DISCOVER,
                ),
                (
                    McpExchangePhase.SERVER_DISCOVER,
                    McpExchangePhase.INITIALIZE,
                ),
            }
        if not valid_phases:
            raise TypeError(
                "MCP incomplete negotiation receipts are not a legal prefix"
            )
        return True

    def _validate_v2_negotiated_pre_list_failure(
        self,
        server: McpServerSpec,
        result: McpProviderCallResult,
        connection: McpConnectionInfo,
        receipts: tuple[McpExchangeReceipt, ...],
    ) -> bool:
        """Accept the built-in wire-certified prefix before first tools/list.

        A custom provider cannot self-certify this narrower dispatch state. A
        built-in failure after any list page is handled by the normal call
        receipt grammar instead.
        """

        if (
            type(getattr(self, "provider", None)) is not SdkMcpProvider
            or result.error_type != "McpPreCallFailure"
        ):
            return False
        negotiation_end = self._validated_v2_negotiation_prefix(
            server,
            connection,
            receipts,
        )
        if negotiation_end != len(receipts):
            return False
        if (
            not result.error
            or result.content is not None
            or result.structured_content is not None
            or result.is_error
            or result.too_large
            or result.call_started
            or any(
                value != 0
                for value in (
                    result.response_bytes,
                    result.list_request_bytes,
                    result.list_response_bytes,
                    result.call_request_bytes,
                    result.call_response_bytes,
                )
            )
        ):
            raise TypeError("MCP negotiated pre-list failure is invalid")
        return True

    def _validate_live_tool_for_call(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        *,
        deadline: float,
        pid: str,
        executable_snapshot: ExecutableSnapshot | None,
        runtime_environment: Mapping[str, str] | None,
    ) -> tuple[McpToolListResult | None, Exception | None, int]:
        """Retain known list bytes while keeping not-started failures exceptional."""

        try:
            result = self._validate_live_tool(
                server,
                tool,
                deadline=deadline,
                pid=pid,
                executable_snapshot=executable_snapshot,
                runtime_environment=runtime_environment,
            )
        except _McpLiveToolValidationError as error:
            return error.result, error, error.result.response_bytes
        except ProviderEffectNotStarted:
            raise
        except Exception as error:
            return None, error, 0
        return result, None, result.response_bytes

    def _dispatch_live_tool_validation(
        self,
        protected: Any,
        server: McpServerSpec,
        tool: McpToolSpec,
        *,
        deadline: float,
        pid: str,
        expected_identity: dict[str, str] | None,
        sink: DataSink,
        context: DataFlowContext,
        payload: Any,
        runtime_environment: Mapping[str, str],
    ) -> tuple[McpToolListResult | None, Exception | None, int]:
        executable_snapshot = self._stdio_snapshot_for_dispatch(
            pid=pid,
            spec=server,
            expected_identity=expected_identity,
            sink=sink,
            context=context,
            payload=payload,
            runtime_environment=runtime_environment,
            deadline=deadline,
        )
        try:
            return protected.call(
                ProviderPhase(
                    "live_validation_not_started_after_dns",
                    information_flow=True,
                ),
                lambda: self._validate_live_tool_for_call(
                    server,
                    tool,
                    deadline=deadline,
                    pid=pid,
                    executable_snapshot=executable_snapshot,
                    runtime_environment=runtime_environment,
                ),
            )
        finally:
            if executable_snapshot is not None:
                executable_snapshot.close()

    @staticmethod
    def _remaining_timeout(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderEffectNotStarted(
                "MCP deadline exhausted before provider dispatch"
            )
        return remaining

    def _provider_dispatch_kwargs(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        pid: str | None,
        runtime_environment: Mapping[str, str] | None,
        executable_snapshot: ExecutableSnapshot | None,
    ) -> dict[str, Any]:
        selected: dict[str, Any] = {
            "timeout_s": self._remaining_timeout(deadline),
            "max_response_bytes": server.max_response_bytes,
            "runtime_environment": runtime_environment,
        }
        if (
            isinstance(self.provider, McpAbsoluteDeadlineProvider)
            and self.provider.supports_mcp_absolute_deadline is True
        ):
            selected["deadline"] = deadline
        if executable_snapshot is not None:
            selected["executable_snapshot"] = executable_snapshot
        if server.transport == "stdio":
            usage_pid = self._resource_usage_pid(pid)
            limits = (
                self._subprocess_limits(usage_pid)
                if usage_pid is not None
                else None
            )
            if limits is not None:
                if (
                    not isinstance(
                        self.provider,
                        McpSubprocessLimitsProvider,
                    )
                    or self.provider.supports_subprocess_limits is not True
                ):
                    raise ValidationError(
                        "MCP provider must explicitly support SubprocessLimits "
                        "before budgeted stdio execution"
                    )
                selected["limits"] = limits
        return selected

    def _subprocess_limits(self, pid: str) -> SubprocessLimits | None:
        if self.resources is None:
            return None
        wall = self.resources.remaining_cumulative(
            pid,
            "max_subprocess_wall_seconds",
            "subprocess_wall_seconds",
        )
        cpu = self.resources.remaining_cumulative(
            pid,
            "max_subprocess_cpu_seconds",
            "subprocess_cpu_seconds",
        )
        memory = self.resources.peak_limit(pid, "max_subprocess_memory_bytes")
        if wall is not None and wall <= 0:
            raise ResourceLimitExceeded(
                f"process {pid} exhausted subprocess wall-time budget"
            )
        if cpu is not None and cpu <= 0:
            raise ResourceLimitExceeded(
                f"process {pid} exhausted subprocess CPU budget"
            )
        if memory is not None and memory <= 0:
            raise ResourceLimitExceeded(
                f"process {pid} exhausted subprocess memory budget"
            )
        if wall is None and cpu is None and memory is None:
            return None
        return SubprocessLimits(
            wall_seconds=wall,
            cpu_seconds=cpu,
            memory_bytes=memory,
        )

    @staticmethod
    def _safe_not_started_error(error: ProviderEffectNotStarted) -> ProviderHostError:
        return ProviderHostError(
            code="mcp_provider_not_started",
            error_type=type(error).__name__,
            correlation_id=new_id("corr"),
        )

    def _require_stdio_process_spawn(
        self,
        actor: str | None,
        server: McpServerSpec,
        *,
        consume: bool = True,
    ) -> list[Any]:
        if actor is None or server.transport != "stdio":
            return []
        decisions = [
            self.capabilities.require(
                actor,
                "process:spawn",
                CapabilityRight.WRITE,
                consume=consume,
            )
        ]
        if server.stdio is None:
            raise ValidationError("MCP stdio transport is missing stdio configuration")
        resource = self.stdio_resource_for_server(server)
        if resource is None:
            raise ValidationError("MCP stdio transport is missing stdio authority resource")
        decisions.append(
            self.capabilities.require(
                actor,
                resource,
                CapabilityRight.EXECUTE,
                {
                    "adapter": "mcp",
                    "operation": "mcp.stdio.spawn",
                    "server_id": server.server_id,
                    "stdio_command": server.stdio.command,
                    "stdio_args": list(server.stdio.args),
                    "stdio_env": dict(server.stdio.env),
                    "stdio_cwd": self._canonical_stdio_cwd(server.stdio.cwd),
                },
                consume=consume,
            )
        )
        return decisions

    def _call_result_from_provider(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        provider_result: McpProviderCallResult,
        *,
        validated: bool = False,
    ) -> McpCallResult:
        if not validated:
            provider_result = self._validated_provider_call_result(
                server,
                provider_result,
            )
        dispatch_state = self._public_call_dispatch_state(server, provider_result)
        retry_class = self._public_call_retry_class(
            ok=not provider_result.error
            and not provider_result.is_error
            and not provider_result.too_large,
            dispatch_state=dispatch_state,
        )
        if provider_result.error_type == "mcp_input_required_unsupported":
            return McpCallResult(
                server_id=server.server_id,
                tool_id=tool.tool_id,
                mcp_name=tool.mcp_name,
                status=McpCallStatus.INPUT_REQUIRED_UNSUPPORTED,
                ok=False,
                error={
                    "code": "mcp_input_required_unsupported",
                    "error_type": "mcp_input_required_unsupported",
                    "correlation_id": provider_result.correlation_id or new_id("corr"),
                    "message": (
                        "MCP server requested multi-round input, which is not "
                        "supported by this release"
                    ),
                    "retryable": False,
                    "automatic_retry_disabled": True,
                    "continuation_present": True,
                },
                response_bytes=provider_result.response_bytes,
                duration_s=provider_result.duration_s,
                connection=provider_result.connection,
                receipts=provider_result.receipts,
                dispatch_state=McpDispatchState.STARTED,
                retry_class=McpRetryClass.UNSAFE_OR_UNKNOWN,
            )
        if provider_result.error:
            safe_message = self._safe_transport_error_message(
                server,
                provider_result.error_type,
            )
            return McpCallResult(
                server_id=server.server_id,
                tool_id=tool.tool_id,
                mcp_name=tool.mcp_name,
                status=McpCallStatus.TRANSPORT_ERROR,
                ok=False,
                error={
                    "code": "mcp_provider_error",
                    "error_type": provider_result.error_type or "TransportError",
                    "correlation_id": provider_result.correlation_id or new_id("corr"),
                    "retryable": False,
                    "automatic_retry_disabled": True,
                    **(
                        {"message": safe_message}
                        if safe_message is not None
                        else {}
                    ),
                },
                response_bytes=provider_result.response_bytes,
                duration_s=provider_result.duration_s,
                connection=provider_result.connection,
                receipts=provider_result.receipts,
                dispatch_state=dispatch_state,
                retry_class=retry_class,
            )
        if provider_result.too_large:
            return self._failure(
                server,
                tool,
                McpCallStatus.RESPONSE_TOO_LARGE,
                f"response exceeded max_response_bytes={server.max_response_bytes}",
                provider_result,
            )
        if provider_result.is_error:
            # Project only the returned/model-facing payload.  Provider byte
            # receipts and settlement fields remain unchanged.
            projected_error = _model_facing_mcp_call_payload(
                provider_result.content,
                provider_result.structured_content,
            )
            return self._failure(
                server,
                tool,
                McpCallStatus.MCP_ERROR,
                "MCP tool returned an error result",
                provider_result,
                extra={
                    key: value
                    for key, value in projected_error.items()
                    if value is not None
                },
            )
        projected = _model_facing_mcp_call_payload(
            provider_result.content,
            provider_result.structured_content,
        )
        return McpCallResult(
            server_id=server.server_id,
            tool_id=tool.tool_id,
            mcp_name=tool.mcp_name,
            status=McpCallStatus.OK,
            ok=True,
            result=to_jsonable(projected),
            response_bytes=provider_result.response_bytes,
            duration_s=provider_result.duration_s,
            connection=provider_result.connection,
            receipts=provider_result.receipts,
            dispatch_state=McpDispatchState.STARTED,
            retry_class=McpRetryClass.NOT_APPLICABLE,
        )

    def _public_call_dispatch_state(
        self,
        server: McpServerSpec,
        provider_result: McpProviderCallResult,
    ) -> McpDispatchState:
        """Project only dispatch evidence trusted by the effect classifier."""

        if (
            provider_result.call_started
            or provider_result.error is None
            or provider_result.is_error
        ):
            return McpDispatchState.STARTED
        if server.schema_version == 1:
            # Manifest v1 retains its released Provider SPI certificate.
            return McpDispatchState.NOT_STARTED
        if (
            type(getattr(self, "provider", None)) is SdkMcpProvider
            and provider_result.call_request_bytes == 0
            and provider_result.call_response_bytes == 0
            and provider_result.response_bytes == 0
            and all(
                receipt.phase is not McpExchangePhase.TOOLS_CALL
                for receipt in provider_result.receipts
            )
        ):
            return McpDispatchState.NOT_STARTED
        return McpDispatchState.UNKNOWN

    @staticmethod
    def _public_call_retry_class(
        *,
        ok: bool,
        dispatch_state: McpDispatchState,
    ) -> McpRetryClass:
        if ok:
            return McpRetryClass.NOT_APPLICABLE
        if dispatch_state is McpDispatchState.NOT_STARTED:
            return McpRetryClass.REOBSERVE_REQUIRED
        return McpRetryClass.UNSAFE_OR_UNKNOWN

    def _invoke_validated_provider_tool(
        self,
        validate_and_call: Any,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        deadline: float,
        pid: str,
        runtime_environment: Mapping[str, str] | None,
        executable_snapshot: ExecutableSnapshot | None,
        started: float,
    ) -> McpProviderCallResult | ProviderEffectNotStartedResult:
        # Build kwargs before entering the untrusted provider boundary. Local
        # resource/deadline/compatibility failures must not become provider errors.
        provider_kwargs = self._provider_dispatch_kwargs(
            server,
            deadline=deadline,
            pid=pid,
            runtime_environment=runtime_environment,
            executable_snapshot=executable_snapshot,
        )
        self._remaining_timeout(deadline)
        try:
            raw_result = validate_and_call(
                server,
                tool,
                arguments,
                **provider_kwargs,
            )
            return self._validated_provider_call_result(
                server,
                raw_result,
                runtime_environment=runtime_environment,
            )
        except ProviderEffectNotStarted as error:
            return ProviderEffectNotStartedResult(
                error=error,
                outcome="validate_and_call_not_started",
                result=McpProviderCallResult(
                    error="provider call did not start",
                    error_type=type(error).__name__,
                    correlation_id=new_id("corr"),
                    duration_s=time.monotonic() - started,
                    call_started=False,
                ),
            )
        except Exception as error:
            post_call_failure = self._wire_bound_post_call_failure_result(
                server,
                error,
                started=started,
                runtime_environment=runtime_environment,
            )
            if post_call_failure is not None:
                return post_call_failure
            if isinstance(error, ProviderHostError):
                raise
            raise ProviderHostError(
                code="mcp_provider_error",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            ) from None

    def _wire_bound_post_call_failure_result(
        self,
        server: McpServerSpec,
        error: BaseException,
        *,
        started: float,
        runtime_environment: Mapping[str, str] | None,
    ) -> McpProviderCallResult | None:
        if server.schema_version != 2 or type(self.provider) is not SdkMcpProvider:
            return None
        certified, receipts, connection = _wire_failure_evidence(error)
        call_receipts = tuple(
            receipt
            for receipt in receipts
            if receipt.phase is McpExchangePhase.TOOLS_CALL
        )
        if (
            not certified
            or connection is None
            or len(call_receipts) != 1
            or not call_receipts[0].call_started
        ):
            return None
        list_receipts = tuple(
            receipt
            for receipt in receipts
            if receipt.phase is McpExchangePhase.TOOLS_LIST
        )
        result = McpProviderCallResult(
            error="MCP tools/call dispatch outcome is unknown",
            error_type="McpPostCallFailure",
            correlation_id=new_id("corr"),
            response_bytes=call_receipts[0].response_bytes,
            duration_s=max(0.0, time.monotonic() - started),
            list_request_bytes=sum(item.request_bytes for item in list_receipts),
            list_response_bytes=sum(item.response_bytes for item in list_receipts),
            call_request_bytes=call_receipts[0].request_bytes,
            call_response_bytes=call_receipts[0].response_bytes,
            call_started=True,
            connection=connection,
            receipts=receipts,
        )
        return self._validated_provider_call_result(
            server,
            result,
            runtime_environment=runtime_environment,
        )

    def _pre_call_failure_classification_override(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        provider_result: McpProviderCallResult,
        *,
        wire_bound_combined_provider: bool,
    ) -> ExternalEffectClassification | None:
        if provider_result.call_started:
            return None

        # Manifest v1 retains its released Provider SPI: call_started=False is
        # the provider's pre-dispatch certificate.  Manifest v2 has exact phase
        # receipts, so require the stronger wire-bound certificate before a
        # mutating tool can be narrowed to an external-read failure.
        certified_pre_call = server.schema_version == 1 or (
            wire_bound_combined_provider
            and provider_result.call_request_bytes == 0
            and provider_result.call_response_bytes == 0
            and provider_result.response_bytes == 0
            and all(
                receipt.phase is not McpExchangePhase.TOOLS_CALL
                for receipt in provider_result.receipts
            )
        )
        if certified_pre_call:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={
                    "outcome": "failed",
                    "failure_kind": "live_validation_failed_before_call",
                    "phase": "live_validation",
                    "call_started": False,
                    "tools_call_receipt_present": False,
                },
            )

        # A v2 result which claims no call but carries call bytes or a
        # tools/call receipt is not a pre-dispatch certificate.  Preserve an
        # unknown mutation for mutating/rollback-unclear tools so an ambiguous
        # provider result can never enable replay.
        rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
        if (
            tool.state_mutation
            or rollback_class is not ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        ):
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                state_mutation=tool.state_mutation,
                information_flow=True,
                metadata={
                    "outcome": "unknown_pre_call_dispatch_state",
                    "phase": "tools/call",
                    "call_started": False,
                    "tools_call_receipt_present": any(
                        receipt.phase is McpExchangePhase.TOOLS_CALL
                        for receipt in provider_result.receipts
                    ),
                },
            )
        return None

    @staticmethod
    def _post_call_failure_classification_override(
        tool: McpToolSpec,
        provider_result: McpProviderCallResult,
    ) -> ExternalEffectClassification | None:
        # Any provider failure after tools/call dispatch leaves a mutating
        # tool's external state ambiguous, including bounded transport errors.
        if provider_result.error is None or not provider_result.call_started:
            return None
        rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
        if (
            not tool.state_mutation
            and rollback_class is ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        ):
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={
                    "outcome": "failed",
                    "failure_kind": "mcp_post_call_read_failure",
                    "phase": "tools/call",
                    "call_started": True,
                    "automatic_retry_disabled": True,
                },
            )
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
            state_mutation=tool.state_mutation,
            information_flow=True,
            metadata={
                "outcome": "unknown_mcp_post_call_failure",
                "failure_kind": "mcp_post_call_failure",
                "phase": "tools/call",
                "call_started": True,
                "automatic_retry_disabled": True,
            },
        )

    def _dispatch_validated_provider_call(
        self,
        protected: Any,
        invoke_validated_tool: Any,
        *,
        server: McpServerSpec,
        tool: McpToolSpec,
        executable_snapshot: ExecutableSnapshot | None,
    ) -> tuple[Any, bool]:
        # Manifest v2 combines negotiation, live catalog validation and the
        # eventual tools/call in one provider entry point. Only the exact
        # built-in SDK provider has wire evidence strong enough to narrow a
        # normal pre-call result; custom providers retain the mutation floor.
        wire_bound = (
            server.schema_version == 2
            and type(self.provider) is SdkMcpProvider
        )
        phase = ProviderPhase(
            "provider_validate_and_call",
            state_mutation=tool.state_mutation and not wire_bound,
            information_flow=True,
        )
        try:
            return protected.call(phase, invoke_validated_tool), wire_bound
        finally:
            if executable_snapshot is not None:
                executable_snapshot.close()

    @staticmethod
    def _input_required_classification_override(
        tool: McpToolSpec,
        provider_result: McpProviderCallResult,
    ) -> ExternalEffectClassification | None:
        if provider_result.error_type != "mcp_input_required_unsupported":
            return None
        rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
        if (
            not tool.state_mutation
            and rollback_class is ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        ):
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={
                    "outcome": "failed",
                    "failure_kind": "mcp_input_required_unsupported",
                    "phase": "tools/call",
                    "automatic_retry_disabled": True,
                    "continuation_present": True,
                },
            )
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
            state_mutation=tool.state_mutation,
            information_flow=True,
            metadata={
                "outcome": "unknown_mcp_input_required_unsupported",
                "phase": "tools/call",
                "automatic_retry_disabled": True,
                "continuation_present": True,
            },
        )

    @staticmethod
    def _safe_transport_error_message(
        server: McpServerSpec,
        error_type: str | None,
    ) -> str | None:
        if error_type == "McpStdioFrameTooLarge":
            return (
                "MCP stdio frame exceeded "
                f"max_response_bytes={server.max_response_bytes}"
            )
        if error_type == "McpStdioStdoutTooLarge":
            max_output_bytes = server.max_response_bytes
            if server.schema_version == 1:
                max_output_bytes *= _MCP_STDIO_PROTOCOL_OUTPUT_MULTIPLIER
            return (
                "MCP stdio stdout exceeded max_output_bytes="
                f"{max_output_bytes}"
            )
        if error_type == "McpStdioStderrTooLarge":
            return (
                "MCP stdio stderr exceeded "
                f"max_output_bytes={server.max_response_bytes}"
            )
        if error_type == "McpHttpSseFrameTooLarge":
            return (
                "MCP HTTP SSE frame exceeded "
                f"max_response_bytes={server.max_response_bytes}"
            )
        if error_type == "McpHttpResponseTooLarge":
            return (
                "MCP HTTP response exceeded "
                f"max_response_bytes={server.max_response_bytes}"
            )
        if error_type == "McpHttpOperationTooLarge":
            return (
                "MCP HTTP operation exceeded "
                f"max_response_bytes={server.max_response_bytes}"
            )
        if error_type == "McpHttpContentEncodingDenied":
            return "MCP HTTP response uses unsupported Content-Encoding"
        return None

    def _failure(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        status: McpCallStatus,
        message: str,
        provider_result: McpProviderCallResult,
        *,
        extra: dict[str, Any] | None = None,
    ) -> McpCallResult:
        code = {
            McpCallStatus.MCP_ERROR: "mcp_tool_error",
            McpCallStatus.INVALID_RESPONSE: "mcp_invalid_response",
            McpCallStatus.RESPONSE_TOO_LARGE: "mcp_response_too_large",
            McpCallStatus.TRANSPORT_ERROR: "mcp_provider_error",
        }.get(status, "mcp_call_failed")
        dispatch_state = self._public_call_dispatch_state(server, provider_result)
        return McpCallResult(
            server_id=server.server_id,
            tool_id=tool.tool_id,
            mcp_name=tool.mcp_name,
            status=status,
            ok=False,
            error={
                "code": code,
                "error_type": provider_result.error_type or status.value,
                "correlation_id": provider_result.correlation_id or new_id("corr"),
                "message": message,
                "retryable": False,
                "automatic_retry_disabled": True,
                **dict(extra or {}),
            },
            response_bytes=provider_result.response_bytes,
            duration_s=provider_result.duration_s,
            connection=provider_result.connection,
            receipts=provider_result.receipts,
            dispatch_state=dispatch_state,
            retry_class=self._public_call_retry_class(
                ok=False,
                dispatch_state=dispatch_state,
            ),
        )

    def _protected(self) -> Any:
        return self.protected_operations

    @staticmethod
    def _effective_protocol_mode(server: McpServerSpec) -> McpProtocolMode:
        return server.protocol_mode or McpProtocolMode.LEGACY

    def _require_modern_protocol_provider(self, server: McpServerSpec) -> None:
        if server.schema_version != 2:
            return
        if (
            not isinstance(self.provider, McpModernProtocolProvider)
            or self.provider.supports_mcp_modern_protocol is not True
        ):
            raise ValidationError(
                "MCP Manifest v2 requires a provider that explicitly supports "
                "modern protocol negotiation"
            )

    def _data_flow(self) -> Any:
        manager = getattr(self, "data_flow", None) or getattr(
            self._protected(),
            "data_flow",
            None,
        )
        if manager is None:
            raise ValidationError("MCP data-flow manager is not attached")
        return manager

    def _server_identity_sha256(
        self,
        spec: McpRegisteredServer,
        tool: McpToolSpec,
        *,
        stdio_executable: dict[str, str] | None | object = (
            _STDIO_EXECUTABLE_IDENTITY_UNSET
        ),
        deadline: float | None = None,
    ) -> str | None:
        if deadline is not None:
            self._remaining_timeout(deadline)
        if stdio_executable is _STDIO_EXECUTABLE_IDENTITY_UNSET:
            stdio_executable = self._stdio_executable_identity(
                spec,
                deadline=deadline,
            )
        if spec.transport == "stdio" and stdio_executable is None:
            # A stdio provider that cannot resolve the exact executable may
            # still handle normal data, but it cannot match Host clearance for
            # data above normal sensitivity.
            return None
        digest = hashlib.sha256(
            dumps(
                to_jsonable(
                    {
                        "schema_version": spec.schema_version,
                        "server": self._registered_server_to_jsonable(spec),
                        "tool": tool,
                        "stdio_executable": stdio_executable,
                    }
                )
            ).encode("utf-8")
        ).hexdigest()
        if deadline is not None:
            self._remaining_timeout(deadline)
        return digest

    def _list_tools_identity_sha256(
        self,
        spec: McpRegisteredServer,
        *,
        stdio_executable: dict[str, str] | None | object = (
            _STDIO_EXECUTABLE_IDENTITY_UNSET
        ),
        deadline: float | None = None,
    ) -> str | None:
        if deadline is not None:
            self._remaining_timeout(deadline)
        if stdio_executable is _STDIO_EXECUTABLE_IDENTITY_UNSET:
            stdio_executable = self._stdio_executable_identity(
                spec,
                deadline=deadline,
            )
        if spec.transport == "stdio" and stdio_executable is None:
            return None
        digest = hashlib.sha256(
            dumps(
                to_jsonable(
                    {
                        "schema_version": spec.schema_version,
                        "server": self._registered_server_to_jsonable(spec),
                        "operation": "tools/list",
                        "stdio_executable": stdio_executable,
                    }
                )
            ).encode("utf-8")
        ).hexdigest()
        if deadline is not None:
            self._remaining_timeout(deadline)
        return digest

    def _discover_identity_sha256(
        self,
        spec: McpServerSpec,
        *,
        stdio_executable: dict[str, str] | None | object = (
            _STDIO_EXECUTABLE_IDENTITY_UNSET
        ),
        deadline: float | None = None,
    ) -> str | None:
        if deadline is not None:
            self._remaining_timeout(deadline)
        if stdio_executable is _STDIO_EXECUTABLE_IDENTITY_UNSET:
            stdio_executable = self._stdio_executable_identity(
                spec,
                deadline=deadline,
            )
        if spec.transport == "stdio" and stdio_executable is None:
            return None
        digest = hashlib.sha256(
            dumps(
                to_jsonable(
                    {
                        "schema_version": 2,
                        "server": self._registered_server_to_jsonable(spec),
                        "operation": "server/discover",
                        "stdio_executable": stdio_executable,
                    }
                )
            ).encode("utf-8")
        ).hexdigest()
        if deadline is not None:
            self._remaining_timeout(deadline)
        return digest

    def _stdio_executable_identity(
        self,
        spec: McpServerSpec,
        *,
        runtime_environment: Mapping[str, str] | None = None,
        deadline: float | None = None,
        fail_closed: bool = False,
    ) -> dict[str, str] | None:
        if spec.transport != "stdio" or spec.stdio is None:
            return None
        resolver = getattr(self.provider, "resolve_stdio_executable", None)
        if not callable(resolver):
            return None
        try:
            if deadline is not None:
                self._remaining_timeout(deadline)
            resolver_kwargs = (
                {"runtime_environment": runtime_environment}
                if runtime_environment is not None
                and bool(
                    getattr(
                        self.provider,
                        "supports_runtime_environment_snapshots",
                        False,
                    )
                )
                else {}
            )
            resolved = Path(resolver(spec, **resolver_kwargs)).resolve(strict=True)
            if deadline is not None:
                self._remaining_timeout(deadline)
            return {
                "path": resolved.as_posix(),
                "content_sha256": executable_content_sha256(
                    resolved,
                    deadline=deadline,
                ),
            }
        except TimeoutError as exc:
            if deadline is not None and time.monotonic() >= deadline:
                raise ProviderEffectNotStarted(
                    "MCP deadline exhausted during executable identity"
                ) from exc
            raise
        except (OSError, ValidationError) as exc:
            if fail_closed:
                raise ValidationError(
                    "MCP stdio executable identity is unavailable"
                ) from exc
            # Preserve normal-sensitivity compatibility, but leave the Sink
            # unidentified so any Host rule above normal fails closed.
            return None

    def _tool_data_sink_for_clearance_precheck(
        self,
        server_id: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
    ) -> DataSink:
        identity = f"mcp:{server_id}:{tool.tool_id}"
        if spec.transport != "stdio":
            return self._tool_data_sink_from_stdio_identity(
                server_id,
                spec,
                tool,
                None,
            )

        # Stdio executable identity may depend on a Host-provided PATH. Use
        # the Host trust record's expected identity only for this negative
        # precheck; exact authorization below uses the resolved executable.
        trust = self._data_flow().resolve_sink_trust(DataSink(identity))
        return DataSink(
            identity,
            trust.identity_sha256 if trust is not None else None,
        )

    def _stdio_executable_resolution_environment(
        self,
        spec: McpServerSpec,
        *,
        deadline: float | None = None,
    ) -> Mapping[str, str]:
        if deadline is not None:
            self._remaining_timeout(deadline)
        if spec.transport != "stdio" or spec.stdio is None:
            return MappingProxyType({})
        command = spec.stdio.command
        if Path(command).is_absolute() or "/" in command or "\\" in command:
            return MappingProxyType({})
        child_names = ("PATH", "PATHEXT") if _MCP_WINDOWS else ("PATH",)
        selected: dict[str, str] = {}
        for child_name in child_names:
            if deadline is not None:
                self._remaining_timeout(deadline)
            host_name = spec.stdio.env.get(child_name)
            if host_name is None:
                if _MCP_WINDOWS:
                    raise ValidationError(
                        "Windows MCP stdio bare commands require "
                        "manifest-mapped child PATH and PATHEXT"
                    )
                continue
            resolved = os.environ.get(host_name)
            if resolved is None:
                raise ValidationError(
                    f"missing environment variable for MCP stdio env {child_name}: "
                    f"{host_name}"
                )
            if "\x00" in resolved:
                raise ValidationError(
                    f"MCP stdio env {child_name} contains NUL byte"
                )
            selected[child_name] = resolved
        if deadline is not None:
            self._remaining_timeout(deadline)
        return MappingProxyType(selected)

    def _tool_data_sink_from_stdio_identity(
        self,
        server_id: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
        stdio_identity: dict[str, str] | None,
        *,
        deadline: float | None = None,
    ) -> DataSink:
        return DataSink(
            f"mcp:{server_id}:{tool.tool_id}",
            self._server_identity_sha256(
                spec,
                tool,
                stdio_executable=stdio_identity,
                deadline=deadline,
            ),
        )

    def _tool_data_sink(
        self,
        server_id: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
        runtime_environment: Mapping[str, str],
        *,
        deadline: float | None = None,
    ) -> DataSink:
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=runtime_environment,
            deadline=deadline,
        )
        return self._tool_data_sink_from_stdio_identity(
            server_id,
            spec,
            tool,
            stdio_identity,
            deadline=deadline,
        )

    def _tool_data_sink_after_runtime_resolution(
        self,
        server_id: str,
        spec: McpServerSpec,
        tool: McpToolSpec,
        runtime_environment: Mapping[str, str] | None,
        *,
        expected: DataSink,
        deadline: float | None = None,
    ) -> DataSink:
        if deadline is not None:
            self._remaining_timeout(deadline)
        if runtime_environment is None:
            return expected
        return self._tool_data_sink(
            server_id,
            spec,
            tool,
            runtime_environment,
            deadline=deadline,
        )

    def _list_tools_data_sink(
        self,
        server_id: str,
        spec: McpServerSpec,
        runtime_environment: Mapping[str, str],
        *,
        deadline: float | None = None,
    ) -> DataSink:
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=runtime_environment,
            deadline=deadline,
        )
        return DataSink(
            f"mcp:{server_id}:list_tools",
            self._list_tools_identity_sha256(
                spec,
                stdio_executable=stdio_identity,
                deadline=deadline,
            ),
        )

    def _discover_data_sink(
        self,
        server_id: str,
        spec: McpServerSpec,
        runtime_environment: Mapping[str, str] | None,
        *,
        deadline: float | None = None,
    ) -> DataSink:
        stdio_identity = self._stdio_executable_identity(
            spec,
            runtime_environment=runtime_environment,
            deadline=deadline,
        )
        return DataSink(
            f"mcp:{server_id}:discover",
            self._discover_identity_sha256(
                spec,
                stdio_executable=stdio_identity,
                deadline=deadline,
            ),
        )

    def _stdio_snapshot_for_dispatch(
        self,
        *,
        pid: str,
        spec: McpServerSpec,
        expected_identity: dict[str, str] | None,
        sink: DataSink,
        context: DataFlowContext,
        payload: Any,
        runtime_environment: Mapping[str, str] | None = None,
        deadline: float,
    ) -> ExecutableSnapshot | None:
        if spec.transport != "stdio" or spec.stdio is None:
            return None
        resolver = getattr(self.provider, "resolve_stdio_executable", None)
        checker = getattr(self.provider, "executable_snapshot_required", None)
        if not callable(resolver) or not callable(checker):
            return None
        resolver_kwargs = (
            {"runtime_environment": runtime_environment}
            if runtime_environment is not None
            and bool(
                getattr(
                    self.provider,
                    "supports_runtime_environment_snapshots",
                    False,
                )
            )
            else {}
        )
        self._remaining_timeout(deadline)
        resolved = Path(resolver(spec, **resolver_kwargs)).resolve(strict=True)
        self._remaining_timeout(deadline)
        snapshot_required = bool(
            checker(spec, str(resolved), **resolver_kwargs)
        )
        self._remaining_timeout(deadline)
        if not snapshot_required:
            if expected_identity is None:
                return None
            try:
                content_sha256 = executable_content_sha256(
                    resolved,
                    deadline=deadline,
                )
            except TimeoutError as error:
                raise ProviderEffectNotStarted(
                    "MCP deadline exhausted during final executable fingerprint"
                ) from error
            actual = {
                "path": resolved.as_posix(),
                "content_sha256": content_sha256,
            }
            if actual != expected_identity:
                self._data_flow().reject_sink_identity_change(
                    pid=pid,
                    sink=sink,
                    context=context,
                    payload=payload,
                    reason="MCP stdio Sink identity changed before dispatch",
                )
                raise AssertionError("data-flow Sink rejection must raise")
            return None
        if not bool(getattr(self.provider, "supports_executable_snapshots", False)):
            self._data_flow().reject_sink_identity_change(
                pid=pid,
                sink=sink,
                context=context,
                payload=payload,
                reason="MCP provider cannot pin a mutable stdio executable",
            )
            raise AssertionError("data-flow Sink rejection must raise")
        try:
            snapshot = snapshot_executable(
                resolved,
                sibling_limit=self.config.tools.executable_snapshot_sibling_limit,
                sibling_policy="scripts",
                deadline=deadline,
            )
        except TimeoutError as error:
            raise ProviderEffectNotStarted(
                "MCP deadline exhausted during final executable snapshot"
            ) from error
        actual = {
            "path": snapshot.source_path.as_posix(),
            "content_sha256": snapshot.content_sha256,
        }
        if expected_identity is None or actual != expected_identity:
            snapshot.close()
            self._data_flow().reject_sink_identity_change(
                pid=pid,
                sink=sink,
                context=context,
                payload=payload,
                reason=(
                    "MCP stdio Sink identity changed before immutable "
                    "executable dispatch snapshot"
                ),
            )
            raise AssertionError("data-flow Sink rejection must raise")
        try:
            self._remaining_timeout(deadline)
        except BaseException:
            snapshot.close()
            raise
        return snapshot

    def _protected_list_evidence(
        self,
        pid: str,
        spec: McpServerSpec,
        context: dict[str, Any],
        result_payload: dict[str, Any],
    ) -> ProtectedOperationEvidence:
        event_payload = {
            "adapter": "mcp",
            "operation": "list_tools",
            "server_id": spec.server_id,
            "transport": spec.transport,
            **result_payload,
        }
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_READ,
            event_source=pid,
            event_target=self.server_resource(spec.server_id),
            event_payload=event_payload,
            audit_action="primitive.mcp.list_tools",
            audit_actor=pid,
            audit_target=self.server_resource(spec.server_id),
            audit_decision={
                "server_id": spec.server_id,
                "transport": spec.transport,
                "request_bytes": context["request_bytes"],
                **result_payload,
            },
            effect_metadata=result_payload,
            provider_receipt={
                "response_bytes": int(result_payload.get("response_bytes", 0) or 0),
                "duration_s": float(result_payload.get("duration_s", 0.0) or 0.0),
            },
        )

    def _protected_discover_evidence(
        self,
        pid: str,
        spec: McpServerSpec,
        context: dict[str, Any],
        result_payload: dict[str, Any],
    ) -> ProtectedOperationEvidence:
        event_payload = {
            "adapter": "mcp",
            "operation": "discover",
            "server_id": spec.server_id,
            "transport": spec.transport,
            **result_payload,
        }
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_READ,
            event_source=pid,
            event_target=self.server_resource(spec.server_id),
            event_payload=event_payload,
            audit_action="primitive.mcp.discover",
            audit_actor=pid,
            audit_target=self.server_resource(spec.server_id),
            audit_decision={
                "server_id": spec.server_id,
                "transport": spec.transport,
                "protocol_mode": self._effective_protocol_mode(spec).value,
                "request_bytes": context["request_bytes"],
                **result_payload,
            },
            effect_metadata=result_payload,
            provider_receipt={
                "request_bytes": int(result_payload.get("request_bytes", 0) or 0),
                "response_bytes": int(result_payload.get("response_bytes", 0) or 0),
                "duration_s": float(result_payload.get("duration_s", 0.0) or 0.0),
                "protocol_revision": (
                    result_payload.get("connection", {}).get("protocol_revision")
                    if isinstance(result_payload.get("connection"), dict)
                    else None
                ),
            },
        )

    def _protected_discover_failure_evidence(
        self,
        pid: str,
        spec: McpServerSpec,
        context: dict[str, Any],
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        return self._protected_discover_evidence(
            pid,
            spec,
            context,
            {
                "ok": False,
                "status": "transport_error",
                "request_bytes": 0,
                "response_bytes": 0,
                "duration_s": 0.0,
                "error_type": type(error).__name__,
                "phase": phase,
            },
        )

    def _protected_list_failure_evidence(
        self,
        pid: str,
        spec: McpServerSpec,
        context: dict[str, Any],
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        return self._protected_list_evidence(
            pid,
            spec,
            context,
            {
                "ok": False,
                "status": "transport_error",
                "response_bytes": 0,
                "duration_s": 0.0,
                "tool_count": 0,
                "error_type": type(error).__name__,
                "phase": phase,
            },
        )

    def _protected_call_evidence(
        self,
        pid: str,
        resource: str,
        result: McpCallResult,
        tool: McpToolSpec,
        operation_context: dict[str, Any],
    ) -> ProtectedOperationEvidence:
        result_payload = self._call_effect_result(result)
        receipt_request_bytes = sum(
            receipt.request_bytes for receipt in result.receipts
        )
        receipt_response_bytes = sum(
            receipt.response_bytes for receipt in result.receipts
        )
        connection_payload = (
            to_jsonable(result.connection)
            if result.connection is not None
            else None
        )
        return ProtectedOperationEvidence(
            event_type=(
                EventType.EXTERNAL_WRITE
                if tool.state_mutation or tool.right != CapabilityRight.READ.value
                else EventType.EXTERNAL_READ
            ),
            event_source=pid,
            event_target=resource,
            event_payload={
                "adapter": "mcp",
                "server_id": result.server_id,
                "tool_id": result.tool_id,
                "mcp_name": result.mcp_name,
                **result_payload,
            },
            audit_action="primitive.mcp.call",
            audit_actor=pid,
            audit_target=resource,
            audit_decision={
                "server_id": result.server_id,
                "tool_id": result.tool_id,
                "mcp_name": tool.mcp_name,
                "right": tool.right,
                "arguments_sha256": operation_context["arguments_sha256"],
                "arguments_preview": operation_context["arguments_preview"],
                "arguments_observation": operation_context["arguments_observation"],
                "sandbox_profile": operation_context.get("sandbox_profile"),
                **result_payload,
            },
            capability_refs=tuple(operation_context.get("capability_ids") or ()),
            effect_metadata=result_payload,
            provider_receipt={
                "request_bytes": receipt_request_bytes,
                "response_bytes": result.response_bytes,
                "operation_response_bytes": receipt_response_bytes,
                "duration_s": result.duration_s,
                **(
                    {"receipts": to_jsonable(result.receipts)}
                    if result.receipts
                    else {}
                ),
                **(
                    {
                        "protocol_revision": result.connection.protocol_revision,
                        "protocol_era": result.connection.protocol_era.value,
                        "fallback_used": result.connection.fallback_used,
                        "connection": connection_payload,
                    }
                    if result.connection is not None
                    else {}
                ),
            },
        )

    def _protected_call_failure_evidence(
        self,
        pid: str,
        resource: str,
        tool: McpToolSpec,
        operation_context: dict[str, Any],
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        result = McpCallResult(
            server_id=str(operation_context["server_id"]),
            tool_id=tool.tool_id,
            mcp_name=tool.mcp_name,
            status=McpCallStatus.TRANSPORT_ERROR,
            ok=False,
            error={"message": type(error).__name__, "phase": phase},
        )
        return self._protected_call_evidence(pid, resource, result, tool, operation_context)

    @staticmethod
    def _call_effect_result(result: McpCallResult) -> dict[str, Any]:
        payload = {
            "status": result.status.value,
            "ok": result.ok,
            "response_bytes": result.response_bytes,
            "duration_s": result.duration_s,
            "dispatch_state": result.dispatch_state.value,
            "retry_class": result.retry_class.value,
            "automatic_retry_disabled": result.automatic_retry_disabled,
        }
        if result.receipts:
            payload["receipts"] = to_jsonable(result.receipts)
        if result.connection is not None:
            payload.update(
                {
                    "protocol_revision": result.connection.protocol_revision,
                    "connection": to_jsonable(result.connection),
                }
            )
        public_error = provider_error_envelope_from_mapping(result.error or {})
        if public_error is not None:
            payload["error"] = {
                key: public_error[key]
                for key in ("code", "error_type", "correlation_id")
            }
        return payload

    def _live_validation_failure_result(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        error: Exception,
        *,
        duration_s: float,
    ) -> McpCallResult:
        if isinstance(error, ProviderHostError):
            public_error = {
                **error.to_dict(),
                "retryable": False,
                "automatic_retry_disabled": True,
            }
            return McpCallResult(
                server_id=server.server_id,
                tool_id=tool.tool_id,
                mcp_name=tool.mcp_name,
                status=McpCallStatus.INVALID_RESPONSE,
                ok=False,
                error=public_error,
                duration_s=duration_s,
                dispatch_state=McpDispatchState.NOT_STARTED,
                retry_class=McpRetryClass.REOBSERVE_REQUIRED,
            )
        return self._failure(
            server,
            tool,
            McpCallStatus.INVALID_RESPONSE,
            f"MCP live tool metadata validation failed: {error}",
            McpProviderCallResult(
                error=type(error).__name__,
                duration_s=duration_s,
            ),
        )

    @staticmethod
    def _mcp_call_failure_resource(
        error: BaseException,
        phase: str,
        *,
        spec: McpServerSpec,
        request_bytes: int,
        list_request_bytes: int,
        resource_context: dict[str, Any],
        resource_progress: dict[str, int],
    ) -> ResourceSettlement | None:
        list_response_bytes = resource_progress["list_response_bytes"]
        provider_result_failure = _provider_result_was_returned(error)
        if (
            provider_result_failure
            and phase == "provider_validate_and_call"
            and spec.schema_version == 2
        ):
            # A combined v2 provider owns negotiation, live validation and
            # dispatch.  Once it has returned a malformed result, no trusted
            # per-phase receipt remains; settle the same cumulative envelope
            # reserved for that phase instead of the legacy two-response shape.
            return ResourceSettlement(
                usage=ResourceUsage(
                    mcp_request_bytes=spec.max_request_bytes,
                    mcp_response_bytes=spec.max_response_bytes,
                ),
                source="primitive.mcp.call",
                context={
                    **resource_context,
                    "failure_phase": phase,
                    "unknown_request_bytes": spec.max_request_bytes,
                    "unknown_response_bytes": spec.max_response_bytes,
                    "call_started": True,
                    "provider_result_returned": True,
                },
            )
        if phase == "provider_call" and provider_result_failure:
            return ResourceSettlement(
                usage=ResourceUsage(
                    mcp_request_bytes=list_request_bytes + request_bytes,
                    mcp_response_bytes=(
                        list_response_bytes + spec.max_response_bytes
                    ),
                ),
                source="primitive.mcp.call",
                context={
                    **resource_context,
                    "failure_phase": phase,
                    "list_response_bytes": list_response_bytes,
                    "unknown_response_bytes": spec.max_response_bytes,
                    "call_started": True,
                    "provider_result_returned": True,
                },
            )
        if not isinstance(error, ProviderEffectNotStarted):
            return None
        return ResourceSettlement(
            usage=ResourceUsage(
                mcp_request_bytes=list_request_bytes,
                mcp_response_bytes=list_response_bytes,
            ),
            source="primitive.mcp.call",
            context={
                **resource_context,
                "failure_phase": phase,
                "call_started": False,
                "certified_not_started": True,
            },
        )

    @staticmethod
    def _mcp_call_settlement(
        request_bytes: int,
        result: McpCallResult,
        *,
        server_id: str,
        tool_id: str,
        list_request_bytes: int = 0,
        list_response_bytes: int = 0,
        call_started: bool = True,
        unknown_response_bytes: int = 0,
    ) -> ResourceSettlement:
        call_request_bytes = request_bytes if call_started else 0
        return ResourceSettlement(
            usage=ResourceUsage(
                mcp_request_bytes=list_request_bytes + call_request_bytes,
                mcp_response_bytes=(
                    list_response_bytes
                    + result.response_bytes
                    + unknown_response_bytes
                ),
            ),
            source="primitive.mcp.call",
            context={
                "server_id": server_id,
                "tool_id": tool_id,
                "request_bytes": request_bytes,
                "response_bytes": result.response_bytes,
                "list_request_bytes": list_request_bytes,
                "list_response_bytes": list_response_bytes,
                "unknown_response_bytes": unknown_response_bytes,
                "call_started": call_started,
                "status": result.status.value,
            },
        )

    @classmethod
    def _mcp_live_validation_failure_settlement(
        cls,
        server: McpServerSpec,
        request_bytes: int,
        result: McpCallResult,
        *,
        server_id: str,
        tool_id: str,
        list_request_bytes: int,
        live_list_result: McpToolListResult | None,
    ) -> ResourceSettlement:
        list_response_bytes = (
            live_list_result.response_bytes if live_list_result is not None else 0
        )
        return cls._mcp_call_settlement(
            request_bytes,
            result,
            server_id=server_id,
            tool_id=tool_id,
            list_request_bytes=list_request_bytes,
            list_response_bytes=list_response_bytes,
            call_started=False,
            unknown_response_bytes=(
                server.max_response_bytes if live_list_result is None else 0
            ),
        )

    @classmethod
    def _mcp_legacy_call_completion_settlement(
        cls,
        server: McpServerSpec,
        request_bytes: int,
        result: McpCallResult,
        provider_result: McpProviderCallResult,
        classification_override: ExternalEffectClassification | None,
        *,
        server_id: str,
        tool_id: str,
        list_request_bytes: int,
        live_list_result: McpToolListResult | None,
    ) -> ResourceSettlement:
        if provider_result.receipts:
            return cls._mcp_exchange_settlement(
                provider_result,
                fallback_list_request_bytes=list_request_bytes,
                fallback_call_request_bytes=request_bytes,
                server_id=server_id,
                tool_id=tool_id,
                status=result.status.value,
            )
        list_response_bytes = (
            live_list_result.response_bytes if live_list_result is not None else 0
        )
        provider_outcome_unknown = (
            classification_override is not None and provider_result.error is not None
        )
        return cls._mcp_call_settlement(
            request_bytes,
            result,
            server_id=server_id,
            tool_id=tool_id,
            list_request_bytes=list_request_bytes,
            list_response_bytes=list_response_bytes,
            call_started=True,
            unknown_response_bytes=(
                server.max_response_bytes if provider_outcome_unknown else 0
            ),
        )

    @staticmethod
    def _mcp_exchange_settlement(
        result: McpProviderCallResult,
        *,
        fallback_list_request_bytes: int,
        fallback_call_request_bytes: int,
        server_id: str,
        tool_id: str,
        status: str,
    ) -> ResourceSettlement:
        if result.receipts:
            request_bytes = sum(item.request_bytes for item in result.receipts)
            response_bytes = sum(item.response_bytes for item in result.receipts)
            return ResourceSettlement(
                usage=ResourceUsage(
                    mcp_request_bytes=request_bytes,
                    mcp_response_bytes=response_bytes,
                ),
                source="primitive.mcp.call",
                context={
                    "server_id": server_id,
                    "tool_id": tool_id,
                    "request_bytes": request_bytes,
                    "response_bytes": response_bytes,
                    "call_started": result.call_started,
                    "status": status,
                    "receipts": to_jsonable(result.receipts),
                },
            )
        list_request_bytes = result.list_request_bytes or fallback_list_request_bytes
        call_request_bytes = (
            result.call_request_bytes or fallback_call_request_bytes
            if result.call_started
            else 0
        )
        call_response_bytes = result.call_response_bytes or result.response_bytes
        return ResourceSettlement(
            usage=ResourceUsage(
                mcp_request_bytes=list_request_bytes + call_request_bytes,
                mcp_response_bytes=result.list_response_bytes + call_response_bytes,
            ),
            source="primitive.mcp.call",
            context={
                "server_id": server_id,
                "tool_id": tool_id,
                "list_request_bytes": list_request_bytes,
                "list_response_bytes": result.list_response_bytes,
                "call_request_bytes": call_request_bytes,
                "call_response_bytes": call_response_bytes,
                "call_started": result.call_started,
                "status": status,
            },
        )

    def _complete_expired_legacy_exchange(
        self,
        *,
        protected: Any,
        spec: McpServerSpec,
        tool: McpToolSpec,
        operation_context: dict[str, Any],
        effect_context: dict[str, Any],
        live_list_result: McpToolListResult | None,
        deadline: float,
        started: float,
        list_request_bytes: int,
        request_bytes: int,
    ) -> McpCallResult | None:
        if time.monotonic() < deadline:
            return None
        provider_result = McpProviderCallResult(
            error="MCP exchange deadline exhausted before tool dispatch",
            error_type="McpDeadlineExceeded",
            correlation_id=new_id("corr"),
            duration_s=time.monotonic() - started,
            list_request_bytes=list_request_bytes,
            list_response_bytes=(
                live_list_result.response_bytes
                if live_list_result is not None
                else 0
            ),
            call_request_bytes=request_bytes,
            call_started=False,
        )
        result = self._call_result_from_provider(spec, tool, provider_result)
        pid = str(operation_context["pid"])
        resource = self.tool_resource(spec.server_id, tool.tool_id)
        return protected.complete(
            result,
            self._protected_call_evidence(
                pid,
                resource,
                result,
                tool,
                operation_context,
            ),
            classification_context=effect_context,
            classification_result=self._call_effect_result(result),
            classification_override=ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={
                    "outcome": "deadline_exhausted_before_call",
                    "phase": "live_validation",
                },
            ),
            resource=self._mcp_exchange_settlement(
                provider_result,
                fallback_list_request_bytes=list_request_bytes,
                fallback_call_request_bytes=request_bytes,
                server_id=spec.server_id,
                tool_id=tool.tool_id,
                status=result.status.value,
            ),
        )

    def _list_tools_success_payload(self, result: McpToolListResult) -> dict[str, Any]:
        payload = {
            "ok": True,
            "status": "ok",
            "response_bytes": result.response_bytes,
            "duration_s": result.duration_s,
            "tool_count": len(result.tools),
        }
        if result.connection is not None:
            payload["connection"] = to_jsonable(result.connection)
            payload["receipts"] = to_jsonable(result.receipts)
        return payload

    def _list_tools_failure_payload(self, exc: Exception, *, duration_s: float) -> dict[str, Any]:
        safe_error = (
            exc.to_dict()
            if isinstance(exc, ProviderHostError)
            else {
                "code": "mcp_provider_error",
                "error_type": type(exc).__name__,
                "correlation_id": new_id("corr"),
            }
        )
        return {
            "ok": False,
            "status": "transport_error",
            "response_bytes": 0,
            "duration_s": duration_s,
            "tool_count": 0,
            **safe_error,
            "error_observation": sanitize_for_observability(safe_error),
        }

    def _resource_usage_pid(self, actor: str | None) -> str | None:
        if actor is None:
            return None
        process = self.processes.get_process(actor)
        if process is None:
            return None
        if process.status in {
            ProcessStatus.EXITED,
            ProcessStatus.FAILED,
            ProcessStatus.KILLED,
        }:
            # A persisted process id must never fall through to the Host actor
            # path after reopen.  Runtime-local MCP owner latches close races
            # in the live instance; this durable status check closes the same
            # authority boundary across Runtime incarnations.
            raise CapabilityDenied("terminal process cannot use MCP")
        return actor

    def _list_tools_effect_context(self, server: McpServerSpec, *, request_bytes: int) -> dict[str, Any]:
        return {
            "server_id": server.server_id,
            "transport": server.transport,
            "rollback_class": ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED.value,
            "rollback_status": ExternalEffectRollbackStatus.NOT_REQUIRED.value,
            "state_mutation": False,
            "information_flow": True,
            "request_bytes": request_bytes,
        }

    def _discover_effect_context(
        self,
        server: McpServerSpec,
        *,
        request_bytes: int,
    ) -> dict[str, Any]:
        return {
            "server_id": server.server_id,
            "transport": server.transport,
            "protocol_mode": self._effective_protocol_mode(server).value,
            "rollback_class": ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED.value,
            "rollback_status": ExternalEffectRollbackStatus.NOT_REQUIRED.value,
            "state_mutation": False,
            "information_flow": True,
            "request_bytes": request_bytes,
        }

    def _operation_context(
        self,
        pid: str,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        registry_binding: dict[str, Any],
    ) -> dict[str, Any]:
        arguments_json = dumps(arguments)
        arguments_observation = sanitize_for_observability(
            {"arguments": arguments},
            preview_chars=self.config.mcp.audit_preview_chars,
        )
        return {
            "pid": pid,
            "primitive": "runtime.mcp.call",
            "operation": "mcp.call",
            "authority_operation": "mcp.call",
            "server_id": server.server_id,
            "transport": server.transport,
            **(
                {"protocol_mode": self._effective_protocol_mode(server).value}
                if server.schema_version == 2
                else {}
            ),
            "tool_id": tool.tool_id,
            "mcp_name": tool.mcp_name,
            "right": tool.right,
            **registry_binding,
            "arguments_sha256": hashlib.sha256(arguments_json.encode("utf-8")).hexdigest(),
            "arguments_preview": arguments_observation["preview"],
            "arguments_observation": arguments_observation,
        }

    def _visibility_operation_context(self, pid: str, server_id: str, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        arguments_json = dumps(arguments)
        return {
            "pid": pid,
            "primitive": "runtime.mcp.call",
            "operation": "mcp.call",
            "authority_operation": "mcp.call",
            "server_id": server_id,
            "tool_id": tool_id,
            "arguments_sha256": hashlib.sha256(arguments_json.encode("utf-8")).hexdigest(),
        }

    def _approval_constraints(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            AUTHORITY_RULES_KEY: [
                {
                    "rule_id": f"mcp.approval.{context['server_id']}.{context['tool_id']}",
                    "operation": "mcp.call",
                    "effect": CapabilityEffect.ALLOW.value,
                    "risk": "high",
                    "conditions": {
                        "server_id": context["server_id"],
                        "tool_id": context["tool_id"],
                        "registry_spec_sha256": context["registry_spec_sha256"],
                        "registry_generation": context["registry_generation"],
                        "arguments_sha256": context["arguments_sha256"],
                    },
                    "description": "one-shot human approval for exact MCP tool payload",
                }
            ]
        }

    @staticmethod
    def _registered_server_to_jsonable(
        server: McpRegisteredServer,
    ) -> dict[str, Any]:
        if isinstance(server, McpServerManifestV3):
            selected = json.loads(canonical_mcp_v3_manifest_json(server))
            if not isinstance(selected, dict):  # pragma: no cover - canonical invariant
                raise ValidationError("canonical MCP Manifest v3 must be an object")
            return selected
        return mcp_server_spec_to_jsonable(server)

    @staticmethod
    def _server_spec_sha256(server: McpRegisteredServer) -> str:
        canonical = (
            canonical_mcp_v3_manifest_json(server)
            if isinstance(server, McpServerManifestV3)
            else canonical_mcp_server_spec_json(server)
        )
        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    def _registry_binding_context(self, server_id: str) -> dict[str, Any]:
        binding = self.extensions.get_mcp_registry_binding(server_id)
        if not isinstance(binding, dict):
            raise ValidationError("MCP registry binding must be an object")
        generation = binding.get("registry_generation")
        digest = binding.get("registry_spec_sha256")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValidationError("MCP registry generation is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValidationError("MCP registry spec digest is invalid")
        return {
            "registry_spec_sha256": digest,
            "registry_generation": generation,
        }

    def _registry_binding_for_server_spec(
        self,
        server: McpRegisteredServer,
    ) -> dict[str, Any]:
        binding = self._registry_binding_context(server.server_id)
        if binding["registry_spec_sha256"] != self._server_spec_sha256(server):
            raise CapabilityDenied("MCP server registry changed before call authorization")
        return binding

    def _protected_registry_guard(
        self,
        binding: dict[str, Any],
        server_id: str,
    ) -> dict[str, Any]:
        return {
            "provider_registry_binding": ProviderRegistryBinding.from_context(binding),
            "provider_registry_binding_resolver": lambda: ProviderRegistryBinding.from_context(
                self._registry_binding_context(server_id)
            ),
            "provider_registry_phase_guard": lambda: self._registry_phase_lock,
        }

    def _effect_context(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        operation_context: dict[str, Any],
        *,
        request_bytes: int,
    ) -> dict[str, Any]:
        return {
            "server_id": server.server_id,
            "transport": server.transport,
            **(
                {"protocol_mode": self._effective_protocol_mode(server).value}
                if server.schema_version == 2
                else {}
            ),
            "tool_id": tool.tool_id,
            "mcp_name": tool.mcp_name,
            "right": tool.right,
            "rollback_class": tool.rollback_class,
            "rollback_status": tool.rollback_status,
            "state_mutation": tool.state_mutation,
            "information_flow": tool.information_flow,
            "arguments_sha256": operation_context["arguments_sha256"],
            "arguments_observation": operation_context["arguments_observation"],
            "request_bytes": request_bytes,
        }

    def _coerce_server(
        self,
        value: McpRegisteredServer | dict[str, Any],
    ) -> McpRegisteredServer:
        if isinstance(value, McpServerManifestV3):
            validate_mcp_v3_manifest(
                value,
                host_policy=self._v3_host_policy(),
                enforce_host_policy=True,
            )
            return value
        if isinstance(value, dict) and value.get("schema_version") == 3:
            manifest = parse_mcp_v3_manifest_mapping(value)
            validate_mcp_v3_manifest(
                manifest,
                host_policy=self._v3_host_policy(),
                enforce_host_policy=True,
            )
            return manifest
        typed_input = isinstance(value, McpServerSpec)
        if typed_input:
            # Normalize typed and mapping/YAML inputs identically.  Runtime
            # dataclass annotations do not coerce ``timeout_s=1`` to ``1.0``;
            # leaving the typed value untouched would make the durable raw
            # spec and its decoded live model hash differently.
            value = to_jsonable(value)
        if isinstance(value, dict):
            _reject_unknown_fields(value, _SERVER_FIELDS, context="MCP server")
            schema_version = self._coerce_positive_int(
                value.get("schema_version", 1),
                "schema_version",
            )
            if (
                not typed_input
                and schema_version == 1
                and "protocol_mode" in value
            ):
                raise ValidationError(
                    "MCP server schema_version 1 must omit protocol_mode"
                )
            raw_protocol_mode = value.get("protocol_mode")
            protocol_mode: McpProtocolMode | None = None
            if raw_protocol_mode is not None:
                if type(raw_protocol_mode) is not str:
                    raise ValidationError("MCP protocol_mode must be a string")
                try:
                    protocol_mode = McpProtocolMode(raw_protocol_mode)
                except ValueError as exc:
                    raise ValidationError(
                        "MCP protocol_mode must be legacy, auto, or 2026-07-28"
                    ) from exc
            transport = self._required_string(
                value,
                "transport",
                "MCP server",
            ).strip()
            server_id = self._required_string(
                value,
                "server_id",
                "MCP server",
            )
            spec = McpServerSpec(
                schema_version=schema_version,
                server_id=server_id,
                transport=transport,
                # Preserve every supplied transport block through canonical
                # coercion. _validate_server owns the strict tagged-union
                # check; dropping the inactive block here would silently turn
                # an invalid typed or mapping manifest into a valid one.
                stdio=(
                    self._stdio_spec(value.get("stdio"))
                    if value.get("stdio") is not None
                    else None
                ),
                http=(
                    self._http_spec(value.get("http"))
                    if value.get("http") is not None
                    else None
                ),
                tools=[
                    self._tool_spec(item)
                    for item in self._list_field(value, "tools", "MCP server")
                ],
                timeout_s=self._coerce_positive_float(value.get("timeout_s", self.config.mcp.timeout_s), "timeout_s"),
                max_request_bytes=self._coerce_positive_int(
                    value.get("max_request_bytes", self.config.mcp.max_request_bytes),
                    "max_request_bytes",
                ),
                max_response_bytes=self._coerce_positive_int(
                    value.get("max_response_bytes", self.config.mcp.max_response_bytes),
                    "max_response_bytes",
                ),
                metadata=self._mapping_field(value, "metadata", "MCP server"),
                protocol_mode=protocol_mode,
            )
        else:
            raise ValidationError("MCP server must be an object")
        self._validate_server(spec)
        return spec

    def _v3_host_policy(self) -> McpManifestV3HostPolicy:
        selected = self.config.mcp
        return McpManifestV3HostPolicy(
            server_id_max_chars=selected.server_id_max_chars,
            tool_id_max_chars=selected.tool_id_max_chars,
            mcp_name_max_chars=selected.mcp_name_max_chars,
            header_name_max_chars=selected.header_name_max_chars,
            timeout_hard_limit_s=selected.timeout_hard_limit_s,
            max_request_hard_limit_bytes=selected.max_request_hard_limit_bytes,
            max_response_hard_limit_bytes=selected.max_response_hard_limit_bytes,
            tool_catalog_limit=selected.tool_catalog_limit,
            resource_catalog_limit=selected.resource_catalog_limit,
            resource_template_limit=selected.resource_template_limit,
            prompt_catalog_limit=selected.prompt_catalog_limit,
            schema_max_depth=selected.schema_max_depth,
            schema_max_nodes=selected.schema_max_nodes,
            schema_max_ref_hops=selected.schema_max_ref_hops,
            schema_max_composition_expansions=(
                selected.schema_max_composition_expansions
            ),
            schema_regex_pattern_max_bytes=(
                selected.schema_regex_pattern_max_bytes
            ),
            schema_regex_max_evaluations=selected.schema_regex_max_evaluations,
            schema_regex_match_timeout_s=selected.schema_regex_match_timeout_s,
            header_env_allowlist=tuple(selected.header_env_allowlist),
            stdio_env_allowlist=tuple(selected.stdio_env_allowlist),
            oauth_enabled=selected.oauth_enabled,
            tasks_extension_enabled=selected.tasks_extension_enabled,
            tasks_extension_spec_sha256=selected.tasks_extension_spec_sha256,
        )

    def _stdio_spec(self, value: Any) -> McpStdioTransportSpec:
        if not isinstance(value, dict):
            raise ValidationError("MCP stdio transport requires stdio object")
        _reject_unknown_fields(value, _STDIO_FIELDS, context="MCP stdio")
        args = self._list_field(value, "args", "MCP stdio")
        if any(type(item) is not str for item in args):
            raise ValidationError("MCP stdio args must be a list of strings")
        environment = self._mapping_field(value, "env", "MCP stdio")
        if any(
            type(name) is not str or type(host_name) is not str
            for name, host_name in environment.items()
        ):
            raise ValidationError("MCP stdio env must map strings to strings")
        cwd = value.get("cwd")
        if cwd is not None and type(cwd) is not str:
            raise ValidationError("MCP stdio cwd must be a string or null")
        return McpStdioTransportSpec(
            command=self._required_string(value, "command", "MCP stdio"),
            args=list(args),
            env=dict(environment),
            cwd=cwd,
        )

    def _http_spec(self, value: Any) -> McpHttpTransportSpec:
        if not isinstance(value, dict):
            raise ValidationError("MCP streamable_http transport requires http object")
        _reject_unknown_fields(value, _HTTP_FIELDS, context="MCP HTTP")
        return McpHttpTransportSpec(
            url=self._required_string(value, "url", "MCP HTTP"),
            headers=self._header_specs(
                self._mapping_field(value, "headers", "MCP HTTP")
            ),
        )

    def _tool_spec(self, value: Any) -> McpToolSpec:
        if not isinstance(value, dict):
            raise ValidationError("MCP tools entries must be objects")
        _reject_unknown_fields(value, _TOOL_FIELDS, context="MCP tool")
        rollback_status = value.get("rollback_status")
        if rollback_status is not None and type(rollback_status) is not str:
            raise ValidationError("MCP tool rollback_status must be a string or null")
        return McpToolSpec(
            tool_id=self._required_string(value, "tool_id", "MCP tool"),
            mcp_name=self._required_string(value, "mcp_name", "MCP tool"),
            right=self._required_string(value, "right", "MCP tool"),
            rollback_class=self._required_string(
                value,
                "rollback_class",
                "MCP tool",
            ),
            rollback_status=rollback_status,
            state_mutation=self._coerce_bool(
                self._required(value, "state_mutation", "MCP tool"),
                "state_mutation",
            ),
            information_flow=self._coerce_bool(
                self._required(value, "information_flow", "MCP tool"),
                "information_flow",
            ),
            input_schema=self._mapping_field(value, "input_schema", "MCP tool"),
            metadata=self._mapping_field(value, "metadata", "MCP tool"),
        )

    def _validate_server(self, server: McpRegisteredServer) -> None:
        if isinstance(server, McpServerManifestV3):
            validate_mcp_v3_manifest(
                server,
                host_policy=self._v3_host_policy(),
                enforce_host_policy=True,
            )
            return
        self._validate_server_protocol(server)
        self._validate_identifier(server.server_id, "server_id", self.config.mcp.server_id_max_chars)
        self._validate_server_transport(server)
        self._validate_server_limits(server)
        self._validate_server_tools(server)
        self._validate_json_value(server.metadata, "metadata")
        if server.schema_version == 2:
            self._validate_no_reserved_mcp_meta(server.metadata, "metadata")

    @staticmethod
    def _validate_server_protocol(server: McpServerSpec) -> None:
        if server.schema_version not in {1, 2}:
            raise ValidationError("MCP server schema_version must be 1 or 2")
        if server.schema_version == 1 and server.protocol_mode is not None:
            raise ValidationError(
                "MCP server schema_version 1 must omit protocol_mode"
            )
        if server.schema_version == 2 and server.protocol_mode is None:
            raise ValidationError(
                "MCP server schema_version 2 requires protocol_mode"
            )

    def _validate_server_transport(self, server: McpServerSpec) -> None:
        if server.transport not in _TRANSPORTS:
            raise ValidationError("MCP transport must be stdio or streamable_http")
        if server.transport == "stdio":
            self._validate_stdio(server.stdio)
            if server.http is not None:
                raise ValidationError("MCP stdio server cannot include http configuration")
        if server.transport == "streamable_http":
            self._validate_http(
                server.http,
                schema_version=server.schema_version,
            )
            if server.stdio is not None:
                raise ValidationError("MCP streamable_http server cannot include stdio configuration")

    def _validate_server_limits(self, server: McpServerSpec) -> None:
        if not server.tools:
            raise ValidationError("MCP server must declare at least one allowed tool")
        if (
            server.schema_version == 2
            and len(server.tools) > self.config.mcp.list_limit
        ):
            raise ValidationError(
                "MCP schema_version 2 tool allowlist exceeds "
                f"list_limit={self.config.mcp.list_limit}"
            )
        if server.timeout_s > self.config.mcp.timeout_hard_limit_s:
            raise ValidationError("MCP timeout_s exceeds configured hard limit")
        if server.max_request_bytes > self.config.mcp.max_request_hard_limit_bytes:
            raise ValidationError("MCP max_request_bytes exceeds configured hard limit")
        if server.max_response_bytes > self.config.mcp.max_response_hard_limit_bytes:
            raise ValidationError("MCP max_response_bytes exceeds configured hard limit")

    def _validate_server_tools(self, server: McpServerSpec) -> None:
        seen_tool_ids: set[str] = set()
        seen_mcp_names: set[str] = set()
        for tool in server.tools:
            self._validate_tool(tool, schema_version=server.schema_version)
            if tool.tool_id in seen_tool_ids:
                raise ValidationError(f"duplicate MCP tool_id: {tool.tool_id}")
            if tool.mcp_name in seen_mcp_names:
                raise ValidationError(f"duplicate MCP mcp_name: {tool.mcp_name}")
            seen_tool_ids.add(tool.tool_id)
            seen_mcp_names.add(tool.mcp_name)

    def _validate_stdio(self, stdio: McpStdioTransportSpec | None) -> None:
        if stdio is None:
            raise ValidationError("MCP stdio transport requires stdio configuration")
        self._validate_stdio_command(stdio)
        self._validate_stdio_args(stdio)
        self._validate_stdio_environment(stdio)
        self._validate_stdio_cwd(stdio)

    @staticmethod
    def _validate_stdio_command(stdio: McpStdioTransportSpec) -> None:
        command = stdio.command.strip()
        if not command:
            raise ValidationError("MCP stdio command must be non-empty")
        if (
            command != stdio.command
            or any(char.isspace() for char in command)
            or any(char in command for char in "\r\n;&|<>")
        ):
            raise ValidationError("MCP stdio command must be a single argv token, not a shell string")
        if command.startswith("~"):
            raise ValidationError(
                "MCP stdio command must not use Host home-directory expansion"
            )
        if not _MCP_WINDOWS:
            return
        windows_path = PureWindowsPath(command)
        path_qualified = (
            windows_path.is_absolute() or "/" in command or "\\" in command
        )
        if (
            path_qualified
            and windows_path.suffix.casefold()
            not in _MCP_WINDOWS_EXECUTABLE_SUFFIXES
        ):
            raise ValidationError(
                "Windows MCP stdio executables must end in .exe or .com"
            )
        if not path_qualified and not {"PATH", "PATHEXT"}.issubset(stdio.env):
            raise ValidationError(
                "Windows MCP stdio bare commands require manifest-mapped "
                "child PATH and PATHEXT"
            )

    @staticmethod
    def _validate_stdio_args(stdio: McpStdioTransportSpec) -> None:
        for arg in stdio.args:
            if not isinstance(arg, str) or "\x00" in arg:
                raise ValidationError("MCP stdio args must be strings without NUL bytes")

    def _validate_stdio_environment(self, stdio: McpStdioTransportSpec) -> None:
        for child_name, host_name in stdio.env.items():
            self._validate_env_name(child_name, "stdio env name")
            self._validate_env_name(host_name, "stdio env source")
            if not self._env_allowed(host_name, self.config.mcp.stdio_env_allowlist):
                raise ValidationError(f"MCP stdio env source is not allowlisted: {host_name}")

    @staticmethod
    def _validate_stdio_cwd(stdio: McpStdioTransportSpec) -> None:
        if stdio.cwd is not None:
            raw = stdio.cwd.replace("\\", "/").strip()
            if not raw or PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
                raise ValidationError("MCP stdio cwd must be a non-empty relative path")
            parts: list[str] = []
            for part in raw.split("/"):
                if part in {"", "."}:
                    continue
                if part == "..":
                    if not parts:
                        raise ValidationError("MCP stdio cwd escapes workspace root")
                    parts.pop()
                    continue
                parts.append(part)

    def _validate_http(
        self,
        http: McpHttpTransportSpec | None,
        *,
        schema_version: int,
    ) -> None:
        if http is None:
            raise ValidationError("MCP streamable_http transport requires http configuration")
        self._validate_url(http.url)
        for name, header in http.headers.items():
            self._validate_header_name(name, schema_version=schema_version)
            self._validate_env_name(header.env, f"header {name} env")
            if not self._env_allowed(header.env, self.config.mcp.header_env_allowlist):
                raise ValidationError(f"MCP header env is not allowlisted: {header.env}")
            if header.prefix not in _ALLOWED_HEADER_PREFIXES:
                raise ValidationError(f"MCP header {name} prefix is not allowed")
            if header.suffix not in _ALLOWED_HEADER_SUFFIXES:
                raise ValidationError(f"MCP header {name} suffix is not allowed")

    def _validate_tool(self, tool: McpToolSpec, *, schema_version: int) -> None:
        self._validate_identifier(tool.tool_id, "tool_id", self.config.mcp.tool_id_max_chars)
        if not tool.mcp_name or len(tool.mcp_name) > self.config.mcp.mcp_name_max_chars:
            raise ValidationError("MCP mcp_name must be non-empty and within configured length")
        if tool.right not in _CALL_RIGHTS:
            raise ValidationError("MCP tool right must be read, write, or execute")
        try:
            rollback_class = ExternalEffectRollbackClass(tool.rollback_class)
        except ValueError as exc:
            raise ValidationError("MCP rollback_class is invalid") from exc
        if tool.rollback_status is not None:
            try:
                ExternalEffectRollbackStatus(tool.rollback_status)
            except ValueError as exc:
                raise ValidationError("MCP rollback_status is invalid") from exc
        if rollback_class == ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED and tool.rollback_status is None:
            pass
        if rollback_class == ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED and tool.state_mutation:
            raise ValidationError("MCP tool with state_mutation=true cannot use no_rollback_required")
        self._validate_json_schema(
            tool.input_schema,
            "input_schema",
            modern=schema_version == 2,
        )
        self._validate_json_value(tool.metadata, "tool metadata")
        if schema_version == 2:
            self._validate_no_reserved_mcp_meta(tool.metadata, "tool metadata")

    def _header_specs(self, value: Any) -> dict[str, McpHeaderSpec]:
        if not isinstance(value, dict):
            raise ValidationError("MCP headers must be an object")
        headers: dict[str, McpHeaderSpec] = {}
        for name, spec in value.items():
            if type(name) is not str:
                raise ValidationError("MCP header names must be strings")
            if not isinstance(spec, dict):
                raise ValidationError(f"MCP header {name} must be an object")
            _reject_unknown_fields(
                spec,
                _HEADER_FIELDS,
                context=f"MCP header {name}",
            )
            prefix = spec.get("prefix", "")
            suffix = spec.get("suffix", "")
            if type(prefix) is not str or type(suffix) is not str:
                raise ValidationError(
                    f"MCP header {name} prefix and suffix must be strings"
                )
            headers[name] = McpHeaderSpec(
                env=self._required_string(spec, "env", f"MCP header {name}"),
                prefix=prefix,
                suffix=suffix,
            )
        return headers

    def _required(self, value: dict[str, Any], key: str, context: str) -> Any:
        if key not in value:
            raise ValidationError(f"{context} requires {key}")
        return value[key]

    def _required_string(
        self,
        value: dict[str, Any],
        key: str,
        context: str,
    ) -> str:
        selected = self._required(value, key, context)
        if type(selected) is not str:
            raise ValidationError(f"{context} {key} must be a string")
        return selected

    @staticmethod
    def _mapping_field(
        value: dict[str, Any],
        key: str,
        context: str,
    ) -> dict[str, Any]:
        if key not in value:
            return {}
        selected = value[key]
        if not isinstance(selected, dict):
            raise ValidationError(f"{context} {key} must be an object")
        return dict(selected)

    @staticmethod
    def _list_field(
        value: dict[str, Any],
        key: str,
        context: str,
    ) -> list[Any]:
        if key not in value:
            return []
        selected = value[key]
        if not isinstance(selected, list):
            raise ValidationError(f"{context} {key} must be an array")
        return list(selected)

    def _validate_header_name(self, name: str, *, schema_version: int) -> None:
        lowered = name.lower()
        if len(name) > self.config.mcp.header_name_max_chars or not _HEADER_PATTERN.match(name):
            raise ValidationError(f"invalid MCP header name: {name!r}")
        forbidden_headers = (
            _MODERN_FORBIDDEN_HEADERS
            if schema_version == 2
            else _LEGACY_FORBIDDEN_HEADERS
        )
        if lowered in forbidden_headers or (
            schema_version == 2 and lowered.startswith("mcp-param-")
        ):
            raise ValidationError(f"MCP header is forbidden: {name}")

    def _env_allowed(self, name: str, patterns: tuple[str, ...]) -> bool:
        for pattern in patterns:
            if pattern.endswith("*") and name.startswith(pattern[:-1]):
                return True
            if name == pattern:
                return True
        return False

    def _require_runtime_environment(
        self,
        server: McpServerSpec,
        *,
        host_environment: Mapping[str, str] | None = None,
        pinned_stdio_environment: Mapping[str, str] | None = None,
        deadline: float | None = None,
    ) -> Mapping[str, str]:
        if deadline is not None:
            self._remaining_timeout(deadline)
        selected_host_environment = self._runtime_environment_input_snapshot(
            server,
            host_environment=host_environment,
            pinned_stdio_environment=pinned_stdio_environment,
        )
        if deadline is not None:
            self._remaining_timeout(deadline)
        resolved = self._runtime_environment_from_host(
            server,
            selected_host_environment,
            pinned_stdio_environment=pinned_stdio_environment,
        )
        if deadline is not None:
            self._remaining_timeout(deadline)
        return resolved

    def snapshot_modern_transport_environment(
        self,
        server: McpServerSpec,
    ) -> McpTransportEnvironmentSnapshot:
        """Capture one immutable Host input snapshot for modern composition.

        This narrow bridge prevents the Runtime binding resolver from reaching
        through the primitive's private environment normalization helpers.
        It returns a repr-safe value object and performs no Provider I/O.
        """

        selected_input = self._runtime_environment_input_snapshot(server)
        resolved = self._require_runtime_environment(
            server,
            host_environment=selected_input,
        )
        return McpTransportEnvironmentSnapshot(
            runtime_environment=selected_input,
            sensitive_values=mcp_runtime_secret_values(server, resolved),
        )

    def _runtime_environment_input_snapshot(
        self,
        server: McpServerSpec,
        *,
        host_environment: Mapping[str, str] | None = None,
        pinned_stdio_environment: Mapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        source = os.environ if host_environment is None else host_environment
        pinned_stdio = pinned_stdio_environment or {}
        names: set[str] = set()
        pinned_host: dict[str, str] = {}
        if server.transport == "stdio" and server.stdio is not None:
            names.update(_MCP_PLATFORM_ENV_KEYS)
            pinned_host = {
                server.stdio.env[child_name]: value
                for child_name, value in pinned_stdio.items()
                if child_name in server.stdio.env
            }
            names.update(
                host_name
                for host_name in server.stdio.env.values()
                if host_name not in pinned_host
            )
        elif server.transport == "streamable_http" and server.http is not None:
            names.update(header.env for header in server.http.headers.values())
        names.difference_update(pinned_host)
        selected = dict(pinned_host)
        for name in sorted(names):
            resolved = source.get(name)
            if resolved is not None:
                selected[name] = resolved
        return MappingProxyType(selected)

    def _runtime_environment_from_host(
        self,
        server: McpServerSpec,
        host_environment: Mapping[str, str],
        *,
        pinned_stdio_environment: Mapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        resolved_environment: dict[str, str] = {}
        pinned_stdio = pinned_stdio_environment or {}
        if server.transport == "stdio" and server.stdio is not None:
            # Windows needs these bootstrap variables to create a child
            # process.  Capture them at the primitive boundary alongside the
            # manifest-declared child environment so provider dispatch never
            # has to consult a newer ambient environment.
            for name in _MCP_PLATFORM_ENV_KEYS:
                resolved = host_environment.get(name)
                if resolved is None:
                    continue
                if "\x00" in resolved:
                    raise ValidationError(f"MCP stdio env {name} contains NUL byte")
                resolved_environment[name] = resolved
            for child_name, host_name in server.stdio.env.items():
                resolved = pinned_stdio.get(child_name)
                if resolved is None:
                    resolved = host_environment.get(host_name)
                if resolved is None:
                    raise ValidationError(f"missing environment variable for MCP stdio env {child_name}: {host_name}")
                if "\x00" in resolved:
                    raise ValidationError(f"MCP stdio env {child_name} contains NUL byte")
                resolved_environment[child_name] = resolved
        if server.transport == "streamable_http" and server.http is not None:
            for name, header in server.http.headers.items():
                resolved = host_environment.get(header.env)
                if resolved is None:
                    raise ValidationError(f"missing environment variable for MCP header {name}: {header.env}")
                header_value = f"{header.prefix}{resolved}{header.suffix}"
                if len(header_value) > self.config.mcp.header_value_max_chars or "\r" in header_value or "\n" in header_value:
                    raise ValidationError(f"MCP header {name} resolved value is invalid")
                resolved_environment[name] = header_value
        # One immutable snapshot spans all provider stages in this operation,
        # including legacy MCP providers that use separate tools/list and
        # call_tool sessions.  The SDK provider must not re-read os.environ.
        return MappingProxyType(resolved_environment)

    def _validate_url(self, url: str) -> None:
        try:
            parsed = urlsplit(url)
        except ValueError as exc:
            raise ValidationError("MCP HTTP URL is invalid") from exc
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValidationError("MCP HTTP URL has invalid port") from exc
        if parsed.scheme not in {"http", "https"}:
            raise ValidationError("MCP HTTP URL must use http or https")
        if parsed.username is not None or parsed.password is not None:
            raise ValidationError("MCP HTTP URL must not include userinfo")
        if parsed.fragment:
            raise ValidationError("MCP HTTP URL must not include a fragment")
        host = parsed.hostname
        if not host:
            raise ValidationError("MCP HTTP URL must include a host")
        if host.lower() in _FORBIDDEN_MCP_HOSTS:
            raise ValidationError("MCP HTTP host is not allowed")
        if parsed.scheme == "http" and host not in _LOCAL_HTTP_HOSTS:
            raise ValidationError("MCP plain HTTP is allowed only for local development hosts")
        self._validate_host_literal(host, allow_local=host in _LOCAL_HTTP_HOSTS)

    def _validate_runtime_resolution(
        self,
        server: McpServerSpec,
        *,
        deadline: float | None = None,
    ) -> tuple[str, ...]:
        if server.http is None:
            return ()
        parsed = urlsplit(server.http.url)
        host = parsed.hostname
        if not host:
            raise ValidationError("MCP HTTP URL must include a host")
        if host in _LOCAL_HTTP_HOSTS:
            return ()
        selected_deadline = (
            time.monotonic() + server.timeout_s if deadline is None else deadline
        )
        try:
            addresses = _allowed_mcp_connect_addresses(
                host,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                deadline=selected_deadline,
            )
        except TimeoutError as exc:
            raise ProviderHostError(
                code="mcp_dns_timeout",
                error_type=type(exc).__name__,
                correlation_id=new_id("corr"),
            ) from None
        return tuple(addresses)

    def _runtime_resolution_observes_host(self, server: McpServerSpec) -> bool:
        if server.http is None:
            return False
        host = urlsplit(server.http.url).hostname
        return bool(host and host.lower() not in _LOCAL_HTTP_HOSTS)

    def _validate_host_literal(self, host: str, *, allow_local: bool) -> None:
        try:
            ip = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            return
        if allow_local:
            return
        if not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValidationError("MCP HTTP IP address is not allowed")

    def _disable_replaced_server_tool_capabilities(self, server_id: str, *, actor: str) -> None:
        prefix = f"mcp:{server_id}:"
        for cap in self.authority.list_capabilities():
            if not cap.active or cap.revoked:
                continue
            if cap.resource == f"mcp:{server_id}:*" or cap.resource.startswith(prefix):
                self.capabilities.disable_subject_capability(
                    cap.cap_id,
                    actor=actor,
                    reason="MCP server spec replaced; tool authority must be reissued",
                )

    def _validate_identifier(self, value: str, field: str, max_chars: int) -> None:
        if not isinstance(value, str) or not value or len(value) > max_chars or not _ID_PATTERN.match(value):
            raise ValidationError(f"invalid MCP {field}: {value!r}")

    def _validate_env_name(self, value: str, field: str) -> None:
        if not value or not _ENV_PATTERN.match(value):
            raise ValidationError(f"invalid MCP {field}: {value!r}")

    def _coerce_bool(self, value: Any, field: str) -> bool:
        if not isinstance(value, bool):
            raise ValidationError(f"MCP {field} must be a boolean")
        return value

    def _coerce_positive_float(self, value: Any, field: str) -> float:
        if type(value) not in {int, float}:
            raise ValidationError(f"MCP {field} must be a number")
        selected = float(value)
        if not math.isfinite(selected) or selected <= 0:
            raise ValidationError(f"MCP {field} must be > 0")
        return selected

    def _coerce_positive_int(self, value: Any, field: str) -> int:
        if type(value) is not int:
            raise ValidationError(f"MCP {field} must be an integer")
        selected = value
        if selected <= 0:
            raise ValidationError(f"MCP {field} must be > 0")
        return selected

    def _validate_json_value(self, value: Any, field: str) -> None:
        try:
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValidationError(f"MCP {field} must be JSON-serializable") from exc

    def _validate_no_reserved_mcp_meta(self, value: Any, field: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str:
                    raise ValidationError(
                        f"MCP {field} object keys must be strings"
                    )
                if key == "_meta" or key.startswith("io.modelcontextprotocol/"):
                    raise ValidationError(
                        f"MCP {field} must not define protocol-reserved _meta"
                    )
                self._validate_no_reserved_mcp_meta(item, field)
        elif isinstance(value, list):
            for item in value:
                self._validate_no_reserved_mcp_meta(item, field)

    def _validate_json_schema(
        self,
        schema: dict[str, Any],
        field: str,
        *,
        modern: bool = False,
    ) -> None:
        if not schema:
            return
        self._validate_json_value(schema, field)
        if modern:
            self._validate_v2_json_schema_safety(schema, field)
        self._validate_schema_regex_patterns(schema, field)
        try:
            jsonschema_validator_for(schema).check_schema(schema)
        except JsonSchemaSchemaError as exc:
            raise ValidationError(f"MCP {field} is not a valid JSON Schema") from exc

    def _validate_arguments_against_schema(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> None:
        if not tool.input_schema:
            if deadline is not None:
                self._remaining_timeout(deadline)
            return
        if deadline is not None:
            self._remaining_timeout(deadline)
        if server.schema_version == 2:
            self._validate_v2_json_schema_safety(
                tool.input_schema,
                "input_schema",
                deadline=deadline,
            )
        regex_budget = self._schema_regex_budget(deadline=deadline)
        self._validate_schema_regex_patterns(
            tool.input_schema,
            "input_schema",
            budget=regex_budget,
        )
        try:
            regex_budget.checkpoint()
            validator = jsonschema_validator_for(tool.input_schema)
            validator.check_schema(tool.input_schema)
            self._bounded_schema_validator(
                tool.input_schema,
                field="input_schema",
                budget=regex_budget,
            ).validate(arguments)
            regex_budget.checkpoint()
        except JsonSchemaValidationError as exc:
            raise ValidationError(f"MCP tool arguments failed schema validation: {exc.message}") from exc
        except JsonSchemaSchemaError as exc:
            raise ValidationError("MCP tool input_schema is invalid") from exc

    def _schema_regex_budget(
        self,
        *,
        deadline: float | None = None,
    ) -> _McpSchemaRegexBudget:
        config = getattr(self, "config", DEFAULT_CONFIG).mcp
        return _McpSchemaRegexBudget(
            pattern_max_bytes=config.schema_regex_pattern_max_bytes,
            max_evaluations=config.schema_regex_max_evaluations,
            timeout_s=config.schema_regex_match_timeout_s,
            operation_deadline=deadline,
        )

    @staticmethod
    def _iter_schema_nodes(schema: dict[str, Any]) -> Any:
        """Yield actual schema nodes without treating property names as keywords."""

        single_schema_keywords = {
            "additionalItems",
            "additionalProperties",
            "contains",
            "contentSchema",
            "else",
            "if",
            "items",
            "not",
            "propertyNames",
            "then",
            "unevaluatedItems",
            "unevaluatedProperties",
        }
        schema_array_keywords = {
            "allOf",
            "anyOf",
            "oneOf",
            "prefixItems",
        }
        schema_map_keywords = {
            "$defs",
            "definitions",
            "dependentSchemas",
            "patternProperties",
            "properties",
        }
        pending: list[Any] = [schema]
        seen: set[int] = set()
        while pending:
            node = pending.pop()
            if type(node) is not dict or id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            for keyword in single_schema_keywords:
                child = node.get(keyword)
                if type(child) is dict:
                    pending.append(child)
                elif keyword == "items" and type(child) is list:
                    pending.extend(child)
            for keyword in schema_array_keywords:
                children = node.get(keyword)
                if type(children) is list:
                    pending.extend(children)
            for keyword in schema_map_keywords:
                children = node.get(keyword)
                if type(children) is dict:
                    pending.extend(children.values())
            dependencies = node.get("dependencies")
            if type(dependencies) is dict:
                pending.extend(
                    child
                    for child in dependencies.values()
                    if type(child) is dict
                )

    def _validate_schema_regex_patterns(
        self,
        schema: dict[str, Any],
        field: str,
        *,
        budget: _McpSchemaRegexBudget | None = None,
    ) -> None:
        selected_budget = budget or self._schema_regex_budget()
        for node in self._iter_schema_nodes(schema):
            selected_budget.checkpoint()
            pattern = node.get("pattern")
            if pattern is not None:
                selected_budget.compile(pattern, field=field)
            pattern_properties = node.get("patternProperties")
            if type(pattern_properties) is dict:
                for candidate in pattern_properties:
                    selected_budget.compile(candidate, field=field)
        selected_budget.checkpoint()

    def _bounded_schema_validator(
        self,
        schema: dict[str, Any],
        *,
        field: str,
        budget: _McpSchemaRegexBudget | None = None,
    ) -> Any:
        base_validator = jsonschema_validator_for(schema)
        callbacks = _McpBoundedSchemaCallbacks(
            budget or self._schema_regex_budget(),
            field=field,
        )
        bounded_validator = extend_jsonschema_validator(
            base_validator,
            validators=callbacks.overrides(base_validator),
        )
        return bounded_validator(schema)

    def _validate_v2_json_schema_safety(
        self,
        schema: dict[str, Any],
        field: str,
        *,
        deadline: float | None = None,
    ) -> None:
        """Validate the bounded JSON Schema 2020-12 subset used by Manifest v2."""

        if schema.get("type") != "object":
            raise ValidationError(
                f"MCP {field} schema_version 2 root type must be object"
            )
        node_count = 0
        combinator_expansion = 1
        local_refs: list[tuple[dict[str, Any], str]] = []

        def walk(value: Any, *, depth: int, path: str) -> None:
            nonlocal node_count, combinator_expansion
            if deadline is not None:
                self._remaining_timeout(deadline)
            if depth > self.config.mcp.schema_max_depth:
                raise ValidationError(
                    f"MCP {field} exceeds schema depth={self.config.mcp.schema_max_depth}"
                )
            node_count += 1
            if node_count > self.config.mcp.schema_max_nodes:
                raise ValidationError(
                    f"MCP {field} exceeds schema nodes={self.config.mcp.schema_max_nodes}"
                )
            if isinstance(value, dict):
                if "if" in value and ("then" in value or "else" in value):
                    combinator_expansion *= 2
                    if (
                        combinator_expansion
                        > self.config.mcp.schema_max_composition_expansions
                    ):
                        raise ValidationError(
                            "MCP "
                            f"{field} exceeds combinator expansion="
                            f"{self.config.mcp.schema_max_composition_expansions}"
                        )
                for key, item in value.items():
                    if key in _MCP_V2_DYNAMIC_REFERENCE_KEYS:
                        raise ValidationError(
                            f"MCP {field} does not allow dynamic or recursive references"
                        )
                    if key == "$ref":
                        if not isinstance(item, str) or not item.startswith("#"):
                            raise ValidationError(
                                f"MCP {field} external $ref is not allowed"
                            )
                        local_refs.append((value, item))
                    if key in {"allOf", "anyOf", "oneOf"}:
                        if not isinstance(item, list):
                            raise ValidationError(
                                f"MCP {field} {key} must be an array"
                            )
                        combinator_expansion *= max(1, len(item))
                        if (
                            combinator_expansion
                            > self.config.mcp.schema_max_composition_expansions
                        ):
                            raise ValidationError(
                                "MCP "
                                f"{field} exceeds combinator expansion="
                                f"{self.config.mcp.schema_max_composition_expansions}"
                            )
                    walk(item, depth=depth + 1, path=f"{path}/{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, depth=depth + 1, path=f"{path}/{index}")

        walk(schema, depth=0, path="#")
        if deadline is not None:
            self._remaining_timeout(deadline)
        for _source, reference in local_refs:
            if deadline is not None:
                self._remaining_timeout(deadline)
            self._resolve_local_schema_ref(schema, reference, field)
        self._reject_cyclic_schema_refs(schema, local_refs, field)
        if deadline is not None:
            self._remaining_timeout(deadline)

    @staticmethod
    def _resolve_local_schema_ref(
        schema: dict[str, Any],
        reference: str,
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

    def _reject_cyclic_schema_refs(
        self,
        schema: dict[str, Any],
        references: list[tuple[dict[str, Any], str]],
        field: str,
    ) -> None:
        # Build one bounded graph for both containment and reference edges.
        # The former implementation recursively scanned a reference target for
        # every `$ref`, turning a valid schema with many references to one large
        # shared definition into quadratic work.
        nodes, edges = self._schema_containment_graph(schema)
        self._add_schema_reference_edges(
            schema,
            references,
            field,
            nodes=nodes,
            edges=edges,
        )
        self._validate_schema_reference_graph(nodes, edges, field)

    @staticmethod
    def _schema_containment_graph(
        schema: dict[str, Any],
    ) -> tuple[
        dict[int, dict[str, Any] | list[Any]],
        dict[int, list[tuple[int, int]]],
    ]:
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
            children = (
                (item for key, item in value.items() if key != "$ref")
                if isinstance(value, dict)
                else iter(value)
            )
            for child in children:
                if not isinstance(child, (dict, list)):
                    continue
                child_identity = id(child)
                edges[identity].append((child_identity, 0))
                pending.append(child)
        return nodes, edges

    def _add_schema_reference_edges(
        self,
        schema: dict[str, Any],
        references: list[tuple[dict[str, Any], str]],
        field: str,
        *,
        nodes: dict[int, dict[str, Any] | list[Any]],
        edges: dict[int, list[tuple[int, int]]],
    ) -> None:
        for source, reference in references:
            target = self._resolve_local_schema_ref(schema, reference, field)
            if not isinstance(target, (dict, list)):
                continue
            source_identity = id(source)
            target_identity = id(target)
            nodes.setdefault(source_identity, source)
            nodes.setdefault(target_identity, target)
            edges.setdefault(source_identity, []).append((target_identity, 1))
            edges.setdefault(target_identity, [])

    def _validate_schema_reference_graph(
        self,
        nodes: dict[int, dict[str, Any] | list[Any]],
        edges: dict[int, list[tuple[int, int]]],
        field: str,
    ) -> None:
        indegree = {identity: 0 for identity in nodes}
        for outgoing in edges.values():
            for target_identity, _ref_edge in outgoing:
                indegree[target_identity] += 1
        ready = deque(
            identity for identity, degree in indegree.items() if degree == 0
        )
        ref_hops = {identity: 0 for identity in nodes}
        processed = 0
        while ready:
            identity = ready.popleft()
            processed += 1
            for target_identity, ref_edge in edges[identity]:
                candidate_hops = ref_hops[identity] + ref_edge
                if candidate_hops > ref_hops[target_identity]:
                    ref_hops[target_identity] = candidate_hops
                if (
                    ref_hops[target_identity]
                    > self.config.mcp.schema_max_ref_hops
                ):
                    raise ValidationError(
                        "MCP "
                        f"{field} exceeds local $ref hops="
                        f"{self.config.mcp.schema_max_ref_hops}"
                    )
                indegree[target_identity] -= 1
                if indegree[target_identity] == 0:
                    ready.append(target_identity)

        if processed != len(nodes):
            raise ValidationError(
                f"MCP {field} recursive local $ref is not allowed"
            )

    def _bounded_list_limit(self, limit: int | None) -> int:
        maximum = self.config.mcp.server_page_limit
        selected = maximum if limit is None else limit
        if not isinstance(selected, int):
            raise ValidationError("MCP server list limit must be an integer")
        if selected < 1:
            raise ValidationError("MCP server list limit must be >= 1")
        if selected > maximum:
            raise ValidationError(
                f"MCP server list limit exceeds configured maximum {maximum}"
            )
        return selected

    def _load_server(
        self,
        server_id: str,
    ) -> tuple[McpRegisteredServer, dict[str, Any]]:
        self._validate_identifier(server_id, "server_id", self.config.mcp.server_id_max_chars)
        found = self.extensions.get_mcp_server_manifest(server_id)
        if found is None:
            raise NotFound(f"MCP server not found: {server_id}")
        spec, metadata = found
        self._validate_server(spec)
        return spec, metadata

    def _server_to_json(
        self,
        server: McpRegisteredServer,
        metadata: dict[str, Any],
        *,
        include_sensitive_fields: bool,
    ) -> dict[str, Any]:
        transport: dict[str, Any]
        if server.transport == "stdio" and server.stdio is not None:
            transport = {
                "type": "stdio",
                "command": server.stdio.command,
                "args": list(server.stdio.args),
                "env": {name: {"env": host_name} for name, host_name in server.stdio.env.items()},
                "cwd": server.stdio.cwd,
            }
        elif server.http is not None:
            transport = {
                "type": "streamable_http",
                "url": server.http.url if include_sensitive_fields else None,
                "headers": {
                    name: {
                        "env": header.env,
                        "prefix": header.prefix,
                        "suffix": header.suffix,
                    }
                    for name, header in server.http.headers.items()
                },
            }
        else:
            transport = {"type": server.transport}
        payload = {
            "schema_version": server.schema_version,
            "server_id": server.server_id,
            "protocol_mode": self._effective_protocol_mode(server).value,
            "transport": transport,
            "stdio_authority_resource": self.stdio_resource_for_server(server),
            "tools": [self._tool_to_json(server.server_id, tool) for tool in server.tools],
            "timeout_s": server.timeout_s,
            "max_request_bytes": server.max_request_bytes,
            "max_response_bytes": server.max_response_bytes,
            "metadata": server.metadata,
            **metadata,
        }
        if isinstance(server, McpServerManifestV3):
            payload["manifest_sha256"] = self._server_spec_sha256(server)
            if include_sensitive_fields:
                payload.update(
                    {
                        "resources": [to_jsonable(item) for item in server.resources],
                        "resource_templates": [
                            to_jsonable(item) for item in server.resource_templates
                        ],
                        "prompts": [to_jsonable(item) for item in server.prompts],
                        "auth_profile_id": server.auth_profile_id,
                        "subscriptions": list(server.subscriptions),
                        "tasks_extension": (
                            to_jsonable(server.tasks_extension)
                            if server.tasks_extension is not None
                            else None
                        ),
                    }
                )
        return payload

    def _tool_to_json(self, server_id: str, tool: McpToolSpec, *, live: McpProviderTool | None = None) -> dict[str, Any]:
        payload = {
            "tool_id": tool.tool_id,
            "mcp_name": tool.mcp_name,
            "right": tool.right,
            "resource": self.tool_resource(server_id, tool.tool_id),
            "rollback_class": tool.rollback_class,
            "rollback_status": (
                tool.rollback_status
                if tool.rollback_status is not None
                else default_external_effect_rollback_status(
                    ExternalEffectRollbackClass(tool.rollback_class)
                ).value
            ),
            "state_mutation": tool.state_mutation,
            "information_flow": tool.information_flow,
            "input_schema": tool.input_schema,
            "metadata": tool.metadata,
        }
        if live is not None:
            payload["live"] = {
                "name": live.name,
                "description": live.description,
                "input_schema": live.input_schema,
                "schema_matches_manifest": not tool.input_schema
                or _json_values_equivalent(
                    live.input_schema,
                    tool.input_schema,
                ),
            }
        return payload

    def _profile_json(self, profile: Any) -> dict[str, Any]:
        return to_jsonable(profile)
