"""Bounded parsing for untrusted MCP MRTR and Tasks payloads.

This module intentionally has no transport or Store access.  Raw server
request keys and ``requestState`` values are returned only for placement in a
Host credential broker; public projections use local request identifiers.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Mapping
from urllib.parse import urlsplit

from agent_libos.mcp.app_policy import (
    is_mcp_app_metadata_key,
    reject_mcp_app_text,
)
from agent_libos.mcp.types import (
    JsonValue,
    McpInputRequest,
    McpInputRequestKind,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.redaction import redact_sensitive_text


_MAX_TREE_DEPTH = 64
_MAX_TREE_NODES = 10_000
_MAX_JSON_BYTES = 256 * 1024
_MAX_INPUT_REQUESTS = 16
_MAX_TEXT_CHARS = 8_192
_MAX_SCHEMA_PROPERTIES = 64
_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def strict_json_value(
    value: Any,
    *,
    label: str,
    max_bytes: int = _MAX_JSON_BYTES,
) -> JsonValue:
    """Return the same strict JSON tree while rejecting coercible Python data."""

    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_TREE_NODES:
            raise ValidationError(f"{label} exceeds the JSON node limit")
        if depth > _MAX_TREE_DEPTH:
            raise ValidationError(f"{label} exceeds the JSON depth limit")
        if current is None or type(current) in {bool, int, str}:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise ValidationError(f"{label} contains a non-finite number")
            continue
        if type(current) is list:
            pending.extend((item, depth + 1) for item in current)
            continue
        if type(current) is dict:
            for key, item in current.items():
                if type(key) is not str:
                    raise ValidationError(f"{label} contains a non-string object key")
                pending.append((item, depth + 1))
            continue
        raise ValidationError(f"{label} contains a non-JSON value")
    encoded = canonical_json_bytes(value, label=label, validate=False)
    if len(encoded) > max_bytes:
        raise ValidationError(f"{label} exceeds the JSON byte limit")
    return value


def canonical_json_bytes(
    value: Any,
    *,
    label: str,
    validate: bool = True,
    max_bytes: int = _MAX_JSON_BYTES,
) -> bytes:
    if validate:
        strict_json_value(value, label=label, max_bytes=max_bytes)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError(f"{label} is not canonical JSON") from exc
    if len(encoded) > max_bytes:
        raise ValidationError(f"{label} exceeds the JSON byte limit")
    return encoded


def json_sha256(value: Any, *, label: str) -> str:
    return sha256(canonical_json_bytes(value, label=label)).hexdigest()


def decode_broker_json(
    value: bytes,
    *,
    label: str,
    max_bytes: int = _MAX_JSON_BYTES,
) -> dict[str, JsonValue]:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValidationError(f"{label} byte limit is invalid")
    if type(value) is not bytes or len(value) > max_bytes:
        raise ValidationError(f"{label} is invalid")
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValidationError(f"{label} is invalid") from exc
    selected = strict_json_value(decoded, label=label, max_bytes=max_bytes)
    if type(selected) is not dict:
        raise ValidationError(f"{label} is invalid")
    if canonical_json_bytes(selected, label=label, max_bytes=max_bytes) != value:
        raise ValidationError(f"{label} is not canonical")
    return selected


def sanitize_provider_json(
    value: Any,
    *,
    sensitive_values: tuple[str, ...] = (),
    label: str = "MCP Provider result",
) -> JsonValue:
    """Create an exact-secret-redacted public tree and exclude MCP Apps data."""

    strict_json_value(value, label=label)

    def visit(current: JsonValue) -> JsonValue:
        if type(current) is str:
            reject_mcp_app_text(current)
            return redact_sensitive_text(current, sensitive_values=sensitive_values)
        if type(current) is list:
            return [visit(item) for item in current]
        if type(current) is dict:
            selected: dict[str, JsonValue] = {}
            for key, item in current.items():
                if is_mcp_app_metadata_key(key):
                    continue
                reject_mcp_app_text(key)
                public_key = redact_sensitive_text(
                    key, sensitive_values=sensitive_values
                )
                if public_key in selected:
                    raise ValidationError(
                        "MCP Provider result keys collide after secret redaction"
                    )
                selected[public_key] = visit(item)
            return selected
        return current

    sanitized = visit(value)
    strict_json_value(sanitized, label=f"sanitized {label}")
    return sanitized


def reject_opaque_secret_reflection(
    value: Any,
    *,
    sensitive_values: tuple[str, ...],
    label: str,
) -> str:
    """Reject an opaque echo value that cannot be safely redacted in place."""

    if type(value) is not str:
        raise ValidationError(f"{label} is invalid")
    if redact_sensitive_text(value, sensitive_values=sensitive_values) != value:
        raise ValidationError(f"{label} reflected an operation secret")
    return value


def sdk_json_mapping(value: Any, *, label: str) -> dict[str, JsonValue]:
    """Detach an official SDK model without permissive Python coercions."""

    def convert(current: Any) -> JsonValue:
        if current is None or type(current) in {bool, int, float, str}:
            return current
        if type(current) is list:
            return [convert(item) for item in current]
        if isinstance(current, Mapping):
            selected: dict[str, JsonValue] = {}
            for key, item in current.items():
                if type(key) is not str:
                    raise ValidationError(f"{label} contains a non-string key")
                selected[key] = convert(item)
            return selected
        model_dump = getattr(current, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_none=True,
                )
            except TypeError:
                dumped = model_dump()
            return convert(dumped)
        raise ValidationError(f"{label} contains a non-JSON SDK value")

    selected = convert(value)
    if type(selected) is not dict:
        raise ValidationError(f"{label} must be an object")
    strict_json_value(selected, label=label)
    return selected


@dataclass(frozen=True)
class ParsedInputRequests:
    public: tuple[McpInputRequest, ...]
    remote_keys: tuple[str, ...]
    raw: dict[str, JsonValue]

    @property
    def has_unsupported(self) -> bool:
        return any(
            item.kind is not McpInputRequestKind.ELICITATION for item in self.public
        )


def parse_input_requests(
    value: Any,
    *,
    sensitive_values: tuple[str, ...] = (),
    max_requests: int = _MAX_INPUT_REQUESTS,
) -> ParsedInputRequests:
    if (
        type(max_requests) is not int
        or max_requests < 1
        or max_requests > _MAX_INPUT_REQUESTS
    ):
        raise ValidationError("MCP input request limit is invalid")
    if type(value) is not dict or len(value) > max_requests:
        raise ValidationError("MCP inputRequests must be a bounded object")
    strict_json_value(value, label="MCP inputRequests")
    public: list[McpInputRequest] = []
    remote_keys: list[str] = []
    raw: dict[str, JsonValue] = {}
    for index, remote_key in enumerate(sorted(value), start=1):
        if not remote_key or len(remote_key) > 256 or "\x00" in remote_key:
            raise ValidationError("MCP input request key is invalid")
        reject_opaque_secret_reflection(
            remote_key,
            sensitive_values=sensitive_values,
            label="MCP input request key",
        )
        request = value[remote_key]
        if type(request) is not dict or set(request) != {"method", "params"}:
            raise ValidationError("MCP input request shape is invalid")
        method = request.get("method")
        params = request.get("params")
        local_id = f"input-{index}"
        if method == "elicitation/create":
            parsed = _parse_elicitation(local_id, params, sensitive_values)
        elif method == "sampling/createMessage":
            parsed = McpInputRequest(
                request_id=local_id,
                kind=McpInputRequestKind.SAMPLING_UNSUPPORTED,
            )
        elif method == "roots/list":
            parsed = McpInputRequest(
                request_id=local_id,
                kind=McpInputRequestKind.ROOTS_UNSUPPORTED,
            )
        else:
            raise ValidationError("MCP input request method is unsupported")
        public.append(parsed)
        remote_keys.append(remote_key)
        sanitized_request = sanitize_provider_json(
            request,
            sensitive_values=sensitive_values,
            label="MCP input request",
        )
        if type(sanitized_request) is not dict:  # pragma: no cover - request invariant
            raise ValidationError("MCP input request shape is invalid")
        raw[remote_key] = sanitized_request
    return ParsedInputRequests(tuple(public), tuple(remote_keys), raw)


def _parse_elicitation(
    local_id: str,
    params: Any,
    sensitive_values: tuple[str, ...],
) -> McpInputRequest:
    if type(params) is not dict:
        raise ValidationError("MCP elicitation params are invalid")
    mode = params.get("mode", "form")
    message = params.get("message")
    if type(message) is not str or not message or len(message) > _MAX_TEXT_CHARS:
        raise ValidationError("MCP elicitation message is invalid")
    public_message = redact_sensitive_text(message, sensitive_values=sensitive_values)
    if mode == "form":
        if set(params) - {"mode", "message", "requestedSchema"}:
            raise ValidationError("MCP form elicitation contains unsupported fields")
        schema = _validate_elicitation_schema(params.get("requestedSchema"))
        sanitized = sanitize_provider_json(
            schema,
            sensitive_values=sensitive_values,
            label="MCP elicitation schema",
        )
        assert type(sanitized) is dict
        return McpInputRequest(
            request_id=local_id,
            kind=McpInputRequestKind.ELICITATION,
            mode="form",
            prompt=public_message,
            schema=sanitized,
        )
    if mode == "url":
        if set(params) != {"mode", "message", "url"}:
            raise ValidationError("MCP URL elicitation shape is invalid")
        url = params.get("url")
        if type(url) is not str or len(url) > _MAX_TEXT_CHARS:
            raise ValidationError("MCP URL elicitation URL is invalid")
        try:
            parsed_url = urlsplit(url)
            hostname = parsed_url.hostname
        except ValueError as exc:
            raise ValidationError("MCP URL elicitation URL is invalid") from exc
        if parsed_url.scheme not in {"https", "http"} or not hostname:
            raise ValidationError("MCP URL elicitation requires an absolute HTTP(S) URL")
        return McpInputRequest(
            request_id=local_id,
            kind=McpInputRequestKind.ELICITATION,
            mode="url",
            prompt=public_message,
            inert_url=redact_sensitive_text(url, sensitive_values=sensitive_values),
        )
    raise ValidationError("MCP elicitation mode is unsupported")


def _validate_elicitation_schema(value: Any) -> dict[str, JsonValue]:
    if type(value) is not dict or set(value) - {
        "$schema",
        "type",
        "properties",
        "required",
    }:
        raise ValidationError("MCP elicitation requestedSchema is invalid")
    if value.get("type") != "object":
        raise ValidationError("MCP elicitation requestedSchema type must be object")
    properties = _validated_schema_properties(value.get("properties"))
    required = _validated_required_properties(
        value.get("required", []),
        properties,
    )
    selected: dict[str, JsonValue] = {
        "type": "object",
        "properties": properties,
    }
    if "$schema" in value:
        selected["$schema"] = _validated_schema_identifier(value["$schema"])
    if required:
        selected["required"] = required
    strict_json_value(selected, label="MCP elicitation requestedSchema")
    return selected


def _validated_schema_properties(value: Any) -> dict[str, JsonValue]:
    if type(value) is not dict or len(value) > _MAX_SCHEMA_PROPERTIES:
        raise ValidationError("MCP elicitation properties are invalid")
    selected: dict[str, JsonValue] = {}
    for name, spec in value.items():
        if (
            type(name) is not str
            or not name
            or len(name) > 128
            or type(spec) is not dict
        ):
            raise ValidationError("MCP elicitation property is invalid")
        _validate_primitive_schema(spec)
        selected[name] = spec
    return selected


def _validated_required_properties(
    value: Any,
    properties: Mapping[str, JsonValue],
) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValidationError("MCP elicitation required properties are invalid")
    if len(set(value)) != len(value) or any(item not in properties for item in value):
        raise ValidationError("MCP elicitation required properties are invalid")
    return value


def _validated_schema_identifier(value: Any) -> str:
    if type(value) is not str or len(value) > 512:
        raise ValidationError("MCP elicitation schema identifier is invalid")
    return value


def _validate_primitive_schema(spec: dict[str, Any]) -> None:
    selected_type = spec.get("type")
    common = {"type", "title", "description", "default"}
    if selected_type == "string":
        allowed = common | _validate_string_schema(spec)
    elif selected_type in {"number", "integer"}:
        allowed = common | _validate_numeric_schema(spec)
    elif selected_type == "boolean":
        allowed = common
    elif selected_type == "array":
        allowed = common | _validate_array_schema(spec)
    else:
        raise ValidationError("MCP elicitation property type is unsupported")
    if set(spec) - allowed:
        raise ValidationError("MCP elicitation property contains unsupported fields")
    for key in ("title", "description", "format"):
        item = spec.get(key)
        if item is not None and (type(item) is not str or len(item) > 2_048):
            raise ValidationError("MCP elicitation property text is invalid")
    strict_json_value(spec, label="MCP elicitation property schema")
    if "default" in spec:
        _validate_primitive_value(spec["default"], spec)


def _validate_string_schema(spec: Mapping[str, Any]) -> set[str]:
    selected_format = spec.get("format")
    if selected_format is not None and selected_format not in {
        "email",
        "uri",
        "date",
        "date-time",
    }:
        raise ValidationError("MCP elicitation string format is unsupported")
    _validate_nonnegative_pair(spec, "minLength", "maxLength")
    enum = spec.get("enum")
    one_of = spec.get("oneOf")
    if enum is not None and one_of is not None:
        raise ValidationError("MCP elicitation enum schema is ambiguous")
    if enum is not None:
        _validate_string_options(enum, "MCP elicitation enum")
        names = spec.get("enumNames")
        if names is not None:
            _validate_string_options(names, "MCP elicitation enum names")
            if len(names) != len(enum):
                raise ValidationError("MCP elicitation enum names are misaligned")
    elif "enumNames" in spec:
        raise ValidationError("MCP elicitation enumNames requires enum")
    if one_of is not None:
        _validate_titled_options(one_of, "MCP elicitation oneOf")
    return {"minLength", "maxLength", "format", "enum", "oneOf", "enumNames"}


def _validate_numeric_schema(spec: Mapping[str, Any]) -> set[str]:
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    for selected in (minimum, maximum):
        if selected is not None and (
            type(selected) not in {int, float} or not math.isfinite(float(selected))
        ):
            raise ValidationError("MCP elicitation numeric bound is invalid")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValidationError("MCP elicitation numeric bounds are inconsistent")
    return {"minimum", "maximum"}


def _validate_array_schema(spec: Mapping[str, Any]) -> set[str]:
    # Multi-select enums are the only array form allowed by MCP Elicitation.
    items = spec.get("items")
    if type(items) is not dict:
        raise ValidationError("MCP elicitation multi-select schema is invalid")
    if set(items) == {"type", "enum"} and items.get("type") == "string":
        _validate_string_options(items.get("enum"), "MCP elicitation multi-select enum")
    elif set(items) == {"anyOf"}:
        _validate_titled_options(items.get("anyOf"), "MCP elicitation multi-select anyOf")
    else:
        raise ValidationError("MCP elicitation multi-select schema is invalid")
    _validate_nonnegative_pair(spec, "minItems", "maxItems")
    return {"items", "minItems", "maxItems"}


def _validate_nonnegative_pair(
    value: Mapping[str, Any],
    minimum_name: str,
    maximum_name: str,
) -> None:
    minimum = value.get(minimum_name)
    maximum = value.get(maximum_name)
    for selected in (minimum, maximum):
        if selected is not None and (type(selected) is not int or selected < 0):
            raise ValidationError("MCP elicitation size bound is invalid")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValidationError("MCP elicitation size bounds are inconsistent")


def _validate_string_options(value: Any, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or not value
        or len(value) > 256
        or any(
            type(item) is not str or not item or len(item) > 2_048
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValidationError(f"{label} is invalid")
    return tuple(value)


def _validate_titled_options(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value or len(value) > 256:
        raise ValidationError(f"{label} is invalid")
    selected: list[str] = []
    for option in value:
        if (
            type(option) is not dict
            or set(option) != {"const", "title"}
            or type(option.get("const")) is not str
            or not option["const"]
            or len(option["const"]) > 2_048
            or type(option.get("title")) is not str
            or not option["title"]
            or len(option["title"]) > 2_048
        ):
            raise ValidationError(f"{label} is invalid")
        selected.append(option["const"])
    if len(set(selected)) != len(selected):
        raise ValidationError(f"{label} contains duplicate values")
    return tuple(selected)


def validate_input_responses(
    parsed: ParsedInputRequests,
    responses: Any,
) -> dict[str, JsonValue]:
    if parsed.has_unsupported:
        raise ValidationError("MCP Sampling and Roots input requests are unsupported")
    if type(responses) is not dict or any(type(key) is not str for key in responses):
        raise ValidationError("MCP input responses must be an object")
    expected_ids = {item.request_id for item in parsed.public}
    if set(responses) != expected_ids:
        raise ValidationError("MCP input response identifiers do not match the request")
    remote: dict[str, JsonValue] = {}
    for public_request, remote_key in zip(parsed.public, parsed.remote_keys, strict=True):
        response = responses[public_request.request_id]
        if type(response) is not dict or set(response) - {"action", "content"}:
            raise ValidationError("MCP elicitation response shape is invalid")
        action = response.get("action")
        if action not in {"accept", "decline", "cancel"}:
            raise ValidationError("MCP elicitation response action is invalid")
        content = response.get("content")
        if action != "accept":
            if "content" in response:
                raise ValidationError("declined MCP elicitation cannot include content")
            remote[remote_key] = {"action": action}
            continue
        if public_request.mode == "url":
            if "content" in response:
                raise ValidationError("URL MCP elicitation cannot include content")
            remote[remote_key] = {"action": "accept"}
            continue
        validated = _validate_form_content(content, public_request.schema)
        remote[remote_key] = {"action": "accept", "content": validated}
    strict_json_value(remote, label="MCP input responses")
    return remote


def _validate_form_content(value: Any, schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValidationError("accepted MCP form content must be an object")
    properties = schema.get("properties")
    required = schema.get("required", [])
    if type(properties) is not dict or type(required) is not list:
        raise ValidationError("MCP elicitation schema binding is invalid")
    if set(value) - set(properties) or any(item not in value for item in required):
        raise ValidationError("MCP form content does not match requested properties")
    selected: dict[str, JsonValue] = {}
    for name, item in value.items():
        spec = properties[name]
        if type(spec) is not dict:
            raise ValidationError("MCP elicitation schema binding is invalid")
        _validate_primitive_value(item, spec)
        selected[name] = item
    strict_json_value(selected, label="MCP form content")
    return selected


def _validate_primitive_value(value: Any, spec: Mapping[str, Any]) -> None:
    selected_type = spec.get("type")
    if selected_type == "string":
        _validate_string_value(value, spec)
        return
    if selected_type == "boolean":
        if type(value) is not bool:
            raise ValidationError("MCP form boolean value is invalid")
        return
    if selected_type == "integer":
        _validate_number_value(value, spec, integer=True)
    elif selected_type == "number":
        _validate_number_value(value, spec, integer=False)
    elif selected_type == "array":
        _validate_array_value(value, spec)
        return
    else:
        raise ValidationError("MCP form value type is unsupported")


def _validate_string_value(value: Any, spec: Mapping[str, Any]) -> None:
    if type(value) is not str:
        raise ValidationError("MCP form string value is invalid")
    minimum = spec.get("minLength")
    maximum = spec.get("maxLength")
    if type(minimum) is int and len(value) < minimum:
        raise ValidationError("MCP form string is too short")
    if type(maximum) is int and len(value) > maximum:
        raise ValidationError("MCP form string is too long")
    options = spec.get("enum")
    if type(options) is list and value not in options:
        raise ValidationError("MCP form enum value is invalid")
    one_of = spec.get("oneOf")
    if type(one_of) is list and value not in {
        item.get("const") for item in one_of if type(item) is dict
    }:
        raise ValidationError("MCP form titled enum value is invalid")
    _validate_string_format(value, spec.get("format"))


def _validate_string_format(value: str, selected_format: Any) -> None:
    if selected_format == "email" and _EMAIL.fullmatch(value) is None:
        raise ValidationError("MCP form email value is invalid")
    if selected_format == "uri" and not urlsplit(value).scheme:
        raise ValidationError("MCP form URI value is invalid")
    if selected_format not in {"date", "date-time"}:
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("MCP form date value is invalid") from exc


def _validate_number_value(
    value: Any,
    spec: Mapping[str, Any],
    *,
    integer: bool,
) -> None:
    if integer:
        if type(value) is not int:
            raise ValidationError("MCP form integer value is invalid")
    elif type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValidationError("MCP form number value is invalid")
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    if type(minimum) in {int, float} and value < minimum:
        raise ValidationError("MCP form number is below minimum")
    if type(maximum) in {int, float} and value > maximum:
        raise ValidationError("MCP form number is above maximum")


def _validate_array_value(value: Any, spec: Mapping[str, Any]) -> None:
    items = spec.get("items")
    if type(items) is dict and type(items.get("enum")) is list:
        options = items["enum"]
    elif type(items) is dict and type(items.get("anyOf")) is list:
        options = [
            item.get("const") for item in items["anyOf"] if type(item) is dict
        ]
    else:
        options = None
    if (
        type(value) is not list
        or any(type(item) is not str for item in value)
        or type(options) is not list
        or any(item not in options for item in value)
    ):
        raise ValidationError("MCP form multi-select value is invalid")
    minimum_items = spec.get("minItems")
    maximum_items = spec.get("maxItems")
    if type(minimum_items) is int and len(value) < minimum_items:
        raise ValidationError("MCP form multi-select has too few values")
    if type(maximum_items) is int and len(value) > maximum_items:
        raise ValidationError("MCP form multi-select has too many values")
