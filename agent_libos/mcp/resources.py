"""Safe projections for MCP Resources and Resource Templates.

Remote resource identifiers are transport selectors, not local URLs.  This
module therefore never opens a URI.  It converts provider-controlled content
to bounded public values, replaces binary payloads with Host artifact
receipts, and turns ResourceLink targets into inert one-way handles.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, Protocol

from agent_libos.mcp.app_policy import (
    is_mcp_app_metadata_key,
    is_mcp_app_mime,
    reject_mcp_app_selector,
    reject_mcp_app_text,
)
from agent_libos.mcp.types import (
    JsonValue,
    McpAnnotations,
    McpArtifactReceipt,
    McpBlobContent,
    McpCacheHint,
    McpCacheScope,
    McpContentBlock,
    McpIcon,
    McpResource,
    McpResourceContents,
    McpResourceLinkContent,
    McpResourceTemplate,
    McpTextContent,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.redaction import redact_sensitive_text
from agent_libos.utils.serde import dumps, to_jsonable


_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000


class McpArtifactWriter(Protocol):
    """Host sink for a binary MCP payload.

    Implementations own retention and access control.  The public MCP result
    receives only the returned receipt; Agent libOS never places ``data`` in
    a model, audit event, TaskRun, or RuntimeStore row.
    """

    def write_mcp_artifact(
        self,
        data: bytes,
        *,
        server_id: str,
        logical_id: str,
        mime_type: str | None,
    ) -> McpArtifactReceipt: ...


def sanitize_provider_json(
    value: Any,
    *,
    sensitive_values: Iterable[str] = (),
    depth: int = 0,
    _nodes: list[int] | None = None,
) -> JsonValue:
    """Return strict, recursively redacted JSON and discard MCP Apps metadata."""

    if _nodes is None:
        _nodes = [0]
    _nodes[0] += 1
    if _nodes[0] > _MAX_JSON_NODES:
        raise ValidationError("MCP provider JSON exceeds maximum node count")
    if depth > _MAX_JSON_DEPTH:
        raise ValidationError("MCP provider JSON exceeds maximum depth")
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError("MCP provider JSON contains a non-finite number")
        return value
    if type(value) is str:
        reject_mcp_app_text(value)
        return redact_sensitive_text(value, sensitive_values=sensitive_values)
    if type(value) in {list, tuple}:
        return [
            sanitize_provider_json(
                item,
                sensitive_values=sensitive_values,
                depth=depth + 1,
                _nodes=_nodes,
            )
            for item in value
        ]
    if isinstance(value, Mapping):
        selected: dict[str, JsonValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValidationError("MCP provider JSON keys must be strings")
            if is_mcp_app_metadata_key(key):
                continue
            reject_mcp_app_text(key)
            public_key = redact_sensitive_text(key, sensitive_values=sensitive_values)
            if public_key in selected:
                raise ValidationError("MCP provider JSON keys collide after redaction")
            selected[public_key] = sanitize_provider_json(
                item,
                sensitive_values=sensitive_values,
                depth=depth + 1,
                _nodes=_nodes,
            )
        return selected
    raise ValidationError("MCP provider value must be strict JSON")


def redact_public_dataclass(value: Any, *, sensitive_values: Iterable[str]) -> Any:
    """Recursively redact already-projected provider dataclasses.

    Custom providers are not trusted to call the SDK adapter, so the manager
    applies this second projection before returning or persisting a value.
    """

    selected_secrets = tuple(sensitive_values)
    if type(value) is str:
        reject_mcp_app_text(value)
        return redact_sensitive_text(value, sensitive_values=selected_secrets)
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError("MCP provider result contains a non-finite number")
        return value
    if type(value) is tuple:
        return tuple(
            redact_public_dataclass(item, sensitive_values=selected_secrets)
            for item in value
        )
    if type(value) is list:
        return [
            redact_public_dataclass(item, sensitive_values=selected_secrets)
            for item in value
        ]
    if isinstance(value, Mapping):
        return sanitize_provider_json(value, sensitive_values=selected_secrets)
    if is_dataclass(value):
        updates = {
            field.name: redact_public_dataclass(
                getattr(value, field.name), sensitive_values=selected_secrets
            )
            for field in fields(value)
        }
        return replace(value, **updates)
    # StrEnum and other string enums are already constrained public values.
    if isinstance(value, str):
        reject_mcp_app_text(str(value))
        return type(value)(
            redact_sensitive_text(str(value), sensitive_values=selected_secrets)
        )
    raise ValidationError("MCP provider returned an unsupported public value")


def bounded_public_size(value: Any, *, maximum: int, label: str) -> int:
    """Measure a public projection only after a bounded structural walk."""

    if type(maximum) is not int or maximum <= 0:
        raise ValidationError("MCP response bound must be a positive integer")
    _bounded_walk(value)
    try:
        encoded = dumps(to_jsonable(value)).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError(f"{label} is not serializable strict JSON") from exc
    if len(encoded) > maximum:
        raise ValidationError(f"{label} exceeds max_response_bytes={maximum}")
    return len(encoded)


def _bounded_walk(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValidationError("MCP provider result exceeds maximum node count")
        if depth > _MAX_JSON_DEPTH:
            raise ValidationError("MCP provider result exceeds maximum depth")
        if current is None or type(current) in {bool, int, str}:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise ValidationError("MCP provider result contains a non-finite number")
            continue
        if isinstance(current, Mapping):
            for key, item in current.items():
                if type(key) is not str:
                    raise ValidationError("MCP provider result keys must be strings")
                stack.append((item, depth + 1))
            continue
        if type(current) in {tuple, list}:
            stack.extend((item, depth + 1) for item in current)
            continue
        if is_dataclass(current):
            stack.extend(
                (getattr(current, field.name), depth + 1)
                for field in fields(current)
            )
            continue
        if isinstance(current, str):
            continue
        raise ValidationError("MCP provider returned an unsupported public value")


def cache_hint_from_sdk(
    value: Any,
    *,
    maximum_ttl_ms: int,
) -> McpCacheHint | None:
    ttl_ms = getattr(value, "ttl_ms", getattr(value, "ttlMs", 0))
    if type(ttl_ms) is not int or ttl_ms < 0:
        raise ValidationError("MCP cache ttlMs must be a non-negative integer")
    if ttl_ms == 0:
        return None
    scope_value = getattr(value, "cache_scope", getattr(value, "cacheScope", "private"))
    try:
        scope = McpCacheScope(str(scope_value))
    except ValueError as exc:
        raise ValidationError("MCP cacheScope is invalid") from exc
    return McpCacheHint(ttl_ms=min(ttl_ms, maximum_ttl_ms), scope=scope)


def sdk_resource(
    item: Any,
    *,
    sensitive_values: Iterable[str] = (),
) -> McpResource:
    remote_uri = str(getattr(item, "uri", ""))
    reject_mcp_app_selector(remote_uri)
    mime_type = getattr(item, "mime_type", getattr(item, "mimeType", None))
    if is_mcp_app_mime(mime_type):
        raise ValidationError("MCP Apps HTML resources are unsupported")
    return McpResource(
        resource_id=remote_uri,
        name=redact_sensitive_text(
            str(getattr(item, "name", "")), sensitive_values=sensitive_values
        ),
        title=_optional_redacted(getattr(item, "title", None), sensitive_values),
        description=_optional_redacted(
            getattr(item, "description", None), sensitive_values
        ),
        mime_type=_optional_redacted(mime_type, sensitive_values),
        size=_optional_nonnegative_int(getattr(item, "size", None), "resource size"),
        # Remote icon URLs are never model/GUI fetch instructions.
        icons=(),
        annotations=sdk_annotations(getattr(item, "annotations", None), sensitive_values),
        metadata=_metadata(getattr(item, "meta", None), sensitive_values),
    )


def sdk_resource_template(
    item: Any,
    *,
    sensitive_values: Iterable[str] = (),
) -> McpResourceTemplate:
    remote_template = str(
        getattr(item, "uri_template", getattr(item, "uriTemplate", ""))
    )
    reject_mcp_app_selector(remote_template, label="resource template selector")
    mime_type = getattr(item, "mime_type", getattr(item, "mimeType", None))
    if is_mcp_app_mime(mime_type):
        raise ValidationError("MCP Apps HTML resource templates are unsupported")
    return McpResourceTemplate(
        template_id=remote_template,
        name=redact_sensitive_text(
            str(getattr(item, "name", "")), sensitive_values=sensitive_values
        ),
        title=_optional_redacted(getattr(item, "title", None), sensitive_values),
        description=_optional_redacted(
            getattr(item, "description", None), sensitive_values
        ),
        mime_type=_optional_redacted(mime_type, sensitive_values),
        icons=(),
        annotations=sdk_annotations(getattr(item, "annotations", None), sensitive_values),
        metadata=_metadata(getattr(item, "meta", None), sensitive_values),
    )


def sdk_annotations(
    value: Any,
    sensitive_values: Iterable[str] = (),
) -> McpAnnotations | None:
    if value is None:
        return None
    audience = getattr(value, "audience", None) or ()
    if any(item not in {"user", "assistant"} for item in audience):
        raise ValidationError("MCP annotations audience is invalid")
    priority = getattr(value, "priority", None)
    if priority is not None and (
        type(priority) not in {int, float}
        or isinstance(priority, bool)
        or not math.isfinite(float(priority))
        or not 0 <= float(priority) <= 1
    ):
        raise ValidationError("MCP annotations priority is invalid")
    return McpAnnotations(
        audience=tuple(audience),
        priority=(float(priority) if priority is not None else None),
        last_modified=_optional_redacted(
            getattr(value, "last_modified", getattr(value, "lastModified", None)),
            sensitive_values,
        ),
    )


def sdk_content_block(
    value: Any,
    *,
    server_id: str,
    logical_id: str,
    artifact_writer: McpArtifactWriter | None,
    sensitive_values: Iterable[str] = (),
    _depth: int = 0,
) -> McpContentBlock:
    """Project one SDK content value without dereferencing remote selectors."""

    selected_secrets = tuple(sensitive_values)
    validate_sdk_content_block(
        value,
        artifact_writer=artifact_writer,
        sensitive_values=selected_secrets,
        depth=_depth,
    )
    type_name = type(value).__name__
    if type_name in {"TextContent", "TextResourceContents"}:
        mime_type = getattr(value, "mime_type", getattr(value, "mimeType", None))
        if is_mcp_app_mime(mime_type):
            raise ValidationError("MCP Apps HTML content is unsupported")
        return McpTextContent(
            text=redact_sensitive_text(
                str(getattr(value, "text", "")), sensitive_values=selected_secrets
            ),
            annotations=sdk_annotations(
                getattr(value, "annotations", None), selected_secrets
            ),
            metadata=_metadata(getattr(value, "meta", None), selected_secrets),
        )
    if type_name in {"BlobResourceContents", "ImageContent", "AudioContent"}:
        encoded = getattr(value, "blob", getattr(value, "data", None))
        if type(encoded) is not str:
            raise ValidationError("MCP binary content must be base64 text")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValidationError("MCP binary content is invalid base64") from exc
        _reject_reflected_secret_bytes(data, selected_secrets)
        if artifact_writer is None:
            raise ValidationError(
                "MCP binary content requires a Host artifact writer"
            )
        mime_type = getattr(value, "mime_type", getattr(value, "mimeType", None))
        if is_mcp_app_mime(mime_type):
            raise ValidationError("MCP Apps HTML content is unsupported")
        receipt = artifact_writer.write_mcp_artifact(
            data,
            server_id=server_id,
            logical_id=logical_id,
            mime_type=mime_type,
        )
        _validate_artifact_receipt(receipt, data)
        return McpBlobContent(
            artifact=receipt,
            annotations=sdk_annotations(
                getattr(value, "annotations", None), selected_secrets
            ),
            metadata=_metadata(getattr(value, "meta", None), selected_secrets),
        )
    if type_name == "ResourceLink":
        uri = str(getattr(value, "uri", ""))
        reject_mcp_app_selector(uri, label="ResourceLink selector")
        mime_type = getattr(value, "mime_type", getattr(value, "mimeType", None))
        if is_mcp_app_mime(mime_type):
            raise ValidationError("MCP Apps ResourceLink content is unsupported")
        return McpResourceLinkContent(
            resource_handle=inert_resource_handle(server_id, uri),
            name=redact_sensitive_text(
                str(getattr(value, "name", "")), sensitive_values=selected_secrets
            ),
            title=_optional_redacted(getattr(value, "title", None), selected_secrets),
            description=_optional_redacted(
                getattr(value, "description", None), selected_secrets
            ),
            mime_type=_optional_redacted(mime_type, selected_secrets),
            annotations=sdk_annotations(
                getattr(value, "annotations", None), selected_secrets
            ),
            metadata=_metadata(getattr(value, "meta", None), selected_secrets),
        )
    if type_name == "EmbeddedResource":
        resource = getattr(value, "resource", None)
        if resource is None:
            raise ValidationError("MCP embedded resource is missing content")
        # The server already supplied this payload.  We project it once but
        # never follow its URI or any links contained in it.
        return sdk_content_block(
            resource,
            server_id=server_id,
            logical_id=logical_id,
            artifact_writer=artifact_writer,
            sensitive_values=selected_secrets,
            _depth=_depth + 1,
        )
    raise ValidationError(f"unsupported MCP content block: {type_name}")


def sdk_resource_contents(
    value: Any,
    *,
    server_id: str,
    logical_id: str,
    artifact_writer: McpArtifactWriter | None,
    sensitive_values: Iterable[str] = (),
    maximum_content_blocks: int = 256,
) -> McpResourceContents:
    contents = getattr(value, "contents", None)
    if type(contents) is not list:
        raise ValidationError("MCP resources/read contents must be a list")
    _require_content_block_count(
        contents,
        maximum=maximum_content_blocks,
        label="resources/read",
    )
    selected_secrets = tuple(sensitive_values)
    for item in contents:
        validate_sdk_content_block(
            item,
            artifact_writer=artifact_writer,
            sensitive_values=selected_secrets,
        )
    return McpResourceContents(
        resource_id=logical_id,
        contents=tuple(
            sdk_content_block(
                item,
                server_id=server_id,
                logical_id=logical_id,
                artifact_writer=artifact_writer,
                sensitive_values=selected_secrets,
            )
            for item in contents
        ),
    )


def validate_sdk_content_block(
    value: Any,
    *,
    artifact_writer: McpArtifactWriter | None,
    sensitive_values: Iterable[str] = (),
    depth: int = 0,
) -> None:
    """Validate one SDK content tree without performing artifact writes."""

    if type(depth) is not int or depth < 0 or depth > _MAX_JSON_DEPTH:
        raise ValidationError("MCP embedded content exceeds maximum depth")
    selected_secrets = tuple(sensitive_values)
    type_name = type(value).__name__
    if type_name in {"TextContent", "TextResourceContents"}:
        text = getattr(value, "text", None)
        if type(text) is not str:
            raise ValidationError("MCP text content must contain text")
        reject_mcp_app_text(text)
        mime_type = getattr(value, "mime_type", getattr(value, "mimeType", None))
        if is_mcp_app_mime(mime_type):
            raise ValidationError("MCP Apps HTML content is unsupported")
    elif type_name in {"BlobResourceContents", "ImageContent", "AudioContent"}:
        encoded = getattr(value, "blob", getattr(value, "data", None))
        if type(encoded) is not str:
            raise ValidationError("MCP binary content must be base64 text")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValidationError("MCP binary content is invalid base64") from exc
        _reject_reflected_secret_bytes(data, selected_secrets)
        if artifact_writer is None:
            raise ValidationError("MCP binary content requires a Host artifact writer")
        mime_type = getattr(value, "mime_type", getattr(value, "mimeType", None))
        if is_mcp_app_mime(mime_type):
            raise ValidationError("MCP Apps HTML content is unsupported")
    elif type_name == "ResourceLink":
        uri = getattr(value, "uri", None)
        if type(uri) is not str or not uri:
            raise ValidationError("MCP ResourceLink selector must be non-empty text")
        reject_mcp_app_selector(uri, label="ResourceLink selector")
        name = getattr(value, "name", None)
        if type(name) is not str or not name:
            raise ValidationError("MCP ResourceLink name must be non-empty text")
        mime_type = getattr(value, "mime_type", getattr(value, "mimeType", None))
        if is_mcp_app_mime(mime_type):
            raise ValidationError("MCP Apps ResourceLink content is unsupported")
    elif type_name == "EmbeddedResource":
        resource = getattr(value, "resource", None)
        if resource is None:
            raise ValidationError("MCP embedded resource is missing content")
        validate_sdk_content_block(
            resource,
            artifact_writer=artifact_writer,
            sensitive_values=selected_secrets,
            depth=depth + 1,
        )
        return
    else:
        raise ValidationError(f"unsupported MCP content block: {type_name}")
    sdk_annotations(getattr(value, "annotations", None), selected_secrets)
    _metadata(getattr(value, "meta", None), selected_secrets)


def sanitize_resource_contents(
    value: McpResourceContents,
    *,
    server_id: str,
    logical_id: str,
    sensitive_values: Iterable[str],
    maximum_content_blocks: int = 256,
) -> McpResourceContents:
    """Validate a custom-provider public result and remove selector leakage."""

    _require_content_block_count(
        value.contents,
        maximum=maximum_content_blocks,
        label="resource",
    )
    selected: list[McpContentBlock] = []
    for content in value.contents:
        if isinstance(content, McpTextContent):
            selected.append(
                replace(
                    content,
                    text=redact_sensitive_text(
                        content.text, sensitive_values=sensitive_values
                    ),
                    annotations=_sanitize_public_annotations(
                        content.annotations, sensitive_values
                    ),
                    metadata=_metadata(content.metadata, sensitive_values),
                )
            )
        elif isinstance(content, McpBlobContent):
            if content.artifact is None:
                raise ValidationError("MCP blob projection is missing artifact receipt")
            _validate_public_artifact_receipt(content.artifact)
            public_artifact_id = redact_sensitive_text(
                content.artifact.artifact_id, sensitive_values=sensitive_values
            )
            if public_artifact_id != content.artifact.artifact_id:
                raise ValidationError("MCP artifact id reflected an operation secret")
            selected.append(
                replace(
                    content,
                    artifact=replace(
                        content.artifact,
                        mime_type=_optional_redacted(
                            content.artifact.mime_type, sensitive_values
                        ),
                    ),
                    annotations=_sanitize_public_annotations(
                        content.annotations, sensitive_values
                    ),
                    metadata=_metadata(content.metadata, sensitive_values),
                )
            )
        elif isinstance(content, McpResourceLinkContent):
            reject_mcp_app_selector(
                content.resource_handle, label="ResourceLink selector"
            )
            if is_mcp_app_mime(content.mime_type):
                raise ValidationError("MCP Apps ResourceLink content is unsupported")
            selected.append(
                replace(
                    content,
                    resource_handle=inert_resource_handle(
                        server_id, content.resource_handle
                    ),
                    name=redact_sensitive_text(
                        content.name, sensitive_values=sensitive_values
                    ),
                    title=_optional_redacted(content.title, sensitive_values),
                    description=_optional_redacted(
                        content.description, sensitive_values
                    ),
                    mime_type=_optional_redacted(content.mime_type, sensitive_values),
                    annotations=_sanitize_public_annotations(
                        content.annotations, sensitive_values
                    ),
                    metadata=_metadata(content.metadata, sensitive_values),
                )
            )
        else:  # pragma: no cover - closed public union, defensive for custom SPI
            raise ValidationError("MCP resource contains an unsupported content block")
    return McpResourceContents(resource_id=logical_id, contents=tuple(selected))


def inert_resource_handle(server_id: str, selector: str) -> str:
    digest = hashlib.sha256(f"{server_id}\0{selector}".encode("utf-8")).hexdigest()
    return f"mcp-link:{digest[:32]}"


def _sanitize_public_annotations(
    value: McpAnnotations | None,
    sensitive_values: Iterable[str],
) -> McpAnnotations | None:
    if value is None:
        return None
    if any(item not in {"user", "assistant"} for item in value.audience):
        raise ValidationError("MCP annotations audience is invalid")
    if value.priority is not None and (
        type(value.priority) not in {int, float}
        or isinstance(value.priority, bool)
        or not math.isfinite(float(value.priority))
        or not 0 <= float(value.priority) <= 1
    ):
        raise ValidationError("MCP annotations priority is invalid")
    return replace(
        value,
        last_modified=_optional_redacted(value.last_modified, sensitive_values),
    )


def _metadata(value: Any, sensitive_values: Iterable[str]) -> dict[str, JsonValue]:
    if value is None:
        return {}
    selected = sanitize_provider_json(value, sensitive_values=sensitive_values)
    if not isinstance(selected, dict):
        raise ValidationError("MCP provider metadata must be an object")
    return selected


def _optional_redacted(value: Any, sensitive_values: Iterable[str]) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValidationError("MCP provider text field must be a string")
    return redact_sensitive_text(value, sensitive_values=sensitive_values)


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValidationError(f"MCP {label} must be a non-negative integer")
    return value


def _require_content_block_count(
    contents: Any,
    *,
    maximum: int,
    label: str,
) -> None:
    if type(maximum) is not int or maximum <= 0:
        raise ValidationError("MCP maximum content block count is invalid")
    if type(contents) not in {list, tuple}:
        raise ValidationError(f"MCP {label} content blocks must be a list")
    if len(contents) > maximum:
        raise ValidationError(f"MCP {label} exceeded maximum content block count")


def _reject_reflected_secret_bytes(data: bytes, sensitive_values: Iterable[str]) -> None:
    for value in sensitive_values:
        if type(value) is str and value and value.encode("utf-8") in data:
            raise ValidationError("MCP binary content reflected an operation secret")


def _validate_artifact_receipt(receipt: Any, data: bytes) -> None:
    if not isinstance(receipt, McpArtifactReceipt):
        raise ValidationError("MCP artifact writer returned an invalid receipt")
    _validate_public_artifact_receipt(receipt)
    expected = hashlib.sha256(data).hexdigest()
    if receipt.byte_length != len(data) or receipt.sha256 != expected:
        raise ValidationError("MCP artifact receipt does not match binary content")


def _validate_public_artifact_receipt(receipt: McpArtifactReceipt) -> None:
    if type(receipt.artifact_id) is not str or not receipt.artifact_id:
        raise ValidationError("MCP artifact receipt id is invalid")
    if type(receipt.byte_length) is not int or receipt.byte_length < 0:
        raise ValidationError("MCP artifact receipt byte length is invalid")
    if (
        type(receipt.sha256) is not str
        or len(receipt.sha256) != 64
        or any(character not in "0123456789abcdef" for character in receipt.sha256)
    ):
        raise ValidationError("MCP artifact receipt SHA-256 is invalid")
    if is_mcp_app_mime(receipt.mime_type):
        raise ValidationError("MCP Apps artifact MIME is unsupported")


__all__ = [
    "McpArtifactWriter",
    "bounded_public_size",
    "cache_hint_from_sdk",
    "inert_resource_handle",
    "is_mcp_app_mime",
    "redact_public_dataclass",
    "reject_mcp_app_selector",
    "sanitize_provider_json",
    "sanitize_resource_contents",
    "sdk_annotations",
    "sdk_content_block",
    "sdk_resource",
    "sdk_resource_contents",
    "sdk_resource_template",
    "validate_sdk_content_block",
]
