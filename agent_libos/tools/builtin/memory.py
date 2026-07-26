from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any

from pydantic import BaseModel, Field, WithJsonSchema, field_validator, model_validator

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.memory.data_labels import propagate_object_labels
from agent_libos.models import ObjectMetadata, ObjectType, Provenance, ViewMode
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.tools.observability import json_size_bytes

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_CREATE_MEMORY_METADATA_FIELDS = frozenset(
    {
        "title",
        "summary",
        "tags",
        "mime_type",
        "sensitivity",
        "retention_policy",
        "trust_level",
        "integrity",
        "tenant",
        "principal",
    }
)

# The broker persists a wrapper around the model-facing tool data.  Account for
# fields whose rendered width can vary slightly between this preflight and the
# broker's final serialization (notably duration_ms) without coupling this tool
# to broker internals.
_TOOL_RESULT_ENVELOPE_SAFETY_BYTES = 512

_DIRECT_JSON_SCHEMA = {
    "anyOf": [
        {"type": "object", "additionalProperties": True},
        {"type": "array", "items": {}},
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "null"},
    ]
}
DirectJsonValue = Annotated[Any, WithJsonSchema(_DIRECT_JSON_SCHEMA)]


def _empty_optional_text_is_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


class CreateMemoryObjectArgs(BaseModel):
    name: str | None = Field(default=None, description="Optional namespace-local object name.")
    namespace: str | None = Field(default=None, description="Object Memory namespace. Defaults to this process namespace.")
    type: str = Field(description="Agent libOS object type, for example summary, plan, observation, or artifact.")
    payload: DirectJsonValue = Field(
        description=(
            "Direct JSON value to store. JSON strings are stored literally; pass an "
            "object/array value, not a JSON-encoded string, when a container is intended."
        )
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_oids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional confirmed Object Memory OIDs returned by memory tools. "
            "Do not pass filesystem or generic tool-result object IDs."
        ),
    )
    immutable: bool = True

    @field_validator("namespace", mode="before")
    @classmethod
    def normalize_empty_namespace(cls, value: Any) -> Any:
        return _empty_optional_text_is_none(value)


class CreateMemoryObjectOutput(BaseModel):
    oid: str
    namespace: str
    name: str
    type: str


class ReadMemoryObjectArgs(BaseModel):
    name: str = Field(
        description=(
            "Exact namespace-local name returned by list_memory_namespace. "
            "There is no bare `goal` alias, and a runtime-only goal object may "
            "be unavailable after reopen; use cumulative process_exit review then."
        )
    )
    namespace: str | None = Field(default=None, description="Object Memory namespace. Defaults to this process namespace.")
    max_payload_chars: int = Field(
        default=_TOOL_DEFAULTS.memory_payload_chars,
        ge=1,
        le=_TOOL_DEFAULTS.memory_payload_hard_limit_chars,
        description=(
            "Maximum canonical JSON characters returned in one payload page. "
            "The runtime also applies a UTF-8 byte and ToolResult-envelope budget. "
            "Use next_cursor to continue a truncated read."
        ),
    )
    json_pointer: str = Field(
        default="",
        description=(
            "Optional RFC 6901 JSON Pointer selecting a payload subtree. "
            "The empty string selects the whole payload."
        ),
    )
    cursor: int = Field(
        default=0,
        ge=0,
        description=(
            "UTF-8 byte cursor returned as next_cursor by the preceding read. "
            "Cursors are relative to the selected subtree's canonical JSON."
        ),
    )
    expected_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Optional sha256 from a preceding page. The read fails if the selected "
            "payload changed before continuation."
        ),
    )

    @field_validator("namespace", mode="before")
    @classmethod
    def normalize_empty_namespace(cls, value: Any) -> Any:
        return _empty_optional_text_is_none(value)

    @model_validator(mode="after")
    def continuation_requires_hash(self) -> ReadMemoryObjectArgs:
        if self.cursor > 0 and self.expected_sha256 is None:
            raise ValueError("expected_sha256 is required when cursor is greater than zero")
        return self


class ReadMemoryObjectOutput(BaseModel):
    oid: str
    namespace: str
    name: str
    type: str
    version: int
    json_pointer: str
    payload_type: str
    shape: dict[str, int | str]
    serialized_bytes: int
    sha256: str
    representation: str
    payload: Any | None = Field(
        default=None,
        description="Complete selected JSON value when it fits in one page; otherwise null.",
    )
    preview: str | None = Field(
        default=None,
        description="Exact canonical JSON page when representation is canonical_json_page.",
        exclude_if=lambda value: value is None,
    )
    preview_encoding: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    page_offset_bytes: int
    page_bytes: int
    truncated: bool
    omitted_bytes: int
    next_cursor: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class AppendMemoryObjectArgs(BaseModel):
    name: str = Field(description="Namespace-local mutable Object Memory name to append to.")
    namespace: str | None = Field(default=None, description="Object Memory namespace. Defaults to this process namespace.")
    entry: DirectJsonValue = Field(
        description=(
            "Direct JSON entry to append. JSON strings are appended literally; pass an "
            "object/array value, not a JSON-encoded string, when a container is intended."
        )
    )
    list_field: str = Field(
        default="entries",
        description="Payload list field to append into when the object payload is a JSON object.",
    )

    @field_validator("namespace", mode="before")
    @classmethod
    def normalize_empty_namespace(cls, value: Any) -> Any:
        return _empty_optional_text_is_none(value)


class AppendMemoryObjectOutput(BaseModel):
    oid: str
    namespace: str
    name: str
    version: int
    appended: bool
    list_field: str | None = None
    length: int


class CreateMemoryNamespaceArgs(BaseModel):
    namespace: str = Field(description="Namespace path to create, for example project/research or child-results.")
    parent_namespace: str | None = Field(
        default=None,
        description="Parent namespace. Defaults to the path parent; top-level namespaces have no parent.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parent_namespace", mode="before")
    @classmethod
    def normalize_empty_parent_namespace(cls, value: Any) -> Any:
        return _empty_optional_text_is_none(value)


class CreateMemoryNamespaceOutput(BaseModel):
    namespace: str
    parent_namespace: str | None
    created: bool


class ListMemoryNamespaceArgs(BaseModel):
    namespace: str | None = Field(
        default=None,
        description=(
            "Exact namespace to list; null defaults to this process namespace. "
            "Do not broaden to the parent `process` namespace as a fallback."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of visible namespace entries to return. Defaults to the runtime memory query limit.",
    )

    @field_validator("namespace", mode="before")
    @classmethod
    def normalize_empty_namespace(cls, value: Any) -> Any:
        return _empty_optional_text_is_none(value)


class MemoryNamespaceObjectEntry(BaseModel):
    oid: str
    namespace: str
    name: str
    type: str
    version: int


class MemoryNamespaceEntry(BaseModel):
    namespace: str
    parent_namespace: str | None


class ListMemoryNamespaceOutput(BaseModel):
    namespace: str
    objects: list[MemoryNamespaceObjectEntry]
    namespaces: list[MemoryNamespaceEntry]


class CreateMemoryObjectTool(SyncAgentTool[CreateMemoryObjectArgs]):
    name = "create_memory_object"
    description = (
        "Create a typed object in Agent libOS Object Memory and attach it to this process MemoryView. "
        "This is a Skills/Tools Layer wrapper over the memory manager."
    )
    args_schema = CreateMemoryObjectArgs
    output_schema = CreateMemoryObjectOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object"]

    def run(self, args: CreateMemoryObjectArgs, ctx: ToolContext) -> CreateMemoryObjectOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        if args.metadata.get("declassification_authority") not in {None, ""}:
            raise ToolExecutionError(
                "LLM-created objects cannot assert declassification authority.",
                code=ToolErrorCode.PERMISSION_DENIED,
            )
        unknown_metadata_fields = sorted(
            set(args.metadata) - _CREATE_MEMORY_METADATA_FIELDS
        )
        if unknown_metadata_fields:
            raise ToolExecutionError(
                f"Unsupported object metadata fields: {unknown_metadata_fields}",
                code=ToolErrorCode.VALIDATION_ERROR,
            )
        if args.metadata.get("trust_level", "unknown") not in {"untrusted", "unknown"}:
            raise ToolExecutionError(
                "LLM-created objects cannot elevate trust_level.",
                code=ToolErrorCode.PERMISSION_DENIED,
            )
        if args.metadata.get("integrity", "unknown") not in {"untrusted", "unknown"}:
            raise ToolExecutionError(
                "LLM-created objects cannot elevate integrity.",
                code=ToolErrorCode.PERMISSION_DENIED,
            )
        try:
            metadata = ObjectMetadata(
                title=args.metadata.get("title"),
                summary=args.metadata.get("summary"),
                tags=args.metadata.get("tags", []),
                mime_type=args.metadata.get("mime_type"),
                sensitivity=args.metadata.get(
                    "sensitivity",
                    runtime.config.memory.metadata_sensitivity,
                ),
                retention_policy=args.metadata.get(
                    "retention_policy",
                    runtime.config.memory.metadata_retention_policy,
                ),
                trust_level=args.metadata.get("trust_level", "unknown"),
                integrity=args.metadata.get("integrity", "unknown"),
                origin="llm",
                tenant=args.metadata.get("tenant"),
                principal=args.metadata.get("principal"),
                declassification_authority=None,
            )
        except ValueError as exc:
            raise ToolExecutionError(str(exc), code=ToolErrorCode.VALIDATION_ERROR) from exc
        flow = runtime.data_flow.current_context()
        metadata = propagate_object_labels(
            metadata,
            [ObjectMetadata(**flow.labels.to_dict())],
        )
        flow_parent_oids, durable_source_refs = (
            runtime.data_flow.provenance_sources(flow)
        )
        parent_oids = list(
            dict.fromkeys([*args.parent_oids, *flow_parent_oids])
        )
        # Preserve Object Memory's global lock order (ownership, then store) and
        # keep Object/capability publication atomic with MemoryView attachment.
        # The nested create/view transactions use savepoints under this outer
        # transaction, so a view revision/update failure cannot leave an
        # unreported Object behind.
        with runtime.memory.ownership_locked(), runtime.store.transaction(
            include_object_payloads=True
        ):
            handle = runtime.memory.create_object(
                pid=ctx.pid,
                object_type=ObjectType(args.type),
                payload=args.payload,
                metadata=metadata,
                provenance=Provenance(
                    created_from_action="llm.create_memory_object",
                    parent_oids=parent_oids,
                    source_refs=list(durable_source_refs),
                ),
                immutable=args.immutable,
                name=args.name,
                namespace=args.namespace,
            )
            obj = runtime.memory.get_object(ctx.pid, handle)
            process = runtime.process.get(ctx.pid)
            if process.memory_view is None:
                process.memory_view = runtime.memory.create_view(ctx.pid, [handle], mode=ViewMode.READ_ONLY)
                runtime.store.patch_process(
                    ctx.pid,
                    {"memory_view": process.memory_view},
                    expected_revision=process.revision,
                )
            elif all(existing.oid != handle.oid for existing in process.memory_view.roots):
                runtime.store.append_process_memory_roots(ctx.pid, [handle])
            runtime.data_flow.observe_ingress(
                runtime.data_flow.context_from_trusted_source_oids([obj.oid])
            )
            return CreateMemoryObjectOutput(
                oid=handle.oid,
                namespace=obj.namespace,
                name=obj.name,
                type=args.type,
            )


class CreateMemoryNamespaceTool(SyncAgentTool[CreateMemoryNamespaceArgs]):
    name = "create_memory_namespace"
    description = (
        "Create an Object Memory namespace. Namespaces provide directory-like name scopes; "
        "object capabilities still control object reads and writes."
    )
    args_schema = CreateMemoryNamespaceArgs
    output_schema = CreateMemoryNamespaceOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object", "namespace"]

    def run(self, args: CreateMemoryNamespaceArgs, ctx: ToolContext) -> CreateMemoryNamespaceOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        namespace = runtime.memory.create_namespace(
            pid=ctx.pid,
            namespace=args.namespace,
            parent_namespace=args.parent_namespace,
            metadata=args.metadata,
        )
        return CreateMemoryNamespaceOutput(
            namespace=namespace.namespace,
            parent_namespace=namespace.parent_namespace,
            created=True,
        )


class ListMemoryNamespaceTool(SyncAgentTool[ListMemoryNamespaceArgs]):
    name = "list_memory_namespace"
    description = (
        "List process-visible objects and child namespaces within an Object Memory namespace. "
        "The list contains only objects the process can read."
    )
    args_schema = ListMemoryNamespaceArgs
    output_schema = ListMemoryNamespaceOutput
    policy = ToolPolicy(
        side_effects=False,
        idempotent=True,
        declared_permissions={"object.read"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object", "namespace", "read"]

    def run(self, args: ListMemoryNamespaceArgs, ctx: ToolContext) -> ListMemoryNamespaceOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        listing = runtime.memory.list_namespace(ctx.pid, args.namespace, limit=args.limit)
        objects = [
            MemoryNamespaceObjectEntry(
                oid=obj.oid,
                namespace=obj.namespace,
                name=obj.name,
                type=obj.type.value,
                version=obj.version,
            )
            for obj in listing["objects"]
        ]
        namespaces = [
            MemoryNamespaceEntry(namespace=namespace.namespace, parent_namespace=namespace.parent_namespace)
            for namespace in listing["namespaces"]
        ]
        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_trusted_source_oids(
                [obj.oid for obj in listing["objects"]]
            )
        )
        return ListMemoryNamespaceOutput(
            namespace=listing["namespace"],
            objects=objects,
            namespaces=namespaces,
        )


def _normalize_memory_json(value: Any) -> Any:
    """Return a deterministic, standards-compliant JSON value.

    Object Memory payloads are expected to be JSON values.  This defensive
    normalization keeps a legacy or directly-created non-JSON value from
    falling back to Python ``repr`` or producing invalid JSON such as NaN.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            label = "NaN"
        else:
            label = "Infinity" if value > 0 else "-Infinity"
        return {"_non_finite_number": label}
    if isinstance(value, dict):
        return {
            str(key): _normalize_memory_json(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_memory_json(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "_binary_bytes": len(raw),
            "_binary_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {
        "_unsupported_type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _canonical_memory_json(value: Any) -> tuple[Any, str, bytes]:
    normalized = _normalize_memory_json(value)
    rendered = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return normalized, rendered, rendered.encode("utf-8")


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _json_value_shape(value: Any) -> dict[str, int | str]:
    kind = _json_value_type(value)
    shape: dict[str, int | str] = {"kind": kind}
    if isinstance(value, dict):
        shape["field_count"] = len(value)
    elif isinstance(value, list):
        shape["item_count"] = len(value)
    elif isinstance(value, str):
        shape["character_count"] = len(value)
        shape["utf8_bytes"] = len(value.encode("utf-8"))
    return shape


def _decode_json_pointer_token(token: str) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(token):
        char = token[index]
        if char != "~":
            decoded.append(char)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise ToolExecutionError(
                "json_pointer contains an invalid RFC 6901 escape.",
                code=ToolErrorCode.VALIDATION_ERROR,
            )
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def _select_json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ToolExecutionError(
            "json_pointer must be empty or start with '/'.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )

    selected = payload
    for raw_token in pointer[1:].split("/"):
        token = _decode_json_pointer_token(raw_token)
        if isinstance(selected, dict):
            if token not in selected:
                raise ToolExecutionError(
                    "json_pointer does not exist in the selected object.",
                    code=ToolErrorCode.VALIDATION_ERROR,
                )
            selected = selected[token]
            continue
        if isinstance(selected, list):
            if not token.isdecimal() or (len(token) > 1 and token.startswith("0")):
                raise ToolExecutionError(
                    "json_pointer array tokens must be canonical non-negative indexes.",
                    code=ToolErrorCode.VALIDATION_ERROR,
                )
            item_index = int(token)
            if item_index >= len(selected):
                raise ToolExecutionError(
                    "json_pointer array index is out of range.",
                    code=ToolErrorCode.VALIDATION_ERROR,
                )
            selected = selected[item_index]
            continue
        raise ToolExecutionError(
            "json_pointer cannot traverse a scalar value.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    return selected


def _canonical_json_page(
    rendered_bytes: bytes,
    *,
    cursor: int,
    max_chars: int,
    max_bytes: int,
) -> tuple[str, int]:
    if cursor >= len(rendered_bytes):
        raise ToolExecutionError(
            "cursor is outside the selected payload; use next_cursor from the preceding page.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    try:
        remaining = rendered_bytes[cursor:].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolExecutionError(
            "cursor is not on a UTF-8 character boundary; use next_cursor from the preceding page.",
            code=ToolErrorCode.VALIDATION_ERROR,
        ) from exc
    upper_chars = min(len(remaining), max_chars)
    preview = remaining[:upper_chars]
    page_bytes = len(preview.encode("utf-8"))
    if page_bytes <= max_bytes:
        return preview, page_bytes

    low = 1
    high = upper_chars
    best_preview = ""
    best_bytes = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = remaining[:middle]
        candidate_bytes = len(candidate.encode("utf-8"))
        if candidate_bytes <= max_bytes:
            best_preview = candidate
            best_bytes = candidate_bytes
            low = middle + 1
        else:
            high = middle - 1
    if not best_preview:
        raise ToolExecutionError(
            "read_memory_object byte budget is too small for one UTF-8 character.",
            code=ToolErrorCode.EXECUTION_ERROR,
        )
    return best_preview, best_bytes


def _memory_read_output_budget(
    *,
    runtime: Any,
    ctx: ToolContext,
    tool_name: str,
    tool_version: str,
) -> int:
    """Return the JSON-byte budget for data nested in the broker envelope.

    The broker checks both the model-facing value and the durable ToolResult
    wrapper against the smaller of its result and Object Memory limits.  Build
    the fixed portion of that wrapper from the live context, then reserve a
    small margin for final telemetry whose width is only known after execution.
    """

    limit = min(
        runtime.config.tools.tool_result_payload_hard_limit_bytes,
        runtime.config.tools.memory_payload_hard_limit_bytes,
    )
    metadata = {
        "data_flow_context": runtime.data_flow.current_context().to_dict(),
        "tool_name": tool_name,
        "tool_version": tool_version,
        "trace_id": ctx.trace_id,
        "call_id": ctx.call_id,
        # The real value is added after execution. Its exact width is covered
        # by _TOOL_RESULT_ENVELOPE_SAFETY_BYTES below.
        "duration_ms": 0.0,
    }
    empty_envelope = {
        "tool_id": str(ctx.metadata.get("tool_id") or tool_name),
        "tool_name": tool_name,
        "result": None,
        "artifacts": [],
        "metadata": metadata,
    }
    fixed_envelope_bytes = json_size_bytes(empty_envelope) - json_size_bytes(None)
    return max(
        0,
        limit - fixed_envelope_bytes - _TOOL_RESULT_ENVELOPE_SAFETY_BYTES,
    )


class ReadMemoryObjectTool(SyncAgentTool[ReadMemoryObjectArgs]):
    name = "read_memory_object"
    description = (
        "Read a named Object Memory object as deterministic canonical JSON. "
        "Large values return a bounded preview and next_cursor for exact, non-overlapping continuation. "
        "Name lookup does not grant authority; the memory primitive still enforces object read capability."
    )
    args_schema = ReadMemoryObjectArgs
    output_schema = ReadMemoryObjectOutput
    policy = ToolPolicy(
        side_effects=False,
        idempotent=True,
        declared_permissions={"object.read"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object", "read"]

    def run(self, args: ReadMemoryObjectArgs, ctx: ToolContext) -> ReadMemoryObjectOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        obj = runtime.memory.get_object_by_name(ctx.pid, args.name, namespace=args.namespace)
        selected = _select_json_pointer(obj.payload, args.json_pointer)
        normalized, rendered, rendered_bytes = _canonical_memory_json(selected)
        # Normalize a legacy non-finite payload before any JSON-backed
        # observability work.  The frozen snapshot context uses the same finite
        # projection while preserving the exact Object version and labels.
        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_object_snapshot(obj)
        )
        digest = hashlib.sha256(rendered_bytes).hexdigest()
        if args.expected_sha256 is not None and args.expected_sha256 != digest:
            raise ToolExecutionError(
                "selected payload changed since the preceding page (sha256 mismatch).",
                code=ToolErrorCode.VALIDATION_ERROR,
            )

        output_budget = _memory_read_output_budget(
            runtime=runtime,
            ctx=ctx,
            tool_name=self.name,
            tool_version=self.version,
        )

        def build_output(
            *,
            payload: Any | None,
            preview: str | None,
            representation: str,
            page_bytes: int,
        ) -> ReadMemoryObjectOutput:
            end_cursor = args.cursor + page_bytes
            omitted_bytes = len(rendered_bytes) - end_cursor
            return ReadMemoryObjectOutput(
                oid=obj.oid,
                namespace=obj.namespace,
                name=obj.name,
                type=obj.type.value,
                version=obj.version,
                json_pointer=args.json_pointer,
                payload_type=_json_value_type(normalized),
                shape=_json_value_shape(normalized),
                serialized_bytes=len(rendered_bytes),
                sha256=digest,
                representation=representation,
                payload=payload,
                preview=preview,
                preview_encoding=(
                    "canonical_json_utf8" if preview is not None else None
                ),
                page_offset_bytes=args.cursor,
                page_bytes=page_bytes,
                truncated=omitted_bytes > 0,
                omitted_bytes=omitted_bytes,
                next_cursor=(end_cursor if end_cursor < len(rendered_bytes) else None),
            )

        def fits_output_budget(output: ReadMemoryObjectOutput) -> bool:
            return json_size_bytes(output.model_dump(mode="json")) <= output_budget

        if (
            args.cursor == 0
            and len(rendered) <= args.max_payload_chars
            and len(rendered_bytes) <= output_budget
        ):
            full_output = build_output(
                payload=normalized,
                preview=None,
                representation="json_value",
                page_bytes=len(rendered_bytes),
            )
            if fits_output_budget(full_output):
                return full_output

        preview, page_bytes = _canonical_json_page(
            rendered_bytes,
            cursor=args.cursor,
            max_chars=args.max_payload_chars,
            max_bytes=output_budget,
        )
        page_output = build_output(
            payload=None,
            preview=preview,
            representation="canonical_json_page",
            page_bytes=page_bytes,
        )
        if fits_output_budget(page_output):
            return page_output

        # JSON string escaping can consume more broker-envelope bytes than the
        # page's raw UTF-8 width. Find the largest prefix satisfying both.
        low = 1
        high = len(preview)
        best_output: ReadMemoryObjectOutput | None = None
        while low <= high:
            middle = (low + high) // 2
            candidate_preview = preview[:middle]
            candidate_bytes = len(candidate_preview.encode("utf-8"))
            candidate_output = build_output(
                payload=None,
                preview=candidate_preview,
                representation="canonical_json_page",
                page_bytes=candidate_bytes,
            )
            if fits_output_budget(candidate_output):
                best_output = candidate_output
                low = middle + 1
            else:
                high = middle - 1
        if best_output is None:
            raise ToolExecutionError(
                "read_memory_object result metadata exceeds the broker envelope budget.",
                code=ToolErrorCode.EXECUTION_ERROR,
            )
        return best_output


class AppendMemoryObjectTool(SyncAgentTool[AppendMemoryObjectArgs]):
    name = "append_memory_object"
    description = (
        "Append a structured entry to a mutable named Object Memory object. "
        "This is the preferred write pattern for LLM context objects because it preserves prompt-cache-friendly prefixes."
    )
    args_schema = AppendMemoryObjectArgs
    output_schema = AppendMemoryObjectOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "object", "write", "append"]

    def run(self, args: AppendMemoryObjectArgs, ctx: ToolContext) -> AppendMemoryObjectOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        source_context = runtime.data_flow.current_context()
        parent_oids, durable_source_refs = runtime.data_flow.provenance_sources(
            source_context
        )
        updated, list_field, length = runtime.memory.append_object_by_name(
            ctx.pid,
            args.name,
            args.entry,
            args.list_field,
            namespace=args.namespace,
            issued_by="append_memory_object_tool",
            source_oids=parent_oids,
            provenance_source_refs=durable_source_refs,
            source_context=source_context,
        )
        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_trusted_source_oids([updated.oid])
        )
        return AppendMemoryObjectOutput(
            oid=updated.oid,
            namespace=updated.namespace,
            name=updated.name,
            version=updated.version,
            appended=True,
            list_field=list_field,
            length=length,
        )
