from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import ValidationError
from agent_libos.models import ObjectMetadata, ObjectType, Provenance
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.tools.builtin.filesystem import normalize_process_path_argument
from agent_libos.tools.observability import ensure_json_size, json_size_bytes
from agent_libos.utils.ids import estimate_tokens

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_PROCESS_FILE_PATH_DESCRIPTION = (
    "File path relative to the process's current working directory; do not "
    "prepend that directory or the workspace root. The resolved path must "
    "remain inside the runtime workspace."
)


class CreateObjectFromFileArgs(BaseModel):
    name: str = Field(description="Namespace-local Object Memory name to create.")
    namespace: str | None = Field(default=None, description="Object Memory namespace. Defaults to this process namespace.")
    path: str = Field(description=_PROCESS_FILE_PATH_DESCRIPTION)
    encoding: str = Field(default=_TOOL_DEFAULTS.default_text_encoding, description="Text encoding.")
    max_bytes: int = Field(
        default=_TOOL_DEFAULTS.object_file_max_bytes,
        ge=1,
        le=_TOOL_DEFAULTS.object_file_hard_limit_bytes,
        description="Maximum bytes to import.",
    )
    allow_truncated: bool = Field(default=False, description="Whether to create the object if the file is truncated.")
    object_type: str = Field(default=ObjectType.ARTIFACT.value, description="ObjectType for the created object.")


class CreateObjectFromFileOutput(BaseModel):
    oid: str
    namespace: str
    name: str
    type: str
    source_path: str
    bytes_read: int
    truncated: bool


class WriteObjectToFileArgs(BaseModel):
    name: str = Field(description="Namespace-local Object Memory name to resolve and write.")
    namespace: str | None = Field(default=None, description="Object Memory namespace. Defaults to this process namespace.")
    path: str = Field(description=_PROCESS_FILE_PATH_DESCRIPTION)
    encoding: str = Field(default=_TOOL_DEFAULTS.default_text_encoding, description="Text encoding.")
    overwrite: bool = Field(default=True, description="Whether to overwrite an existing file.")


class WriteObjectToFileOutput(BaseModel):
    oid: str
    namespace: str
    name: str
    path: str
    bytes_written: int
    created: bool


class CreateObjectFromFileTool(SyncAgentTool[CreateObjectFromFileArgs]):
    name = "create_object_from_file"
    description = (
        "Create a named Object Memory object from a text file path relative to "
        "the process's current working directory; do not prepend that directory. "
        "The resolved path must remain under the runtime workspace root. "
        "The file content is stored inside Object Memory but is not returned in the tool result."
    )
    args_schema = CreateObjectFromFileArgs
    output_schema = CreateObjectFromFileOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"filesystem.read", "object.write"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "filesystem", "object"]

    def run(self, args: CreateObjectFromFileArgs, ctx: ToolContext) -> CreateObjectFromFileOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        cwd = runtime.process.working_directory(ctx.pid)
        path = normalize_process_path_argument(args.path, cwd)
        try:
            result = runtime.filesystem.read_text(
                pid=ctx.pid,
                path=path,
                encoding=args.encoding,
                max_bytes=args.max_bytes,
                cwd=cwd,
            )
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(
                "File could not be decoded with the requested encoding.",
                code=ToolErrorCode.EXECUTION_ERROR,
                details={"encoding": args.encoding, "path": args.path},
            ) from exc
        if result.truncated and not args.allow_truncated:
            raise ToolExecutionError(
                "File exceeded max_bytes; no object was created.",
                code=ToolErrorCode.EXECUTION_ERROR,
                details={"path": result.path, "bytes_read": result.bytes_read, "max_bytes": args.max_bytes},
            )
        # The content moves into Object Memory, but the tool result exposes only
        # metadata so a process can copy files without seeing the bytes.
        payload = {
            "kind": "workspace_text_file",
            "source_path": result.path,
            "encoding": args.encoding,
            "content": result.content,
            "bytes_read": result.bytes_read,
            "truncated": result.truncated,
        }
        payload = self._fit_payload_to_memory_limit(payload, args.allow_truncated, runtime.config.tools.memory_payload_hard_limit_bytes)
        flow = runtime.data_flow.current_context()
        labels = flow.labels
        parent_oids, durable_source_refs = runtime.data_flow.provenance_sources(flow)
        handle = runtime.memory.create_object(
            pid=ctx.pid,
            object_type=ObjectType(args.object_type),
            payload=payload,
            metadata=ObjectMetadata(
                title=args.name,
                tags=["file_object", "workspace_file"],
                mime_type="text/plain",
                token_estimate=estimate_tokens(payload),
                sensitivity=labels.sensitivity.value,
                trust_level=labels.trust_level.value,
                integrity=labels.integrity.value,
                origin=labels.origin,
                tenant=labels.tenant,
                principal=labels.principal,
                declassification_authority=labels.declassification_authority,
            ),
            provenance=Provenance(
                created_from_action="tool.create_object_from_file",
                parent_oids=list(parent_oids),
                source_refs=list(durable_source_refs),
            ),
            immutable=True,
            name=args.name,
            namespace=args.namespace,
        )
        obj = runtime.memory.get_object(ctx.pid, handle)
        return CreateObjectFromFileOutput(
            oid=handle.oid,
            namespace=obj.namespace,
            name=obj.name,
            type=args.object_type,
            source_path=result.path,
            bytes_read=result.bytes_read,
            truncated=bool(payload["truncated"]),
        )

    def _fit_payload_to_memory_limit(
        self,
        payload: dict[str, Any],
        allow_truncated: bool,
        limit_bytes: int,
    ) -> dict[str, Any]:
        try:
            ensure_json_size(payload, limit_bytes, "file object payload")
            return payload
        except ValidationError as exc:
            if not allow_truncated:
                raise ToolExecutionError(
                    "File object payload exceeded Object Memory payload limit; no object was created.",
                    code=ToolErrorCode.EXECUTION_ERROR,
                    details={"limit_bytes": limit_bytes, "payload_bytes": json_size_bytes(payload)},
                ) from exc
        content = payload.get("content")
        if not isinstance(content, str):
            raise ToolExecutionError(
                "File object payload does not contain text content to truncate.",
                code=ToolErrorCode.EXECUTION_ERROR,
                details={"limit_bytes": limit_bytes},
            )
        base = dict(payload)
        base["truncated"] = True
        low = 0
        high = len(content)
        best: dict[str, Any] | None = None
        while low <= high:
            mid = (low + high) // 2
            candidate = dict(base)
            candidate["content"] = content[:mid]
            candidate["stored_bytes"] = len(candidate["content"].encode(str(payload.get("encoding") or "utf-8")))
            if json_size_bytes(candidate) <= limit_bytes:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        if best is None:
            raise ToolExecutionError(
                "File object metadata alone exceeded Object Memory payload limit; no object was created.",
                code=ToolErrorCode.EXECUTION_ERROR,
                details={"limit_bytes": limit_bytes},
            )
        return best


class WriteObjectToFileTool(SyncAgentTool[WriteObjectToFileArgs]):
    name = "write_object_to_file"
    description = (
        "Write a named Object Memory object's text to a path relative to the "
        "process's current working directory; do not prepend that directory. "
        "The resolved path must remain under the runtime workspace root. "
        "The object content is not returned in the tool result."
    )
    args_schema = WriteObjectToFileArgs
    output_schema = WriteObjectToFileOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"filesystem.write", "object.read"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["memory", "filesystem", "object", "side_effect"]

    def run(self, args: WriteObjectToFileArgs, ctx: ToolContext) -> WriteObjectToFileOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        obj = runtime.memory.get_object_by_name(ctx.pid, args.name, namespace=args.namespace)
        text = self._extract_text(obj.payload)
        # The object payload is handed directly to the filesystem primitive; the
        # process-visible result below still omits the concrete content.
        try:
            cwd = runtime.process.working_directory(ctx.pid)
            path = normalize_process_path_argument(args.path, cwd)
            result = runtime.filesystem.write_text(
                pid=ctx.pid,
                path=path,
                text=text,
                encoding=args.encoding,
                overwrite=args.overwrite,
                cwd=cwd,
                source_oids=[obj.oid],
            )
        except FileExistsError as exc:
            raise ToolExecutionError(
                "File already exists and overwrite is false.",
                code=ToolErrorCode.EXECUTION_ERROR,
                details={"path": args.path},
            ) from exc
        return WriteObjectToFileOutput(
            oid=obj.oid,
            namespace=obj.namespace,
            name=obj.name,
            path=result.path,
            bytes_written=result.bytes_written,
            created=result.created,
        )

    def _extract_text(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("content"), str):
            return payload["content"]
        raise ToolExecutionError(
            "Object payload does not contain text content.",
            code=ToolErrorCode.EXECUTION_ERROR,
            details={"expected": "string payload or dict content string"},
        )
