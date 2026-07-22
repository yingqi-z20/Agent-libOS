from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.memory.data_labels import propagate_object_labels
from agent_libos.models import ObjectMetadata, ObjectType, Provenance, ViewMode
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class CreateMemoryObjectArgs(BaseModel):
    name: str | None = Field(default=None, description="Optional namespace-local object name.")
    namespace: str | None = Field(default=None, description="Object Memory namespace. Defaults to this process namespace.")
    type: str = Field(description="Agent libOS object type, for example summary, plan, observation, or artifact.")
    payload: Any = Field(description="Structured payload to store.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_oids: list[str] = Field(
        default_factory=list,
        description="Optional Object Memory source ids used for conservative data-label propagation.",
    )
    immutable: bool = True


class CreateMemoryObjectOutput(BaseModel):
    oid: str
    namespace: str
    name: str
    type: str


class ReadMemoryObjectArgs(BaseModel):
    name: str = Field(description="Namespace-local Object Memory name to read.")
    namespace: str | None = Field(default=None, description="Object Memory namespace. Defaults to this process namespace.")
    max_payload_chars: int = Field(
        default=_TOOL_DEFAULTS.memory_payload_chars,
        ge=1,
        le=_TOOL_DEFAULTS.memory_payload_hard_limit_chars,
        description="Maximum rendered payload chars.",
    )


class ReadMemoryObjectOutput(BaseModel):
    oid: str
    namespace: str
    name: str
    type: str
    version: int
    payload: Any
    truncated: bool


class AppendMemoryObjectArgs(BaseModel):
    name: str = Field(description="Namespace-local mutable Object Memory name to append to.")
    namespace: str | None = Field(default=None, description="Object Memory namespace. Defaults to this process namespace.")
    entry: Any = Field(description="Structured entry to append.")
    list_field: str = Field(
        default="entries",
        description="Payload list field to append into when the object payload is a JSON object.",
    )


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


class CreateMemoryNamespaceOutput(BaseModel):
    namespace: str
    parent_namespace: str | None
    created: bool


class ListMemoryNamespaceArgs(BaseModel):
    namespace: str | None = Field(default=None, description="Namespace to list. Defaults to this process namespace.")
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of visible namespace entries to return. Defaults to the runtime memory query limit.",
    )


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
        with runtime.store.locked():
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
        return CreateMemoryObjectOutput(oid=handle.oid, namespace=obj.namespace, name=obj.name, type=args.type)


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


class ReadMemoryObjectTool(SyncAgentTool[ReadMemoryObjectArgs]):
    name = "read_memory_object"
    description = (
        "Read a named Object Memory object. Name lookup does not grant authority; "
        "the memory primitive still enforces object read capability."
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
        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_trusted_source_oids([obj.oid])
        )
        payload = obj.payload
        rendered = repr(payload)
        truncated = len(rendered) > args.max_payload_chars
        if truncated:
            payload = rendered[: args.max_payload_chars]
        return ReadMemoryObjectOutput(
            oid=obj.oid,
            namespace=obj.namespace,
            name=obj.name,
            type=obj.type.value,
            version=obj.version,
            payload=payload,
            truncated=truncated,
        )


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
