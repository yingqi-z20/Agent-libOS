from __future__ import annotations

import builtins
import json
import math
import re
import threading
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.human.manager import HumanObjectManager
from agent_libos.utils.openai_schema import openai_chat_tool_schema
from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.models import (
    AgentObject,
    JIT_MULTIPLEXER_TOOL_NAME,
    JITRehydrationSummary,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    ObjectOwnerKind,
    ObjectType,
    ToolCallResult,
    ToolCandidate,
    ToolHandle,
    ToolSpec,
    ValidationResult,
)
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.ports import (
    AuditPort,
    DataFlowPort,
    EventPort,
    OperationPort,
    RuntimePublicationReceiptRecorder,
)
from agent_libos.storage import UnitOfWork
from agent_libos.tools.base import BaseAgentTool, SyncAgentTool
from agent_libos.tools.execution import JITSyscallSession, ToolExecutionService
from agent_libos.tools.jit import JITToolService
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.tools.registry import ToolRegistry
from agent_libos.tools.sandbox import DenoTypescriptSandbox, SandboxBackend
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import dumps

_JIT_DECLARED_PERMISSIONS = (
    "checkpoint.read",
    "checkpoint.write",
    "filesystem.delete",
    "filesystem.read",
    "filesystem.write",
    "human.ask",
    "human.output",
    "image.write",
    "jsonrpc.call",
    "libos.syscall",
    "object.link",
    "object.read",
    "object.write",
    "process.lifecycle",
    "process.message",
    "process.signal",
    "process.spawn",
    "shell.execute",
    "skill.read",
    "skill.write",
)

_REMOVED_TOOL_GROUP_METADATA_KEYS = frozenset(
    {
        "initial_tool_groups",
        "lazy_tool_groups",
    }
)

_CANONICAL_JSON_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_CANONICAL_JSON_NUMBER = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z"
)
_MODEL_SCALAR_STRING_MAX_CHARS = 128
_MODEL_ARGUMENT_NORMALIZATION_MAX_DEPTH = 32
_MODEL_ARGUMENT_NORMALIZATION_MAX_FIELDS = 64

_SKILL_TOOL_BOOTSTRAP = (
    "discover_skills",
    "activate_skill",
    "read_skill_resource",
    "unload_skill",
    "process_exit",
)

_JIT_MULTIPLEXER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "description": "Name of a visible process-local JIT tool to execute.",
        },
        "arguments": {
            "type": "object",
            "description": "Arguments for the selected JIT tool.",
            "additionalProperties": True,
        },
    },
    "required": ["tool_name", "arguments"],
    "additionalProperties": False,
}
# This spec is only an LLM protocol surface. It is never inserted into a
# process tool table, so direct runtime calls still resolve to real tools only.
_JIT_MULTIPLEXER_SPEC = ToolSpec(
    name=JIT_MULTIPLEXER_TOOL_NAME,
    description=(
        "Execute one visible process-local Deno/TypeScript JIT tool by name. "
        "The image prompt must describe available JIT tool names and argument shapes."
    ),
    input_schema=_JIT_MULTIPLEXER_INPUT_SCHEMA,
    output_schema={"type": "object"},
    policy={
        "side_effects": True,
        "idempotent": False,
        "declared_permissions": list(_JIT_DECLARED_PERMISSIONS),
    },
    tags=["jit", "tool", "multiplexer", "protocol"],
    side_effects=list(_JIT_DECLARED_PERMISSIONS),
)


def _normalize_schema_scalar_strings(
    value: Any,
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    path: tuple[str, ...],
    changed: list[str],
    depth: int,
) -> Any:
    """Normalize canonical scalar strings only when every schema variant forbids strings."""

    if (
        depth > _MODEL_ARGUMENT_NORMALIZATION_MAX_DEPTH
        or len(changed) >= _MODEL_ARGUMENT_NORMALIZATION_MAX_FIELDS
    ):
        return value
    variants = _expanded_schema_variants(
        schema,
        root_schema=root_schema,
        depth=depth,
        seen_refs=frozenset(),
    )
    if not variants:
        return value

    if isinstance(value, str):
        allowed_types = _schema_variant_types(variants)
        normalized = _normalize_scalar_string(value, allowed_types)
        if normalized is value:
            return value
        changed.append(_normalization_path(path))
        return normalized

    if isinstance(value, dict):
        return _normalize_schema_mapping(
            value,
            variants,
            root_schema=root_schema,
            path=path,
            changed=changed,
            depth=depth,
        )

    if isinstance(value, list):
        return _normalize_schema_items(
            value,
            variants,
            root_schema=root_schema,
            path=path,
            changed=changed,
            depth=depth,
        )

    return value


def _normalize_schema_mapping(
    value: dict[Any, Any],
    variants: list[dict[str, Any]],
    *,
    root_schema: dict[str, Any],
    path: tuple[str, ...],
    changed: list[str],
    depth: int,
) -> dict[Any, Any]:
    normalized_mapping = dict(value)
    for key, item in value.items():
        if len(changed) >= _MODEL_ARGUMENT_NORMALIZATION_MAX_FIELDS:
            break
        property_schemas = [
            properties[key]
            for variant in variants
            if isinstance((properties := variant.get("properties")), dict)
            and key in properties
            and isinstance(properties[key], dict)
        ]
        if not property_schemas:
            continue
        selected_schema = (
            property_schemas[0]
            if len(property_schemas) == 1
            else {"anyOf": property_schemas}
        )
        normalized_mapping[key] = _normalize_schema_scalar_strings(
            item,
            selected_schema,
            root_schema=root_schema,
            path=(*path, str(key)),
            changed=changed,
            depth=depth + 1,
        )
    return normalized_mapping


def _normalize_schema_items(
    value: list[Any],
    variants: list[dict[str, Any]],
    *,
    root_schema: dict[str, Any],
    path: tuple[str, ...],
    changed: list[str],
    depth: int,
) -> list[Any]:
    item_schemas = [
        item_schema
        for variant in variants
        if isinstance((item_schema := variant.get("items")), dict)
    ]
    if not item_schemas:
        return value
    selected_schema = (
        item_schemas[0] if len(item_schemas) == 1 else {"anyOf": item_schemas}
    )
    normalized_items = list(value)
    for index, item in enumerate(value):
        if len(changed) >= _MODEL_ARGUMENT_NORMALIZATION_MAX_FIELDS:
            break
        normalized_items[index] = _normalize_schema_scalar_strings(
            item,
            selected_schema,
            root_schema=root_schema,
            path=(*path, f"[{index}]"),
            changed=changed,
            depth=depth + 1,
        )
    return normalized_items


def _expanded_schema_variants(
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    depth: int,
    seen_refs: frozenset[str],
) -> list[dict[str, Any]]:
    if depth > _MODEL_ARGUMENT_NORMALIZATION_MAX_DEPTH:
        return []
    selected = dict(schema)
    ref = selected.get("$ref")
    if isinstance(ref, str):
        if ref in seen_refs:
            return []
        target = _local_schema_ref(root_schema, ref)
        if target is None:
            return []
        selected = {**target, **{key: item for key, item in selected.items() if key != "$ref"}}
        seen_refs = seen_refs | {ref}

    raw_variants = selected.get("anyOf") or selected.get("oneOf")
    if isinstance(raw_variants, list):
        common = {
            key: item
            for key, item in selected.items()
            if key not in {"anyOf", "oneOf"}
        }
        expanded: list[dict[str, Any]] = []
        for raw_variant in raw_variants:
            if not isinstance(raw_variant, dict):
                continue
            expanded.extend(
                _expanded_schema_variants(
                    {**common, **raw_variant},
                    root_schema=root_schema,
                    depth=depth + 1,
                    seen_refs=seen_refs,
                )
            )
        return expanded
    return [selected]


def _local_schema_ref(
    root_schema: dict[str, Any],
    ref: str,
) -> dict[str, Any] | None:
    if not ref.startswith("#/"):
        return None
    selected: Any = root_schema
    for raw_token in ref[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(selected, dict) or token not in selected:
            return None
        selected = selected[token]
    return dict(selected) if isinstance(selected, dict) else None


def _schema_variant_types(variants: list[dict[str, Any]]) -> set[str]:
    selected: set[str] = set()
    for variant in variants:
        schema_type = variant.get("type")
        if isinstance(schema_type, str):
            selected.add(schema_type)
        elif isinstance(schema_type, list):
            selected.update(item for item in schema_type if isinstance(item, str))
    return selected


def _normalize_scalar_string(value: str, allowed_types: set[str]) -> Any:
    if not allowed_types or "string" in allowed_types:
        return value
    stripped = value.strip()
    if not stripped or len(stripped) > _MODEL_SCALAR_STRING_MAX_CHARS:
        return value
    lowered = stripped.casefold()
    if "null" in allowed_types and lowered == "null":
        return None
    if "boolean" in allowed_types and lowered in {"true", "false"}:
        return lowered == "true"
    if "integer" in allowed_types and _CANONICAL_JSON_INTEGER.fullmatch(stripped):
        try:
            return int(stripped)
        except ValueError:
            return value
    if "number" in allowed_types and _CANONICAL_JSON_NUMBER.fullmatch(stripped):
        try:
            normalized = json.loads(stripped)
        except (TypeError, ValueError):
            return value
        if (
            isinstance(normalized, (int, float))
            and not isinstance(normalized, bool)
            and (not isinstance(normalized, float) or math.isfinite(normalized))
        ):
            return normalized
    return value


def _normalization_path(path: tuple[str, ...]) -> str:
    if not path:
        return "$"
    rendered = ""
    for item in path:
        if item.startswith("["):
            rendered += item
        else:
            rendered += ("." if rendered else "") + item
    return rendered[:512]


class ToolBroker:
    """Registry and dispatch boundary for model-facing tools."""

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        memory: ObjectMemoryManager,
        capabilities: CapabilityManager,
        human: HumanObjectManager,
        audit: AuditPort,
        events: EventPort,
        operations: OperationPort,
        data_flow: DataFlowPort,
        jit_session_factory: Callable[[str], JITSyscallSession],
        tool_context_host: Any,
        images: dict[str, Any],
        registry_lifecycle_lock: threading.RLock,
        sandbox: SandboxBackend | None = None,
        workspace_root: str | Path | None = None,
        config: AgentLibOSConfig | None = None,
        resources: Any | None = None,
        lifecycle: Any | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.unit_of_work = unit_of_work
        self.processes = unit_of_work.processes
        self.objects = unit_of_work.objects
        self.extensions = unit_of_work.extensions
        self.memory = memory
        self.capabilities = capabilities
        self.human = human
        self.audit = audit
        self.events = events
        self.resources = resources
        self.data_flow = data_flow
        self._jit_session_factory = jit_session_factory
        self._tool_context_host = tool_context_host
        self._images = images
        self._registry_lifecycle_lock_value = registry_lifecycle_lock
        self._lifecycle = lifecycle
        self.sandbox = sandbox or DenoTypescriptSandbox(
            deno_executable=self.config.tools.deno_executable,
            default_timeout_s=self.config.tools.deno_timeout_s,
            max_rpc_calls=self.config.tools.deno_max_rpc_calls,
            max_stdout_bytes=self.config.tools.deno_max_stdout_bytes,
            max_stderr_bytes=self.config.tools.deno_max_stderr_bytes,
            jsr_allowlist=self.config.tools.deno_jsr_allowlist,
            max_validation_log_chars=self.config.tools.jit_validation_log_max_chars,
            forbidden_executable_roots=[Path(workspace_root or Path.cwd()).resolve()],
        )
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.registry = ToolRegistry(unit_of_work, audit, self.config)
        self.jit = JITToolService(
            unit_of_work=unit_of_work,
            memory=memory,
            audit=audit,
            sandbox=self.sandbox,
            registry=self.registry,
            config=self.config,
            declared_permissions=_JIT_DECLARED_PERMISSIONS,
            resources=resources,
            images=images,
            require_recovery_lease=(
                lifecycle.require_recovery_lease if lifecycle is not None else None
            ),
        )
        self.execution = ToolExecutionService(
            data_flow=data_flow,
            operations=operations,
            evidence=unit_of_work.evidence,
            processes=unit_of_work.processes,
            extensions=unit_of_work.extensions,
            memory=memory,
            audit=audit,
            events=events,
            resources=resources,
            registry=self.registry,
            sandbox=self.sandbox,
            config=self.config,
            jit_session_factory=jit_session_factory,
            tool_context_host=tool_context_host,
            workspace_root=self.workspace_root,
            registry_lifecycle_lock=registry_lifecycle_lock,
        )

    @property
    def sandbox(self) -> SandboxBackend:
        return self._sandbox

    @sandbox.setter
    def sandbox(self, value: SandboxBackend) -> None:
        self._sandbox = value
        if hasattr(self, "jit"):
            self.jit.sandbox = value
        if hasattr(self, "execution"):
            self.execution.sandbox = value

    def _registry_lifecycle_lock(self) -> threading.RLock:
        return self._registry_lifecycle_lock_value

    def _mutation_admission(self) -> Any:
        if self._lifecycle is None:
            return nullcontext()
        return self._lifecycle.admit()

    def has_side_effects(self, tool: ToolHandle | str) -> bool:
        """Inspect the declared side-effect policy through the public broker."""

        return self.execution.has_side_effects(self.resolve(tool))

    def register_tool(
        self,
        tool: BaseAgentTool,
        registered_by: str = "runtime",
        scope: str = "static",
        ephemeral: bool = False,
        publication_id: str | None = None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None = None,
    ) -> ToolHandle:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            if publication_id is None:
                return self._register_tool_locked(
                    tool,
                    registered_by=registered_by,
                    scope=scope,
                    ephemeral=ephemeral,
                )
            handle: ToolHandle | None = None
            try:
                with self.unit_of_work.transaction():
                    handle = self._register_tool_locked(
                        tool,
                        registered_by=registered_by,
                        scope=scope,
                        ephemeral=ephemeral,
                    )
                    self._record_publication_tool(
                        publication_id,
                        handle,
                        receipt_recorder=receipt_recorder,
                    )
            except BaseException:
                if handle is not None:
                    self.registry.discard_loaded_registration(handle)
                raise
            return handle

    def _register_tool_locked(
        self,
        tool: BaseAgentTool,
        registered_by: str = "runtime",
        scope: str = "static",
        ephemeral: bool = False,
    ) -> ToolHandle:
        return self.registry.register(
            tool,
            registered_by=registered_by,
            scope=scope,
            ephemeral=ephemeral,
        )

    def unregister_tool(
        self,
        tool: ToolHandle | str,
        *,
        registered_by: str | None = None,
    ) -> bool:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            return self._unregister_tool_locked(
                tool,
                registered_by=registered_by,
            )

    def discard_tool_registration(self, handle: ToolHandle) -> bool:
        """Remove a captured in-memory tool after a failed module rollback."""

        with self._mutation_admission(), self._registry_lifecycle_lock():
            return self.registry.discard_loaded_registration(handle)

    def _unregister_tool_locked(
        self,
        tool: ToolHandle | str,
        *,
        registered_by: str | None = None,
    ) -> bool:
        return self.registry.unregister(tool, registered_by=registered_by)

    def configure_process_tools(
        self,
        pid: str,
        tools: builtins.list[ToolHandle | str],
        assigned_by: str = "tool_broker",
    ) -> dict[str, str]:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            return self._configure_process_tools(pid, tools, assigned_by=assigned_by)

    def _configure_process_tools(
        self,
        pid: str,
        tools: builtins.list[ToolHandle | str],
        *,
        assigned_by: str,
    ) -> dict[str, str]:
        if self.processes.get_process(pid) is None:
            raise NotFound(f"process not found: {pid}")
        resolved: dict[str, ToolHandle] = {}
        for tool in tools:
            try:
                handle = self.resolve(tool)
            except NotFound:
                if isinstance(tool, ToolHandle):
                    self.jit.preflight_process_bindings(
                        pid,
                        (tool,),
                        assigned_by=assigned_by,
                    )
                raise
            resolved[handle.name] = handle
        self.jit.preflight_process_bindings(
            pid,
            resolved.values(),
            assigned_by=assigned_by,
        )
        with self.unit_of_work.transaction():
            process = self.processes.get_process(pid)
            if process is None:
                raise NotFound(f"process not found: {pid}")
            table = {
                name: handle.tool_id
                for name, handle in resolved.items()
            }
            updated_at = utc_now()
            self.processes.patch_process(
                pid,
                {
                    "tool_table": dict(table),
                    "model_tool_table": dict(table),
                    "updated_at": updated_at,
                },
                expected_revision=process.revision,
            )
            self.audit.record(
                actor=assigned_by,
                action="process.tools.configure",
                target=f"process:{pid}",
                decision={"tools": sorted(table)},
            )
        return dict(table)

    def grant_execute(self, pid: str, tool: ToolHandle | str, issued_by: str = "tool_broker") -> str:
        with self._mutation_admission():
            handle = self.resolve(tool, pid=pid)
            if not self.registry.process_has_tool(pid, handle):
                raise ValidationError(
                    "tool execute capabilities are not a resource authority; configure process tools at creation time"
                )
            self.audit.record(
                actor=issued_by,
                action="tool.execute_grant_ignored",
                target=f"tool:{handle.tool_id}",
                decision={
                    "tool": handle.name,
                    "reason": "tool calls are allowed by process tool table, not execute capability",
                },
            )
        return handle.tool_id

    def process_has_tool(self, pid: str, tool: ToolHandle | str) -> bool:
        handle = self.resolve(tool, pid=pid)
        return self.registry.process_has_tool(pid, handle)

    def is_sync_side_effect_tool(self, tool: ToolHandle | str) -> bool:
        handle = self.resolve(tool)
        implementation = self.registry.implementation(handle.tool_id)
        if implementation is None:
            return False
        return isinstance(implementation, SyncAgentTool) and bool(implementation.spec().policy.get("side_effects"))

    def propose(
        self,
        pid: str,
        spec: ToolSpec | dict[str, Any],
        source_code: str,
        tests: builtins.list[dict[str, Any]] | None = None,
        requested_capabilities: builtins.list[dict[str, Any]] | None = None,
        publication_id: str | None = None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None = None,
    ) -> str:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            if publication_id is None:
                return self.jit.propose(
                    pid,
                    spec,
                    source_code,
                    tests=tests,
                    requested_capabilities=requested_capabilities,
                )
            with self.unit_of_work.transaction(include_object_payloads=True):
                candidate_id, descriptor_oid = self.jit.propose_with_descriptor(
                    pid,
                    spec,
                    source_code,
                    tests=tests,
                    requested_capabilities=requested_capabilities,
                )
                self._record_publication_artifact(
                    publication_id,
                    {
                        "artifact_id": f"candidate:{candidate_id}",
                        "kind": "tool_candidate",
                        "candidate_id": candidate_id,
                        "descriptor_state": "object",
                        "descriptor_oid": descriptor_oid,
                        "pid": pid,
                    },
                    receipt_recorder=receipt_recorder,
                )
            return candidate_id

    def validate(
        self,
        candidate_id: str,
        *,
        pid: str | None = None,
    ) -> ValidationResult:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            return self.jit.validate(candidate_id, pid=pid)

    def register(
        self,
        pid: str,
        candidate_id: str,
        approver: str = "policy:local",
        scope: str = "ephemeral_process",
        replace_tool_id: str | None = None,
        publication_id: str | None = None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None = None,
    ) -> ToolHandle:
        with self._mutation_admission():
            with self._registry_lifecycle_lock():
                if publication_id is None:
                    return self._register_jit_locked(
                        pid,
                        candidate_id,
                        approver=approver,
                        scope=scope,
                        replace_tool_id=replace_tool_id,
                    )
                handle: ToolHandle | None = None
                try:
                    with self.unit_of_work.transaction():
                        handle = self._register_jit_locked(
                            pid,
                            candidate_id,
                            approver=approver,
                            scope=scope,
                            replace_tool_id=replace_tool_id,
                        )
                        self._record_publication_tool(
                            publication_id,
                            handle,
                            receipt_recorder=receipt_recorder,
                        )
                except BaseException:
                    if handle is not None:
                        self.registry.forget_jit(handle.tool_id)
                    raise
                return handle

    def _register_jit_locked(
        self,
        pid: str,
        candidate_id: str,
        approver: str = "policy:local",
        scope: str = "ephemeral_process",
        replace_tool_id: str | None = None,
    ) -> ToolHandle:
        return self.jit.register(
            pid,
            candidate_id,
            approver=approver,
            scope=scope,
            replace_tool_id=replace_tool_id,
        )

    def _record_publication_tool(
        self,
        publication_id: str,
        handle: ToolHandle,
        *,
        receipt_recorder: RuntimePublicationReceiptRecorder | None,
    ) -> None:
        self._record_publication_artifact(
            publication_id,
            {
                "artifact_id": f"tool:{handle.tool_id}",
                "kind": "tool",
                "tool_id": handle.tool_id,
                "name": handle.name,
            },
            receipt_recorder=receipt_recorder,
        )

    def _record_publication_artifact(
        self,
        publication_id: str,
        artifact: dict[str, Any],
        *,
        receipt_recorder: RuntimePublicationReceiptRecorder | None,
    ) -> None:
        recorder = (
            receipt_recorder
            if receipt_recorder is not None
            else self.unit_of_work.publications
        )
        if not recorder.record_runtime_publication_artifact(
            publication_id,
            artifact,
            expected_states={"planning", "applying"},
        ):
            raise ValidationError(
                "runtime publication changed while recording tool artifact: "
                f"{publication_id}"
            )

    def discard_candidate(
        self,
        pid: str,
        candidate_id: str,
        *,
        descriptor_oid: str | None = None,
        exact_descriptor: bool = False,
        discarded_by: str = "tool_broker",
        reason: str = "candidate_abandoned",
    ) -> bool:
        """Delete an unpublished candidate and its Object Memory descriptor."""

        with (
            self._mutation_admission(),
            self._registry_lifecycle_lock(),
            self.memory.ownership_locked(),
            self.unit_of_work.transaction(include_object_payloads=True),
        ):
            candidate = self.extensions.get_tool_candidate(candidate_id)
            if candidate is not None:
                self.jit.require_candidate_owner(candidate, pid)
                if self._candidate_has_registered_artifacts(
                    pid,
                    candidate,
                    exact_lookup=exact_descriptor,
                ):
                    raise ValidationError(f"cannot discard a registered tool candidate: {candidate_id}")
            candidate_objects = self._candidate_descriptor_objects(
                pid,
                candidate_id,
                descriptor_oid=descriptor_oid,
                exact_lookup=exact_descriptor,
            )
            if candidate is None and not candidate_objects:
                return False
            for obj in candidate_objects:
                self.memory.delete_object_trusted(
                    discarded_by,
                    obj.oid,
                    reason=f"tool candidate discarded: {reason}",
                )
            self.extensions.delete_tool_candidate(candidate_id, pid)
            self.audit.record(
                actor=discarded_by,
                action="tool.candidate.discard",
                target=f"tool_candidate:{candidate_id}",
                input_refs=[obj.oid for obj in candidate_objects],
                decision={"pid": pid, "reason": reason},
            )
        return True

    def _candidate_has_registered_artifacts(
        self,
        pid: str,
        candidate: ToolCandidate,
        *,
        exact_lookup: bool,
    ) -> bool:
        registered_tool_id = candidate.registered_tool_id
        if not registered_tool_id:
            return False
        if exact_lookup:
            tool_exists = registered_tool_id in self.extensions.get_existing_tool_ids(
                (registered_tool_id,)
            )
            process = self.processes.get_process(pid)
            local_alias_exists = bool(
                process is not None
                and registered_tool_id
                in {
                    *process.tool_table.values(),
                    *process.model_tool_table.values(),
                }
            )
            outside_alias_exists = (
                self.processes.tool_id_referenced_outside_process(
                    registered_tool_id,
                    excluding_pid=pid,
                )
            )
            return tool_exists or local_alias_exists or outside_alias_exists
        tool_exists = any(
            row["tool_id"] == registered_tool_id
            for row in self.extensions.list_tools()
        )
        alias_exists = any(
            registered_tool_id
            in {
                *process.tool_table.values(),
                *process.model_tool_table.values(),
            }
            for process in self.processes.list_processes()
        )
        return tool_exists or alias_exists

    def _candidate_descriptor_objects(
        self,
        pid: str,
        candidate_id: str,
        *,
        descriptor_oid: str | None,
        exact_lookup: bool,
    ) -> list[AgentObject]:
        if not exact_lookup:
            return [
                obj
                for obj in self.objects.list_objects_owned_by(
                    ObjectOwnerKind.PROCESS,
                    pid,
                )
                if obj.type == ObjectType.TOOL_CANDIDATE
                and isinstance(obj.payload, dict)
                and obj.payload.get("candidate_id") == candidate_id
            ]
        if descriptor_oid is None:
            return []
        obj = self.objects.get_object(descriptor_oid)
        if obj is None:
            return []
        if (
            obj.owner_kind != ObjectOwnerKind.PROCESS
            or obj.owner_id != pid
            or obj.type != ObjectType.TOOL_CANDIDATE
            or not isinstance(obj.payload, dict)
            or obj.payload.get("candidate_id") != candidate_id
        ):
            raise ValidationError(
                "tool candidate descriptor receipt identity mismatch: "
                f"{candidate_id} -> {descriptor_oid}"
            )
        return [obj]

    def call(
        self,
        pid: str,
        tool: ToolHandle | str,
        args: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        if self._lifecycle is None:
            return self.execution.call(
                pid,
                tool,
                args,
                context_metadata=context_metadata,
            )
        with self._lifecycle.admit():
            return self.execution.call(
                pid,
                tool,
                args,
                context_metadata=context_metadata,
            )

    async def acall(
        self,
        pid: str,
        tool: ToolHandle | str,
        args: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        if self._lifecycle is None:
            return await self.execution.acall(
                pid,
                tool,
                args,
                context_metadata=context_metadata,
            )
        with self._lifecycle.admit():
            return await self.execution.acall(
                pid,
                tool,
                args,
                context_metadata=context_metadata,
            )

    def resolve(self, tool: ToolHandle | str, pid: str | None = None) -> ToolHandle:
        with self._registry_lifecycle_lock():
            return self._resolve_locked(tool, pid=pid)

    def _resolve_locked(
        self,
        tool: ToolHandle | str,
        pid: str | None = None,
    ) -> ToolHandle:
        return self.registry.resolve(tool, pid=pid)

    def list(self, *, limit: int | None = None) -> builtins.list[dict[str, Any]]:
        return self.registry.list(limit=limit)

    def visible_tools(self, pid: str) -> builtins.list[dict[str, Any]]:
        visible_ids = self._visible_tool_ids(pid)
        return [
            row
            for row in self.extensions.list_tools()
            if row["tool_id"] in visible_ids
        ]

    def model_visible_tools(self, pid: str) -> builtins.list[dict[str, Any]]:
        visible_ids = self._model_visible_tool_ids(pid)
        rows = [
            row
            for row in self.extensions.list_tools()
            if row["tool_id"] in visible_ids
        ]
        if self._jit_exposure_for_process(pid) != JIT_TOOL_EXPOSURE_MULTIPLEXED:
            return rows
        static_rows = [
            row for row in rows
            if not self.registry.is_jit(str(row.get("tool_id")))
        ]
        if any(self.registry.is_jit(str(row.get("tool_id"))) for row in rows):
            static_rows.append(self._jit_multiplexer_row())
        return static_rows

    def initial_tool_projection(self, image: Any) -> list[str]:
        metadata = getattr(image, "metadata", {})
        removed_keys = sorted(_REMOVED_TOOL_GROUP_METADATA_KEYS.intersection(metadata))
        if removed_keys:
            raise ValidationError(
                "image metadata contains removed tool-group fields: "
                f"{', '.join(removed_keys)}; run the offline tool-group migration"
            )
        if "tool_projection" not in metadata:
            return list(image.default_tools)
        projection = metadata["tool_projection"]
        if projection != "skills":
            raise ValidationError(
                "image metadata tool_projection must be 'skills' when present"
            )
        allowed = set(image.default_tools)
        missing = [name for name in _SKILL_TOOL_BOOTSTRAP if name not in allowed]
        if missing:
            raise ValidationError(
                "image metadata tool_projection='skills' requires all bootstrap "
                f"tools; missing: {', '.join(missing)}"
            )
        return list(_SKILL_TOOL_BOOTSTRAP)

    def configure_model_tool_projection(
        self,
        pid: str,
        tools: builtins.list[ToolHandle | str],
        *,
        assigned_by: str,
    ) -> dict[str, str]:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            return self._configure_model_tool_projection(
                pid,
                tools,
                assigned_by=assigned_by,
            )

    def _configure_model_tool_projection(
        self,
        pid: str,
        tools: builtins.list[ToolHandle | str],
        *,
        assigned_by: str,
    ) -> dict[str, str]:
        with self.unit_of_work.transaction():
            process = self.processes.get_process(pid)
            if process is None:
                raise NotFound(f"process not found: {pid}")
            table: dict[str, str] = {}
            for tool in tools:
                handle = self.resolve(tool, pid=pid)
                if process.tool_table.get(handle.name) != handle.tool_id:
                    raise ValidationError(
                        f"tool is not authorized by process image: {handle.name}"
                    )
                table[handle.name] = handle.tool_id
            updated_at = utc_now()
            self.processes.patch_process(
                pid,
                {
                    "model_tool_table": dict(table),
                    "updated_at": updated_at,
                },
                expected_revision=process.revision,
            )
            self.audit.record(
                actor=assigned_by,
                action="process.tools.project",
                target=f"process:{pid}",
                decision={"tools": sorted(table), "authority_changed": False},
            )
        return dict(table)

    def model_tool_names(self, pid: str) -> builtins.list[str]:
        names = [str(row.get("name") or "") for row in self.model_visible_tools(pid)]
        return sorted(name for name in names if name)

    def model_tool_table(self, pid: str) -> dict[str, str]:
        return {
            str(row["name"]): str(row["tool_id"])
            for row in self.model_visible_tools(pid)
            if row.get("name") and row.get("tool_id")
        }

    def model_loaded_skills(self, pid: str) -> dict[str, Any]:
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        hidden = self._hidden_jit_tool_names(pid)
        model_loaded: dict[str, Any] = {}
        for skill_id, loaded in process.loaded_skills.items():
            if not isinstance(loaded, dict):
                model_loaded[skill_id] = {"invalid_snapshot": True}
                continue
            # Keep only fields shared by every loaded Skill. Host provenance,
            # activation kind, receipts, and base-binding restoration state do
            # not belong in model context or its provider cache fingerprint.
            entry = {
                key: loaded[key]
                for key in (
                    "skill_id",
                    "version",
                    "package_sha256",
                    "instructions_hash",
                    "package_snapshot",
                    "tool_names",
                    "tool_ids",
                    "jit_tool_ids",
                )
                if key in loaded
            }
            entry = self._redact_hidden_jit_names(entry, hidden)
            if hidden:
                entry["jit_tool_ids"] = {}
                if isinstance(entry.get("tool_names"), list):
                    entry["tool_names"] = [
                        name for name in entry["tool_names"]
                        if name != "<multiplexed_jit_tool>"
                    ]
            model_loaded[skill_id] = entry
        return model_loaded

    def redact_model_context(self, pid: str, value: Any) -> Any:
        hidden = self._hidden_jit_tool_names(pid)
        if not hidden:
            return value
        return self._redact_hidden_jit_names(value, hidden)

    def openai_tool_schemas(self, pid: str | None = None) -> builtins.list[dict[str, Any]]:
        tool_ids = (
            self._model_visible_tool_ids(pid)
            if pid is not None
            else set(self.registry.implementation_ids())
        )
        multiplex_jit = pid is not None and self._jit_exposure_for_process(pid) == JIT_TOOL_EXPOSURE_MULTIPLEXED
        has_visible_jit = any(self.registry.is_jit(tool_id) for tool_id in tool_ids)
        schemas: builtins.list[dict[str, Any]] = []
        for tool_id in sorted(tool_ids, key=self._tool_sort_key):
            implementation = self.registry.implementation(tool_id)
            if implementation is not None:
                schemas.append(implementation.to_openai_chat_tool(config=self.config))
                continue
            if not self.registry.is_jit(tool_id):
                continue
            if multiplex_jit:
                continue
            spec = self.extensions.get_tool_spec(tool_id)
            if spec is None:
                continue
            schemas.append(openai_chat_tool_schema(spec.name, spec.description, spec.input_schema))
        if multiplex_jit and has_visible_jit:
            schemas.append(self._jit_multiplexer_openai_schema())
        return schemas

    def normalize_model_action(self, pid: str, action: dict[str, Any]) -> dict[str, Any]:
        name = str(action.get("action") or "").strip()
        if name in {"read_process_messages", "receive_process_messages"}:
            action, normalized_fields = self._normalize_process_message_model_action(
                action
            )
            if normalized_fields:
                self.audit.record(
                    actor=pid,
                    action="llm.process_message_arguments_normalized",
                    target=name,
                    decision={
                        "tool": name,
                        "normalized_fields": normalized_fields,
                        "normalization": "reversible_json_string_encoding",
                    },
                )
        elif name != JIT_MULTIPLEXER_TOOL_NAME:
            action, normalized_fields = self._normalize_schema_model_action(
                pid,
                name,
                action,
            )
            if normalized_fields:
                self.audit.record(
                    actor=pid,
                    action="llm.tool_arguments_normalized",
                    target=name,
                    decision={
                        "tool": name,
                        "normalized_fields": normalized_fields,
                        "normalization": "schema_guided_canonical_scalar_string",
                    },
                )
        if name == JIT_MULTIPLEXER_TOOL_NAME:
            if self._jit_exposure_for_process(pid) != JIT_TOOL_EXPOSURE_MULTIPLEXED:
                raise ValueError(f"{JIT_MULTIPLEXER_TOOL_NAME} is not available for this image")
            return self._normalize_multiplexed_jit_action(pid, action)
        if self._jit_exposure_for_process(pid) == JIT_TOOL_EXPOSURE_MULTIPLEXED and self._is_visible_jit_name(pid, name):
            raise ValueError(f"JIT tool {name!r} must be called through {JIT_MULTIPLEXER_TOOL_NAME}")
        if self._is_visible_jit_name(pid, name):
            handle = self.resolve(name, pid=pid)
            args = {key: value for key, value in action.items() if key != "action"}
            self.execution.validate_jit_arguments(handle, args)
        return action

    def _normalize_schema_model_action(
        self,
        pid: str,
        name: str,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Repair only unambiguous scalar stringification at the LLM boundary."""

        process = self.processes.get_process(pid)
        if process is None or name not in process.model_tool_table:
            return action, []
        try:
            handle = self.resolve(name, pid=pid)
        except NotFound:
            return action, []
        spec = self.extensions.get_tool_spec(handle.tool_id)
        if spec is None or not isinstance(spec.input_schema, dict):
            return action, []
        arguments = {
            key: value for key, value in action.items() if key != "action"
        }
        changed: list[str] = []
        normalized = _normalize_schema_scalar_strings(
            arguments,
            spec.input_schema,
            root_schema=spec.input_schema,
            path=(),
            changed=changed,
            depth=0,
        )
        if not changed or not isinstance(normalized, dict):
            return action, []
        return {**normalized, "action": name}, sorted(set(changed))

    @staticmethod
    def _normalize_process_message_model_action(
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Undo reversible JSON stringification from non-conforming providers."""

        normalized = dict(action)
        changed: list[str] = []
        nullable_fields = {
            "kind",
            "sender",
            "channel",
            "correlation_id",
            "reply_to",
        }
        for field in nullable_fields:
            value = normalized.get(field)
            if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
                normalized[field] = None
                changed.append(field)
        message_ids = normalized.get("message_ids")
        if isinstance(message_ids, str):
            stripped = message_ids.strip()
            if stripped.lower() in {"none", "null"}:
                normalized["message_ids"] = None
                changed.append("message_ids")
            else:
                try:
                    decoded_ids = json.loads(stripped)
                except (TypeError, ValueError):
                    decoded_ids = None
                if (
                    isinstance(decoded_ids, list)
                    and all(isinstance(item, str) for item in decoded_ids)
                ):
                    normalized["message_ids"] = decoded_ids
                    changed.append("message_ids")
        limit = normalized.get("limit")
        if isinstance(limit, str) and limit.strip().isdigit():
            normalized["limit"] = int(limit.strip())
            changed.append("limit")
        for field in ("include_acked", "ack", "block"):
            value = normalized.get(field)
            if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                normalized[field] = value.strip().lower() == "true"
                changed.append(field)
        return normalized, sorted(changed)

    def _normalize_multiplexed_jit_action(self, pid: str, action: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(action.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError(f"{JIT_MULTIPLEXER_TOOL_NAME} requires a non-empty tool_name")
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"{JIT_MULTIPLEXER_TOOL_NAME}.arguments must be an object")
        try:
            handle = self.resolve(tool_name, pid=pid)
        except NotFound as exc:
            raise ValueError(f"JIT tool is not available in this process: {tool_name}") from exc
        if not self.registry.is_jit(handle.tool_id):
            raise ValueError(f"{JIT_MULTIPLEXER_TOOL_NAME} can only dispatch process-local JIT tools: {tool_name}")
        if not self.registry.process_has_tool(pid, handle):
            raise ValueError(f"JIT tool is not in process tool table: {tool_name}")
        self.execution.validate_jit_arguments(handle, arguments)
        return {**arguments, "action": tool_name}

    def _is_visible_jit_name(self, pid: str, name: str) -> bool:
        if not name:
            return False
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        tool_id = process.tool_table.get(name)
        return self.registry.is_jit(tool_id) if tool_id is not None else False

    def _hidden_jit_tool_names(self, pid: str) -> set[str]:
        if self._jit_exposure_for_process(pid) != JIT_TOOL_EXPOSURE_MULTIPLEXED:
            return set()
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return {
            name
            for name, tool_id in process.tool_table.items()
            if self.registry.is_jit(tool_id)
        }

    def _redact_hidden_jit_names(self, value: Any, hidden: set[str]) -> Any:
        if isinstance(value, str):
            redacted = value
            for name in hidden:
                redacted = redacted.replace(name, "<multiplexed_jit_tool>")
            return redacted
        if isinstance(value, list):
            return [self._redact_hidden_jit_names(item, hidden) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_hidden_jit_names(item, hidden) for item in value)
        if isinstance(value, dict):
            return {
                self._redact_hidden_jit_names(key, hidden): self._redact_hidden_jit_names(item, hidden)
                for key, item in value.items()
            }
        return value

    def _jit_exposure_for_process(self, pid: str | None) -> str:
        if pid is None:
            return ""
        return self.jit.exposure_for_process(pid)

    def _jit_multiplexer_row(self) -> dict[str, Any]:
        return {
            "tool_id": f"protocol:{JIT_MULTIPLEXER_TOOL_NAME}",
            "name": JIT_MULTIPLEXER_TOOL_NAME,
            "spec_json": dumps(_JIT_MULTIPLEXER_SPEC),
            "scope": "llm_protocol",
            "registered_by": "runtime",
            "created_at": "",
            "ephemeral": 1,
        }

    def _jit_multiplexer_openai_schema(self) -> dict[str, Any]:
        return openai_chat_tool_schema(
            _JIT_MULTIPLEXER_SPEC.name,
            _JIT_MULTIPLEXER_SPEC.description,
            _JIT_MULTIPLEXER_SPEC.input_schema,
        )

    def _tool_sort_key(self, tool_id: str) -> tuple[str, str]:
        handle = self.registry.handle(tool_id)
        if handle is not None:
            return (handle.name, tool_id)
        spec = self.extensions.get_tool_spec(tool_id)
        if spec is not None:
            return (spec.name, tool_id)
        return (tool_id, tool_id)

    def name_collides_with_static_tool(self, name: str) -> bool:
        """Report whether a process-local name conflicts with a static tool."""

        return self.registry.name_collides_with_static_tool(name)

    def static_check_jit_source(self, source: str) -> ValidationResult:
        """Validate image-bundled JIT source through the configured sandbox."""

        return self.jit.static_check_source(source)

    def is_jit_tool_id(self, tool_id: str) -> bool:
        """Return whether a tool id names a registered process-local JIT."""

        return self.registry.is_jit(tool_id)

    def jit_source(self, tool_id: str) -> str | None:
        """Return a loaded JIT source for trusted host snapshotting."""

        return self.registry.jit_source(tool_id)

    def loaded_tool_handle(self, tool_id: str) -> ToolHandle | None:
        return self.registry.handle(tool_id)

    def reconstruct_persisted_jit_handles(
        self,
        sources: dict[str, str],
    ) -> dict[str, ToolHandle]:
        """Rebuild trusted rollback handles from matching durable JIT rows."""

        if not sources:
            return {}
        tool_rows = {
            str(row["tool_id"]): row
            for row in self.extensions.list_tools()
        }
        candidates = {
            str(row.get("registered_tool_id") or ""): row
            for row in self.extensions.list_registered_tool_candidate_rows()
            if row.get("registered_tool_id")
        }
        handles: dict[str, ToolHandle] = {}
        for tool_id, source in sources.items():
            row = tool_rows.get(str(tool_id))
            candidate = candidates.get(str(tool_id))
            if (
                row is None
                or candidate is None
                or not bool(row.get("ephemeral"))
                or str(candidate.get("source_code") or "") != str(source)
            ):
                raise ValidationError(
                    f"durable JIT rollback state is incomplete: {tool_id}"
                )
            handles[str(tool_id)] = ToolHandle(
                tool_id=str(tool_id),
                name=str(row["name"]),
                capability_id=None,
                scope=str(row["scope"]),
            )
        return handles

    def loaded_tool_handles(self) -> tuple[ToolHandle, ...]:
        return self.registry.loaded_handles()

    def loaded_tool_ids(self) -> frozenset[str]:
        return frozenset(handle.tool_id for handle in self.registry.loaded_handles())

    def loaded_jit_tool_ids(self) -> frozenset[str]:
        return self.registry.jit_ids()

    def snapshot_loaded_tool_state(
        self,
        tool_ids: Iterable[str],
    ) -> tuple[dict[str, ToolHandle], dict[str, str]]:
        return self.registry.snapshot_loaded_state(tool_ids)

    def restore_loaded_jit_state(
        self,
        handles: dict[str, ToolHandle],
        sources: dict[str, str],
    ) -> None:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            self.registry.restore_loaded_jit_state(handles, sources)

    def forget_loaded_jit(self, tool_id: str) -> None:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            self.registry.forget_jit(tool_id)

    def install_committed_jit(
        self,
        pid: str,
        *,
        name: str,
        scope: str,
        spec: ToolSpec,
        source_code: str,
        registered_by: str,
        publication_id: str | None = None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None = None,
    ) -> ToolHandle:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            if publication_id is None:
                handle, _candidate_id = self.jit.install_committed(
                    pid,
                    name=name,
                    scope=scope,
                    spec=spec,
                    source_code=source_code,
                    registered_by=registered_by,
                )
                return handle
            handle: ToolHandle | None = None
            try:
                with self.unit_of_work.transaction():
                    handle, candidate_id = self.jit.install_committed(
                        pid,
                        name=name,
                        scope=scope,
                        spec=spec,
                        source_code=source_code,
                        registered_by=registered_by,
                    )
                    self._record_publication_artifact(
                        publication_id,
                        {
                            "artifact_id": f"candidate:{candidate_id}",
                            "kind": "tool_candidate",
                            "candidate_id": candidate_id,
                            "descriptor_state": "not_created",
                            "descriptor_oid": None,
                            "pid": pid,
                        },
                        receipt_recorder=receipt_recorder,
                    )
                    self._record_publication_tool(
                        publication_id,
                        handle,
                        receipt_recorder=receipt_recorder,
                    )
            except BaseException:
                if handle is not None:
                    self.registry.forget_jit(handle.tool_id)
                raise
            return handle

    def rehydrate_registered_jit_tools(self) -> JITRehydrationSummary:
        with self._mutation_admission(), self._registry_lifecycle_lock():
            return self.jit.rehydrate_registered()

    def _visible_tool_ids(self, pid: str) -> set[str]:
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return set(process.tool_table.values())

    def _model_visible_tool_ids(self, pid: str) -> set[str]:
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return set(process.model_tool_table.values())

    def _error_observation(self, text: str) -> dict[str, Any]:
        return sanitize_for_observability(
            text,
            preview_chars=self.config.tools.tool_observability_preview_chars,
        )
