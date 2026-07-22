from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.validators import validator_for as jsonschema_validator_for

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    CapabilityDecision,
    CapabilityRight,
    EventType,
    JIT_MULTIPLEXER_TOOL_NAME,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    OPENAI_TOOL_NAME_MAX_CHARS,
    ToolCandidateStatus,
    is_openai_tool_name,
)
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, NotFound, ValidationError
from agent_libos.ports import AuditPort, EventPort, RuntimePublicationReceiptRecorder
from agent_libos.skills.schema import ActionSchema, JitToolSpec, LoadedSkill, SkillPackage, SkillResource
from agent_libos.storage import UnitOfWork
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import bounded_json_loads, dumps, to_jsonable
from agent_libos.utils.yaml_loader import load_yaml_mapping

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
_SOURCE_TYPES = {"workspace", "global", "runtime"}
_FRONTMATTER_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
_AGENT_LIBOS_METADATA_KEYS = {
    "agent-libos.version",
    "agent-libos.actions",
    "agent-libos.required-capabilities",
    "agent-libos.jit-tools",
}


@dataclass(slots=True)
class _DeferredJitRegistryFinalization:
    """JIT registry changes finalized by an enclosing authority transaction."""

    published_tool_ids: set[str] = field(default_factory=set)
    retired_tool_ids: set[str] = field(default_factory=set)

    def capture(self, handles: Mapping[str, Any], retired_tool_ids: Iterable[str]) -> None:
        self.published_tool_ids.update(
            str(handle.tool_id) for handle in handles.values()
        )
        self.retired_tool_ids.update(str(tool_id) for tool_id in retired_tool_ids)


def _with_registry_lifecycle_lock(method: Callable[..., Any]) -> Callable[..., Any]:
    """Guard direct manager calls before their first durable/registry read."""

    @wraps(method)
    def guarded(self: "SkillManager", *args: Any, **kwargs: Any) -> Any:
        with self._lifecycle_lock:
            return method(self, *args, **kwargs)

    return guarded


class SkillManager:
    """Capability-controlled primitive for standard Agent Skill packages.

    Skills use the standard package shape rooted at ``SKILL.md``. Activation
    changes only prompt materialization and process-local tool visibility; all
    external authority still comes from capability-checked primitives.
    """

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        capabilities: CapabilityManager,
        audit: AuditPort,
        events: EventPort,
        tools: Any,
        filesystem: Any,
        process: Any,
        images: Mapping[str, Any],
        lifecycle_lock: Any,
        *,
        config: AgentLibOSConfig | None = None,
        human: Any | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.unit_of_work = unit_of_work
        self.store = unit_of_work.extensions
        self.processes = unit_of_work.processes
        self.publications = unit_of_work.publications
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.human = human
        self._tools = tools
        self._filesystem = filesystem
        self._process = process
        self._images = images
        self._lifecycle_lock = lifecycle_lock

    def resource_for(self, skill_id: str) -> str:
        return f"skill:{skill_id}"

    def trust_resource(self, package_sha256: str = "*") -> str:
        return self.config.skills.trust_resource if package_sha256 == "*" else f"skill_trust:{package_sha256}"

    def source_resource(self, source_type: str, source: str) -> str:
        return f"skill_source:{source_type}:{source}"

    def validate_package_path(self, path: str | Path) -> dict[str, Any]:
        package, source = self._load_package_from_host_path(path)
        return {
            "skill_id": package.skill_id,
            "name": package.name,
            "description": package.description,
            "instructions_sha256": self._hash_text(package.instructions),
            "version": package.version,
            "source": source,
            "package_sha256": package.package_sha256,
            "resources": [resource.path for resource in package.resources],
            "allowed_tools": list(package.allowed_tools),
            "jit_tools": [tool.name for tool in package.jit_tools],
            "actions": [action.name for action in package.actions],
            "diagnostics": list(package.diagnostics),
            "valid": True,
        }

    def register_skill_package(
        self,
        package: SkillPackage,
        *,
        actor: str,
        replace: bool = False,
        require_capability: bool = True,
        source_type: str = "runtime",
        source: str | None = None,
        package_sha256: str | None = None,
    ) -> dict[str, Any]:
        spec = self._coerce_package(package)
        self._validate_package(spec)
        selected_source_type = self._validate_source_type(source_type)
        selected_source = source or selected_source_type
        selected_sha = package_sha256 or spec.package_sha256 or self._package_hash(spec)
        if selected_source_type == "global":
            self._require_trusted_global_source(selected_source, selected_sha)
        if require_capability:
            decisions = self._require_skill_right(actor, spec.skill_id, CapabilityRight.WRITE)
        else:
            decisions = []
        now = utc_now()
        if spec.package_sha256 != selected_sha:
            spec = self._replace_package_hash(spec, selected_sha)
        with self.capabilities.authority_transaction(
            decisions,
            actor=actor,
            operation="skill registration",
        ):
            if selected_source_type == "global":
                self._require_trusted_global_source(selected_source, selected_sha)
            with self.unit_of_work.transaction():
                existing = self.store.get_skill(spec.skill_id)
                if existing is not None and not replace:
                    raise ValidationError(f"skill already registered: {spec.skill_id}")
                self.store.upsert_skill(
                    spec,
                    source_type=selected_source_type,
                    source=selected_source,
                    package_sha256=selected_sha,
                    registered_by=actor,
                    created_at=now,
                )
                self.events.emit(
                    EventType.SKILL_REGISTERED,
                    source=actor,
                    target=self.resource_for(spec.skill_id),
                    payload={"skill_id": spec.skill_id, "version": spec.version, "source_type": selected_source_type},
                )
                self.audit.record(
                    actor=actor,
                    action="skill.register",
                    target=self.resource_for(spec.skill_id),
                    decision={
                        "replace": existing is not None,
                        "source_type": selected_source_type,
                        "source": selected_source,
                        "package_sha256": selected_sha,
                        "allowed_tools": list(spec.allowed_tools),
                        "jit_tools": [tool.name for tool in spec.jit_tools],
                        "resources": [resource.path for resource in spec.resources],
                    },
                )
        return self.inspect_skill(spec.skill_id, actor=actor, require_capability=False)

    def register_skill_from_path(
        self,
        path: str | Path,
        *,
        actor: str,
        replace: bool = False,
        require_capability: bool = True,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        package, source = self._load_package_from_host_path(path)
        selected_source_type = source_type or self._source_type_for_host_path(Path(source))
        return self.register_skill_package(
            package,
            actor=actor,
            replace=replace,
            require_capability=require_capability,
            source_type=selected_source_type,
            source=source,
            package_sha256=package.package_sha256,
        )

    def register_global_skill_from_path(
        self,
        path: str | Path,
        *,
        actor: str,
        replace: bool = False,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        absolute, source_id = self._normalize_global_path(path)
        package, _source = self._load_package_from_host_path(absolute)
        return self.register_skill_package(
            package,
            actor=actor,
            replace=replace,
            require_capability=require_capability,
            source_type="global",
            source=source_id,
            package_sha256=package.package_sha256,
        )

    def global_package_info(self, path: str | Path) -> dict[str, Any]:
        absolute, source_id = self._normalize_global_path(path)
        package, _source = self._load_package_from_host_path(absolute)
        return {
            "path": str(absolute),
            "source": source_id,
            "package_sha256": package.package_sha256,
            "skill_id": package.skill_id,
            "bytes": sum(resource.size_bytes for resource in package.resources),
        }

    def register_skill_from_workspace_path(
        self,
        pid: str,
        path: str,
        *,
        replace: bool = False,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        package, source = self._load_package_from_workspace(pid, path)
        return self.register_skill_package(
            package,
            actor=pid,
            replace=replace,
            require_capability=require_capability,
            source_type="workspace",
            source=source,
            package_sha256=package.package_sha256,
        )

    def activate_skill_from_workspace_path(
        self,
        pid: str,
        path: str,
        *,
        replace: bool = False,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        package, source = self._load_package_from_workspace(pid, path)
        if require_capability:
            decisions = self._require_skill_rights(pid, package.skill_id, [CapabilityRight.WRITE, CapabilityRight.EXECUTE])
        else:
            decisions = []
        write_uses = self._decision_consume_ids(
            decision
            for decision in decisions
            if decision.right == CapabilityRight.WRITE.value
        )
        execute_uses = self._decision_consume_ids(
            decision
            for decision in decisions
            if decision.right == CapabilityRight.EXECUTE.value
        )
        shared_one_time_authority = write_uses & execute_uses
        if shared_one_time_authority:
            activation_error: Exception | None = None
            result: dict[str, Any] | None = None
            deferred_jit = _DeferredJitRegistryFinalization()
            with self._lifecycle_lock:
                try:
                    with self.capabilities.authority_transaction(
                        decisions,
                        actor=pid,
                        operation="workspace skill registration and activation",
                    ):
                        self.register_skill_package(
                            package,
                            actor=pid,
                            replace=replace,
                            require_capability=False,
                            source_type="workspace",
                            source=source,
                            package_sha256=package.package_sha256,
                        )
                        try:
                            result = self.activate_skill(
                                pid,
                                package.skill_id,
                                actor=pid,
                                require_capability=False,
                                _deferred_jit_finalization=deferred_jit,
                            )
                        except Exception as exc:
                            activation_error = exc
                except BaseException as exc:
                    try:
                        self._forget_jit_tool_ids(deferred_jit.published_tool_ids)
                    except Exception as cleanup_exc:
                        exc.add_note(
                            "failed to discard workspace Skill JIT publications "
                            "after authority rollback: "
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
                    raise
                self._forget_jit_tool_ids(deferred_jit.retired_tool_ids)
            if activation_error is not None:
                raise activation_error
            assert result is not None
        else:
            self.register_skill_package(
                package,
                actor=pid,
                replace=replace,
                require_capability=require_capability,
                source_type="workspace",
                source=source,
                package_sha256=package.package_sha256,
            )
            result = self.activate_skill(
                pid,
                package.skill_id,
                actor=pid,
                require_capability=require_capability,
            )
        return {**result, "source": source, "registered": True}

    def discover_skills(
        self,
        text: str | None = None,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        skills, _has_more = self.discover_skills_window(
            text,
            actor=actor,
            require_capability=require_capability,
            limit=limit,
        )
        return skills

    def discover_skills_window(
        self,
        text: str | None = None,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return one bounded catalog page plus an exact next-item signal."""

        reservations: dict[str, str] = {}
        if require_capability and actor is not None:
            decision = self.capabilities.require(
                actor,
                self.config.skills.registry_resource,
                CapabilityRight.READ,
                consume=False,
            )
            reservations = self._reserve_skill_rights([decision], used_by="skill")
        try:
            selected_limit = self._bounded_discover_limit(limit)
            registered = [
                self._skill_summary(skill, metadata)
                for skill, metadata in self.store.list_skills(text=text, limit=selected_limit + 1)
            ]
            if actor is None and len(registered) <= selected_limit:
                seen = {item["skill_id"] for item in registered}
                registered.extend(
                    self._discover_host_skill_catalog(
                        text=text,
                        limit=selected_limit + 1 - len(registered),
                        exclude_skill_ids=seen,
                    )
                )
        except Exception:
            self._restore_skill_rights(reservations)
            raise
        self._commit_skill_rights(reservations)
        return registered[:selected_limit], len(registered) > selected_limit

    def _bounded_discover_limit(self, limit: int | None) -> int:
        selected = self.config.skills.discover_limit if limit is None else limit
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise ValidationError("Skill discover limit must be an integer")
        if selected < 1:
            raise ValidationError("Skill discover limit must be >= 1")
        if selected > self.config.skills.discover_limit:
            raise ValidationError(
                f"Skill discover limit exceeds configured maximum {self.config.skills.discover_limit}"
            )
        return selected

    def inspect_skill(
        self,
        skill_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        skill, metadata = self._get_skill(skill_id)
        reservations: dict[str, str] = {}
        if require_capability and actor is not None:
            decisions = self._require_skill_right(actor, skill_id, CapabilityRight.READ)
            reservations = self._reserve_skill_rights(decisions, used_by="skill")
        try:
            result = {
                **self._skill_summary(skill, metadata),
                "instructions": self._prompt_instructions(skill),
                "allowed_tools": list(skill.allowed_tools),
                "actions": [asdict(action) for action in skill.actions],
                "jit_tools": [self._jit_summary(tool) for tool in skill.jit_tools],
                "required_capabilities": list(skill.required_capabilities),
                "metadata": dict(skill.metadata),
                "resources": [self._resource_summary(resource) for resource in skill.resources],
                "license": skill.license,
                "compatibility": skill.compatibility,
                "diagnostics": list(skill.diagnostics),
            }
        except Exception:
            self._restore_skill_rights(reservations)
            raise
        self._commit_skill_rights(reservations)
        return result

    def prompt_context(self, pid: str) -> list[dict[str, Any]]:
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        include_jit_catalog = not self._process_uses_multiplexed_jit(process)
        result: list[dict[str, Any]] = []
        for skill_id, loaded in process.loaded_skills.items():
            try:
                skill = self._skill_for_loaded_record(skill_id, loaded)
            except ValidationError as exc:
                entry = {"skill_id": skill_id, "invalid_snapshot": True, "error": str(exc)}
                if include_jit_catalog:
                    entry["loaded"] = loaded
                result.append(entry)
                continue
            entry = {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "version": skill.version,
                "description": skill.description,
                "instructions": self._prompt_instructions(skill),
                "allowed_tools": list(skill.allowed_tools),
                "actions": [asdict(action) for action in skill.actions],
                "jit_tools": [self._jit_summary(tool) for tool in skill.jit_tools] if include_jit_catalog else [],
                "required_capabilities": list(skill.required_capabilities),
                "resources": self._prompt_resource_summaries(skill, include_jit_catalog=include_jit_catalog),
                "metadata": dict(skill.metadata),
            }
            if include_jit_catalog:
                entry["loaded"] = loaded
            result.append(entry)
        return result

    @_with_registry_lifecycle_lock
    def activate_skill(
        self,
        pid: str,
        skill_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        publication_id: str | None = None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None = None,
        _deferred_jit_finalization: _DeferredJitRegistryFinalization | None = None,
    ) -> dict[str, Any]:
        selected_actor = actor or pid
        skill, metadata = self._get_skill(skill_id)
        if require_capability:
            decisions = self._require_skill_right(selected_actor, skill_id, CapabilityRight.EXECUTE)
            admin_decision = self._require_process_admin_if_cross_actor(selected_actor, pid)
            if admin_decision is not None:
                decisions.append(admin_decision)
        else:
            decisions = []
        process = self.processes.get_process(pid)
        jit_handles: dict[str, Any] = {}
        retired_jit_ids: set[str] = set()
        if process is None:
            raise NotFound(f"process not found: {pid}")
        preflight_loaded = process.loaded_skills.get(skill.skill_id)
        preflight_jit_ids = self._loaded_tool_id_map(preflight_loaded, "jit_tool_ids")
        self._validate_loadable(
            pid,
            skill,
            process.tool_table,
            replacing_jit_tool_ids=preflight_jit_ids,
        )
        # Candidate creation is durable state, so it must be enrolled in the
        # same authority transaction as registration and process publication.
        # This also lets a recovery-fence commit rejection roll it back without
        # attempting a new mutation through the now-stale admission lease.
        with self._activation_authority_scope(
            decisions,
            actor=selected_actor,
            jit_state=lambda: (jit_handles, retired_jit_ids),
            deferred_jit_finalization=_deferred_jit_finalization,
        ):
            prepared_jit_tools = self._prepare_jit_tools(
                pid, skill, publication_id=publication_id, receipt_recorder=receipt_recorder
            )
            # Tool registry lifecycle operations acquire this lock before the store
            # lock; activation must keep that order while its store transaction is open.
            try:
                with self.unit_of_work.transaction():
                    process = self.processes.get_process(pid)
                    if process is None:
                        raise NotFound(f"process not found: {pid}")
                    previous_loaded = process.loaded_skills.get(skill.skill_id)
                    previous_tool_ids = self._loaded_tool_id_map(previous_loaded, "tool_ids")
                    previous_jit_ids = self._loaded_tool_id_map(previous_loaded, "jit_tool_ids")
                    self._validate_loadable(
                        pid,
                        skill,
                        process.tool_table,
                        replacing_jit_tool_ids=previous_jit_ids,
                    )
                    existing_handles = self._resolve_existing_tools(skill.allowed_tools)
                    base_tool_ids, base_model_tool_ids = self._activation_base_bindings(
                        process,
                        skill.skill_id,
                        set(existing_handles)
                        | {jit.name for jit, _candidate_id in prepared_jit_tools},
                    )
                    jit_handles = self._register_prepared_jit_tools(
                        pid,
                        skill,
                        prepared_jit_tools,
                        replacing_jit_tool_ids=previous_jit_ids,
                        approver=selected_actor,
                        publication_id=publication_id,
                        receipt_recorder=receipt_recorder,
                    )
                    # JIT registration advances process CAS; continue from
                    # that committed row, not the pre-registration revision.
                    process = self.processes.get_process(pid)
                    if process is None:
                        raise NotFound(f"process not found: {pid}")
                    tool_ids = {name: handle.tool_id for name, handle in existing_handles.items()}
                    jit_tool_ids = {name: handle.tool_id for name, handle in jit_handles.items()}
                    updated_table = dict(process.tool_table)
                    updated_model_table = dict(process.model_tool_table)
                    for name, tool_id in {**previous_tool_ids, **previous_jit_ids}.items():
                        if updated_table.get(name) == tool_id:
                            updated_table.pop(name, None)
                        if updated_model_table.get(name) == tool_id:
                            updated_model_table.pop(name, None)
                    for name, handle in {**existing_handles, **jit_handles}.items():
                        updated_table[name] = handle.tool_id
                        # Loading a Skill is an explicit tool-visibility action,
                        # independent of the base image's lazy built-in groups.
                        updated_model_table[name] = handle.tool_id
                    loaded = LoadedSkill(
                        skill_id=skill.skill_id,
                        version=skill.version,
                        source=metadata.get("source"),
                        package_sha256=skill.package_sha256,
                        loaded_at=utc_now(),
                        tool_names=sorted([*tool_ids, *jit_tool_ids]),
                        tool_ids=tool_ids,
                        jit_tool_ids=jit_tool_ids,
                        instructions_hash=self._hash_text(skill.instructions),
                        package_snapshot=self._skill_snapshot(skill),
                        base_tool_ids=base_tool_ids,
                        base_model_tool_ids=base_model_tool_ids,
                    )
                    process = self._persist_loaded_skill(
                        process,
                        loaded=loaded,
                        tool_table=updated_table,
                        model_tool_table=updated_model_table,
                        publication_id=publication_id,
                        receipt_recorder=receipt_recorder,
                    )
                    retired_jit_ids = set(previous_jit_ids.values()) - set(jit_tool_ids.values())
                    self._delete_jit_rows(pid, retired_jit_ids)
                    self.events.emit(
                        EventType.SKILL_LOADED,
                        source=selected_actor,
                        target=pid,
                        payload={"skill_id": skill.skill_id, "tool_names": loaded.tool_names},
                    )
                    self.audit.record(
                        actor=selected_actor,
                        action="skill.activate",
                        target=f"process:{pid}",
                        decision={
                            "skill_id": skill.skill_id,
                            "version": skill.version,
                            "replaced_loaded_version": self._loaded_version(previous_loaded),
                            "tool_ids": tool_ids,
                            "jit_tool_ids": jit_tool_ids,
                            "retired_jit_tool_ids": sorted(retired_jit_ids),
                            "source": metadata.get("source"),
                            "package_sha256": skill.package_sha256,
                        },
                    )
            except BaseException:
                # Do not expose in-memory JIT handles after the outer store
                # transaction rolled back and before releasing the lock.
                self._discard_uncommitted_jit_tools(jit_handles)
                raise
        return {
            "pid": pid,
            "skill_id": skill.skill_id,
            "name": skill.name,
            "version": skill.version,
            "tool_names": loaded.tool_names,
            "tool_ids": tool_ids,
            "jit_tool_ids": jit_tool_ids,
            "instructions_hash": loaded.instructions_hash,
            "package_sha256": skill.package_sha256,
        }

    def unload_skill(
        self,
        pid: str,
        skill_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        selected_actor = actor or pid
        if require_capability:
            decisions = self._require_skill_right(selected_actor, skill_id, CapabilityRight.EXECUTE)
            admin_decision = self._require_process_admin_if_cross_actor(selected_actor, pid)
            if admin_decision is not None:
                decisions.append(admin_decision)
        else:
            decisions = []
        removed: list[str] = []
        jit_tool_ids: dict[str, str] = {}
        retired_jit_ids: set[str] = set()
        with self._lifecycle_lock:
            with self.capabilities.authority_transaction(
                decisions,
                actor=selected_actor,
                operation="skill unload",
            ):
                with self.unit_of_work.transaction():
                    process = self.processes.get_process(pid)
                    if process is None:
                        raise NotFound(f"process not found: {pid}")
                    loaded = process.loaded_skills.get(skill_id)
                    if loaded is None:
                        raise NotFound(f"skill is not loaded in process {pid}: {skill_id}")
                    self._require_loaded_skill_provenance(loaded)
                    tool_ids = self._loaded_tool_id_map(loaded, "tool_ids")
                    jit_tool_ids = self._loaded_tool_id_map(loaded, "jit_tool_ids")
                    base_tool_ids = self._loaded_tool_id_map(loaded, "base_tool_ids")
                    base_model_tool_ids = self._loaded_tool_id_map(loaded, "base_model_tool_ids")
                    process.loaded_skills.pop(skill_id, None)
                    for name, tool_id in {**tool_ids, **jit_tool_ids}.items():
                        if process.tool_table.get(name) == tool_id:
                            replacement = self._remaining_skill_binding(process.loaded_skills, name)
                            if replacement is None:
                                replacement = base_tool_ids.get(name)
                            if replacement is None:
                                process.tool_table.pop(name, None)
                                removed.append(name)
                            else:
                                process.tool_table[name] = replacement
                        if process.model_tool_table.get(name) == tool_id:
                            replacement = self._remaining_skill_binding(process.loaded_skills, name)
                            if replacement is None:
                                replacement = base_model_tool_ids.get(name)
                            if replacement is None:
                                process.model_tool_table.pop(name, None)
                            else:
                                process.model_tool_table[name] = replacement
                    process.updated_at = utc_now()
                    process = self.processes.patch_process(
                        pid,
                        {
                            "tool_table": process.tool_table,
                            "model_tool_table": process.model_tool_table,
                            "loaded_skills": process.loaded_skills,
                            "updated_at": process.updated_at,
                        },
                        expected_revision=process.revision,
                    )
                    remaining_jit_ids = {
                        tool_id
                        for remaining in process.loaded_skills.values()
                        for tool_id in self._loaded_tool_id_map(remaining, "jit_tool_ids").values()
                    }
                    retired_jit_ids = set(jit_tool_ids.values()) - remaining_jit_ids
                    self._delete_jit_rows(pid, retired_jit_ids)
                    self.events.emit(
                        EventType.SKILL_UNLOADED,
                        source=selected_actor,
                        target=pid,
                        payload={"skill_id": skill_id, "removed_tools": sorted(removed)},
                    )
                    self.audit.record(
                        actor=selected_actor,
                        action="skill.unload",
                        target=f"process:{pid}",
                        decision={
                            "skill_id": skill_id,
                            "removed_tools": sorted(removed),
                            "retired_jit_tool_ids": sorted(retired_jit_ids),
                        },
                    )
            # A failed authority settlement rolls back the process/tool rows and
            # their evidence, so retire in-memory implementations only after the
            # enclosing AuthorityTransaction has committed successfully.
            self._forget_jit_tool_ids(retired_jit_ids)
        return {"pid": pid, "skill_id": skill_id, "removed_tools": sorted(removed)}

    def read_skill_resource(
        self,
        pid: str,
        skill_id: str,
        path: str,
        *,
        actor: str | None = None,
        max_bytes: int | None = None,
        require_loaded: bool = True,
    ) -> dict[str, Any]:
        selected_actor = actor or pid
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        loaded = process.loaded_skills.get(skill_id)
        if require_loaded:
            if loaded is None:
                raise CapabilityDenied(f"skill is not loaded in process {pid}: {skill_id}")
            skill = self._skill_for_loaded_record(skill_id, loaded)
        else:
            skill, _metadata = self._get_skill(skill_id)
        normalized = self._normalize_relative_resource_path(path)
        selected = next((resource for resource in skill.resources if resource.path == normalized), None)
        if selected is None:
            raise NotFound(f"skill resource not found: {skill_id}/{normalized}")
        limit = max_bytes or self.config.skills.resource_read_max_bytes
        if selected.size_bytes > limit:
            raise ValidationError(f"skill resource exceeds max_bytes={limit}: {normalized}")
        self.audit.record(
            actor=selected_actor,
            action="skill.read_resource",
            target=f"{self.resource_for(skill_id)}:{normalized}",
            decision={"skill_id": skill_id, "path": normalized, "size_bytes": selected.size_bytes},
        )
        payload = {
            "skill_id": skill_id,
            "path": selected.path,
            "kind": selected.kind,
            "size_bytes": selected.size_bytes,
            "sha256": selected.sha256,
            "content": selected.content,
            "content_base64": selected.content_base64,
        }
        return payload

    def trust_skill_source(
        self,
        *,
        actor: str,
        source_type: str,
        source: str,
        package_sha256: str,
        require_capability: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected_source_type = self._validate_source_type(source_type)
        decisions: list[CapabilityDecision] = []
        if require_capability:
            decisions.append(
                self.capabilities.require(
                    actor,
                    self.config.skills.trust_resource,
                    CapabilityRight.ADMIN,
                    consume=False,
                )
            )
        with self.capabilities.authority_transaction(
            decisions,
            actor=actor,
            operation="skill source trust",
        ):
            self.store.insert_skill_trust(
                trust_id=new_id("strust"),
                source_type=selected_source_type,
                source=source,
                package_sha256=package_sha256,
                trusted_by=actor,
                created_at=utc_now(),
                metadata=metadata or {},
            )
            self.events.emit(
                EventType.SKILL_TRUSTED,
                source=actor,
                target=self.trust_resource(package_sha256),
                payload={"source_type": selected_source_type, "source": source},
            )
            self.audit.record(
                actor=actor,
                action="skill.trust",
                target=self.trust_resource(package_sha256),
                decision={"source_type": selected_source_type, "source": source, "package_sha256": package_sha256},
            )
        return {"source_type": selected_source_type, "source": source, "package_sha256": package_sha256, "trusted": True}

    def untrust_skill_source(
        self,
        *,
        actor: str,
        source_type: str,
        source: str,
        package_sha256: str,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        selected_source_type = self._validate_source_type(source_type)
        decisions: list[CapabilityDecision] = []
        if require_capability:
            decisions.append(
                self.capabilities.require(
                    actor,
                    self.config.skills.trust_resource,
                    CapabilityRight.ADMIN,
                    consume=False,
                )
            )
        with self.capabilities.authority_transaction(
            decisions,
            actor=actor,
            operation="skill source untrust",
        ):
            self.store.delete_skill_trust(source_type=selected_source_type, source=source, package_sha256=package_sha256)
            self.audit.record(
                actor=actor,
                action="skill.untrust",
                target=self.trust_resource(package_sha256),
                decision={"source_type": selected_source_type, "source": source, "package_sha256": package_sha256},
            )
        return {"source_type": selected_source_type, "source": source, "package_sha256": package_sha256, "trusted": False}

    def _load_package_from_host_path(self, path: str | Path) -> tuple[SkillPackage, str]:
        skill_md = self._resolve_host_skill_md(path)
        root = skill_md.parent
        if skill_md.suffix.lower() in {".yaml", ".yml"}:
            raise ValidationError("legacy YAML Skill manifests are not supported; use a SKILL.md package")
        raw_skill = self._read_bytes_limited(skill_md, self.config.skills.skill_md_hard_limit_bytes)
        frontmatter, body = self._parse_skill_markdown(raw_skill.decode("utf-8"), expected_dir_name=root.name)
        resources = self._read_host_resources(root, raw_skill)
        package = self._package_from_parts(frontmatter, body, resources)
        return package, str(root.resolve())

    def _load_package_from_workspace(self, pid: str, path: str) -> tuple[SkillPackage, str]:
        cwd = self._process.working_directory(pid)
        package_root, skill_md_path = self._workspace_package_paths(path)
        read = self._filesystem.read_bytes(
            pid,
            skill_md_path,
            max_bytes=self.config.skills.skill_md_max_bytes,
            cwd=cwd,
        )
        if read.truncated:
            raise ValidationError(
                "SKILL.md exceeds "
                f"skill_md_max_bytes={self.config.skills.skill_md_max_bytes}: {skill_md_path}"
            )
        frontmatter, body = self._parse_skill_markdown(
            read.content.decode("utf-8"),
            expected_dir_name=Path(package_root).name,
        )
        _target, workspace_package_root = self._filesystem.resolve_path(package_root, cwd=cwd)
        references = self._frontmatter_reference_paths(frontmatter)
        raw_resources: dict[str, bytes] = {"SKILL.md": read.content}
        for ref in references:
            ref_read = self._filesystem.read_bytes(
                pid,
                self._join_relative(package_root, ref),
                max_bytes=self.config.skills.resource_read_max_bytes,
                cwd=cwd,
            )
            if ref_read.truncated:
                raise ValidationError(
                    "skill metadata resource exceeds "
                    f"resource_read_max_bytes={self.config.skills.resource_read_max_bytes}: {ref}"
                )
            raw_resources[ref] = ref_read.content
        jit_tools = self._load_jit_specs_from_resources(frontmatter, raw_resources)
        for tool in jit_tools:
            if tool.source_path not in raw_resources:
                script_read = self._filesystem.read_bytes(
                    pid,
                    self._join_relative(package_root, tool.source_path),
                    max_bytes=self.config.skills.resource_read_max_bytes,
                    cwd=cwd,
                )
                if script_read.truncated:
                    raise ValidationError(
                        "Skill JIT source exceeds "
                        f"resource_read_max_bytes={self.config.skills.resource_read_max_bytes}: "
                        f"{tool.source_path}"
                    )
                raw_resources[tool.source_path] = script_read.content
        self._read_workspace_resource_dirs(pid, workspace_package_root, raw_resources)
        resources = [self._resource_from_bytes(path, content) for path, content in sorted(raw_resources.items())]
        package = self._package_from_parts(frontmatter, body, resources)
        return package, package_root

    def _read_workspace_resource_dirs(
        self,
        pid: str,
        workspace_package_root: str,
        raw_resources: dict[str, bytes],
    ) -> None:
        max_files = self.config.skills.max_package_files
        visited_dirs: set[str] = set()

        def visit(directory: str) -> None:
            normalized_dir = directory.strip("/")
            if normalized_dir in visited_dirs:
                return
            visited_dirs.add(normalized_dir)
            if not self._has_read_authority(pid, self._filesystem.directory_resource_for_path(normalized_dir, cwd=None)):
                return
            try:
                listing = self._filesystem.read_directory(pid, normalized_dir, limit=max_files, cwd=None)
            except NotFound:
                return
            if listing.truncated:
                raise ValidationError(f"skill package exceeds max_package_files={max_files}")
            for entry in listing.entries:
                relative = self._workspace_resource_relative_path(workspace_package_root, entry.path)
                if relative is None:
                    continue
                if entry.kind == "directory":
                    visit(entry.path)
                    continue
                if entry.kind != "file" or relative in raw_resources:
                    continue
                self._validate_resource_path(relative)
                if not self._has_read_authority(pid, self._filesystem.resource_for_path(entry.path, cwd=None)):
                    continue
                read = self._filesystem.read_bytes(
                    pid,
                    entry.path,
                    max_bytes=self.config.skills.resource_read_max_bytes,
                    cwd=None,
                )
                if read.truncated:
                    raise ValidationError(f"skill resource exceeds resource_read_max_bytes={self.config.skills.resource_read_max_bytes}: {relative}")
                raw_resources[relative] = read.content
                if len(raw_resources) > max_files:
                    raise ValidationError(f"skill package exceeds max_package_files={max_files}")

        for directory in self.config.skills.resource_dirs:
            visit(self._join_relative(workspace_package_root, directory))

    def _workspace_resource_relative_path(self, workspace_package_root: str, workspace_path: str) -> str | None:
        root = workspace_package_root.strip("/")
        path = workspace_path.strip("/")
        if root in {"", "."}:
            return self._normalize_relative_resource_path(path) if path else None
        if path == root:
            return None
        prefix = f"{root}/"
        if not path.startswith(prefix):
            return None
        return self._normalize_relative_resource_path(path[len(prefix) :])

    def _has_read_authority(self, pid: str, resource: str) -> bool:
        return self.capabilities.check(pid, resource, CapabilityRight.READ)

    def _parse_skill_markdown(self, text: str, *, expected_dir_name: str | None = None) -> tuple[dict[str, Any], str]:
        normalized = text.replace("\r\n", "\n")
        lines = normalized.split("\n")
        if not lines or lines[0].strip() != "---":
            raise ValidationError("SKILL.md must start with YAML frontmatter delimited by ---")
        end_index = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_index = index
                break
        if end_index is None:
            raise ValidationError("SKILL.md frontmatter is missing closing ---")
        frontmatter_text = "\n".join(lines[1:end_index])
        body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
        data = load_yaml_mapping(frontmatter_text)
        unknown = sorted(set(data) - _FRONTMATTER_FIELDS)
        if unknown:
            raise ValidationError(f"unknown SKILL.md frontmatter fields: {unknown}")
        name = self._require_string(data.get("name"), "name")
        self._validate_skill_name(name)
        if expected_dir_name is not None and expected_dir_name != name:
            raise ValidationError(f"skill directory name must match frontmatter name: {expected_dir_name!r} != {name!r}")
        description = self._require_string(data.get("description"), "description")
        metadata = self._metadata(data.get("metadata"))
        for key in metadata:
            if key.startswith("agent-libos.") and key not in _AGENT_LIBOS_METADATA_KEYS:
                raise ValidationError(f"unknown agent-libos skill metadata key: {key}")
        raw_allowed_tools = data.get("allowed-tools")
        if raw_allowed_tools == {}:
            raw_allowed_tools = []
        allowed_tools = self._string_list(raw_allowed_tools, "allowed-tools")
        for tool in allowed_tools:
            self._validate_tool_identifier(tool, "allowed-tools[]", self.config.skills.id_max_chars)
        return {
            "name": name,
            "description": description,
            "license": self._optional_string(data.get("license"), "license") or "",
            "compatibility": self._optional_string(data.get("compatibility"), "compatibility") or "",
            "metadata": metadata,
            "allowed_tools": allowed_tools,
        }, body

    def _package_from_parts(self, frontmatter: dict[str, Any], body: str, resources: list[SkillResource]) -> SkillPackage:
        resource_map = {resource.path: resource for resource in resources}
        actions = self._load_actions_from_resources(frontmatter, resource_map)
        required_capabilities = self._load_required_capabilities_from_resources(frontmatter, resource_map)
        jit_tools = self._load_jit_specs_from_resource_map(frontmatter, resource_map)
        package = SkillPackage(
            skill_id=frontmatter["name"],
            name=frontmatter["name"],
            description=frontmatter["description"],
            instructions=body,
            version=frontmatter["metadata"].get("agent-libos.version", "v0"),
            license=frontmatter["license"],
            compatibility=frontmatter["compatibility"],
            metadata=dict(frontmatter["metadata"]),
            allowed_tools=list(frontmatter["allowed_tools"]),
            actions=actions,
            jit_tools=jit_tools,
            required_capabilities=required_capabilities,
            resources=resources,
            package_sha256="",
        )
        self._validate_package(package)
        return self._replace_package_hash(package, self._package_hash(package))

    def _load_actions_from_resources(self, frontmatter: dict[str, Any], resources: dict[str, SkillResource]) -> list[ActionSchema]:
        path = frontmatter["metadata"].get("agent-libos.actions")
        if not path:
            return []
        data = self._json_resource(resources, self._normalize_metadata_reference(path, "agent-libos.actions"))
        if not isinstance(data, list):
            raise ValidationError("agent-libos.actions JSON must be a list")
        return [self._coerce_action(item) for item in data]

    def _load_required_capabilities_from_resources(self, frontmatter: dict[str, Any], resources: dict[str, SkillResource]) -> list[dict[str, Any]]:
        path = frontmatter["metadata"].get("agent-libos.required-capabilities")
        if not path:
            return []
        data = self._json_resource(resources, self._normalize_metadata_reference(path, "agent-libos.required-capabilities"))
        return self._capability_specs(data)

    def _load_jit_specs_from_resource_map(self, frontmatter: dict[str, Any], resources: dict[str, SkillResource]) -> list[JitToolSpec]:
        path = frontmatter["metadata"].get("agent-libos.jit-tools")
        if not path:
            return []
        data = self._json_resource(resources, self._normalize_metadata_reference(path, "agent-libos.jit-tools"))
        if not isinstance(data, list):
            raise ValidationError("agent-libos.jit-tools JSON must be a list")
        result: list[JitToolSpec] = []
        for item in data:
            tool = self._coerce_jit_tool(item)
            script = resources.get(tool.source_path)
            if script is None:
                raise ValidationError(f"JIT script is missing from package snapshot: {tool.source_path}")
            if script.content is None:
                raise ValidationError(f"JIT script must be UTF-8 text: {tool.source_path}")
            result.append(
                JitToolSpec(
                    name=tool.name,
                    description=tool.description,
                    source_path=tool.source_path,
                    input_schema=tool.input_schema,
                    output_schema=tool.output_schema,
                    source=script.content,
                    tests=tool.tests,
                    metadata=tool.metadata,
                    timeout_s=tool.timeout_s,
                )
            )
        return result

    def _load_jit_specs_from_resources(self, frontmatter: dict[str, Any], raw_resources: dict[str, bytes]) -> list[JitToolSpec]:
        path = frontmatter["metadata"].get("agent-libos.jit-tools")
        if not path:
            return []
        normalized = self._normalize_metadata_reference(path, "agent-libos.jit-tools")
        raw = raw_resources.get(normalized)
        if raw is None:
            return []
        try:
            data = bounded_json_loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise ValidationError(f"invalid JSON skill metadata resource {normalized}: {exc}") from exc
        if not isinstance(data, list):
            raise ValidationError("agent-libos.jit-tools JSON must be a list")
        return [self._coerce_jit_tool(item) for item in data]

    def _read_host_resources(self, root: Path, raw_skill: bytes) -> list[SkillResource]:
        raw_resources: dict[str, bytes] = {"SKILL.md": raw_skill}
        root_resolved = root.resolve()
        for directory in self.config.skills.resource_dirs:
            candidate = root / directory
            if not candidate.exists():
                continue
            if candidate.is_symlink():
                raise ValidationError(f"skill resource path is a symlink: {directory}")
            if not candidate.is_dir():
                raise ValidationError(f"skill resource path is not a directory: {directory}")
            for file in sorted(candidate.rglob("*")):
                stat_result = file.lstat()
                if stat.S_ISLNK(stat_result.st_mode):
                    raise ValidationError(f"skill package symlinks are not supported: {file}")
                if not stat.S_ISREG(stat_result.st_mode):
                    continue
                try:
                    relative = file.relative_to(root_resolved).as_posix()
                except ValueError as exc:
                    raise ValidationError(f"skill resource escapes package root: {file}") from exc
                self._validate_resource_path(relative)
                raw_resources[relative] = self._read_bytes_limited(file, self.config.skills.resource_read_max_bytes)
                if len(raw_resources) > self.config.skills.max_package_files:
                    raise ValidationError(f"skill package exceeds max_package_files={self.config.skills.max_package_files}")
        return [self._resource_from_bytes(path, content) for path, content in sorted(raw_resources.items())]

    def _resource_from_bytes(self, path: str, content: bytes) -> SkillResource:
        sha = hashlib.sha256(content).hexdigest()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return SkillResource(
                path=path,
                size_bytes=len(content),
                sha256=sha,
                kind="base64",
                content_base64=base64.b64encode(content).decode("ascii"),
            )
        return SkillResource(path=path, size_bytes=len(content), sha256=sha, kind="text", content=text)

    def _discover_host_skill_catalog(
        self,
        *,
        text: str | None,
        limit: int,
        exclude_skill_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen = set(exclude_skill_ids or ())
        roots = self._skill_catalog_roots()
        needle = text.lower() if text else None
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                try:
                    package, source = self._load_package_from_host_path(child)
                    summary = self._skill_summary(
                        package,
                        {
                            "source_type": self._source_type_for_host_path(Path(source)),
                            "source": source,
                            "package_sha256": package.package_sha256,
                            "registered_by": None,
                            "created_at": None,
                            "updated_at": None,
                        },
                    )
                except Exception as exc:
                    summary = {
                        "skill_id": child.name,
                        "name": child.name,
                        "description": "",
                        "source_type": "diagnostic",
                        "source": str(child),
                        "registered": False,
                        "diagnostics": [str(exc)],
                    }
                if needle and needle not in dumps(summary).lower():
                    continue
                skill_id = str(summary["skill_id"])
                if skill_id in seen:
                    continue
                result.append(summary)
                seen.add(skill_id)
                if len(result) >= limit:
                    return result
        return result

    def _skill_catalog_roots(self) -> list[Path]:
        """Return configured workspace/global catalog roots without aliases."""

        roots: list[Path] = []
        seen: set[str] = set()
        for configured in (*self.config.skills.workspace_dirs, *self.config.skills.global_dirs):
            root = Path(configured).expanduser()
            # ``strict=False`` also normalizes relative spellings and symlinked
            # aliases for roots that do not exist yet.  Keep the configured
            # spelling for diagnostics and source summaries.
            key = os.path.normcase(os.fspath(root.resolve(strict=False)))
            if key in seen:
                continue
            seen.add(key)
            roots.append(root)
        return roots

    def _validate_package(self, skill: SkillPackage) -> None:
        defaults = self.config.skills
        if skill.schema_version != defaults.schema_version:
            raise ValidationError(f"unsupported Skill schema_version: {skill.schema_version}")
        self._validate_skill_name(skill.skill_id)
        if skill.skill_id != skill.name:
            raise ValidationError("SkillPackage skill_id must equal standard frontmatter name")
        if not skill.description.strip():
            raise ValidationError("SkillPackage description is required")
        self._validate_string_length(skill.version, "version", defaults.version_max_chars)
        self._validate_string_length(skill.description, "description", defaults.description_max_chars)
        if len(skill.instructions) > defaults.max_prompt_instruction_chars:
            raise ValidationError(f"instructions exceeds max_prompt_instruction_chars={defaults.max_prompt_instruction_chars}")
        if len(skill.allowed_tools) > defaults.max_tools:
            raise ValidationError(f"allowed-tools exceeds max_tools={defaults.max_tools}")
        if len(skill.actions) > defaults.max_actions:
            raise ValidationError(f"actions exceeds max_actions={defaults.max_actions}")
        if len(skill.jit_tools) > defaults.max_jit_tools:
            raise ValidationError(f"jit_tools exceeds max_jit_tools={defaults.max_jit_tools}")
        if len(skill.required_capabilities) > defaults.max_required_capabilities:
            raise ValidationError(f"required_capabilities exceeds max_required_capabilities={defaults.max_required_capabilities}")
        names = [*skill.allowed_tools, *(tool.name for tool in skill.jit_tools)]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValidationError(f"duplicate Skill tool names: {duplicates}")
        self._validate_package_resources(skill)
        for tool in skill.jit_tools:
            self._validate_jit_tool_name(tool.name, "jit_tools[].name")
            self._validate_jit_script_path(tool.source_path)
            self._coerce_jit_timeout(tool.timeout_s)
            if len(tool.source) > defaults.max_jit_source_chars:
                raise ValidationError(f"JIT source for {tool.name} exceeds max_jit_source_chars={defaults.max_jit_source_chars}")
        for spec in skill.required_capabilities:
            self._validate_capability_spec(spec)

    def _validate_package_resources(self, skill: SkillPackage) -> None:
        defaults = self.config.skills
        if len(skill.resources) > defaults.max_package_files:
            raise ValidationError(f"skill package exceeds max_package_files={defaults.max_package_files}")
        seen_paths: set[str] = set()
        total_bytes = 0
        for resource in skill.resources:
            self._validate_resource_path(resource.path)
            if resource.path in seen_paths:
                raise ValidationError(f"duplicate skill resource path: {resource.path}")
            seen_paths.add(resource.path)
            self._validate_resource_content(resource)
            if resource.path != "SKILL.md" and resource.size_bytes > defaults.resource_read_max_bytes:
                raise ValidationError(
                    "skill resource exceeds "
                    f"resource_read_max_bytes={defaults.resource_read_max_bytes}: {resource.path}"
                )
            total_bytes += resource.size_bytes
        if total_bytes > defaults.package_max_bytes:
            raise ValidationError(f"skill package exceeds package_max_bytes={defaults.package_max_bytes}")

    def _validate_loadable(
        self,
        pid: str,
        skill: SkillPackage,
        process_tool_table: dict[str, str],
        *,
        replacing_jit_tool_ids: dict[str, str] | None = None,
    ) -> None:
        static_names = {row["name"] for row in self._tools.list() if not bool(row.get("ephemeral"))}
        process = self._process.get(pid)
        image = self._images.get(process.image_id) if process is not None else None
        multiplexed_jit = getattr(image, "jit_tool_exposure", None) == JIT_TOOL_EXPOSURE_MULTIPLEXED
        replaceable = replacing_jit_tool_ids or {}
        for name in skill.allowed_tools:
            self._tools.resolve(name)
        for tool in skill.jit_tools:
            if multiplexed_jit and tool.name == JIT_MULTIPLEXER_TOOL_NAME:
                raise ValidationError(f"{JIT_MULTIPLEXER_TOOL_NAME} is reserved by multiplexed JIT tool exposure")
            existing_tool_id = process_tool_table.get(tool.name)
            if existing_tool_id is not None and replaceable.get(tool.name) != existing_tool_id:
                raise ValidationError(f"process already has a tool named: {tool.name}")
            if tool.name in static_names:
                raise ValidationError(f"JIT skill tool cannot shadow static tool: {tool.name}")

    def _resolve_existing_tools(self, names: list[str]) -> dict[str, Any]:
        return {name: self._tools.resolve(name) for name in names}

    def _prepare_jit_tools(
        self,
        pid: str,
        skill: SkillPackage,
        *,
        publication_id: str | None = None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None = None,
    ) -> list[tuple[JitToolSpec, str]]:
        prepared: list[tuple[JitToolSpec, str]] = []
        for jit in skill.jit_tools:
            candidate_id = self._tools.propose(
                pid,
                {
                    "name": jit.name,
                    "description": jit.description,
                    "input_schema": jit.input_schema,
                    "output_schema": jit.output_schema,
                    "policy": (
                        {"sandbox_timeout_s": jit.timeout_s}
                        if jit.timeout_s is not None
                        else {}
                    ),
                    "metadata": {"skill_id": skill.skill_id, "source_path": jit.source_path, **jit.metadata},
                },
                source_code=jit.source,
                tests=jit.tests,
                publication_id=publication_id,
                receipt_recorder=receipt_recorder,
            )
            prepared.append((jit, candidate_id))
            validation = self._tools.validate(candidate_id, pid=pid)
            if not validation.ok:
                raise ValidationError(f"JIT skill tool {jit.name} failed validation: {'; '.join(validation.errors)}")
            candidate = self.store.get_tool_candidate(candidate_id)
            if candidate is None:
                raise NotFound(f"tool candidate not found after validation: {candidate_id}")
            candidate.status = ToolCandidateStatus.VALIDATED
            self.store.update_tool_candidate(candidate)
        return prepared

    def _record_publication_activation(
        self,
        publication_id: str,
        *,
        pid: str,
        skill_id: str,
        loaded_record: dict[str, Any],
        receipt_recorder: RuntimePublicationReceiptRecorder | None,
    ) -> None:
        loaded_at = str(loaded_record.get("loaded_at") or "")
        if not loaded_at:
            raise ValidationError("loaded Skill publication receipt has no loaded_at locator")
        jit_tool_ids = loaded_record.get("jit_tool_ids")
        recorder = (
            receipt_recorder
            if receipt_recorder is not None
            else self.publications
        )
        if not recorder.record_runtime_publication_artifact(
            publication_id,
            {
                "artifact_id": f"skill:{pid}:{skill_id}:{loaded_at}",
                "kind": "loaded_skill",
                "pid": pid,
                "skill_id": skill_id,
                "loaded_at": loaded_at,
                "package_sha256": str(loaded_record.get("package_sha256") or ""),
                "jit_tool_ids": sorted(
                    str(tool_id)
                    for tool_id in (
                        jit_tool_ids.values()
                        if isinstance(jit_tool_ids, dict)
                        else []
                    )
                ),
            },
            expected_states={"planning", "applying"},
        ):
            raise ValidationError(
                "runtime publication changed while recording loaded Skill: "
                f"{publication_id}"
            )

    def _persist_loaded_skill(
        self,
        process: Any,
        *,
        loaded: LoadedSkill,
        tool_table: dict[str, str],
        model_tool_table: dict[str, str],
        publication_id: str | None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None,
    ) -> Any:
        loaded_record = to_jsonable(loaded)
        process.tool_table = tool_table
        process.model_tool_table = model_tool_table
        process.loaded_skills[loaded.skill_id] = loaded_record
        process.updated_at = utc_now()
        committed = self.processes.patch_process(
            process.pid,
            {
                "tool_table": process.tool_table,
                "model_tool_table": process.model_tool_table,
                "loaded_skills": process.loaded_skills,
                "updated_at": process.updated_at,
            },
            expected_revision=process.revision,
        )
        if publication_id is not None:
            self._record_publication_activation(
                publication_id,
                pid=process.pid,
                skill_id=loaded.skill_id,
                loaded_record=loaded_record,
                receipt_recorder=receipt_recorder,
            )
        return committed

    def _register_prepared_jit_tools(
        self,
        pid: str,
        skill: SkillPackage,
        prepared: list[tuple[JitToolSpec, str]],
        *,
        replacing_jit_tool_ids: dict[str, str] | None = None,
        approver: str | None = None,
        publication_id: str | None = None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None = None,
    ) -> dict[str, Any]:
        replaceable = replacing_jit_tool_ids or {}
        handles: dict[str, Any] = {}
        try:
            for jit, candidate_id in prepared:
                handles[jit.name] = self._tools.register(
                    pid,
                    candidate_id,
                    approver=approver or f"skill:{skill.skill_id}",
                    replace_tool_id=replaceable.get(jit.name),
                    publication_id=publication_id,
                    receipt_recorder=receipt_recorder,
                )
        except BaseException:
            self._discard_uncommitted_jit_tools(handles)
            raise
        return handles

    def _discard_uncommitted_jit_tools(self, handles: dict[str, Any]) -> None:
        """Remove process-local runtime aliases after the enclosing DB transaction rolled back."""

        for handle in handles.values():
            # Activation owns the shared registry lifecycle lock here.  This is
            # rollback of unpublished in-memory state, not a new mutation that
            # may acquire admission (the active lease can already be fenced).
            self._tools.registry.forget_jit(handle.tool_id)

    @contextmanager
    def _activation_authority_scope(
        self,
        decisions: Iterable[CapabilityDecision],
        *,
        actor: str,
        jit_state: Callable[[], tuple[dict[str, Any], set[str]]],
        deferred_jit_finalization: _DeferredJitRegistryFinalization | None = None,
    ) -> Iterator[None]:
        """Keep JIT handle publication aligned with authority settlement."""

        with self._lifecycle_lock:
            try:
                with self.capabilities.authority_transaction(
                    decisions,
                    actor=actor,
                    operation="skill activation",
                ):
                    yield
            except BaseException:
                handles, _retired = jit_state()
                self._discard_uncommitted_jit_tools(handles)
                raise
            handles, retired = jit_state()
            if deferred_jit_finalization is None:
                self._forget_jit_tool_ids(retired)
            else:
                deferred_jit_finalization.capture(handles, retired)

    def _delete_jit_rows(self, pid: str, tool_ids: Iterable[str]) -> None:
        self.store.delete_jit_tool_rows(pid, set(tool_ids))

    def _forget_jit_tool_ids(self, tool_ids: Iterable[str]) -> None:
        for tool_id in set(tool_ids):
            self._tools.registry.forget_jit(tool_id)

    def _loaded_tool_id_map(self, loaded: Any, field: str) -> dict[str, str]:
        if not isinstance(loaded, dict) or not isinstance(loaded.get(field), dict):
            return {}
        return {str(name): str(tool_id) for name, tool_id in loaded[field].items()}

    @staticmethod
    def _require_loaded_skill_provenance(loaded: Any) -> None:
        if not isinstance(loaded, dict) or not all(
            isinstance(loaded.get(field), dict)
            for field in ("base_tool_ids", "base_model_tool_ids")
        ):
            raise ValidationError(
                "0.3 loaded Skill state is missing canonical tool provenance"
            )

    def _activation_base_bindings(
        self,
        process: Any,
        skill_id: str,
        names: set[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        previous = process.loaded_skills.get(skill_id)
        base_tools = self._loaded_tool_id_map(previous, "base_tool_ids")
        base_model_tools = self._loaded_tool_id_map(previous, "base_model_tool_ids")
        previous_skill_bindings = {
            **self._loaded_tool_id_map(previous, "tool_ids"),
            **self._loaded_tool_id_map(previous, "jit_tool_ids"),
        }
        other_skills = {
            loaded_skill_id: loaded
            for loaded_skill_id, loaded in process.loaded_skills.items()
            if loaded_skill_id != skill_id
        }
        for name in names:
            if name not in base_tools:
                inherited = self._remaining_skill_base_binding(other_skills, name, "base_tool_ids")
                if inherited is not None:
                    base_tools[name] = inherited
                elif (
                    name not in previous_skill_bindings
                    and self._remaining_skill_binding(other_skills, name) is None
                ):
                    current = process.tool_table.get(name)
                    if current is not None:
                        base_tools[name] = str(current)
            if name not in base_model_tools:
                inherited = self._remaining_skill_base_binding(other_skills, name, "base_model_tool_ids")
                if inherited is not None:
                    base_model_tools[name] = inherited
                elif (
                    name not in previous_skill_bindings
                    and self._remaining_skill_binding(other_skills, name) is None
                ):
                    current = process.model_tool_table.get(name)
                    if current is not None:
                        base_model_tools[name] = str(current)
        return base_tools, base_model_tools

    def _remaining_skill_binding(self, loaded_skills: dict[str, Any], name: str) -> str | None:
        for loaded in reversed(list(loaded_skills.values())):
            jit_tool_ids = self._loaded_tool_id_map(loaded, "jit_tool_ids")
            if name in jit_tool_ids:
                return jit_tool_ids[name]
            tool_ids = self._loaded_tool_id_map(loaded, "tool_ids")
            if name in tool_ids:
                return tool_ids[name]
        return None

    def _remaining_skill_base_binding(
        self,
        loaded_skills: dict[str, Any],
        name: str,
        field: str,
    ) -> str | None:
        for loaded in reversed(list(loaded_skills.values())):
            binding = self._loaded_tool_id_map(loaded, field).get(name)
            if binding is not None:
                return binding
        return None

    def _loaded_version(self, loaded: Any) -> str | None:
        if not isinstance(loaded, dict):
            return None
        value = loaded.get("version")
        return str(value) if value is not None else None

    def _require_skill_right(self, actor: str, skill_id: str, right: CapabilityRight) -> list[CapabilityDecision]:
        return self._require_skill_rights(actor, skill_id, [right])

    def _require_skill_rights(self, actor: str, skill_id: str, rights: Iterable[CapabilityRight]) -> list[CapabilityDecision]:
        resource = self.resource_for(skill_id)
        missing: list[str] = []
        decisions: list[CapabilityDecision] = []
        for right in rights:
            decision = self.capabilities.authorize(actor, resource, right)
            if decision.allowed:
                decisions.append(decision)
                continue
            missing.append(str(right))
        if not missing:
            return decisions
        if self.human is None:
            raise CapabilityDenied(f"{actor} lacks {missing} on {resource}")
        request_id = self.human.query(
            pid=actor,
            human=self.config.runtime.default_human,
            request={
                "type": "permission_request",
                "question": f"Allow process {actor} to use skill {skill_id} rights={missing} once?",
                "requested_once_capability": {
                    "subject": actor,
                    "resource": resource,
                    "rights": missing,
                },
                "context": {"primitive": "skill", "skill_id": skill_id},
            },
            blocking=True,
        )
        raise HumanApprovalRequired(request_id, f"human approval required for skill {skill_id}")

    def _reserve_skill_rights(self, decisions: Iterable[CapabilityDecision], *, used_by: str) -> dict[str, str]:
        reserved: dict[str, str] = {}
        # A single finite grant may authorize several rights for one skill
        # operation.  Revalidate the complete decision set before reserving
        # any use so the first reservation cannot make a sibling decision
        # appear revoked.  Keeping both phases in one store transaction also
        # prevents a policy mutation from being interleaved between them.
        with self.capabilities.store.transaction():
            current = tuple(
                self.capabilities.reauthorize_decision(prepared)
                for prepared in decisions
            )
            for decision in current:
                cap_id = str(decision.consume_capability_id) if decision.consume_capability_id is not None else None
                if cap_id is None or cap_id in reserved:
                    continue
                reservation_id = self.capabilities.reserve_decision_use(
                    decision,
                    used_by=used_by,
                    reason="one-time skill permission reserved",
                )
                if reservation_id is not None:
                    reserved[cap_id] = reservation_id
        return reserved

    def _commit_skill_rights(
        self,
        reservations: dict[str, str],
        *,
        capability_ids: set[str] | None = None,
        exclude_capability_ids: set[str] | None = None,
    ) -> None:
        selected = self._select_skill_reservations(
            reservations,
            capability_ids=capability_ids,
            exclude_capability_ids=exclude_capability_ids,
        )
        for cap_id, reservation_id in selected.items():
            committed = self.capabilities.commit_reserved_use(
                reservation_id,
                committed_by="skill",
                reason=f"one-time skill permission committed: {cap_id}",
            )
            if not committed:
                raise CapabilityDenied(
                    "skill authority reservation is no longer active"
                )
            reservations.pop(cap_id, None)

    def _restore_skill_rights(
        self,
        reservations: dict[str, str],
        *,
        capability_ids: set[str] | None = None,
        exclude_capability_ids: set[str] | None = None,
    ) -> None:
        selected = self._select_skill_reservations(
            reservations,
            capability_ids=capability_ids,
            exclude_capability_ids=exclude_capability_ids,
        )
        for cap_id, reservation_id in selected.items():
            self.capabilities.restore_reserved_use(
                reservation_id,
                restored_by="skill",
                reason="one-time skill permission restored before commit",
            )
            reservations.pop(cap_id, None)

    def _select_skill_reservations(
        self,
        reservations: dict[str, str],
        *,
        capability_ids: set[str] | None,
        exclude_capability_ids: set[str] | None,
    ) -> dict[str, str]:
        return {
            cap_id: reservation_id
            for cap_id, reservation_id in reservations.items()
            if (capability_ids is None or cap_id in capability_ids)
            and (exclude_capability_ids is None or cap_id not in exclude_capability_ids)
        }

    def _decision_consume_ids(self, decisions: Iterable[CapabilityDecision]) -> set[str]:
        return {str(decision.consume_capability_id) for decision in decisions if decision.consume_capability_id is not None}

    def _require_process_admin_if_cross_actor(self, actor: str, pid: str) -> CapabilityDecision | None:
        if actor == pid:
            return None
        return self.capabilities.require(actor, f"process:{pid}", CapabilityRight.ADMIN, consume=False)

    def _require_trusted_global_source(self, source: str, package_sha256: str) -> None:
        if not self.config.skills.global_requires_trust:
            return
        if package_sha256 in set(self.config.skills.trusted_global_package_sha256):
            return
        if self.store.is_skill_trusted(source_type="global", source=source, package_sha256=package_sha256):
            return
        raise CapabilityDenied(f"global skill source is not trusted: {source} sha256={package_sha256}")

    def _normalize_global_path(self, path: str | Path) -> tuple[Path, str]:
        skill_md = self._resolve_host_skill_md(path)
        selected = skill_md.parent.resolve()
        roots = [Path(root).expanduser().resolve() for root in self.config.skills.global_dirs]
        for root in roots:
            try:
                relative = selected.relative_to(root)
            except ValueError:
                continue
            return selected, relative.as_posix()
        raise CapabilityDenied(f"global skill path is outside configured global_dirs: {selected}")

    def _get_skill(self, skill_id: str) -> tuple[SkillPackage, dict[str, Any]]:
        found = self.store.get_skill(skill_id)
        if found is None:
            raise NotFound(f"skill not found: {skill_id}")
        return found

    def _skill_snapshot(self, skill: SkillPackage) -> dict[str, Any]:
        return dict(to_jsonable(skill))

    def _skill_for_loaded_record(self, skill_id: str, loaded: Any) -> SkillPackage:
        if not isinstance(loaded, dict) or "package_snapshot" not in loaded:
            # Legacy in-memory rows did not carry package snapshots. New
            # activations always do, which prevents registry replacement from
            # mutating already loaded prompt/resources.
            skill, _metadata = self._get_skill(skill_id)
            return skill
        snapshot = loaded.get("package_snapshot")
        if not isinstance(snapshot, dict):
            raise ValidationError(f"loaded skill snapshot must be an object: {skill_id}")
        skill = self._package_from_snapshot(snapshot, context=f"loaded skill {skill_id}")
        if skill.skill_id != skill_id:
            raise ValidationError(f"loaded skill snapshot id mismatch: {skill.skill_id} != {skill_id}")
        expected_sha = str(loaded.get("package_sha256") or "")
        if expected_sha and skill.package_sha256 != expected_sha:
            raise ValidationError(
                f"loaded skill snapshot hash mismatch for {skill_id}: {skill.package_sha256} != {expected_sha}"
            )
        return skill

    def _package_from_snapshot(self, data: dict[str, Any], *, context: str) -> SkillPackage:
        try:
            package = SkillPackage(
                schema_version=int(data.get("schema_version", self.config.skills.schema_version)),
                skill_id=str(data["skill_id"]),
                name=str(data["name"]),
                description=str(data.get("description", "")),
                instructions=str(data.get("instructions", "")),
                version=str(data.get("version", "v0")),
                license=str(data.get("license", "")),
                compatibility=str(data.get("compatibility", "")),
                metadata={str(key): str(value) for key, value in self._mapping(data.get("metadata"), "metadata").items()},
                allowed_tools=self._string_list(data.get("allowed_tools"), "allowed_tools"),
                actions=[ActionSchema(**dict(item)) for item in self._list(data.get("actions"), "actions")],
                jit_tools=[JitToolSpec(**dict(item)) for item in self._list(data.get("jit_tools"), "jit_tools")],
                required_capabilities=[
                    dict(item) for item in self._list(data.get("required_capabilities"), "required_capabilities")
                ],
                resources=[SkillResource(**dict(item)) for item in self._list(data.get("resources"), "resources")],
                package_sha256=str(data.get("package_sha256", "")),
                diagnostics=self._string_list(data.get("diagnostics"), "diagnostics"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"invalid {context} package snapshot: {exc}") from exc
        actual_sha = self._package_hash(package)
        if package.package_sha256 and package.package_sha256 != actual_sha:
            raise ValidationError(f"invalid {context} package snapshot hash")
        if not package.package_sha256:
            package = self._replace_package_hash(package, actual_sha)
        self._validate_package(package)
        return package

    def _skill_summary(self, skill: SkillPackage, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "version": skill.version,
            "description": skill.description,
            "allowed_tools": list(skill.allowed_tools),
            "actions": [action.name for action in skill.actions],
            "jit_tools": [tool.name for tool in skill.jit_tools],
            "required_capabilities": list(skill.required_capabilities),
            "source_type": metadata.get("source_type"),
            "source": metadata.get("source"),
            "package_sha256": metadata.get("package_sha256") or skill.package_sha256,
            "registered": bool(metadata.get("registered_by")),
            "registered_by": metadata.get("registered_by"),
        }

    def _jit_summary(self, tool: JitToolSpec) -> dict[str, Any]:
        summary = {
            "name": tool.name,
            "description": tool.description,
            "source_path": tool.source_path,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "tests": tool.tests,
            "source_sha256": self._hash_text(tool.source),
        }
        if tool.timeout_s is not None:
            summary["timeout_s"] = tool.timeout_s
        return summary

    def _resource_summary(self, resource: SkillResource) -> dict[str, Any]:
        return {
            "path": resource.path,
            "kind": resource.kind,
            "size_bytes": resource.size_bytes,
            "sha256": resource.sha256,
        }

    def _prompt_resource_summaries(self, skill: SkillPackage, *, include_jit_catalog: bool) -> list[dict[str, Any]]:
        if include_jit_catalog:
            return [self._resource_summary(resource) for resource in skill.resources]
        hidden_paths = {"references/agent-libos/jit-tools.json"}
        hidden_paths.update(tool.source_path for tool in skill.jit_tools)
        return [
            self._resource_summary(resource)
            for resource in skill.resources
            if resource.path not in hidden_paths
        ]

    def _prompt_instructions(self, skill: SkillPackage) -> str:
        return skill.instructions[: self.config.skills.max_prompt_instruction_chars]

    def _coerce_package(self, skill: SkillPackage) -> SkillPackage:
        if isinstance(skill, SkillPackage):
            return skill
        raise ValidationError("skill registration requires a parsed SKILL.md package")

    def _coerce_action(self, value: Any) -> ActionSchema:
        if not isinstance(value, dict):
            raise ValidationError("actions entries must be mappings")
        allowed = {
            "name",
            "use_cases",
            "input_schema",
            "output_schema",
            "required_capabilities",
            "side_effects",
            "failure_modes",
            "examples",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValidationError(f"unknown Skill action fields: {unknown}")
        examples: list[dict[str, Any]] = []
        for item in self._list(value.get("examples"), "actions[].examples"):
            if not isinstance(item, dict):
                raise ValidationError("actions[].examples entries must be mappings")
            examples.append(dict(item))
        return ActionSchema(
            name=self._require_string(value.get("name"), "actions[].name"),
            use_cases=self._string_list(value.get("use_cases"), "actions[].use_cases"),
            input_schema=self._mapping(value.get("input_schema"), "actions[].input_schema"),
            output_schema=self._mapping(value.get("output_schema"), "actions[].output_schema"),
            required_capabilities=self._capability_specs(value.get("required_capabilities")),
            side_effects=self._string_list(value.get("side_effects"), "actions[].side_effects"),
            failure_modes=self._string_list(value.get("failure_modes"), "actions[].failure_modes"),
            examples=examples,
        )

    def _coerce_jit_tool(self, value: Any) -> JitToolSpec:
        if not isinstance(value, dict):
            raise ValidationError("jit_tools entries must be mappings")
        allowed = {
            "name",
            "description",
            "input_schema",
            "output_schema",
            "source_path",
            "tests",
            "metadata",
            "timeout_s",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValidationError(f"unknown Skill JIT tool fields: {unknown}")
        source_path = self._require_string(value.get("source_path"), "jit_tools[].source_path")
        self._validate_jit_script_path(source_path)
        tests: list[dict[str, Any]] = []
        for item in self._list(value.get("tests"), "jit_tools[].tests"):
            if not isinstance(item, dict):
                raise ValidationError("jit_tools[].tests entries must be mappings")
            tests.append(dict(item))
        name = self._require_string(value.get("name"), "jit_tools[].name")
        self._validate_jit_tool_name(name, "jit_tools[].name")
        input_schema = self._mapping(value.get("input_schema"), "jit_tools[].input_schema")
        output_schema = self._mapping(value.get("output_schema"), "jit_tools[].output_schema")
        self._validate_json_schema(input_schema or {"type": "object"}, "jit_tools[].input_schema")
        self._validate_json_schema(output_schema or {"type": "object"}, "jit_tools[].output_schema")
        return JitToolSpec(
            name=name,
            description=self._require_string(value.get("description"), "jit_tools[].description"),
            source_path=source_path,
            input_schema=input_schema,
            output_schema=output_schema,
            tests=tests,
            metadata=self._mapping(value.get("metadata"), "jit_tools[].metadata"),
            timeout_s=self._coerce_jit_timeout(value.get("timeout_s")),
        )

    def _coerce_jit_timeout(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError("jit_tools[].timeout_s must be a number")
        if isinstance(value, int):
            if value <= 0:
                raise ValidationError("jit_tools[].timeout_s must be finite and > 0")
            if value > self.config.tools.deno_timeout_hard_limit_s:
                raise ValidationError(
                    "jit_tools[].timeout_s exceeds tools.deno_timeout_hard_limit_s="
                    f"{self.config.tools.deno_timeout_hard_limit_s}"
                )
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValidationError("jit_tools[].timeout_s must be finite and > 0")
        if timeout > self.config.tools.deno_timeout_hard_limit_s:
            raise ValidationError(
                "jit_tools[].timeout_s exceeds tools.deno_timeout_hard_limit_s="
                f"{self.config.tools.deno_timeout_hard_limit_s}"
            )
        return timeout

    def _json_resource(self, resources: dict[str, SkillResource], path: str) -> Any:
        resource = resources.get(path)
        if resource is None:
            raise ValidationError(f"referenced skill metadata resource is missing: {path}")
        if resource.content is None:
            raise ValidationError(f"referenced skill metadata resource must be text: {path}")
        try:
            return bounded_json_loads(resource.content)
        except (ValueError, RecursionError) as exc:
            raise ValidationError(f"invalid JSON skill metadata resource {path}: {exc}") from exc

    def _frontmatter_reference_paths(self, frontmatter: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        metadata = dict(frontmatter.get("metadata") or {})
        for key in ["agent-libos.actions", "agent-libos.required-capabilities", "agent-libos.jit-tools"]:
            value = metadata.get(key)
            if value:
                paths.append(self._normalize_metadata_reference(value, key))
        return sorted(set(paths))

    def _normalize_metadata_reference(self, value: str, key: str) -> str:
        path = self._normalize_relative_resource_path(value)
        if not path.startswith("references/agent-libos/") or not path.endswith(".json"):
            raise ValidationError(f"{key} must point to references/agent-libos/*.json")
        return path

    def _validate_jit_script_path(self, path: str) -> None:
        normalized = self._normalize_relative_resource_path(path)
        if normalized != path:
            raise ValidationError(f"JIT source_path must be normalized: {path}")
        if not normalized.startswith("scripts/") or not normalized.endswith(".ts"):
            raise ValidationError("Skill JIT source_path must point to scripts/*.ts")

    def _validate_resource_path(self, path: str) -> None:
        normalized = self._normalize_relative_resource_path(path)
        if normalized != path:
            raise ValidationError(f"skill resource path must be normalized: {path}")
        if normalized == "SKILL.md":
            return
        if not any(normalized.startswith(f"{directory}/") for directory in self.config.skills.resource_dirs):
            raise ValidationError(f"skill resource must live under one of {self.config.skills.resource_dirs}: {path}")

    def _normalize_relative_resource_path(self, path: str) -> str:
        raw = os.fspath(path).replace("\\", "/").strip()
        if not raw or raw.startswith("/") or ":" in raw.split("/", 1)[0]:
            raise ValidationError(f"skill resource path must be relative: {path!r}")
        parts: list[str] = []
        for part in raw.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                raise ValidationError(f"skill resource path escapes package root: {path!r}")
            parts.append(part)
        if not parts:
            raise ValidationError("skill resource path cannot be empty")
        return "/".join(parts)

    def _workspace_package_paths(self, path: str) -> tuple[str, str]:
        normalized = self._normalize_relative_resource_path(path)
        if normalized.endswith("/SKILL.md"):
            return normalized[: -len("/SKILL.md")], normalized
        if normalized == "SKILL.md":
            return ".", normalized
        if normalized.endswith(".yaml") or normalized.endswith(".yml"):
            raise ValidationError("legacy YAML Skill manifests are not supported; use a SKILL.md package")
        return normalized, self._join_relative(normalized, "SKILL.md")

    def _join_relative(self, root: str, path: str) -> str:
        if root in {"", "."}:
            return path
        return f"{root.rstrip('/')}/{path}"

    def _resolve_host_skill_md(self, path: str | Path) -> Path:
        selected = Path(path).expanduser()
        if not selected.is_absolute():
            selected = Path.cwd() / selected
        selected = selected.resolve()
        if selected.suffix.lower() in {".yaml", ".yml"}:
            raise ValidationError("legacy YAML Skill manifests are not supported; use a SKILL.md package")
        if selected.is_dir():
            selected = selected / "SKILL.md"
        if selected.name != "SKILL.md":
            raise ValidationError("skill path must be a skill directory or SKILL.md")
        if not selected.exists() or not selected.is_file():
            raise NotFound(f"SKILL.md not found: {selected}")
        return selected

    def _source_type_for_host_path(self, path: Path) -> str:
        selected = path.expanduser().resolve()
        roots = [Path(root).expanduser().resolve() for root in self.config.skills.global_dirs]
        for root in roots:
            try:
                selected.relative_to(root)
            except ValueError:
                continue
            return "global"
        return "workspace"

    def _read_bytes_limited(self, path: Path, max_bytes: int) -> bytes:
        if not path.exists() or not path.is_file():
            raise NotFound(f"skill package file not found: {path}")
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValidationError(f"skill package file is not a regular file: {path}")
        if before.st_nlink > 1:
            raise ValidationError(f"skill package hard links are not supported: {path}")
        if before.st_size > max_bytes:
            raise ValidationError(f"skill package file exceeds limit {max_bytes}: {path}")
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValidationError(f"skill package symlinks are not supported: {path}") from exc
            raise
        with os.fdopen(fd, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ValidationError(f"skill package file is not a regular file: {path}")
            if opened.st_nlink > 1:
                raise ValidationError(f"skill package hard links are not supported: {path}")
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ValidationError(f"skill package file changed during read: {path}")
            raw = handle.read()
        if len(raw) > max_bytes:
            raise ValidationError(f"skill package file exceeds limit {max_bytes}: {path}")
        return raw

    def _package_hash(self, package: SkillPackage) -> str:
        payload = {
            "schema_version": package.schema_version,
            "skill_id": package.skill_id,
            "name": package.name,
            "description": package.description,
            "instructions_sha256": self._hash_text(package.instructions),
            "version": package.version,
            "license": package.license,
            "compatibility": package.compatibility,
            "metadata": package.metadata,
            "allowed_tools": package.allowed_tools,
            "actions": [asdict(action) for action in package.actions],
            "jit_tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "source_path": tool.source_path,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "source_sha256": self._hash_text(tool.source),
                    "tests": tool.tests,
                    "metadata": tool.metadata,
                    **(
                        {"timeout_s": tool.timeout_s}
                        if tool.timeout_s is not None
                        else {}
                    ),
                }
                for tool in package.jit_tools
            ],
            "required_capabilities": package.required_capabilities,
            "resources": [
                {
                    "path": resource.path,
                    "sha256": resource.sha256,
                    "size_bytes": resource.size_bytes,
                    "kind": resource.kind,
                    "content_sha256": self._resource_content_sha256(resource),
                }
                for resource in package.resources
            ],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _replace_package_hash(self, package: SkillPackage, package_sha256: str) -> SkillPackage:
        return SkillPackage(
            schema_version=package.schema_version,
            skill_id=package.skill_id,
            name=package.name,
            description=package.description,
            instructions=package.instructions,
            version=package.version,
            license=package.license,
            compatibility=package.compatibility,
            metadata=dict(package.metadata),
            allowed_tools=list(package.allowed_tools),
            actions=list(package.actions),
            jit_tools=list(package.jit_tools),
            required_capabilities=list(package.required_capabilities),
            resources=list(package.resources),
            package_sha256=package_sha256,
            diagnostics=list(package.diagnostics),
        )

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _validate_resource_content(self, resource: SkillResource) -> None:
        content = self._resource_content_bytes(resource)
        if len(content) != resource.size_bytes:
            raise ValidationError(f"skill resource size mismatch: {resource.path}")
        if hashlib.sha256(content).hexdigest() != resource.sha256:
            raise ValidationError(f"skill resource sha256 mismatch: {resource.path}")

    def _resource_content_sha256(self, resource: SkillResource) -> str:
        return hashlib.sha256(self._resource_content_bytes(resource)).hexdigest()

    def _resource_content_bytes(self, resource: SkillResource) -> bytes:
        if resource.kind == "text":
            if resource.content is None:
                raise ValidationError(f"text skill resource is missing content: {resource.path}")
            return resource.content.encode("utf-8")
        if resource.kind == "base64":
            if resource.content_base64 is None:
                raise ValidationError(f"base64 skill resource is missing content: {resource.path}")
            try:
                return base64.b64decode(resource.content_base64.encode("ascii"), validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValidationError(f"base64 skill resource content is invalid: {resource.path}") from exc
        raise ValidationError(f"unsupported skill resource kind: {resource.kind}")

    def _validate_source_type(self, source_type: str) -> str:
        if source_type not in _SOURCE_TYPES:
            raise ValidationError(f"unsupported skill source_type: {source_type}")
        return source_type

    def _list(self, value: Any, field: str) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError(f"{field} must be a list")
        return list(value)

    def _string_list(self, value: Any, field: str) -> list[str]:
        return [self._require_string(item, f"{field}[]") for item in self._list(value, field)]

    def _mapping(self, value: Any, field: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValidationError(f"{field} must be a mapping")
        return dict(value)

    def _metadata(self, value: Any) -> dict[str, str]:
        raw = self._mapping(value, "metadata")
        result: dict[str, str] = {}
        for key, item in raw.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise ValidationError("SKILL.md metadata must contain string keys and string values")
            result[key] = item
        return result

    def _capability_specs(self, value: Any) -> list[dict[str, Any]]:
        specs = self._list(value, "required_capabilities")
        normalized: list[dict[str, Any]] = []
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValidationError("capability spec entries must be mappings")
            item = dict(spec)
            self._validate_capability_spec(item)
            normalized.append(item)
        return normalized

    def _validate_capability_spec(self, spec: dict[str, Any]) -> None:
        resource = spec.get("resource")
        rights = spec.get("rights")
        if not isinstance(resource, str) or not resource:
            raise ValidationError("capability spec requires a non-empty resource")
        try:
            self.capabilities.parse_resource_pattern(resource)
        except CapabilityDenied as exc:
            raise ValidationError(str(exc)) from exc
        if not isinstance(rights, list) or not rights or not all(isinstance(right, str) and right for right in rights):
            raise ValidationError("capability spec requires a non-empty rights list")
        for right in rights:
            try:
                CapabilityRight(str(right))
            except ValueError as exc:
                raise ValidationError(f"unknown capability right: {right}") from exc
        constraints = spec.get("constraints")
        if constraints is not None and not isinstance(constraints, dict):
            raise ValidationError("capability spec constraints must be a mapping")

    def _require_string(self, value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{field} must be a non-empty string")
        return value.strip()

    def _optional_string(self, value: Any, field: str) -> str | None:
        if value is None:
            return None
        return self._require_string(value, field)

    def _validate_skill_name(self, value: str) -> None:
        self._validate_string_length(value, "name", self.config.skills.name_max_chars)
        if not _SKILL_NAME_PATTERN.match(value):
            raise ValidationError(f"skill name must use lowercase letters, digits, and hyphens: {value!r}")

    def _validate_tool_identifier(self, value: str, field: str, max_chars: int) -> None:
        self._validate_string_length(value, field, max_chars)
        if not _TOOL_NAME_PATTERN.match(value):
            raise ValidationError(f"{field} contains unsupported characters: {value!r}")

    def _validate_jit_tool_name(self, value: str, field: str) -> None:
        self._validate_string_length(value, field, OPENAI_TOOL_NAME_MAX_CHARS)
        if not is_openai_tool_name(value):
            raise ValidationError(
                f"{field} must match OpenAI tool name syntax [A-Za-z0-9_-]{{1,{OPENAI_TOOL_NAME_MAX_CHARS}}}: {value!r}"
            )

    def _validate_json_schema(self, schema: dict[str, Any], field: str) -> None:
        if not isinstance(schema, dict):
            raise ValidationError(f"{field} must be a JSON schema object")
        try:
            jsonschema_validator_for(schema).check_schema(schema)
        except JsonSchemaSchemaError as exc:
            raise ValidationError(f"{field} is not a valid JSON schema: {exc.message}") from exc

    def _validate_string_length(self, value: str, field: str, max_chars: int) -> None:
        if len(value) > max_chars:
            raise ValidationError(f"{field} exceeds max length {max_chars}")
        if any(ord(char) < 32 for char in value):
            raise ValidationError(f"{field} contains control characters")

    def _process_uses_multiplexed_jit(self, process: Any) -> bool:
        image = self._images.get(process.image_id)
        return getattr(image, "jit_tool_exposure", None) == JIT_TOOL_EXPOSURE_MULTIPLEXED
