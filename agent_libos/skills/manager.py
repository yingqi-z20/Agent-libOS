from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.validators import validator_for as jsonschema_validator_for

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    CapabilityRight,
    EventType,
    JIT_MULTIPLEXER_TOOL_NAME,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    OPENAI_TOOL_NAME_MAX_CHARS,
    ToolCandidateStatus,
    is_openai_tool_name,
)
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, NotFound, ValidationError
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.skills.schema import ActionSchema, JitToolSpec, LoadedSkill, SkillPackage, SkillResource
from agent_libos.storage import SQLiteStore
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, to_jsonable
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


class SkillManager:
    """Capability-controlled primitive for standard Agent Skill packages.

    Skills use the standard package shape rooted at ``SKILL.md``. Activation
    changes only prompt materialization and process-local tool visibility; all
    external authority still comes from capability-checked primitives.
    """

    def __init__(
        self,
        store: SQLiteStore,
        capabilities: CapabilityManager,
        audit: AuditManager,
        events: EventBus,
        *,
        config: AgentLibOSConfig | None = None,
        human: Any | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.store = store
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.human = human
        self.runtime: Any | None = None

    def bind_runtime(self, runtime: Any) -> None:
        self.runtime = runtime

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
            self._require_skill_right(actor, spec.skill_id, CapabilityRight.WRITE)
        existing = self.store.get_skill(spec.skill_id)
        if existing is not None and not replace:
            raise ValidationError(f"skill already registered: {spec.skill_id}")
        now = utc_now()
        if spec.package_sha256 != selected_sha:
            spec = self._replace_package_hash(spec, selected_sha)
        self.store.upsert_skill(
            spec,
            source_type=selected_source_type,
            source=selected_source,
            package_sha256=selected_sha,
            registered_by=actor,
            created_at=now,
        )
        self.capabilities.consume_allow_once(actor, self.resource_for(spec.skill_id), CapabilityRight.WRITE, "skill")
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
        if require_capability:
            self._require_skill_right(pid, package.skill_id, CapabilityRight.WRITE)
        result = self.register_skill_package(
            package,
            actor=pid,
            replace=replace,
            require_capability=False,
            source_type="workspace",
            source=source,
            package_sha256=package.package_sha256,
        )
        self.capabilities.consume_allow_once(pid, self.resource_for(package.skill_id), CapabilityRight.WRITE, "skill")
        return result

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
            self._require_skill_rights(pid, package.skill_id, [CapabilityRight.WRITE, CapabilityRight.EXECUTE])
        self.register_skill_package(
            package,
            actor=pid,
            replace=replace,
            require_capability=False,
            source_type="workspace",
            source=source,
            package_sha256=package.package_sha256,
        )
        result = self.activate_skill(pid, package.skill_id, actor=pid, require_capability=False)
        self.capabilities.consume_allow_once(pid, self.resource_for(package.skill_id), CapabilityRight.WRITE, "skill")
        self.capabilities.consume_allow_once(pid, self.resource_for(package.skill_id), CapabilityRight.EXECUTE, "skill")
        return {**result, "source": source, "registered": True}

    def discover_skills(
        self,
        text: str | None = None,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if require_capability and actor is not None:
            self.capabilities.require(actor, self.config.skills.registry_resource, CapabilityRight.READ)
        selected_limit = self.config.skills.discover_limit if limit is None else limit
        registered = [self._skill_summary(skill, metadata) for skill, metadata in self.store.list_skills(text=text, limit=selected_limit)]
        if actor is not None:
            return registered
        discovered = self._discover_host_skill_catalog(text=text, limit=selected_limit)
        seen = {item["skill_id"] for item in registered}
        for item in discovered:
            if item["skill_id"] not in seen:
                registered.append(item)
                seen.add(item["skill_id"])
            if len(registered) >= selected_limit:
                break
        return registered

    def inspect_skill(
        self,
        skill_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        skill, metadata = self._get_skill(skill_id)
        if require_capability and actor is not None:
            self._require_skill_right(actor, skill_id, CapabilityRight.READ)
        return {
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

    def prompt_context(self, pid: str) -> list[dict[str, Any]]:
        process = self.store.get_process(pid)
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

    def activate_skill(
        self,
        pid: str,
        skill_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        selected_actor = actor or pid
        skill, metadata = self._get_skill(skill_id)
        if require_capability:
            self._require_skill_right(selected_actor, skill_id, CapabilityRight.EXECUTE)
            self._require_process_admin_if_cross_actor(selected_actor, pid)
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        self._validate_loadable(pid, skill, process.tool_table)
        existing_handles = self._resolve_existing_tools(skill.allowed_tools)
        jit_handles = self._register_jit_tools(pid, skill)
        tool_ids = {name: handle.tool_id for name, handle in existing_handles.items()}
        jit_tool_ids = {name: handle.tool_id for name, handle in jit_handles.items()}
        updated_table = dict(process.tool_table)
        for name, handle in {**existing_handles, **jit_handles}.items():
            updated_table[name] = handle.tool_id
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
        )
        process.tool_table = updated_table
        process.loaded_skills[skill.skill_id] = to_jsonable(loaded)
        process.updated_at = utc_now()
        self.store.update_process(process)
        self.capabilities.consume_allow_once(selected_actor, self.resource_for(skill_id), CapabilityRight.EXECUTE, "skill")
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
                "tool_ids": tool_ids,
                "jit_tool_ids": jit_tool_ids,
                "source": metadata.get("source"),
                "package_sha256": skill.package_sha256,
            },
        )
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
            self._require_skill_right(selected_actor, skill_id, CapabilityRight.EXECUTE)
            self._require_process_admin_if_cross_actor(selected_actor, pid)
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        loaded = process.loaded_skills.get(skill_id)
        if loaded is None:
            raise NotFound(f"skill is not loaded in process {pid}: {skill_id}")
        tool_ids = dict(loaded.get("tool_ids", {})) if isinstance(loaded, dict) else {}
        jit_tool_ids = dict(loaded.get("jit_tool_ids", {})) if isinstance(loaded, dict) else {}
        removed: list[str] = []
        for name, tool_id in {**tool_ids, **jit_tool_ids}.items():
            if process.tool_table.get(name) == tool_id:
                process.tool_table.pop(name, None)
                removed.append(name)
        process.loaded_skills.pop(skill_id, None)
        process.updated_at = utc_now()
        self.store.update_process(process)
        self.capabilities.consume_allow_once(selected_actor, self.resource_for(skill_id), CapabilityRight.EXECUTE, "skill")
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
            decision={"skill_id": skill_id, "removed_tools": sorted(removed)},
        )
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
        process = self.store.get_process(pid)
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
        if require_capability:
            self.capabilities.require(actor, self.config.skills.trust_resource, CapabilityRight.ADMIN)
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
        if require_capability:
            self.capabilities.require(actor, self.config.skills.trust_resource, CapabilityRight.ADMIN)
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
        runtime = self._runtime()
        cwd = runtime.process.working_directory(pid)
        package_root, skill_md_path = self._workspace_package_paths(path)
        read = runtime.filesystem.read_text(
            pid,
            skill_md_path,
            max_bytes=self.config.skills.skill_md_max_bytes,
            cwd=cwd,
        )
        frontmatter, body = self._parse_skill_markdown(read.content, expected_dir_name=Path(package_root).name)
        _target, workspace_package_root = runtime.filesystem.resolve_path(package_root, cwd=cwd)
        references = self._frontmatter_reference_paths(frontmatter)
        raw_resources: dict[str, bytes] = {"SKILL.md": read.content.encode("utf-8")}
        for ref in references:
            ref_read = runtime.filesystem.read_text(
                pid,
                self._join_relative(package_root, ref),
                max_bytes=self.config.skills.resource_read_max_bytes,
                cwd=cwd,
            )
            raw_resources[ref] = ref_read.content.encode("utf-8")
        jit_tools = self._load_jit_specs_from_resources(frontmatter, raw_resources)
        for tool in jit_tools:
            if tool.source_path not in raw_resources:
                script_read = runtime.filesystem.read_text(
                    pid,
                    self._join_relative(package_root, tool.source_path),
                    max_bytes=self.config.skills.max_jit_source_chars,
                    cwd=cwd,
                )
                raw_resources[tool.source_path] = script_read.content.encode("utf-8")
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
        runtime = self._runtime()
        max_files = self.config.skills.max_package_files
        visited_dirs: set[str] = set()

        def visit(directory: str) -> None:
            normalized_dir = directory.strip("/")
            if normalized_dir in visited_dirs:
                return
            visited_dirs.add(normalized_dir)
            if not self._has_read_authority(pid, runtime.filesystem.directory_resource_for_path(normalized_dir, cwd=None)):
                return
            try:
                listing = runtime.filesystem.read_directory(pid, normalized_dir, limit=max_files, cwd=None)
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
                if not self._has_read_authority(pid, runtime.filesystem.resource_for_path(entry.path, cwd=None)):
                    continue
                read = runtime.filesystem.read_bytes(
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
        data = json.loads(raw.decode("utf-8"))
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

    def _discover_host_skill_catalog(self, *, text: str | None, limit: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        roots = [Path("skills"), Path(".agents") / "skills", Path(".claude") / "skills"]
        roots.extend(Path(root).expanduser() for root in self.config.skills.global_dirs)
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
                result.append(summary)
                if len(result) >= limit:
                    return result
        return result

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
        seen_paths: set[str] = set()
        total_bytes = 0
        for resource in skill.resources:
            self._validate_resource_path(resource.path)
            if resource.path in seen_paths:
                raise ValidationError(f"duplicate skill resource path: {resource.path}")
            seen_paths.add(resource.path)
            total_bytes += resource.size_bytes
        if total_bytes > defaults.package_max_bytes:
            raise ValidationError(f"skill package exceeds package_max_bytes={defaults.package_max_bytes}")
        for tool in skill.jit_tools:
            self._validate_jit_tool_name(tool.name, "jit_tools[].name")
            self._validate_jit_script_path(tool.source_path)
            if len(tool.source) > defaults.max_jit_source_chars:
                raise ValidationError(f"JIT source for {tool.name} exceeds max_jit_source_chars={defaults.max_jit_source_chars}")
        for spec in skill.required_capabilities:
            self._validate_capability_spec(spec)

    def _validate_loadable(self, pid: str, skill: SkillPackage, process_tool_table: dict[str, str]) -> None:
        runtime = self._runtime()
        static_names = {row["name"] for row in runtime.tools.list() if not bool(row.get("ephemeral"))}
        process = runtime.process.get(pid)
        image = runtime.images.get(process.image_id) if process is not None else None
        multiplexed_jit = getattr(image, "jit_tool_exposure", None) == JIT_TOOL_EXPOSURE_MULTIPLEXED
        for name in skill.allowed_tools:
            runtime.tools.resolve(name)
        for tool in skill.jit_tools:
            if multiplexed_jit and tool.name == JIT_MULTIPLEXER_TOOL_NAME:
                raise ValidationError(f"{JIT_MULTIPLEXER_TOOL_NAME} is reserved by multiplexed JIT tool exposure")
            if tool.name in process_tool_table:
                raise ValidationError(f"process already has a tool named: {tool.name}")
            if tool.name in static_names:
                raise ValidationError(f"JIT skill tool cannot shadow static tool: {tool.name}")

    def _resolve_existing_tools(self, names: list[str]) -> dict[str, Any]:
        runtime = self._runtime()
        return {name: runtime.tools.resolve(name) for name in names}

    def _register_jit_tools(self, pid: str, skill: SkillPackage) -> dict[str, Any]:
        runtime = self._runtime()
        prepared: list[tuple[JitToolSpec, str]] = []
        for jit in skill.jit_tools:
            candidate_id = runtime.tools.propose(
                pid,
                {
                    "name": jit.name,
                    "description": jit.description,
                    "input_schema": jit.input_schema,
                    "output_schema": jit.output_schema,
                    "metadata": {"skill_id": skill.skill_id, "source_path": jit.source_path, **jit.metadata},
                },
                source_code=jit.source,
                tests=jit.tests,
            )
            validation = runtime.tools.validate(candidate_id, pid=pid)
            if not validation.ok:
                raise ValidationError(f"JIT skill tool {jit.name} failed validation: {'; '.join(validation.errors)}")
            candidate = runtime.store.get_tool_candidate(candidate_id)
            if candidate is None:
                raise NotFound(f"tool candidate not found after validation: {candidate_id}")
            candidate.status = ToolCandidateStatus.VALIDATED
            runtime.store.update_tool_candidate(candidate)
            prepared.append((jit, candidate_id))
        handles: dict[str, Any] = {}
        try:
            for jit, candidate_id in prepared:
                handles[jit.name] = runtime.tools.register(pid, candidate_id, approver=f"skill:{skill.skill_id}")
        except Exception:
            self._remove_registered_jit_tools(pid, handles)
            raise
        return handles

    def _remove_registered_jit_tools(self, pid: str, handles: dict[str, Any]) -> None:
        if not handles:
            return
        runtime = self._runtime()
        process = runtime.store.get_process(pid)
        if process is not None:
            for name, handle in handles.items():
                if process.tool_table.get(name) == handle.tool_id:
                    process.tool_table.pop(name, None)
            process.updated_at = utc_now()
            runtime.store.update_process(process)
        for handle in handles.values():
            getattr(runtime.tools, "_jit_sources", {}).pop(handle.tool_id, None)
            getattr(runtime.tools, "_handles", {}).pop(handle.tool_id, None)
            names = getattr(runtime.tools, "_tool_ids_by_name", None)
            if names is not None and names.get(handle.name) == handle.tool_id:
                names.pop(handle.name, None)

    def _require_skill_right(self, actor: str, skill_id: str, right: CapabilityRight) -> None:
        self._require_skill_rights(actor, skill_id, [right])

    def _require_skill_rights(self, actor: str, skill_id: str, rights: Iterable[CapabilityRight]) -> None:
        resource = self.resource_for(skill_id)
        missing: list[str] = []
        for right in rights:
            policy = self.capabilities.permission_policy(actor, resource, right)
            if policy in {CapabilityManager.ALWAYS_ALLOW, CapabilityManager.ALLOW_ONCE}:
                continue
            missing.append(str(right))
        if not missing:
            return
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

    def _require_process_admin_if_cross_actor(self, actor: str, pid: str) -> None:
        if actor == pid:
            return
        self.capabilities.require(actor, f"process:{pid}", CapabilityRight.ADMIN)

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
        return {
            "name": tool.name,
            "description": tool.description,
            "source_path": tool.source_path,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "tests": tool.tests,
            "source_sha256": self._hash_text(tool.source),
        }

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
        allowed = {"name", "description", "input_schema", "output_schema", "source_path", "tests", "metadata"}
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
        )

    def _json_resource(self, resources: dict[str, SkillResource], path: str) -> Any:
        resource = resources.get(path)
        if resource is None:
            raise ValidationError(f"referenced skill metadata resource is missing: {path}")
        if resource.content is None:
            raise ValidationError(f"referenced skill metadata resource must be text: {path}")
        try:
            return json.loads(resource.content)
        except json.JSONDecodeError as exc:
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

    def _validate_source_type(self, source_type: str) -> str:
        if source_type not in _SOURCE_TYPES:
            raise ValidationError(f"unsupported skill source_type: {source_type}")
        return source_type

    def _runtime(self) -> Any:
        if self.runtime is None:
            raise RuntimeError("SkillManager is not bound to a Runtime")
        return self.runtime

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
        runtime = self.runtime
        if runtime is None:
            return False
        image = getattr(runtime, "images", {}).get(process.image_id)
        return getattr(image, "jit_tool_exposure", None) == JIT_TOOL_EXPOSURE_MULTIPLEXED
