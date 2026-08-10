from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import itertools
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
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    SkillPackageChanged,
    ValidationError,
)
from agent_libos.ports import AuditPort, EventPort, RuntimePublicationReceiptRecorder
from agent_libos.skills.builtin_catalog import (
    BUILTIN_SKILL_PREFIX,
    BuiltinSkillCatalog,
    get_builtin_skill_catalog,
)
from agent_libos.skills.schema import ActionSchema, JitToolSpec, LoadedSkill, SkillPackage, SkillResource
from agent_libos.storage import UnitOfWork
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.secure_host_files import (
    SecureDirectoryGuard,
    SecureFileChanged,
    SecureFileLimitExceeded,
    SecureFileReadUnavailable,
    StablePathSnapshot,
    open_secure_directory,
    open_secure_file,
    read_stable_file_limited,
    stable_identity_available,
)
from agent_libos.utils.serde import bounded_json_loads, dumps, to_jsonable
from agent_libos.utils.skill_search import (
    skill_metadata_exact_match,
    skill_metadata_search_score,
)
from agent_libos.utils.yaml_loader import load_yaml_mapping

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SKILL_NAME_MAX_CHARS = 64
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
_SOURCE_TYPES = {"workspace", "global", "runtime"}
_FRONTMATTER_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
_AGENT_LIBOS_METADATA_KEYS = {
    "agent-libos.version",
    "agent-libos.actions",
    "agent-libos.required-capabilities",
    "agent-libos.jit-tools",
}
_BUILTIN_PROJECTION_RECEIPT_ACTION = "skill.builtin_projection.receipt"
_BUILTIN_PROJECTION_RECEIPT_SCHEMA_VERSION = 1
_BUILTIN_PROJECTION_RECEIPT_FIELD = "builtin_projection_receipt_id"
_SKILL_PACKAGE_READ_CHUNK_BYTES = 64 * 1024
_LOADED_SKILL_FIELDS = frozenset(LoadedSkill.__dataclass_fields__)
_ACTIVATED_SKILL_RESULT_FIELDS = frozenset(
    {
        "pid",
        "skill_id",
        "name",
        "version",
        "tool_names",
        "tool_ids",
        "jit_tool_ids",
        "instructions_hash",
        "package_sha256",
    }
)


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


@dataclass(slots=True)
class _SkillPackageTraversalBudget:
    """Aggregate directory/depth budget shared by one package traversal."""

    max_directories: int
    max_depth: int
    directories: int = 0

    def charge_directory(self, *, depth: int) -> None:
        if depth > self.max_depth:
            raise ValidationError(
                "skill package exceeds max_package_depth="
                f"{self.max_depth}"
            )
        if self.directories >= self.max_directories:
            raise ValidationError(
                "skill package exceeds max_package_directories="
                f"{self.max_directories}"
            )
        self.directories += 1


@dataclass(frozen=True, slots=True)
class _WorkspaceSkillManifest:
    """Bounded immutable SKILL.md snapshot used for pre-authorization."""

    requested_path: str
    cwd: str
    package_root: str
    skill_md_path: str
    workspace_package_root: str
    raw_skill: bytes
    frontmatter: dict[str, Any]
    body: str
    total_bytes: int

    @property
    def skill_id(self) -> str:
        return str(self.frontmatter["name"])


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
        self._builtin_catalog: BuiltinSkillCatalog = get_builtin_skill_catalog()
        self._validate_builtin_registry_boundary()

    def resource_for(self, skill_id: str) -> str:
        return f"skill:{skill_id}"

    def builtin_skill_for_tool(self, tool_name: str) -> str | None:
        """Return the unique built-in guidance Skill for a static tool."""

        return self._builtin_catalog.skill_for_tool(str(tool_name))

    def _validate_builtin_registry_boundary(self) -> None:
        for package in self._builtin_catalog.list():
            # Re-validate immutable assets against the active runtime limits,
            # not only the package-distribution defaults used by the catalog
            # loader. A stricter Host configuration must fail before any
            # built-in snapshot can be published.
            self._validate_package(package)
        collisions = [
            package.skill_id
            for package, _metadata in self.store.list_skills(limit=None)
            if self._builtin_catalog.is_builtin_id(package.skill_id)
        ]
        if collisions:
            raise ValidationError(
                "registered Skills collide with reserved built-in ids: "
                + ", ".join(sorted(collisions))
            )

    def trust_resource(self, package_sha256: str = "*") -> str:
        return self.config.skills.trust_resource if package_sha256 == "*" else f"skill_trust:{package_sha256}"

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

    @_with_registry_lifecycle_lock
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
        if self._builtin_catalog.is_builtin_id(spec.skill_id):
            raise ValidationError(
                f"Skill id uses the reserved built-in prefix {BUILTIN_SKILL_PREFIX!r}: {spec.skill_id}"
            )
        actual_sha = self._package_hash(spec)
        for label, claimed_sha in (
            ("SkillPackage.package_sha256", spec.package_sha256),
            ("package_sha256", package_sha256),
        ):
            if claimed_sha and claimed_sha != actual_sha:
                raise ValidationError(
                    f"{label} does not match the Skill package content hash"
                )
        selected_source_type = self._validate_source_type(source_type)
        selected_source = source or selected_source_type
        selected_sha = actual_sha
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
        package, source = self._load_package_from_host_path(path)
        _, source_id = self._normalize_global_source(source)
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
        package, source = self._load_package_from_host_path(path)
        absolute, source_id = self._normalize_global_source(source)
        return {
            "path": str(absolute),
            "source": source_id,
            "package_sha256": package.package_sha256,
            "skill_id": package.skill_id,
            "bytes": sum(resource.size_bytes for resource in package.resources),
        }

    @_with_registry_lifecycle_lock
    def register_skill_from_workspace_path(
        self,
        pid: str,
        path: str,
        *,
        replace: bool = False,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        if not require_capability:
            _package, _source, registration = self._load_and_register_workspace_skill(
                pid,
                path,
                replace=replace,
            )
            return registration
        manifest = self._read_workspace_skill_manifest(pid, path)
        decisions = self._require_skill_right(
            pid,
            manifest.skill_id,
            CapabilityRight.WRITE,
        )
        with self.capabilities.authority_transaction(
            decisions,
            actor=pid,
            operation="workspace Skill package registration",
        ):
            _package, _source, registration = self._load_and_register_workspace_skill(
                pid,
                path,
                replace=replace,
                manifest=manifest,
            )
        return registration

    def _load_and_register_workspace_skill(
        self,
        pid: str,
        path: str,
        *,
        replace: bool,
        manifest: _WorkspaceSkillManifest | None = None,
    ) -> tuple[SkillPackage, str, dict[str, Any]]:
        package, source = self._load_package_from_workspace(
            pid,
            path,
            manifest=manifest,
        )
        registration = self.register_skill_package(
            package,
            actor=pid,
            replace=replace,
            require_capability=False,
            source_type="workspace",
            source=source,
            package_sha256=package.package_sha256,
        )
        return package, source, registration

    @_with_registry_lifecycle_lock
    def activate_skill_from_workspace_path(
        self,
        pid: str,
        path: str,
        *,
        replace: bool = False,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        if not require_capability:
            package, source, _registration = self._load_and_register_workspace_skill(
                pid,
                path,
                replace=replace,
            )
            result = self.activate_skill(
                pid,
                package.skill_id,
                actor=pid,
                require_capability=False,
                expected_package_sha256=package.package_sha256,
            )
            return {**result, "source": source, "registered": True}

        manifest = self._read_workspace_skill_manifest(pid, path)
        decisions = self._require_skill_rights(
            pid,
            manifest.skill_id,
            [CapabilityRight.WRITE, CapabilityRight.EXECUTE],
        )
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
            result, source = self._activate_workspace_skill_with_shared_authority(
                pid,
                path,
                replace=replace,
                manifest=manifest,
                decisions=decisions,
            )
        else:
            write_decisions = [
                decision
                for decision in decisions
                if decision.right == CapabilityRight.WRITE.value
            ]
            with self.capabilities.authority_transaction(
                write_decisions,
                actor=pid,
                operation="workspace Skill package registration",
            ):
                package, source, _registration = self._load_and_register_workspace_skill(
                    pid,
                    path,
                    replace=replace,
                    manifest=manifest,
                )
            result = self.activate_skill(
                pid,
                package.skill_id,
                actor=pid,
                require_capability=True,
                expected_package_sha256=package.package_sha256,
            )
        return {**result, "source": source, "registered": True}

    def _activate_workspace_skill_with_shared_authority(
        self,
        pid: str,
        path: str,
        *,
        replace: bool,
        manifest: _WorkspaceSkillManifest,
        decisions: list[CapabilityDecision],
    ) -> tuple[dict[str, Any], str]:
        activation_error: Exception | None = None
        result: dict[str, Any] | None = None
        source = manifest.package_root
        deferred_jit = _DeferredJitRegistryFinalization()
        with self._lifecycle_lock:
            try:
                with self.capabilities.authority_transaction(
                    decisions,
                    actor=pid,
                    operation="workspace skill registration and activation",
                ):
                    package, source, _registration = (
                        self._load_and_register_workspace_skill(
                            pid,
                            path,
                            replace=replace,
                            manifest=manifest,
                        )
                    )
                    try:
                        result = self.activate_skill(
                            pid,
                            package.skill_id,
                            actor=pid,
                            require_capability=False,
                            expected_package_sha256=package.package_sha256,
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
        return result, source

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
        """Return one uniformly filtered and bounded visible Skill page."""

        skills, has_more, _scope = self._discover_skills_window_with_scope(
            text,
            actor=actor,
            require_capability=require_capability,
            limit=limit,
        )
        return skills, has_more

    def discover_skills_result(
        self,
        text: str | None = None,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return discoverable Skills and the authority-bounded catalog scope."""

        skills, has_more, scope = self._discover_skills_window_with_scope(
            text,
            actor=actor,
            require_capability=require_capability,
            limit=limit,
        )
        return {"skills": skills, "catalog_scope": scope, "has_more": has_more}

    def skill_declares_any_tool(
        self,
        skill_id: str,
        tool_names: set[str] | frozenset[str],
    ) -> bool:
        """Return whether a catalog package declares any selected tool name.

        This source-neutral Host predicate is used only to validate mandatory
        control-flow activations before the normal activation path enforces
        package trust, process compatibility, and authority.
        """

        skill, _metadata = self._get_skill(skill_id)
        declared = {
            *skill.allowed_tools,
            *(tool.name for tool in skill.jit_tools),
        }
        return not declared.isdisjoint(tool_names)

    def _discover_skills_window_with_scope(
        self,
        text: str | None,
        *,
        actor: str | None,
        require_capability: bool,
        limit: int | None,
    ) -> tuple[list[dict[str, Any]], bool, str]:
        """Return one uniformly bounded page across every visible Skill source."""

        reservations: dict[str, str] = {}
        include_registered = (
            actor is None
            or not require_capability
            or self.capabilities.check(
                actor,
                self.config.skills.registry_resource,
                CapabilityRight.READ,
            )
        )
        if include_registered and require_capability and actor is not None:
            decision = self.capabilities.require(
                actor,
                self.config.skills.registry_resource,
                CapabilityRight.READ,
                consume=False,
            )
            reservations = self._reserve_skill_rights([decision], used_by="skill")
        try:
            selected_limit = self._bounded_discover_limit(limit)
            process = self.processes.get_process(actor) if actor is not None else None
            builtins = self._available_builtin_summaries(actor, text=text)
            if include_registered:
                registered, registered_has_more = (
                    self._registered_discovery_summaries(
                        process,
                        text=text,
                        limit=selected_limit,
                    )
                )
            else:
                registered, registered_has_more = [], False
            host_entries, host_has_more = self._host_discovery_summaries(
                actor,
                builtins=builtins,
                registered=registered,
                text=text,
                limit=selected_limit,
            )
            combined, exact_match = self._rank_discovery_summaries(
                [*builtins, *registered, *host_entries],
                text=text,
            )
            if exact_match:
                registered_has_more = False
                host_has_more = False
        except BaseException as exc:
            self._restore_skill_rights_after_failure(reservations, exc)
            raise
        self._commit_skill_rights(reservations)
        scope = "all_visible_sources" if include_registered else "visibility_limited"
        page: list[dict[str, Any]] = []
        for item in combined[:selected_limit]:
            visible = dict(item)
            visible.pop("_discovery_score", None)
            page.append(visible)
        return (
            page,
            (
                len(combined) > selected_limit
                or registered_has_more
                or host_has_more
            ),
            scope,
        )

    def _registered_discovery_summaries(
        self,
        process: Any | None,
        *,
        text: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        rows = self.store.list_skills(text=text, limit=limit + 1)
        result: list[dict[str, Any]] = []
        for skill, metadata in rows:
            summary = {
                **self._skill_summary(skill, metadata),
                "active": self._loaded_skill_matches_package(process, skill),
            }
            score = self._skill_discovery_score(summary, text)
            if score is None:
                continue
            summary["_discovery_score"] = score
            result.append(summary)
        return result, len(rows) > limit

    def _host_discovery_summaries(
        self,
        actor: str | None,
        *,
        builtins: list[dict[str, Any]],
        registered: list[dict[str, Any]],
        text: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        if actor is not None:
            return [], False
        seen = {item["skill_id"] for item in (*builtins, *registered)}
        entries = self._discover_host_skill_catalog(
            text=text,
            limit=limit + 1,
            exclude_skill_ids=seen,
        )
        for item in entries:
            item["active"] = False
        return entries, len(entries) > limit

    @staticmethod
    def _rank_discovery_summaries(
        summaries: list[dict[str, Any]],
        *,
        text: str | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        ranked = sorted(
            summaries,
            key=lambda item: (
                -int(item.get("_discovery_score") or 0),
                str(item.get("name") or "").casefold(),
                str(item.get("skill_id") or ""),
            ),
        )
        exact = [
            item
            for item in ranked
            if skill_metadata_exact_match(
                skill_id=str(item.get("skill_id") or ""),
                name=str(item.get("name") or ""),
                text=text,
            )
        ]
        return (exact, True) if exact else (ranked, False)

    def _available_builtin_summaries(
        self,
        actor: str | None,
        *,
        text: str | None = None,
    ) -> list[dict[str, Any]]:
        process = self.processes.get_process(actor) if actor is not None else None
        result: list[dict[str, Any]] = []
        for skill in self._builtin_catalog.list():
            if process is not None and not self._builtin_supported_by_process(skill, process):
                continue
            metadata = self._builtin_catalog.metadata(skill.skill_id)
            assert metadata is not None
            summary = {
                **self._skill_summary(skill, metadata),
                "active": self._loaded_skill_matches_package(process, skill),
            }
            score = self._skill_discovery_score(summary, text)
            if score is None:
                continue
            summary["_discovery_score"] = score
            result.append(summary)
        return result

    @staticmethod
    def _skill_discovery_score(
        summary: Mapping[str, Any],
        text: str | None,
    ) -> int | None:
        return skill_metadata_search_score(
            skill_id=str(summary.get("skill_id") or ""),
            name=str(summary.get("name") or ""),
            description=str(summary.get("description") or ""),
            text=text,
        )

    def _loaded_skill_is_trusted(self, process: Any | None, skill_id: str) -> bool:
        if process is None or skill_id not in process.loaded_skills:
            return False
        loaded = process.loaded_skills[skill_id]
        try:
            if self._builtin_catalog.is_builtin_id(skill_id):
                return self._loaded_builtin_projection_is_trusted(
                    skill_id,
                    loaded,
                    process,
                )
            activation_kind = (
                loaded.get("activation_kind", "registered")
                if isinstance(loaded, dict)
                else "registered"
            )
            if activation_kind != "registered":
                return False
            self._skill_for_loaded_record(skill_id, loaded)
            return True
        except ValidationError:
            return False

    def _loaded_skill_matches_package(
        self,
        process: Any | None,
        skill: SkillPackage,
    ) -> bool:
        """Return whether the trusted loaded snapshot is this catalog content."""

        if not self._loaded_skill_is_trusted(process, skill.skill_id):
            return False
        assert process is not None
        loaded = process.loaded_skills.get(skill.skill_id)
        if not isinstance(loaded, dict):
            return False
        loaded_sha256 = loaded.get("package_sha256")
        return (
            isinstance(loaded_sha256, str)
            and bool(loaded_sha256)
            and loaded_sha256 == skill.package_sha256
        )

    def _builtin_supported_by_process(self, skill: SkillPackage, process: Any) -> bool:
        image = self._images.get(process.image_id)
        if image is None:
            return False
        image_default_tools = set(image.default_tools)
        for name in skill.allowed_tools:
            if name not in image_default_tools:
                return False
            configured_tool_id = process.tool_table.get(name)
            if configured_tool_id is None:
                return False
            try:
                handle = self._tools.resolve(name)
            except (NotFound, ValidationError):
                return False
            if str(handle.tool_id) != str(configured_tool_id):
                return False
        return True

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
        decisions: list[CapabilityDecision] = []
        reservations: dict[str, str] = {}
        if (
            require_capability
            and actor is not None
            and self._builtin_catalog.metadata(skill_id) is None
        ):
            decisions = self._require_skill_right(actor, skill_id, CapabilityRight.READ)
            reservations = self._reserve_skill_rights(decisions, used_by="skill")
        try:
            # Keep reauthorization, lookup, response construction, and
            # finite-use settlement under one store transaction.  This
            # preserves durable restored-reservation evidence on failure while
            # preventing unlimited-grant revoke races from reopening the
            # absent-vs-inaccessible identity oracle.
            with self.capabilities.store.transaction():
                for decision in decisions:
                    self._reauthorize_skill_read_decision(
                        decision,
                        reservations,
                    )
                skill, metadata = self._get_skill(skill_id)
                result = {
                    **self._skill_summary(skill, metadata),
                    "instructions": self._prompt_instructions(skill),
                    "allowed_tools": list(skill.allowed_tools),
                    "actions": [asdict(action) for action in skill.actions],
                    "jit_tools": [self._jit_summary(tool) for tool in skill.jit_tools],
                    "required_capabilities": list(skill.required_capabilities),
                    "metadata": dict(skill.metadata),
                    "resources": [
                        self._resource_summary(resource)
                        for resource in skill.resources
                    ],
                    "license": skill.license,
                    "compatibility": skill.compatibility,
                    "diagnostics": list(skill.diagnostics),
                }
                self._commit_skill_rights(reservations)
        except BaseException as exc:
            self._restore_skill_rights_after_failure(reservations, exc)
            raise
        return result

    def prompt_context(self, pid: str) -> list[dict[str, Any]]:
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        include_jit_catalog = not self._process_uses_multiplexed_jit(process)
        result: list[dict[str, Any]] = []
        for skill_id, loaded in process.loaded_skills.items():
            try:
                activation_kind = (
                    loaded.get("activation_kind", "registered")
                    if isinstance(loaded, dict)
                    else "registered"
                )
                if activation_kind == "builtin_projection":
                    skill, _ = self._validate_loaded_builtin_projection_record(
                        skill_id,
                        loaded,
                    )
                    if not self.builtin_projection_supported_by_process(
                        process,
                        skill_id,
                        loaded,
                    ):
                        raise ValidationError(
                            "built-in loaded Skill escapes current image default_tools "
                            f"or tool table: {skill_id}"
                        )
                elif activation_kind != "registered":
                    raise ValidationError(
                        f"unknown loaded Skill activation_kind: {activation_kind}"
                    )
                elif self._builtin_catalog.is_builtin_id(skill_id):
                    raise ValidationError(
                        f"built-in loaded Skill is missing trusted projection provenance: {skill_id}"
                    )
                else:
                    skill = self._skill_for_loaded_record(skill_id, loaded)
            except ValidationError as exc:
                result.append(
                    {
                        "skill_id": skill_id,
                        "invalid_snapshot": True,
                        "error": str(exc),
                    }
                )
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
        expected_package_sha256: str | None = None,
        publication_id: str | None = None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None = None,
        _deferred_jit_finalization: _DeferredJitRegistryFinalization | None = None,
    ) -> dict[str, Any]:
        self._validate_expected_package_sha256(expected_package_sha256)
        selected_actor = actor or pid
        known_builtin = self._builtin_catalog.metadata(skill_id) is not None
        if known_builtin:
            # Built-in catalog identities and packages are immutable public
            # runtime assets.  Target state remains protected inside the
            # projection authority transaction below.
            skill, metadata = self._get_skill(skill_id)
            self._require_expected_package_sha256(
                skill,
                expected_package_sha256,
            )
            return self._activate_builtin_projection(
                pid,
                skill,
                metadata,
                actor=selected_actor,
                require_capability=require_capability,
                publication_id=publication_id,
                receipt_recorder=receipt_recorder,
            )
        return self._activate_registered_skill(
            pid,
            skill_id,
            actor=selected_actor,
            require_capability=require_capability,
            expected_package_sha256=expected_package_sha256,
            publication_id=publication_id,
            receipt_recorder=receipt_recorder,
            deferred_jit_finalization=_deferred_jit_finalization,
        )

    def validate_activated_skill_result(
        self,
        pid: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate a completed activation against its durable process state.

        This is a read-only certification seam for TaskRun safe-point
        settlement.  It does not make an arbitrary tool-table change trusted:
        callers must separately prove the complete pre/post table delta.
        """

        selected = self._validated_activation_result_shape(pid, result)
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        skill_id = selected["skill_id"]
        loaded = process.loaded_skills.get(skill_id)
        if not isinstance(loaded, dict):
            raise ValidationError(
                f"activated Skill has no durable loaded record: {skill_id}"
            )
        self._require_loaded_skill_provenance(loaded)
        activation_kind = str(loaded.get("activation_kind") or "")
        allowed_loaded_fields = set(_LOADED_SKILL_FIELDS)
        if activation_kind == "builtin_projection":
            allowed_loaded_fields.add(_BUILTIN_PROJECTION_RECEIPT_FIELD)
            skill, trusted_tool_ids = self._validate_loaded_builtin_projection_record(
                skill_id,
                loaded,
            )
            if not self.builtin_projection_supported_by_process(
                process,
                skill_id,
                loaded,
            ):
                raise ValidationError(
                    f"activated built-in Skill escapes its Image binding: {skill_id}"
                )
            if self._loaded_tool_id_map(loaded, "tool_ids") != trusted_tool_ids:
                raise ValidationError(
                    f"activated built-in Skill tool binding changed: {skill_id}"
                )
        elif activation_kind == "registered":
            if self._builtin_catalog.is_builtin_id(skill_id):
                raise ValidationError(
                    f"built-in Skill lost projection provenance: {skill_id}"
                )
            skill = self._skill_for_loaded_record(skill_id, loaded)
            self._validate_registered_loaded_tool_sets(skill_id, skill, loaded)
        else:
            raise ValidationError(
                f"activated Skill has an invalid activation kind: {skill_id}"
            )
        if set(loaded) != allowed_loaded_fields:
            raise ValidationError(
                f"activated Skill loaded record has an invalid shape: {skill_id}"
            )
        self._require_activation_result_matches_loaded(
            selected,
            loaded,
            skill=skill,
        )
        return {
            "activation_kind": activation_kind,
            "skill_id": skill_id,
            "package_sha256": skill.package_sha256,
            "instructions_hash": self._hash_text(skill.instructions),
        }

    def _validated_activation_result_shape(
        self,
        pid: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(result, Mapping) or set(result) != _ACTIVATED_SKILL_RESULT_FIELDS:
            raise ValidationError("TaskRun activate_skill result has an invalid shape")
        selected = dict(result)
        if selected.get("pid") != pid:
            raise ValidationError("TaskRun activate_skill result changed process")
        for field in (
            "skill_id",
            "name",
            "version",
            "instructions_hash",
            "package_sha256",
        ):
            if not isinstance(selected.get(field), str) or not selected[field]:
                raise ValidationError(
                    f"TaskRun activate_skill result has an invalid {field}"
                )
        if not isinstance(selected.get("tool_names"), list) or any(
            not isinstance(name, str) or not name
            for name in selected["tool_names"]
        ):
            raise ValidationError("TaskRun activate_skill result has invalid tool names")
        for field in ("tool_ids", "jit_tool_ids"):
            if not isinstance(selected.get(field), dict) or any(
                not isinstance(name, str)
                or not name
                or not isinstance(tool_id, str)
                or not tool_id
                for name, tool_id in selected[field].items()
            ):
                raise ValidationError(
                    f"TaskRun activate_skill result has invalid {field}"
                )
        return selected

    def _validate_registered_loaded_tool_sets(
        self,
        skill_id: str,
        skill: SkillPackage,
        loaded: Mapping[str, Any],
    ) -> None:
        tool_ids = self._loaded_tool_id_map(loaded, "tool_ids")
        jit_tool_ids = self._loaded_tool_id_map(loaded, "jit_tool_ids")
        expected_jit_names = {tool.name for tool in skill.jit_tools}
        if set(tool_ids) != set(skill.allowed_tools):
            raise ValidationError(
                f"activated Skill static tool provenance changed: {skill_id}"
            )
        if set(jit_tool_ids) != expected_jit_names:
            raise ValidationError(
                f"activated Skill JIT tool provenance changed: {skill_id}"
            )

    def _require_activation_result_matches_loaded(
        self,
        result: Mapping[str, Any],
        loaded: Mapping[str, Any],
        *,
        skill: SkillPackage,
    ) -> None:
        tool_ids = self._loaded_tool_id_map(loaded, "tool_ids")
        jit_tool_ids = self._loaded_tool_id_map(loaded, "jit_tool_ids")
        expected = (
            result.get("skill_id"),
            result.get("name"),
            result.get("version"),
            result.get("package_sha256"),
            result.get("instructions_hash"),
            result.get("tool_names"),
            result.get("tool_ids"),
            result.get("jit_tool_ids"),
        )
        actual = (
            skill.skill_id,
            skill.name,
            skill.version,
            skill.package_sha256,
            self._hash_text(skill.instructions),
            sorted([*tool_ids, *jit_tool_ids]),
            tool_ids,
            jit_tool_ids,
        )
        if expected != actual:
            raise ValidationError(
                f"TaskRun activate_skill result does not match durable state: {skill.skill_id}"
            )

    def _activate_registered_skill(
        self,
        pid: str,
        skill_id: str,
        *,
        actor: str,
        require_capability: bool,
        expected_package_sha256: str | None,
        publication_id: str | None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None,
        deferred_jit_finalization: _DeferredJitRegistryFinalization | None,
    ) -> dict[str, Any]:
        selected_actor = actor
        if require_capability:
            decisions = self._require_skill_right(
                selected_actor,
                skill_id,
                CapabilityRight.EXECUTE,
            )
            admin_decision = self._require_process_admin_if_cross_actor(selected_actor, pid)
            if admin_decision is not None:
                decisions.append(admin_decision)
        else:
            decisions = []
        jit_handles: dict[str, Any] = {}
        retired_jit_ids: set[str] = set()
        # Candidate creation is durable state, so it must be enrolled in the
        # same authority transaction as registration and process publication.
        # This also lets a recovery-fence commit rejection roll it back without
        # attempting a new mutation through the now-stale admission lease.  In
        # addition, mutable registry lookup happens only after reauthorization
        # and finite-use reservation, preventing a revoke/consume race from
        # restoring an absent-vs-inaccessible identity oracle.
        with self._activation_authority_scope(
            decisions,
            actor=selected_actor,
            jit_state=lambda: (jit_handles, retired_jit_ids),
            deferred_jit_finalization=deferred_jit_finalization,
        ):
            skill, metadata = self._get_skill(skill_id)
            self._require_expected_package_sha256(
                skill,
                expected_package_sha256,
            )
            if metadata.get("source_type") == "builtin":
                raise ValidationError(
                    f"registered Skill activation resolved a built-in id: {skill_id}"
                )
            process = self.processes.get_process(pid)
            if process is None:
                raise NotFound(f"process not found: {pid}")
            preflight_loaded = process.loaded_skills.get(skill.skill_id)
            preflight_jit_ids = self._loaded_tool_id_map(
                preflight_loaded,
                "jit_tool_ids",
            )
            self._validate_loadable(
                pid,
                skill,
                process.tool_table,
                replacing_jit_tool_ids=preflight_jit_ids,
            )
            prepared_jit_tools = self._prepare_jit_tools(
                pid, skill, publication_id=publication_id, receipt_recorder=receipt_recorder
            )
            # Tool registry lifecycle operations acquire this lock before the store
            # lock; activation must keep that order while its store transaction is open.
            loaded, tool_ids, jit_tool_ids = self._publish_registered_skill_activation(
                pid,
                skill,
                metadata,
                actor=selected_actor,
                prepared_jit_tools=prepared_jit_tools,
                jit_handles=jit_handles,
                retired_jit_ids=retired_jit_ids,
                publication_id=publication_id,
                receipt_recorder=receipt_recorder,
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

    def _publish_registered_skill_activation(
        self,
        pid: str,
        skill: SkillPackage,
        metadata: dict[str, Any],
        *,
        actor: str,
        prepared_jit_tools: list[tuple[JitToolSpec, str]],
        jit_handles: dict[str, Any],
        retired_jit_ids: set[str],
        publication_id: str | None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None,
    ) -> tuple[LoadedSkill, dict[str, str], dict[str, str]]:
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
                jit_handles.update(
                    self._register_prepared_jit_tools(
                        pid,
                        skill,
                        prepared_jit_tools,
                        replacing_jit_tool_ids=previous_jit_ids,
                        approver=actor,
                        publication_id=publication_id,
                        receipt_recorder=receipt_recorder,
                    )
                )
                # JIT registration advances process CAS; continue from
                # that committed row, not the pre-registration revision.
                process = self.processes.get_process(pid)
                if process is None:
                    raise NotFound(f"process not found: {pid}")
                tool_ids = {
                    name: handle.tool_id for name, handle in existing_handles.items()
                }
                jit_tool_ids = {
                    name: handle.tool_id for name, handle in jit_handles.items()
                }
                updated_table = dict(process.tool_table)
                updated_model_table = dict(process.model_tool_table)
                for name, tool_id in {**previous_tool_ids, **previous_jit_ids}.items():
                    if updated_table.get(name) == tool_id:
                        updated_table.pop(name, None)
                    if updated_model_table.get(name) == tool_id:
                        updated_model_table.pop(name, None)
                for name, handle in {**existing_handles, **jit_handles}.items():
                    updated_table[name] = handle.tool_id
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
                self._persist_loaded_skill(
                    process,
                    loaded=loaded,
                    tool_table=updated_table,
                    model_tool_table=updated_model_table,
                    publication_id=publication_id,
                    receipt_recorder=receipt_recorder,
                )
                retired_jit_ids.update(
                    set(previous_jit_ids.values()) - set(jit_tool_ids.values())
                )
                self._delete_jit_rows(pid, retired_jit_ids)
                self.events.emit(
                    EventType.SKILL_LOADED,
                    source=actor,
                    target=pid,
                    payload={"skill_id": skill.skill_id, "tool_names": loaded.tool_names},
                )
                self.audit.record(
                    actor=actor,
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
                return loaded, tool_ids, jit_tool_ids
        except BaseException:
            # Do not expose in-memory JIT handles after the store transaction
            # rolled back and before releasing the lifecycle lock.
            self._discard_uncommitted_jit_tools(jit_handles)
            raise

    def _activate_builtin_projection(
        self,
        pid: str,
        skill: SkillPackage,
        metadata: dict[str, Any],
        *,
        actor: str,
        require_capability: bool,
        publication_id: str | None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None,
    ) -> dict[str, Any]:
        """Load trusted guidance while revealing only image-owned bindings."""

        decisions: list[CapabilityDecision] = []
        if require_capability:
            admin_decision = self._require_process_admin_if_cross_actor(actor, pid)
            if admin_decision is not None:
                decisions.append(admin_decision)
        with self.capabilities.authority_transaction(
            decisions,
            actor=actor,
            operation="built-in Skill projection activation",
        ):
            # Reauthorization and finite-use reservation above must precede
            # every target-sensitive read, including process existence,
            # image compatibility, tool bindings, and prior projection state.
            self._prompt_instructions(skill)
            process = self.processes.get_process(pid)
            if process is None:
                raise NotFound(f"process not found: {pid}")
            tool_ids = self._builtin_projection_tool_ids(skill, process)
            self._validate_existing_builtin_projection(skill.skill_id, process)
            before_schemas = self._tools.openai_tool_schemas(pid)
            with self.unit_of_work.transaction():
                process = self.processes.get_process(pid)
                if process is None:
                    raise NotFound(f"process not found: {pid}")
                # Repeat the complete-subset check under the publication
                # transaction so a concurrent image/exec transition cannot
                # turn validation into a partial projection.
                tool_ids = self._builtin_projection_tool_ids(skill, process)
                previous_loaded = process.loaded_skills.get(skill.skill_id)
                self._validate_existing_builtin_projection(skill.skill_id, process)
                previous_tool_ids = self._loaded_tool_id_map(previous_loaded, "tool_ids")
                base_tool_ids, base_model_tool_ids = self._activation_base_bindings(
                    process,
                    skill.skill_id,
                    set(tool_ids),
                )
                updated_model_table = dict(process.model_tool_table)
                for name, tool_id in previous_tool_ids.items():
                    if updated_model_table.get(name) == tool_id:
                        updated_model_table.pop(name, None)
                updated_model_table.update(tool_ids)
                loaded = self._persist_builtin_projection(
                    process,
                    skill=skill,
                    source=str(metadata.get("source") or f"builtin:{skill.skill_id}"),
                    actor=actor,
                    tool_ids=tool_ids,
                    base_tool_ids=base_tool_ids,
                    base_model_tool_ids=base_model_tool_ids,
                    updated_model_table=updated_model_table,
                    publication_id=publication_id,
                    receipt_recorder=receipt_recorder,
                )
                after_schemas = self._tools.openai_tool_schemas(pid)
                self.events.emit(
                    EventType.SKILL_LOADED,
                    source=actor,
                    target=pid,
                    # Model-visible lifecycle events use the same payload for
                    # every Skill source. Provenance stays in Host audit state.
                    payload={"skill_id": skill.skill_id, "tool_names": loaded.tool_names},
                )
                self.audit.record(
                    actor=actor,
                    action="skill.activate",
                    target=f"process:{pid}",
                    decision={
                        "skill_id": skill.skill_id,
                        "version": skill.version,
                        "activation_kind": "builtin_projection",
                        "replaced_loaded_version": self._loaded_version(previous_loaded),
                        "tool_ids": tool_ids,
                        "jit_tool_ids": {},
                        "source": metadata.get("source"),
                        "package_sha256": skill.package_sha256,
                        "authority_changed": False,
                        "tool_count_before": len(before_schemas),
                        "tool_count_after": len(after_schemas),
                        "schema_bytes_before": len(dumps(before_schemas).encode("utf-8")),
                        "schema_bytes_after": len(dumps(after_schemas).encode("utf-8")),
                    },
                )
        return {
            "pid": pid,
            "skill_id": skill.skill_id,
            "name": skill.name,
            "version": skill.version,
            # Host callers retain provenance/effect detail. The model-facing
            # ActivateSkillOutput deliberately projects the common fields only.
            "activation_kind": "builtin_projection",
            "tool_names": loaded.tool_names,
            "tool_ids": tool_ids,
            "jit_tool_ids": {},
            "instructions_hash": loaded.instructions_hash,
            "package_sha256": skill.package_sha256,
            "authority_changed": False,
        }

    def _builtin_projection_tool_ids(self, skill: SkillPackage, process: Any) -> dict[str, str]:
        if skill.jit_tools or skill.actions or skill.required_capabilities:
            raise ValidationError(
                f"built-in tool Skill must contain guidance and static allowed-tools only: {skill.skill_id}"
            )
        image = self._images.get(process.image_id)
        if image is None:
            raise ValidationError(
                f"built-in Skill cannot resolve process image: {process.image_id}"
            )
        image_default_tools = set(image.default_tools)
        selected: dict[str, str] = {}
        unavailable: list[str] = []
        for name in skill.allowed_tools:
            configured_tool_id = process.tool_table.get(name)
            try:
                handle = self._tools.resolve(name)
            except (NotFound, ValidationError):
                handle = None
            if (
                name not in image_default_tools
                or configured_tool_id is None
                or handle is None
                or str(handle.tool_id) != str(configured_tool_id)
            ):
                unavailable.append(name)
                continue
            selected[name] = str(configured_tool_id)
        if unavailable:
            raise ValidationError(
                f"built-in Skill is not fully authorized by image {process.image_id}: "
                f"{skill.skill_id} missing={sorted(unavailable)}"
            )
        return selected

    def _validate_existing_builtin_projection(self, skill_id: str, process: Any) -> None:
        """Reject a forged prior record before its bindings influence replacement."""

        previous = process.loaded_skills.get(skill_id)
        if previous is None:
            return
        if not self._loaded_builtin_projection_is_trusted(skill_id, previous, process):
            raise ValidationError(
                f"built-in loaded Skill is missing trusted projection provenance: {skill_id}"
            )

    def _persist_builtin_projection(
        self,
        process: Any,
        *,
        skill: SkillPackage,
        source: str,
        actor: str,
        tool_ids: dict[str, str],
        base_tool_ids: dict[str, str],
        base_model_tool_ids: dict[str, str],
        updated_model_table: dict[str, str],
        publication_id: str | None,
        receipt_recorder: RuntimePublicationReceiptRecorder | None,
    ) -> LoadedSkill:
        loaded_at = utc_now()
        receipt_id = self._record_builtin_projection_receipt(
            pid=process.pid,
            actor=actor,
            skill=skill,
            loaded_at=loaded_at,
        )
        loaded = LoadedSkill(
            skill_id=skill.skill_id,
            version=skill.version,
            source=source,
            package_sha256=skill.package_sha256,
            loaded_at=loaded_at,
            tool_names=sorted(tool_ids),
            tool_ids=dict(tool_ids),
            jit_tool_ids={},
            instructions_hash=self._hash_text(skill.instructions),
            package_snapshot=self._skill_snapshot(skill),
            activation_kind="builtin_projection",
            base_tool_ids=base_tool_ids,
            base_model_tool_ids=base_model_tool_ids,
        )
        self._persist_loaded_skill(
            process,
            loaded=loaded,
            tool_table=dict(process.tool_table),
            model_tool_table=updated_model_table,
            publication_id=publication_id,
            receipt_recorder=receipt_recorder,
            loaded_record_extensions={
                _BUILTIN_PROJECTION_RECEIPT_FIELD: receipt_id,
            },
        )
        return loaded

    def _record_builtin_projection_receipt(
        self,
        *,
        pid: str,
        actor: str,
        skill: SkillPackage,
        loaded_at: str,
    ) -> str:
        """Persist append-only Host evidence for one catalog-authenticated snapshot."""

        receipt = self.audit.record(
            actor=actor,
            action=_BUILTIN_PROJECTION_RECEIPT_ACTION,
            target=self.resource_for(skill.skill_id),
            decision={
                "schema_version": _BUILTIN_PROJECTION_RECEIPT_SCHEMA_VERSION,
                "skill_id": skill.skill_id,
                "activation_kind": "builtin_projection",
                "source": f"builtin:{skill.skill_id}",
                "package_sha256": skill.package_sha256,
                "instructions_hash": self._hash_text(skill.instructions),
                "allowed_tools": sorted(skill.allowed_tools),
                "loaded_at": loaded_at,
                "source_pid": pid,
                "authority_changed": False,
            },
        )
        return str(receipt.record_id)

    def unload_skill(
        self,
        pid: str,
        skill_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        selected_actor = actor or pid
        self._validate_local_unload_state(
            selected_actor,
            pid,
            skill_id,
            require_capability=require_capability,
        )
        decisions = self._unload_identity_authority_decisions(
            selected_actor,
            pid,
            skill_id,
            require_capability=require_capability,
        )
        removed: list[str] = []
        jit_tool_ids: dict[str, str] = {}
        retired_jit_ids: set[str] = set()
        with self._lifecycle_lock:
            with self.capabilities.authority_transaction(
                decisions,
                actor=selected_actor,
                operation="skill unload",
            ):
                # Resolve target existence, loaded state, and projection
                # provenance only after the complete exact/cross-process
                # authority bundle has been revalidated and reserved.
                process, builtin_projection = self._resolve_unload_target_state(
                    pid,
                    skill_id,
                )
                with self.unit_of_work.transaction():
                    process = self.processes.get_process(pid)
                    if process is None:
                        raise NotFound(f"process not found: {pid}")
                    loaded = process.loaded_skills.get(skill_id)
                    if loaded is None:
                        raise NotFound(f"skill is not loaded in process {pid}: {skill_id}")
                    self._require_stable_unload_provenance(
                        skill_id, loaded, process, builtin_projection
                    )
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
                            replacement = self._unload_model_replacement(
                                process,
                                name,
                                base_model_tool_ids,
                                builtin_projection=builtin_projection,
                            )
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
                    self._record_skill_unload(
                        pid=pid,
                        actor=selected_actor,
                        skill_id=skill_id,
                        builtin_projection=builtin_projection,
                        removed=removed,
                        retired_jit_ids=retired_jit_ids,
                    )
            # A failed authority settlement rolls back the process/tool rows and
            # their evidence, so retire in-memory implementations only after the
            # enclosing AuthorityTransaction has committed successfully.
            self._forget_jit_tool_ids(retired_jit_ids)
        return {
            "pid": pid,
            "skill_id": skill_id,
            # Host callers and audit tooling may inspect these fields; the
            # model-facing UnloadSkillOutput omits them for every Skill source.
            "activation_kind": "builtin_projection" if builtin_projection else "registered",
            "removed_tools": sorted(removed),
            "authority_changed": False,
        }

    def _validate_local_unload_state(
        self,
        actor: str,
        pid: str,
        skill_id: str,
        *,
        require_capability: bool,
    ) -> None:
        if not require_capability or actor != pid:
            return
        # A process already receives its own loaded Skill ids and prompt
        # context. Preserve fail-fast validation of forged local provenance
        # without weakening the cross-process error-order boundary.
        self._resolve_unload_target_state(pid, skill_id)

    def _unload_identity_authority_decisions(
        self,
        actor: str,
        pid: str,
        skill_id: str,
        *,
        require_capability: bool,
    ) -> list[CapabilityDecision]:
        if not require_capability:
            return []
        decisions = (
            self._require_skill_right(actor, skill_id, CapabilityRight.EXECUTE)
            if self._builtin_catalog.metadata(skill_id) is None
            else []
        )
        admin_decision = self._require_process_admin_if_cross_actor(actor, pid)
        if admin_decision is not None:
            decisions.append(admin_decision)
        return decisions

    def _resolve_unload_target_state(
        self,
        pid: str,
        skill_id: str,
    ) -> tuple[Any, bool]:
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        loaded = process.loaded_skills.get(skill_id)
        if loaded is None:
            raise NotFound(f"skill is not loaded in process {pid}: {skill_id}")
        builtin_projection = self._loaded_builtin_projection_is_trusted(
            skill_id,
            loaded,
            process,
        )
        return process, builtin_projection

    def _record_skill_unload(
        self,
        *,
        pid: str,
        actor: str,
        skill_id: str,
        builtin_projection: bool,
        removed: list[str],
        retired_jit_ids: set[str],
    ) -> None:
        self.events.emit(
            EventType.SKILL_UNLOADED,
            source=actor,
            target=pid,
            payload={"skill_id": skill_id, "removed_tools": sorted(removed)},
        )
        self.audit.record(
            actor=actor,
            action="skill.unload",
            target=f"process:{pid}",
            decision={
                "skill_id": skill_id,
                "activation_kind": (
                    "builtin_projection" if builtin_projection else "registered"
                ),
                "authority_changed": False,
                "removed_tools": sorted(removed),
                "retired_jit_tool_ids": sorted(retired_jit_ids),
            },
        )

    def _require_stable_unload_provenance(
        self,
        skill_id: str,
        loaded: Any,
        process: Any,
        expected_builtin_projection: bool,
    ) -> None:
        current = self._loaded_builtin_projection_is_trusted(
            skill_id,
            loaded,
            process,
        )
        if current != expected_builtin_projection:
            raise ValidationError(
                f"loaded Skill activation provenance changed during unload: {skill_id}"
            )

    def _loaded_builtin_projection_is_trusted(
        self,
        skill_id: str,
        loaded: Any,
        process: Any,
    ) -> bool:
        activation_kind = (
            loaded.get("activation_kind", "registered")
            if isinstance(loaded, dict)
            else "registered"
        )
        if self._builtin_catalog.is_builtin_id(skill_id) and activation_kind != "builtin_projection":
            raise ValidationError(
                f"built-in loaded Skill is missing trusted projection provenance: {skill_id}"
            )
        if activation_kind != "builtin_projection":
            return False
        if not isinstance(loaded, dict):
            raise ValidationError(f"invalid built-in loaded Skill record: {skill_id}")
        _skill, tool_ids = self._validate_loaded_builtin_projection_record(skill_id, loaded)
        if any(process.tool_table.get(name) != tool_id for name, tool_id in tool_ids.items()):
            raise ValidationError(f"built-in loaded Skill escapes current image tool table: {skill_id}")
        return True

    def reconcile_builtin_projection_image_ceilings(self) -> None:
        """Drop trusted projections that no longer fit a replaced image definition."""

        for process in self.processes.list_processes(limit=None):
            candidates = [
                str(skill_id)
                for skill_id, loaded in process.loaded_skills.items()
                if isinstance(loaded, dict)
                and loaded.get("activation_kind") == "builtin_projection"
            ]
            for skill_id in candidates:
                current = self.processes.get_process(process.pid)
                if current is None:
                    break
                loaded = current.loaded_skills.get(skill_id)
                try:
                    trusted = self._loaded_builtin_projection_is_trusted(
                        skill_id,
                        loaded,
                        current,
                    )
                    supported = self.builtin_projection_supported_by_process(
                        current,
                        skill_id,
                        loaded,
                    )
                except ValidationError:
                    # Preserve malformed state for the normal invalid-snapshot
                    # diagnostics. It must never be trusted as a free unload.
                    continue
                if trusted and not supported:
                    self.unload_skill(
                        current.pid,
                        skill_id,
                        actor="runtime",
                        require_capability=False,
                    )

    def _unload_model_replacement(
        self,
        process: Any,
        name: str,
        base_model_tool_ids: dict[str, str],
        *,
        builtin_projection: bool,
    ) -> str | None:
        replacement = self._remaining_skill_binding(process.loaded_skills, name)
        if replacement is None:
            replacement = base_model_tool_ids.get(name)
        if (
            builtin_projection
            and replacement is not None
            and not self._builtin_image_allows_model_binding(
                process,
                name,
                replacement,
            )
        ):
            return None
        return replacement

    def _builtin_image_allows_model_binding(
        self,
        process: Any,
        name: str,
        tool_id: str,
    ) -> bool:
        image = self._images.get(process.image_id)
        return (
            image is not None
            and name in image.default_tools
            and process.tool_table.get(name) == tool_id
        )

    def builtin_projection_supported_by_process(
        self,
        process: Any,
        skill_id: str,
        loaded: Any,
    ) -> bool:
        """Validate trusted provenance and test target-image compatibility."""

        if not isinstance(loaded, dict) or loaded.get("activation_kind") != "builtin_projection":
            return False
        _skill, tool_ids = self._validate_loaded_builtin_projection_record(skill_id, loaded)
        image = self._images.get(process.image_id)
        return (
            image is not None
            and set(tool_ids).issubset(image.default_tools)
            and all(process.tool_table.get(name) == tool_id for name, tool_id in tool_ids.items())
        )

    def _validate_loaded_builtin_projection_record(
        self,
        skill_id: str,
        loaded: dict[str, Any],
    ) -> tuple[SkillPackage, dict[str, str]]:
        catalog_skill = self._builtin_catalog.get(skill_id)
        if catalog_skill is None:
            raise ValidationError(f"unknown built-in loaded Skill id: {skill_id}")
        if loaded.get("activation_kind") != "builtin_projection":
            raise ValidationError(f"invalid built-in loaded Skill activation kind: {skill_id}")
        if str(loaded.get("source") or "") != f"builtin:{skill_id}":
            raise ValidationError(f"invalid built-in loaded Skill source: {skill_id}")
        if not isinstance(loaded.get("package_snapshot"), dict):
            raise ValidationError(f"built-in loaded Skill is missing its package snapshot: {skill_id}")
        if not str(loaded.get("package_sha256") or ""):
            raise ValidationError(f"built-in loaded Skill is missing its package hash: {skill_id}")
        skill = self._skill_for_loaded_record(skill_id, loaded)
        if skill.jit_tools or skill.actions or skill.required_capabilities:
            raise ValidationError(f"invalid built-in loaded Skill package: {skill_id}")
        if loaded.get("instructions_hash") != self._hash_text(skill.instructions):
            raise ValidationError(f"invalid built-in loaded Skill instructions hash: {skill_id}")
        tool_ids = self._loaded_tool_id_map(loaded, "tool_ids")
        if set(tool_ids) != set(skill.allowed_tools):
            raise ValidationError(f"invalid built-in loaded Skill tool provenance: {skill_id}")
        if loaded.get("tool_names") != sorted(skill.allowed_tools):
            raise ValidationError(f"invalid built-in loaded Skill tool names: {skill_id}")
        if self._loaded_tool_id_map(loaded, "jit_tool_ids"):
            raise ValidationError(f"built-in loaded Skill cannot publish JIT tools: {skill_id}")
        if self._loaded_tool_id_map(loaded, "base_tool_ids") != tool_ids:
            raise ValidationError(f"invalid built-in loaded Skill base tool provenance: {skill_id}")
        base_model_tool_ids = self._loaded_tool_id_map(loaded, "base_model_tool_ids")
        if any(tool_ids.get(name) != tool_id for name, tool_id in base_model_tool_ids.items()):
            raise ValidationError(f"invalid built-in loaded Skill model provenance: {skill_id}")
        self._validate_builtin_projection_receipt(skill_id, skill, loaded)
        return skill, tool_ids

    def _validate_builtin_projection_receipt(
        self,
        skill_id: str,
        skill: SkillPackage,
        loaded: dict[str, Any],
    ) -> None:
        receipt_id = loaded.get(_BUILTIN_PROJECTION_RECEIPT_FIELD)
        if not isinstance(receipt_id, str) or not receipt_id:
            raise ValidationError(
                f"built-in loaded Skill is missing its activation receipt: {skill_id}"
            )
        receipt = self.unit_of_work.evidence.get_audit(receipt_id)
        decision = receipt.decision if receipt is not None else None
        expected = {
            "schema_version": _BUILTIN_PROJECTION_RECEIPT_SCHEMA_VERSION,
            "skill_id": skill_id,
            "activation_kind": "builtin_projection",
            "source": f"builtin:{skill_id}",
            "package_sha256": skill.package_sha256,
            "instructions_hash": self._hash_text(skill.instructions),
            "allowed_tools": sorted(skill.allowed_tools),
            "loaded_at": str(loaded.get("loaded_at") or ""),
            "source_pid": (
                str(decision.get("source_pid") or "")
                if isinstance(decision, dict)
                else ""
            ),
            "authority_changed": False,
        }
        if (
            receipt is None
            or receipt.action != _BUILTIN_PROJECTION_RECEIPT_ACTION
            or receipt.target != self.resource_for(skill_id)
            or not isinstance(decision, dict)
            or decision != expected
        ):
            raise ValidationError(
                f"built-in loaded Skill activation receipt does not match its package: {skill_id}"
            )

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
            if (
                isinstance(loaded, dict)
                and loaded.get("activation_kind") == "builtin_projection"
            ):
                if not self._loaded_builtin_projection_is_trusted(
                    skill_id,
                    loaded,
                    process,
                ):
                    raise ValidationError(
                        f"invalid built-in loaded Skill projection: {skill_id}"
                    )
                skill, _tool_ids = self._validate_loaded_builtin_projection_record(
                    skill_id,
                    loaded,
                )
            elif self._builtin_catalog.is_builtin_id(skill_id):
                raise ValidationError(
                    f"built-in loaded Skill is missing trusted projection provenance: {skill_id}"
                )
            else:
                skill = self._skill_for_loaded_record(skill_id, loaded)
        else:
            skill, _metadata = self._get_skill(skill_id)
        normalized = self._normalize_relative_resource_path(path)
        selected = next((resource for resource in skill.resources if resource.path == normalized), None)
        if selected is None:
            raise NotFound(f"skill resource not found: {skill_id}/{normalized}")
        if max_bytes is None:
            limit = self.config.skills.resource_read_max_bytes
        else:
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
                raise ValidationError("skill resource max_bytes must be an integer")
            if max_bytes < 1:
                raise ValidationError("skill resource max_bytes must be >= 1")
            limit = max_bytes
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
        raw_resources, source = self._read_host_resources(root)
        raw_skill = raw_resources["SKILL.md"]
        frontmatter, body = self._parse_skill_markdown(raw_skill.decode("utf-8"), expected_dir_name=root.name)
        resources = [
            self._resource_from_bytes(resource_path, content)
            for resource_path, content in sorted(raw_resources.items())
        ]
        package = self._package_from_parts(frontmatter, body, resources)
        return package, source

    def _read_workspace_skill_manifest(
        self,
        pid: str,
        path: str,
    ) -> _WorkspaceSkillManifest:
        cwd = self._process.working_directory(pid)
        package_root, skill_md_path = self._workspace_package_paths(path)
        raw_skill, total_bytes = self._read_workspace_package_file(
            pid,
            skill_md_path,
            display_path=skill_md_path,
            file_max_bytes=self.config.skills.skill_md_max_bytes,
            file_limit_name="skill_md_max_bytes",
            file_error_prefix="SKILL.md",
            total_bytes=0,
            cwd=cwd,
        )
        frontmatter, body = self._parse_skill_markdown(
            raw_skill.decode("utf-8"),
            expected_dir_name=Path(package_root).name,
        )
        _target, workspace_package_root = self._filesystem.resolve_path(package_root, cwd=cwd)
        return _WorkspaceSkillManifest(
            requested_path=path,
            cwd=cwd,
            package_root=package_root,
            skill_md_path=skill_md_path,
            workspace_package_root=workspace_package_root,
            raw_skill=raw_skill,
            frontmatter=frontmatter,
            body=body,
            total_bytes=total_bytes,
        )

    def _load_package_from_workspace(
        self,
        pid: str,
        path: str,
        *,
        manifest: _WorkspaceSkillManifest | None = None,
    ) -> tuple[SkillPackage, str]:
        selected = manifest or self._read_workspace_skill_manifest(pid, path)
        if selected.requested_path != path:
            raise ValidationError("workspace Skill manifest path changed before load")
        references = self._frontmatter_reference_paths(selected.frontmatter)
        total_bytes = selected.total_bytes
        raw_resources: dict[str, bytes] = {"SKILL.md": selected.raw_skill}
        for ref in references:
            self._require_workspace_package_file_slot(raw_resources)
            content, total_bytes = self._read_workspace_package_file(
                pid,
                self._join_relative(selected.package_root, ref),
                display_path=ref,
                file_max_bytes=self.config.skills.resource_read_max_bytes,
                file_limit_name="resource_read_max_bytes",
                file_error_prefix="skill metadata resource",
                total_bytes=total_bytes,
                cwd=selected.cwd,
            )
            raw_resources[ref] = content
        jit_tools = self._load_jit_specs_from_resources(
            selected.frontmatter,
            raw_resources,
        )
        for tool in jit_tools:
            if tool.source_path not in raw_resources:
                self._require_workspace_package_file_slot(raw_resources)
                content, total_bytes = self._read_workspace_package_file(
                    pid,
                    self._join_relative(selected.package_root, tool.source_path),
                    display_path=tool.source_path,
                    file_max_bytes=self.config.skills.resource_read_max_bytes,
                    file_limit_name="resource_read_max_bytes",
                    file_error_prefix="Skill JIT source",
                    total_bytes=total_bytes,
                    cwd=selected.cwd,
                )
                raw_resources[tool.source_path] = content
        self._read_workspace_resource_dirs(
            pid,
            selected.workspace_package_root,
            raw_resources,
            total_bytes=total_bytes,
        )
        resources = [self._resource_from_bytes(path, content) for path, content in sorted(raw_resources.items())]
        package = self._package_from_parts(
            selected.frontmatter,
            selected.body,
            resources,
        )
        if package.skill_id != selected.skill_id or package.name != selected.skill_id:
            raise ValidationError(
                "workspace Skill identity changed after pre-authorization: "
                f"{selected.skill_id} != {package.skill_id}"
            )
        return package, selected.package_root

    def _read_workspace_resource_dirs(
        self,
        pid: str,
        workspace_package_root: str,
        raw_resources: dict[str, bytes],
        *,
        total_bytes: int,
    ) -> None:
        max_files = self.config.skills.max_package_files
        traversal = _SkillPackageTraversalBudget(
            max_directories=self.config.skills.max_package_directories,
            max_depth=self.config.skills.max_package_depth,
        )
        visited_dirs: set[str] = set()
        pending: list[tuple[str, int]] = [
            (self._join_relative(workspace_package_root, directory), 1)
            for directory in reversed(self.config.skills.resource_dirs)
        ]

        while pending:
            directory, depth = pending.pop()
            normalized_dir = directory.strip("/")
            if normalized_dir in visited_dirs:
                continue
            visited_dirs.add(normalized_dir)
            if not self._has_read_authority(pid, self._filesystem.directory_resource_for_path(normalized_dir, cwd=None)):
                continue
            traversal.charge_directory(depth=depth)
            remaining_entries = (
                max_files
                - len(raw_resources)
                + traversal.max_directories
                - traversal.directories
            )
            try:
                listing = self._filesystem.read_directory(
                    pid,
                    normalized_dir,
                    # Read one extra entry so a wide directory fails closed
                    # instead of silently omitting package topology.
                    limit=max(1, remaining_entries + 1),
                    cwd=None,
                )
            except NotFound:
                # Configured resource roots are optional.  A path that did
                # not exist consumed no traversal work beyond this bounded
                # probe and therefore is not package topology.
                traversal.directories -= 1
                continue
            if listing.truncated or len(listing.entries) > remaining_entries:
                raise ValidationError(
                    "skill package directory exceeds the remaining aggregate "
                    f"max_package_files={max_files} and "
                    "max_package_directories="
                    f"{traversal.max_directories} budget"
                )
            child_directories: list[tuple[str, int]] = []
            for entry in listing.entries:
                relative = self._workspace_resource_relative_path(workspace_package_root, entry.path)
                if relative is None:
                    continue
                if entry.kind == "directory":
                    child_directories.append((entry.path, depth + 1))
                    continue
                if entry.kind != "file" or relative in raw_resources:
                    continue
                self._validate_resource_path(relative)
                if not self._has_read_authority(pid, self._filesystem.resource_for_path(entry.path, cwd=None)):
                    continue
                self._require_workspace_package_file_slot(
                    raw_resources,
                    directory_count=traversal.directories,
                )
                content, total_bytes = self._read_workspace_package_file(
                    pid,
                    entry.path,
                    display_path=relative,
                    file_max_bytes=self.config.skills.resource_read_max_bytes,
                    file_limit_name="resource_read_max_bytes",
                    file_error_prefix="skill resource",
                    total_bytes=total_bytes,
                    cwd=None,
                )
                raw_resources[relative] = content
            pending.extend(reversed(child_directories))

    def _require_workspace_package_file_slot(
        self,
        raw_resources: Mapping[str, bytes],
        *,
        directory_count: int = 0,
    ) -> None:
        if len(raw_resources) >= self.config.skills.max_package_files:
            raise ValidationError(
                "skill package exceeds max_package_files="
                f"{self.config.skills.max_package_files}"
            )
        if directory_count > self.config.skills.max_package_directories:
            raise ValidationError(
                "skill package exceeds max_package_directories="
                f"{self.config.skills.max_package_directories}"
            )

    def _read_workspace_package_file(
        self,
        pid: str,
        path: str,
        *,
        display_path: str,
        file_max_bytes: int,
        file_limit_name: str,
        file_error_prefix: str,
        total_bytes: int,
        cwd: str | None,
    ) -> tuple[bytes, int]:
        remaining = self.config.skills.package_max_bytes - total_bytes
        selected_limit = min(file_max_bytes, max(remaining, 1))
        package_limited = remaining < file_max_bytes
        read = self._filesystem.read_bytes(
            pid,
            path,
            max_bytes=selected_limit,
            cwd=cwd,
        )
        if read.truncated or len(read.content) > max(remaining, 0):
            if package_limited or len(read.content) > max(remaining, 0):
                raise ValidationError(
                    "skill package exceeds package_max_bytes="
                    f"{self.config.skills.package_max_bytes}: {display_path}"
                )
            raise ValidationError(
                f"{file_error_prefix} exceeds {file_limit_name}={file_max_bytes}: "
                f"{display_path}"
            )
        return read.content, total_bytes + len(read.content)

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
        allowed_tools = self._allowed_tools(data.get("allowed-tools"))
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

    def _read_host_resources(self, root: Path) -> tuple[dict[str, bytes], str]:
        root_absolute = Path(os.path.abspath(root))
        try:
            root_guard = open_secure_directory(root_absolute)
        except OSError as exc:
            raise ValidationError(
                f"cannot securely open Skill package directory: {root_absolute}"
            ) from exc
        raw_resources: dict[str, bytes] = {}
        total_bytes = 0
        traversal = _SkillPackageTraversalBudget(
            max_directories=self.config.skills.max_package_directories,
            max_depth=self.config.skills.max_package_depth,
        )
        with root_guard:
            source_root = root_guard.path
            opened_root = self._validate_host_package_directory_snapshot(
                root_guard.snapshot(),
                path=source_root,
                after_read=False,
            )
            try:
                linked_root = self._validate_host_package_directory_snapshot(
                    root_guard.linked_snapshot(),
                    path=source_root,
                    after_read=False,
                )
            except OSError as exc:
                raise ValidationError(
                    f"Skill package directory changed during enumeration: {source_root}"
                ) from exc
            if linked_root != opened_root:
                raise ValidationError(
                    f"Skill package directory changed during enumeration: {source_root}"
                )
            raw_skill, total_bytes = self._read_host_package_file_with_budget(
                source_root / "SKILL.md",
                parent=root_guard,
                relative_name="SKILL.md",
                file_max_bytes=self.config.skills.skill_md_hard_limit_bytes,
                file_limit_name="skill_md_hard_limit_bytes",
                total_bytes=total_bytes,
            )
            raw_resources["SKILL.md"] = raw_skill
            for configured_directory in self.config.skills.resource_dirs:
                normalized_directory = self._normalize_relative_resource_path(
                    configured_directory
                )
                directory_path = source_root / normalized_directory
                try:
                    directory_guard = open_secure_directory(directory_path)
                except OSError as exc:
                    if exc.errno == errno.ENOENT:
                        continue
                    if exc.errno == errno.ELOOP:
                        raise ValidationError(
                            "skill package symlinks are not supported: "
                            f"{directory_path}"
                        ) from exc
                    if exc.errno == errno.ENOTDIR:
                        raise ValidationError(
                            f"skill resource path is not a directory: {configured_directory}"
                        ) from exc
                    raise ValidationError(
                        f"cannot securely open Skill resource directory: {directory_path}"
                    ) from exc
                with directory_guard:
                    traversal.charge_directory(depth=1)
                    total_bytes = self._read_host_resource_directory(
                        directory_guard,
                        source_root=source_root,
                        raw_resources=raw_resources,
                        total_bytes=total_bytes,
                        traversal=traversal,
                        depth=1,
                    )
            after_root = self._validate_host_package_directory_snapshot(
                root_guard.snapshot(),
                path=source_root,
                after_read=True,
            )
            try:
                linked_after = self._validate_host_package_directory_snapshot(
                    root_guard.linked_snapshot(),
                    path=source_root,
                    after_read=True,
                )
            except OSError as exc:
                raise ValidationError(
                    f"Skill package directory changed during enumeration: {source_root}"
                ) from exc
            if after_root != opened_root or linked_after != after_root:
                raise ValidationError(
                    f"Skill package directory changed during enumeration: {source_root}"
                )
        return raw_resources, str(source_root)

    def _read_host_resource_directory(
        self,
        directory: SecureDirectoryGuard,
        *,
        source_root: Path,
        raw_resources: dict[str, bytes],
        total_bytes: int,
        traversal: _SkillPackageTraversalBudget,
        depth: int,
    ) -> int:
        opened = self._validate_host_package_directory_snapshot(
            directory.snapshot(),
            path=directory.path,
            after_read=False,
        )
        try:
            linked = self._validate_host_package_directory_snapshot(
                directory.linked_snapshot(),
                path=directory.path,
                after_read=False,
            )
        except OSError as exc:
            raise ValidationError(
                f"Skill package directory changed during enumeration: {directory.path}"
            ) from exc
        if linked != opened:
            raise ValidationError(
                f"Skill package directory changed during enumeration: {directory.path}"
            )
        try:
            iterator = directory.scandir()
        except OSError as exc:
            raise ValidationError(
                f"Skill package directory changed during enumeration: {directory.path}"
            ) from exc
        remaining_entries = (
            self.config.skills.max_package_files
            - len(raw_resources)
            + traversal.max_directories
            - traversal.directories
        )
        with iterator:
            entries = list(
                itertools.islice(iterator, max(1, remaining_entries + 1))
            )
        if len(entries) > remaining_entries:
            raise ValidationError(
                "skill package directory exceeds the remaining aggregate "
                "max_package_files="
                f"{self.config.skills.max_package_files} and "
                "max_package_directories="
                f"{traversal.max_directories} budget"
            )
        entries.sort(key=lambda entry: entry.name)
        for entry in entries:
            file = directory.path / entry.name
            try:
                relative = file.relative_to(source_root).as_posix()
            except ValueError as exc:
                raise ValidationError(
                    f"skill resource escapes package root: {file}"
                ) from exc
            self._validate_resource_path(relative)
            try:
                before = directory.lstat_child(entry.name)
            except OSError as exc:
                raise ValidationError(
                    f"skill package path changed during enumeration: {file}"
                ) from exc
            if before.is_reparse_point or stat.S_ISLNK(before.mode):
                raise ValidationError(
                    f"skill package symlinks are not supported: {file}"
                )
            if stat.S_ISDIR(before.mode):
                traversal.charge_directory(depth=depth + 1)
                try:
                    child = directory.open_child_directory(entry.name)
                except OSError as exc:
                    raise ValidationError(
                        f"skill package directory changed during enumeration: {file}"
                    ) from exc
                with child:
                    total_bytes = self._read_host_resource_directory(
                        child,
                        source_root=source_root,
                        raw_resources=raw_resources,
                        total_bytes=total_bytes,
                        traversal=traversal,
                        depth=depth + 1,
                    )
                continue
            if not stat.S_ISREG(before.mode):
                raise ValidationError(
                    f"skill package path is not a regular file or directory: {file}"
                )
            if relative in raw_resources:
                continue
            if len(raw_resources) >= self.config.skills.max_package_files:
                raise ValidationError(
                    "skill package exceeds max_package_files="
                    f"{self.config.skills.max_package_files}"
                )
            content, total_bytes = self._read_host_package_file_with_budget(
                file,
                parent=directory,
                relative_name=entry.name,
                file_max_bytes=self.config.skills.resource_read_max_bytes,
                file_limit_name="resource_read_max_bytes",
                total_bytes=total_bytes,
            )
            raw_resources[relative] = content
        after = self._validate_host_package_directory_snapshot(
            directory.snapshot(),
            path=directory.path,
            after_read=True,
        )
        try:
            linked_after = self._validate_host_package_directory_snapshot(
                directory.linked_snapshot(),
                path=directory.path,
                after_read=True,
            )
        except OSError as exc:
            raise ValidationError(
                f"Skill package directory changed during enumeration: {directory.path}"
            ) from exc
        if after != opened or linked_after != after:
            raise ValidationError(
                f"Skill package directory changed during enumeration: {directory.path}"
            )
        return total_bytes

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
        scanned_entries = 0
        scan_limit = self.config.skills.catalog_scan_limit
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            remaining = scan_limit - scanned_entries
            iterator = root.iterdir()
            try:
                bounded_children = list(
                    itertools.islice(iterator, max(1, remaining + 1))
                )
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            if len(bounded_children) > remaining:
                raise ValidationError(
                    "Skill catalog exceeds catalog_scan_limit="
                    f"{scan_limit}: {root}"
                )
            scanned_entries += len(bounded_children)
            for child in sorted(bounded_children, key=lambda entry: entry.name):
                if not child.is_dir():
                    continue
                if self._builtin_catalog.is_builtin_id(child.name):
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
                score = self._skill_discovery_score(summary, text)
                if score is None:
                    continue
                skill_id = str(summary["skill_id"])
                if skill_id in seen:
                    continue
                summary["_discovery_score"] = score
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
        loaded_record_extensions: Mapping[str, Any] | None = None,
    ) -> Any:
        loaded_record = to_jsonable(loaded)
        if loaded_record_extensions:
            loaded_record.update(dict(loaded_record_extensions))
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
                "loaded Skill state is missing canonical tool provenance"
            )
        activation_kind = loaded.get("activation_kind", "registered")
        if activation_kind not in {"registered", "builtin_projection"}:
            raise ValidationError(f"unknown loaded Skill activation_kind: {activation_kind}")

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
        missing: list[tuple[str, dict[str, Any]]] = []
        decisions: list[CapabilityDecision] = []
        for right in rights:
            requested_right = right.value
            context = {
                "adapter": "skill",
                "authority_operation": "skill.use",
                "primitive": "runtime.skills.activate",
                "operation": "use",
                "pid": actor,
                "resource": resource,
                "right": requested_right,
                "skill_id": skill_id,
                "target_state_version": None,
            }
            # A Host Human one-shot carries an approval binding over this exact
            # context.  Reuse the same deterministic context on the resumed
            # authorization pass; omitting it would make the approved grant
            # permanently unusable and repeatedly prompt for the first right.
            decision = self.capabilities.authorize(
                actor,
                resource,
                right,
                context,
            )
            if decision.allowed:
                decisions.append(decision)
                continue
            missing.append((requested_right, context))
        if not missing:
            return decisions
        if self.human is None:
            raise CapabilityDenied(
                f"{actor} lacks {[right for right, _context in missing]} on {resource}"
            )
        # Each Human decision is bound to one exact right.  If an operation
        # needs several rights, the resumed authorization pass requests the
        # next missing right rather than hiding a bundled grant in a generic
        # permission payload.
        requested_right, approval_context = missing[0]
        request_id = self.human.query_authority_request(
            pid=actor,
            human=self.config.runtime.default_human,
            request={
                "type": "external_operation_approval",
                "question": (
                    f"Allow process {actor} to use skill {skill_id} "
                    f"right={requested_right} once?"
                ),
                "requested_once_capability": {
                    "subject": actor,
                    "resource": resource,
                    "rights": [requested_right],
                    "constraints": {},
                },
                "context": approval_context,
            },
            blocking=True,
            authority_origin="external_operation",
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

    def _reauthorize_skill_read_decision(
        self,
        decision: CapabilityDecision,
        reservations: Mapping[str, str],
    ) -> None:
        cap_id = (
            str(decision.consume_capability_id)
            if decision.consume_capability_id is not None
            else None
        )
        reservation_id = reservations.get(cap_id) if cap_id is not None else None
        if reservation_id is None:
            self.capabilities.reauthorize_decision(decision)
            return
        reservation = self.capabilities.store.get_capability_use_reservation(
            reservation_id
        )
        if (
            reservation is None
            or reservation.get("status") != "reserved"
            or str(reservation.get("cap_id")) != cap_id
        ):
            raise CapabilityDenied(
                "Skill read authority reservation is no longer active"
            )

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

    def _restore_skill_rights_after_failure(
        self,
        reservations: dict[str, str],
        error: BaseException,
    ) -> None:
        """Compensate a failed read without replacing its original failure."""

        try:
            self._restore_skill_rights(reservations)
        except BaseException as cleanup_error:
            error.add_note(
                "failed to restore reserved Skill authority after the operation "
                "failed; authority remains fail closed: "
                f"{type(cleanup_error).__name__}"
            )

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

    def _normalize_global_source(self, path: str | Path) -> tuple[Path, str]:
        selected = Path(path).expanduser().resolve()
        roots = [Path(root).expanduser().resolve() for root in self.config.skills.global_dirs]
        for root in roots:
            relative = self._relative_to_configured_host_root(selected, root)
            if relative is None:
                continue
            return selected, relative.as_posix()
        raise CapabilityDenied(f"global skill path is outside configured global_dirs: {selected}")

    @staticmethod
    def _relative_to_configured_host_root(
        selected: Path,
        root: Path,
    ) -> Path | None:
        try:
            return selected.relative_to(root)
        except ValueError:
            pass
        if os.name != "nt":
            return None

        # Windows can name the same existing directory through a DOS 8.3 alias
        # or different preserved casing. Compare each ancestor by filesystem
        # identity, then rebuild the relative suffix without trusting spelling.
        suffix: list[str] = []
        current = selected
        while True:
            try:
                if os.path.samefile(current, root):
                    return Path(*reversed(suffix)) if suffix else Path(".")
            except OSError:
                pass
            parent = current.parent
            if parent == current:
                return None
            suffix.append(current.name)
            current = parent

    def _get_skill(self, skill_id: str) -> tuple[SkillPackage, dict[str, Any]]:
        builtin = self._builtin_catalog.get(skill_id)
        if builtin is not None:
            metadata = self._builtin_catalog.metadata(skill_id)
            assert metadata is not None
            return builtin, {
                **metadata,
                "registered_by": None,
                "created_at": None,
                "updated_at": None,
            }
        if self._builtin_catalog.is_builtin_id(skill_id):
            raise ValidationError(f"unknown reserved built-in Skill id: {skill_id}")
        found = self.store.get_skill(skill_id)
        if found is None:
            raise NotFound(f"skill not found: {skill_id}")
        return found

    @staticmethod
    def _validate_expected_package_sha256(
        expected_package_sha256: str | None,
    ) -> None:
        if expected_package_sha256 is None:
            return
        if (
            not isinstance(expected_package_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_package_sha256) is None
        ):
            raise ValidationError(
                "expected_package_sha256 must be 64 lowercase hexadecimal characters"
            )

    @classmethod
    def _require_expected_package_sha256(
        cls,
        skill: SkillPackage,
        expected_package_sha256: str | None,
    ) -> None:
        cls._validate_expected_package_sha256(expected_package_sha256)
        if expected_package_sha256 is None:
            return
        if expected_package_sha256 != skill.package_sha256:
            raise SkillPackageChanged(
                f"Skill package changed since discovery: {skill.skill_id}"
            )

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
            "package_sha256": skill.package_sha256 or metadata.get("package_sha256"),
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
        limit = self.config.skills.max_prompt_instruction_chars
        if len(skill.instructions) > limit:
            raise ValidationError(
                "Skill instructions exceed model-visible prompt limit "
                f"max_prompt_instruction_chars={limit}: {skill.skill_id}"
            )
        return skill.instructions

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
        if selected.is_symlink():
            raise ValidationError(f"skill package path is a symlink: {selected}")
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
            if self._relative_to_configured_host_root(selected, root) is not None:
                return "global"
        return "workspace"

    def _validate_host_package_file_snapshot(
        self,
        snapshot: StablePathSnapshot,
        *,
        path: Path,
        after_read: bool,
    ) -> StablePathSnapshot:
        if snapshot.is_reparse_point or not stat.S_ISREG(snapshot.mode):
            raise ValidationError(
                f"skill package file is not a regular file: {path}"
            )
        if snapshot.links > 1:
            raise ValidationError(
                f"skill package hard links are not supported: {path}"
            )
        if snapshot.links < 1:
            message = "changed during read" if after_read else "is not linked"
            raise ValidationError(f"skill package file {message}: {path}")
        if not stable_identity_available(snapshot):
            raise ValidationError(
                "secure Host Skill package file identity is unavailable on this platform"
            )
        if snapshot.size < 0:
            raise ValidationError(
                f"skill package file has an invalid size: {path}"
            )
        return snapshot

    def _validate_host_package_directory_snapshot(
        self,
        snapshot: StablePathSnapshot,
        *,
        path: Path,
        after_read: bool,
    ) -> StablePathSnapshot:
        if snapshot.is_reparse_point or not stat.S_ISDIR(snapshot.mode):
            raise ValidationError(
                f"skill package directory is not a regular directory: {path}"
            )
        if snapshot.links < 1:
            message = (
                "changed during enumeration" if after_read else "is not linked"
            )
            raise ValidationError(f"skill package directory {message}: {path}")
        if not stable_identity_available(snapshot):
            raise ValidationError(
                "secure Host Skill package directory identity is unavailable on this platform"
            )
        if snapshot.size < 0:
            raise ValidationError(
                f"skill package directory has an invalid size: {path}"
            )
        return snapshot

    def _read_host_package_file_with_budget(
        self,
        path: Path,
        *,
        parent: SecureDirectoryGuard,
        relative_name: str,
        file_max_bytes: int,
        file_limit_name: str,
        total_bytes: int,
    ) -> tuple[bytes, int]:
        remaining = self.config.skills.package_max_bytes - total_bytes
        selected_limit = min(file_max_bytes, max(remaining, 0))
        package_limited = remaining < file_max_bytes
        try:
            secure_file = open_secure_file(
                path,
                parent=parent,
                relative_name=relative_name,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValidationError(
                    f"skill package symlinks are not supported: {path}"
                ) from exc
            if exc.errno in {
                errno.ENOENT,
                errno.ENOTDIR,
                getattr(errno, "ESTALE", -1),
            }:
                raise ValidationError(
                    f"skill package file changed during read: {path}"
                ) from exc
            raise ValidationError(
                f"cannot securely open Skill package file: {path}"
            ) from exc
        try:
            content = read_stable_file_limited(
                secure_file,
                max_bytes=selected_limit,
                chunk_bytes=_SKILL_PACKAGE_READ_CHUNK_BYTES,
                validate_snapshot=lambda snapshot, after_read: self._validate_host_package_file_snapshot(
                    snapshot,
                    path=path,
                    after_read=after_read,
                ),
            )
        except SecureFileLimitExceeded as exc:
            if package_limited:
                raise ValidationError(
                    "skill package exceeds package_max_bytes="
                    f"{self.config.skills.package_max_bytes}: {path}"
                ) from exc
            raise ValidationError(
                f"skill package file exceeds {file_limit_name}={file_max_bytes}: {path}"
            ) from exc
        except SecureFileReadUnavailable as exc:
            raise ValidationError(
                f"cannot securely read Skill package file to EOF: {path}"
            ) from exc
        except (SecureFileChanged, OSError) as exc:
            raise ValidationError(
                f"skill package file changed during read: {path}"
            ) from exc
        return content, total_bytes + len(content)

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
            if resource.content_base64 is not None:
                raise ValidationError(
                    f"text skill resource must not contain content_base64: {resource.path}"
                )
            return resource.content.encode("utf-8")
        if resource.kind == "base64":
            if resource.content_base64 is None:
                raise ValidationError(f"base64 skill resource is missing content: {resource.path}")
            if resource.content is not None:
                raise ValidationError(
                    f"base64 skill resource must not contain text content: {resource.path}"
                )
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

    def _allowed_tools(self, value: Any) -> list[str]:
        """Parse canonical Agent Skills syntax while retaining legacy lists."""

        if value is None or value == {}:
            return []
        if isinstance(value, str):
            return self._require_string(value, "allowed-tools").split()
        return self._string_list(value, "allowed-tools")

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
        allowed = {"resource", "rights", "constraints"}
        unknown = sorted(
            key if isinstance(key, str) else f"<non-string:{type(key).__name__}>"
            for key in spec
            if not isinstance(key, str) or key not in allowed
        )
        if unknown:
            raise ValidationError(
                f"capability spec contains unknown fields: {unknown}"
            )
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
        self._validate_string_length(
            value,
            "name",
            min(self.config.skills.name_max_chars, _SKILL_NAME_MAX_CHARS),
        )
        if not _SKILL_NAME_PATTERN.match(value):
            raise ValidationError(
                "skill name must use lowercase letters, digits, and single internal hyphens: "
                f"{value!r}"
            )

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
