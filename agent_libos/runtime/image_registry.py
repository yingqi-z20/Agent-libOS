from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import stat
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.validators import validator_for as jsonschema_validator_for

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    AgentImage,
    Capability,
    CapabilityRight,
    EventType,
    JIT_MULTIPLEXER_TOOL_NAME,
    JIT_TOOL_EXPOSURE_DIRECT,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    JIT_TOOL_EXPOSURES,
    OPENAI_TOOL_NAME_MAX_CHARS,
    ObjectOwnerKind,
    PROMPT_MODE_IMAGE_ONLY,
    PROMPT_MODES,
    ToolHandle,
    ToolSpec,
    is_openai_tool_name,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.skills.schema import JitToolSpec
from agent_libos.storage import SQLiteStore
from agent_libos.tools.observability import ensure_json_size
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, loads
from agent_libos.utils.yaml_loader import load_yaml_mapping

_IMAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
_WINDOWS_FORBIDDEN_PATH_CHARS = set('<>:"|?*')
_WINDOWS_RESERVED_PATH_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_SENSITIVE_PACKAGE_FILENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
_SENSITIVE_PACKAGE_SUFFIXES = (".pem", ".p12", ".pfx", ".key")
_CACHE_PACKAGE_SEGMENTS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
}
_PACKAGE_BOOT_KIND = "image_package"
_CHECKPOINT_BOOT_KIND = "checkpoint_commit"


@dataclass(frozen=True)
class ImageRegistrationResult:
    image: AgentImage
    replaced: bool
    source: str | None = None


class ImageRegistryPrimitive:
    """Registers AgentImage definitions under capability, audit, and event control."""

    IMAGE_FIELDS = {
        "image_id",
        "name",
        "version",
        "system_prompt",
        "prompt_mode",
        "jit_tool_exposure",
        "planner",
        "action_schema",
        "default_skills",
        "default_tools",
        "context_policy",
        "safety_profile",
        "llm_profile_id",
        "required_capabilities",
        "metadata",
        "signature",
        "boot",
    }

    def __init__(
        self,
        images: dict[str, AgentImage],
        capabilities: CapabilityManager,
        audit: AuditManager,
        events: EventBus,
        tool_exists: Any,
        store: SQLiteStore | None = None,
        config: AgentLibOSConfig | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.store = store
        self.images = images
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.tool_exists = tool_exists
        self.runtime: Any | None = None

    def bind_runtime(self, runtime: Any) -> None:
        self.runtime = runtime

    def register(
        self,
        image: AgentImage | dict[str, Any],
        *,
        actor: str = "runtime",
        replace: bool = False,
        require_capability: bool = False,
        source: str | None = None,
    ) -> ImageRegistrationResult:
        candidate = self._coerce_image(image)
        if require_capability:
            self.capabilities.require(actor, self.resource_for(candidate.image_id), CapabilityRight.WRITE)
        existing = self.images.get(candidate.image_id)
        if existing is not None and not replace:
            raise ValidationError(f"agent image already exists: {candidate.image_id}")
        self._validate_image(candidate)
        self.images[candidate.image_id] = candidate
        now = utc_now()
        if self.store is not None:
            self.store.upsert_image(candidate, registered_by=actor, source=source, created_at=now)
        action = "image.replace" if existing is not None else "image.register"
        self.events.emit(
            EventType.IMAGE_REGISTERED,
            source=actor,
            target=self.resource_for(candidate.image_id),
            payload={
                "image_id": candidate.image_id,
                "name": candidate.name,
                "version": candidate.version,
                "replaced": existing is not None,
                "source": source,
                "boot_kind": candidate.boot.get("kind", "fresh"),
            },
        )
        self.audit.record(
            actor=actor,
            action=action,
            target=self.resource_for(candidate.image_id),
            decision={
                "image_id": candidate.image_id,
                "name": candidate.name,
                "version": candidate.version,
                "default_tools": list(candidate.default_tools),
                "required_capabilities": len(candidate.required_capabilities),
                "replaced": existing is not None,
                "source": source,
                "boot_kind": candidate.boot.get("kind", "fresh"),
            },
        )
        return ImageRegistrationResult(image=candidate, replaced=existing is not None, source=source)

    def validate_package_path(self, path: str | Path) -> dict[str, Any]:
        files, source = self._read_host_package(path)
        image, artifact = self._image_package_from_files(files, source=source)
        return self._package_validation_summary(image, artifact)

    def validate_workspace_package(self, pid: str, path: str) -> dict[str, Any]:
        files, source = self._read_workspace_package(pid, path)
        image, artifact = self._image_package_from_files(files, source=source)
        return self._package_validation_summary(image, artifact)

    def register_from_package_path(
        self,
        path: str | Path,
        *,
        actor: str,
        replace: bool = False,
        require_capability: bool = False,
        source: str | None = None,
    ) -> ImageRegistrationResult:
        files, detected_source = self._read_host_package(path)
        return self.register_from_package_files(
            files,
            actor=actor,
            replace=replace,
            require_capability=require_capability,
            source=source or detected_source,
        )

    def register_from_workspace_package(
        self,
        pid: str,
        path: str,
        *,
        replace: bool = False,
    ) -> ImageRegistrationResult:
        files, source = self._read_workspace_package(pid, path)
        return self.register_from_package_files(
            files,
            actor=pid,
            replace=replace,
            require_capability=True,
            source=source,
        )

    def register_from_package_files(
        self,
        files: dict[str, bytes | str],
        *,
        actor: str,
        replace: bool = False,
        require_capability: bool = False,
        source: str | None = None,
    ) -> ImageRegistrationResult:
        if self.store is None:
            raise ValidationError("image package registration requires SQLiteStore artifact persistence")
        normalized_files = self._normalize_package_files(files)
        image, artifact = self._image_package_from_files(normalized_files, source=source)
        if require_capability:
            self.capabilities.require(actor, self.resource_for(image.image_id), CapabilityRight.WRITE)
        existing = self.images.get(image.image_id)
        if existing is not None and not replace:
            raise ValidationError(f"agent image already exists: {image.image_id}")
        artifact_json = dumps(artifact)
        artifact_bytes = len(artifact_json.encode("utf-8"))
        if artifact_bytes > self.config.image_commit.artifact_hard_limit_bytes:
            raise ValidationError(
                "image package artifact exceeded "
                f"artifact_hard_limit_bytes={self.config.image_commit.artifact_hard_limit_bytes}"
            )
        artifact_sha256 = hashlib.sha256(artifact_json.encode("utf-8")).hexdigest()
        artifact_id = f"imgpkg_{artifact_sha256[:24]}"
        image = AgentImage(
            image_id=image.image_id,
            name=image.name,
            version=image.version,
            system_prompt=image.system_prompt,
            prompt_mode=image.prompt_mode,
            jit_tool_exposure=image.jit_tool_exposure,
            planner=dict(image.planner),
            action_schema=dict(image.action_schema),
            default_skills=list(image.default_skills),
            default_tools=list(image.default_tools),
            context_policy=image.context_policy,
            safety_profile=image.safety_profile,
            llm_profile_id=image.llm_profile_id,
            required_capabilities=list(image.required_capabilities),
            metadata={
                **dict(image.metadata),
                "package_sha256": artifact["package_sha256"],
                "artifact_sha256": artifact_sha256,
                "artifact_bytes": artifact_bytes,
                "package_kind": _PACKAGE_BOOT_KIND,
                "package_jit_tools": [tool["name"] for tool in artifact.get("jit_tools", [])],
            },
            signature=image.signature,
            boot={
                "kind": _PACKAGE_BOOT_KIND,
                "artifact_id": artifact_id,
                "artifact_sha256": artifact_sha256,
                "package_sha256": artifact["package_sha256"],
                "workspace": artifact.get("workspace", {}),
            },
        )
        self._validate_image(image)
        created_at = utc_now()
        if self.store.get_image_artifact(artifact_id) is None:
            self.store.insert_image_artifact(
                artifact_id=artifact_id,
                kind=_PACKAGE_BOOT_KIND,
                artifact=artifact,
                sha256=artifact_sha256,
                created_by=actor,
                created_at=created_at,
                metadata={
                    "source": source,
                    "package_sha256": artifact["package_sha256"],
                    "artifact_bytes": artifact_bytes,
                    "workspace_files": artifact.get("counts", {}).get("workspace_files", 0),
                    "jit_tools": len(artifact.get("jit_tools", [])),
                },
            )
        result = self.register(
            image,
            actor=actor,
            replace=replace,
            require_capability=False,
            source=source,
        )
        self.audit.record(
            actor=actor,
            action="image.package.register",
            target=self.resource_for(image.image_id),
            decision={
                "source": source,
                "artifact_id": artifact_id,
                "artifact_sha256": artifact_sha256,
                "package_sha256": artifact["package_sha256"],
                "files": artifact.get("counts", {}).get("files", 0),
                "workspace_files": artifact.get("counts", {}).get("workspace_files", 0),
                "jit_tools": len(artifact.get("jit_tools", [])),
                "jit_tool_exposure": image.jit_tool_exposure,
                "replaced": result.replaced,
            },
        )
        return result

    def _package_validation_summary(self, image: AgentImage, artifact: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_id": image.image_id,
            "name": image.name,
            "version": image.version,
            "package_sha256": artifact["package_sha256"],
            "prompt_mode": image.prompt_mode,
            "jit_tool_exposure": image.jit_tool_exposure,
            "default_tools": list(image.default_tools),
            "default_skills": list(image.default_skills),
            "jit_tools": [tool["name"] for tool in artifact.get("jit_tools", [])],
            "workspace": artifact.get("workspace", {}),
            "counts": artifact.get("counts", {}),
        }

    def _read_host_package(self, path: str | Path) -> tuple[dict[str, bytes], str]:
        manifest_path = self._resolve_host_image_manifest(path)
        root = manifest_path.parent
        root_resolved = root.resolve()
        files: dict[str, bytes] = {}
        for file in sorted(root_resolved.rglob("*")):
            stat_result = file.lstat()
            if stat.S_ISLNK(stat_result.st_mode):
                raise ValidationError(f"image package symlinks are not supported: {file}")
            if not stat.S_ISREG(stat_result.st_mode):
                if file.exists() and not stat.S_ISDIR(stat_result.st_mode):
                    raise ValidationError(f"image package path is not a regular file or directory: {file}")
                continue
            relative = file.relative_to(root_resolved).as_posix()
            self._validate_package_relative_path(relative)
            if ".git" in Path(relative).parts:
                raise ValidationError("image packages must not include .git directories")
            files[relative] = self._read_package_file_limited(file)
        if self.config.image.package_manifest_name not in files:
            raise ValidationError(f"image package is missing {self.config.image.package_manifest_name}")
        self._validate_package_size(files)
        return files, str(root_resolved)

    def _resolve_host_image_manifest(self, path: str | Path) -> Path:
        selected = Path(path).expanduser()
        if not selected.is_absolute():
            selected = Path.cwd() / selected
        if selected.is_symlink():
            raise ValidationError(f"image package path is a symlink: {selected}")
        if selected.is_dir():
            selected = selected / self.config.image.package_manifest_name
        if selected.name != self.config.image.package_manifest_name:
            raise ValidationError(f"image package path must be a directory or {self.config.image.package_manifest_name}")
        if not selected.exists():
            raise NotFound(f"image package manifest not found: {selected}")
        if selected.is_symlink() or not selected.is_file():
            raise ValidationError(f"image package manifest is not a regular file: {selected}")
        return selected

    def _read_package_file_limited(self, path: Path) -> bytes:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValidationError(f"image package path is not a regular file: {path}")
        if before.st_nlink > 1:
            raise ValidationError(f"image package hard links are not supported: {path}")
        size = before.st_size
        if size > self.config.image.package_file_max_bytes:
            raise ValidationError(f"image package file exceeds package_file_max_bytes={self.config.image.package_file_max_bytes}: {path}")
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValidationError(f"image package symlinks are not supported: {path}") from exc
            raise
        with os.fdopen(fd, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ValidationError(f"image package path is not a regular file: {path}")
            if opened.st_nlink > 1:
                raise ValidationError(f"image package hard links are not supported: {path}")
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ValidationError(f"image package file changed during read: {path}")
            return handle.read()

    def _read_workspace_package(self, pid: str, path: str) -> tuple[dict[str, bytes], str]:
        runtime = self._runtime()
        cwd = runtime.process.working_directory(pid)
        package_root, manifest_path = self._workspace_package_paths(path)
        manifest = runtime.filesystem.read_bytes(
            pid,
            manifest_path,
            max_bytes=self.config.image.package_manifest_max_bytes,
            cwd=cwd,
        )
        if manifest.truncated:
            raise ValidationError(
                f"image package manifest exceeds package_manifest_max_bytes={self.config.image.package_manifest_max_bytes}"
            )
        _target, workspace_package_root = runtime.filesystem.resolve_path(package_root, cwd=cwd)
        files: dict[str, bytes] = {self.config.image.package_manifest_name: manifest.content}
        self._read_workspace_package_tree(pid, workspace_package_root, files)
        self._validate_package_size(files)
        return files, workspace_package_root

    def _workspace_package_paths(self, path: str) -> tuple[str, str]:
        normalized = self._normalize_package_reference(path)
        manifest_name = self.config.image.package_manifest_name
        if normalized.endswith(f"/{manifest_name}"):
            return normalized[: -len(f"/{manifest_name}")], normalized
        if normalized == manifest_name:
            return ".", manifest_name
        if Path(normalized).suffix.lower() in {".yaml", ".yml"}:
            raise ValidationError(f"image package path must be a directory or {manifest_name}")
        return normalized, self._join_relative(normalized, manifest_name)

    def _read_workspace_package_tree(
        self,
        pid: str,
        workspace_package_root: str,
        files: dict[str, bytes],
    ) -> None:
        runtime = self._runtime()
        visited_dirs: set[str] = set()

        def visit(directory: str) -> None:
            normalized_dir = directory.strip("/")
            if normalized_dir in visited_dirs:
                return
            visited_dirs.add(normalized_dir)
            listing = runtime.filesystem.read_directory(
                pid,
                normalized_dir or ".",
                limit=self.config.image.package_max_files,
                cwd=None,
            )
            if listing.truncated:
                raise ValidationError(f"image package exceeds package_max_files={self.config.image.package_max_files}")
            for entry in listing.entries:
                relative = self._workspace_package_relative_path(workspace_package_root, entry.path)
                if relative is None:
                    continue
                if ".git" in Path(relative).parts:
                    raise ValidationError("image packages must not include .git directories")
                if entry.kind == "directory":
                    visit(entry.path)
                    continue
                if entry.kind != "file":
                    raise ValidationError(f"image package path is not a regular file: {relative}")
                self._validate_package_relative_path(relative)
                if relative in files:
                    continue
                read = runtime.filesystem.read_bytes(
                    pid,
                    entry.path,
                    max_bytes=self.config.image.package_file_max_bytes,
                    cwd=None,
                )
                if read.truncated:
                    raise ValidationError(
                        f"image package file exceeds package_file_max_bytes={self.config.image.package_file_max_bytes}: {relative}"
                    )
                files[relative] = read.content
                if len(files) > self.config.image.package_max_files:
                    raise ValidationError(f"image package exceeds package_max_files={self.config.image.package_max_files}")

        visit(workspace_package_root)

    def _workspace_package_relative_path(self, workspace_package_root: str, workspace_path: str) -> str | None:
        root = workspace_package_root.strip("/")
        path = workspace_path.strip("/")
        if root in {"", "."}:
            return self._normalize_package_reference(path) if path else None
        if path == root:
            return None
        prefix = f"{root}/"
        if not path.startswith(prefix):
            return None
        return self._normalize_package_reference(path[len(prefix) :])

    def _normalize_package_files(self, files: dict[str, bytes | str]) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for path, content in files.items():
            relative = self._normalize_package_reference(path)
            self._validate_package_relative_path(relative)
            if ".git" in Path(relative).parts:
                raise ValidationError("image packages must not include .git directories")
            if isinstance(content, str):
                data = content.encode("utf-8")
            elif isinstance(content, bytes):
                data = content
            else:
                raise ValidationError(f"image package file content must be bytes or text: {relative}")
            if len(data) > self.config.image.package_file_max_bytes:
                raise ValidationError(f"image package file exceeds package_file_max_bytes={self.config.image.package_file_max_bytes}: {relative}")
            if relative in result:
                raise ValidationError(f"duplicate image package file: {relative}")
            result[relative] = data
        if self.config.image.package_manifest_name not in result:
            raise ValidationError(f"image package is missing {self.config.image.package_manifest_name}")
        self._validate_package_size(result)
        return result

    def _validate_package_size(self, files: dict[str, bytes]) -> None:
        if len(files) > self.config.image.package_max_files:
            raise ValidationError(f"image package exceeds package_max_files={self.config.image.package_max_files}")
        total = sum(len(content) for content in files.values())
        if total > self.config.image.package_max_bytes:
            raise ValidationError(f"image package exceeds package_max_bytes={self.config.image.package_max_bytes}")

    def _image_package_from_files(self, files: dict[str, bytes], *, source: str | None) -> tuple[AgentImage, dict[str, Any]]:
        manifest_name = self.config.image.package_manifest_name
        manifest_raw = files.get(manifest_name)
        if manifest_raw is None:
            raise ValidationError(f"image package is missing {manifest_name}")
        if len(manifest_raw) > self.config.image.package_manifest_hard_limit_bytes:
            raise ValidationError(
                f"image package manifest exceeds package_manifest_hard_limit_bytes={self.config.image.package_manifest_hard_limit_bytes}"
            )
        try:
            manifest_text = manifest_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(f"{manifest_name} must be UTF-8 text") from exc
        data = load_yaml_mapping(manifest_text)
        if set(data) == {"image"} and isinstance(data["image"], dict):
            data = data["image"]
        if not isinstance(data, dict):
            raise ValidationError("IMAGE.yaml must contain an image mapping")
        image_data = self._normalize_image_package_manifest(data)
        prompt_path = self._normalize_package_reference(image_data["prompt"])
        prompt_raw = files.get(prompt_path)
        if prompt_raw is None:
            raise ValidationError(f"image package prompt file is missing: {prompt_path}")
        try:
            prompt = prompt_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(f"image package prompt must be UTF-8 text: {prompt_path}") from exc
        if len(prompt) > self.config.image.prompt_max_chars:
            raise ValidationError(f"image package prompt exceeds prompt_max_chars={self.config.image.prompt_max_chars}")
        jit_tools = self._load_package_jit_tools(files, image_data.get("jit_tools"))
        jit_names = {tool.name for tool in jit_tools}
        jit_tool_exposure = (
            self._optional_string(image_data.get("jit_tool_exposure"), "jit_tool_exposure")
            or JIT_TOOL_EXPOSURE_DIRECT
        )
        if jit_tool_exposure == JIT_TOOL_EXPOSURE_MULTIPLEXED and JIT_MULTIPLEXER_TOOL_NAME in jit_names:
            raise ValidationError(f"{JIT_MULTIPLEXER_TOOL_NAME} is reserved by multiplexed JIT tool exposure")
        default_tools = [
            tool for tool in self._string_list(image_data.get("default_tools"), "default_tools")
            if tool not in jit_names
        ]
        workspace = self._coerce_package_workspace(image_data.get("workspace"))
        self._validate_package_workspace_paths(files, workspace)
        self._validate_package_artifact_scope(
            files,
            manifest_path=manifest_name,
            prompt_path=prompt_path,
            workspace=workspace,
            jit_tools_path=image_data.get("jit_tools"),
        )
        image = AgentImage(
            image_id=self._require_string(image_data["image_id"], "image_id"),
            name=self._require_string(image_data["name"], "name"),
            version=self._optional_string(image_data.get("version"), "version") or "v0",
            system_prompt=prompt,
            prompt_mode=self._optional_string(image_data.get("prompt_mode"), "prompt_mode") or PROMPT_MODE_IMAGE_ONLY,
            jit_tool_exposure=jit_tool_exposure,
            planner=self._mapping(image_data.get("planner"), "planner"),
            action_schema=self._mapping(image_data.get("action_schema"), "action_schema"),
            default_skills=self._string_list(image_data.get("default_skills"), "default_skills"),
            default_tools=default_tools,
            context_policy=self._optional_string(image_data.get("context_policy"), "context_policy") or "plan_first",
            safety_profile=self._optional_string(image_data.get("safety_profile"), "safety_profile") or "default",
            llm_profile_id=self._optional_string(image_data.get("llm_profile"), "llm_profile"),
            required_capabilities=self._capability_specs(image_data.get("required_capabilities")),
            metadata=self._mapping(image_data.get("metadata"), "metadata"),
            signature=self._optional_string(image_data.get("signature"), "signature"),
            boot={"kind": "fresh"},
        )
        self._validate_image(image)
        file_records = [self._package_file_record(path, content) for path, content in sorted(files.items())]
        package_sha256 = self._package_hash(file_records)
        workspace_source = workspace.get("source")
        workspace_files = [
            record for record in file_records
            if workspace_source and self._is_under_package_path(record["path"], workspace_source)
        ]
        artifact = {
            "artifact_version": 1,
            "kind": _PACKAGE_BOOT_KIND,
            "source": source,
            "package_sha256": package_sha256,
            "manifest_path": manifest_name,
            "prompt_path": prompt_path,
            "files": file_records,
            "jit_tools": [self._jit_tool_to_artifact(tool) for tool in jit_tools],
            "workspace": workspace,
            "counts": {
                "files": len(file_records),
                "bytes": sum(record["size_bytes"] for record in file_records),
                "workspace_files": len(workspace_files),
                "jit_tools": len(jit_tools),
            },
        }
        return image, artifact

    def _normalize_image_package_manifest(self, data: dict[str, Any]) -> dict[str, Any]:
        fields = {
            "image_id",
            "name",
            "version",
            "prompt",
            "prompt_mode",
            "jit_tool_exposure",
            "planner",
            "action_schema",
            "default_skills",
            "default_tools",
            "context_policy",
            "safety_profile",
            "llm_profile",
            "required_capabilities",
            "metadata",
            "signature",
            "jit_tools",
            "workspace",
        }
        unknown = sorted(set(data) - fields)
        if unknown:
            raise ValidationError(f"unknown IMAGE.yaml fields: {unknown}")
        missing = sorted(key for key in ["image_id", "name", "prompt"] if key not in data)
        if missing:
            raise ValidationError(f"missing required IMAGE.yaml fields: {missing}")
        return dict(data)

    def _load_package_jit_tools(self, files: dict[str, bytes], jit_tools_path: Any) -> list[JitToolSpec]:
        if jit_tools_path is None:
            return []
        path = self._normalize_package_reference(self._require_string(jit_tools_path, "jit_tools"))
        if not path.startswith(f"{self.config.image.package_tools_dir}/") or not path.endswith(".json"):
            raise ValidationError("image package jit_tools must point to tools/*.json")
        raw = files.get(path)
        if raw is None:
            raise ValidationError(f"image package jit_tools file is missing: {path}")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid image package jit_tools JSON: {path}: {exc}") from exc
        if not isinstance(data, list):
            raise ValidationError("image package jit_tools JSON must be a list")
        if len(data) > self.config.image.max_package_jit_tools:
            raise ValidationError(f"image package exceeds max_package_jit_tools={self.config.image.max_package_jit_tools}")
        result: list[JitToolSpec] = []
        names: list[str] = []
        for item in data:
            tool = self._coerce_package_jit_tool(item)
            if tool.name in names:
                raise ValidationError(f"duplicate image package JIT tool: {tool.name}")
            self._validate_package_jit_name_available(tool.name)
            names.append(tool.name)
            source_raw = files.get(tool.source_path)
            if source_raw is None:
                raise ValidationError(f"image package JIT source is missing: {tool.source_path}")
            try:
                source = source_raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValidationError(f"image package JIT source must be UTF-8 text: {tool.source_path}") from exc
            if len(source) > self.config.skills.max_jit_source_chars:
                raise ValidationError(f"image package JIT source exceeds max_jit_source_chars={self.config.skills.max_jit_source_chars}")
            self._static_check_package_jit_source(source, tool.name)
            result.append(
                JitToolSpec(
                    name=tool.name,
                    description=tool.description,
                    source_path=tool.source_path,
                    input_schema=tool.input_schema,
                    output_schema=tool.output_schema,
                    source=source,
                    tests=tool.tests,
                    metadata=tool.metadata,
                )
            )
        return result

    def _validate_package_jit_name_available(self, name: str) -> None:
        runtime = self.runtime
        tools = getattr(runtime, "tools", None)
        if tools is not None and tools._name_collides_with_static_tool(name):
            raise ValidationError(f"image package JIT tool conflicts with static tool: {name}")
        try:
            self.tool_exists(name)
        except NotFound:
            return
        raise ValidationError(f"image package JIT tool conflicts with existing tool: {name}")

    def _coerce_package_jit_tool(self, value: Any) -> JitToolSpec:
        if not isinstance(value, dict):
            raise ValidationError("image package jit_tools entries must be mappings")
        allowed = {"name", "description", "input_schema", "output_schema", "source_path", "tests", "metadata"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValidationError(f"unknown image package JIT tool fields: {unknown}")
        source_path = self._normalize_package_reference(self._require_string(value.get("source_path"), "jit_tools[].source_path"))
        expected_prefix = f"{self.config.image.package_tools_dir}/scripts/"
        if not source_path.startswith(expected_prefix) or not source_path.endswith(".ts"):
            raise ValidationError("image package JIT source_path must point to tools/scripts/*.ts")
        tests: list[dict[str, Any]] = []
        for item in value.get("tests") or []:
            if not isinstance(item, dict):
                raise ValidationError("jit_tools[].tests entries must be mappings")
            tests.append(dict(item))
        if len(tests) > self.config.tools.jit_tests_max_count:
            raise ValidationError(f"image package JIT tests exceed jit_tests_max_count={self.config.tools.jit_tests_max_count}")
        for index, test in enumerate(tests, start=1):
            ensure_json_size(test, self.config.tools.jit_test_case_max_bytes, f"image package JIT test {index}")
        name = self._require_string(value.get("name"), "jit_tools[].name")
        self._validate_openai_tool_name(name, "jit_tools[].name")
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

    def _static_check_package_jit_source(self, source: str, name: str) -> None:
        runtime = self.runtime
        sandbox = getattr(getattr(runtime, "tools", None), "sandbox", None)
        static_check = getattr(sandbox, "static_check", None)
        if not callable(static_check):
            return
        result = static_check(source)
        if not result.ok:
            raise ValidationError(f"image package JIT tool {name} failed static validation: {'; '.join(result.errors)}")

    def _coerce_package_workspace(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValidationError("workspace must be a mapping")
        allowed = {"source", "working_directory", "grants"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValidationError(f"unknown workspace fields: {unknown}")
        source = self._normalize_package_reference(
            self._optional_string(value.get("source"), "workspace.source")
            or self.config.image.package_workspace_dir
        )
        if source != self.config.image.package_workspace_dir and not source.startswith(f"{self.config.image.package_workspace_dir}/"):
            raise ValidationError("workspace.source must point inside workspace/")
        working_directory = self._normalize_package_reference(
            self._optional_string(value.get("working_directory"), "workspace.working_directory") or "."
        )
        if working_directory != "." and not self._is_under_package_path(self._join_relative(source, working_directory), source):
            raise ValidationError("workspace.working_directory must stay inside workspace.source")
        grants = self._coerce_workspace_grants(value.get("grants"), source)
        return {
            "source": source,
            "working_directory": working_directory,
            "grants": grants,
        }

    def _coerce_workspace_grants(self, value: Any, source: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError("workspace.grants must be a list")
        if len(value) > self.config.image.max_workspace_grants:
            raise ValidationError(f"workspace.grants exceeds max_workspace_grants={self.config.image.max_workspace_grants}")
        grants: list[dict[str, Any]] = []
        allowed_rights = {CapabilityRight.READ.value, CapabilityRight.WRITE.value, CapabilityRight.DELETE.value}
        for item in value:
            if not isinstance(item, dict):
                raise ValidationError("workspace.grants entries must be mappings")
            unknown = sorted(set(item) - {"path", "rights", "recursive", "delegable"})
            if unknown:
                raise ValidationError(f"unknown workspace grant fields: {unknown}")
            grant_path = self._normalize_package_reference(self._optional_string(item.get("path"), "workspace.grants[].path") or ".")
            target = source if grant_path == "." else self._join_relative(source, grant_path)
            if not self._is_under_package_path(target, source):
                raise ValidationError("workspace grant path must stay inside workspace.source")
            rights = self._string_list(item.get("rights"), "workspace.grants[].rights")
            if not rights or any(right not in allowed_rights for right in rights):
                raise ValidationError("workspace grants may include only read, write, and delete rights")
            grants.append(
                {
                    "path": grant_path,
                    "rights": sorted(set(rights)),
                    "recursive": bool(item.get("recursive", False)),
                    "delegable": bool(item.get("delegable", False)),
                }
            )
        return grants

    def _validate_package_workspace_paths(self, files: dict[str, bytes], workspace: dict[str, Any]) -> None:
        source = workspace.get("source")
        if not source:
            return
        file_paths = set(files)
        if source in file_paths:
            raise ValidationError("workspace.source must point to a directory, not a file")
        working_directory = str(workspace.get("working_directory") or ".")
        working_directory_path = source if working_directory == "." else self._join_relative(source, working_directory)
        if working_directory_path in file_paths:
            raise ValidationError("workspace.working_directory must point to a directory, not a file")

    def _validate_package_artifact_scope(
        self,
        files: dict[str, bytes],
        *,
        manifest_path: str,
        prompt_path: str,
        workspace: dict[str, Any],
        jit_tools_path: Any,
    ) -> None:
        """Keep image artifacts limited to package-declared content.

        Registration reads a directory tree to discover the manifest and its
        referenced files, but artifact persistence should not silently capture
        unrelated local files such as notes, caches, or secrets that happened to
        be placed next to IMAGE.yaml.
        """
        roots = [self.config.image.package_resources_dir]
        if workspace.get("source"):
            roots.append(str(workspace["source"]))
        if jit_tools_path is not None:
            roots.append(self.config.image.package_tools_dir)
        allowed_files = {manifest_path, prompt_path}
        unexpected = [
            path
            for path in sorted(files)
            if path not in allowed_files and not any(self._is_under_package_path(path, root) for root in roots)
        ]
        if unexpected:
            raise ValidationError(f"image package contains undeclared files: {unexpected[:5]}")

    def _package_file_record(self, path: str, content: bytes) -> dict[str, Any]:
        sha = hashlib.sha256(content).hexdigest()
        record: dict[str, Any] = {
            "path": path,
            "size_bytes": len(content),
            "sha256": sha,
        }
        try:
            record["kind"] = "text"
            record["content"] = content.decode("utf-8")
        except UnicodeDecodeError:
            record["kind"] = "base64"
            record["content_base64"] = base64.b64encode(content).decode("ascii")
        return record

    def _package_hash(self, file_records: list[dict[str, Any]]) -> str:
        canonical = [
            {"path": record["path"], "size_bytes": record["size_bytes"], "sha256": record["sha256"]}
            for record in file_records
        ]
        return hashlib.sha256(dumps({"kind": _PACKAGE_BOOT_KIND, "files": canonical}).encode("utf-8")).hexdigest()

    def _jit_tool_to_artifact(self, tool: JitToolSpec) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "source_path": tool.source_path,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "source": tool.source,
            "tests": list(tool.tests),
            "metadata": dict(tool.metadata),
        }

    def _normalize_package_reference(self, path: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise ValidationError("image package path must be a non-empty string")
        raw = path.replace("\\", "/").strip()
        if raw.startswith("/"):
            raise ValidationError(f"image package path must be relative: {path!r}")
        parts: list[str] = []
        for part in raw.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                raise ValidationError(f"image package path escapes package root: {path!r}")
            parts.append(part)
        normalized = "/".join(parts) or "."
        if normalized != ".":
            self._validate_package_path_safety(normalized, original=path)
        return normalized

    def _validate_package_relative_path(self, path: str) -> None:
        normalized = self._normalize_package_reference(path)
        if normalized != path:
            raise ValidationError(f"image package path must be normalized: {path!r}")
        if normalized == ".":
            raise ValidationError("image package file path cannot be package root")
        if any(ord(char) < 32 for char in normalized):
            raise ValidationError(f"image package path contains control characters: {path!r}")

    def _validate_package_path_safety(self, normalized: str, *, original: str) -> None:
        for part in normalized.split("/"):
            lower = part.lower()
            stem = part.split(".", 1)[0].upper()
            if any(char in _WINDOWS_FORBIDDEN_PATH_CHARS for char in part):
                raise ValidationError(f"image package path contains a Windows-unsafe character: {original!r}")
            if part.endswith((" ", ".")):
                raise ValidationError(f"image package path contains a Windows-unsafe segment: {original!r}")
            if stem in _WINDOWS_RESERVED_PATH_NAMES:
                raise ValidationError(f"image package path uses a reserved Windows device name: {original!r}")
            if lower in _CACHE_PACKAGE_SEGMENTS:
                raise ValidationError(f"image package must not include cache or VCS paths: {original!r}")
            if lower in _SENSITIVE_PACKAGE_FILENAMES or lower.endswith(_SENSITIVE_PACKAGE_SUFFIXES):
                raise ValidationError(f"image package must not include likely secret material: {original!r}")

    def _join_relative(self, base: str, path: str) -> str:
        normalized_base = self._normalize_package_reference(base)
        normalized_path = self._normalize_package_reference(path)
        if normalized_base == ".":
            return normalized_path
        if normalized_path == ".":
            return normalized_base
        return f"{normalized_base}/{normalized_path}"

    def _is_under_package_path(self, path: str, root: str) -> bool:
        normalized_path = self._normalize_package_reference(path)
        normalized_root = self._normalize_package_reference(root)
        return normalized_path == normalized_root or normalized_path.startswith(f"{normalized_root}/")

    def _runtime(self) -> Any:
        if self.runtime is None:
            raise ValidationError("image package operation requires a bound Runtime")
        return self.runtime

    def load_persisted_images(self) -> None:
        if self.store is None:
            return
        for image, _metadata in self.store.list_images():
            # Persisted images may depend on startup modules that are not loaded
            # in this Runtime.open() invocation. Keep the manifest inspectable
            # and defer concrete tool resolution to spawn/exec.
            self._validate_persisted_image(image)
            self.images[image.image_id] = image

    def list_images(self) -> list[dict[str, Any]]:
        if self.store is None:
            return [self._image_summary(image, {}) for image in sorted(self.images.values(), key=lambda item: item.image_id)]
        return [
            self._image_summary(self._validate_persisted_image(image), metadata)
            for image, metadata in self.store.list_images()
        ]

    def inspect(self, image_id: str) -> dict[str, Any]:
        image = self.images.get(image_id)
        metadata: dict[str, Any] = {}
        if self.store is not None:
            persisted = self.store.get_image(image_id)
            if persisted is not None:
                image, metadata = persisted
                image = self._validate_persisted_image(image)
        if image is None:
            raise NotFound(f"agent image not found: {image_id}")
        artifact = None
        boot = image.boot or {"kind": "fresh"}
        if boot.get("kind") in {_CHECKPOINT_BOOT_KIND, _PACKAGE_BOOT_KIND} and self.store is not None:
            found = self.store.get_image_artifact(str(boot.get("artifact_id")))
            if found is not None:
                artifact_data, artifact_meta = found
                if boot.get("kind") == _CHECKPOINT_BOOT_KIND:
                    artifact = {
                        **artifact_meta,
                        "source_checkpoint_id": artifact_data.get("source_checkpoint_id"),
                        "source_pid": artifact_data.get("source_pid"),
                        "counts": artifact_data.get("counts", {}),
                        "modules": artifact_data.get("modules", []),
                    }
                else:
                    artifact = {
                        **artifact_meta,
                        "source": artifact_data.get("source"),
                        "package_sha256": artifact_data.get("package_sha256"),
                        "counts": artifact_data.get("counts", {}),
                        "workspace": artifact_data.get("workspace", {}),
                        "jit_tools": [tool.get("name") for tool in artifact_data.get("jit_tools", [])],
                    }
        return {
            "image": self._image_to_dict(image),
            "registry": metadata,
            "artifact": artifact,
        }

    def _validate_persisted_image(self, image: AgentImage) -> AgentImage:
        # Persisted images can outlive code upgrades or manual DB repairs. Keep
        # startup/module tool resolution deferred, but fail closed on malformed
        # model fields before the image reaches spawn or prompt selection.
        self._validate_image(image, validate_tools=False)
        return image

    def commit_from_checkpoint(
        self,
        *,
        actor: str,
        checkpoint_id: str,
        image_id: str,
        name: str,
        version: str = "v0",
        replace: bool = False,
        metadata: dict[str, Any] | None = None,
        require_capability: bool = True,
    ) -> ImageRegistrationResult:
        if self.runtime is None or self.store is None:
            raise ValidationError("checkpoint image commit requires a bound Runtime and SQLiteStore")
        found = self.store.get_checkpoint_snapshot(checkpoint_id)
        if found is None:
            raise NotFound(f"checkpoint not found: {checkpoint_id}")
        checkpoint, snapshot = found
        if require_capability:
            self.capabilities.require(actor, self.resource_for(image_id), CapabilityRight.WRITE)
            self.runtime.checkpoint._require_checkpoint_or_process_read(actor, checkpoint)
        self.runtime.checkpoint._require_snapshot_modules(snapshot)
        self._validate_identifier(image_id, "image_id", self.config.image.id_max_chars)
        self._validate_string_length(name, "name", self.config.image.name_max_chars)
        self._validate_string_length(version, "version", self.config.image.version_max_chars)
        if image_id in self.images and not replace:
            raise ValidationError(f"agent image already exists: {image_id}")
        artifact = self._build_commit_artifact(snapshot, checkpoint_id=checkpoint_id)
        artifact_json = dumps(artifact)
        artifact_bytes = len(artifact_json.encode("utf-8"))
        if artifact_bytes > self.config.image_commit.artifact_hard_limit_bytes:
            raise ValidationError(
                "image artifact exceeded "
                f"artifact_hard_limit_bytes={self.config.image_commit.artifact_hard_limit_bytes}"
            )
        artifact_sha256 = hashlib.sha256(artifact_json.encode("utf-8")).hexdigest()
        artifact_id = f"imgart_{artifact_sha256[:24]}"
        created_at = utc_now()
        if self.store.get_image_artifact(artifact_id) is None:
            self.store.insert_image_artifact(
                artifact_id=artifact_id,
                kind="checkpoint_commit",
                artifact=artifact,
                sha256=artifact_sha256,
                created_by=actor,
                created_at=created_at,
                metadata={
                    "source_checkpoint_id": checkpoint_id,
                    "source_pid": checkpoint.pid,
                    "artifact_bytes": artifact_bytes,
                },
            )
        source_image = self.images.get(str(artifact["source_image_id"]))
        image = AgentImage(
            image_id=image_id,
            name=name,
            version=version,
            system_prompt=source_image.system_prompt if source_image is not None else "",
            prompt_mode=source_image.prompt_mode if source_image is not None else PROMPT_MODE_IMAGE_ONLY,
            jit_tool_exposure=source_image.jit_tool_exposure if source_image is not None else JIT_TOOL_EXPOSURE_DIRECT,
            planner=dict(source_image.planner) if source_image is not None else {},
            action_schema=dict(source_image.action_schema) if source_image is not None else {},
            default_skills=list(artifact.get("default_skills", [])),
            default_tools=list(artifact.get("static_default_tools", [])),
            context_policy=source_image.context_policy if source_image is not None else "plan_first",
            safety_profile=source_image.safety_profile if source_image is not None else "default",
            llm_profile_id=source_image.llm_profile_id if source_image is not None else None,
            required_capabilities=self._dedupe_capability_specs(artifact.get("required_capabilities", [])),
            metadata={
                **(metadata or {}),
                "committed_from_checkpoint": checkpoint_id,
                "committed_from_pid": checkpoint.pid,
                "artifact_sha256": artifact_sha256,
                "artifact_bytes": artifact_bytes,
                "commit_kind": "checkpoint_commit",
            },
            boot={
                "kind": "checkpoint_commit",
                "artifact_id": artifact_id,
                "artifact_sha256": artifact_sha256,
                "source_checkpoint_id": checkpoint_id,
                "source_pid": checkpoint.pid,
                "root_only": True,
            },
        )
        result = self.register(
            image,
            actor=actor,
            replace=replace,
            require_capability=False,
            source=f"checkpoint:{checkpoint_id}",
        )
        self.events.emit(
            EventType.IMAGE_COMMITTED,
            source=actor,
            target=self.resource_for(image_id),
            payload={
                "image_id": image_id,
                "checkpoint_id": checkpoint_id,
                "artifact_id": artifact_id,
                "artifact_sha256": artifact_sha256,
                "artifact_bytes": artifact_bytes,
            },
        )
        self.audit.record(
            actor=actor,
            action="image.commit",
            target=self.resource_for(image_id),
            decision={
                "checkpoint_id": checkpoint_id,
                "source_pid": checkpoint.pid,
                "artifact_id": artifact_id,
                "artifact_sha256": artifact_sha256,
                "artifact_bytes": artifact_bytes,
                "required_capabilities": len(image.required_capabilities),
            },
        )
        return result

    def grant_register(
        self,
        pid: str,
        image_id: str = "*",
        issued_by: str = "image_registry",
    ) -> Capability:
        resource = self.config.image.registry_resource if image_id == "*" else self.resource_for(image_id)
        return self.capabilities.grant(
            subject=pid,
            resource=resource,
            rights=[CapabilityRight.WRITE],
            issued_by=issued_by,
        )

    def resource_for(self, image_id: str) -> str:
        return f"image:{image_id}"

    def registry_resource(self) -> str:
        return self.config.image.registry_resource

    def _coerce_image(self, image: AgentImage | dict[str, Any]) -> AgentImage:
        if isinstance(image, AgentImage):
            return image
        if not isinstance(image, dict):
            raise ValidationError("image registration requires an AgentImage or mapping")
        unknown = sorted(set(image) - self.IMAGE_FIELDS)
        if unknown:
            raise ValidationError(f"unknown AgentImage fields: {unknown}")
        required = {"image_id", "name"}
        missing = sorted(key for key in required if key not in image)
        if missing:
            raise ValidationError(f"missing required AgentImage fields: {missing}")
        return AgentImage(
            image_id=self._require_string(image["image_id"], "image_id"),
            name=self._require_string(image["name"], "name"),
            version=self._optional_string(image.get("version"), "version") or "v0",
            system_prompt=self._optional_text(image.get("system_prompt"), "system_prompt") or "",
            prompt_mode=self._optional_string(image.get("prompt_mode"), "prompt_mode") or PROMPT_MODE_IMAGE_ONLY,
            jit_tool_exposure=self._optional_string(image.get("jit_tool_exposure"), "jit_tool_exposure")
            or JIT_TOOL_EXPOSURE_DIRECT,
            planner=self._mapping(image.get("planner"), "planner"),
            action_schema=self._mapping(image.get("action_schema"), "action_schema"),
            default_skills=self._string_list(image.get("default_skills"), "default_skills"),
            default_tools=self._string_list(image.get("default_tools"), "default_tools"),
            context_policy=self._optional_string(image.get("context_policy"), "context_policy") or "plan_first",
            safety_profile=self._optional_string(image.get("safety_profile"), "safety_profile") or "default",
            llm_profile_id=self._optional_string(image.get("llm_profile_id"), "llm_profile_id"),
            required_capabilities=self._capability_specs(image.get("required_capabilities")),
            metadata=self._mapping(image.get("metadata"), "metadata"),
            signature=self._optional_string(image.get("signature"), "signature"),
            boot=self._boot_mapping(image.get("boot")),
        )

    def _validate_image(self, image: AgentImage, *, validate_tools: bool = True) -> None:
        self._validate_identifier(image.image_id, "image_id", self.config.image.id_max_chars)
        self._validate_string_length(image.name, "name", self.config.image.name_max_chars)
        self._validate_string_length(image.version, "version", self.config.image.version_max_chars)
        if image.prompt_mode not in PROMPT_MODES:
            raise ValidationError(f"unknown prompt_mode: {image.prompt_mode}")
        if image.jit_tool_exposure not in JIT_TOOL_EXPOSURES:
            raise ValidationError(f"unknown jit_tool_exposure: {image.jit_tool_exposure}")
        if image.jit_tool_exposure == JIT_TOOL_EXPOSURE_MULTIPLEXED and JIT_MULTIPLEXER_TOOL_NAME in image.default_tools:
            raise ValidationError(f"{JIT_MULTIPLEXER_TOOL_NAME} is reserved by multiplexed JIT tool exposure")
        if image.llm_profile_id is not None:
            self._validate_string_length(image.llm_profile_id, "llm_profile_id", self.config.image.id_max_chars)
            if not image.llm_profile_id.strip():
                raise ValidationError("llm_profile_id must be non-empty when provided")
        if len(image.system_prompt) > self.config.image.prompt_max_chars:
            raise ValidationError(f"system_prompt exceeds prompt_max_chars={self.config.image.prompt_max_chars}")
        self._validate_mapping_size(image.planner, "planner")
        self._validate_mapping_size(image.action_schema, "action_schema")
        self._validate_mapping_size(image.metadata, "metadata")
        self._validate_mapping_size(image.boot, "boot")
        manifest_bytes = len(dumps(self._image_to_dict(image)).encode("utf-8"))
        if manifest_bytes > self.config.image.manifest_hard_limit_bytes:
            raise ValidationError(
                f"AgentImage manifest exceeds manifest_hard_limit_bytes={self.config.image.manifest_hard_limit_bytes}"
            )
        if len(image.default_tools) > self.config.image.max_default_tools:
            raise ValidationError(f"default_tools exceeds max_default_tools={self.config.image.max_default_tools}")
        if len(image.default_skills) > self.config.skills.max_tools:
            raise ValidationError(f"default_skills exceeds max_tools={self.config.skills.max_tools}")
        if len(image.required_capabilities) > self.config.image.max_required_capabilities:
            raise ValidationError(
                "required_capabilities exceeds "
                f"max_required_capabilities={self.config.image.max_required_capabilities}"
            )
        for skill_id in image.default_skills:
            self._validate_identifier(skill_id, "default_skills[]", self.config.skills.id_max_chars)
        for tool_name in image.default_tools:
            self._validate_identifier(tool_name, "default_tools[]", self.config.image.id_max_chars)
            if not validate_tools:
                continue
            try:
                self.tool_exists(tool_name)
            except Exception as exc:
                raise ValidationError(f"unknown tool in AgentImage default_tools: {tool_name}") from exc
        for spec in image.required_capabilities:
            self._validate_capability_spec(spec)
        self._validate_boot(image.boot)

    def _validate_identifier(self, value: str, field: str, max_chars: int) -> None:
        self._validate_string_length(value, field, max_chars)
        if not _IMAGE_ID_PATTERN.match(value):
            raise ValidationError(f"{field} contains unsupported characters: {value!r}")

    def _validate_openai_tool_name(self, value: str, field: str) -> None:
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
        if not isinstance(value, str) or not value:
            raise ValidationError(f"{field} must be a non-empty string")
        if len(value) > max_chars:
            raise ValidationError(f"{field} exceeds max length {max_chars}")
        if any(ord(char) < 32 for char in value):
            raise ValidationError(f"{field} contains control characters")

    def _validate_mapping_size(self, value: dict[str, Any], field: str) -> None:
        size = len(dumps(value).encode("utf-8"))
        if size > self.config.image.structured_field_hard_limit_bytes:
            raise ValidationError(
                f"{field} exceeds structured_field_hard_limit_bytes="
                f"{self.config.image.structured_field_hard_limit_bytes}"
            )

    def _require_string(self, value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{field} must be a non-empty string")
        return value.strip()

    def _optional_string(self, value: Any, field: str) -> str | None:
        if value is None:
            return None
        return self._require_string(value, field)

    def _optional_text(self, value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValidationError(f"{field} must be a string")
        return value

    def _string_list(self, value: Any, field: str) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError(f"{field} must be a list")
        return [self._require_string(item, f"{field}[]") for item in value]

    def _mapping(self, value: Any, field: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValidationError(f"{field} must be a mapping")
        return dict(value)

    def _boot_mapping(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {"kind": "fresh"}
        if not isinstance(value, dict):
            raise ValidationError("boot must be a mapping")
        return dict(value)

    def _validate_boot(self, boot: dict[str, Any]) -> None:
        kind = boot.get("kind", "fresh")
        if kind not in {"fresh", _CHECKPOINT_BOOT_KIND, _PACKAGE_BOOT_KIND}:
            raise ValidationError(f"unsupported image boot kind: {kind}")
        if kind == "fresh":
            return
        artifact_id = boot.get("artifact_id")
        artifact_sha256 = boot.get("artifact_sha256")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValidationError(f"{kind} boot requires artifact_id")
        if not isinstance(artifact_sha256, str) or not artifact_sha256:
            raise ValidationError(f"{kind} boot requires artifact_sha256")

    def _capability_specs(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError("required_capabilities must be a list")
        specs: list[dict[str, Any]] = []
        for spec in value:
            if not isinstance(spec, dict):
                raise ValidationError("required_capabilities entries must be mappings")
            normalized = dict(spec)
            self._validate_capability_spec(normalized)
            specs.append(normalized)
        return specs

    def _validate_capability_spec(self, spec: dict[str, Any]) -> None:
        resource = spec.get("resource")
        if not isinstance(resource, str) or not resource:
            raise ValidationError("capability spec requires a non-empty resource")
        try:
            self.capabilities.parse_resource_pattern(resource)
        except CapabilityDenied as exc:
            raise ValidationError(str(exc)) from exc
        rights = spec.get("rights")
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

    def _build_commit_artifact(self, snapshot: dict[str, Any], *, checkpoint_id: str) -> dict[str, Any]:
        source_pid = str(snapshot["pid"])
        process_rows = [row for row in snapshot.get("rows", {}).get("processes", []) if row["pid"] == source_pid]
        if not process_rows:
            raise ValidationError(f"checkpoint root process row is missing: {source_pid}")
        process_row = dict(process_rows[0])
        # Resource constraints are launch-time policy, not image state. A
        # checkpoint-committed image may replay reconstructable memory/tool
        # context, but it must not carry the source process budget or usage into
        # a newly started process.
        process_row.pop("resource_budget_json", None)
        process_row.pop("resource_usage_json", None)
        object_oids = self._root_object_oids(snapshot, process_row, source_pid)
        object_rows = [row for row in snapshot.get("rows", {}).get("objects", []) if row["oid"] in object_oids]
        namespace_names = self._root_namespace_names(snapshot, object_rows, source_pid)
        tool_table = loads(process_row.get("tool_table_json"), {})
        source_image_id = str(process_row.get("image_id"))
        source_image = self.images.get(source_image_id)
        static_tool_names = self._static_tool_names_for_commit(tool_table)
        if len(tool_table) > self.config.image_commit.max_committed_tools:
            raise ValidationError(
                f"committed tool table exceeds max_committed_tools={self.config.image_commit.max_committed_tools}"
            )
        required_capabilities, internal_capabilities = self._split_commit_capabilities(
            snapshot.get("rows", {}).get("capabilities", []),
            source_pid=source_pid,
            object_oids=object_oids,
            namespace_names=namespace_names,
        )
        object_payloads: dict[str, Any] = {}
        for oid in object_oids:
            payload = deepcopy(snapshot.get("object_payloads", {}).get(oid))
            payload_bytes = len(dumps(payload).encode("utf-8"))
            if payload_bytes > self.config.image_commit.payload_capture_limit_bytes:
                raise ValidationError(
                    f"object payload {oid} exceeds image_commit.payload_capture_limit_bytes="
                    f"{self.config.image_commit.payload_capture_limit_bytes}"
                )
            object_payloads[oid] = payload
        visible_tool_ids = set(tool_table.values())
        jit_sources = {
            tool_id: source
            for tool_id, source in snapshot.get("jit_sources", {}).items()
            if tool_id in visible_tool_ids
        }
        if len(jit_sources) > self.config.image_commit.max_committed_jit_sources:
            raise ValidationError(
                "committed JIT sources exceed "
                f"max_committed_jit_sources={self.config.image_commit.max_committed_jit_sources}"
            )
        return {
            "artifact_version": self.config.image_commit.artifact_version,
            "kind": "checkpoint_commit",
            "source_checkpoint_id": checkpoint_id,
            "source_pid": source_pid,
            "source_image_id": source_image_id,
            "source_process": process_row,
            "rows": {
                "object_namespaces": [
                    row for row in snapshot.get("rows", {}).get("object_namespaces", [])
                    if row["namespace"] in namespace_names
                ],
                "objects": object_rows,
                "object_links": [
                    row for row in snapshot.get("rows", {}).get("object_links", [])
                    if row["src_oid"] in object_oids and row["dst_oid"] in object_oids
                ],
                "capabilities": internal_capabilities,
                "skills": snapshot.get("rows", {}).get("skills", []),
                "tools": [
                    row for row in snapshot.get("rows", {}).get("tools", [])
                    if row["tool_id"] in visible_tool_ids
                ],
                "tool_candidates": [
                    row for row in snapshot.get("rows", {}).get("tool_candidates", [])
                    if row["pid"] == source_pid
                ],
            },
            "object_oids": sorted(object_oids),
            "namespaces": sorted(namespace_names),
            "object_payloads": object_payloads,
            "tool_table": tool_table,
            "loaded_skills": loads(process_row.get("loaded_skills_json"), {}),
            "jit_sources": jit_sources,
            "working_directory": process_row.get("working_directory", "."),
            "default_skills": list(source_image.default_skills) if source_image is not None else [],
            "static_default_tools": static_tool_names,
            "required_capabilities": required_capabilities,
            "modules": list(snapshot.get("modules", [])),
            "counts": {
                "objects": len(object_rows),
                "namespaces": len(namespace_names),
                "internal_capabilities": len(internal_capabilities),
                "required_capabilities": len(required_capabilities),
                "tools": len(tool_table),
                "jit_sources": len(jit_sources),
            },
        }

    def _root_object_oids(self, snapshot: dict[str, Any], process_row: dict[str, Any], source_pid: str) -> set[str]:
        oids: set[str] = set()
        if process_row.get("goal_oid"):
            oids.add(str(process_row["goal_oid"]))
        view = loads(process_row.get("memory_view_json"), {}) if process_row.get("memory_view_json") else {}
        for root in view.get("roots", []):
            if isinstance(root, dict) and root.get("oid"):
                oids.add(str(root["oid"]))
        for row in snapshot.get("rows", {}).get("objects", []):
            if row.get("owner_kind") == ObjectOwnerKind.PROCESS.value and row.get("owner_id") == source_pid:
                oids.add(str(row["oid"]))
        available = set(snapshot.get("object_payloads", {}).keys())
        return {oid for oid in oids if oid in available}

    def _root_namespace_names(self, snapshot: dict[str, Any], object_rows: list[dict[str, Any]], source_pid: str) -> set[str]:
        process_namespace = f"{self.config.memory.process_namespace_prefix}:{source_pid}"
        names = {process_namespace}
        names.update(str(row["namespace"]) for row in object_rows)
        for row in snapshot.get("rows", {}).get("object_namespaces", []):
            if row.get("created_by") == source_pid or row.get("namespace") in names:
                names.add(str(row["namespace"]))
        return names

    def _split_commit_capabilities(
        self,
        rows: list[dict[str, Any]],
        *,
        source_pid: str,
        object_oids: set[str],
        namespace_names: set[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        required: list[dict[str, Any]] = []
        internal: list[dict[str, Any]] = []
        for row in rows:
            if row.get("subject") != source_pid or row.get("status") != "active":
                continue
            resource = str(row.get("resource") or "")
            rights = loads(row.get("rights_json"), [])
            if self._is_internal_commit_capability(resource, object_oids, namespace_names):
                internal.append(dict(row))
            else:
                # Unknown provider resource kinds are still external authority.
                # Preserve them as declarations, never as live boot grants.
                required.append(
                    {
                        "resource": resource,
                        "rights": rights,
                        "constraints": loads(row.get("constraints_json"), {}),
                    }
                )
        return self._dedupe_capability_specs(required), internal

    def _is_internal_commit_capability(self, resource: str, object_oids: set[str], namespace_names: set[str]) -> bool:
        if resource.startswith("object:"):
            return resource.split(":", 1)[1] in object_oids
        if resource.startswith("object_namespace:"):
            return resource.split(":", 1)[1] in namespace_names
        return False

    def _static_tool_names_for_commit(self, tool_table: dict[str, str]) -> list[str]:
        static: list[str] = []
        runtime = self.runtime
        tools = getattr(runtime, "tools", None)
        if tools is None:
            return static
        jit_sources = getattr(tools, "_jit_sources", {})
        for name, tool_id in sorted(tool_table.items()):
            if tool_id in jit_sources:
                continue
            try:
                tools.resolve(name)
            except Exception as exc:
                raise ValidationError(f"committed image references unavailable static tool: {name}") from exc
            static.append(name)
        return static

    def _dedupe_capability_specs(self, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_key: dict[str, dict[str, Any]] = {}
        for spec in specs:
            normalized = {
                "resource": spec["resource"],
                "rights": sorted(str(right) for right in spec.get("rights", [])),
            }
            constraints = spec.get("constraints") or {}
            if constraints:
                normalized["constraints"] = constraints
            key = dumps(normalized)
            by_key[key] = normalized
        result = list(by_key.values())
        if len(result) > self.config.image_commit.max_required_capabilities:
            raise ValidationError(
                "committed required_capabilities exceeds "
                f"max_required_capabilities={self.config.image_commit.max_required_capabilities}"
            )
        return result

    def _image_to_dict(self, image: AgentImage) -> dict[str, Any]:
        return {
            "image_id": image.image_id,
            "name": image.name,
            "version": image.version,
            "system_prompt": image.system_prompt,
            "prompt_mode": image.prompt_mode,
            "jit_tool_exposure": image.jit_tool_exposure,
            "planner": image.planner,
            "action_schema": image.action_schema,
            "default_skills": list(image.default_skills),
            "default_tools": list(image.default_tools),
            "context_policy": image.context_policy,
            "safety_profile": image.safety_profile,
            "llm_profile_id": image.llm_profile_id,
            "required_capabilities": list(image.required_capabilities),
            "metadata": dict(image.metadata),
            "signature": image.signature,
            "boot": dict(image.boot),
        }

    def _image_summary(self, image: AgentImage, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_id": image.image_id,
            "name": image.name,
            "version": image.version,
            "boot_kind": (image.boot or {}).get("kind", "fresh"),
            "prompt_mode": image.prompt_mode,
            "jit_tool_exposure": image.jit_tool_exposure,
            "llm_profile_id": image.llm_profile_id,
            "default_tools": list(image.default_tools),
            "default_skills": list(image.default_skills),
            "required_capabilities_count": len(image.required_capabilities),
            **metadata,
        }
