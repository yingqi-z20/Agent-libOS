from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import re
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.modules.schema import ModuleManifest, ModuleProvides, ModuleSource
from agent_libos.utils.yaml_loader import load_yaml_mapping

_MODULE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
_PYTHON_OBJECT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PYTHON_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_SYSCALL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_HEX_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
_IMPORT_LOCK = threading.RLock()


class _FreshSourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        source_bytes = self.get_data(self.path)
        return self.source_to_code(source_bytes, self.path)


class ModuleLoader:
    """Loads trusted startup module manifests and Python entrypoints."""

    MANIFEST_FIELDS = {
        "schema_version",
        "module_id",
        "name",
        "version",
        "entrypoint",
        "provides",
        "metadata",
        "sha256",
    }
    PROVIDES_FIELDS = {"tools", "images", "syscalls", "provider_hooks", "startup_hooks"}

    def __init__(
        self,
        config: AgentLibOSConfig | None = None,
        *,
        trusted_modules: tuple[str, ...] = (),
        trusted_sha256: tuple[str, ...] = (),
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.trusted_modules = tuple(self.config.modules.trusted_modules) + tuple(trusted_modules)
        self.trusted_sha256 = tuple(self.config.modules.trusted_sha256) + tuple(trusted_sha256)

    def load(self, manifest_path: str | Path) -> tuple[ModuleSource, Any]:
        source = self.resolve(manifest_path)
        if not self.is_trusted(source.manifest.module_id, source.source_sha256):
            raise CapabilityDenied(
                "startup module is not trusted: "
                f"{source.manifest.module_id}:{source.source_sha256}"
            )
        return source, self.import_entrypoint(source)

    def verify(self, manifest_path: str | Path) -> dict[str, Any]:
        source = self.resolve(manifest_path)
        return {
            "module_id": source.manifest.module_id,
            "name": source.manifest.name,
            "version": source.manifest.version,
            "entrypoint": source.manifest.entrypoint,
            "manifest_path": source.manifest_path,
            "manifest_sha256": source.manifest_sha256,
            "source_path": source.source_path,
            "source_sha256": source.source_sha256,
            "trusted": self.is_trusted(source.manifest.module_id, source.source_sha256),
            "provides": {
                "tools": list(source.manifest.provides.tools),
                "images": list(source.manifest.provides.images),
                "syscalls": list(source.manifest.provides.syscalls),
                "provider_hooks": list(source.manifest.provides.provider_hooks),
                "startup_hooks": list(source.manifest.provides.startup_hooks),
            },
        }

    def resolve(self, manifest_path: str | Path) -> ModuleSource:
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise NotFound(f"module manifest not found: {manifest_path}")
        text = self._read_manifest(path)
        manifest = self.parse_manifest(text)
        source_path, entrypoint_object = self._resolve_entrypoint(path, manifest.entrypoint)
        source_bytes = self._read_source_bytes(source_path)
        source_sha = self._sha256_bytes(source_bytes)
        expected_sha = manifest.sha256.lower()
        if source_sha != expected_sha:
            raise ValidationError(
                "module source sha256 mismatch: "
                f"expected {expected_sha}, got {source_sha}"
            )
        return ModuleSource(
            manifest=manifest,
            manifest_path=str(path),
            manifest_sha256=self._sha256_bytes(text.encode("utf-8")),
            source_path=str(source_path),
            source_sha256=source_sha,
            entrypoint_object=entrypoint_object,
            source_bytes=source_bytes,
        )

    def parse_manifest(self, text: str) -> ModuleManifest:
        if len(text.encode("utf-8")) > self.config.modules.manifest_hard_limit_bytes:
            raise ValidationError(
                "module manifest exceeded "
                f"manifest_hard_limit_bytes={self.config.modules.manifest_hard_limit_bytes}"
            )
        data = self._load_mapping(text)
        if set(data) == {"module"} and isinstance(data["module"], dict):
            data = dict(data["module"])
        unknown = sorted(set(data) - self.MANIFEST_FIELDS)
        if unknown:
            raise ValidationError(f"unknown module manifest fields: {unknown}")
        missing = sorted(field for field in ["schema_version", "module_id", "name", "entrypoint", "provides", "sha256"] if field not in data)
        if missing:
            raise ValidationError(f"missing required module manifest fields: {missing}")
        schema_version = data["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValidationError("schema_version must be an integer")
        if schema_version != self.config.modules.schema_version:
            raise ValidationError(f"unsupported module schema_version: {schema_version}")
        provides = self._coerce_provides(data["provides"])
        manifest = ModuleManifest(
            schema_version=int(schema_version),
            module_id=self._identifier(data["module_id"], "module_id", self.config.modules.id_max_chars),
            name=self._string(data["name"], "name", self.config.modules.name_max_chars),
            version=self._string(data.get("version") or "v0", "version", self.config.modules.version_max_chars),
            entrypoint=self._string(data["entrypoint"], "entrypoint", self.config.modules.entrypoint_max_chars),
            provides=provides,
            sha256=self._sha256(data["sha256"], "sha256"),
            metadata=self._mapping(data.get("metadata"), "metadata"),
        )
        self._validate_provides(manifest.provides)
        return manifest

    def import_entrypoint(self, source: ModuleSource) -> Any:
        module_ref, object_name = self._split_entrypoint(source.manifest.entrypoint)
        with _IMPORT_LOCK:
            module = self._import_file(
                Path(source.source_path),
                source.manifest.module_id,
                source.source_sha256,
                source.source_bytes,
            )
        self._verify_imported_module_source(module, source)
        entrypoint = getattr(module, object_name, None)
        if not callable(entrypoint):
            raise ValidationError(f"module entrypoint is not callable: {source.manifest.entrypoint}")
        return entrypoint

    def is_trusted(self, module_id: str, source_sha256: str) -> bool:
        accepted = {
            f"{module_id}:{source_sha256}",
            f"{module_id}@{source_sha256}",
        }
        return bool(accepted & set(self.trusted_modules)) or source_sha256 in set(self.trusted_sha256)

    def _read_manifest(self, path: Path) -> str:
        size = path.stat().st_size
        if size > self.config.modules.manifest_max_bytes:
            raise ValidationError(
                "module manifest exceeded "
                f"manifest_max_bytes={self.config.modules.manifest_max_bytes}"
            )
        return path.read_text(encoding="utf-8")

    def _load_mapping(self, text: str) -> dict[str, Any]:
        stripped = text.lstrip()
        if stripped.startswith("{"):
            data = self._load_json_mapping(text)
            if not isinstance(data, dict):
                raise ValidationError("module manifest JSON must be a mapping")
            return data
        return load_yaml_mapping(text)

    def _load_json_mapping(self, text: str) -> dict[str, Any]:
        try:
            data = json.loads(text, object_pairs_hook=_unique_json_object)
        except ValidationError:
            raise
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid module manifest JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValidationError("module manifest JSON must be a mapping")
        return data

    def _coerce_provides(self, value: Any) -> ModuleProvides:
        if not isinstance(value, dict):
            raise ValidationError("provides must be a mapping")
        unknown = sorted(set(value) - self.PROVIDES_FIELDS)
        if unknown:
            raise ValidationError(f"unknown module provides fields: {unknown}")
        return ModuleProvides(
            tools=self._string_list(value.get("tools"), "provides.tools", self.config.modules.max_declared_tools),
            images=self._string_list(value.get("images"), "provides.images", self.config.modules.max_declared_images),
            syscalls=self._string_list(value.get("syscalls"), "provides.syscalls", self.config.modules.max_declared_syscalls),
            provider_hooks=self._string_list(
                value.get("provider_hooks"),
                "provides.provider_hooks",
                self.config.modules.max_declared_provider_hooks,
            ),
            startup_hooks=self._string_list(
                value.get("startup_hooks"),
                "provides.startup_hooks",
                self.config.modules.max_declared_startup_hooks,
            ),
        )

    def _validate_provides(self, provides: ModuleProvides) -> None:
        for field, values in {
            "tools": provides.tools,
            "images": provides.images,
            "syscalls": provides.syscalls,
            "provider_hooks": provides.provider_hooks,
            "startup_hooks": provides.startup_hooks,
        }.items():
            duplicates = sorted({value for value in values if values.count(value) > 1})
            if duplicates:
                raise ValidationError(f"duplicate module provides.{field}: {duplicates}")
        for name in provides.syscalls:
            if not _SYSCALL_PATTERN.match(name):
                raise ValidationError(f"invalid syscall name in module manifest: {name}")
        for field, values in {
            "provider_hooks": provides.provider_hooks,
            "startup_hooks": provides.startup_hooks,
        }.items():
            for name in values:
                if not _SYSCALL_PATTERN.match(name):
                    raise ValidationError(f"invalid {field[:-1]} name in module manifest: {name}")

    def _resolve_entrypoint(self, manifest_path: Path, entrypoint: str) -> tuple[Path, str]:
        module_ref, object_name = self._split_entrypoint(entrypoint)
        if self._is_path_ref(module_ref):
            source = (manifest_path.parent / module_ref).resolve()
            self._require_under(source, manifest_path.parent.resolve())
        else:
            source = self._resolve_import_source(manifest_path.parent.resolve(), module_ref)
        if not source.is_file():
            raise NotFound(f"module entrypoint source not found: {source}")
        return source, object_name

    def _split_entrypoint(self, entrypoint: str) -> tuple[str, str]:
        if ":" not in entrypoint:
            raise ValidationError("module entrypoint must use '<module-or-path>:<callable>'")
        module_ref, object_name = entrypoint.rsplit(":", 1)
        module_ref = module_ref.strip()
        object_name = object_name.strip()
        if not module_ref or not object_name:
            raise ValidationError("module entrypoint must include both module/path and callable")
        if not self._is_path_ref(module_ref) and not _PYTHON_MODULE_PATTERN.match(module_ref):
            raise ValidationError(f"module entrypoint import is not a valid Python module path: {module_ref}")
        if not _PYTHON_OBJECT_PATTERN.match(object_name):
            raise ValidationError(f"module entrypoint callable is not a valid Python identifier: {object_name}")
        return module_ref, object_name

    def _is_path_ref(self, module_ref: str) -> bool:
        return module_ref.endswith(".py") or module_ref.startswith(".") or "/" in module_ref or "\\" in module_ref

    def _resolve_import_source(self, manifest_dir: Path, module_ref: str) -> Path:
        """Resolve import-string entrypoints without executing package code."""

        parts = module_ref.split(".")
        self._require_package_parent_files(manifest_dir, parts[:-1], module_ref)
        module_path = manifest_dir.joinpath(*parts)
        file_source = module_path.with_suffix(".py")
        package_source = module_path / "__init__.py"
        if file_source.is_file():
            source = file_source.resolve()
            self._require_under(source, manifest_dir)
            return source
        if package_source.is_file():
            source = package_source.resolve()
            self._require_under(source, manifest_dir)
            return source
        raise NotFound(f"module entrypoint import not found under manifest directory: {module_ref}")

    def _require_package_parent_files(self, manifest_dir: Path, package_parts: list[str], module_ref: str) -> None:
        for index in range(1, len(package_parts) + 1):
            init_path = manifest_dir.joinpath(*package_parts[:index], "__init__.py")
            if not init_path.is_file():
                package_name = ".".join(package_parts[:index])
                raise NotFound(
                    "module entrypoint import requires package parent "
                    f"{package_name!r} with __init__.py under the manifest directory: {module_ref}"
                )
            self._require_under(init_path.resolve(), manifest_dir)

    def _require_under(self, path: Path, root: Path) -> None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValidationError(f"module entrypoint path escapes manifest directory: {path}") from exc

    def _import_file(self, path: Path, module_id: str, source_sha256: str, source_bytes: bytes) -> ModuleType:
        module_name = (
            "_agent_libos_module_"
            f"{hashlib.sha256((module_id + str(path) + source_sha256).encode('utf-8')).hexdigest()}"
        )
        return self._exec_source_module(module_name, path, source_bytes)

    def _exec_source_module(self, module_name: str, path: Path, source_bytes: bytes) -> ModuleType:
        spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(path))
        if spec is None:
            raise ValidationError(f"cannot import module entrypoint source: {path}")
        module = ModuleType(module_name)
        module.__file__ = str(path)
        module.__spec__ = spec
        module.__package__ = ""
        sys.modules[module_name] = module
        try:
            exec(compile(source_bytes, str(path), "exec"), module.__dict__)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    def _verify_imported_module_source(self, module: ModuleType, source: ModuleSource) -> None:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ValidationError(f"module entrypoint has no source file: {source.manifest.entrypoint}")
        imported_path = Path(module_file).resolve()
        expected_path = Path(source.source_path).resolve()
        if imported_path != expected_path:
            raise ValidationError(
                "module entrypoint import resolved to a different source file: "
                f"expected {expected_path}, got {imported_path}"
            )
        imported_sha = self._sha256_file(imported_path)
        if imported_sha != source.source_sha256:
            raise ValidationError(
                "module entrypoint source changed after verification: "
                f"expected {source.source_sha256}, got {imported_sha}"
            )

    def _identifier(self, value: Any, field: str, max_chars: int) -> str:
        text = self._string(value, field, max_chars)
        if not _MODULE_ID_PATTERN.match(text):
            raise ValidationError(f"{field} contains unsupported characters: {text!r}")
        return text

    def _string(self, value: Any, field: str, max_chars: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{field} must be a non-empty string")
        text = value.strip()
        if len(text) > max_chars:
            raise ValidationError(f"{field} exceeds max length {max_chars}")
        if any(ord(char) < 32 for char in text):
            raise ValidationError(f"{field} contains control characters")
        return text

    def _string_list(self, value: Any, field: str, max_items: int) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError(f"{field} must be a list")
        if len(value) > max_items:
            raise ValidationError(f"{field} exceeds max item count {max_items}")
        return [self._string(item, f"{field}[]", self.config.modules.id_max_chars) for item in value]

    def _mapping(self, value: Any, field: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValidationError(f"{field} must be a mapping")
        return dict(value)

    def _sha256(self, value: Any, field: str) -> str:
        if not isinstance(value, str) or not _HEX_SHA256_PATTERN.match(value):
            raise ValidationError(f"{field} must be a sha256 hex digest")
        return value.lower()

    def _sha256_file(self, path: Path) -> str:
        return self._sha256_bytes(self._read_source_bytes(path))

    def _read_source_bytes(self, path: Path) -> bytes:
        size = path.stat().st_size
        if size > self.config.modules.source_max_bytes:
            raise ValidationError(
                "module source exceeded "
                f"source_max_bytes={self.config.modules.source_max_bytes}"
            )
        return path.read_bytes()

    def _sha256_bytes(self, value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValidationError(f"duplicate module manifest JSON key: {key!r}")
        mapping[key] = value
    return mapping
