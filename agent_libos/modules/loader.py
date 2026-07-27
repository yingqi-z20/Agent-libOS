from __future__ import annotations

import hashlib
import errno
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import re
import stat
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.modules.schema import ModuleManifest, ModuleProvides, ModuleSource, ModuleSourceFile
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
from agent_libos.utils.ids import new_id
from agent_libos.utils.serde import bounded_json_loads
from agent_libos.utils.yaml_loader import load_yaml_mapping

_MODULE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
_PYTHON_OBJECT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PYTHON_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_SYSCALL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_HEX_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
_WINDOWS_FORBIDDEN_PATH_CHARS = set('<>:"|?*')
_WINDOWS_RESERVED_PATH_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_CACHE_PACKAGE_SEGMENTS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "node_modules",
}
_SENSITIVE_PACKAGE_FILENAMES = {
    ".env",
    ".netrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
_SENSITIVE_PACKAGE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_IMPORT_LOCK = threading.RLock()
_IMPORT_CLEANUP_ATTR = "__agent_libos_package_cleanup__"
_MODULE_FILE_READ_CHUNK_BYTES = 64 * 1024


def _is_ignored_package_path(parts: tuple[str, ...]) -> bool:
    return any(part.lower() in _CACHE_PACKAGE_SEGMENTS for part in parts)


class _FreshSourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        source_bytes = self.get_data(self.path)
        return self.source_to_code(source_bytes, self.path)


class _SnapshotPackageImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, package_name: str, records: tuple[ModuleSourceFile, ...], entry_module_path: str):
        self.package_name = package_name
        self.entry_module_path = entry_module_path
        self._modules: dict[str, ModuleSourceFile] = {}
        self._packages: set[str] = {package_name}
        for record in records:
            module_name = self._module_name_for_record(record)
            if module_name is not None:
                existing = self._modules.get(module_name)
                if existing is not None:
                    raise ValidationError(
                        "module package contains paths with the same import identity: "
                        f"{existing.module_path!r} and {record.module_path!r}"
                    )
                self._modules[module_name] = record
            self._add_package_dirs(record.module_path)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> importlib.machinery.ModuleSpec | None:
        if fullname in self._modules:
            record = self._modules[fullname]
            is_package = fullname == self.package_name and record.module_path == "__init__.py"
            return importlib.util.spec_from_loader(fullname, self, origin=record.absolute_path, is_package=is_package)
        if fullname in self._packages:
            spec = importlib.util.spec_from_loader(fullname, self, origin="<agent-libos-module-snapshot>", is_package=True)
            if spec is not None:
                spec.submodule_search_locations = []
            return spec
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        record = self._modules.get(module.__name__)
        if record is None:
            module.__path__ = []  # type: ignore[attr-defined]
            module.__package__ = module.__name__
            return
        module.__file__ = record.absolute_path
        if module.__name__ == self.package_name and record.module_path == "__init__.py":
            module.__package__ = module.__name__
            module.__path__ = []  # type: ignore[attr-defined]
        else:
            module.__package__ = module.__name__.rsplit(".", 1)[0]
        exec(compile(record.content, record.absolute_path, "exec"), module.__dict__)

    def _module_name_for_record(self, record: ModuleSourceFile) -> str | None:
        if record.module_path == "__init__.py" or record.module_path.endswith("/__init__.py"):
            if record.module_path != self.entry_module_path:
                return None
            parent = record.module_path[: -len("/__init__.py")] if record.module_path.endswith("/__init__.py") else ""
            suffix = ".".join(part for part in parent.split("/") if part)
            return f"{self.package_name}.{suffix}" if suffix else self.package_name
        if not record.module_path.endswith(".py"):
            return None
        relative = record.module_path[: -len(".py")]
        suffix = ".".join(part for part in relative.split("/") if part)
        return f"{self.package_name}.{suffix}" if suffix else self.package_name

    def _add_package_dirs(self, module_path: str) -> None:
        parts = module_path.split("/")[:-1]
        prefix = self.package_name
        for part in parts:
            prefix = f"{prefix}.{part}"
            self._packages.add(prefix)


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
    PROVIDES_FIELDS = {
        "tools",
        "images",
        "syscalls",
        "provider_hooks",
        "startup_hooks",
        "durable_object_release_finalizers",
    }

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
        if not self.is_trusted(source.manifest.module_id, source.source_sha256, source.manifest_sha256):
            raise CapabilityDenied(
                "startup module is not trusted: "
                f"{source.manifest.module_id}:{source.manifest_sha256}:{source.source_sha256}"
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
            "source_kind": source.source_kind,
            "source_root": source.source_root,
            "source_files": self._source_file_summaries(source),
            "trusted": self.is_trusted(source.manifest.module_id, source.source_sha256, source.manifest_sha256),
            "trust_key": self.trust_key(source.manifest.module_id, source.manifest_sha256, source.source_sha256),
            "provides": {
                "tools": list(source.manifest.provides.tools),
                "images": list(source.manifest.provides.images),
                "syscalls": list(source.manifest.provides.syscalls),
                "provider_hooks": list(source.manifest.provides.provider_hooks),
                "startup_hooks": list(source.manifest.provides.startup_hooks),
                "durable_object_release_finalizers": list(
                    source.manifest.provides.durable_object_release_finalizers
                ),
            },
        }

    def resolve(self, manifest_path: str | Path) -> ModuleSource:
        selected_path = Path(manifest_path).expanduser()
        if not selected_path.is_absolute():
            selected_path = Path.cwd() / selected_path
        selected_path = self._absolute_lexical_path(selected_path)
        text = self._read_manifest(selected_path)
        # Keep the checked lexical path. Canonicalizing it here would erase a
        # symlinked ancestor before the source/package secure-open boundary.
        path = selected_path
        manifest = self.parse_manifest(text)
        source_path, entrypoint_object = self._resolve_entrypoint(path, manifest.entrypoint)
        source_bytes = self._read_source_bytes(source_path)
        source_sha = self._sha256_bytes(source_bytes)
        expected_sha = manifest.sha256.lower()
        source_root = self._infer_source_root(path.parent, source_path, manifest.entrypoint)
        source_files = self._entry_source_files(path.parent, source_root, source_path, source_bytes)
        source_kind = "file"
        if source_sha != expected_sha:
            source_files = self._read_package_source_files(path.parent, source_root)
            package_sha = self._package_sha256(source_files)
            if package_sha != expected_sha:
                raise ValidationError(
                    "module source sha256 mismatch: "
                    f"expected {expected_sha}, got entry={source_sha}, package={package_sha}"
                )
            source_sha = package_sha
            source_kind = "package"
            entry = next(
                (
                    record
                    for record in source_files
                    if self._absolute_lexical_path(Path(record.absolute_path))
                    == self._absolute_lexical_path(source_path)
                ),
                None,
            )
            if entry is None:
                raise ValidationError(f"module entrypoint source is missing from package snapshot: {source_path}")
            source_bytes = entry.content
        if source_kind == "file" and source_sha != expected_sha:
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
            source_kind=source_kind,
            source_root=str(source_root),
            source_files=tuple(source_files),
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
            if source.source_kind == "package":
                module = self._import_package(source)
            else:
                module = self._import_file(
                    Path(source.source_path),
                    source.manifest.module_id,
                    source.source_sha256,
                    source.source_bytes,
                )
        try:
            self._verify_imported_module_source(module, source)
            entrypoint = getattr(module, object_name, None)
            if not callable(entrypoint):
                raise ValidationError(f"module entrypoint is not callable: {source.manifest.entrypoint}")
            return entrypoint
        except BaseException:
            self._cleanup_imported_package(module)
            raise

    @classmethod
    def import_cleanup_for_entrypoint(cls, entrypoint: Any) -> tuple[str, Any] | None:
        module_name = getattr(entrypoint, "__module__", None)
        if not isinstance(module_name, str):
            return None
        module = sys.modules.get(module_name)
        if module is None:
            return None
        cleanup = getattr(module, _IMPORT_CLEANUP_ATTR, None)
        if isinstance(cleanup, tuple) and len(cleanup) == 2:
            return cleanup
        return None

    @classmethod
    def cleanup_imported_package(cls, cleanup: Any) -> None:
        if not isinstance(cleanup, tuple) or len(cleanup) != 2:
            return
        package_name, importer = cleanup
        cls._clear_import_namespace(str(package_name))
        try:
            sys.meta_path.remove(importer)
        except ValueError:
            pass

    @staticmethod
    def trust_key(module_id: str, manifest_sha256: str, source_sha256: str) -> str:
        return f"{module_id}:{manifest_sha256}:{source_sha256}"

    def is_trusted(self, module_id: str, source_sha256: str, manifest_sha256: str) -> bool:
        accepted = {
            self.trust_key(module_id, manifest_sha256, source_sha256),
            f"{module_id}@{manifest_sha256}:{source_sha256}",
        }
        accepted_hashes = {f"{manifest_sha256}:{source_sha256}"}
        return bool(accepted & set(self.trusted_modules)) or bool(accepted_hashes & set(self.trusted_sha256))

    def _read_manifest(self, path: Path) -> str:
        raw = self._read_module_file_limited(
            path,
            kind="manifest",
            max_bytes=self.config.modules.manifest_max_bytes,
            limit_name="manifest_max_bytes",
        )
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(f"module manifest is not valid UTF-8: {path}") from exc

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
            # Preflight through the shared strict parser before preserving the
            # module-specific duplicate-key error contract below.  This fixes
            # depth, node, integer, and non-finite-number limits even when the
            # interpreter-wide integer guard has been disabled.
            bounded_json_loads(text, reject_duplicate_keys=False)
            data = json.loads(text, object_pairs_hook=_unique_json_object)
        except ValidationError:
            raise
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
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
            durable_object_release_finalizers=self._string_list(
                value.get("durable_object_release_finalizers"),
                "provides.durable_object_release_finalizers",
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
            "durable_object_release_finalizers": (
                provides.durable_object_release_finalizers
            ),
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
            "durable_object_release_finalizers": (
                provides.durable_object_release_finalizers
            ),
        }.items():
            for name in values:
                if not _SYSCALL_PATTERN.match(name):
                    raise ValidationError(f"invalid {field[:-1]} name in module manifest: {name}")

    def _resolve_entrypoint(self, manifest_path: Path, entrypoint: str) -> tuple[Path, str]:
        module_ref, object_name = self._split_entrypoint(entrypoint)
        if self._is_path_ref(module_ref):
            source = self._absolute_lexical_path(manifest_path.parent / module_ref)
            self._require_under(source, manifest_path.parent)
        else:
            source = self._resolve_import_source(manifest_path.parent, module_ref)
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
            source = self._absolute_lexical_path(file_source)
            self._require_under(source, manifest_dir)
            return source
        if package_source.is_file():
            source = self._absolute_lexical_path(package_source)
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
            self._require_under(
                self._absolute_lexical_path(init_path),
                manifest_dir,
            )

    def _require_under(self, path: Path, root: Path) -> None:
        path = self._absolute_lexical_path(path)
        root = self._absolute_lexical_path(root)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValidationError(f"module entrypoint path escapes manifest directory: {path}") from exc

    @staticmethod
    def _absolute_lexical_path(path: Path) -> Path:
        """Return an absolute normalized path without following symlinks."""

        selected = Path(os.path.abspath(path))
        if sys.platform == "darwin" and len(selected.parts) > 1:
            alias = selected.parts[1]
            if alias in {"etc", "tmp", "var"}:
                return Path("/private", alias, *selected.parts[2:])
        return selected

    def _infer_source_root(self, manifest_dir: Path, source_path: Path, entrypoint: str) -> Path:
        module_ref, _object_name = self._split_entrypoint(entrypoint)
        manifest_dir = self._absolute_lexical_path(manifest_dir)
        source = self._absolute_lexical_path(source_path)
        self._require_under(source, manifest_dir)
        if not self._is_path_ref(module_ref):
            first = module_ref.split(".", 1)[0]
            candidate = self._absolute_lexical_path(manifest_dir / first)
            if candidate.is_dir():
                self._require_under(candidate, manifest_dir)
                return candidate
        if source.name == "__init__.py":
            return source.parent
        current = source.parent
        root = current
        while current != manifest_dir and (current / "__init__.py").is_file():
            root = current
            current = current.parent
        return root

    def _entry_source_files(
        self,
        manifest_dir: Path,
        source_root: Path,
        source_path: Path,
        source_bytes: bytes,
    ) -> tuple[ModuleSourceFile, ...]:
        manifest_dir = self._absolute_lexical_path(manifest_dir)
        source_root = self._absolute_lexical_path(source_root)
        source_path = self._absolute_lexical_path(source_path)
        relative = self._manifest_relative_path(manifest_dir, source_path)
        module_path = self._source_root_relative_path(source_root, source_path)
        return (
            ModuleSourceFile(
                path=relative,
                module_path=module_path,
                absolute_path=str(source_path),
                size_bytes=len(source_bytes),
                sha256=self._sha256_bytes(source_bytes),
                content=source_bytes,
            ),
        )

    def _read_package_source_files(self, manifest_dir: Path, source_root: Path) -> tuple[ModuleSourceFile, ...]:
        manifest_dir = Path(os.path.abspath(manifest_dir))
        root = Path(os.path.abspath(source_root))
        self._require_under(root, manifest_dir)
        records: list[ModuleSourceFile] = []
        total_bytes = 0
        visited_paths = 0

        def visit(directory: SecureDirectoryGuard) -> None:
            nonlocal total_bytes, visited_paths
            opened_directory = self._validate_module_directory_snapshot(
                directory.snapshot(),
                path=directory.path,
                after_read=False,
            )
            try:
                linked_directory = self._validate_module_directory_snapshot(
                    directory.linked_snapshot(),
                    path=directory.path,
                    after_read=False,
                )
            except OSError as exc:
                raise ValidationError(
                    f"module package directory changed during enumeration: {directory.path}"
                ) from exc
            if linked_directory != opened_directory:
                raise ValidationError(
                    f"module package directory changed during enumeration: {directory.path}"
                )
            try:
                entries = directory.scandir()
            except OSError as exc:
                raise ValidationError(
                    f"module package directory changed during enumeration: {directory.path}"
                ) from exc
            with entries:
                for entry in entries:
                    item = directory.path / entry.name
                    try:
                        source_relative_parts = item.relative_to(root).parts
                    except ValueError as exc:
                        raise ValidationError(
                            f"module package path escapes source root: {item}"
                        ) from exc
                    visited_paths += 1
                    if visited_paths > self.config.modules.max_package_files:
                        raise ValidationError(
                            "module package exceeded "
                            f"max_package_files={self.config.modules.max_package_files}"
                        )
                    if _is_ignored_package_path(source_relative_parts):
                        continue
                    relative = self._manifest_relative_path(manifest_dir, item)
                    self._validate_source_relative_path(relative)
                    try:
                        before = directory.lstat_child(entry.name)
                    except OSError as exc:
                        raise ValidationError(
                            f"module package path changed during enumeration: {item}"
                        ) from exc
                    if before.is_reparse_point or stat.S_ISLNK(before.mode):
                        raise ValidationError(
                            f"module package symlinks are not supported: {item}"
                        )
                    if stat.S_ISDIR(before.mode):
                        try:
                            child = directory.open_child_directory(entry.name)
                        except OSError as exc:
                            raise ValidationError(
                                f"module package directory changed during enumeration: {item}"
                            ) from exc
                        with child:
                            visit(child)
                        continue
                    if not stat.S_ISREG(before.mode):
                        raise ValidationError(
                            "module package path is not a regular file or directory: "
                            f"{item}"
                        )
                    if before.links > 1:
                        raise ValidationError(
                            f"module package hard links are not supported: {item}"
                        )
                    if item.suffix != ".py":
                        continue
                    remaining_package_bytes = (
                        self.config.modules.package_max_bytes - total_bytes
                    )
                    if remaining_package_bytes < self.config.modules.source_max_bytes:
                        selected_max_bytes = max(remaining_package_bytes, 0)
                        limit_name = "package_max_bytes"
                        reported_max_bytes = self.config.modules.package_max_bytes
                    else:
                        selected_max_bytes = self.config.modules.source_max_bytes
                        limit_name = "source_max_bytes"
                        reported_max_bytes = selected_max_bytes
                    content = self._read_source_bytes(
                        item,
                        max_bytes=selected_max_bytes,
                        limit_name=limit_name,
                        reported_max_bytes=reported_max_bytes,
                        parent=directory,
                        relative_name=entry.name,
                    )
                    total_bytes += len(content)
                    records.append(
                        ModuleSourceFile(
                            path=relative,
                            module_path=self._source_root_relative_path(root, item),
                            absolute_path=str(item),
                            size_bytes=len(content),
                            sha256=self._sha256_bytes(content),
                            content=content,
                        )
                    )
            after_directory = self._validate_module_directory_snapshot(
                directory.snapshot(),
                path=directory.path,
                after_read=True,
            )
            try:
                linked_after = self._validate_module_directory_snapshot(
                    directory.linked_snapshot(),
                    path=directory.path,
                    after_read=True,
                )
            except OSError as exc:
                raise ValidationError(
                    f"module package directory changed during enumeration: {directory.path}"
                ) from exc
            if (
                after_directory != opened_directory
                or linked_after != after_directory
            ):
                raise ValidationError(
                    f"module package directory changed during enumeration: {directory.path}"
                )

        try:
            root_guard = open_secure_directory(root)
        except OSError as exc:
            raise ValidationError(
                f"cannot securely open module package directory: {root}"
            ) from exc
        with root_guard:
            visit(root_guard)
        if not records:
            raise NotFound(f"module package contains no Python source files: {root}")
        return tuple(sorted(records, key=lambda record: record.path))

    def _validate_module_directory_snapshot(
        self,
        snapshot: StablePathSnapshot,
        *,
        path: Path,
        after_read: bool,
    ) -> StablePathSnapshot:
        if snapshot.is_reparse_point or not stat.S_ISDIR(snapshot.mode):
            raise ValidationError(
                f"module package directory is not a regular directory: {path}"
            )
        if snapshot.links < 1:
            message = "changed during enumeration" if after_read else "is not linked"
            raise ValidationError(f"module package directory {message}: {path}")
        if not stable_identity_available(snapshot):
            raise ValidationError(
                "secure Runtime Module directory identity is unavailable on this platform"
            )
        if snapshot.size < 0:
            raise ValidationError(f"module package directory has an invalid size: {path}")
        return snapshot

    def _manifest_relative_path(self, manifest_dir: Path, path: Path) -> str:
        try:
            return path.relative_to(manifest_dir).as_posix()
        except ValueError as exc:
            raise ValidationError(f"module package path escapes manifest directory: {path}") from exc

    def _source_root_relative_path(self, source_root: Path, path: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ValidationError(f"module package path escapes source root: {path}") from exc

    def _validate_source_relative_path(self, path: str) -> None:
        normalized = path.replace("\\", "/").strip()
        if not normalized or normalized.startswith("/") or ":" in normalized.split("/", 1)[0]:
            raise ValidationError(f"module package path must be relative: {path!r}")
        parts: list[str] = []
        for part in normalized.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                raise ValidationError(f"module package path escapes source root: {path!r}")
            parts.append(part)
        if not parts or "/".join(parts) != normalized:
            raise ValidationError(f"module package path must be normalized: {path!r}")
        if any(ord(char) < 32 for char in normalized):
            raise ValidationError(f"module package path contains control characters: {path!r}")
        for part in parts:
            lower = part.lower()
            stem = part.split(".", 1)[0].upper()
            if any(char in _WINDOWS_FORBIDDEN_PATH_CHARS for char in part):
                raise ValidationError(f"module package path contains a Windows-unsafe character: {path!r}")
            if part.endswith((" ", ".")):
                raise ValidationError(f"module package path contains a Windows-unsafe segment: {path!r}")
            if stem in _WINDOWS_RESERVED_PATH_NAMES:
                raise ValidationError(f"module package path uses a reserved Windows device name: {path!r}")
            if lower in _CACHE_PACKAGE_SEGMENTS:
                raise ValidationError(f"module package must not include cache or VCS paths: {path!r}")
            if lower in _SENSITIVE_PACKAGE_FILENAMES or lower.endswith(_SENSITIVE_PACKAGE_SUFFIXES):
                raise ValidationError(f"module package must not include likely secret material: {path!r}")

    def _package_sha256(self, source_files: tuple[ModuleSourceFile, ...]) -> str:
        canonical = [
            {"path": record.path, "size_bytes": record.size_bytes, "sha256": record.sha256}
            for record in source_files
        ]
        payload = {"kind": "agent_libos_runtime_module_package", "files": canonical}
        return self._sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def _source_file_summaries(self, source: ModuleSource) -> list[dict[str, Any]]:
        return [
            {
                "path": record.path,
                "module_path": record.module_path,
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
            }
            for record in source.source_files
        ]

    def _import_file(self, path: Path, module_id: str, source_sha256: str, source_bytes: bytes) -> ModuleType:
        module_name = (
            "_agent_libos_module_"
            f"{hashlib.sha256((module_id + str(path) + source_sha256).encode('utf-8')).hexdigest()}"
        )
        return self._exec_source_module(module_name, path, source_bytes)

    def _import_package(self, source: ModuleSource) -> ModuleType:
        package_name = (
            "_agent_libos_module_pkg_"
            f"{hashlib.sha256((source.manifest.module_id + source.source_root + source.source_sha256).encode('utf-8')).hexdigest()}"
            f"_{new_id('load')}"
        )
        entry_module_path = self._source_root_relative_path(
            self._absolute_lexical_path(Path(source.source_root)),
            self._absolute_lexical_path(Path(source.source_path)),
        )
        importer = _SnapshotPackageImporter(package_name, tuple(source.source_files), entry_module_path)
        if entry_module_path == "__init__.py":
            entry_name = package_name
        elif entry_module_path.endswith("/__init__.py"):
            entry_name = f"{package_name}.{entry_module_path[:-len('/__init__.py')].replace('/', '.')}"
        else:
            entry_name = f"{package_name}.{entry_module_path[:-3].replace('/', '.')}"
        self._clear_import_namespace(package_name)
        sys.meta_path.insert(0, importer)
        try:
            module = importlib.import_module(entry_name)
            cleanup = (package_name, importer)
            # The manifest entry may re-export a callable defined by a sibling
            # module. Cleanup ownership still belongs to this package load, so
            # publish the same immutable handle on every module imported from
            # the synthetic namespace before returning the callable.
            for name, imported_module in tuple(sys.modules.items()):
                if (
                    name == package_name or name.startswith(f"{package_name}.")
                ) and isinstance(imported_module, ModuleType):
                    setattr(imported_module, _IMPORT_CLEANUP_ATTR, cleanup)
            return module
        except BaseException:
            self._clear_import_namespace(package_name)
            try:
                sys.meta_path.remove(importer)
            except ValueError:
                pass
            raise

    def _cleanup_imported_package(self, module: ModuleType) -> None:
        self.cleanup_imported_package(getattr(module, _IMPORT_CLEANUP_ATTR, None))

    @staticmethod
    def _clear_import_namespace(module_name: str) -> None:
        for name in list(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name, None)

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
        finally:
            sys.modules.pop(module_name, None)
        return module

    def _verify_imported_module_source(self, module: ModuleType, source: ModuleSource) -> None:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ValidationError(f"module entrypoint has no source file: {source.manifest.entrypoint}")
        imported_path = self._absolute_lexical_path(Path(module_file))
        expected_path = self._absolute_lexical_path(Path(source.source_path))
        if imported_path != expected_path:
            raise ValidationError(
                "module entrypoint import resolved to a different source file: "
                f"expected {expected_path}, got {imported_path}"
            )
        if source.source_kind == "package":
            source_root = self._absolute_lexical_path(Path(source.source_root))
            manifest_dir = self._absolute_lexical_path(
                Path(source.manifest_path)
            ).parent
            imported_sha = self._package_sha256(self._read_package_source_files(manifest_dir, source_root))
            if imported_sha != source.source_sha256:
                raise ValidationError(
                    "module package source changed after verification: "
                    f"expected {source.source_sha256}, got {imported_sha}"
                )
            return
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
        try:
            json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValidationError(f"{field} must contain only finite JSON values") from exc
        return dict(value)

    def _sha256(self, value: Any, field: str) -> str:
        if not isinstance(value, str) or not _HEX_SHA256_PATTERN.match(value):
            raise ValidationError(f"{field} must be a sha256 hex digest")
        return value.lower()

    def _sha256_file(self, path: Path) -> str:
        return self._sha256_bytes(self._read_source_bytes(path))

    def _read_source_bytes(
        self,
        path: Path,
        *,
        max_bytes: int | None = None,
        limit_name: str = "source_max_bytes",
        reported_max_bytes: int | None = None,
        parent: SecureDirectoryGuard | None = None,
        relative_name: str | None = None,
    ) -> bytes:
        selected_max_bytes = (
            self.config.modules.source_max_bytes
            if max_bytes is None
            else max_bytes
        )
        return self._read_module_file_limited(
            path,
            kind="source",
            max_bytes=selected_max_bytes,
            limit_name=limit_name,
            reported_max_bytes=reported_max_bytes,
            parent=parent,
            relative_name=relative_name,
        )

    def _module_file_snapshot(
        self,
        snapshot: StablePathSnapshot,
        *,
        path: Path,
        kind: str,
        after_read: bool,
    ) -> StablePathSnapshot:
        if snapshot.is_reparse_point or not stat.S_ISREG(snapshot.mode):
            raise ValidationError(f"module {kind} is not a regular file: {path}")
        if snapshot.links > 1:
            raise ValidationError(f"module {kind} hard links are not supported: {path}")
        if snapshot.links < 1:
            message = "changed during read" if after_read else "is not linked"
            raise ValidationError(f"module {kind} {message}: {path}")
        if not stable_identity_available(snapshot):
            raise ValidationError(
                "secure Runtime Module file identity is unavailable on this platform"
            )
        if snapshot.size < 0:
            raise ValidationError(f"module {kind} has an invalid size: {path}")
        return snapshot

    @staticmethod
    def _module_file_limit_error(
        *,
        kind: str,
        path: Path,
        limit_name: str,
        max_bytes: int,
    ) -> ValidationError:
        return ValidationError(
            f"module {kind} exceeded {limit_name}={max_bytes}: {path}"
        )

    def _read_module_file_limited(
        self,
        path: Path,
        *,
        kind: str,
        max_bytes: int,
        limit_name: str,
        reported_max_bytes: int | None = None,
        parent: SecureDirectoryGuard | None = None,
        relative_name: str | None = None,
    ) -> bytes:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 0
        ):
            raise ValidationError("Runtime Module file read limit must be non-negative")
        displayed_max_bytes = max_bytes if reported_max_bytes is None else reported_max_bytes
        try:
            secure_file = open_secure_file(
                path,
                parent=parent,
                relative_name=relative_name,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValidationError(f"module {kind} symlinks are not supported: {path}") from exc
            if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
                raise NotFound(f"module {kind} not found: {path}") from exc
            if exc.errno == getattr(errno, "ESTALE", -1):
                raise ValidationError(f"module {kind} changed during read: {path}") from exc
            raise ValidationError(f"cannot securely open module {kind}: {path}") from exc
        try:
            return read_stable_file_limited(
                secure_file,
                max_bytes=max_bytes,
                chunk_bytes=_MODULE_FILE_READ_CHUNK_BYTES,
                validate_snapshot=lambda snapshot, after_read: self._module_file_snapshot(
                    snapshot,
                    path=path,
                    kind=kind,
                    after_read=after_read,
                ),
            )
        except SecureFileLimitExceeded as exc:
            raise self._module_file_limit_error(
                kind=kind,
                path=path,
                limit_name=limit_name,
                max_bytes=displayed_max_bytes,
            ) from exc
        except SecureFileReadUnavailable as exc:
            raise ValidationError(
                f"cannot securely read module {kind} to EOF: {path}"
            ) from exc
        except (SecureFileChanged, OSError) as exc:
            raise ValidationError(f"module {kind} changed during read: {path}") from exc

    def _sha256_bytes(self, value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValidationError(f"duplicate module manifest JSON key: {key!r}")
        mapping[key] = value
    return mapping
