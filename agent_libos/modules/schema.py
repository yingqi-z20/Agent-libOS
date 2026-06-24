from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModuleProvides:
    tools: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    syscalls: list[str] = field(default_factory=list)
    provider_hooks: list[str] = field(default_factory=list)
    startup_hooks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModuleManifest:
    schema_version: int
    module_id: str
    name: str
    version: str
    entrypoint: str
    provides: ModuleProvides
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModuleSource:
    manifest: ModuleManifest
    manifest_path: str
    manifest_sha256: str
    source_path: str
    source_sha256: str
    entrypoint_object: str
    source_bytes: bytes = b""


@dataclass(frozen=True)
class LoadedModule:
    module_id: str
    name: str
    version: str
    entrypoint: str
    manifest_path: str
    manifest_sha256: str
    source_path: str
    source_sha256: str
    status: str
    loaded_at: str | None
    registered: dict[str, Any]
    error: str | None
    metadata: dict[str, Any]
