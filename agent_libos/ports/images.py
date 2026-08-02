from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_libos.substrate.base import ResolvedPath
else:
    class ResolvedPath(Protocol):
        """Runtime-local structural view; avoids importing the substrate graph."""

        @property
        def relative(self) -> str:
            ...

        @property
        def display(self) -> str:
            ...

        @property
        def is_root(self) -> bool:
            ...


class ImageCheckpointPort(Protocol):
    """Checkpoint operations needed to build a checkpoint-derived image."""

    def load_checkpoint_artifact_for_read(
        self,
        checkpoint_id: str,
        *,
        actor: str | None,
        require_capability: bool = ...,
    ) -> tuple[Any, dict[str, Any]]:
        ...

    def checkpoint_or_process_read_scope(
        self,
        actor: str,
        checkpoint: Any,
        *,
        purpose: str,
    ) -> AbstractContextManager[Any]:
        ...

    def preflight_checkpoint_read(
        self,
        checkpoint_id: str,
        *,
        actor: str | None,
        require_capability: bool = ...,
    ) -> None:
        ...

    def require_snapshot_modules(self, snapshot: dict[str, Any]) -> None:
        ...


class ImageFilesystemPort(Protocol):
    """Workspace reads used while importing a process-visible image package."""

    def read_bytes(
        self,
        pid: str,
        path: str | os.PathLike[str],
        max_bytes: int = ...,
        cwd: str | os.PathLike[str] | None = None,
    ) -> Any:
        ...

    def read_directory(
        self,
        pid: str,
        path: str | os.PathLike[str],
        limit: int = ...,
        cwd: str | os.PathLike[str] | None = None,
    ) -> Any:
        ...

    def resolve_path(
        self,
        path: str | os.PathLike[str],
        cwd: str | os.PathLike[str] | None = None,
    ) -> tuple[ResolvedPath, str]:
        ...


class ImageToolPort(Protocol):
    """Tool catalog and JIT validation surface used by image registration."""

    def resolve(self, name: str, *, pid: str | None = None) -> Any:
        ...

    def name_collides_with_static_tool(self, name: str) -> bool:
        ...

    def static_check_jit_source(self, source: str) -> Any:
        ...

    def is_jit_tool_id(self, tool_id: str) -> bool:
        ...

    def initial_tool_projection(self, image: Any) -> list[str]:
        ...


__all__ = [
    "ImageCheckpointPort",
    "ImageFilesystemPort",
    "ImageToolPort",
]
