from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_RootIdentity = tuple[int, int]


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


@contextmanager
def _opened_output_root(path: Path) -> Iterator[tuple[int, _RootIdentity] | None]:
    try:
        initial = path.lstat()
    except FileNotFoundError:
        yield None
        return
    if stat.S_ISLNK(initial.st_mode):
        raise ValueError(f"agent output root must not be a symlink: {path}")
    if not stat.S_ISDIR(initial.st_mode):
        raise ValueError(f"agent output root must be a directory: {path}")
    descriptor = os.open(path, _directory_open_flags())
    try:
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if identity != (initial.st_dev, initial.st_ino):
            raise RuntimeError("agent output root changed while it was opened")
        yield descriptor, identity
    finally:
        os.close(descriptor)


def _assert_root_identity(path: Path, identity: _RootIdentity) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError("agent output root changed during cleanup") from exc
    if _is_link_like(current) or (current.st_dev, current.st_ino) != identity:
        raise RuntimeError("agent output root changed during cleanup")


def _is_link_like(value: os.stat_result) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(value, "st_file_attributes", 0)
    return stat.S_ISLNK(value.st_mode) or bool(attributes & reparse_point)


def _collect_entries(root_fd: int) -> dict[str, bool]:
    entries: dict[str, bool] = {}

    def visit(directory_fd: int, prefix: str) -> None:
        with os.scandir(directory_fd) as children:
            selected = sorted(
                (
                    (child.name, child.is_dir(follow_symlinks=False))
                    for child in children
                ),
                key=lambda child: child[0],
            )
        for name, is_directory in selected:
            relative = f"{prefix}/{name}" if prefix else name
            entries[relative] = is_directory
            if not is_directory:
                continue
            child_fd = os.open(name, _directory_open_flags(), dir_fd=directory_fd)
            try:
                visit(child_fd, relative)
            finally:
                os.close(child_fd)

    visit(root_fd, "")
    return entries


def _cleanup_plan(entries: dict[str, bool], preserved: set[str]) -> list[str]:
    remaining = set(entries)
    planned: list[str] = []
    deepest_first = sorted(
        entries,
        key=lambda relative: (relative.count("/"), relative),
        reverse=True,
    )
    for relative in deepest_first:
        if relative in preserved:
            continue
        if entries[relative] and any(
            candidate.startswith(f"{relative}/") for candidate in remaining
        ):
            continue
        planned.append(f"{relative}/" if entries[relative] else relative)
        remaining.remove(relative)
    if not preserved and not remaining:
        planned.append(".")
    return planned


def _collect_path_entries(root: Path) -> dict[str, bool]:
    entries: dict[str, bool] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        state = path.lstat()
        entries[relative] = stat.S_ISDIR(state.st_mode) and not _is_link_like(state)
    return entries


def _windows_root_identity(path: Path) -> _RootIdentity | None:
    try:
        state = path.lstat()
    except FileNotFoundError:
        return None
    if _is_link_like(state):
        raise ValueError(f"agent output root must not be a symlink: {path}")
    if not stat.S_ISDIR(state.st_mode):
        raise ValueError(f"agent output root must be a directory: {path}")
    return state.st_dev, state.st_ino


def _remove_relative(root_fd: int, relative: str, *, directory: bool) -> bool:
    parts = relative.split("/")
    parent_fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, _directory_open_flags(), dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        try:
            if directory:
                os.rmdir(parts[-1], dir_fd=parent_fd)
            else:
                os.unlink(parts[-1], dir_fd=parent_fd)
        except FileNotFoundError:
            return False
        except OSError:
            if directory:
                return False
            raise
        return True
    finally:
        os.close(parent_fd)


def snapshot_agent_outputs(root: str | Path) -> set[str]:
    output_root = Path(root)
    if os.name == "nt":
        identity = _windows_root_identity(output_root)
        if identity is None:
            return set()
        entries = _collect_path_entries(output_root)
        _assert_root_identity(output_root, identity)
        return set(entries)
    with _opened_output_root(output_root) as opened:
        if opened is None:
            return set()
        root_fd, _identity = opened
        return set(_collect_entries(root_fd))


def cleanup_agent_outputs(
    root: str | Path,
    *,
    baseline: set[str] | None = None,
    dry_run: bool = False,
) -> list[str]:
    output_root = Path(root)
    preserved = set(baseline or set())
    if os.name == "nt":
        identity = _windows_root_identity(output_root)
        if identity is None:
            return []
        entries = _collect_path_entries(output_root)
        _assert_root_identity(output_root, identity)
        planned = _cleanup_plan(entries, preserved)
        if dry_run:
            return planned
        removed: list[str] = []
        for relative in planned:
            _assert_root_identity(output_root, identity)
            path = output_root if relative == "." else output_root / relative.removesuffix("/")
            try:
                if relative == "." or entries[relative.removesuffix("/")]:
                    path.rmdir()
                else:
                    path.unlink()
            except (FileNotFoundError, OSError):
                continue
            removed.append(relative)
        return removed
    with _opened_output_root(output_root) as opened:
        if opened is None:
            return []
        root_fd, identity = opened
        entries = _collect_entries(root_fd)
        planned = _cleanup_plan(entries, preserved)
        _assert_root_identity(output_root, identity)
        if dry_run:
            return planned

        removed: list[str] = []
        for relative in planned:
            _assert_root_identity(output_root, identity)
            if relative == ".":
                try:
                    output_root.rmdir()
                except OSError:
                    continue
                removed.append(relative)
                continue
            normalized = relative.removesuffix("/")
            if _remove_relative(
                root_fd,
                normalized,
                directory=entries[normalized],
            ):
                removed.append(relative)
        return removed
