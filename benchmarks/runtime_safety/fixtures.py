from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from benchmarks.runtime_safety.loader import _safe_relative_path
from benchmarks.runtime_safety.models import BenchmarkTask, BenchmarkValidationError


def prepare_workspace(task: BenchmarkTask, suite_root: str | Path, run_root: str | Path, runner: str) -> Path:
    suite = Path(suite_root)
    source = suite / task.workspace
    if not source.exists() or not source.is_dir():
        raise BenchmarkValidationError(f"{task.id}: workspace fixture does not exist: {source}")
    _reject_symlinks(source, task.id)
    target = Path(run_root) / "workspaces" / runner / task.id
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    _apply_setup_files(task, target)
    return target


def safe_workspace_path(workspace: Path, raw_path: str) -> Path:
    relative = _safe_relative_path(raw_path, Path("<runtime>"), "path")
    target = (workspace / relative).resolve()
    root = workspace.resolve()
    if root not in target.parents and target != root:
        raise BenchmarkValidationError(f"path escapes workspace: {raw_path}")
    return target


def _apply_setup_files(task: BenchmarkTask, workspace: Path) -> None:
    setup = task.setup or {}
    for index, item in enumerate(setup.get("files", []) or []):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"{task.id}: setup.files[{index}] must be a mapping")
        path = item.get("path")
        if not isinstance(path, str):
            raise BenchmarkValidationError(f"{task.id}: setup.files[{index}].path must be a string")
        content = item.get("content", "")
        target = safe_workspace_path(workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding=str(item.get("encoding") or "utf-8"), newline="\n")
    for index, item in enumerate(setup.get("delete", []) or []):
        path = str(item.get("path") if isinstance(item, dict) else item)
        target = safe_workspace_path(workspace, path)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _reject_symlinks(source: Path, task_id: str) -> None:
    for item in source.rglob("*"):
        if item.is_symlink():
            raise BenchmarkValidationError(f"{task_id}: workspace fixture contains symlink: {item}")
