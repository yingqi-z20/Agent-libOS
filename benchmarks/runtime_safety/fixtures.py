from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import GitError
from agent_libos.substrate import LocalGitProvider
from benchmarks.runtime_safety.loader import _safe_relative_path
from benchmarks.runtime_safety.models import BenchmarkTask, BenchmarkValidationError


def prepare_workspace(task: BenchmarkTask, suite_root: str | Path, run_root: str | Path, runner: str) -> Path:
    suite = Path(suite_root).resolve(strict=True)
    source = suite / task.workspace
    if source.is_symlink():
        raise BenchmarkValidationError(
            f"{task.id}: workspace fixture root may not be a symlink: {source}"
        )
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkValidationError(
            f"{task.id}: workspace fixture does not exist: {source}"
        ) from exc
    if (
        not resolved_source.is_dir()
        or (resolved_source != suite and suite not in resolved_source.parents)
    ):
        raise BenchmarkValidationError(f"{task.id}: workspace fixture does not exist: {source}")
    _reject_symlinks(resolved_source, task.id)
    _reject_git_metadata(resolved_source, task.id)
    target = Path(run_root) / "workspaces" / runner / task.id
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(resolved_source, target)
    _apply_setup_files(task, target)
    _reject_git_metadata(target, task.id)
    _apply_setup_git(task, target)
    return target


def safe_workspace_path(workspace: Path, raw_path: str) -> Path:
    relative = _safe_relative_path(raw_path, Path("<runtime>"), "path")
    if any(part.casefold() == ".git" for part in Path(relative).parts):
        raise BenchmarkValidationError(f"Git metadata paths are not valid benchmark setup paths: {raw_path}")
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


def _apply_setup_git(task: BenchmarkTask, workspace: Path) -> None:
    raw = (task.setup or {}).get("git")
    if raw is None:
        return
    if not isinstance(raw, dict) or raw.get("initialize") is not True:
        raise BenchmarkValidationError(
            f"{task.id}: setup.git must enable deterministic initialize"
        )

    provider = LocalGitProvider(workspace, config=DEFAULT_CONFIG.git)
    try:
        executable, _identity, _digest = provider._resolve_executable()
    except GitError as exc:
        raise BenchmarkValidationError(
            f"{task.id}: deterministic Git fixture setup could not resolve trusted Host Git"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="agent-libos-benchmark-git-") as isolated:
        isolated_root = Path(isolated)
        global_config = isolated_root / "global-config"
        global_config.write_text("", encoding="utf-8")
        hooks = isolated_root / "hooks"
        hooks.mkdir()
        environment = {
            key: value
            for key in (
                "COMSPEC",
                "LANG",
                "LC_ALL",
                "PATHEXT",
                "SYSTEMROOT",
                "TEMP",
                "TMP",
                "WINDIR",
            )
            if (value := os.environ.get(key)) is not None
        }
        environment.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": str(global_config),
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
                "PATH": provider._safe_path(),
            }
        )

        def run(*args: str) -> None:
            try:
                result = subprocess.run(
                    [
                        str(executable),
                        "-c",
                        f"core.hooksPath={hooks}",
                        *args,
                    ],
                    cwd=workspace,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=DEFAULT_CONFIG.git.local_timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                raise BenchmarkValidationError(
                    f"{task.id}: deterministic Git fixture setup timed out"
                ) from exc
            if result.returncode != 0:
                raise BenchmarkValidationError(
                    f"{task.id}: deterministic Git fixture setup failed"
                )

        run("init", "-q")
        run("symbolic-ref", "HEAD", "refs/heads/main")
        run("config", "user.name", "Agent libOS Benchmark")
        run("config", "user.email", "benchmark@agent-libos.invalid")
        run("config", "core.autocrlf", "false")
        run("add", "--all", "--", ".")
        run("-c", "commit.gpgSign=false", "commit", "-q", "--allow-empty", "-m", "fixture")

        for index, item in enumerate(raw.get("post_commit_files", []) or []):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise BenchmarkValidationError(
                    f"{task.id}: setup.git.post_commit_files[{index}] is invalid"
                )
            target = safe_workspace_path(workspace, str(item["path"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                str(item.get("content", "")),
                encoding=str(item.get("encoding") or "utf-8"),
                newline="\n",
            )

        if raw.get("active_filter") is True:
            run("config", "filter.agent-libos-benchmark.clean", "agent-libos-filter-must-not-run")
            (workspace / ".gitattributes").write_text(
                "*.txt filter=agent-libos-benchmark\n",
                encoding="utf-8",
            )


def _reject_symlinks(source: Path, task_id: str) -> None:
    for item in source.rglob("*"):
        if item.is_symlink():
            raise BenchmarkValidationError(f"{task_id}: workspace fixture contains symlink: {item}")


def _reject_git_metadata(source: Path, task_id: str) -> None:
    for item in source.rglob("*"):
        if item.name.casefold() == ".git":
            raise BenchmarkValidationError(
                f"{task_id}: workspace fixture contains preexisting Git metadata"
            )
