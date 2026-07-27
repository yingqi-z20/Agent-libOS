from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import ValidationError
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.skills import write_skill_package


def _write_many_resources(root: Path, *, count: int, size: int) -> Path:
    return write_skill_package(
        root,
        "bounded-package",
        extra_resources={
            f"assets/item-{index:03d}.txt": "x" * size
            for index in range(count)
        },
    )


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlink semantics")
def test_host_skill_package_rejects_symlinked_intermediate_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    package = write_skill_package(real_parent, "secure-skill")
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    runtime = Runtime.open("local")
    try:
        with pytest.raises(ValidationError, match="securely open Skill package"):
            runtime.skills.validate_package_path(alias_parent / package.name)
        assert runtime.store.get_skill("secure-skill") is None
    finally:
        runtime.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX replacement fixture")
def test_host_skill_package_leaf_swap_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = write_skill_package(
        tmp_path,
        "swap-skill",
        extra_resources={"references/guide.md": "original\n"},
    )
    target = package / "references" / "guide.md"
    parked = tmp_path / "original-guide.md"
    replacement = tmp_path / "replacement-guide.md"
    replacement.write_text("replacement\n", encoding="utf-8")
    identity = (target.stat().st_dev, target.stat().st_ino)
    real_fstat = os.fstat
    swapped = False

    def swap_after_open(fd: int) -> os.stat_result:
        nonlocal swapped
        result = real_fstat(fd)
        if not swapped and (result.st_dev, result.st_ino) == identity:
            swapped = True
            os.replace(target, parked)
            os.replace(replacement, target)
        return result

    monkeypatch.setattr(os, "fstat", swap_after_open)
    runtime = Runtime.open("local")
    try:
        with pytest.raises(ValidationError, match="changed during read"):
            runtime.skills.validate_package_path(package)
        assert swapped
        assert runtime.store.get_skill("swap-skill") is None
    finally:
        runtime.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX growth fixture")
def test_host_skill_package_growth_uses_bounded_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = write_skill_package(
        tmp_path,
        "growth-skill",
        extra_resources={"references/guide.md": "original\n"},
    )
    target = package / "references" / "guide.md"
    file_limit = target.stat().st_size + 64
    identity = (target.stat().st_dev, target.stat().st_ino)
    config = replace(
        DEFAULT_CONFIG,
        skills=replace(
            DEFAULT_CONFIG.skills,
            resource_read_max_bytes=file_limit,
        ),
    )
    real_fstat = os.fstat
    real_fdopen = os.fdopen
    grew = False
    read_sizes: list[int] = []

    class RecordingHandle:
        def __init__(self, handle: Any):
            self._handle = handle

        def __enter__(self) -> RecordingHandle:
            self._handle.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._handle.__exit__(*args)

        def fileno(self) -> int:
            return self._handle.fileno()

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self._handle.read(size)

    def grow_after_open(fd: int) -> os.stat_result:
        nonlocal grew
        result = real_fstat(fd)
        if not grew and (result.st_dev, result.st_ino) == identity:
            grew = True
            with target.open("ab") as output:
                output.write(b"x" * file_limit)
        return result

    def record_target_reads(fd: int, *args: object, **kwargs: object) -> Any:
        handle = real_fdopen(fd, *args, **kwargs)
        opened = real_fstat(fd)
        return (
            RecordingHandle(handle)
            if (opened.st_dev, opened.st_ino) == identity
            else handle
        )

    monkeypatch.setattr(os, "fstat", grow_after_open)
    monkeypatch.setattr(os, "fdopen", record_target_reads)
    runtime = Runtime.open("local", config=config)
    try:
        with pytest.raises(
            ValidationError,
            match=rf"resource_read_max_bytes={file_limit}",
        ):
            runtime.skills.validate_package_path(package)
        assert grew
        assert read_sizes
        assert all(0 < size <= file_limit + 1 for size in read_sizes)
        assert runtime.store.get_skill("growth-skill") is None
    finally:
        runtime.close()


def test_host_skill_package_stops_at_cumulative_budget_before_reading_all_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SKILL.md plus 255 resources exercises the full 256-file default package
    # surface while staying valid under the fixture helper's default runtime.
    package = _write_many_resources(tmp_path, count=255, size=8192)
    config = replace(
        DEFAULT_CONFIG,
        skills=replace(
            DEFAULT_CONFIG.skills,
            resource_read_max_bytes=8192,
            package_max_bytes=16384,
            max_package_files=300,
        ),
    )
    from agent_libos.skills import manager as skill_manager_module

    real_open = skill_manager_module.open_secure_file
    opened: list[Path] = []

    def record_open(path: str | Path, **kwargs: object) -> Any:
        opened.append(Path(path))
        return real_open(path, **kwargs)

    monkeypatch.setattr(skill_manager_module, "open_secure_file", record_open)
    runtime = Runtime.open("local", config=config)
    try:
        with pytest.raises(ValidationError, match="package_max_bytes=16384"):
            runtime.skills.validate_package_path(package)
        assert len(opened) <= 3
        assert runtime.store.get_skill("bounded-package") is None
    finally:
        runtime.close()


def test_workspace_skill_package_stops_at_cumulative_budget_before_reading_all_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _write_many_resources(tmp_path, count=255, size=8192)
    config = replace(
        DEFAULT_CONFIG,
        skills=replace(
            DEFAULT_CONFIG.skills,
            resource_read_max_bytes=8192,
            package_max_bytes=16384,
            max_package_files=300,
        ),
    )
    runtime = Runtime.open(
        "local",
        config=config,
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        pid = runtime.process.spawn(goal="reject an oversized workspace Skill")
        runtime.filesystem.grant_path(
            pid,
            f"{package.name}/SKILL.md",
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        runtime.filesystem.grant_directory(
            pid,
            f"{package.name}/assets",
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        real_read = runtime.filesystem.read_bytes
        reads: list[str] = []

        def record_read(
            selected_pid: str,
            path: str,
            **kwargs: object,
        ) -> Any:
            reads.append(path)
            return real_read(selected_pid, path, **kwargs)

        monkeypatch.setattr(runtime.filesystem, "read_bytes", record_read)
        with pytest.raises(ValidationError, match="package_max_bytes=16384"):
            runtime.skills.register_skill_from_workspace_path(
                pid,
                package.name,
                require_capability=False,
            )
        assert len(reads) <= 3
        assert runtime.store.get_skill("bounded-package") is None
    finally:
        runtime.close()


def test_host_and_workspace_stable_snapshots_produce_the_same_package_hash(
    tmp_path: Path,
) -> None:
    package = write_skill_package(
        tmp_path,
        "stable-skill",
        allowed_tools=["echo"],
        extra_resources={"references/guide.md": "stable resource\n"},
    )
    host = Runtime.open("local")
    try:
        host_hash = host.skills.validate_package_path(package)["package_sha256"]
    finally:
        host.close()

    workspace = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        pid = workspace.process.spawn(goal="load a stable workspace Skill")
        workspace.filesystem.grant_path(
            pid,
            "stable-skill/SKILL.md",
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        workspace.filesystem.grant_directory(
            pid,
            "stable-skill/references",
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        workspace_hash = workspace.skills.register_skill_from_workspace_path(
            pid,
            "stable-skill",
            require_capability=False,
        )["package_sha256"]
        assert workspace_hash == host_hash
    finally:
        workspace.close()
