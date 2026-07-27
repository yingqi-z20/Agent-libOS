from __future__ import annotations

import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos.utils.secure_host_files import (
    StablePathSnapshot,
    WindowsOpenContract,
    open_secure_directory,
    open_secure_file,
    snapshot_from_stat,
    stable_identity_available,
    windows_open_contract,
)


class _FakeWindowsAPI:
    def __init__(self) -> None:
        self.file_contracts: list[WindowsOpenContract] = []
        self.directory_contracts: list[WindowsOpenContract] = []
        self._directory_snapshots: dict[int, StablePathSnapshot] = {}
        self._next_directory_handle = -1

    def open_file_descriptor(
        self,
        path: Path,
        contract: WindowsOpenContract,
    ) -> tuple[int, int]:
        self.file_contracts.append(contract)
        descriptor = os.open(path, os.O_RDONLY)
        return descriptor, descriptor

    def open_directory(
        self,
        path: Path,
        contract: WindowsOpenContract,
    ) -> int:
        self.directory_contracts.append(contract)
        handle = self._next_directory_handle
        self._next_directory_handle -= 1
        self._directory_snapshots[handle] = snapshot_from_stat(path.stat())
        return handle

    def snapshot(self, handle: int) -> StablePathSnapshot:
        directory = self._directory_snapshots.get(handle)
        if directory is not None:
            return directory
        return snapshot_from_stat(os.fstat(handle))

    def close_handle(self, handle: int) -> None:
        if handle in self._directory_snapshots:
            del self._directory_snapshots[handle]
            return
        os.close(handle)


def test_windows_file_open_contract_blocks_replacement_and_opens_reparse_point(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_bytes(b"safe module source")
    api = _FakeWindowsAPI()

    secure_file = open_secure_file(
        source,
        platform="nt",
        windows_api=api,
    )
    try:
        with secure_file.open_binary() as handle:
            assert secure_file.snapshot().size == len(b"safe module source")
            assert handle.read() == b"safe module source"
    finally:
        secure_file.release_path_guards()

    contract = api.file_contracts[0]
    assert contract.share_mode == 0x00000001
    assert contract.share_mode & (0x00000002 | 0x00000004) == 0
    assert contract.flags_and_attributes & 0x00200000
    assert all(
        ancestor.share_mode & 0x00000004 == 0
        for ancestor in api.directory_contracts
    )


def test_zero_file_index_requires_a_held_win32_replacement_lock() -> None:
    snapshot = StablePathSnapshot(
        device=0,
        inode=0,
        mode=0,
        links=1,
        size=0,
        modified_ns=0,
        changed_ns=0,
    )

    assert not stable_identity_available(snapshot)
    assert stable_identity_available(
        replace(snapshot, replacement_locked=True)
    )


def test_windows_directory_open_contract_guards_each_enumerated_parent(
    tmp_path: Path,
) -> None:
    (tmp_path / "child.txt").write_text("content", encoding="utf-8")
    api = _FakeWindowsAPI()

    with open_secure_directory(
        tmp_path,
        platform="nt",
        windows_api=api,
    ) as directory:
        opened = directory.snapshot()
        with directory.scandir() as entries:
            assert {entry.name for entry in entries} == {"child.txt"}
        assert directory.linked_snapshot() == opened

    contract = api.directory_contracts[-1]
    assert contract == windows_open_contract(directory=True)
    assert contract.share_mode == 0x00000001
    assert contract.share_mode & (0x00000002 | 0x00000004) == 0
    assert contract.flags_and_attributes & 0x00200000
    assert contract.flags_and_attributes & 0x02000000
    assert all(
        ancestor.share_mode & 0x00000004 == 0
        for ancestor in api.directory_contracts[:-1]
    )


@pytest.mark.parametrize(
    "child_name",
    ["../outside.txt", "subdir/child.txt", "/tmp/outside.txt", ".", "..", "", "bad\0name"],
)
def test_secure_directory_lstat_child_rejects_non_child_names(
    tmp_path: Path,
    child_name: str,
) -> None:
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")

    with open_secure_directory(guarded) as directory:
        with pytest.raises(OSError, match="relative name|one path component"):
            directory.lstat_child(child_name)


def test_secure_directory_lstat_child_rejects_non_string_name(tmp_path: Path) -> None:
    guarded = tmp_path / "guarded"
    guarded.mkdir()

    with open_secure_directory(guarded) as directory:
        with pytest.raises(OSError, match="relative name"):
            directory.lstat_child(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("child_name", [r"subdir\child.txt", r"C:child.txt"])
def test_windows_secure_directory_lstat_child_rejects_windows_paths(
    tmp_path: Path,
    child_name: str,
) -> None:
    api = _FakeWindowsAPI()

    with open_secure_directory(tmp_path, platform="nt", windows_api=api) as directory:
        with pytest.raises(OSError, match="one path component"):
            directory.lstat_child(child_name)


@pytest.mark.skipif(os.name == "nt", reason="backslash is a separator on Windows")
def test_posix_secure_directory_lstat_child_accepts_literal_backslash(
    tmp_path: Path,
) -> None:
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    child = guarded / r"literal\name"
    child.write_text("content", encoding="utf-8")

    with open_secure_directory(guarded) as directory:
        snapshot = directory.lstat_child(child.name)

    assert snapshot.size == len(b"content")


def test_secure_directory_lstat_child_accepts_one_direct_child(tmp_path: Path) -> None:
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    child = guarded / "child.txt"
    child.write_text("content", encoding="utf-8")

    with open_secure_directory(guarded) as directory:
        snapshot = directory.lstat_child("child.txt")

    assert snapshot.size == len(b"content")


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX dir_fd semantics")
@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_posix_secure_open_rejects_symlinked_intermediate_component(
    tmp_path: Path,
    target_kind: str,
) -> None:
    real_parent = tmp_path / "real-parent"
    package = real_parent / "package"
    package.mkdir(parents=True)
    target = package if target_kind == "directory" else package / "module.py"
    if target_kind == "file":
        target.write_text("VALUE = 1\n", encoding="utf-8")
    alias = tmp_path / "alias-parent"
    alias.symlink_to(real_parent, target_is_directory=True)
    selected = alias / target.relative_to(real_parent)

    with pytest.raises(OSError):
        if target_kind == "directory":
            open_secure_directory(selected)
        else:
            open_secure_file(selected)


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX dir_fd semantics")
@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_posix_secure_open_detects_intermediate_directory_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_kind: str,
) -> None:
    ancestor = tmp_path / "guarded-parent"
    package = ancestor / "package"
    package.mkdir(parents=True)
    source = package / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    parked = tmp_path / "parked-parent"
    ancestor_identity = (ancestor.stat().st_dev, ancestor.stat().st_ino)
    real_open = os.open
    real_fstat = os.fstat
    swapped = False

    def swap_after_ancestor_open(
        path: os.PathLike[str] | str,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        opened = real_fstat(descriptor)
        if not swapped and (opened.st_dev, opened.st_ino) == ancestor_identity:
            swapped = True
            os.replace(ancestor, parked)
            shutil.copytree(parked, ancestor)
        return descriptor

    monkeypatch.setattr(os, "open", swap_after_ancestor_open)
    selected = package if target_kind == "directory" else source

    with pytest.raises(OSError):
        if target_kind == "directory":
            open_secure_directory(selected)
        else:
            open_secure_file(selected)

    assert swapped


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="requires real Win32 sharing and rename semantics",
)
def test_real_windows_directory_guards_block_root_and_ancestor_replacement(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "guarded-parent"
    package = parent / "package"
    package.mkdir(parents=True)
    (package / "source.py").write_text("VALUE = 1\n", encoding="utf-8")

    with open_secure_directory(package) as directory:
        with pytest.raises(OSError):
            os.replace(parent, tmp_path / "moved-parent")
        with pytest.raises(OSError):
            os.replace(package, parent / "moved-package")
        with directory.scandir() as entries:
            assert {entry.name for entry in entries} == {"source.py"}


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="requires real Win32 sharing and replacement semantics",
)
def test_real_windows_file_guard_blocks_write_and_replacement_until_close(
    tmp_path: Path,
) -> None:
    target = tmp_path / "guarded.py"
    parked = tmp_path / "parked.py"
    target.write_bytes(b"VALUE = 1\n")
    secure_file = open_secure_file(target)
    try:
        with secure_file.open_binary() as handle:
            assert handle.read() == b"VALUE = 1\n"
            with pytest.raises(OSError):
                with target.open("r+b"):
                    pass
            with pytest.raises(OSError):
                os.replace(target, parked)
    finally:
        secure_file.release_path_guards()

    os.replace(target, parked)
    os.replace(parked, target)
