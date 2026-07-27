from __future__ import annotations

import errno
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000


@dataclass(frozen=True)
class WindowsOpenContract:
    desired_access: int
    share_mode: int
    creation_disposition: int
    flags_and_attributes: int


@dataclass(frozen=True)
class StablePathSnapshot:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int
    is_reparse_point: bool = False
    replacement_locked: bool = False


@dataclass(frozen=True)
class _PosixAncestorEdge:
    parent_descriptor: int
    name: str
    snapshot: StablePathSnapshot


class SecureFileLimitExceeded(Exception):
    pass


class SecureFileChanged(Exception):
    pass


class SecureFileReadUnavailable(Exception):
    pass


def windows_open_contract(
    *,
    directory: bool,
    ancestor: bool = False,
) -> WindowsOpenContract:
    """Return the non-following, replacement-blocking Win32 open contract."""

    flags = _FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _FILE_FLAG_BACKUP_SEMANTICS
    else:
        flags |= _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_SEQUENTIAL_SCAN
    return WindowsOpenContract(
        desired_access=_GENERIC_READ,
        # Target files/directories admit readers only, preventing replacement
        # or mutation while their bytes and identity are validated. Ancestor
        # guards permit ordinary child writes but never FILE_SHARE_DELETE, so
        # no path component can be renamed while a target is reached by name.
        share_mode=(
            _FILE_SHARE_READ | _FILE_SHARE_WRITE
            if ancestor
            else _FILE_SHARE_READ
        ),
        creation_disposition=_OPEN_EXISTING,
        flags_and_attributes=flags,
    )


def snapshot_from_stat(value: os.stat_result) -> StablePathSnapshot:
    attributes = int(getattr(value, "st_file_attributes", 0))
    return StablePathSnapshot(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=stat.S_IFMT(value.st_mode),
        links=int(value.st_nlink),
        size=int(value.st_size),
        modified_ns=_stat_ns(value, "st_mtime_ns", "st_mtime"),
        changed_ns=_stat_ns(value, "st_ctime_ns", "st_ctime"),
        is_reparse_point=bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT),
    )


def stable_identity_available(snapshot: StablePathSnapshot) -> bool:
    """Return whether identity or an equivalent held-handle lock is present."""

    return (
        snapshot.device >= 0
        and snapshot.inode >= 0
        and (snapshot.inode > 0 or snapshot.replacement_locked)
    )


class SecureFileDescriptor:
    def __init__(
        self,
        path: Path,
        descriptor: int,
        *,
        parent_descriptor: int | None = None,
        parent_guard: SecureDirectoryGuard | None = None,
        relative_name: str | None = None,
        windows_handle: int | None = None,
        windows_api: Any | None = None,
        windows_ancestor_handles: tuple[int, ...] = (),
        posix_ancestor_descriptors: tuple[int, ...] = (),
        posix_ancestor_edges: tuple[_PosixAncestorEdge, ...] = (),
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self._parent_descriptor = parent_descriptor
        self._parent_guard = parent_guard
        self._relative_name = relative_name
        self._windows_handle = windows_handle
        self._windows_api = windows_api
        self._windows_ancestor_handles = windows_ancestor_handles
        self._posix_ancestor_descriptors = posix_ancestor_descriptors
        self._posix_ancestor_edges = posix_ancestor_edges
        self._descriptor_owned = True
        self._path_guards_released = False

    def snapshot(self) -> StablePathSnapshot:
        if self._windows_api is not None:
            return self._windows_api.snapshot(self._windows_handle)
        return snapshot_from_stat(os.fstat(self.descriptor))

    def linked_snapshot(self) -> StablePathSnapshot:
        if self._windows_api is not None:
            # The Win32 handle deliberately omits FILE_SHARE_DELETE and
            # FILE_SHARE_WRITE, so its final path cannot be replaced and new
            # writers cannot be admitted until this descriptor is closed.
            return self.snapshot()
        if self._parent_guard is not None:
            self._parent_guard.verify_path_guard()
        _verify_posix_ancestor_edges(self._posix_ancestor_edges)
        if self._parent_descriptor is None:
            value = os.stat(self.path, follow_symlinks=False)
        else:
            value = os.stat(
                self._relative_name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
        return snapshot_from_stat(value)

    def open_binary(self) -> Any:
        handle = os.fdopen(self.descriptor, "rb")
        self._descriptor_owned = False
        return handle

    def close(self) -> None:
        if self._descriptor_owned:
            self._descriptor_owned = False
            os.close(self.descriptor)
        self.release_path_guards()

    def release_path_guards(self) -> None:
        if self._path_guards_released:
            return
        self._path_guards_released = True
        first_error: BaseException | None = None
        if self._windows_api is not None:
            for handle in reversed(self._windows_ancestor_handles):
                try:
                    self._windows_api.close_handle(handle)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        else:
            for descriptor in reversed(self._posix_ancestor_descriptors):
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error


def read_stable_file_limited(
    secure_file: SecureFileDescriptor,
    *,
    max_bytes: int,
    chunk_bytes: int,
    validate_snapshot: Callable[
        [StablePathSnapshot, bool],
        StablePathSnapshot,
    ],
) -> bytes:
    """Read one identity-stable descriptor to EOF with a strict byte cap."""

    try:
        handle = secure_file.open_binary()
    except BaseException:
        try:
            secure_file.close()
        except OSError:
            pass
        raise
    try:
        with handle:
            opened = validate_snapshot(secure_file.snapshot(), False)
            if opened.size > max_bytes:
                raise SecureFileLimitExceeded
            content = bytearray()
            reached_eof = False
            while len(content) <= max_bytes:
                read_size = min(chunk_bytes, max_bytes + 1 - len(content))
                chunk = handle.read(read_size)
                if chunk == b"":
                    reached_eof = True
                    break
                if chunk is None:
                    raise SecureFileReadUnavailable
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise SecureFileLimitExceeded
            if not reached_eof:
                raise SecureFileLimitExceeded
            after_read = validate_snapshot(secure_file.snapshot(), True)
            linked = validate_snapshot(secure_file.linked_snapshot(), True)
            if (
                after_read != opened
                or linked != after_read
                or len(content) != opened.size
            ):
                raise SecureFileChanged
            return bytes(content)
    finally:
        secure_file.release_path_guards()


class SecureDirectoryGuard:
    def __init__(
        self,
        path: Path,
        *,
        descriptor: int | None = None,
        parent_descriptor: int | None = None,
        parent_guard: SecureDirectoryGuard | None = None,
        relative_name: str | None = None,
        windows_handle: int | None = None,
        windows_api: Any | None = None,
        windows_ancestor_handles: tuple[int, ...] = (),
        posix_ancestor_descriptors: tuple[int, ...] = (),
        posix_ancestor_edges: tuple[_PosixAncestorEdge, ...] = (),
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self._parent_descriptor = parent_descriptor
        self._parent_guard = parent_guard
        self._relative_name = relative_name
        self._windows_handle = windows_handle
        self._windows_api = windows_api
        self._windows_ancestor_handles = windows_ancestor_handles
        self._posix_ancestor_descriptors = posix_ancestor_descriptors
        self._posix_ancestor_edges = posix_ancestor_edges
        self._closed = False

    def snapshot(self) -> StablePathSnapshot:
        if self._windows_api is not None:
            return self._windows_api.snapshot(self._windows_handle)
        if self.descriptor is None:
            raise OSError("secure directory descriptor is unavailable")
        return snapshot_from_stat(os.fstat(self.descriptor))

    def linked_snapshot(self) -> StablePathSnapshot:
        if self._windows_api is not None:
            return self.snapshot()
        self.verify_path_guard()
        if self._parent_descriptor is None:
            return self.snapshot()
        else:
            value = os.stat(
                self._relative_name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
        return snapshot_from_stat(value)

    def verify_path_guard(self) -> None:
        if self._windows_api is not None:
            return
        if self._parent_guard is not None:
            self._parent_guard.verify_path_guard()
        _verify_posix_ancestor_edges(self._posix_ancestor_edges)

    def scandir(self) -> Any:
        if self._windows_api is not None:
            return os.scandir(self.path)
        if self.descriptor is None:
            raise OSError("secure directory descriptor is unavailable")
        return os.scandir(self.descriptor)

    def lstat_child(self, name: str) -> StablePathSnapshot:
        _validate_child_path(
            self.path / name if isinstance(name, str) else self.path,
            parent=self,
            relative_name=name,
        )
        if self._windows_api is not None:
            return snapshot_from_stat(os.stat(self.path / name, follow_symlinks=False))
        if self.descriptor is None:
            raise OSError("secure directory descriptor is unavailable")
        return snapshot_from_stat(
            os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        )

    def open_child_directory(self, name: str) -> "SecureDirectoryGuard":
        return open_secure_directory(
            self.path / name,
            parent=self,
            relative_name=name,
            windows_api=self._windows_api,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._windows_api is not None:
            first_error: BaseException | None = None
            for handle in (
                self._windows_handle,
                *reversed(self._windows_ancestor_handles),
            ):
                try:
                    self._windows_api.close_handle(handle)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            if first_error is not None:
                raise first_error
        else:
            first_error = None
            if self.descriptor is not None:
                try:
                    os.close(self.descriptor)
                except BaseException as exc:
                    first_error = exc
            for descriptor in reversed(self._posix_ancestor_descriptors):
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            if first_error is not None:
                raise first_error

    def __enter__(self) -> "SecureDirectoryGuard":
        return self

    def __exit__(self, _exc_type: Any, _exc: BaseException | None, _tb: Any) -> None:
        self.close()


def open_secure_file(
    path: str | Path,
    *,
    parent: SecureDirectoryGuard | None = None,
    relative_name: str | None = None,
    platform: str | None = None,
    windows_api: Any | None = None,
) -> SecureFileDescriptor:
    selected = Path(path)
    _validate_child_path(selected, parent=parent, relative_name=relative_name)
    selected_platform = (
        "nt"
        if windows_api is not None
        else os.name if platform is None else platform
    )
    if selected_platform == "nt":
        api = _WINDOWS_API if windows_api is None else windows_api
        if api is None:
            raise OSError("Win32 secure file APIs are unavailable")
        ancestor_handles: tuple[int, ...] = ()
        if parent is None:
            selected = Path(os.path.abspath(selected))
            ancestor_handles = _open_windows_ancestor_guards(selected, api)
        try:
            descriptor, native_handle = api.open_file_descriptor(
                selected,
                windows_open_contract(directory=False),
            )
        except BaseException:
            for handle in reversed(ancestor_handles):
                api.close_handle(handle)
            raise
        return SecureFileDescriptor(
            selected,
            descriptor,
            parent_guard=parent,
            relative_name=relative_name,
            windows_handle=native_handle,
            windows_api=api,
            windows_ancestor_handles=ancestor_handles,
        )
    flags = _posix_open_flags(directory=False)
    parent_descriptor = parent.descriptor if parent is not None else None
    ancestor_descriptors: tuple[int, ...] = ()
    ancestor_edges: tuple[_PosixAncestorEdge, ...] = ()
    if parent_descriptor is None:
        selected, ancestor_descriptors, ancestor_edges, final_name = (
            _open_posix_ancestor_guards(selected)
        )
        if final_name is None:
            _close_posix_descriptors(ancestor_descriptors)
            raise OSError("secure file path names a directory root")
        parent_descriptor = ancestor_descriptors[-1]
        relative_name = final_name
    descriptor: int | None = None
    try:
        descriptor = os.open(relative_name, flags, dir_fd=parent_descriptor)
        if parent is not None:
            parent.verify_path_guard()
        _verify_posix_ancestor_edges(ancestor_edges)
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _close_posix_descriptors(ancestor_descriptors)
        raise
    return SecureFileDescriptor(
        selected,
        descriptor,
        parent_descriptor=parent_descriptor,
        parent_guard=parent,
        relative_name=relative_name,
        posix_ancestor_descriptors=ancestor_descriptors,
        posix_ancestor_edges=ancestor_edges,
    )


def open_secure_directory(
    path: str | Path,
    *,
    parent: SecureDirectoryGuard | None = None,
    relative_name: str | None = None,
    platform: str | None = None,
    windows_api: Any | None = None,
) -> SecureDirectoryGuard:
    selected = Path(path)
    _validate_child_path(selected, parent=parent, relative_name=relative_name)
    selected_platform = (
        "nt"
        if windows_api is not None
        else os.name if platform is None else platform
    )
    if selected_platform == "nt":
        api = _WINDOWS_API if windows_api is None else windows_api
        if api is None:
            raise OSError("Win32 secure file APIs are unavailable")
        ancestor_handles: tuple[int, ...] = ()
        if parent is None:
            selected = Path(os.path.abspath(selected))
            ancestor_handles = _open_windows_ancestor_guards(selected, api)
        native_handle: int | None = None
        try:
            native_handle = api.open_directory(
                selected,
                windows_open_contract(directory=True),
            )
            _require_stable_directory_snapshot(api.snapshot(native_handle))
        except BaseException:
            if native_handle is not None:
                api.close_handle(native_handle)
            for handle in reversed(ancestor_handles):
                api.close_handle(handle)
            raise
        return SecureDirectoryGuard(
            selected,
            parent_guard=parent,
            relative_name=relative_name,
            windows_handle=native_handle,
            windows_api=api,
            windows_ancestor_handles=ancestor_handles,
        )
    flags = _posix_open_flags(directory=True)
    parent_descriptor = parent.descriptor if parent is not None else None
    ancestor_descriptors: tuple[int, ...] = ()
    ancestor_edges: tuple[_PosixAncestorEdge, ...] = ()
    if parent_descriptor is None:
        selected, ancestor_descriptors, ancestor_edges, final_name = (
            _open_posix_ancestor_guards(selected)
        )
        if final_name is None:
            descriptor = ancestor_descriptors[-1]
            ancestor_descriptors = ancestor_descriptors[:-1]
            parent_descriptor = None
            relative_name = None
        else:
            parent_descriptor = ancestor_descriptors[-1]
            relative_name = final_name
            try:
                descriptor = os.open(relative_name, flags, dir_fd=parent_descriptor)
            except BaseException:
                _close_posix_descriptors(ancestor_descriptors)
                raise
    else:
        descriptor = os.open(relative_name, flags, dir_fd=parent_descriptor)
    try:
        if parent is not None:
            parent.verify_path_guard()
        _verify_posix_ancestor_edges(ancestor_edges)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        _close_posix_descriptors(ancestor_descriptors)
        raise
    return SecureDirectoryGuard(
        selected,
        descriptor=descriptor,
        parent_descriptor=parent_descriptor,
        parent_guard=parent,
        relative_name=relative_name,
        posix_ancestor_descriptors=ancestor_descriptors,
        posix_ancestor_edges=ancestor_edges,
    )


def _posix_open_flags(*, directory: bool) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        raise OSError("O_NOFOLLOW is unavailable")
    flags = (
        os.O_RDONLY
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if directory:
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if not isinstance(directory_flag, int) or directory_flag <= 0:
            raise OSError("O_DIRECTORY is unavailable")
        flags |= directory_flag
    return flags


def _open_posix_ancestor_guards(
    path: Path,
) -> tuple[
    Path,
    tuple[int, ...],
    tuple[_PosixAncestorEdge, ...],
    str | None,
]:
    """Open an absolute path root-to-parent without following any component."""

    selected = Path(os.path.abspath(path))
    anchor = selected.anchor
    if not anchor:
        raise OSError("secure POSIX path has no filesystem anchor")
    relative_parts = selected.relative_to(Path(anchor)).parts
    directory_flags = _posix_open_flags(directory=True)
    descriptors: list[int] = []
    edges: list[_PosixAncestorEdge] = []
    try:
        root_descriptor = os.open(anchor, directory_flags)
        descriptors.append(root_descriptor)
        _require_stable_directory_snapshot(
            snapshot_from_stat(os.fstat(root_descriptor))
        )
        original_parts = relative_parts
        relative_parts = _expand_darwin_root_alias(
            root_descriptor,
            relative_parts,
        )
        if relative_parts != original_parts:
            selected = Path(anchor).joinpath(*relative_parts)
        for component in relative_parts[:-1]:
            parent_descriptor = descriptors[-1]
            try:
                descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    try:
                        blocked = snapshot_from_stat(
                            os.stat(
                                component,
                                dir_fd=parent_descriptor,
                                follow_symlinks=False,
                            )
                        )
                    except OSError:
                        pass
                    else:
                        if stat.S_ISLNK(blocked.mode):
                            raise OSError(
                                errno.ELOOP,
                                "secure POSIX ancestor is a symlink",
                                component,
                            ) from exc
                raise
            descriptors.append(descriptor)
            opened = snapshot_from_stat(os.fstat(descriptor))
            _require_stable_directory_snapshot(opened)
            linked = snapshot_from_stat(
                os.stat(
                    component,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            )
            _require_stable_directory_snapshot(linked)
            if not _same_stable_identity(linked, opened):
                raise OSError(
                    "secure POSIX ancestor changed while it was opened"
                )
            edges.append(
                _PosixAncestorEdge(
                    parent_descriptor=parent_descriptor,
                    name=component,
                    snapshot=opened,
                )
            )
        _verify_posix_ancestor_edges(tuple(edges))
        final_name = relative_parts[-1] if relative_parts else None
        return selected, tuple(descriptors), tuple(edges), final_name
    except BaseException:
        _close_posix_descriptors(tuple(descriptors))
        raise


def _verify_posix_ancestor_edges(
    edges: tuple[_PosixAncestorEdge, ...],
) -> None:
    for edge in edges:
        linked = snapshot_from_stat(
            os.stat(
                edge.name,
                dir_fd=edge.parent_descriptor,
                follow_symlinks=False,
            )
        )
        _require_stable_directory_snapshot(linked)
        if not _same_stable_identity(linked, edge.snapshot):
            raise OSError("secure POSIX ancestor changed during access")


def _same_stable_identity(
    current: StablePathSnapshot,
    expected: StablePathSnapshot,
) -> bool:
    return (
        current.device == expected.device
        and current.inode == expected.inode
        and current.mode == expected.mode
        and current.is_reparse_point == expected.is_reparse_point
        and current.links >= 1
        and expected.links >= 1
    )


def _expand_darwin_root_alias(
    root_descriptor: int,
    parts: tuple[str, ...],
) -> tuple[str, ...]:
    """Map Darwin's fixed root aliases without allowing general symlinks."""

    if sys.platform != "darwin" or not parts:
        return parts
    alias = parts[0]
    if alias not in {"etc", "tmp", "var"}:
        return parts
    try:
        alias_stat = os.stat(
            alias,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        target = os.readlink(alias, dir_fd=root_descriptor)
    except OSError:
        return parts
    if not stat.S_ISLNK(alias_stat.st_mode):
        return parts
    expected = f"private/{alias}"
    if target not in {expected, f"/{expected}"}:
        return parts
    return ("private", alias, *parts[1:])


def _close_posix_descriptors(descriptors: tuple[int, ...]) -> None:
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _validate_child_path(
    path: Path,
    *,
    parent: SecureDirectoryGuard | None,
    relative_name: str | None,
) -> None:
    if parent is None:
        if relative_name is not None:
            raise OSError("relative name requires a secure parent directory")
        return
    if not isinstance(relative_name, str) or not relative_name:
        raise OSError("secure child open requires a relative name")
    invalid_component = (
        relative_name in {".", ".."}
        or "\0" in relative_name
        or Path(relative_name).name != relative_name
    )
    if parent._windows_api is not None:
        windows_name = PureWindowsPath(relative_name)
        invalid_component = invalid_component or bool(
            windows_name.drive
            or windows_name.root
            or windows_name.name != relative_name
        )
    if invalid_component:
        raise OSError("secure child open name must be one path component")
    expected = Path(os.path.abspath(parent.path / relative_name))
    if Path(os.path.abspath(path)) != expected:
        raise OSError("secure child path does not match its guarded parent")


def _stat_ns(
    value: os.stat_result,
    nanoseconds_field: str,
    seconds_field: str,
) -> int:
    nanoseconds = getattr(value, nanoseconds_field, None)
    if isinstance(nanoseconds, int):
        return nanoseconds
    return int(float(getattr(value, seconds_field)) * 1_000_000_000)


def _open_windows_ancestor_guards(path: Path, api: Any) -> tuple[int, ...]:
    handles: list[int] = []
    try:
        for ancestor in reversed(path.parents):
            handle = api.open_directory(
                ancestor,
                windows_open_contract(directory=True, ancestor=True),
            )
            handles.append(handle)
            _require_stable_directory_snapshot(api.snapshot(handle))
    except BaseException:
        for handle in reversed(handles):
            api.close_handle(handle)
        raise
    return tuple(handles)


def _require_stable_directory_snapshot(snapshot: StablePathSnapshot) -> None:
    if (
        snapshot.is_reparse_point
        or not stat.S_ISDIR(snapshot.mode)
        or snapshot.links < 1
        or not stable_identity_available(snapshot)
    ):
        raise OSError("secure directory identity is unavailable or is a reparse point")


if os.name == "nt":
    import ctypes
    import msvcrt

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", ctypes.c_uint32),
            ("dwHighDateTime", ctypes.c_uint32),
        ]

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", ctypes.c_uint32),
            ("ftCreationTime", _FILETIME),
            ("ftLastAccessTime", _FILETIME),
            ("ftLastWriteTime", _FILETIME),
            ("dwVolumeSerialNumber", ctypes.c_uint32),
            ("nFileSizeHigh", ctypes.c_uint32),
            ("nFileSizeLow", ctypes.c_uint32),
            ("nNumberOfLinks", ctypes.c_uint32),
            ("nFileIndexHigh", ctypes.c_uint32),
            ("nFileIndexLow", ctypes.c_uint32),
        ]

    class _WindowsHostFileAPI:
        def __init__(self) -> None:
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._kernel32.CreateFileW.argtypes = [
                ctypes.c_wchar_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
            ]
            self._kernel32.CreateFileW.restype = ctypes.c_void_p
            self._kernel32.GetFileInformationByHandle.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
            ]
            self._kernel32.GetFileInformationByHandle.restype = ctypes.c_int
            self._kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            self._kernel32.CloseHandle.restype = ctypes.c_int
            self._invalid_handle = ctypes.c_void_p(-1).value

        def _open(self, path: Path, contract: WindowsOpenContract) -> int:
            handle = self._kernel32.CreateFileW(
                os.fspath(path),
                contract.desired_access,
                contract.share_mode,
                None,
                contract.creation_disposition,
                contract.flags_and_attributes,
                None,
            )
            if not handle or handle == self._invalid_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            return int(handle)

        def open_file_descriptor(
            self,
            path: Path,
            contract: WindowsOpenContract,
        ) -> tuple[int, int]:
            handle = self._open(path, contract)
            descriptor: int | None = None
            try:
                descriptor = msvcrt.open_osfhandle(
                    handle,
                    os.O_RDONLY | getattr(os, "O_BINARY", 0),
                )
                os.set_inheritable(descriptor, False)
            except BaseException:
                if descriptor is not None:
                    os.close(descriptor)
                else:
                    self.close_handle(handle)
                raise
            if descriptor is None:
                self.close_handle(handle)
                raise OSError("Win32 file descriptor conversion returned no descriptor")
            return descriptor, handle

        def open_directory(
            self,
            path: Path,
            contract: WindowsOpenContract,
        ) -> int:
            return self._open(path, contract)

        def snapshot(self, handle: int) -> StablePathSnapshot:
            info = _BY_HANDLE_FILE_INFORMATION()
            if not self._kernel32.GetFileInformationByHandle(
                handle,
                ctypes.byref(info),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            attributes = int(info.dwFileAttributes)
            mode = (
                stat.S_IFDIR
                if attributes & _FILE_ATTRIBUTE_DIRECTORY
                else stat.S_IFREG
            )
            return StablePathSnapshot(
                device=int(info.dwVolumeSerialNumber),
                inode=(int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow),
                mode=mode,
                links=int(info.nNumberOfLinks),
                size=(int(info.nFileSizeHigh) << 32) | int(info.nFileSizeLow),
                modified_ns=_filetime_value(info.ftLastWriteTime) * 100,
                # Windows descriptor ctime can disagree with path ctime on
                # supported Python versions; creation time is stable for the
                # same file ID and is sufficient while replacement is locked.
                changed_ns=_filetime_value(info.ftCreationTime) * 100,
                is_reparse_point=bool(
                    attributes & _FILE_ATTRIBUTE_REPARSE_POINT
                ),
                # A zero legacy file index is possible on some Win32-backed
                # filesystems. The no-write/no-delete target handle and the
                # no-delete ancestor chain, rather than a nonzero index alone,
                # provide the replacement guarantee for this snapshot.
                replacement_locked=True,
            )

        def close_handle(self, handle: int) -> None:
            if not self._kernel32.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())

    def _filetime_value(value: _FILETIME) -> int:
        return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

    _WINDOWS_API: Any | None = _WindowsHostFileAPI()
else:
    _WINDOWS_API = None
