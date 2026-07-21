from __future__ import annotations

import asyncio
import contextlib
import ctypes
import errno
import hashlib
import http.client
import heapq
import ipaddress
import os
import re
import signal
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

import psutil

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    JsonRpcEndpointSpec,
    JsonRpcMethodSpec,
    JsonRpcTransportResult,
    McpProviderCallResult,
    McpProviderTool,
    McpServerSpec,
    McpToolListResult,
    McpToolSpec,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.primitives.git_command_policy import trusted_git_read_operation
from agent_libos.models.external_effect import default_external_effect_rollback_status
from agent_libos.substrate.base import (
    CommandMetrics,
    CommandResult,
    DirectoryEntrySnapshot,
    ExecutableSnapshot,
    HierarchicalPathLock,
    PathState,
    ResolvedPath,
    resolve_runtime_python_alias,
    snapshot_executable,
    SubprocessLimitExceeded,
    SubprocessLimits,
    SubprocessTimeoutExpired,
)
from agent_libos.substrate.git import LocalGitProvider
from agent_libos.utils.ids import new_id
from agent_libos.utils.serde import dumps, to_jsonable

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_SHELL_DEFAULTS = DEFAULT_CONFIG.shell
_MCP_LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}
_MCP_FORBIDDEN_HOSTS = {"metadata.google.internal"}
_MCP_STDIO_READ_CHUNK_BYTES = 64 * 1024
_SAFE_SHELL_ENV_KEYS = {
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
}

if os.name == "nt":
    import msvcrt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    _kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    _kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.SetInformationJobObject.restype = ctypes.c_int
    _kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    _kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    _kernel32.CreateFileW.restype = ctypes.c_void_p
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_int
    _kernel32.GetFinalPathNameByHandleW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
    _kernel32.GetFinalPathNameByHandleW.restype = ctypes.c_uint32

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_NAME_NORMALIZED = 0
    _VOLUME_NAME_DOS = 0
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class WindowsJobObject:
    def __init__(self, handle: int):
        self.handle = handle
        self._closed = False

    @classmethod
    def create(cls) -> "WindowsJobObject":
        if os.name != "nt":
            raise OSError("Windows job objects are only available on Windows")
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.get_last_error()
            _kernel32.CloseHandle(handle)
            raise ctypes.WinError(error)
        return cls(int(handle))

    def assign(self, proc: subprocess.Popen[str]) -> None:
        process_handle = getattr(proc, "_handle", None)
        if process_handle is None:
            raise OSError("subprocess handle is unavailable for job assignment")
        if not _kernel32.AssignProcessToJobObject(self.handle, int(process_handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_pid(self, pid: int) -> None:
        """Attach an asynchronously launched process by pid.

        asyncio does not expose the ``subprocess.Popen`` handle portably.  A
        short-lived OpenProcess handle is sufficient for job assignment and is
        closed immediately after the process joins the job.
        """

        if os.name != "nt":
            raise OSError("Windows job objects are only available on Windows")
        process_handle = _kernel32.OpenProcess(
            _PROCESS_TERMINATE | _PROCESS_SET_QUOTA,
            False,
            int(pid),
        )
        if not process_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not _kernel32.AssignProcessToJobObject(self.handle, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _kernel32.CloseHandle(process_handle)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if os.name == "nt":
            _kernel32.CloseHandle(self.handle)


class _WindowsDirectoryGuard:
    def __init__(self, handle: int):
        self.handle = handle
        self._closed = False

    @classmethod
    def open(cls, path: Path) -> "_WindowsDirectoryGuard":
        if os.name != "nt":
            raise OSError("Windows directory guards are only available on Windows")
        handle = _kernel32.CreateFileW(
            os.fspath(path),
            _GENERIC_READ,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS,
            None,
        )
        if not handle or handle == _INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())
        return cls(int(handle))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if os.name == "nt":
            _kernel32.CloseHandle(self.handle)


class LocalFilesystemProvider:
    """Local-workspace implementation of the filesystem substrate."""

    def __init__(self, root: str | Path, namespace: str = _RUNTIME_DEFAULTS.workspace_namespace):
        self.root = Path(root).resolve()
        self.namespace = namespace
        self.root_display = str(self.root)
        self._path_lock = HierarchicalPathLock()

    def resolve(self, path: Any) -> ResolvedPath:
        raw = Path(path)
        candidate = raw if raw.is_absolute() else self.root / raw
        # Resource derivation runs before capability authorization.  Keep this
        # step purely lexical: Path.resolve() would touch the host filesystem,
        # follow symlinks, and expose their canonical target before the caller
        # has any read authority.  Provider sinks call _target() after
        # authorization; that method performs real-path containment and rejects
        # symlink/junction components on the original lexical path.
        target = Path(os.path.abspath(os.path.normpath(os.fspath(candidate))))
        try:
            relative_path = target.relative_to(self.root)
        except ValueError as exc:
            raise CapabilityDenied(f"path escapes filesystem adapter root: {path}")
        relative = relative_path.as_posix()
        return ResolvedPath(relative=relative, display=str(target), is_root=target == self.root)

    def state(self, path: ResolvedPath) -> PathState:
        with self._path_lock.hold(path.relative):
            target = self._target(path)
            if not target.exists():
                return PathState(exists=False, kind="missing")
            stat = target.stat()
            kind = "file" if target.is_file() else "directory" if target.is_dir() else "other"
            return PathState(
                exists=True,
                kind=kind,
                size_bytes=stat.st_size if target.is_file() else None,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            )

    def read_bytes(self, path: ResolvedPath, *, max_bytes: int | None = None) -> bytes:
        with self._path_lock.hold(path.relative):
            target = self._target(path)
            self._before_path_sink("read_bytes", target)
            target = self._target(path)
            with self._open_existing_file(target, os.O_RDONLY) as handle:
                if max_bytes is None:
                    return handle.read()
                return handle.read(max(0, max_bytes))

    def write_text(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None = "\n",
        *,
        overwrite: bool = True,
    ) -> None:
        with self._path_lock.hold(HierarchicalPathLock.creation_scope(path.relative)):
            target = self._target(path)
            self._before_path_sink_checked("write_parent", target.parent)
            self._ensure_parent_dirs_under_root(target)
            target = self._target(path)
            self._before_path_sink("write_text", target)
            target = self._target(path)
            with self._open_write_file(target, encoding=encoding, newline=newline, overwrite=overwrite) as handle:
                handle.write(text)
            self._target(path)

    def make_directory(self, path: ResolvedPath, *, parents: bool, exist_ok: bool) -> None:
        with self._path_lock.hold(HierarchicalPathLock.creation_scope(path.relative)):
            target = self._target(path)
            self._before_path_sink_checked("make_directory", target)
            self._make_directory_under_root(target, parents=parents, exist_ok=exist_ok)
            self._target(path)

    def list_directory(self, path: ResolvedPath, *, limit: int | None = None) -> list[DirectoryEntrySnapshot]:
        with self._path_lock.hold(path.relative):
            target = self._target(path)
            self._before_path_sink_checked("list_directory", target)
            return self._list_directory_under_root(target, limit=limit)

    def delete_file(self, path: ResolvedPath) -> None:
        with self._path_lock.hold(path.relative):
            target = self._target(path)
            self._delete_file_under_root(path, target)
            self._target(path)

    def delete_directory(self, path: ResolvedPath, *, recursive: bool) -> None:
        with self._path_lock.hold(path.relative):
            target = self._target(path)
            self._delete_directory_under_root(path, target, recursive=recursive)
            self._target(path)

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation in {"write_text", "make_directory", "delete_file", "delete_directory"}:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                state_mutation=True,
                information_flow=False,
                metadata={"namespace": self.namespace, "path": context.get("path")},
            )
        if operation in {"state", "read_bytes", "list_directory"}:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={"namespace": self.namespace, "path": context.get("path")},
            )
        raise ValueError(f"unsupported filesystem external effect operation: {operation}")

    def _target(self, path: ResolvedPath) -> Path:
        target = Path(path.display)
        resolved = target.resolve()
        if self.root not in resolved.parents and resolved != self.root:
            raise CapabilityDenied(f"path escapes filesystem adapter root: {path.relative}")
        self._reject_reparse_components(target)
        return target

    def _before_path_sink(self, operation: str, target: Path) -> None:
        return None

    def _reject_reparse_components(self, target: Path) -> None:
        try:
            relative_parts = target.relative_to(self.root).parts
        except ValueError as exc:
            raise CapabilityDenied(f"path escapes filesystem adapter root: {target}") from exc
        current = self.root
        for part in relative_parts:
            current = current / part
            if not current.exists() and not current.is_symlink():
                break
            if self._is_reparse_path(current):
                raise CapabilityDenied(f"filesystem path contains a symlink or junction: {current}")

    def _is_reparse_path(self, path: Path) -> bool:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction()) if callable(is_junction) else False

    def _open_existing_file(self, target: Path, flags: int) -> Any:
        fd = self._open_under_root(target, flags)
        try:
            self._validate_open_regular_file(fd, target)
            return os.fdopen(fd, "rb")
        except Exception:
            os.close(fd)
            raise

    def _open_write_file(self, target: Path, *, encoding: str, newline: str | None, overwrite: bool) -> Any:
        try:
            fd = self._open_under_root(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            if not overwrite:
                raise
            fd = self._open_under_root(target, os.O_WRONLY)
        try:
            self._validate_open_regular_file(fd, target)
            os.ftruncate(fd, 0)
            return os.fdopen(fd, "w", encoding=encoding, newline=newline)
        except Exception:
            os.close(fd)
            raise

    def _open_under_root(self, target: Path, flags: int, mode: int = 0o666) -> int:
        if os.open not in os.supports_dir_fd:
            return self._open_under_root_fallback(target, flags, mode)
        parts = self._relative_parts(target)
        if not parts:
            raise CapabilityDenied("filesystem operation requires a file path below the adapter root")
        dir_fd = self._open_root_dir_fd()
        try:
            for part in parts[:-1]:
                next_fd = self._open_dir_component(dir_fd, part)
                os.close(dir_fd)
                dir_fd = next_fd
            return self._open_file_component(dir_fd, parts[-1], flags, mode)
        finally:
            with contextlib.suppress(OSError):
                os.close(dir_fd)

    def _delete_file_under_root(self, path: ResolvedPath, target: Path) -> None:
        if self._supports_dir_fd_deletes():
            dir_fd, name = self._open_parent_dir_fd(target)
            try:
                self._before_path_sink_checked("delete_file", target)
                self._require_file_component_for_delete(dir_fd, name, target)
                os.unlink(name, dir_fd=dir_fd)
            finally:
                os.close(dir_fd)
            return

        guard = self._windows_parent_directory_guard(target)
        if guard is None:
            raise CapabilityDenied("file delete requires dir_fd support on this platform")
        try:
            self._require_existing_single_link_file(target)
            self._before_path_sink_checked("delete_file", target)
            self._target(path)
            self._require_existing_single_link_file(target)
            target.unlink()
        finally:
            guard.close()

    def _delete_directory_under_root(self, path: ResolvedPath, target: Path, *, recursive: bool) -> None:
        if self._supports_dir_fd_deletes():
            dir_fd, name = self._open_parent_dir_fd(target)
            try:
                self._before_path_sink_checked("delete_directory", target)
                self._require_directory_component_for_delete(dir_fd, name, target)
                if recursive:
                    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
                        raise CapabilityDenied("recursive directory delete requires symlink-safe rmtree support")
                    shutil.rmtree(name, dir_fd=dir_fd)
                else:
                    os.rmdir(name, dir_fd=dir_fd)
            finally:
                os.close(dir_fd)
            return

        guard = self._windows_parent_directory_guard(target)
        try:
            self._before_path_sink_checked("delete_directory", target)
            self._target(path)
            if recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        finally:
            if guard is not None:
                guard.close()

    def _supports_dir_fd_deletes(self) -> bool:
        return (
            os.open in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
            and os.unlink in os.supports_dir_fd
            and os.rmdir in os.supports_dir_fd
        )

    def _open_parent_dir_fd(self, target: Path) -> tuple[int, str]:
        parts = self._relative_parts(target)
        if not parts:
            raise CapabilityDenied("filesystem operation requires a path below the adapter root")
        dir_fd = self._open_root_dir_fd()
        try:
            for part in parts[:-1]:
                next_fd = self._open_dir_component(dir_fd, part)
                os.close(dir_fd)
                dir_fd = next_fd
            return dir_fd, parts[-1]
        except Exception:
            os.close(dir_fd)
            raise

    def _supports_dir_fd_directory_ops(self, *, require_list: bool = False) -> bool:
        supported = (
            os.open in os.supports_dir_fd
            and os.mkdir in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
        )
        if require_list:
            supported = supported and os.listdir in os.supports_fd
        return supported

    def _ensure_parent_dirs_under_root(self, target: Path) -> None:
        parts = self._relative_parts(target)
        if len(parts) <= 1:
            return
        if not self._supports_dir_fd_directory_ops():
            self._fallback_create_parent_dirs(target)
            return
        dir_fd = self._open_root_dir_fd()
        try:
            for part in parts[:-1]:
                next_fd = self._mkdir_or_open_dir_component(dir_fd, part, exist_ok=True)
                os.close(dir_fd)
                dir_fd = next_fd
        finally:
            os.close(dir_fd)

    def _make_directory_under_root(self, target: Path, *, parents: bool, exist_ok: bool) -> None:
        parts = self._relative_parts(target)
        if not parts:
            if exist_ok:
                return
            raise FileExistsError(os.fspath(target))
        if not self._supports_dir_fd_directory_ops():
            self._fallback_make_directory(target, parents=parents, exist_ok=exist_ok)
            return
        if parents:
            dir_fd = self._open_root_dir_fd()
            try:
                for index, part in enumerate(parts):
                    if index == len(parts) - 1:
                        self._mkdir_component(dir_fd, part, target, exist_ok=exist_ok)
                        next_fd = self._open_dir_component(dir_fd, part)
                        os.close(dir_fd)
                        dir_fd = next_fd
                    else:
                        next_fd = self._mkdir_or_open_dir_component(dir_fd, part, exist_ok=True)
                        os.close(dir_fd)
                        dir_fd = next_fd
            finally:
                os.close(dir_fd)
            return
        dir_fd, name = self._open_parent_dir_fd(target)
        try:
            self._mkdir_component(dir_fd, name, target, exist_ok=exist_ok)
        finally:
            os.close(dir_fd)

    def _mkdir_or_open_dir_component(self, dir_fd: int, name: str, *, exist_ok: bool) -> int:
        try:
            return self._open_dir_component(dir_fd, name)
        except FileNotFoundError:
            self._mkdir_component(dir_fd, name, Path(name), exist_ok=False)
            return self._open_dir_component(dir_fd, name)
        except NotADirectoryError:
            raise

    def _mkdir_component(self, dir_fd: int, name: str, target: Path, *, exist_ok: bool) -> None:
        try:
            os.mkdir(name, mode=0o777, dir_fd=dir_fd)
        except FileExistsError:
            if not exist_ok:
                raise
            opened_fd = self._open_dir_component(dir_fd, name)
            os.close(opened_fd)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise CapabilityDenied(f"filesystem path contains a symlink or non-directory component: {target}") from exc
            raise

    def _list_directory_under_root(self, target: Path, *, limit: int | None) -> list[DirectoryEntrySnapshot]:
        if not self._supports_dir_fd_directory_ops(require_list=True):
            return self._fallback_list_directory(target, limit=limit)
        dir_fd = self._open_directory_under_root(target)
        try:
            names = os.listdir(dir_fd)
            selected_names = heapq.nsmallest(limit, names) if limit is not None and limit > 0 else sorted(names)
            entries: list[DirectoryEntrySnapshot] = []
            for name in selected_names:
                stat_result = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                entries.append(self._directory_entry_from_stat(target, name, stat_result))
            return entries
        finally:
            os.close(dir_fd)

    def _open_directory_under_root(self, target: Path) -> int:
        parts = self._relative_parts(target)
        dir_fd = self._open_root_dir_fd()
        try:
            for part in parts:
                next_fd = self._open_dir_component(dir_fd, part)
                os.close(dir_fd)
                dir_fd = next_fd
            return dir_fd
        except Exception:
            os.close(dir_fd)
            raise

    def _require_file_component_for_delete(self, dir_fd: int, name: str, target: Path) -> None:
        try:
            stat_result = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise CapabilityDenied(f"filesystem path changed during delete: {target}") from exc
        if stat.S_ISLNK(stat_result.st_mode):
            raise CapabilityDenied(f"filesystem path contains a symlink or junction: {target}")
        if stat.S_ISREG(stat_result.st_mode) and stat_result.st_nlink > 1:
            raise CapabilityDenied(f"filesystem path is a hard link with multiple names: {target}")

    def _require_directory_component_for_delete(self, dir_fd: int, name: str, target: Path) -> None:
        try:
            stat_result = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise CapabilityDenied(f"filesystem path changed during delete: {target}") from exc
        if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISDIR(stat_result.st_mode):
            raise CapabilityDenied(f"filesystem path is not a directory below the adapter root: {target}")

    def _before_path_sink_checked(self, operation: str, target: Path) -> None:
        try:
            self._before_path_sink(operation, target)
        except OSError as exc:
            raise CapabilityDenied(
                f"filesystem path contains a symlink or junction, or changed before {operation}: {target}"
            ) from exc

    def _fallback_create_parent_dirs(self, target: Path) -> None:
        if target.parent == self.root:
            return
        if os.name != "nt":
            raise CapabilityDenied(
                "filesystem parent creation requires descriptor-bound directory operations"
            )
        parts = self._relative_parts(target)
        current = self.root
        guard = self._windows_directory_guard(current)
        try:
            for part in parts[:-1]:
                child = current / part
                try:
                    child.mkdir()
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise CapabilityDenied(
                        f"filesystem parent path could not be created safely: {child}"
                    ) from exc
                child_guard = self._windows_directory_guard(child)
                guard.close()
                guard = child_guard
                current = child
        finally:
            guard.close()

    def _fallback_make_directory(self, target: Path, *, parents: bool, exist_ok: bool) -> None:
        if parents:
            self._fallback_create_parent_dirs(target)
        guard = self._windows_parent_directory_guard(target)
        try:
            self._target(ResolvedPath(display=os.fspath(target), relative=target.relative_to(self.root).as_posix()))
            target.mkdir(parents=False, exist_ok=exist_ok)
        finally:
            if guard is not None:
                guard.close()

    def _fallback_list_directory(self, target: Path, *, limit: int | None) -> list[DirectoryEntrySnapshot]:
        guard = _WindowsDirectoryGuard.open(target) if os.name == "nt" else None
        try:
            if guard is None:
                raise CapabilityDenied("directory listing requires descriptor-bound directory operations")
            opened = self._windows_final_path_from_handle(guard.handle)
            requested = Path(os.path.abspath(os.fspath(target)))
            if os.path.normcase(os.fspath(opened)) != os.path.normcase(os.fspath(requested)):
                raise CapabilityDenied(f"filesystem directory path changed during validation: {target}")
            if self.root not in opened.parents and opened != self.root:
                raise CapabilityDenied(f"filesystem directory path escapes adapter root: {target}")
            if limit is not None and limit > 0:
                children = heapq.nsmallest(limit, target.iterdir(), key=lambda item: item.name)
            else:
                children = sorted(target.iterdir(), key=lambda item: item.name)
            return [self._directory_entry(child) for child in children]
        finally:
            if guard is not None:
                guard.close()

    def _windows_parent_directory_guard(self, target: Path) -> _WindowsDirectoryGuard | None:
        if os.name != "nt":
            return None
        return self._windows_directory_guard(target.parent)

    def _windows_directory_guard(self, directory: Path) -> _WindowsDirectoryGuard:
        try:
            if self._is_reparse_path(directory) or not directory.is_dir():
                raise CapabilityDenied(
                    f"filesystem path contains a symlink, junction, or non-directory: {directory}"
                )
            guard = _WindowsDirectoryGuard.open(directory)
            opened = self._windows_final_path_from_handle(guard.handle)
            requested = Path(os.path.abspath(os.fspath(directory)))
            if os.path.normcase(os.fspath(opened)) != os.path.normcase(os.fspath(requested)):
                guard.close()
                raise CapabilityDenied(
                    f"filesystem directory path changed during validation: {directory}"
                )
            if self.root not in opened.parents and opened != self.root:
                guard.close()
                raise CapabilityDenied(
                    f"filesystem directory path escapes adapter root: {directory}"
                )
            return guard
        except OSError as exc:
            raise CapabilityDenied(
                f"filesystem directory path could not be guarded: {directory}"
            ) from exc

    def _open_under_root_fallback(self, target: Path, flags: int, mode: int) -> int:
        guard = self._windows_parent_directory_guard(target)
        try:
            self._require_existing_single_link_file(
                target,
                allow_missing=bool(flags & os.O_CREAT),
            )
            try:
                self._before_fallback_open(target, flags)
            except OSError as exc:
                raise CapabilityDenied(
                    f"filesystem opened path changed while its parent was guarded: {target}"
                ) from exc
            fd = os.open(target, flags, mode)
            try:
                self._validate_open_target_matches_request(fd, target)
            except Exception:
                os.close(fd)
                raise
            return fd
        finally:
            if guard is not None:
                guard.close()

    def _before_fallback_open(self, target: Path, flags: int) -> None:
        return None

    def _validate_open_target_matches_request(self, fd: int, target: Path) -> None:
        if os.name != "nt":
            return
        opened = self._windows_final_path_from_fd(fd)
        requested = Path(os.path.abspath(os.fspath(target)))
        if os.path.normcase(os.fspath(opened)) != os.path.normcase(os.fspath(requested)):
            raise CapabilityDenied(f"filesystem opened path changed during validation: {target}")
        if self.root not in opened.parents and opened != self.root:
            raise CapabilityDenied(f"filesystem opened path escapes adapter root: {target}")

    def _windows_final_path_from_fd(self, fd: int) -> Path:
        if os.name != "nt":
            raise OSError("Windows final path validation is only available on Windows")
        handle = msvcrt.get_osfhandle(fd)
        return self._windows_final_path_from_handle(int(handle))

    def _windows_final_path_from_handle(self, handle: int) -> Path:
        if os.name != "nt":
            raise OSError("Windows final path validation is only available on Windows")
        size = 512
        while True:
            buffer = ctypes.create_unicode_buffer(size)
            result = _kernel32.GetFinalPathNameByHandleW(
                ctypes.c_void_p(handle),
                buffer,
                size,
                _FILE_NAME_NORMALIZED | _VOLUME_NAME_DOS,
            )
            if result == 0:
                raise ctypes.WinError(ctypes.get_last_error())
            if result < size:
                value = buffer.value
                if value.startswith("\\\\?\\UNC\\"):
                    value = "\\\\" + value[8:]
                elif value.startswith("\\\\?\\"):
                    value = value[4:]
                return Path(value)
            size = int(result) + 1

    def _relative_parts(self, target: Path) -> tuple[str, ...]:
        try:
            parts = target.relative_to(self.root).parts
        except ValueError as exc:
            raise CapabilityDenied(f"path escapes filesystem adapter root: {target}") from exc
        if any(part in {"", ".", ".."} for part in parts):
            raise CapabilityDenied(f"invalid filesystem path component: {target}")
        return tuple(parts)

    def _open_root_dir_fd(self) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.open(self.root, flags)

    def _open_dir_component(self, dir_fd: int, name: str) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            return os.open(name, flags, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise CapabilityDenied(f"filesystem path contains a symlink or non-directory component: {name}") from exc
            raise

    def _open_file_component(self, dir_fd: int, name: str, flags: int, mode: int) -> int:
        try:
            return os.open(name, flags | getattr(os, "O_NOFOLLOW", 0), mode, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise CapabilityDenied(f"filesystem path contains a symlink or non-file component: {name}") from exc
            raise

    def _validate_open_regular_file(self, fd: int, target: Path) -> None:
        stat_result = os.fstat(fd)
        if not stat.S_ISREG(stat_result.st_mode):
            raise CapabilityDenied(f"filesystem path is not a regular file: {target}")
        if stat_result.st_nlink > 1:
            raise CapabilityDenied(f"filesystem path is a hard link with multiple names: {target}")

    def _require_existing_single_link_file(self, target: Path, *, allow_missing: bool = False) -> None:
        try:
            stat_result = target.lstat()
        except FileNotFoundError:
            if allow_missing:
                return
            raise
        if stat.S_ISLNK(stat_result.st_mode):
            raise CapabilityDenied(f"filesystem path contains a symlink or junction: {target}")
        if stat.S_ISREG(stat_result.st_mode) and stat_result.st_nlink > 1:
            raise CapabilityDenied(f"filesystem path is a hard link with multiple names: {target}")

    def _directory_entry(self, target: Path) -> DirectoryEntrySnapshot:
        stat_result = target.lstat()
        return self._directory_entry_from_stat(target.parent, target.name, stat_result)

    def _directory_entry_from_stat(self, parent: Path, name: str, stat_result: os.stat_result) -> DirectoryEntrySnapshot:
        mode = stat_result.st_mode
        kind = (
            "symlink"
            if stat.S_ISLNK(mode)
            else "file"
            if stat.S_ISREG(mode)
            else "directory"
            if stat.S_ISDIR(mode)
            else "other"
        )
        target = parent / name
        return DirectoryEntrySnapshot(
            name=name,
            path=target.relative_to(self.root).as_posix(),
            kind=kind,
            size_bytes=stat_result.st_size if kind == "file" else None,
            modified_at=datetime.fromtimestamp(stat_result.st_mtime, timezone.utc).isoformat(),
        )


class LocalClockProvider:
    """Host clock implementation used by the default local substrate."""

    def now(self, timezone_: tzinfo) -> datetime:
        return datetime.now(timezone_)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        # Async sleep lets one sleeping AgentProcess yield to other runnable
        # processes in the cooperative scheduler.
        await asyncio.sleep(seconds)

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation == "now":
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={"timezone": context.get("timezone")},
            )
        if operation == "sleep":
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=False,
                metadata={"requested_seconds": context.get("requested_seconds")},
            )
        raise ValueError(f"unsupported clock external effect operation: {operation}")


class LocalShellProvider:
    """Subprocess-backed shell provider scoped to a configured working directory."""

    supports_subprocess_limits = os.name != "nt"
    supports_executable_snapshots = True

    def __init__(self, cwd: str | Path, *, git_config: Any | None = None):
        self.cwd = Path(cwd).resolve()
        self._git_config = git_config
        self._git_read_guard: LocalGitProvider | None = None

    def resolve_argv(self, argv: list[str], *, cwd: str | None = None) -> list[str]:
        return self._resolve_argv0(argv, self._resolve_cwd(cwd))

    def run(
        self,
        argv: list[str],
        *,
        timeout: float = _TOOL_DEFAULTS.shell_timeout_s,
        cwd: str | None = None,
        limits: SubprocessLimits | None = None,
        stdout_limit_chars: int | None = None,
        stderr_limit_chars: int | None = None,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> CommandResult:
        selected_cwd = self._prepare_run_cwd(argv, cwd=cwd, limits=limits)
        stdout_limit, stderr_limit = self._selected_output_limits(
            stdout_limit_chars,
            stderr_limit_chars,
        )
        requested_argv0 = argv[0] if argv else None
        checked_argv = self._resolve_argv0(argv, selected_cwd)
        popen_executable: str | None = None
        if executable_snapshot is not None:
            executable_snapshot.verify()
            if executable_snapshot.source_path != Path(checked_argv[0]).resolve(
                strict=False
            ):
                raise ValidationError(
                    "shell executable snapshot does not match resolved argv[0]"
                )
            if os.name == "nt":
                checked_argv = [
                    str(executable_snapshot.executable_path),
                    *checked_argv[1:],
                ]
            else:
                # The executable parameter selects the pinned bytes, while
                # argv[0] retains the caller's invocation spelling. Launchers
                # such as .venv/bin/python use that spelling to locate
                # pyvenv.cfg; replacing it with the symlink-resolved base
                # interpreter silently drops the virtual environment.
                popen_executable = str(executable_snapshot.executable_path)
                if requested_argv0 is not None:
                    checked_argv = [requested_argv0, *checked_argv[1:]]
        started_at = time.monotonic()
        with tempfile.TemporaryFile("w+b") as stdout_file, tempfile.TemporaryFile("w+b") as stderr_file:
            job = self._windows_job_for_run(limits)
            try:
                proc = subprocess.Popen(
                    checked_argv,
                    executable=popen_executable,
                    cwd=selected_cwd,
                    env=self._safe_env(),
                    shell=False,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    **self._process_group_kwargs(),
                )
            except Exception:
                if job is not None:
                    job.close()
                raise
            try:
                if job is not None:
                    job.assign(proc)
            except OSError as exc:
                job.close()
                if limits is not None:
                    with contextlib.suppress(Exception):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=1.0)
                    raise ValidationError("shell provider could not attach Windows Job Object for budgeted execution") from exc
                job = None
            require_complete_metrics = bool(
                limits is not None
                and (limits.cpu_seconds is not None or limits.memory_bytes is not None)
            )
            ps_proc: psutil.Process | None = None
            try:
                ps_proc = psutil.Process(proc.pid)
            except (psutil.Error, OSError) as exc:
                if require_complete_metrics:
                    self._kill_process_tree(None, proc)
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=1.0)
                    if job is not None:
                        job.close()
                    raise ValidationError(
                        "shell provider cannot enforce CPU/memory SubprocessLimits because process metrics are unavailable"
                    ) from exc
            except Exception:
                if job is not None:
                    job.close()
                self._kill_process_tree(None, proc)
                with contextlib.suppress(Exception):
                    proc.wait(timeout=1.0)
                raise
            peak_memory = 0
            cpu_seconds = 0.0
            limit_kind: str | None = None
            timed_out = False
            try:
                while True:
                    wall_seconds = time.monotonic() - started_at
                    if ps_proc is not None:
                        cpu_seconds, peak_memory = self._sample_process_tree(
                            ps_proc,
                            peak_memory,
                            require_complete=require_complete_metrics,
                        )
                    limit_kind = self._limit_kind(
                        wall_seconds=wall_seconds,
                        cpu_seconds=cpu_seconds,
                        peak_memory=peak_memory,
                        limits=limits,
                    )
                    if limit_kind is None:
                        limit_kind = self._output_limit_kind(stdout_file, stderr_file, stdout_limit, stderr_limit)
                    if limit_kind is not None:
                        self._kill_process_tree(ps_proc, proc)
                        try:
                            proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
                        break
                    if timeout is not None and wall_seconds > timeout:
                        timed_out = True
                        self._kill_process_tree(ps_proc, proc)
                        try:
                            proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
                        break
                    if proc.poll() is not None:
                        self._terminate_process_group(proc)
                        break
                    time.sleep(0.02)
            finally:
                if proc.poll() is None:
                    self._kill_process_tree(ps_proc, proc)
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                if job is not None:
                    job.close()
            stdout, stdout_truncated = self._read_limited_output(stdout_file, stdout_limit)
            stderr, stderr_truncated = self._read_limited_output(stderr_file, stderr_limit)
            wall_seconds = time.monotonic() - started_at
            if ps_proc is not None:
                final_cpu_seconds, peak_memory = self._sample_process_tree(
                    ps_proc,
                    peak_memory,
                    require_complete=require_complete_metrics,
                )
                cpu_seconds = max(cpu_seconds, final_cpu_seconds)
            metrics = CommandMetrics(
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                peak_memory_bytes=peak_memory,
                killed=timed_out or limit_kind is not None,
                limit_kind="subprocess_timeout" if timed_out else limit_kind,
            )
            result = CommandResult(
                argv=list(argv),
                returncode=proc.returncode if proc.returncode is not None else -9,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                metrics=metrics,
            )
            if timed_out:
                raise SubprocessTimeoutExpired(
                    f"subprocess timed out after {timeout}s",
                    metrics=metrics,
                    result=result,
                )
            if limit_kind is not None:
                raise SubprocessLimitExceeded(
                    f"subprocess exceeded {limit_kind}",
                    metrics=metrics,
                    result=result,
                )
            return result

    def _prepare_run_cwd(
        self,
        argv: list[str],
        *,
        cwd: str | None,
        limits: SubprocessLimits | None,
    ) -> Path:
        if limits is not None and not self.supports_subprocess_limits:
            raise ValidationError(
                "shell provider cannot enforce SubprocessLimits on this platform"
            )
        selected_cwd = self._resolve_cwd(cwd)
        self._validate_legacy_git_dispatch(argv, selected_cwd)
        return selected_cwd

    def _validate_legacy_git_dispatch(self, argv: list[str], selected_cwd: Path) -> None:
        git_operation = trusted_git_read_operation(argv, hardened_only=True)
        if git_operation is None:
            return
        if selected_cwd != self.cwd:
            raise ValidationError(
                "legacy raw Git reads are fixed to the Runtime workspace root"
            )
        if self._git_read_guard is None:
            self._git_read_guard = LocalGitProvider(
                self.cwd,
                config=self._git_config,
            )
        self._git_read_guard.validate_read_only_operation(git_operation)

    @staticmethod
    def _selected_output_limits(
        stdout_limit_chars: int | None,
        stderr_limit_chars: int | None,
    ) -> tuple[int, int]:
        stdout_limit = (
            _SHELL_DEFAULTS.stdout_hard_limit_chars
            if stdout_limit_chars is None
            else max(0, int(stdout_limit_chars))
        )
        stderr_limit = (
            _SHELL_DEFAULTS.stderr_hard_limit_chars
            if stderr_limit_chars is None
            else max(0, int(stderr_limit_chars))
        )
        return stdout_limit, stderr_limit

    def executable_snapshot_required(
        self,
        executable: str,
        *,
        requested_argv0: str | None = None,
        cwd: str | None = None,
    ) -> bool:
        selected_cwd = self._resolve_cwd(cwd)

        def is_workspace_path(path: Path) -> bool:
            return path == self.cwd or self.cwd in path.parents

        if is_workspace_path(Path(executable).resolve(strict=False)):
            return True
        if requested_argv0 and self._argv0_has_path(requested_argv0):
            raw = Path(requested_argv0).expanduser()
            candidate = raw if raw.is_absolute() else selected_cwd / raw
            lexical = Path(os.path.abspath(candidate))
            return is_workspace_path(lexical)
        return False

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation != "run":
            raise ValueError(f"unsupported shell external effect operation: {operation}")
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
            state_mutation=True,
            information_flow=True,
            metadata={"argv": context.get("argv"), "cwd": context.get("cwd")},
        )

    def _resolve_cwd(self, cwd: str | None) -> Path:
        if cwd is None or cwd in {"", "."}:
            return self.cwd
        raw = Path(cwd)
        target = raw.resolve() if raw.is_absolute() else (self.cwd / raw).resolve()
        if self.cwd not in target.parents and target != self.cwd:
            raise CapabilityDenied(f"shell working directory escapes workspace root: {cwd}")
        return target

    def _safe_env(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key.upper() in _SAFE_SHELL_ENV_KEYS}
        env["PATH"] = self._safe_path()
        env["HOME"] = str(self.cwd)
        env["USERPROFILE"] = str(self.cwd)
        env["GIT_NO_LAZY_FETCH"] = "1"
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_PAGER"] = "cat"
        env["GIT_NO_REPLACE_OBJECTS"] = "1"
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_ATTR_NOSYSTEM"] = "1"
        if os.name == "nt":
            env["GCM_INTERACTIVE"] = "Never"
        return env

    def _resolve_argv0(self, argv: list[str], selected_cwd: Path) -> list[str]:
        if not argv:
            return argv
        if self._argv0_has_path(argv[0]):
            raw = Path(argv[0])
            target = raw if raw.is_absolute() else selected_cwd / raw
            return [str(target.resolve(strict=False)), *argv[1:]]
        resolved = shutil.which(argv[0], path=self._safe_path())
        if resolved is None:
            resolved = resolve_runtime_python_alias(
                argv[0],
                workspace_root=self.cwd,
            )
        if resolved is None:
            raise FileNotFoundError(f"shell executable not found on safe PATH: {argv[0]}")
        target = Path(resolved).resolve()
        if self.cwd in target.parents or target == self.cwd or selected_cwd in target.parents or target == selected_cwd:
            raise CapabilityDenied(f"bare shell executable resolves inside workspace: {argv[0]}")
        return [str(target), *argv[1:]]

    def _safe_path(self) -> str:
        entries: list[str] = []
        for item in os.environ.get("PATH", "").split(os.pathsep):
            if not item:
                continue
            raw = Path(item).expanduser()
            if not raw.is_absolute():
                continue
            resolved = raw.resolve(strict=False)
            if self.cwd in resolved.parents or resolved == self.cwd:
                continue
            entries.append(str(resolved))
        return os.pathsep.join(entries)

    def _argv0_has_path(self, value: str) -> bool:
        return "/" in value or "\\" in value or Path(value).is_absolute()

    def _output_limit_kind(
        self,
        stdout_file: Any,
        stderr_file: Any,
        stdout_limit: int,
        stderr_limit: int,
    ) -> str | None:
        if os.fstat(stdout_file.fileno()).st_size > stdout_limit:
            return "subprocess_stdout_bytes"
        if os.fstat(stderr_file.fileno()).st_size > stderr_limit:
            return "subprocess_stderr_bytes"
        return None

    def _read_limited_output(self, handle: Any, limit: int) -> tuple[str, bool]:
        handle.flush()
        handle.seek(0)
        data = handle.read(limit + 1)
        truncated = len(data) > limit
        if truncated:
            data = data[:limit]
        return data.decode("utf-8", errors="replace"), truncated

    def _limit_kind(
        self,
        *,
        wall_seconds: float,
        cpu_seconds: float,
        peak_memory: int,
        limits: SubprocessLimits | None,
    ) -> str | None:
        if limits is None:
            return None
        if limits.wall_seconds is not None and wall_seconds > limits.wall_seconds:
            return "subprocess_wall_seconds"
        if limits.cpu_seconds is not None and cpu_seconds > limits.cpu_seconds:
            return "subprocess_cpu_seconds"
        if limits.memory_bytes is not None and peak_memory > limits.memory_bytes:
            return "subprocess_memory_bytes"
        return None

    def _sample_process_tree(
        self,
        proc: psutil.Process,
        peak_memory: int,
        *,
        require_complete: bool,
    ) -> tuple[float, int]:
        cpu_seconds = 0.0
        memory_bytes = 0
        processes = [proc]
        if require_complete:
            try:
                processes.extend(proc.children(recursive=True))
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
            except (psutil.Error, OSError) as exc:
                raise ValidationError(
                    "shell provider cannot enforce CPU/memory SubprocessLimits because complete process metrics are unavailable"
                ) from exc
        for item in processes:
            try:
                times = item.cpu_times()
                cpu_seconds += float(times.user) + float(times.system)
                memory_bytes += int(item.memory_info().rss)
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
            except (psutil.Error, OSError) as exc:
                if require_complete:
                    raise ValidationError(
                        "shell provider cannot enforce CPU/memory SubprocessLimits because complete process metrics are unavailable"
                    ) from exc
        return cpu_seconds, max(peak_memory, memory_bytes)

    def _process_group_kwargs(self) -> dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    def _windows_job_for_run(self, limits: SubprocessLimits | None) -> WindowsJobObject | None:
        if os.name != "nt":
            return None
        try:
            return WindowsJobObject.create()
        except OSError as exc:
            if limits is not None:
                raise ValidationError("shell provider could not create Windows Job Object for budgeted execution") from exc
            return None

    def _kill_process_tree(self, ps_proc: psutil.Process | None, proc: subprocess.Popen[str]) -> None:
        # The direct child may exit after spawning background work, at which
        # point psutil no longer sees those processes as descendants. A process
        # group gives the provider one cleanup handle for the whole shell run.
        self._terminate_process_group(proc)
        processes: list[psutil.Process] = []
        if ps_proc is not None:
            try:
                processes.extend(ps_proc.children(recursive=True))
            except (psutil.Error, OSError):
                pass
            processes.append(ps_proc)
        for item in processes:
            try:
                item.terminate()
            except (psutil.Error, OSError):
                continue
        try:
            alive = psutil.wait_procs(processes, timeout=1.0)[1] if processes else []
        except (psutil.Error, OSError):
            alive = processes
        for item in alive:
            try:
                item.kill()
            except (psutil.Error, OSError):
                continue
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def _terminate_process_group(self, proc: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            return
        time.sleep(0.05)
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)


class LocalHumanProvider:
    """Terminal-backed human I/O provider for the local substrate."""

    def __init__(
        self,
        *,
        output_sink: Callable[[str], None] | None = None,
        input_reader: Callable[[str], str] | None = None,
    ) -> None:
        self.output_sink = output_sink or (lambda message: print(message, flush=True))
        self.input_reader = input_reader or input

    def write(self, message: str) -> None:
        self.output_sink(message)

    def read(self, prompt: str) -> str:
        return self.input_reader(prompt)

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation == "write":
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={"channel": context.get("channel"), "chars": context.get("chars")},
            )
        if operation == "read":
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={"prompt": context.get("prompt")},
            )
        raise ValueError(f"unsupported human external effect operation: {operation}")


class HttpJsonRpcProvider:
    """HTTP JSON-RPC client provider used by the default substrate."""

    class _NoRedirectHandler(urlrequest.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            return None

    def call(
        self,
        endpoint: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
        request_body: bytes,
        *,
        timeout_s: float,
        max_response_bytes: int,
        resolved_addresses: tuple[str, ...] | None = None,
    ) -> JsonRpcTransportResult:
        if resolved_addresses:
            return self._call_pinned(
                endpoint,
                request_body,
                timeout_s=timeout_s,
                max_response_bytes=max_response_bytes,
                resolved_addresses=resolved_addresses,
            )
        started = time.monotonic()
        request = urlrequest.Request(
            endpoint.url,
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **self._resolved_headers(endpoint),
            },
            method="POST",
        )
        opener = urlrequest.build_opener(self._NoRedirectHandler)
        try:
            with opener.open(request, timeout=timeout_s) as response:
                body = response.read(max_response_bytes + 1)
                too_large = len(body) > max_response_bytes
                if too_large:
                    body = body[:max_response_bytes]
                return JsonRpcTransportResult(
                    status_code=int(response.status),
                    body=body,
                    elapsed_s=time.monotonic() - started,
                    response_bytes=len(body),
                    too_large=too_large,
                )
        except urlerror.HTTPError as exc:
            try:
                body = exc.read(max_response_bytes + 1)
                too_large = len(body) > max_response_bytes
                if too_large:
                    body = body[:max_response_bytes]
                return JsonRpcTransportResult(
                    status_code=int(exc.code),
                    body=body,
                    elapsed_s=time.monotonic() - started,
                    response_bytes=len(body),
                    too_large=too_large,
                    error=str(exc),
                )
            finally:
                exc.close()
        except Exception as exc:
            return JsonRpcTransportResult(
                status_code=None,
                body=b"",
                elapsed_s=time.monotonic() - started,
                response_bytes=0,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _call_pinned(
        self,
        endpoint: JsonRpcEndpointSpec,
        request_body: bytes,
        *,
        timeout_s: float,
        max_response_bytes: int,
        resolved_addresses: tuple[str, ...],
    ) -> JsonRpcTransportResult:
        # Keep DNS policy and the actual socket target coupled. urlopen()
        # re-resolves hostnames internally, which can reopen DNS rebinding
        # after the primitive has already accepted a safe address set.
        started = time.monotonic()
        parsed = urlsplit(endpoint.url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        request_target = parsed.path or "/"
        if parsed.query:
            request_target = f"{request_target}?{parsed.query}"
        headers = {
            "Host": self._host_header(host, port, parsed.scheme),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(len(request_body)),
            "Connection": "close",
            **self._resolved_headers(endpoint),
        }
        request_head = self._http_request_head("POST", request_target, headers)
        last_error: str | None = None
        deadline = started + timeout_s
        for address in resolved_addresses:
            remaining_timeout = deadline - time.monotonic()
            if remaining_timeout <= 0:
                last_error = "TimeoutError: JSON-RPC pinned request timed out"
                break
            try:
                with self._pinned_socket(
                    address,
                    port,
                    host=host,
                    scheme=parsed.scheme,
                    timeout_s=remaining_timeout,
                ) as sock:
                    sock.sendall(request_head + request_body)
                    response = http.client.HTTPResponse(sock)
                    response.begin()
                    body = response.read(max_response_bytes + 1)
                    too_large = len(body) > max_response_bytes
                    if too_large:
                        body = body[:max_response_bytes]
                    return JsonRpcTransportResult(
                        status_code=int(response.status),
                        body=body,
                        elapsed_s=time.monotonic() - started,
                        response_bytes=len(body),
                        too_large=too_large,
                    )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
        return JsonRpcTransportResult(
            status_code=None,
            body=b"",
            elapsed_s=time.monotonic() - started,
            response_bytes=0,
            error=last_error or "no pinned JSON-RPC addresses were available",
        )

    def _pinned_socket(
        self,
        address: str,
        port: int,
        *,
        host: str,
        scheme: str,
        timeout_s: float,
    ) -> socket.socket:
        raw = socket.create_connection((address, port), timeout=timeout_s)
        raw.settimeout(timeout_s)
        try:
            if scheme == "https":
                context = ssl.create_default_context()
                return context.wrap_socket(raw, server_hostname=host)
            return raw
        except Exception:
            raw.close()
            raise

    def _host_header(self, host: str, port: int, scheme: str) -> str:
        default_port = 443 if scheme == "https" else 80
        if port == default_port:
            return host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"{host}:{port}"

    def _http_request_head(self, method: str, target: str, headers: dict[str, str]) -> bytes:
        lines = [f"{method} {target} HTTP/1.1"]
        lines.extend(f"{name}: {value}" for name, value in headers.items())
        lines.append("")
        lines.append("")
        return "\r\n".join(lines).encode("iso-8859-1")

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation != "call":
            raise ValueError(f"unsupported JSON-RPC external effect operation: {operation}")
        method = context.get("method") if isinstance(context.get("method"), dict) else {}
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass(str(method.get("rollback_class"))),
            rollback_status=ExternalEffectRollbackStatus(str(method.get("rollback_status"))),
            state_mutation=bool(method.get("state_mutation")),
            information_flow=bool(method.get("information_flow")),
            metadata={
                "endpoint_id": context.get("endpoint_id"),
                "method_id": context.get("method_id"),
                "rpc_method": context.get("rpc_method"),
                "status": result.get("status") if isinstance(result, dict) else None,
            },
        )

    def _resolved_headers(self, endpoint: JsonRpcEndpointSpec) -> dict[str, str]:
        headers: dict[str, str] = {}
        for name, spec in endpoint.headers.items():
            value = os.environ.get(spec.env)
            if value is None:
                raise RuntimeError(f"missing environment variable for JSON-RPC header {name}: {spec.env}")
            headers[name] = f"{spec.prefix}{value}{spec.suffix}"
        return headers


class SdkMcpProvider:
    """MCP client provider backed by the optional official Python SDK."""

    supports_executable_snapshots = True

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()

    def list_tools(
        self,
        server: McpServerSpec,
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> McpToolListResult:
        with self._stdio_dispatch_snapshot(
            server,
            executable_snapshot,
        ) as selected_snapshot:
            try:
                return _run_mcp_async(
                    self._alist_tools(
                        server,
                        timeout_s=timeout_s,
                        max_response_bytes=max_response_bytes,
                        executable_snapshot=selected_snapshot,
                    )
                )
            except BaseExceptionGroup as exc:
                self._raise_mcp_transport_limit_error(exc)
                raise

    def call_tool(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> McpProviderCallResult:
        with self._stdio_dispatch_snapshot(
            server,
            executable_snapshot,
        ) as selected_snapshot:
            try:
                return _run_mcp_async(
                    self._acall_tool(
                        server,
                        tool,
                        arguments,
                        timeout_s=timeout_s,
                        max_response_bytes=max_response_bytes,
                        executable_snapshot=selected_snapshot,
                    )
                )
            except BaseExceptionGroup as exc:
                self._raise_mcp_transport_limit_error(exc)
                raise

    def validate_and_call(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> McpProviderCallResult:
        """Validate and invoke through one MCP session and one wall-clock deadline."""

        with self._stdio_dispatch_snapshot(server, executable_snapshot) as selected_snapshot:
            try:
                return _run_mcp_async(
                    asyncio.wait_for(
                        self._avalidate_and_call(
                            server,
                            tool,
                            arguments,
                            timeout_s=timeout_s,
                            max_response_bytes=max_response_bytes,
                            executable_snapshot=selected_snapshot,
                        ),
                        timeout=timeout_s,
                    )
                )
            except BaseExceptionGroup as exc:
                self._raise_mcp_transport_limit_error(exc)
                raise

    def resolve_stdio_executable(self, server: McpServerSpec) -> str:
        """Resolve the exact stdio executable used by the local MCP transport."""

        if server.transport != "stdio" or server.stdio is None:
            raise ValidationError("MCP stdio executable resolution requires stdio configuration")
        candidate = self._stdio_command_candidate(server)
        resolved_candidate = candidate.resolve(strict=True)
        if not resolved_candidate.is_file():
            raise ValidationError(
                f"MCP stdio executable is not a regular file: {resolved_candidate}"
            )
        return str(resolved_candidate)

    def _stdio_command_candidate(self, server: McpServerSpec) -> Path:
        if server.transport != "stdio" or server.stdio is None:
            raise ValidationError("MCP stdio executable resolution requires stdio configuration")
        command = server.stdio.command
        selected_cwd = Path(self._resolved_stdio_cwd(server))
        raw = Path(command).expanduser()
        if raw.is_absolute() or "/" in command or "\\" in command:
            candidate = raw if raw.is_absolute() else selected_cwd / raw
        else:
            child_env = self._resolved_stdio_env(server)
            resolved = shutil.which(command, path=child_env.get("PATH", os.defpath))
            if resolved is None:
                raise FileNotFoundError(f"MCP stdio executable not found: {command}")
            candidate = Path(resolved)
        return Path(os.path.abspath(candidate))

    def executable_snapshot_required(
        self,
        server: McpServerSpec,
        resolved_executable: str,
    ) -> bool:
        if server.transport != "stdio" or server.stdio is None:
            return False

        def is_workspace_path(path: Path) -> bool:
            return path == self.workspace_root or self.workspace_root in path.parents

        if is_workspace_path(Path(resolved_executable).resolve(strict=False)):
            return True
        candidate = self._stdio_command_candidate(server)
        return is_workspace_path(candidate)

    @contextlib.contextmanager
    def _stdio_dispatch_snapshot(
        self,
        server: McpServerSpec,
        executable_snapshot: ExecutableSnapshot | None,
    ) -> Iterator[ExecutableSnapshot | None]:
        if executable_snapshot is not None:
            executable_snapshot.verify()
            yield executable_snapshot
            return
        if server.transport != "stdio" or server.stdio is None:
            yield None
            return
        resolved = self.resolve_stdio_executable(server)
        if not self.executable_snapshot_required(server, resolved):
            yield None
            return
        with snapshot_executable(resolved) as owned_snapshot:
            yield owned_snapshot

    @staticmethod
    def _raise_mcp_transport_limit_error(error: BaseException) -> None:
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            message = str(current)
            if message.startswith(
                (
                    "MCP stdio frame exceeded max_response_bytes=",
                    "MCP HTTP response exceeded max_response_bytes=",
                    "MCP HTTP SSE frame exceeded max_response_bytes=",
                    "MCP HTTP response uses unsupported Content-Encoding=",
                )
            ):
                raise RuntimeError(message) from error
            if isinstance(current, BaseExceptionGroup):
                pending.extend(current.exceptions)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)

    async def _alist_tools(
        self,
        server: McpServerSpec,
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> McpToolListResult:
        started = time.monotonic()
        async with self._session(
            server,
            timeout_s=timeout_s,
            max_response_bytes=max_response_bytes,
            executable_snapshot=executable_snapshot,
        ) as session:
            result = await asyncio.wait_for(session.list_tools(), timeout=timeout_s)
        tools = [
            McpProviderTool(
                name=str(getattr(item, "name", "")),
                description=getattr(item, "description", None),
                input_schema=dict(getattr(item, "inputSchema", None) or getattr(item, "input_schema", None) or {}),
                metadata=_mcp_metadata(item),
            )
            for item in list(getattr(result, "tools", []) or [])
        ]
        encoded = dumps([to_jsonable(tool) for tool in tools]).encode("utf-8")
        if len(encoded) > max_response_bytes:
            raise RuntimeError(f"MCP tools/list response exceeded max_response_bytes={max_response_bytes}")
        return McpToolListResult(
            server_id=server.server_id,
            tools=tools,
            response_bytes=len(encoded),
            duration_s=time.monotonic() - started,
        )

    async def _acall_tool(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> McpProviderCallResult:
        started = time.monotonic()
        async with self._session(
            server,
            timeout_s=timeout_s,
            max_response_bytes=max_response_bytes,
            executable_snapshot=executable_snapshot,
        ) as session:
            result = await asyncio.wait_for(session.call_tool(tool.mcp_name, arguments), timeout=timeout_s)
        content = _jsonable_mcp_value(getattr(result, "content", None))
        structured = _jsonable_mcp_value(
            getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
        )
        payload = {"content": _bounded_mcp_content(content), "structured_content": structured}
        encoded = dumps(payload).encode("utf-8")
        too_large = len(encoded) > max_response_bytes
        if too_large:
            payload = {"content": _mcp_oversize_observation(encoded), "structured_content": None}
        return McpProviderCallResult(
            content=payload["content"],
            structured_content=payload["structured_content"],
            is_error=bool(getattr(result, "isError", False) or getattr(result, "is_error", False)),
            response_bytes=min(len(encoded), max_response_bytes),
            duration_s=time.monotonic() - started,
            too_large=too_large,
        )

    async def _avalidate_and_call(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> McpProviderCallResult:
        started = time.monotonic()
        list_request_bytes = len(
            dumps({"method": "tools/list", "server_id": server.server_id}).encode("utf-8")
        )
        call_request_bytes = len(
            dumps({"name": tool.mcp_name, "arguments": arguments}).encode("utf-8")
        )
        async with self._session(
            server,
            timeout_s=timeout_s,
            max_response_bytes=max_response_bytes,
            executable_snapshot=executable_snapshot,
        ) as session:
            live_result = await session.list_tools()
            live_tools = [
                McpProviderTool(
                    name=str(getattr(item, "name", "")),
                    description=getattr(item, "description", None),
                    input_schema=dict(
                        getattr(item, "inputSchema", None)
                        or getattr(item, "input_schema", None)
                        or {}
                    ),
                    metadata=_mcp_metadata(item),
                )
                for item in list(getattr(live_result, "tools", []) or [])
            ]
            list_encoded = dumps([to_jsonable(item) for item in live_tools]).encode("utf-8")
            if len(list_encoded) > max_response_bytes:
                return McpProviderCallResult(
                    error="MCP tools/list response exceeded limit",
                    error_type="ResponseTooLarge",
                    correlation_id=new_id("corr"),
                    duration_s=time.monotonic() - started,
                    list_request_bytes=list_request_bytes,
                    list_response_bytes=max_response_bytes,
                    call_request_bytes=call_request_bytes,
                    call_started=False,
                )
            live = next((item for item in live_tools if item.name == tool.mcp_name), None)
            if live is None or (
                tool.input_schema
                and live.input_schema
                and live.input_schema != tool.input_schema
            ):
                return McpProviderCallResult(
                    error="MCP live tool validation failed",
                    error_type="LiveToolValidationError",
                    correlation_id=new_id("corr"),
                    duration_s=time.monotonic() - started,
                    list_request_bytes=list_request_bytes,
                    list_response_bytes=len(list_encoded),
                    call_request_bytes=call_request_bytes,
                    call_started=False,
                )
            result = await session.call_tool(tool.mcp_name, arguments)
        content = _jsonable_mcp_value(getattr(result, "content", None))
        structured = _jsonable_mcp_value(
            getattr(result, "structuredContent", None)
            or getattr(result, "structured_content", None)
        )
        payload = {"content": _bounded_mcp_content(content), "structured_content": structured}
        encoded = dumps(payload).encode("utf-8")
        too_large = len(encoded) > max_response_bytes
        if too_large:
            payload = {"content": _mcp_oversize_observation(encoded), "structured_content": None}
        call_response_bytes = min(len(encoded), max_response_bytes)
        return McpProviderCallResult(
            content=payload["content"],
            structured_content=payload["structured_content"],
            is_error=bool(getattr(result, "isError", False) or getattr(result, "is_error", False)),
            response_bytes=call_response_bytes,
            duration_s=time.monotonic() - started,
            too_large=too_large,
            list_request_bytes=list_request_bytes,
            list_response_bytes=len(list_encoded),
            call_request_bytes=call_request_bytes,
            call_response_bytes=call_response_bytes,
            call_started=True,
        )

    @contextlib.asynccontextmanager
    async def _session(
        self,
        server: McpServerSpec,
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ):
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters
            from mcp.client.streamable_http import streamable_http_client
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP provider requires the optional dependency; install with `uv sync --extra mcp --all-groups`"
            ) from exc
        if server.transport == "stdio":
            if server.stdio is None:
                raise RuntimeError("MCP stdio transport is missing stdio configuration")
            if executable_snapshot is not None:
                executable_snapshot.verify()
                command = str(executable_snapshot.executable_path)
            else:
                resolved_executable = Path(self.resolve_stdio_executable(server))
                command_candidate = self._stdio_command_candidate(server)
                try:
                    dispatch_target = command_candidate.resolve(strict=True)
                except OSError as exc:
                    raise ValidationError(
                        "MCP stdio executable is no longer available"
                    ) from exc
                if dispatch_target != resolved_executable:
                    raise ValidationError(
                        "MCP stdio executable changed before dispatch"
                    )
                # Preserve a virtual-environment launcher's lexical path so
                # Python can discover its pyvenv.cfg. The resolved target above
                # remains the identity that was validated by the primitive.
                command = str(command_candidate)
            params = StdioServerParameters(
                command=command,
                args=list(server.stdio.args),
                env=self._stdio_dispatch_env(server, executable_snapshot),
                cwd=self._resolved_stdio_cwd(server),
            )
            async with _strict_stdio_client(
                params,
                max_frame_bytes=max_response_bytes,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=timeout_s)
                    yield session
            return
        if server.transport == "streamable_http":
            if server.http is None:
                raise RuntimeError("MCP streamable_http transport is missing HTTP configuration")
            async with self._http_client(
                server,
                timeout_s=timeout_s,
                max_response_bytes=max_response_bytes,
            ) as http_client:
                async with streamable_http_client(
                    server.http.url,
                    http_client=http_client,
                ) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await asyncio.wait_for(session.initialize(), timeout=timeout_s)
                        yield session
            return
        raise RuntimeError(f"unsupported MCP transport: {server.transport}")

    @contextlib.asynccontextmanager
    async def _http_client(
        self,
        server: McpServerSpec,
        *,
        timeout_s: float,
        max_response_bytes: int,
    ):
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP provider requires httpx from the optional MCP dependency; "
                "install with `uv sync --extra mcp --all-groups`"
            ) from exc
        timeout = httpx.Timeout(timeout_s, read=timeout_s)
        headers = self._resolved_http_headers(server)
        headers["Accept-Encoding"] = "identity"
        transport = _McpPolicyAsyncHTTPTransport(max_response_bytes=max_response_bytes)
        try:
            async with httpx.AsyncClient(
                headers=headers,
                follow_redirects=False,
                timeout=timeout,
                transport=transport,
                trust_env=False,
            ) as client:
                yield client
                if transport.limit_error is not None:
                    raise transport.limit_error
        except BaseException as exc:
            if transport.limit_error is not None and exc is not transport.limit_error:
                raise transport.limit_error from exc
            raise

    def _resolved_http_headers(self, server: McpServerSpec) -> dict[str, str]:
        if server.http is None:
            return {}
        headers: dict[str, str] = {}
        for name, spec in server.http.headers.items():
            value = os.environ.get(spec.env)
            if value is None:
                raise RuntimeError(f"missing environment variable for MCP header {name}: {spec.env}")
            headers[name] = f"{spec.prefix}{value}{spec.suffix}"
        return headers

    def _resolved_stdio_env(self, server: McpServerSpec) -> dict[str, str]:
        if server.stdio is None:
            return {}
        env = _mcp_platform_env()
        for child_name, host_name in server.stdio.env.items():
            value = os.environ.get(host_name)
            if value is None:
                raise RuntimeError(f"missing environment variable for MCP stdio env {child_name}: {host_name}")
            env[child_name] = value
        return env

    def _stdio_dispatch_env(
        self,
        server: McpServerSpec,
        executable_snapshot: ExecutableSnapshot | None,
    ) -> dict[str, str]:
        env = self._resolved_stdio_env(server)
        if executable_snapshot is None:
            return env
        candidate = self._stdio_command_candidate(server)
        name = candidate.name.lower()
        if re.fullmatch(r"python(?:w)?(?:\d+(?:\.\d+)*)?(?:\.exe)?", name) is None:
            return env
        venv_root = candidate.parent.parent
        if not (venv_root / "pyvenv.cfg").is_file():
            return env
        try:
            selected_target = candidate.resolve(strict=True)
            snapshot_target = executable_snapshot.source_path.resolve(strict=True)
        except OSError as exc:
            raise ValidationError(
                "MCP stdio Python virtual environment launcher is no longer available"
            ) from exc
        if selected_target != snapshot_target:
            raise ValidationError(
                "MCP stdio Python virtual environment launcher changed after executable snapshot"
            )

        env["VIRTUAL_ENV"] = str(venv_root)
        env["PATH"] = os.pathsep.join(
            (str(candidate.parent), env.get("PATH", os.defpath))
        )
        site_packages = sorted(
            path
            for path in (
                *tuple((venv_root / "lib").glob("python*/site-packages")),
                venv_root / "Lib" / "site-packages",
            )
            if path.is_dir()
        )
        if site_packages:
            python_paths = [str(path) for path in site_packages]
            if env.get("PYTHONPATH"):
                python_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["PYTHONNOUSERSITE"] = "1"
        if sys.platform == "darwin":
            # A copied macOS CPython launcher otherwise derives an invalid
            # prefix from the private snapshot directory and cannot find even
            # the standard library. Preserve the selected virtual environment
            # while the executable bytes remain pinned by the snapshot.
            env["__PYVENV_LAUNCHER__"] = str(candidate)
        return env

    def _resolved_stdio_cwd(self, server: McpServerSpec) -> str:
        if server.stdio is None or server.stdio.cwd is None:
            return str(self.workspace_root)
        target = (self.workspace_root / server.stdio.cwd).resolve()
        if target != self.workspace_root and self.workspace_root not in target.parents:
            raise ValidationError("MCP stdio cwd escapes workspace root")
        return str(target)

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation == "list_tools":
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={
                    "provider": "mcp",
                    "server_id": context.get("server_id"),
                    "transport": context.get("transport"),
                    "operation": operation,
                },
            )
        if operation != "call_tool":
            raise ValueError(f"unsupported MCP external effect operation: {operation}")
        rollback_class = ExternalEffectRollbackClass(str(context["rollback_class"]))
        rollback_status = context.get("rollback_status")
        if rollback_status is None:
            rollback_status = default_external_effect_rollback_status(rollback_class)
        return ExternalEffectClassification(
            rollback_class=rollback_class,
            rollback_status=ExternalEffectRollbackStatus(str(rollback_status)),
            state_mutation=bool(context.get("state_mutation")),
            information_flow=bool(context.get("information_flow")),
            metadata={
                "provider": "mcp",
                "server_id": context.get("server_id"),
                "tool_id": context.get("tool_id"),
                "mcp_name": context.get("mcp_name"),
            },
        )


class _McpPolicyAsyncHTTPTransport:
    """MCP address policy plus pre-materialization HTTP response bounds."""

    def __init__(self, *, max_response_bytes: int) -> None:
        try:
            import httpcore
            import httpx  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP HTTP transport requires httpx/httpcore from the optional MCP dependency; "
                "install with `uv sync --extra mcp --all-groups`"
            ) from exc
        if isinstance(max_response_bytes, bool) or max_response_bytes < 1:
            raise ValidationError("MCP HTTP max_response_bytes must be a positive integer")
        self.max_response_bytes = max_response_bytes
        self.limit_error: RuntimeError | None = None
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(),
            max_connections=8,
            max_keepalive_connections=0,
            keepalive_expiry=0,
            http1=True,
            http2=False,
            retries=0,
            network_backend=_McpPolicyNetworkBackend(),
        )

    async def __aenter__(self) -> "_McpPolicyAsyncHTTPTransport":
        with _map_mcp_httpcore_exceptions():
            await self._pool.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        with _map_mcp_httpcore_exceptions():
            await self._pool.__aexit__(exc_type, exc_value, traceback)

    async def handle_async_request(self, request: Any) -> Any:
        import httpcore
        import httpx

        request.headers["Accept-Encoding"] = "identity"
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        with _map_mcp_httpcore_exceptions():
            core_response = await self._pool.handle_async_request(core_request)
        headers = httpx.Headers(core_response.headers)
        content_encoding = headers.get("content-encoding", "").strip().lower()
        if content_encoding and content_encoding != "identity":
            error = self._limit_failure(
                f"MCP HTTP response uses unsupported Content-Encoding={content_encoding}"
            )
            with _map_mcp_httpcore_exceptions():
                await core_response.aclose()
            raise error
        content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        return httpx.Response(
            status_code=core_response.status,
            headers=headers,
            stream=_bounded_mcp_http_stream(
                core_response.stream,
                max_response_bytes=self.max_response_bytes,
                is_sse=content_type == "text/event-stream",
                fail=self._limit_failure,
            ),
            extensions=core_response.extensions,
        )

    def _limit_failure(self, message: str) -> RuntimeError:
        error = RuntimeError(message)
        self.limit_error = error
        return error

    async def aclose(self) -> None:
        with _map_mcp_httpcore_exceptions():
            await self._pool.aclose()


@contextlib.contextmanager
def _map_mcp_httpcore_exceptions() -> Iterator[None]:
    """Translate public httpcore transport errors to their httpx equivalents."""

    try:
        yield
    except Exception as exc:
        import httpcore
        import httpx

        exception_names = (
            "TimeoutException",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "NetworkError",
            "ConnectError",
            "ReadError",
            "WriteError",
            "ProxyError",
            "UnsupportedProtocol",
            "ProtocolError",
            "LocalProtocolError",
            "RemoteProtocolError",
        )
        mapped: type[Exception] | None = None
        for name in exception_names:
            source = getattr(httpcore, name)
            target = getattr(httpx, name)
            if isinstance(exc, source) and (
                mapped is None or issubclass(target, mapped)
            ):
                mapped = target
        if mapped is None:
            raise
        raise mapped(str(exc)) from exc


class _McpHttpResponseLimiter:
    """Count a complete body or one raw SSE event without buffering it."""

    def __init__(self, *, max_response_bytes: int, is_sse: bool) -> None:
        self.max_response_bytes = max_response_bytes
        self.is_sse = is_sse
        self.body_bytes = 0
        self.frame_bytes = 0
        self.line_has_data = False
        self.pending_cr = False
        self.pending_cr_reset = False

    def feed(self, chunk: bytes) -> str | None:
        if not self.is_sse:
            self.body_bytes += len(chunk)
            if self.body_bytes > self.max_response_bytes:
                return f"MCP HTTP response exceeded max_response_bytes={self.max_response_bytes}"
            return None
        for value in chunk:
            if self.pending_cr:
                self.pending_cr = False
                if value == 0x0A:
                    if not self.pending_cr_reset:
                        self.frame_bytes += 1
                    self.pending_cr_reset = False
                    if self.frame_bytes > self.max_response_bytes:
                        return f"MCP HTTP SSE frame exceeded max_response_bytes={self.max_response_bytes}"
                    continue
                self.pending_cr_reset = False
            self.frame_bytes += 1
            if value == 0x0D:
                blank_line = not self.line_has_data
                self.line_has_data = False
                self.pending_cr = True
                self.pending_cr_reset = blank_line
                if blank_line:
                    self.frame_bytes = 0
            elif value == 0x0A:
                blank_line = not self.line_has_data
                self.line_has_data = False
                if blank_line:
                    self.frame_bytes = 0
            else:
                self.line_has_data = True
            if self.frame_bytes > self.max_response_bytes:
                return f"MCP HTTP SSE frame exceeded max_response_bytes={self.max_response_bytes}"
        return None


def _bounded_mcp_http_stream(
    stream: Any,
    *,
    max_response_bytes: int,
    is_sse: bool,
    fail: Callable[[str], RuntimeError],
) -> Any:
    import httpx

    limiter = _McpHttpResponseLimiter(max_response_bytes=max_response_bytes, is_sse=is_sse)

    class BoundedMcpHttpStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            with _map_mcp_httpcore_exceptions():
                async for chunk in stream:
                    message = limiter.feed(chunk)
                    if message is not None:
                        raise fail(message)
                    yield chunk

        async def aclose(self) -> None:
            with _map_mcp_httpcore_exceptions():
                await stream.aclose()

    return BoundedMcpHttpStream()


class _McpPolicyNetworkBackend:
    """Resolve, validate, then connect to the exact MCP HTTP address."""

    def __init__(self) -> None:
        try:
            import httpcore
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP HTTP transport requires httpcore from the optional MCP dependency; "
                "install with `uv sync --extra mcp --all-groups`"
            ) from exc
        self._backend = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        host_text = host.decode("idna") if isinstance(host, bytes) else str(host)
        addresses = _allowed_mcp_connect_addresses(host_text, port)
        last_exc: Exception | None = None
        for address in addresses:
            try:
                return await self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise OSError(f"MCP host resolved no usable addresses: {host_text}")

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> Any:
        return await self._backend.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


def _allowed_mcp_connect_addresses(host: str, port: int) -> list[str]:
    normalized = host.strip("[]").lower()
    if normalized in _MCP_FORBIDDEN_HOSTS:
        raise ValidationError("MCP HTTP host is not allowed")
    allow_local = normalized in _MCP_LOCAL_HTTP_HOSTS
    literal = _ip_address_or_none(host)
    if literal is not None:
        _validate_mcp_connect_ip(literal, allow_local=allow_local)
        return [host.strip("[]")]
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValidationError(f"MCP host could not be resolved: {host}") from exc
    addresses = sorted({info[4][0] for info in infos})
    if not addresses:
        raise ValidationError(f"MCP host resolved no addresses: {host}")
    for address in addresses:
        _validate_mcp_connect_ip(ipaddress.ip_address(address), allow_local=allow_local)
    return addresses


def _ip_address_or_none(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


def _validate_mcp_connect_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, *, allow_local: bool) -> None:
    if allow_local:
        if ip.is_loopback:
            return
        raise ValidationError("MCP local HTTP host must resolve to loopback")
    if not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        raise ValidationError("MCP HTTP IP address is not allowed")


def _mcp_platform_env() -> dict[str, str]:
    if os.name != "nt":
        return {}
    env: dict[str, str] = {}
    for key in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


@contextlib.asynccontextmanager
async def _strict_stdio_client(server: Any, *, max_frame_bytes: int):
    try:
        import anyio
        import anyio.lowlevel
        import mcp.types as mcp_types
        from mcp.os.posix.utilities import terminate_posix_process_tree
        from mcp.os.win32.utilities import create_windows_process, get_windows_executable_command, terminate_windows_process_tree
        from mcp.shared.message import SessionMessage
    except ModuleNotFoundError as exc:
        raise ValidationError(
            "MCP stdio transport requires the optional MCP dependency; "
            "install with `uv sync --extra mcp --all-groups`"
        ) from exc
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)
    reader_errors: list[RuntimeError] = []
    process = None
    command = get_windows_executable_command(server.command) if sys.platform == "win32" else server.command
    try:
        if sys.platform == "win32":
            process = await create_windows_process(command, list(server.args), dict(server.env or {}), sys.stderr, server.cwd)
        else:
            process = await anyio.open_process(
                [command, *list(server.args)],
                env=dict(server.env or {}),
                stderr=sys.stderr,
                cwd=server.cwd,
                start_new_session=True,
            )
    except OSError:
        await read_stream.aclose()
        await write_stream.aclose()
        await read_stream_writer.aclose()
        await write_stream_reader.aclose()
        raise

    async def stdout_reader() -> None:
        assert process is not None and process.stdout
        try:
            async with read_stream_writer:
                buffer = bytearray()
                while True:
                    # Ask the byte stream for only the remaining frame capacity
                    # plus one sentinel byte. The transport buffer therefore
                    # never grows beyond max_frame_bytes + 1 before rejection,
                    # even when AnyIO's default receive chunk is much larger.
                    read_size = _mcp_stdio_read_size(len(buffer), max_frame_bytes)
                    try:
                        chunk = await process.stdout.receive(max_bytes=read_size)
                    except anyio.EndOfStream:
                        break
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while True:
                        newline = buffer.find(b"\n")
                        if newline < 0:
                            if len(buffer) > max_frame_bytes:
                                error = RuntimeError(
                                    "MCP stdio frame exceeded "
                                    f"max_response_bytes={max_frame_bytes}"
                                )
                                reader_errors.append(error)
                                await read_stream_writer.send(error)
                                return
                            break
                        if newline > max_frame_bytes:
                            error = RuntimeError(
                                "MCP stdio frame exceeded "
                                f"max_response_bytes={max_frame_bytes}"
                            )
                            reader_errors.append(error)
                            await read_stream_writer.send(error)
                            return
                        line = bytes(buffer[:newline])
                        del buffer[: newline + 1]
                        try:
                            message = mcp_types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            await read_stream_writer.send(exc)
                            continue
                        await read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def stdin_writer() -> None:
        assert process is not None and process.stdin
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    raw = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await process.stdin.send(
                        (raw + "\n").encode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as task_group, process:
        task_group.start_soon(stdout_reader)
        task_group.start_soon(stdin_writer)
        try:
            try:
                yield read_stream, write_stream
            except BaseException as exc:
                if reader_errors:
                    raise reader_errors[0] from exc
                raise
        finally:
            if process.stdin:
                with contextlib.suppress(Exception):
                    await process.stdin.aclose()
            try:
                with anyio.fail_after(2.0):
                    await process.wait()
            except TimeoutError:
                if sys.platform == "win32":
                    await terminate_windows_process_tree(process)
                else:
                    await terminate_posix_process_tree(process)
            except ProcessLookupError:
                pass
            await read_stream.aclose()
            await write_stream.aclose()
            await read_stream_writer.aclose()
            await write_stream_reader.aclose()


def _mcp_stdio_read_size(buffered_bytes: int, max_frame_bytes: int) -> int:
    if max_frame_bytes < 1:
        raise ValueError("max_frame_bytes must be positive")
    remaining_with_sentinel = max_frame_bytes + 1 - buffered_bytes
    if remaining_with_sentinel < 1:
        raise ValueError("MCP stdio frame buffer already exceeds its hard limit")
    return min(_MCP_STDIO_READ_CHUNK_BYTES, remaining_with_sentinel)


def _run_mcp_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("MCP provider cannot run inside an active event loop; use the async primitive wrapper")


def _jsonable_mcp_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, list):
        return [_jsonable_mcp_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_mcp_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_mcp_value(item) for key, item in value.items()}
    return to_jsonable(value)


def _mcp_metadata(item: Any) -> dict[str, Any]:
    raw = _jsonable_mcp_value(item)
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in raw.items()
        if key not in {"name", "description", "inputSchema", "input_schema"}
    }


def _bounded_mcp_content(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    bounded: list[Any] = []
    for item in value:
        if not isinstance(item, dict):
            bounded.append(item)
            continue
        item_type = item.get("type")
        if item_type in {"image", "audio", "resource"}:
            raw_data = item.get("data") or item.get("blob") or item.get("text") or ""
            raw_text = str(raw_data)
            bounded.append(
                {
                    "type": item_type,
                    "mimeType": item.get("mimeType") or item.get("mime_type"),
                    "bytes": len(raw_text.encode("utf-8")),
                    "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                }
            )
            continue
        bounded.append(item)
    return bounded


def _mcp_oversize_observation(encoded: bytes) -> dict[str, Any]:
    return {
        "type": "oversize",
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


class LocalResourceProviderSubstrate:
    """Default Resource Provider Substrate backed by the host OS."""

    def __init__(
        self,
        workspace_root: str | Path,
        namespace: str = _RUNTIME_DEFAULTS.workspace_namespace,
        *,
        git_config: Any | None = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_display = str(self.workspace_root)
        self.filesystem = LocalFilesystemProvider(self.workspace_root, namespace=namespace)
        self.clock = LocalClockProvider()
        self.shell = LocalShellProvider(self.workspace_root, git_config=git_config)
        self.git = LocalGitProvider(self.workspace_root, config=git_config)
        self.human = LocalHumanProvider()
        self.jsonrpc = HttpJsonRpcProvider()
        self.mcp = SdkMcpProvider(self.workspace_root)
