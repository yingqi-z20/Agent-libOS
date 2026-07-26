from __future__ import annotations

import asyncio
import base64
import binascii
import codecs
import contextlib
import ctypes
import errno
import hashlib
import http.client
import ipaddress
import math
import os
import re
import signal
import shutil
import socket
import ssl
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable, Iterator
from itertools import islice
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

import psutil

from agent_libos.config import DEFAULT_CONFIG, GitDefaults
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
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.substrate.base import (
    CommandMetrics,
    CommandResult,
    DirectoryEntrySnapshot,
    ExecutableSnapshot,
    FilesystemContentConflict,
    HierarchicalPathLock,
    PathState,
    ProviderEffectNotStarted,
    ResolvedPath,
    resolve_runtime_python_alias,
    snapshot_executable,
    SubprocessLimitExceeded,
    SubprocessLimits,
    SubprocessTimeoutExpired,
    _bounded_provider_getaddrinfo,
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
_MCP_STDIO_PROTOCOL_OUTPUT_MULTIPLIER = 4
_MCP_WINDOWS = os.name == "nt"
_MCP_WINDOWS_EXECUTABLE_SUFFIXES = {".com", ".exe"}
_MCP_STABLE_CWD_SUPPORTED = sys.platform.startswith("linux") and Path(
    "/proc/self/fd"
).is_dir()
_JSONRPC_DEADLINE_THREAD_PREFIX = "agent-libos-jsonrpc-deadline"
_DARWIN_F_GETPATH = 50
_DARWIN_F_GETPATH_BUFFER_BYTES = 1024
_DARWIN_ATTR_BIT_MAP_COUNT = 5
_DARWIN_ATTR_VOL_INFO = 0x80000000
_DARWIN_ATTR_VOL_CAPABILITIES = 0x00020000
_DARWIN_ATTR_VOL_FSTYPENAME = 0x00100000
_DARWIN_VOL_CAP_FMT_CASE_SENSITIVE = 0x00000100
_DARWIN_VOL_CAP_FMT_CASE_PRESERVING = 0x00000200
_DARWIN_NORMALIZATION_INSENSITIVE_FILESYSTEMS = {"apfs"}
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


class _ProtectedMetadataDeleteDenied(CapabilityDenied, ProviderEffectNotStarted):
    """A recursive delete rejected before the provider mutated the tree."""


class _McpStdioDispatchNotStarted(ValidationError, ProviderEffectNotStarted):
    """A stdio target failed closed before a child process was created."""


class _McpStdioDispatchStarted(ValidationError):
    """A stdio child was created but failed post-spawn isolation checks."""


class _ProtectedDeleteState:
    def __init__(self) -> None:
        self.mutation_started = False


@dataclass(frozen=True)
class _FilesystemContentSnapshot:
    sha256: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


class _DarwinAttrList(ctypes.Structure):
    _fields_ = [
        ("bitmapcount", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class _DarwinVolumeIdentityPolicy:
    case_sensitive: bool
    case_preserving: bool
    normalization_insensitive: bool
    filesystem_type: str


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
    _JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
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
    def create(
        cls,
        limits: SubprocessLimits | None = None,
    ) -> "WindowsJobObject":
        if os.name != "nt":
            raise OSError("Windows job objects are only available on Windows")
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        job_limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limit_flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if limits is not None and limits.cpu_seconds is not None:
            job_limits.BasicLimitInformation.PerJobUserTimeLimit = int(
                limits.cpu_seconds * 10_000_000
            )
            limit_flags |= _JOB_OBJECT_LIMIT_JOB_TIME
        if limits is not None and limits.memory_bytes is not None:
            job_limits.JobMemoryLimit = int(limits.memory_bytes)
            limit_flags |= _JOB_OBJECT_LIMIT_JOB_MEMORY
        job_limits.BasicLimitInformation.LimitFlags = limit_flags
        if not _kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(job_limits),
            ctypes.sizeof(job_limits),
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


def _darwin_volume_identity_policy(path: Path) -> _DarwinVolumeIdentityPolicy:
    """Read the mounted volume's future-name comparison policy.

    Darwin exposes case sensitivity through ATTR_VOL_CAPABILITIES. Unicode
    normalization behavior is format-defined rather than a capability bit, so
    only APFS is treated as normalization-insensitive and preserving enough
    for this canonical creation scheme;
    unsupported formats fail closed for non-ASCII future names below.
    """

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        getattrlist = libc.getattrlist
    except (AttributeError, OSError) as exc:
        raise CapabilityDenied(
            "cannot determine Darwin volume path identity semantics"
        ) from exc
    getattrlist.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_DarwinAttrList),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
    ]
    getattrlist.restype = ctypes.c_int
    attributes = _DarwinAttrList(
        _DARWIN_ATTR_BIT_MAP_COUNT,
        0,
        0,
        (
            _DARWIN_ATTR_VOL_INFO
            | _DARWIN_ATTR_VOL_CAPABILITIES
            | _DARWIN_ATTR_VOL_FSTYPENAME
        ),
        0,
        0,
        0,
    )
    buffer = ctypes.create_string_buffer(256)
    if getattrlist(
        os.fsencode(path),
        ctypes.byref(attributes),
        buffer,
        ctypes.sizeof(buffer),
        0,
    ) != 0:
        error_number = ctypes.get_errno()
        raise CapabilityDenied(
            "cannot determine Darwin volume path identity semantics"
        ) from OSError(error_number, os.strerror(error_number))
    payload = buffer.raw
    returned_size = struct.unpack_from("=I", payload, 0)[0]
    capabilities_size = 8 * ctypes.sizeof(ctypes.c_uint32)
    reference_offset = ctypes.sizeof(ctypes.c_uint32) + capabilities_size
    if returned_size < reference_offset + 8 or returned_size > len(payload):
        raise CapabilityDenied("invalid Darwin volume path identity metadata")
    capability_values = struct.unpack_from("=8I", payload, 4)
    format_capabilities = capability_values[0]
    valid_format_capabilities = capability_values[4]
    required_case_capabilities = (
        _DARWIN_VOL_CAP_FMT_CASE_SENSITIVE
        | _DARWIN_VOL_CAP_FMT_CASE_PRESERVING
    )
    if (
        valid_format_capabilities & required_case_capabilities
    ) != required_case_capabilities:
        raise CapabilityDenied(
            "Darwin volume does not report valid case identity semantics"
        )
    string_offset, string_length = struct.unpack_from(
        "=iI",
        payload,
        reference_offset,
    )
    string_start = reference_offset + string_offset
    string_end = string_start + string_length
    if (
        string_offset < 8
        or string_length < 2
        or string_start < 0
        or string_end > returned_size
    ):
        raise CapabilityDenied("invalid Darwin volume filesystem identity")
    try:
        filesystem_type = payload[string_start:string_end].rstrip(b"\0").decode(
            "utf-8",
            errors="strict",
        ).casefold()
    except UnicodeDecodeError as exc:
        raise CapabilityDenied("invalid Darwin volume filesystem identity") from exc
    if not filesystem_type:
        raise CapabilityDenied("invalid Darwin volume filesystem identity")
    return _DarwinVolumeIdentityPolicy(
        case_sensitive=bool(
            format_capabilities & _DARWIN_VOL_CAP_FMT_CASE_SENSITIVE
        ),
        case_preserving=bool(
            format_capabilities & _DARWIN_VOL_CAP_FMT_CASE_PRESERVING
        ),
        normalization_insensitive=(
            filesystem_type in _DARWIN_NORMALIZATION_INSENSITIVE_FILESYSTEMS
        ),
        filesystem_type=filesystem_type,
    )


def _optional_darwin_volume_identity_policy(
    path: Path,
) -> _DarwinVolumeIdentityPolicy | None:
    try:
        return _darwin_volume_identity_policy(path)
    except CapabilityDenied:
        return None


class LocalFilesystemProvider:
    """Local-workspace implementation of the filesystem substrate."""

    def __init__(self, root: str | Path, namespace: str = _RUNTIME_DEFAULTS.workspace_namespace):
        lexical_root = Path(root).resolve()
        self.root = (
            self._darwin_existing_descriptor_path(
                lexical_root,
                require_directory=True,
                purpose="filesystem adapter root",
            )
            if sys.platform == "darwin"
            else lexical_root
        )
        self._darwin_volume_policy = (
            _optional_darwin_volume_identity_policy(self.root)
            if sys.platform == "darwin"
            else None
        )
        self.namespace = namespace
        self.root_display = str(self.root)
        self._path_lock = HierarchicalPathLock()

    def resolve(self, path: Any) -> ResolvedPath:
        raw = Path(path)
        candidate = raw if raw.is_absolute() else self.root / raw
        # Resource derivation runs before capability authorization. Most Hosts
        # keep this step purely lexical. Darwin's default filesystems can map
        # several case/Unicode spellings to one directory entry, however, so a
        # lexical resource would let the same file acquire several authority
        # and data-label identities. F_GETPATH exposes only descriptor identity
        # and the Host-stored spelling; it does not read file content. Existing
        # paths (or the nearest existing parent of a create) are canonicalized
        # before deriving every downstream resource/lock/label key.
        target = Path(os.path.abspath(os.path.normpath(os.fspath(candidate))))
        if sys.platform == "darwin":
            try:
                target.relative_to(self.root)
            except ValueError as exc:
                raise CapabilityDenied(f"path escapes filesystem adapter root: {path}") from exc
            try:
                self._reject_reparse_components(target)
            except CapabilityDenied:
                # Preserve the lexical identity for a reparse path. Following
                # it here would reveal the target before authorization. The
                # provider sink repeats this check after authorization and
                # rejects the operation without traversing the component.
                pass
            else:
                target = self._darwin_canonical_path(target)
        try:
            relative_path = target.relative_to(self.root)
        except ValueError as exc:
            raise CapabilityDenied(f"path escapes filesystem adapter root: {path}") from exc
        relative = relative_path.as_posix()
        return ResolvedPath(relative=relative, display=str(target), is_root=target == self.root)

    @staticmethod
    def _darwin_existing_descriptor_path(
        path: Path,
        *,
        require_directory: bool,
        purpose: str,
    ) -> Path:
        """Return Darwin's stored path spelling for one descriptor identity.

        F_GETPATH is the only supported canonicalization mechanism here. A
        lexical/case-fold fallback would merge distinct names on case-sensitive
        volumes, while realpath-style fallback could follow a path that changed
        between identity derivation and authorization.
        """

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        if require_directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            fd = os.open(path, flags)
        except (FileNotFoundError, NotADirectoryError):
            raise
        except OSError as exc:
            raise CapabilityDenied(f"cannot canonicalize {purpose} identity") from exc
        try:
            descriptor_stat = os.fstat(fd)
            if require_directory and not stat.S_ISDIR(descriptor_stat.st_mode):
                raise CapabilityDenied(f"{purpose} is not a directory")
            try:
                import fcntl

                encoded = fcntl.fcntl(
                    fd,
                    _DARWIN_F_GETPATH,
                    b"\0" * _DARWIN_F_GETPATH_BUFFER_BYTES,
                )
            except (ImportError, OSError, TypeError, ValueError) as exc:
                raise CapabilityDenied(f"cannot canonicalize {purpose} identity") from exc
            if not isinstance(encoded, bytes):
                raise CapabilityDenied(f"cannot canonicalize {purpose} identity")
            raw_path = encoded.split(b"\0", 1)[0]
            if not raw_path:
                raise CapabilityDenied(f"cannot canonicalize {purpose} identity")
            canonical = Path(os.fsdecode(raw_path))
            if not canonical.is_absolute():
                raise CapabilityDenied(f"cannot canonicalize {purpose} identity")
            try:
                path_stat = os.stat(canonical, follow_symlinks=False)
            except OSError as exc:
                raise CapabilityDenied(f"cannot verify {purpose} identity") from exc
            if stat.S_ISLNK(path_stat.st_mode) or (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
            ) != (path_stat.st_dev, path_stat.st_ino):
                raise CapabilityDenied(f"cannot verify {purpose} identity")
            return Path(os.path.abspath(os.path.normpath(os.fspath(canonical))))
        finally:
            os.close(fd)

    def _darwin_canonical_path(self, target: Path) -> Path:
        current = target
        missing_parts: list[str] = []
        while True:
            try:
                canonical = self._darwin_existing_descriptor_path(
                    current,
                    require_directory=bool(missing_parts),
                    purpose="filesystem path",
                )
                break
            except (FileNotFoundError, NotADirectoryError):
                parent = current.parent
                if parent == current:
                    raise CapabilityDenied(
                        "cannot canonicalize filesystem path identity"
                    )
                missing_parts.append(current.name)
                current = parent

        future_policy = (
            _darwin_volume_identity_policy(canonical)
            if missing_parts
            else self._darwin_volume_policy
        )
        for part in reversed(missing_parts):
            canonical = canonical / self._darwin_future_component(
                part,
                policy=future_policy,
            )
        canonical = Path(os.path.abspath(os.path.normpath(os.fspath(canonical))))
        try:
            canonical.relative_to(self.root)
        except ValueError as exc:
            raise CapabilityDenied(
                f"path escapes filesystem adapter root: {target}"
            ) from exc
        return canonical

    def _darwin_future_component(
        self,
        component: str,
        *,
        policy: _DarwinVolumeIdentityPolicy | None,
    ) -> str:
        if policy is None:
            raise CapabilityDenied(
                "Darwin volume path identity policy is unavailable"
            )
        if not policy.case_preserving:
            raise CapabilityDenied(
                "cannot derive a stable future path spelling on a "
                "non-case-preserving Darwin volume"
            )
        if policy.normalization_insensitive:
            selected = unicodedata.normalize("NFC", component)
        elif component.isascii():
            selected = component
        else:
            raise CapabilityDenied(
                "cannot derive a safe non-ASCII future path identity on "
                f"Darwin filesystem {policy.filesystem_type}"
            )
        if not policy.case_sensitive:
            folded_parts: list[str] = []
            for character in selected:
                if character.isascii():
                    folded_parts.append(character.lower())
                elif character.casefold() == character:
                    folded_parts.append(character)
                else:
                    raise CapabilityDenied(
                        "cannot derive a safe non-ASCII case-insensitive "
                        "future path identity"
                    )
            selected = "".join(folded_parts)
        if selected in {"", ".", ".."} or "/" in selected or "\0" in selected:
            raise CapabilityDenied("invalid canonical future path component")
        return selected

    def state(self, path: ResolvedPath) -> PathState:
        with self._path_lock.hold(path.relative):
            target = self._target(path)
            return self._state_under_root(target)

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
        self._write_text(
            path,
            text,
            encoding,
            newline,
            overwrite=overwrite,
            expected_content_sha256=None,
        )

    def write_text_compare_and_swap(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None = "\n",
        *,
        overwrite: bool = True,
        expected_content_sha256: str,
    ) -> None:
        self._write_text(
            path,
            text,
            encoding,
            newline,
            overwrite=overwrite,
            expected_content_sha256=expected_content_sha256,
        )

    def _write_text(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None,
        *,
        overwrite: bool,
        expected_content_sha256: str | None,
    ) -> None:
        self._validate_expected_content_sha256(expected_content_sha256)
        with self._path_lock.hold(HierarchicalPathLock.creation_scope(path.relative)):
            target = self._target(path)
            if expected_content_sha256 is not None:
                initial = self._content_snapshot_under_root(target)
                self._require_expected_content(
                    initial,
                    expected_content_sha256,
                    target=target,
                )
            self._before_path_sink_checked("write_parent", target.parent)
            self._ensure_parent_dirs_under_root(target)
            target = self._target(path)
            self._before_path_sink("write_text", target)
            target = self._target(path)
            expected_snapshot = None
            expected_missing = False
            if expected_content_sha256 is not None:
                expected_snapshot = self._content_snapshot_under_root(target)
                self._require_expected_content(
                    expected_snapshot,
                    expected_content_sha256,
                    target=target,
                )
                expected_missing = expected_content_sha256 == "missing"
            with self._open_write_file(
                target,
                encoding=encoding,
                newline=newline,
                overwrite=overwrite,
                expected_snapshot=expected_snapshot,
                expected_missing=expected_missing,
            ) as handle:
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

    def contains_descendant_name(
        self,
        path: ResolvedPath,
        *,
        names: tuple[str, ...],
    ) -> bool:
        with self._path_lock.hold(path.relative):
            target = self._target(path)
            return self._contains_descendant_name(target, names)

    def delete_directory(self, path: ResolvedPath, *, recursive: bool) -> None:
        with self._path_lock.hold(path.relative):
            target = self._target(path)
            self._delete_directory_under_root(
                path,
                target,
                recursive=recursive,
                protected_descendant_names=(),
            )
            self._target(path)

    def delete_directory_protected(
        self,
        path: ResolvedPath,
        *,
        recursive: bool,
        protected_descendant_names: tuple[str, ...],
    ) -> None:
        with self._path_lock.hold(path.relative):
            target = self._target(path)
            self._delete_directory_under_root(
                path,
                target,
                recursive=recursive,
                protected_descendant_names=protected_descendant_names,
            )
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
        # `relative` is the provider-issued authority identity. Reconstruct the
        # sink path from it instead of trusting a caller-controlled display
        # string, and keep Darwin case/Unicode aliases pinned to the spelling
        # used for capability and data-label settlement.
        target = self.root / path.relative
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

    def _open_write_file(
        self,
        target: Path,
        *,
        encoding: str,
        newline: str | None,
        overwrite: bool,
        expected_snapshot: _FilesystemContentSnapshot | None = None,
        expected_missing: bool = False,
    ) -> Any:
        try:
            if expected_snapshot is not None:
                if not overwrite:
                    raise ValidationError(
                        "filesystem CAS overwrite of existing content requires overwrite=true"
                    )
                fd = self._open_under_root(target, os.O_WRONLY)
            else:
                fd = self._open_under_root(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                )
        except FileExistsError:
            if expected_missing:
                raise FilesystemContentConflict(
                    "filesystem content changed before compare-and-swap write"
                ) from None
            if not overwrite:
                raise
            fd = self._open_under_root(target, os.O_WRONLY)
        except FileNotFoundError:
            if expected_snapshot is not None:
                raise FilesystemContentConflict(
                    "filesystem content changed before compare-and-swap write"
                ) from None
            raise
        try:
            self._validate_open_regular_file(fd, target)
            if expected_snapshot is not None:
                observed = self._snapshot_from_stat(
                    expected_snapshot.sha256,
                    os.fstat(fd),
                )
                if observed != expected_snapshot:
                    raise FilesystemContentConflict(
                        "filesystem content changed before compare-and-swap write"
                    )
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

    def _state_under_root(self, target: Path) -> PathState:
        if self._supports_dir_fd_state():
            return self._state_under_root_dir_fd(target)
        if os.name == "nt":
            return self._state_under_root_windows(target)
        raise CapabilityDenied(
            "filesystem state requires descriptor-bound path inspection"
        )

    def _state_under_root_dir_fd(self, target: Path) -> PathState:
        parts = self._relative_parts(target)
        if not parts:
            root_fd = self._open_root_dir_fd()
            try:
                self._before_path_sink_checked("state", target)
                return self._path_state_from_stat(os.fstat(root_fd))
            finally:
                os.close(root_fd)

        try:
            parent_fd, name = self._open_parent_dir_fd(target)
        except FileNotFoundError:
            return PathState(exists=False, kind="missing")
        try:
            self._before_path_sink_checked("state", target)
            try:
                observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return PathState(exists=False, kind="missing")
            if stat.S_ISLNK(observed.st_mode):
                raise CapabilityDenied(
                    f"filesystem path contains a symlink or junction: {target}"
                )
            return self._path_state_from_stat(observed)
        finally:
            os.close(parent_fd)

    def _state_under_root_windows(self, target: Path) -> PathState:
        guard = (
            self._windows_directory_guard(target)
            if target == self.root
            else self._windows_parent_directory_guard(target)
        )
        if guard is None:
            raise CapabilityDenied(
                "filesystem state requires a guarded Windows path"
            )
        try:
            self._before_path_sink_checked("state", target)
            try:
                observed = target.lstat()
            except FileNotFoundError:
                return PathState(exists=False, kind="missing")
            file_attributes = int(getattr(observed, "st_file_attributes", 0))
            reparse_attribute = int(
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
            )
            if stat.S_ISLNK(observed.st_mode) or file_attributes & reparse_attribute:
                raise CapabilityDenied(
                    f"filesystem path contains a symlink or junction: {target}"
                )
            return self._path_state_from_stat(observed)
        finally:
            guard.close()

    @staticmethod
    def _path_state_from_stat(observed: os.stat_result) -> PathState:
        kind = (
            "file"
            if stat.S_ISREG(observed.st_mode)
            else "directory"
            if stat.S_ISDIR(observed.st_mode)
            else "other"
        )
        return PathState(
            exists=True,
            kind=kind,
            size_bytes=observed.st_size if kind == "file" else None,
            modified_at=datetime.fromtimestamp(
                observed.st_mtime,
                timezone.utc,
            ).isoformat(),
        )

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

    def _delete_directory_under_root(
        self,
        path: ResolvedPath,
        target: Path,
        *,
        recursive: bool,
        protected_descendant_names: tuple[str, ...],
    ) -> None:
        if self._supports_dir_fd_deletes():
            dir_fd, name = self._open_parent_dir_fd(target)
            try:
                self._before_path_sink_checked("delete_directory", target)
                self._require_directory_component_for_delete(dir_fd, name, target)
                if recursive:
                    self._reject_protected_descendants(
                        target,
                        protected_descendant_names,
                    )
                    if protected_descendant_names:
                        delete_state = _ProtectedDeleteState()
                        self._delete_protected_directory_at(
                            dir_fd,
                            name,
                            target,
                            protected_descendant_names,
                            delete_state,
                        )
                    else:
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
                self._reject_protected_descendants(
                    target,
                    protected_descendant_names,
                )
                if protected_descendant_names:
                    if os.name != "nt":
                        raise CapabilityDenied(
                            "protected recursive delete requires descriptor-bound "
                            "directory operations on this platform"
                        )
                    self._delete_protected_directory_path(
                        target,
                        protected_descendant_names,
                        _ProtectedDeleteState(),
                    )
                else:
                    shutil.rmtree(target)
            else:
                target.rmdir()
        finally:
            if guard is not None:
                guard.close()

    def _delete_protected_directory_at(
        self,
        parent_fd: int,
        name: str,
        target: Path,
        protected_descendant_names: tuple[str, ...],
        delete_state: _ProtectedDeleteState,
    ) -> None:
        try:
            expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise CapabilityDenied(
                f"filesystem path changed during recursive delete: {target}"
            ) from exc
        directory_fd = self._open_dir_component(parent_fd, name)
        try:
            opened = os.fstat(directory_fd)
            self._require_same_directory_identity(expected, opened, target)
            self._delete_protected_directory_contents_at(
                directory_fd,
                target,
                protected_descendant_names,
                delete_state,
            )
            try:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise CapabilityDenied(
                    f"filesystem path changed during recursive delete: {target}"
                ) from exc
            self._require_same_directory_identity(opened, current, target)
            os.rmdir(name, dir_fd=parent_fd)
            delete_state.mutation_started = True
            self._after_protected_delete_entry(target)
        finally:
            os.close(directory_fd)

    def _delete_protected_directory_contents_at(
        self,
        directory_fd: int,
        target: Path,
        protected_descendant_names: tuple[str, ...],
        delete_state: _ProtectedDeleteState,
    ) -> None:
        protected = {name.casefold() for name in protected_descendant_names}
        names = sorted(os.listdir(directory_fd))
        if any(name.casefold() in protected for name in names):
            self._raise_protected_metadata_delete_denied(delete_state)
        for name in names:
            child_target = target / name
            try:
                child_state = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(child_state.st_mode):
                try:
                    self._delete_protected_directory_at(
                        directory_fd,
                        name,
                        child_target,
                        protected_descendant_names,
                        delete_state,
                    )
                except FileNotFoundError:
                    continue
                continue
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                continue
            delete_state.mutation_started = True
            self._after_protected_delete_entry(child_target)

    def _raise_protected_metadata_delete_denied(
        self,
        delete_state: _ProtectedDeleteState,
    ) -> None:
        message = "recursive directory delete contains protected metadata"
        if delete_state.mutation_started:
            raise CapabilityDenied(message)
        raise _ProtectedMetadataDeleteDenied(message)

    def _after_protected_delete_entry(self, _target: Path) -> None:
        return None

    def _require_same_directory_identity(
        self,
        expected: os.stat_result,
        observed: os.stat_result,
        target: Path,
    ) -> None:
        if (
            not stat.S_ISDIR(observed.st_mode)
            or expected.st_dev != observed.st_dev
            or expected.st_ino != observed.st_ino
        ):
            raise CapabilityDenied(
                f"filesystem directory changed during recursive delete: {target}"
            )

    def _delete_protected_directory_path(
        self,
        target: Path,
        protected_descendant_names: tuple[str, ...],
        delete_state: _ProtectedDeleteState,
    ) -> None:
        guard = self._windows_directory_guard(target)
        try:
            with os.scandir(target) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
            protected = {name.casefold() for name in protected_descendant_names}
            if any(entry.name.casefold() in protected for entry in entries):
                self._raise_protected_metadata_delete_denied(delete_state)
            for entry in entries:
                child = target / entry.name
                try:
                    if child.is_symlink():
                        child.unlink()
                        delete_state.mutation_started = True
                        self._after_protected_delete_entry(child)
                    elif self._is_reparse_path(child):
                        child.rmdir()
                        delete_state.mutation_started = True
                        self._after_protected_delete_entry(child)
                    elif entry.is_dir(follow_symlinks=False):
                        self._delete_protected_directory_path(
                            child,
                            protected_descendant_names,
                            delete_state,
                        )
                    else:
                        child.unlink()
                        delete_state.mutation_started = True
                        self._after_protected_delete_entry(child)
                except FileNotFoundError:
                    continue
        finally:
            guard.close()
        # Do not rescan after processing the captured entry set. If another
        # actor inserted metadata in the meantime, rmdir fails non-empty and
        # leaves that new entry untouched.
        target.rmdir()
        delete_state.mutation_started = True
        self._after_protected_delete_entry(target)

    def _reject_protected_descendants(
        self,
        target: Path,
        protected_descendant_names: tuple[str, ...],
    ) -> None:
        if self._contains_descendant_name(target, protected_descendant_names):
            raise _ProtectedMetadataDeleteDenied(
                "recursive directory delete contains protected metadata"
            )

    def _contains_descendant_name(
        self,
        target: Path,
        names: tuple[str, ...],
    ) -> bool:
        protected = {name.casefold() for name in names}
        if not protected:
            return False

        scan_failed = False

        def record_walk_error(_error: OSError) -> None:
            nonlocal scan_failed
            scan_failed = True

        for current, directory_names, file_names in os.walk(
            target,
            topdown=True,
            onerror=record_walk_error,
            followlinks=False,
        ):
            if any(
                name.casefold() in protected
                for name in (*directory_names, *file_names)
            ):
                return True
            current_path = Path(current)
            retained_directories: list[str] = []
            for name in directory_names:
                try:
                    if self._is_reparse_path(current_path / name):
                        continue
                except OSError:
                    scan_failed = True
                    continue
                retained_directories.append(name)
            directory_names[:] = retained_directories
        # A recursive delete must fail closed if any subtree could not be
        # inspected. The caller deliberately receives only a boolean so path
        # names discovered during this policy scan are never exposed.
        return scan_failed

    def _supports_dir_fd_deletes(self) -> bool:
        return (
            os.open in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
            and os.unlink in os.supports_dir_fd
            and os.rmdir in os.supports_dir_fd
            and os.listdir in os.supports_fd
        )

    @staticmethod
    def _supports_dir_fd_state() -> bool:
        return (
            os.open in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
            and os.stat in os.supports_follow_symlinks
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
            supported = supported and os.scandir in os.supports_fd
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
        selected_limit, reject_overflow = self._directory_scan_limit(limit)
        dir_fd = self._open_directory_under_root(target)
        try:
            with os.scandir(dir_fd) as iterator:
                names = [
                    entry.name
                    for entry in islice(
                        iterator,
                        selected_limit + (1 if reject_overflow else 0),
                    )
                ]
            if reject_overflow and len(names) > selected_limit:
                raise ValidationError(
                    "filesystem directory exceeds the bounded provider listing limit"
                )
            selected_names = sorted(names)
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
        selected_limit, reject_overflow = self._directory_scan_limit(limit)
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
            with os.scandir(target) as iterator:
                children = [
                    Path(entry.path)
                    for entry in islice(
                        iterator,
                        selected_limit + (1 if reject_overflow else 0),
                    )
                ]
            if reject_overflow and len(children) > selected_limit:
                raise ValidationError(
                    "filesystem directory exceeds the bounded provider listing limit"
                )
            children.sort(key=lambda item: item.name)
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

    @staticmethod
    def _validate_expected_content_sha256(value: str | None) -> None:
        if value is None or value == "missing":
            return
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValidationError(
                "expected_content_sha256 must be 'missing' or 64 lowercase hexadecimal characters"
            )

    def _content_snapshot_under_root(
        self,
        target: Path,
    ) -> _FilesystemContentSnapshot | None:
        try:
            handle = self._open_existing_file(target, os.O_RDONLY)
        except FileNotFoundError:
            return None
        with handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            before_snapshot = self._snapshot_from_stat(digest.hexdigest(), before)
            after_snapshot = self._snapshot_from_stat(digest.hexdigest(), after)
            if before_snapshot != after_snapshot:
                raise FilesystemContentConflict(
                    "filesystem content changed while computing compare-and-swap identity"
                )
            return after_snapshot

    @staticmethod
    def _snapshot_from_stat(
        sha256: str,
        observed: os.stat_result,
    ) -> _FilesystemContentSnapshot:
        return _FilesystemContentSnapshot(
            sha256=sha256,
            device=int(observed.st_dev),
            inode=int(observed.st_ino),
            size=int(observed.st_size),
            modified_ns=int(observed.st_mtime_ns),
            changed_ns=int(observed.st_ctime_ns),
        )

    @staticmethod
    def _require_expected_content(
        snapshot: _FilesystemContentSnapshot | None,
        expected: str,
        *,
        target: Path,
    ) -> None:
        matches = (
            snapshot is None
            if expected == "missing"
            else snapshot is not None and snapshot.sha256 == expected
        )
        if not matches:
            raise FilesystemContentConflict(
                f"filesystem content compare-and-swap conflict: {target}"
            )

    @staticmethod
    def _directory_scan_limit(limit: int | None) -> tuple[int, bool]:
        if limit is None:
            return int(_TOOL_DEFAULTS.directory_entry_hard_limit), True
        if isinstance(limit, bool) or type(limit) is not int or limit <= 0:
            raise ValidationError("filesystem provider listing limit must be positive")
        return limit, False

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


class _BoundedUnicodePipeCapture:
    """Drain one subprocess pipe while retaining at most ``limit`` characters."""

    _READ_BYTES = 4_096

    def __init__(
        self,
        stream: Any,
        *,
        limit: int,
        overflow_event: threading.Event,
    ) -> None:
        self.stream = stream
        self.limit = limit
        self.overflow_event = overflow_event
        self.chunks: list[str] = []
        self.characters = 0
        self.truncated = False
        self.error: BaseException | None = None

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    def run(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                chunk = os.read(self.stream.fileno(), self._READ_BYTES)
                if not chunk:
                    self._append(decoder.decode(b"", final=True))
                    return
                if not self._append(decoder.decode(chunk, final=False)):
                    return
        except BaseException as exc:
            self.error = exc
            self.overflow_event.set()
        finally:
            with contextlib.suppress(OSError, ValueError):
                self.stream.close()

    def _append(self, value: str) -> bool:
        if not value:
            return True
        remaining = self.limit - self.characters
        if len(value) > remaining:
            if remaining > 0:
                self.chunks.append(value[:remaining])
                self.characters += remaining
            self.truncated = True
            self.overflow_event.set()
            return False
        self.chunks.append(value)
        self.characters += len(value)
        return True


@dataclass
class _ShellProcessExecution:
    proc: subprocess.Popen[bytes]
    job: WindowsJobObject | None
    ps_process: psutil.Process | None
    stdout_capture: _BoundedUnicodePipeCapture
    stderr_capture: _BoundedUnicodePipeCapture
    capture_event: threading.Event
    capture_threads: tuple[threading.Thread, threading.Thread]
    require_complete_metrics: bool
    started_at: float
    peak_memory: int = 0
    cpu_seconds: float = 0.0
    limit_kind: str | None = None
    timed_out: bool = False
    cleanup_failed: bool = False


class LocalShellProvider:
    """Subprocess-backed shell provider scoped to a configured working directory."""

    supports_subprocess_limits = os.name != "nt"
    supports_executable_snapshots = True

    def __init__(
        self,
        cwd: str | Path,
        *,
        git_config: GitDefaults | None = None,
    ):
        self.cwd = Path(cwd).resolve()
        self._git_config = git_config
        self._git_read_guard: LocalGitProvider | None = None

    @property
    def git_config(self) -> GitDefaults:
        """Return the effective Git policy used by raw Git read guards."""

        return self._git_config or DEFAULT_CONFIG.git

    def bind_runtime_git_config(self, git_config: GitDefaults) -> None:
        """Bind the Runtime-owned Git policy used by legacy raw read guards."""

        if (
            self._git_read_guard is not None
            and self._git_read_guard.config != git_config
        ):
            raise ValidationError(
                "local shell provider already initialized a Git guard with "
                "a different configuration"
            )
        self._git_config = git_config

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
        checked_argv, popen_executable = self._prepare_shell_dispatch_argv(
            argv,
            selected_cwd,
            executable_snapshot,
        )
        return self._execute_shell_process(
            argv=argv,
            checked_argv=checked_argv,
            popen_executable=popen_executable,
            selected_cwd=selected_cwd,
            timeout=timeout,
            limits=limits,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )

    def _prepare_shell_dispatch_argv(
        self,
        argv: list[str],
        selected_cwd: Path,
        executable_snapshot: ExecutableSnapshot | None,
    ) -> tuple[list[str], str | None]:
        requested_argv0 = argv[0] if argv else None
        checked_argv = self._resolve_argv0(argv, selected_cwd)
        popen_executable: str | None = None
        if executable_snapshot is None:
            return checked_argv, popen_executable
        executable_snapshot.verify()
        if executable_snapshot.source_path != Path(checked_argv[0]).resolve(
            strict=False
        ):
            raise ValidationError(
                "shell executable snapshot does not match resolved argv[0]"
            )
        if os.name == "nt":
            return [
                str(executable_snapshot.executable_path),
                *checked_argv[1:],
            ], None
        # Select pinned executable bytes while retaining the invocation argv[0]
        # used by virtual-environment launchers to locate pyvenv.cfg.
        popen_executable = str(executable_snapshot.executable_path)
        if requested_argv0 is not None:
            checked_argv = [requested_argv0, *checked_argv[1:]]
        return checked_argv, popen_executable

    def _execute_shell_process(
        self,
        *,
        argv: list[str],
        checked_argv: list[str],
        popen_executable: str | None,
        selected_cwd: Path,
        timeout: float,
        limits: SubprocessLimits | None,
        stdout_limit: int,
        stderr_limit: int,
    ) -> CommandResult:
        started_at = time.monotonic()
        execution = self._launch_shell_process(
            checked_argv=checked_argv,
            popen_executable=popen_executable,
            selected_cwd=selected_cwd,
            limits=limits,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
            started_at=started_at,
        )
        try:
            self._supervise_shell_process(
                execution,
                timeout=timeout,
                limits=limits,
            )
        finally:
            execution.cleanup_failed = self._cleanup_shell_process(execution)
        return self._finish_shell_execution(execution, argv=argv, timeout=timeout)

    def _launch_shell_process(
        self,
        *,
        checked_argv: list[str],
        popen_executable: str | None,
        selected_cwd: Path,
        limits: SubprocessLimits | None,
        stdout_limit: int,
        stderr_limit: int,
        started_at: float,
    ) -> _ShellProcessExecution:
        job = self._windows_job_for_run(limits)
        proc: subprocess.Popen[bytes] | None = None
        ps_proc: psutil.Process | None = None
        capture_threads: tuple[threading.Thread, ...] = ()
        require_complete_metrics = bool(
            limits is not None
            and (limits.cpu_seconds is not None or limits.memory_bytes is not None)
        )
        try:
            proc = subprocess.Popen(
                checked_argv,
                executable=popen_executable,
                cwd=selected_cwd,
                env=self._safe_env(),
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **self._process_group_kwargs(),
            )
            job = self._attach_shell_job(job, proc, limits=limits)
            ps_proc = self._shell_ps_process(
                proc,
                require_complete=require_complete_metrics,
            )
            assert proc.stdout is not None and proc.stderr is not None
            capture_event = threading.Event()
            stdout_capture = _BoundedUnicodePipeCapture(
                proc.stdout,
                limit=stdout_limit,
                overflow_event=capture_event,
            )
            stderr_capture = _BoundedUnicodePipeCapture(
                proc.stderr,
                limit=stderr_limit,
                overflow_event=capture_event,
            )
            capture_threads = (
                threading.Thread(
                    target=stdout_capture.run,
                    name=f"agent-libos-shell-stdout-{proc.pid}",
                    daemon=True,
                ),
                threading.Thread(
                    target=stderr_capture.run,
                    name=f"agent-libos-shell-stderr-{proc.pid}",
                    daemon=True,
                ),
            )
            for thread in capture_threads:
                thread.start()
            return _ShellProcessExecution(
                proc=proc,
                job=job,
                ps_process=ps_proc,
                stdout_capture=stdout_capture,
                stderr_capture=stderr_capture,
                capture_event=capture_event,
                capture_threads=capture_threads,
                require_complete_metrics=require_complete_metrics,
                started_at=started_at,
            )
        except BaseException:
            if proc is not None:
                self._cleanup_shell_process_parts(
                    proc,
                    ps_process=ps_proc,
                    job=job,
                    capture_threads=capture_threads,
                )
            elif job is not None:
                job.close()
            raise

    @staticmethod
    def _attach_shell_job(
        job: WindowsJobObject | None,
        proc: subprocess.Popen[bytes],
        *,
        limits: SubprocessLimits | None,
    ) -> WindowsJobObject | None:
        if job is None:
            return None
        try:
            job.assign(proc)
        except OSError as exc:
            job.close()
            if limits is not None:
                raise ValidationError(
                    "shell provider could not attach Windows Job Object for budgeted execution"
                ) from exc
            return None
        return job

    @staticmethod
    def _shell_ps_process(
        proc: subprocess.Popen[bytes],
        *,
        require_complete: bool,
    ) -> psutil.Process | None:
        try:
            return psutil.Process(proc.pid)
        except (psutil.Error, OSError) as exc:
            if require_complete:
                raise ValidationError(
                    "shell provider cannot enforce CPU/memory SubprocessLimits because process metrics are unavailable"
                ) from exc
            return None

    def _supervise_shell_process(
        self,
        execution: _ShellProcessExecution,
        *,
        timeout: float,
        limits: SubprocessLimits | None,
    ) -> None:
        while True:
            wall_seconds = time.monotonic() - execution.started_at
            if execution.ps_process is not None:
                execution.cpu_seconds, execution.peak_memory = self._sample_process_tree(
                    execution.ps_process,
                    execution.peak_memory,
                    require_complete=execution.require_complete_metrics,
                )
            execution.limit_kind = self._limit_kind(
                wall_seconds=wall_seconds,
                cpu_seconds=execution.cpu_seconds,
                peak_memory=execution.peak_memory,
                limits=limits,
            ) or self._shell_capture_limit_kind(execution)
            if execution.limit_kind is not None or self._shell_capture_error(execution):
                self._stop_shell_process(execution)
                return
            if timeout is not None and wall_seconds > timeout:
                execution.timed_out = True
                self._stop_shell_process(execution)
                return
            if execution.proc.poll() is not None:
                self._terminate_process_group(execution.proc)
                return
            execution.capture_event.wait(0.02)

    @staticmethod
    def _shell_capture_limit_kind(
        execution: _ShellProcessExecution,
    ) -> str | None:
        if execution.stdout_capture.truncated:
            return "subprocess_stdout_chars"
        if execution.stderr_capture.truncated:
            return "subprocess_stderr_chars"
        return None

    @staticmethod
    def _shell_capture_error(execution: _ShellProcessExecution) -> bool:
        return (
            execution.stdout_capture.error is not None
            or execution.stderr_capture.error is not None
        )

    def _stop_shell_process(self, execution: _ShellProcessExecution) -> None:
        self._kill_process_tree(execution.ps_process, execution.proc)
        with contextlib.suppress(subprocess.TimeoutExpired):
            execution.proc.wait(timeout=1.0)

    def _cleanup_shell_process(self, execution: _ShellProcessExecution) -> bool:
        return self._cleanup_shell_process_parts(
            execution.proc,
            ps_process=execution.ps_process,
            job=execution.job,
            capture_threads=execution.capture_threads,
        )

    def _cleanup_shell_process_parts(
        self,
        proc: subprocess.Popen[bytes],
        *,
        ps_process: psutil.Process | None,
        job: WindowsJobObject | None,
        capture_threads: tuple[threading.Thread, ...],
    ) -> bool:
        if proc.poll() is None:
            self._kill_process_tree(ps_process, proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=1.0)
        for thread in capture_threads:
            if thread.ident is not None:
                thread.join(timeout=1.0)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
        for thread in capture_threads:
            if thread.ident is not None and thread.is_alive():
                thread.join(timeout=0.25)
        if job is not None:
            job.close()
        return any(
            thread.ident is not None and thread.is_alive()
            for thread in capture_threads
        )

    def _finish_shell_execution(
        self,
        execution: _ShellProcessExecution,
        *,
        argv: list[str],
        timeout: float,
    ) -> CommandResult:
        if execution.cleanup_failed:
            raise RuntimeError("shell output reader did not terminate after process cleanup")
        for capture in (execution.stdout_capture, execution.stderr_capture):
            if capture.error is not None:
                raise capture.error
        execution.limit_kind = (
            execution.limit_kind or self._shell_capture_limit_kind(execution)
        )
        wall_seconds = time.monotonic() - execution.started_at
        if execution.ps_process is not None:
            final_cpu_seconds, execution.peak_memory = self._sample_process_tree(
                execution.ps_process,
                execution.peak_memory,
                require_complete=execution.require_complete_metrics,
            )
            execution.cpu_seconds = max(
                execution.cpu_seconds,
                final_cpu_seconds,
            )
        metrics = CommandMetrics(
            wall_seconds=wall_seconds,
            cpu_seconds=execution.cpu_seconds,
            peak_memory_bytes=execution.peak_memory,
            killed=execution.timed_out or execution.limit_kind is not None,
            limit_kind=(
                "subprocess_timeout"
                if execution.timed_out
                else execution.limit_kind
            ),
        )
        result = CommandResult(
            argv=list(argv),
            returncode=(
                execution.proc.returncode
                if execution.proc.returncode is not None
                else -9
            ),
            stdout=execution.stdout_capture.text,
            stderr=execution.stderr_capture.text,
            stdout_truncated=execution.stdout_capture.truncated,
            stderr_truncated=execution.stderr_capture.truncated,
            metrics=metrics,
        )
        if execution.timed_out:
            raise SubprocessTimeoutExpired(
                f"subprocess timed out after {timeout}s",
                metrics=metrics,
                result=result,
            )
        if execution.limit_kind is not None:
            raise SubprocessLimitExceeded(
                f"subprocess exceeded {execution.limit_kind}",
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


class _JsonRpcSocketDeadline:
    """Abort one exact pinned socket when its absolute request deadline passes."""

    def __init__(self, sock: Any, deadline: float) -> None:
        self.sock = sock
        self.deadline = deadline
        self.expired = threading.Event()
        self._timer: threading.Timer | None = None

    def __enter__(self) -> _JsonRpcSocketDeadline:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self._expire()
            raise TimeoutError("JSON-RPC pinned request timed out")
        timer = threading.Timer(remaining, self._expire)
        timer.name = f"{_JSONRPC_DEADLINE_THREAD_PREFIX}-{id(self.sock):x}"
        timer.daemon = True
        self._timer = timer
        timer.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        timer = self._timer
        if timer is None:
            return
        timer.cancel()
        # A cancelled Timer wakes immediately. Joining is deliberate: no
        # request may return while its watchdog can still close a reused fd.
        timer.join()

    def require_time_remaining(self) -> None:
        if not self.expired.is_set() and time.monotonic() < self.deadline:
            return
        self._expire()
        raise TimeoutError("JSON-RPC pinned request timed out")

    def _expire(self) -> None:
        self.expired.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except BaseException:
            pass
        try:
            self.sock.close()
        except BaseException:
            pass


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
        resolved_headers: Mapping[str, str] | None = None,
    ) -> JsonRpcTransportResult:
        selected_headers = (
            dict(resolved_headers)
            if resolved_headers is not None
            else self._resolved_headers(endpoint)
        )
        if resolved_addresses:
            return self._call_pinned(
                endpoint,
                request_body,
                timeout_s=timeout_s,
                max_response_bytes=max_response_bytes,
                resolved_addresses=resolved_addresses,
                resolved_headers=selected_headers,
            )
        started = time.monotonic()
        request = urlrequest.Request(
            endpoint.url,
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **selected_headers,
            },
            method="POST",
        )
        # Provider calls must not silently inherit ambient proxy routing.
        opener = urlrequest.build_opener(
            urlrequest.ProxyHandler({}),
            self._NoRedirectHandler,
        )
        try:
            with opener.open(request, timeout=timeout_s) as response:
                # Retain one sentinel byte so the primitive can derive the
                # limit outcome without trusting provider-owned metadata.
                body = response.read(max_response_bytes + 1)
                too_large = len(body) > max_response_bytes
                return JsonRpcTransportResult(
                    status_code=int(response.status),
                    body=body,
                    elapsed_s=time.monotonic() - started,
                    response_bytes=min(len(body), max_response_bytes),
                    too_large=too_large,
                )
        except urlerror.HTTPError as exc:
            try:
                body = exc.read(max_response_bytes + 1)
                too_large = len(body) > max_response_bytes
                return JsonRpcTransportResult(
                    status_code=int(exc.code),
                    body=body,
                    elapsed_s=time.monotonic() - started,
                    response_bytes=min(len(body), max_response_bytes),
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
        resolved_headers: Mapping[str, str],
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
            **resolved_headers,
        }
        request_head = self._http_request_head("POST", request_target, headers)
        last_error: str | None = None
        deadline = started + timeout_s
        for address in resolved_addresses:
            request_dispatch_started = False
            deadline_guard: _JsonRpcSocketDeadline | None = None
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
                    deadline_guard = _JsonRpcSocketDeadline(sock, deadline)
                    with deadline_guard:
                        handshake = getattr(sock, "do_handshake", None)
                        if callable(handshake):
                            handshake()
                        request_dispatch_started = True
                        sock.sendall(request_head + request_body)
                        response = http.client.HTTPResponse(sock)
                        response.begin()
                        # Retain one sentinel byte for primitive-side validation.
                        body = response.read(max_response_bytes + 1)
                        deadline_guard.require_time_remaining()
                        too_large = len(body) > max_response_bytes
                        return JsonRpcTransportResult(
                            status_code=int(response.status),
                            body=body,
                            elapsed_s=time.monotonic() - started,
                            response_bytes=min(len(body), max_response_bytes),
                            too_large=too_large,
                        )
            except Exception as exc:
                last_error = (
                    "TimeoutError: JSON-RPC pinned request timed out"
                    if deadline_guard is not None and deadline_guard.expired.is_set()
                    else f"{type(exc).__name__}: {exc}"
                )
                if request_dispatch_started:
                    break
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
                # The caller performs the handshake under the same absolute
                # socket watchdog as request dispatch and response parsing.
                return context.wrap_socket(
                    raw,
                    server_hostname=host,
                    do_handshake_on_connect=False,
                )
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
    supports_runtime_environment_snapshots = True
    supports_subprocess_limits = True

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()

    def list_tools(
        self,
        server: McpServerSpec,
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
        limits: SubprocessLimits | None = None,
    ) -> McpToolListResult:
        deadline = time.monotonic() + timeout_s
        with self._stdio_dispatch_snapshot(
            server,
            executable_snapshot,
            runtime_environment=runtime_environment,
        ) as selected_snapshot:
            try:
                return _run_mcp_async(
                    _mcp_await_with_deadline(
                        self._alist_tools(
                            server,
                            deadline=deadline,
                            max_response_bytes=max_response_bytes,
                            executable_snapshot=selected_snapshot,
                            runtime_environment=runtime_environment,
                            limits=limits,
                        ),
                        deadline=deadline,
                        stage="tools/list",
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
        runtime_environment: Mapping[str, str] | None = None,
        limits: SubprocessLimits | None = None,
    ) -> McpProviderCallResult:
        deadline = time.monotonic() + timeout_s
        with self._stdio_dispatch_snapshot(
            server,
            executable_snapshot,
            runtime_environment=runtime_environment,
        ) as selected_snapshot:
            try:
                return _run_mcp_async(
                    _mcp_await_with_deadline(
                        self._acall_tool(
                            server,
                            tool,
                            arguments,
                            deadline=deadline,
                            max_response_bytes=max_response_bytes,
                            executable_snapshot=selected_snapshot,
                            runtime_environment=runtime_environment,
                            limits=limits,
                        ),
                        deadline=deadline,
                        stage=f"tools/call {tool.mcp_name}",
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
        runtime_environment: Mapping[str, str] | None = None,
        limits: SubprocessLimits | None = None,
    ) -> McpProviderCallResult:
        """Validate and invoke through one MCP session and one wall-clock deadline."""

        deadline = time.monotonic() + timeout_s
        with self._stdio_dispatch_snapshot(
            server,
            executable_snapshot,
            runtime_environment=runtime_environment,
        ) as selected_snapshot:
            try:
                return _run_mcp_async(
                    _mcp_await_with_deadline(
                        self._avalidate_and_call(
                            server,
                            tool,
                            arguments,
                            deadline=deadline,
                            max_response_bytes=max_response_bytes,
                            executable_snapshot=selected_snapshot,
                            runtime_environment=runtime_environment,
                            limits=limits,
                        ),
                        deadline=deadline,
                        stage=f"validated tools/call {tool.mcp_name}",
                    )
                )
            except BaseExceptionGroup as exc:
                message = self._mcp_transport_limit_message(exc)
                if message is not None:
                    return self._mcp_transport_failure_result(
                        server,
                        tool,
                        arguments,
                        message=message,
                        started_at=deadline - timeout_s,
                        max_response_bytes=max_response_bytes,
                    )
                raise
            except RuntimeError as exc:
                message = self._mcp_transport_limit_message(exc)
                if message is None:
                    raise
                return self._mcp_transport_failure_result(
                    server,
                    tool,
                    arguments,
                    message=message,
                    started_at=deadline - timeout_s,
                    max_response_bytes=max_response_bytes,
                )

    def resolve_stdio_executable(
        self,
        server: McpServerSpec,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> str:
        """Resolve the exact stdio executable used by the local MCP transport."""

        if server.transport != "stdio" or server.stdio is None:
            raise ValidationError("MCP stdio executable resolution requires stdio configuration")
        candidate = self._stdio_command_candidate(
            server,
            runtime_environment=runtime_environment,
        )
        if (
            _MCP_WINDOWS
            and candidate.suffix.casefold() not in _MCP_WINDOWS_EXECUTABLE_SUFFIXES
        ):
            raise ValidationError(
                "Windows MCP stdio executables must end in .exe or .com"
            )
        resolved_candidate = candidate.resolve(strict=True)
        if not resolved_candidate.is_file():
            raise ValidationError(
                f"MCP stdio executable is not a regular file: {resolved_candidate}"
            )
        return str(resolved_candidate)

    def _stdio_command_candidate(
        self,
        server: McpServerSpec,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> Path:
        if server.transport != "stdio" or server.stdio is None:
            raise ValidationError("MCP stdio executable resolution requires stdio configuration")
        command = server.stdio.command
        selected_cwd = Path(self._resolved_stdio_cwd(server))
        raw = Path(command)
        if raw.is_absolute() or "/" in command or "\\" in command:
            candidate = raw if raw.is_absolute() else selected_cwd / raw
        else:
            child_env = self._resolved_stdio_env(
                server,
                runtime_environment=runtime_environment,
            )
            if _MCP_WINDOWS:
                resolved = _resolve_windows_mcp_bare_command(
                    command,
                    search_path=child_env.get("PATH"),
                    pathext=child_env.get("PATHEXT"),
                )
            else:
                resolved = shutil.which(
                    command,
                    path=child_env.get("PATH", os.defpath),
                )
            if resolved is None:
                raise FileNotFoundError(f"MCP stdio executable not found: {command}")
            candidate = Path(resolved)
        return Path(os.path.abspath(candidate))

    def executable_snapshot_required(
        self,
        server: McpServerSpec,
        resolved_executable: str,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> bool:
        del resolved_executable, runtime_environment
        return server.transport == "stdio" and server.stdio is not None

    @contextlib.contextmanager
    def _stdio_dispatch_snapshot(
        self,
        server: McpServerSpec,
        executable_snapshot: ExecutableSnapshot | None,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> Iterator[ExecutableSnapshot | None]:
        if executable_snapshot is not None:
            executable_snapshot.verify()
            yield executable_snapshot
            return
        if server.transport != "stdio" or server.stdio is None:
            yield None
            return
        resolved = self.resolve_stdio_executable(
            server,
            runtime_environment=runtime_environment,
        )
        if not self.executable_snapshot_required(
            server,
            resolved,
            runtime_environment=runtime_environment,
        ):
            yield None
            return
        with snapshot_executable(
            resolved,
            sibling_policy="scripts",
        ) as owned_snapshot:
            yield owned_snapshot

    @staticmethod
    def _raise_mcp_transport_limit_error(error: BaseException) -> None:
        message = SdkMcpProvider._mcp_transport_limit_message(error)
        if message is not None:
            raise RuntimeError(message) from error

    @staticmethod
    def _mcp_transport_limit_message(error: BaseException) -> str | None:
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            message = str(current)
            if isinstance(
                current,
                (SubprocessLimitExceeded, SubprocessTimeoutExpired),
            ):
                return None
            if message.startswith(
                (
                    "MCP stdio frame exceeded max_response_bytes=",
                    "MCP stdio stdout exceeded max_output_bytes=",
                    "MCP stdio stderr exceeded max_output_bytes=",
                    "MCP stdio request frame exceeded max_request_bytes=",
                    "MCP stdio stdin exceeded max_output_bytes=",
                    "MCP HTTP response exceeded max_response_bytes=",
                    "MCP HTTP SSE frame exceeded max_response_bytes=",
                    "MCP HTTP response uses unsupported Content-Encoding=",
                )
            ):
                return message
            if isinstance(current, BaseExceptionGroup):
                pending.extend(current.exceptions)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return None

    @staticmethod
    def _mcp_transport_failure_result(
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        message: str,
        started_at: float,
        max_response_bytes: int,
    ) -> McpProviderCallResult:
        list_request_bytes = len(
            dumps({"method": "tools/list", "server_id": server.server_id}).encode(
                "utf-8"
            )
        )
        call_request_bytes = len(
            dumps({"name": tool.mcp_name, "arguments": arguments}).encode("utf-8")
        )
        error_type = "McpTransportLimitError"
        if message.startswith("MCP stdio frame exceeded"):
            error_type = "McpStdioFrameTooLarge"
        elif message.startswith("MCP stdio stdout exceeded"):
            error_type = "McpStdioStdoutTooLarge"
        elif message.startswith("MCP stdio stderr exceeded"):
            error_type = "McpStdioStderrTooLarge"
        elif message.startswith("MCP HTTP SSE frame exceeded"):
            error_type = "McpHttpSseFrameTooLarge"
        elif message.startswith("MCP HTTP response exceeded"):
            error_type = "McpHttpResponseTooLarge"
        elif message.startswith("MCP HTTP response uses unsupported Content-Encoding="):
            error_type = "McpHttpContentEncodingDenied"
        return McpProviderCallResult(
            error="bounded MCP transport failure",
            error_type=error_type,
            correlation_id=new_id("corr"),
            response_bytes=max_response_bytes,
            duration_s=max(0.0, time.monotonic() - started_at),
            list_request_bytes=list_request_bytes,
            list_response_bytes=0,
            call_request_bytes=call_request_bytes,
            call_response_bytes=max_response_bytes,
            call_started=True,
        )

    async def _alist_tools(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
        limits: SubprocessLimits | None = None,
    ) -> McpToolListResult:
        started = time.monotonic()
        async with self._session(
            server,
            deadline=deadline,
            max_response_bytes=max_response_bytes,
            executable_snapshot=executable_snapshot,
            runtime_environment=runtime_environment,
            limits=limits,
        ) as session:
            result = await _mcp_await_with_deadline(
                session.list_tools(),
                deadline=deadline,
                stage="tools/list response",
            )
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
        deadline: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
        limits: SubprocessLimits | None = None,
    ) -> McpProviderCallResult:
        started = time.monotonic()
        async with self._session(
            server,
            deadline=deadline,
            max_response_bytes=max_response_bytes,
            executable_snapshot=executable_snapshot,
            runtime_environment=runtime_environment,
            limits=limits,
        ) as session:
            result = await _mcp_await_with_deadline(
                session.call_tool(tool.mcp_name, arguments),
                deadline=deadline,
                stage=f"tools/call response {tool.mcp_name}",
            )
        content = _jsonable_mcp_value(getattr(result, "content", None))
        structured = _jsonable_mcp_value(_mcp_structured_content(result))
        raw_payload = {"content": content, "structured_content": structured}
        encoded = dumps(raw_payload).encode("utf-8")
        too_large = len(encoded) > max_response_bytes
        if too_large:
            payload = {"content": _mcp_oversize_observation(encoded), "structured_content": None}
        else:
            payload = {
                "content": _bounded_mcp_content(content),
                "structured_content": _bounded_mcp_content(structured),
            }
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
        deadline: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
        limits: SubprocessLimits | None = None,
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
            deadline=deadline,
            max_response_bytes=max_response_bytes,
            executable_snapshot=executable_snapshot,
            runtime_environment=runtime_environment,
            limits=limits,
        ) as session:
            live_result = await _mcp_await_with_deadline(
                session.list_tools(),
                deadline=deadline,
                stage="validated tools/list response",
            )
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
                tool.input_schema and live.input_schema != tool.input_schema
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
            result = await _mcp_await_with_deadline(
                session.call_tool(tool.mcp_name, arguments),
                deadline=deadline,
                stage=f"validated tools/call response {tool.mcp_name}",
            )
        content = _jsonable_mcp_value(getattr(result, "content", None))
        structured = _jsonable_mcp_value(_mcp_structured_content(result))
        raw_payload = {"content": content, "structured_content": structured}
        encoded = dumps(raw_payload).encode("utf-8")
        too_large = len(encoded) > max_response_bytes
        if too_large:
            payload = {"content": _mcp_oversize_observation(encoded), "structured_content": None}
        else:
            payload = {
                "content": _bounded_mcp_content(content),
                "structured_content": _bounded_mcp_content(structured),
            }
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
        deadline: float | None = None,
        timeout_s: float | None = None,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
        limits: SubprocessLimits | None = None,
    ):
        if deadline is None:
            deadline = time.monotonic() + (
                server.timeout_s if timeout_s is None else timeout_s
            )
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
                resolved_executable = Path(
                    self.resolve_stdio_executable(
                        server,
                        runtime_environment=runtime_environment,
                    )
                )
                command_candidate = self._stdio_command_candidate(
                    server,
                    runtime_environment=runtime_environment,
                )
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
            with self._stdio_dispatch_cwd(server) as (dispatch_cwd, cwd_fd):
                params = StdioServerParameters(
                    command=command,
                    args=list(server.stdio.args),
                    env=self._stdio_dispatch_env(
                        server,
                        executable_snapshot,
                        runtime_environment=runtime_environment,
                    ),
                    cwd=dispatch_cwd,
                )
                async with _strict_stdio_client(
                    params,
                    max_frame_bytes=max_response_bytes,
                    max_request_bytes=server.max_request_bytes,
                    cwd_fd=cwd_fd,
                    deadline=deadline,
                    limits=limits,
                ) as (read, write):
                    async with ClientSession(read, write) as session:
                        await _mcp_await_with_deadline(
                            session.initialize(),
                            deadline=deadline,
                            stage="stdio initialize",
                        )
                        yield session
            return
        if server.transport == "streamable_http":
            if server.http is None:
                raise RuntimeError("MCP streamable_http transport is missing HTTP configuration")
            async with self._http_client(
                server,
                timeout_s=_mcp_remaining_timeout(deadline, stage="HTTP client setup"),
                max_response_bytes=max_response_bytes,
                runtime_environment=runtime_environment,
                deadline=deadline,
            ) as http_client:
                async with streamable_http_client(
                    server.http.url,
                    http_client=http_client,
                ) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await _mcp_await_with_deadline(
                            session.initialize(),
                            deadline=deadline,
                            stage="HTTP initialize",
                        )
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
        runtime_environment: Mapping[str, str] | None = None,
        deadline: float | None = None,
    ):
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP provider requires httpx from the optional MCP dependency; "
                "install with `uv sync --extra mcp --all-groups`"
            ) from exc
        selected_deadline = (
            time.monotonic() + timeout_s if deadline is None else deadline
        )
        remaining = _mcp_remaining_timeout(
            selected_deadline,
            requested=timeout_s,
            stage="HTTP client setup",
        )
        timeout = httpx.Timeout(remaining, read=remaining)
        headers = self._resolved_http_headers(
            server,
            runtime_environment=runtime_environment,
        )
        headers["Accept-Encoding"] = "identity"
        transport = _McpPolicyAsyncHTTPTransport(
            max_response_bytes=max_response_bytes,
            deadline=selected_deadline,
        )
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

    def _resolved_http_headers(
        self,
        server: McpServerSpec,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if server.http is None:
            return {}
        if runtime_environment is not None:
            return dict(runtime_environment)
        headers: dict[str, str] = {}
        for name, spec in server.http.headers.items():
            value = os.environ.get(spec.env)
            if value is None:
                raise RuntimeError(f"missing environment variable for MCP header {name}: {spec.env}")
            headers[name] = f"{spec.prefix}{value}{spec.suffix}"
        return headers

    def _resolved_stdio_env(
        self,
        server: McpServerSpec,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if server.stdio is None:
            return {}
        if runtime_environment is not None:
            # The primitive snapshot already includes Windows child-process
            # bootstrap variables.  Do not mix in a later ambient value.
            return dict(runtime_environment)
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
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        env = self._resolved_stdio_env(
            server,
            runtime_environment=runtime_environment,
        )
        if executable_snapshot is None:
            return env
        candidate = self._stdio_command_candidate(
            server,
            runtime_environment=runtime_environment,
        )
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

    @contextlib.contextmanager
    def _stdio_dispatch_cwd(
        self,
        server: McpServerSpec,
    ) -> Iterator[tuple[str, int | None]]:
        """Bind a replaceable workspace cwd to one stable directory object."""

        try:
            resolved = self._resolved_stdio_cwd(server)
        except (OSError, ValidationError) as exc:
            raise _McpStdioDispatchNotStarted(
                "MCP stdio cwd changed before dispatch"
            ) from exc
        if server.stdio is None or server.stdio.cwd is None:
            # The workspace root is a Host-owned boundary. A process confined
            # within it cannot rename or replace that root through workspace
            # paths, so no child-directory handle is needed.
            yield resolved, None
            return
        if not _MCP_STABLE_CWD_SUPPORTED:
            raise _McpStdioDispatchNotStarted(
                "configured MCP stdio cwd requires stable /proc/self/fd support"
            )

        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            cwd_fd = os.open(resolved, flags)
        except OSError as exc:
            raise _McpStdioDispatchNotStarted(
                "MCP stdio cwd changed before dispatch"
            ) from exc
        try:
            fd_path = Path("/proc/self/fd") / str(cwd_fd)
            try:
                actual = fd_path.resolve(strict=True)
            except OSError as exc:
                raise _McpStdioDispatchNotStarted(
                    "MCP stdio cwd handle cannot be resolved"
                ) from exc
            root = self.workspace_root.resolve(strict=True)
            if actual != root and root not in actual.parents:
                raise _McpStdioDispatchNotStarted(
                    "MCP stdio cwd changed outside workspace before dispatch"
                )
            yield str(fd_path), cwd_fd
        finally:
            os.close(cwd_fd)

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


def _mcp_remaining_timeout(
    deadline: float,
    *,
    requested: float | None = None,
    stage: str,
) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"MCP absolute deadline exhausted during {stage}")
    if requested is None:
        return remaining
    return min(remaining, max(0.0, requested))


async def _mcp_await_with_deadline(
    awaitable: Any,
    *,
    deadline: float,
    stage: str,
) -> Any:
    try:
        timeout = _mcp_remaining_timeout(deadline, stage=stage)
    except BaseException:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError as exc:
        raise TimeoutError(
            f"MCP absolute deadline exhausted during {stage}"
        ) from exc


class _McpPolicyAsyncHTTPTransport:
    """MCP address policy plus pre-materialization HTTP response bounds."""

    def __init__(
        self,
        *,
        max_response_bytes: int,
        deadline: float | None = None,
    ) -> None:
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
        self.deadline = float("inf") if deadline is None else deadline
        self.limit_error: RuntimeError | None = None
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(),
            max_connections=8,
            max_keepalive_connections=0,
            keepalive_expiry=0,
            http1=True,
            http2=False,
            retries=0,
            network_backend=_McpPolicyNetworkBackend(deadline=self.deadline),
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
        extensions = dict(request.extensions)
        timeout_extension = dict(extensions.get("timeout", {}))
        for timeout_kind in ("connect", "pool", "read", "write"):
            requested = timeout_extension.get(timeout_kind)
            timeout_extension[timeout_kind] = _mcp_remaining_timeout(
                self.deadline,
                requested=requested,
                stage=f"HTTP {timeout_kind}",
            )
        extensions["timeout"] = timeout_extension
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
            extensions=extensions,
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
                deadline=self.deadline,
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
    deadline: float = float("inf"),
) -> Any:
    import httpx

    limiter = _McpHttpResponseLimiter(max_response_bytes=max_response_bytes, is_sse=is_sse)

    class BoundedMcpHttpStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            iterator = stream.__aiter__()
            while True:
                try:
                    with _map_mcp_httpcore_exceptions():
                        chunk = await _mcp_await_with_deadline(
                            anext(iterator),
                            deadline=deadline,
                            stage="HTTP response body",
                        )
                except StopAsyncIteration:
                    return
                else:
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

    def __init__(self, *, deadline: float | None = None) -> None:
        try:
            import httpcore
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP HTTP transport requires httpcore from the optional MCP dependency; "
                "install with `uv sync --extra mcp --all-groups`"
            ) from exc
        self._backend = httpcore.AnyIOBackend()
        self.deadline = float("inf") if deadline is None else deadline

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        host_text = host.decode("idna") if isinstance(host, bytes) else str(host)
        addresses = await run_blocking_once(
            _allowed_mcp_connect_addresses,
            host_text,
            port,
            deadline=self.deadline,
        )
        last_exc: Exception | None = None
        for address in addresses:
            try:
                selected_timeout = _mcp_remaining_timeout(
                    self.deadline,
                    requested=timeout,
                    stage=f"TCP connect to {address}",
                )
                stream = await self._backend.connect_tcp(
                    address,
                    port,
                    timeout=selected_timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
                return _McpDeadlineNetworkStream(
                    stream,
                    deadline=self.deadline,
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
        selected_timeout = _mcp_remaining_timeout(
            self.deadline,
            requested=timeout,
            stage="Unix socket connect",
        )
        stream = await self._backend.connect_unix_socket(
            path,
            timeout=selected_timeout,
            socket_options=socket_options,
        )
        return _McpDeadlineNetworkStream(stream, deadline=self.deadline)

    async def sleep(self, seconds: float) -> None:
        selected = _mcp_remaining_timeout(
            self.deadline,
            requested=seconds,
            stage="HTTP transport backoff",
        )
        await self._backend.sleep(selected)
        if selected < seconds:
            raise TimeoutError(
                "MCP absolute deadline exhausted during HTTP transport backoff"
            )


class _McpDeadlineNetworkStream:
    """Clamp every socket and TLS operation to one transport deadline."""

    def __init__(self, stream: Any, *, deadline: float) -> None:
        self._stream = stream
        self.deadline = deadline

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return await self._stream.read(
            max_bytes,
            timeout=_mcp_remaining_timeout(
                self.deadline,
                requested=timeout,
                stage="HTTP socket read",
            ),
        )

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        await self._stream.write(
            buffer,
            timeout=_mcp_remaining_timeout(
                self.deadline,
                requested=timeout,
                stage="HTTP socket write",
            ),
        )

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> "_McpDeadlineNetworkStream":
        stream = await self._stream.start_tls(
            ssl_context,
            server_hostname=server_hostname,
            timeout=_mcp_remaining_timeout(
                self.deadline,
                requested=timeout,
                stage="TLS handshake",
            ),
        )
        return _McpDeadlineNetworkStream(stream, deadline=self.deadline)

    def get_extra_info(self, info: str) -> Any:
        return self._stream.get_extra_info(info)


def _allowed_mcp_connect_addresses(
    host: str,
    port: int,
    *,
    deadline: float | None = None,
) -> list[str]:
    normalized = host.strip("[]").lower()
    if normalized in _MCP_FORBIDDEN_HOSTS:
        raise ValidationError("MCP HTTP host is not allowed")
    allow_local = normalized in _MCP_LOCAL_HTTP_HOSTS
    literal = _ip_address_or_none(host)
    if literal is not None:
        _validate_mcp_connect_ip(literal, allow_local=allow_local)
        return [host.strip("[]")]
    infos = _bounded_mcp_getaddrinfo(host, port, deadline=deadline)
    addresses = sorted({info[4][0] for info in infos})
    if not addresses:
        raise ValidationError(f"MCP host resolved no addresses: {host}")
    for address in addresses:
        _validate_mcp_connect_ip(ipaddress.ip_address(address), allow_local=allow_local)
    return addresses


def _bounded_mcp_getaddrinfo(
    host: str,
    port: int,
    *,
    deadline: float | None,
) -> list[Any]:
    """Run the Host resolver behind a bounded, saturation-safe daemon slot."""
    try:
        return _bounded_provider_getaddrinfo(
            host,
            port,
            deadline=deadline,
            operation="MCP",
        )
    except socket.gaierror as error:
        raise ValidationError(f"MCP host could not be resolved: {host}") from error


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
    if not _MCP_WINDOWS:
        return {}
    env: dict[str, str] = {}
    for key in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _resolve_windows_mcp_bare_command(
    command: str,
    *,
    search_path: str | None,
    pathext: str | None,
) -> str | None:
    """Resolve from explicit child snapshots without Windows CWD fallback."""

    if search_path is None or pathext is None:
        raise ValidationError(
            "Windows MCP stdio bare commands require manifest-mapped child "
            "PATH and PATHEXT"
        )
    directories = _windows_mcp_search_directories(search_path)
    selected_extensions = _windows_mcp_executable_extensions(pathext)
    names = _windows_mcp_candidate_names(command, selected_extensions)
    return _find_windows_mcp_command(directories, names)


def _windows_mcp_search_directories(search_path: str) -> list[Path]:
    path_entries = search_path.split(os.pathsep)
    if not path_entries:
        raise ValidationError("Windows MCP stdio PATH must be non-empty")
    directories: list[Path] = []
    for entry in path_entries:
        directory = Path(entry)
        if not entry or not directory.is_absolute():
            raise ValidationError(
                "Windows MCP stdio PATH entries must be non-empty absolute paths"
            )
        directories.append(directory)
    return directories


def _windows_mcp_pathext_entry_invalid(extension: str) -> bool:
    return (
        not extension
        or not extension.startswith(".")
        or extension in {".", ".."}
        or any(char in extension for char in "/\\:\x00")
    )


def _windows_mcp_executable_extensions(pathext: str) -> list[str]:
    extensions = pathext.split(os.pathsep)
    if not extensions or any(
        _windows_mcp_pathext_entry_invalid(extension)
        for extension in extensions
    ):
        raise ValidationError("Windows MCP stdio PATHEXT is invalid")
    selected_extensions = [
        extension
        for extension in extensions
        if extension.casefold() in _MCP_WINDOWS_EXECUTABLE_SUFFIXES
    ]
    if not selected_extensions:
        raise ValidationError(
            "Windows MCP stdio PATHEXT must allow .exe or .com"
        )
    return selected_extensions


def _windows_mcp_candidate_names(
    command: str,
    extensions: list[str],
) -> list[str]:
    return (
        [command]
        if any(
            command.casefold().endswith(extension.casefold())
            for extension in extensions
        )
        else [f"{command}{extension}" for extension in extensions]
    )


def _find_windows_mcp_command(
    directories: list[Path],
    names: list[str],
) -> str | None:
    seen: set[str] = set()
    for directory in directories:
        normalized = os.path.normcase(str(directory))
        if normalized in seen:
            continue
        seen.add(normalized)
        for name in names:
            candidate = directory / name
            if candidate.is_file() and os.access(
                candidate,
                os.F_OK | os.X_OK,
            ):
                return str(candidate)
    return None


@dataclass(frozen=True)
class _McpStdioConfig:
    max_frame_bytes: int
    request_limit_bytes: int
    stdout_limit_bytes: int
    stderr_limit_bytes: int
    deadline: float
    limits: SubprocessLimits | None


def _mcp_stdio_dependencies() -> tuple[Any, Any, Any]:
    try:
        import anyio
        import anyio.lowlevel
        import mcp.types as mcp_types
        from mcp.shared.message import SessionMessage
    except ModuleNotFoundError as exc:
        raise ValidationError(
            "MCP stdio transport requires the optional MCP dependency; "
            "install with `uv sync --extra mcp --all-groups`"
        ) from exc
    return anyio, mcp_types, SessionMessage


def _validated_mcp_stdio_config(
    *,
    max_frame_bytes: int,
    max_request_bytes: int | None,
    deadline: float | None,
    limits: SubprocessLimits | None,
    stdout_limit_bytes: int | None,
    stderr_limit_bytes: int | None,
) -> _McpStdioConfig:
    selected_request_limit = (
        max_frame_bytes if max_request_bytes is None else max_request_bytes
    )
    selected_stdout_limit = (
        max_frame_bytes * _MCP_STDIO_PROTOCOL_OUTPUT_MULTIPLIER
        if stdout_limit_bytes is None
        else stdout_limit_bytes
    )
    selected_stderr_limit = (
        max_frame_bytes if stderr_limit_bytes is None else stderr_limit_bytes
    )
    for field_name, value in (
        ("max_frame_bytes", max_frame_bytes),
        ("max_request_bytes", selected_request_limit),
        ("stdout limit", selected_stdout_limit),
        ("stderr limit", selected_stderr_limit),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"MCP stdio {field_name} must be a positive integer")
    if limits is not None:
        _validate_mcp_stdio_subprocess_limits(limits)
    return _McpStdioConfig(
        max_frame_bytes=max_frame_bytes,
        request_limit_bytes=selected_request_limit,
        stdout_limit_bytes=selected_stdout_limit,
        stderr_limit_bytes=selected_stderr_limit,
        deadline=(
            time.monotonic() + _TOOL_DEFAULTS.shell_timeout_s
            if deadline is None
            else deadline
        ),
        limits=limits,
    )


def _validate_mcp_stdio_subprocess_limits(limits: SubprocessLimits) -> None:
    for field_name, value in (
        ("wall_seconds", limits.wall_seconds),
        ("cpu_seconds", limits.cpu_seconds),
        ("memory_bytes", limits.memory_bytes),
    ):
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or type(value) not in {int, float}
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValidationError(
                f"MCP stdio SubprocessLimits.{field_name} must be positive and finite"
            )


async def _spawn_mcp_stdio_process(
    anyio: Any,
    server: Any,
    *,
    cwd_fd: int | None,
    config: _McpStdioConfig,
) -> tuple[Any, int | None, WindowsJobObject | None]:
    # ``server.command`` is already the verified absolute snapshot path. Do
    # not let the SDK run a second PATH/PATHEXT lookup after authorization.
    if _MCP_WINDOWS:
        return await _spawn_windows_mcp_stdio_process(anyio, server, cwd_fd, config)
    return await _spawn_posix_mcp_stdio_process(anyio, server, cwd_fd, config)


async def _spawn_windows_mcp_stdio_process(
    anyio: Any,
    server: Any,
    cwd_fd: int | None,
    config: _McpStdioConfig,
) -> tuple[Any, None, WindowsJobObject]:
    if cwd_fd is not None:
        raise _McpStdioDispatchNotStarted(
            "Windows MCP stdio cannot receive a POSIX cwd handle"
        )
    try:
        windows_job = WindowsJobObject.create(config.limits)
    except OSError as exc:
        raise _McpStdioDispatchNotStarted(
            "MCP stdio could not create the required Windows Job Object"
        ) from exc
    process = None
    try:
        process = await _mcp_await_with_deadline(
            anyio.open_process(
                [server.command, *list(server.args)],
                env=dict(server.env or {}),
                stderr=subprocess.PIPE,
                cwd=server.cwd,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                ),
            ),
            deadline=config.deadline,
            stage="stdio process spawn",
        )
        windows_job.assign_pid(process.pid)
        return process, None, windows_job
    except BaseException as exc:
        if process is not None:
            with contextlib.suppress(Exception):
                process.kill()
            with anyio.move_on_after(0.25):
                await process.wait()
        windows_job.close()
        if isinstance(exc, TimeoutError):
            raise
        if process is None:
            raise _McpStdioDispatchNotStarted(
                "MCP stdio process creation failed before dispatch"
            ) from exc
        raise _McpStdioDispatchStarted(
            "MCP stdio child could not attach to the required Windows Job Object"
        ) from exc


async def _spawn_posix_mcp_stdio_process(
    anyio: Any,
    server: Any,
    cwd_fd: int | None,
    config: _McpStdioConfig,
) -> tuple[Any, int, None]:
    process = await _mcp_await_with_deadline(
        anyio.open_process(
            [server.command, *list(server.args)],
            env=dict(server.env or {}),
            stderr=subprocess.PIPE,
            cwd=server.cwd,
            start_new_session=True,
            pass_fds=(() if cwd_fd is None else (cwd_fd,)),
        ),
        deadline=config.deadline,
        stage="stdio process spawn",
    )
    process_group_id = process.pid
    try:
        if os.getpgid(process.pid) != process_group_id:
            raise OSError("spawned MCP process is not its process-group leader")
    except OSError as exc:
        await _terminate_mcp_stdio_process(
            process,
            process_group_id=process_group_id,
            windows_job=None,
        )
        raise _McpStdioDispatchStarted(
            "MCP stdio child could not establish an isolated process group"
        ) from exc
    return process, process_group_id, None


async def _close_mcp_stdio_streams(*streams: Any) -> None:
    for stream in streams:
        with contextlib.suppress(Exception):
            await stream.aclose()


def _mcp_stdio_ps_process(
    process: Any,
    *,
    require_complete: bool,
) -> psutil.Process | None:
    try:
        return psutil.Process(process.pid)
    except (psutil.Error, OSError) as exc:
        if require_complete:
            raise ValidationError(
                "MCP stdio cannot enforce CPU/memory SubprocessLimits because "
                "complete process metrics are unavailable"
            ) from exc
        return None


class _StrictMcpStdioTransport:
    def __init__(
        self,
        *,
        anyio: Any,
        mcp_types: Any,
        session_message_type: Any,
        server: Any,
        config: _McpStdioConfig,
        process: Any,
        process_group_id: int | None,
        windows_job: WindowsJobObject | None,
        ps_process: psutil.Process | None,
        require_complete_metrics: bool,
        read_stream_writer: Any,
        read_stream: Any,
        write_stream: Any,
        write_stream_reader: Any,
    ) -> None:
        self.anyio = anyio
        self.mcp_types = mcp_types
        self.session_message_type = session_message_type
        self.server = server
        self.config = config
        self.process = process
        self.process_group_id = process_group_id
        self.windows_job = windows_job
        self.ps_process = ps_process
        self.require_complete_metrics = require_complete_metrics
        self.read_stream_writer = read_stream_writer
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.write_stream_reader = write_stream_reader
        self.transport_errors: list[BaseException] = []
        self.failure_event = anyio.Event()
        self.termination_lock = anyio.Lock()
        self.terminated = False
        self.started_at = time.monotonic()
        self.peak_memory = 0
        self.cpu_seconds = 0.0
        self.stdout_bytes = 0
        self.stderr_bytes = 0
        self.stdin_bytes = 0

    def record_failure(self, error: BaseException) -> None:
        if not self.transport_errors:
            self.transport_errors.append(error)
        self.failure_event.set()

    async def _send_stdout_failure(self, error: BaseException) -> None:
        self.record_failure(error)
        with self.anyio.move_on_after(0.05):
            await self.read_stream_writer.send(error)

    async def stdout_reader(self) -> None:
        assert self.process.stdout
        try:
            async with self.read_stream_writer:
                buffer = bytearray()
                while True:
                    read_size = _mcp_stdio_read_size(
                        len(buffer),
                        self.config.max_frame_bytes,
                    )
                    try:
                        chunk = await self.process.stdout.receive(max_bytes=read_size)
                    except self.anyio.EndOfStream:
                        break
                    if not chunk:
                        break
                    self.stdout_bytes += len(chunk)
                    if self.stdout_bytes > self.config.stdout_limit_bytes:
                        await self._send_stdout_failure(
                            RuntimeError(
                                "MCP stdio stdout exceeded max_output_bytes="
                                f"{self.config.stdout_limit_bytes}"
                            )
                        )
                        return
                    buffer.extend(chunk)
                    if not await self._drain_stdout_frames(buffer):
                        return
        except self.anyio.ClosedResourceError:
            await self.anyio.lowlevel.checkpoint()

    async def _drain_stdout_frames(self, buffer: bytearray) -> bool:
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                if len(buffer) > self.config.max_frame_bytes:
                    await self._send_stdout_failure(self._frame_limit_error())
                    return False
                return True
            if newline > self.config.max_frame_bytes:
                await self._send_stdout_failure(self._frame_limit_error())
                return False
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            try:
                message = self.mcp_types.JSONRPCMessage.model_validate_json(line)
            except Exception as exc:
                await self.read_stream_writer.send(exc)
                continue
            await self.read_stream_writer.send(self.session_message_type(message))

    def _frame_limit_error(self) -> RuntimeError:
        return RuntimeError(
            "MCP stdio frame exceeded "
            f"max_response_bytes={self.config.max_frame_bytes}"
        )

    async def stderr_reader(self) -> None:
        assert self.process.stderr
        try:
            while True:
                try:
                    chunk = await self.process.stderr.receive(
                        max_bytes=min(
                            _MCP_STDIO_READ_CHUNK_BYTES,
                            self.config.stderr_limit_bytes + 1,
                        )
                    )
                except self.anyio.EndOfStream:
                    return
                if not chunk:
                    return
                self.stderr_bytes += len(chunk)
                if self.stderr_bytes > self.config.stderr_limit_bytes:
                    self.record_failure(
                        RuntimeError(
                            "MCP stdio stderr exceeded max_output_bytes="
                            f"{self.config.stderr_limit_bytes}"
                        )
                    )
                    return
        except self.anyio.ClosedResourceError:
            await self.anyio.lowlevel.checkpoint()

    async def stdin_writer(self) -> None:
        assert self.process.stdin
        try:
            async with self.write_stream_reader:
                async for session_message in self.write_stream_reader:
                    encoded = self._encode_request(session_message)
                    if len(encoded) > self.config.request_limit_bytes:
                        self.record_failure(
                            RuntimeError(
                                "MCP stdio request frame exceeded max_request_bytes="
                                f"{self.config.request_limit_bytes}"
                            )
                        )
                        return
                    self.stdin_bytes += len(encoded)
                    aggregate_limit = (
                        self.config.request_limit_bytes
                        * _MCP_STDIO_PROTOCOL_OUTPUT_MULTIPLIER
                    )
                    if self.stdin_bytes > aggregate_limit:
                        self.record_failure(
                            RuntimeError(
                                "MCP stdio stdin exceeded "
                                f"max_output_bytes={aggregate_limit}"
                            )
                        )
                        return
                    await self.process.stdin.send(encoded)
        except self.anyio.ClosedResourceError:
            await self.anyio.lowlevel.checkpoint()

    def _encode_request(self, session_message: Any) -> bytes:
        raw = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
        return (raw + "\n").encode(
            encoding=self.server.encoding,
            errors=self.server.encoding_error_handler,
        )

    def _selected_limit_kind(self, wall_seconds: float) -> tuple[bool, str | None]:
        timed_out = time.monotonic() >= self.config.deadline
        limits = self.config.limits
        if timed_out or limits is None:
            return timed_out, None
        if limits.wall_seconds is not None and wall_seconds > limits.wall_seconds:
            return False, "subprocess_wall_seconds"
        if limits.cpu_seconds is not None and self.cpu_seconds > limits.cpu_seconds:
            return False, "subprocess_cpu_seconds"
        if limits.memory_bytes is not None and self.peak_memory > limits.memory_bytes:
            return False, "subprocess_memory_bytes"
        return False, None

    def _record_resource_failure(
        self,
        *,
        wall_seconds: float,
        timed_out: bool,
        limit_kind: str | None,
    ) -> None:
        metrics = CommandMetrics(
            wall_seconds=wall_seconds,
            cpu_seconds=self.cpu_seconds,
            peak_memory_bytes=self.peak_memory,
            killed=True,
            limit_kind=("subprocess_timeout" if timed_out else limit_kind),
        )
        if timed_out:
            self.record_failure(
                SubprocessTimeoutExpired(
                    "MCP stdio subprocess exceeded the absolute deadline",
                    metrics=metrics,
                )
            )
            return
        self.record_failure(
            SubprocessLimitExceeded(
                f"MCP stdio subprocess exceeded {limit_kind}",
                metrics=metrics,
            )
        )

    async def resource_monitor(self) -> None:
        while True:
            if self.failure_event.is_set():
                await self._terminate()
                return
            wall_seconds = time.monotonic() - self.started_at
            try:
                if self.ps_process is not None:
                    self.cpu_seconds, self.peak_memory = _sample_mcp_process_tree(
                        self.ps_process,
                        self.peak_memory,
                        require_complete=self.require_complete_metrics,
                    )
            except ValidationError as exc:
                self.record_failure(exc)
                continue
            timed_out, limit_kind = self._selected_limit_kind(wall_seconds)
            if timed_out or limit_kind is not None:
                self._record_resource_failure(
                    wall_seconds=wall_seconds,
                    timed_out=timed_out,
                    limit_kind=limit_kind,
                )
                continue
            if self.process.returncode is not None:
                return
            await self.anyio.sleep(0.01)

    async def _terminate(self) -> None:
        async with self.termination_lock:
            if self.terminated:
                return
            await _terminate_mcp_stdio_process(
                self.process,
                process_group_id=self.process_group_id,
                windows_job=self.windows_job,
            )
            self.windows_job = None
            self.terminated = True

    @contextlib.asynccontextmanager
    async def run(self):
        body_error: BaseException | None = None
        try:
            async with self.process:
                async with self.anyio.create_task_group() as task_group:
                    for task in (
                        self.stdout_reader,
                        self.stderr_reader,
                        self.stdin_writer,
                        self.resource_monitor,
                    ):
                        task_group.start_soon(task)
                    try:
                        yield self.read_stream, self.write_stream
                    except BaseException as exc:
                        body_error = exc
                    finally:
                        # Once the task group is cancelled, ordinary awaits in
                        # this scope are cancellation points. Shield cleanup so
                        # every SDK stream and subprocess pipe is deterministically
                        # closed before the asyncio loop can disappear.
                        with self.anyio.CancelScope(shield=True):
                            if self.process.stdin:
                                with contextlib.suppress(Exception):
                                    await self.process.stdin.aclose()
                            await self._terminate()
                            task_group.cancel_scope.cancel()
                            await _close_mcp_stdio_streams(
                                self.process.stdout,
                                self.process.stderr,
                                self.read_stream,
                                self.write_stream,
                                self.read_stream_writer,
                                self.write_stream_reader,
                            )
        except BaseException as exc:
            if body_error is None:
                body_error = exc
        finally:
            if self.windows_job is not None:
                self.windows_job.close()
        self._raise_transport_error(body_error)

    def _raise_transport_error(self, body_error: BaseException | None) -> None:
        if self.transport_errors:
            if body_error is not None:
                raise self.transport_errors[0] from body_error
            raise self.transport_errors[0]
        if body_error is not None:
            raise body_error


@contextlib.asynccontextmanager
async def _strict_stdio_client(
    server: Any,
    *,
    max_frame_bytes: int,
    max_request_bytes: int | None = None,
    cwd_fd: int | None = None,
    deadline: float | None = None,
    limits: SubprocessLimits | None = None,
    stdout_limit_bytes: int | None = None,
    stderr_limit_bytes: int | None = None,
):
    anyio, mcp_types, session_message_type = _mcp_stdio_dependencies()
    config = _validated_mcp_stdio_config(
        max_frame_bytes=max_frame_bytes,
        max_request_bytes=max_request_bytes,
        deadline=deadline,
        limits=limits,
        stdout_limit_bytes=stdout_limit_bytes,
        stderr_limit_bytes=stderr_limit_bytes,
    )
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)
    streams = (read_stream, write_stream, read_stream_writer, write_stream_reader)
    windows_job: WindowsJobObject | None = None
    try:
        process, process_group_id, windows_job = await _spawn_mcp_stdio_process(
            anyio,
            server,
            cwd_fd=cwd_fd,
            config=config,
        )
    except BaseException:
        if windows_job is not None:
            windows_job.close()
        await _close_mcp_stdio_streams(*streams)
        raise

    require_complete_metrics = bool(
        limits is not None
        and (limits.cpu_seconds is not None or limits.memory_bytes is not None)
    )
    try:
        ps_process = _mcp_stdio_ps_process(
            process,
            require_complete=require_complete_metrics,
        )
    except BaseException:
        await _terminate_mcp_stdio_process(
            process,
            process_group_id=process_group_id,
            windows_job=windows_job,
        )
        await _close_mcp_stdio_streams(*streams)
        raise

    transport = _StrictMcpStdioTransport(
        anyio=anyio,
        mcp_types=mcp_types,
        session_message_type=session_message_type,
        server=server,
        config=config,
        process=process,
        process_group_id=process_group_id,
        windows_job=windows_job,
        ps_process=ps_process,
        require_complete_metrics=require_complete_metrics,
        read_stream_writer=read_stream_writer,
        read_stream=read_stream,
        write_stream=write_stream,
        write_stream_reader=write_stream_reader,
    )
    async with transport.run() as client_streams:
        yield client_streams


def _sample_mcp_process_tree(
    process: psutil.Process,
    peak_memory: int,
    *,
    require_complete: bool,
) -> tuple[float, int]:
    cpu_seconds = 0.0
    memory_bytes = 0
    processes = [process]
    if require_complete:
        try:
            processes.extend(process.children(recursive=True))
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
        except (psutil.Error, OSError) as exc:
            raise ValidationError(
                "MCP stdio cannot enforce CPU/memory SubprocessLimits because "
                "complete process metrics are unavailable"
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
                    "MCP stdio cannot enforce CPU/memory SubprocessLimits because "
                    "complete process metrics are unavailable"
                ) from exc
    return cpu_seconds, max(peak_memory, memory_bytes)


async def _terminate_mcp_stdio_process(
    process: Any,
    *,
    process_group_id: int | None,
    windows_job: WindowsJobObject | None,
) -> None:
    try:
        import anyio
    except ModuleNotFoundError:  # pragma: no cover - imported by caller
        return
    if _MCP_WINDOWS:
        if windows_job is not None:
            # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is the fail-closed tree kill.
            windows_job.close()
        with contextlib.suppress(Exception):
            process.terminate()
    elif process_group_id is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process_group_id, signal.SIGTERM)
        await anyio.sleep(0.05)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process_group_id, signal.SIGKILL)
    else:
        with contextlib.suppress(Exception):
            process.terminate()
    with anyio.move_on_after(0.25):
        await process.wait()
    if process.returncode is None:
        with contextlib.suppress(Exception):
            process.kill()
        with anyio.move_on_after(0.25):
            await process.wait()
    # AnyIO's asyncio receive wrapper marks stdout/stderr closed without
    # closing BaseSubprocessTransport itself. Close that transport while the
    # loop is alive so pipe finalizers cannot fire after asyncio.run() exits.
    raw_process = getattr(process, "_process", None)
    raw_transport = getattr(raw_process, "_transport", None)
    if raw_transport is not None:
        with contextlib.suppress(Exception):
            raw_transport.close()
        await anyio.lowlevel.checkpoint()


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


def _mcp_structured_content(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    return getattr(result, "structured_content", None)


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
    """Project MCP binary blocks without placing their base64 in later prompts.

    This is intentionally recursive because embedded resources and provider
    extensions may nest content blocks.  ``response_bytes`` is measured from
    the unprojected JSON payload before this helper runs, so transport/resource
    evidence does not mistake this compact view for the raw response size.
    """

    if isinstance(value, list):
        return [_bounded_mcp_content(item) for item in value]
    if not isinstance(value, dict):
        return value

    item_type = value.get("type")
    is_binary_block = item_type in {"image", "audio"}
    is_resource_block = item_type == "resource"
    projected: dict[str, Any] = {}
    payload_key = "data" if is_binary_block and "data" in value else None
    if is_resource_block and "blob" in value:
        payload_key = "blob"

    for key, item in value.items():
        if key == payload_key:
            continue
        if key == "resource" and is_resource_block and isinstance(item, dict):
            projected[key] = _bounded_mcp_embedded_resource(item)
        else:
            projected[key] = _bounded_mcp_content(item)

    if payload_key is not None:
        _merge_mcp_content_observation(
            projected,
            _mcp_base64_observation(value[payload_key]),
        )
    elif (is_binary_block or is_resource_block) and value.get("content_omitted") is True:
        # Preserve an already-projected provider receipt idempotently.
        projected["raw_content_retained"] = False
    if is_binary_block or is_resource_block:
        return _canonical_mcp_media_metadata(projected, value)
    return projected


def _bounded_mcp_embedded_resource(value: dict[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for key, item in value.items():
        if key == "blob":
            continue
        projected[key] = _bounded_mcp_content(item)
    if "blob" in value:
        _merge_mcp_content_observation(
            projected,
            _mcp_base64_observation(value["blob"]),
        )
    elif value.get("content_omitted") is True:
        projected["raw_content_retained"] = False
    return _canonical_mcp_media_metadata(projected, value)


def _canonical_mcp_media_metadata(
    projected: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    if "mimeType" not in projected and source.get("mime_type") is not None:
        projected["mimeType"] = source["mime_type"]
        projected.pop("mime_type", None)
    return projected


def _merge_mcp_content_observation(
    projected: dict[str, Any],
    observation: dict[str, Any],
) -> None:
    """Prefer computed receipts while retaining conflicting provider claims."""

    conflicts = {
        key: projected[key]
        for key, value in observation.items()
        if key in projected and projected[key] != value
    }
    if conflicts:
        provider_reported = projected.get("provider_reported_content_metadata")
        retained = (
            dict(provider_reported)
            if isinstance(provider_reported, dict)
            else (
                {"original_value": provider_reported}
                if provider_reported is not None
                else {}
            )
        )
        retained.update(conflicts)
        projected["provider_reported_content_metadata"] = retained
    projected.update(observation)


def _mcp_base64_observation(value: Any) -> dict[str, Any]:
    """Return an explicit receipt for valid or malformed base64 content."""

    if isinstance(value, str):
        encoded = value.encode("utf-8")
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, UnicodeEncodeError, ValueError):
            return {
                "content_omitted": True,
                "raw_content_retained": False,
                "content_encoding": "base64",
                "base64_valid": False,
                "encoded_bytes": len(encoded),
                "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        return {
            "content_omitted": True,
            "raw_content_retained": False,
            "content_encoding": "base64",
            "base64_valid": True,
            "bytes": len(decoded),
            "sha256": hashlib.sha256(decoded).hexdigest(),
            "sha256_basis": "decoded_bytes",
        }

    encoded = dumps(value).encode("utf-8")
    return {
        "content_omitted": True,
        "raw_content_retained": False,
        "content_encoding": "base64",
        "base64_valid": False,
        "encoded_bytes": len(encoded),
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _mcp_oversize_observation(encoded: bytes) -> dict[str, Any]:
    return {
        "type": "oversize",
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "sha256_basis": "normalized_raw_payload",
        "content_omitted": True,
        "raw_content_retained": False,
    }


class LocalResourceProviderSubstrate:
    """Default Resource Provider Substrate backed by the host OS."""

    def __init__(
        self,
        workspace_root: str | Path,
        namespace: str = _RUNTIME_DEFAULTS.workspace_namespace,
        *,
        git_config: GitDefaults | None = None,
    ):
        requested_root = Path(workspace_root).resolve()
        self.filesystem = LocalFilesystemProvider(requested_root, namespace=namespace)
        # Use the filesystem provider's descriptor-canonical root everywhere.
        # On Darwin Path.resolve() preserves a caller's case spelling even
        # when APFS resolves it to a differently spelled directory entry.
        self.workspace_root = self.filesystem.root
        self.workspace_display = str(self.workspace_root)
        self._declared_git_config = git_config
        self._bound_runtime_git_config: GitDefaults | None = None
        self._git_config_binding_lock = threading.RLock()
        self.clock = LocalClockProvider()
        self.shell = LocalShellProvider(self.workspace_root, git_config=git_config)
        self.git = LocalGitProvider(self.workspace_root, config=git_config)
        self.human = LocalHumanProvider()
        self.jsonrpc = HttpJsonRpcProvider()
        self.mcp = SdkMcpProvider(self.workspace_root)

    def bind_runtime_git_config(self, git_config: GitDefaults) -> None:
        """Bind built-in Git boundaries to one authoritative Runtime policy.

        An explicit constructor policy is a caller-owned compatibility
        contract, so a conflicting Runtime policy is rejected instead of
        silently choosing one. Substrates created without a policy inherit the
        Runtime policy at assembly time.
        """

        with self._git_config_binding_lock:
            declared = self._declared_git_config
            if declared is not None and declared != git_config:
                raise ValidationError(
                    "local substrate Git configuration does not match the "
                    "Runtime Git configuration"
                )
            bound = self._bound_runtime_git_config
            if bound is not None and bound != git_config:
                raise ValidationError(
                    "local substrate is already bound to a different Runtime "
                    "Git configuration"
                )

            current_git = getattr(self, "git", None)
            replacement_git = current_git
            if isinstance(current_git, LocalGitProvider):
                if type(current_git) is LocalGitProvider:
                    if current_git.config != git_config:
                        replacement_git = LocalGitProvider(
                            self.workspace_root,
                            config=git_config,
                        )
                elif current_git.config != git_config:
                    raise ValidationError(
                        "local Git provider subclass configuration does not "
                        "match the Runtime Git configuration"
                    )

            current_shell = self.shell
            if (
                isinstance(current_shell, LocalShellProvider)
                and type(current_shell) is not LocalShellProvider
                and current_shell.git_config != git_config
            ):
                raise ValidationError(
                    "local shell provider subclass Git configuration does not "
                    "match the Runtime Git configuration"
                )
            if type(current_shell) is LocalShellProvider:
                current_shell.bind_runtime_git_config(git_config)
            if replacement_git is not current_git:
                self.git = replacement_git
            self._bound_runtime_git_config = git_config
